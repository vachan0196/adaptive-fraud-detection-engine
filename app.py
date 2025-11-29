# app.py
import json
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from streamlit_option_menu import option_menu
from streamlit_autorefresh import st_autorefresh

# ---------- page config ----------
st.set_page_config(page_title="Fraud Detection — Cost-Optimised", layout="wide")

# ---------- paths ----------
ART_DIR       = Path("artifacts")
MODEL_PATH    = ART_DIR / "rf_model.pkl"          # legacy default
MODEL_META_PATH = ART_DIR / "model_meta.json"     # new multi-model metadata
FEATS_PATH    = ART_DIR / "feature_cols.json"
META_PATH     = ART_DIR / "metadata.json"
THRESH_PATH   = ART_DIR / "threshold.json"
SAMPLES_PATH  = ART_DIR / "demo_samples.parquet"
SIM_POOL_PATH = ART_DIR / "sim_pool.parquet"

LIVE_LOG      = ART_DIR / "live_events.csv"
CASES_PATH    = ART_DIR / "cases.csv"
AUDIT_PATH    = ART_DIR / "case_audit_log.csv"

MAX_LIVE_ROWS = 200_000   # hard cap for live_events
MAX_CASE_ROWS = 200_000   # hard cap for cases / audit


# ---------- utils ----------
@st.cache_resource
def load_artifacts() -> Tuple[object, list, dict, dict, pd.DataFrame | None, dict | None]:
    """
    Load:
      - Active model (RF or LightGBM) using model_meta.json if available.
      - Feature list, metadata, threshold config, sample data.
      - Model metadata dictionary (for display).
    Falls back to rf_model.pkl if model_meta.json is missing or invalid.
    """
    # common artifacts
    features = json.loads(Path(FEATS_PATH).read_text())
    meta = json.loads(Path(META_PATH).read_text())
    thr = json.loads(Path(THRESH_PATH).read_text())
    samples = pd.read_parquet(SAMPLES_PATH) if SAMPLES_PATH.exists() else None

    model_meta = None
    model = None

    if MODEL_META_PATH.exists():
        try:
            model_meta = json.loads(MODEL_META_PATH.read_text())
            active_key = model_meta.get("active_model")
            models_info = model_meta.get("models", {}) or {}
            active_info = models_info.get(active_key, {})

            # e.g. "rf_model.pkl" or "lgbm_model.pkl"
            rel_path = active_info.get("path", "rf_model.pkl")
            model_path = ART_DIR / rel_path
            if not model_path.exists():
                # fallback to legacy RF if path missing
                model_path = MODEL_PATH
            model = joblib.load(model_path)
        except Exception:
            # any issue: fallback to single RF model
            model_meta = None
            model = joblib.load(MODEL_PATH)
    else:
        # old project version: only RF model
        model = joblib.load(MODEL_PATH)

    return model, features, meta, thr, samples, model_meta


rf, feature_cols, meta, thr, samples, model_meta = load_artifacts()
default_threshold = float(thr.get("threshold", 0.025))
C_FN, C_FP = int(thr.get("C_FN", 100)), int(thr.get("C_FP", 1))


def trim_csv(path: Path, max_rows: int, sort_col: str) -> None:
    """Keep only the most recent max_rows rows based on sort_col."""
    if not path.exists():
        return
    try:
        df = pd.read_csv(path)
        if len(df) > max_rows and sort_col in df.columns:
            df = df.sort_values(sort_col).tail(max_rows)
            df.to_csv(path, index=False)
    except Exception:
        # best-effort; don't crash the app
        return


def shap_for_binary(model, X_row: pd.DataFrame) -> np.ndarray:
    """Return SHAP values for class=1 for a single-row DF, robust across SHAP versions."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_row)
    if isinstance(sv, (list, tuple)):
        return np.array(sv[1][0])
    if hasattr(sv, "ndim") and sv.ndim == 3:
        return sv[0, :, 1]
    if hasattr(sv, "ndim") and sv.ndim == 2:
        return sv[0]
    return np.ravel(sv)


def make_empty_template():
    return pd.DataFrame([[0.0] * len(feature_cols)], columns=feature_cols)


def set_defaults_from_series(s: pd.Series):
    d = {k: float(s.get(k, 0.0)) for k in feature_cols}
    if "id" in d:
        d["id"] = 0.0
    st.session_state["defaults"] = d


def get_default_value(col, fallback):
    return st.session_state.get("defaults", {}).get(col, fallback)


def style_livefeed(df: pd.DataFrame):
    """Color rows: REVIEW=red, APPROVE=green; format proba as %."""
    def _row_style(row):
        is_review = str(row.get("decision", "")).upper() == "REVIEW"
        color = "#5a0e0e" if is_review else "#0e5a2b"
        return [f"background-color:{color}; color:#ffffff;"] * len(row)

    fmt = {"proba": lambda v: f"{float(v)*100:.2f}%"}
    return (df.style
              .apply(_row_style, axis=1)
              .format(fmt)
              .set_properties(**{"border-color": "#222"}))


def ensure_live_log_schema():
    """Make sure live_events.csv has the columns we expect, including event_id."""
    if not LIVE_LOG.exists():
        cols = ["event_id", "ts", "decision", "proba", "payload"]
        pd.DataFrame(columns=cols).to_csv(LIVE_LOG, index=False)
        return

    df = pd.read_csv(LIVE_LOG)
    changed = False
    if "event_id" not in df.columns:
        # create a simple incremental event_id if missing
        df.insert(0, "event_id", range(1, len(df) + 1))
        changed = True
    for col in ["ts", "decision", "proba", "payload"]:
        if col not in df.columns:
            df[col] = np.nan
            changed = True
    if changed:
        df.to_csv(LIVE_LOG, index=False)


def ensure_cases_schema():
    """Make sure cases.csv exists with event_id as primary reference."""
    if not CASES_PATH.exists():
        cols = [
            "event_id",
            "ts",
            "proba",
            "status",
            "customer_response",
            "resolution_source",
            "updated_at",
        ]
        pd.DataFrame(columns=cols).to_csv(CASES_PATH, index=False)
        return

    df = pd.read_csv(CASES_PATH)
    changed = False
    if "event_id" not in df.columns:
        # old version had ts as key; fabricate event_id
        df.insert(0, "event_id", range(1, len(df) + 1))
        changed = True
    for col in ["ts", "proba", "status", "customer_response",
                "resolution_source", "updated_at"]:
        if col not in df.columns:
            df[col] = "" if col in ("status", "customer_response", "resolution_source", "updated_at") else np.nan
            changed = True
    if changed:
        df.to_csv(CASES_PATH, index=False)


def sync_cases_from_live() -> pd.DataFrame:
    """
    Build / update a simple cases table from live_events.csv.
    - Every REVIEW transaction becomes / remains a case.
    - New REVIEWs get status=PENDING by default.
    Return merged DataFrame.
    """
    ensure_live_log_schema()
    ensure_cases_schema()

    trim_csv(LIVE_LOG, MAX_LIVE_ROWS, "event_id")
    trim_csv(CASES_PATH, MAX_CASE_ROWS, "event_id")

    live = pd.read_csv(LIVE_LOG)
    cases = pd.read_csv(CASES_PATH)

    if live.empty:
        return cases

    live["decision"] = live["decision"].astype(str).str.upper()
    reviews = live[live["decision"] == "REVIEW"].copy()

    if reviews.empty:
        return cases

    # Make sure event_id exists and is unique
    if "event_id" not in reviews.columns:
        reviews["event_id"] = range(1, len(reviews) + 1)

    reviews = reviews[["event_id", "ts", "proba"]]

    # Add missing review events as new cases
    if "event_id" not in cases.columns:
        cases["event_id"] = pd.Series(dtype=int)

    new_mask = ~reviews["event_id"].isin(cases["event_id"])
    if new_mask.any():
        new_cases = reviews.loc[new_mask].copy()
        new_cases["status"] = "PENDING"
        new_cases["customer_response"] = ""
        new_cases["resolution_source"] = ""
        new_cases["updated_at"] = ""
        cases = pd.concat([cases, new_cases], ignore_index=True)

    # Update ts/proba in case they changed
    cases = cases.merge(
        reviews,
        on="event_id",
        how="left",
        suffixes=("", "_live"),
    )
    for col in ["ts", "proba"]:
        live_col = f"{col}_live"
        cases[col] = np.where(
            cases[live_col].notna(), cases[live_col], cases[col]
        )
        cases.drop(columns=[live_col], inplace=True)

    cases.to_csv(CASES_PATH, index=False)
    return cases


def append_audit_row(event_id: int, old_status: str, new_status: str,
                     source: str, proba: float, ts: str):
    """Append a row to the case audit log CSV."""
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "event_id": event_id,
        "ts": ts,
        "proba": proba,
        "old_status": old_status,
        "new_status": new_status,
        "resolution_source": source,
        "logged_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    if AUDIT_PATH.exists():
        df = pd.read_csv(AUDIT_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    # trim
    if len(df) > MAX_CASE_ROWS:
        df = df.tail(MAX_CASE_ROWS)
    df.to_csv(AUDIT_PATH, index=False)


def auto_resolve_cases(cases: pd.DataFrame,
                       threshold: float,
                       demo_enabled: bool = True) -> pd.DataFrame:
    """
    Demo auto-resolution logic to simulate customer responses / fraud backoffice.
    """
    if not demo_enabled or cases.empty:
        return cases

    now = datetime.utcnow()
    df = cases.copy()

    pending_mask = df["status"].astype(str).str.upper().eq("PENDING")
    if not pending_mask.any():
        return df

    df_pending = df[pending_mask].copy()

    # parse timestamps, treat as naive UTC
    ts_vals = pd.to_datetime(df_pending["ts"], errors="coerce")
    age_minutes = ((now - ts_vals).dt.total_seconds() / 60.0).fillna(0)

    proba = df_pending["proba"].astype(float).fillna(0.0)

    # Rules (tweakable):
    # - Very high proba (>=0.9) and age >= 2 min  -> CONFIRMED_FRAUD  (auto)
    # - Borderline (between threshold and 0.3) and age >= 5 min -> CONFIRMED_LEGIT
    # - Medium (0.3–0.9) and age >= 10 min -> random YES/NO with bias to fraud
    hi_mask = (proba >= 0.90) & (age_minutes >= 2)
    low_mask = (proba < threshold) & (age_minutes >= 5)
    mid_mask = (proba >= threshold) & (proba < 0.90) & (age_minutes >= 10)

    # apply decisions
    for idx in df_pending.index:
        eid = int(df_pending.loc[idx, "event_id"])
        p = proba.loc[idx]
        ts_str = str(df_pending.loc[idx, "ts"])
        old_status = str(df_pending.loc[idx, "status"])

        if hi_mask.loc[idx]:
            new_status = "CONFIRMED_FRAUD"
            cust_resp = "NO"
        elif low_mask.loc[idx]:
            new_status = "CONFIRMED_LEGIT"
            cust_resp = "YES"
        elif mid_mask.loc[idx]:
            # biased coin: 70% fraud, 30% legit for mid band
            if np.random.rand() < 0.7:
                new_status = "CONFIRMED_FRAUD"
                cust_resp = "NO"
            else:
                new_status = "CONFIRMED_LEGIT"
                cust_resp = "YES"
        else:
            continue  # still pending

        df.loc[df["event_id"] == eid, "status"] = new_status
        df.loc[df["event_id"] == eid, "customer_response"] = cust_resp
        df.loc[df["event_id"] == eid, "resolution_source"] = "AUTO_DEMO"
        df.loc[df["event_id"] == eid, "updated_at"] = now.isoformat(
            timespec="seconds"
        )

        append_audit_row(
            event_id=eid,
            old_status=old_status,
            new_status=new_status,
            source="AUTO_DEMO",
            proba=p,
            ts=ts_str,
        )

    df.to_csv(CASES_PATH, index=False)
    return df


# ---------- header ----------
st.title("Adaptive Fraud Detection Engine ▸ Real-Time ML + Auto-Case Resolution")

# Build caption with active model info (if available)
model_name = "RandomForest (legacy)"
model_auc = None
if model_meta:
    active_key = model_meta.get("active_model")
    models_info = model_meta.get("models", {}) or {}
    info = models_info.get(active_key, {})
    # try a friendly name if present
    model_name = info.get("name") or active_key or model_name
    auc_val = info.get("auc")
    if isinstance(auc_val, (int, float)):
        model_auc = auc_val

caption_parts = [
    f"Default operating threshold: **{default_threshold:.4f}**",
    f"Cost ratio: FN = {C_FN}×, FP = {C_FP}×",
    f"Active model: {model_name}" + (f" (AUC={model_auc:.4f})" if model_auc is not None else ""),
]
st.caption("  |  ".join(caption_parts))

# exact 4-decimal threshold selector
options = [i / 10000 for i in range(0, 2001)]  # 0.0000 .. 0.2000
threshold = st.select_slider(
    "Operating threshold (demo override)",
    options=options,
    value=min(options, key=lambda v: abs(v - default_threshold)),
    format_func=lambda v: f"{v:.4f}",
    help="Overrides the default threshold from artifacts/threshold.json for demo purposes.",
)
st.caption(f"Current threshold: **{threshold:.4f}**")

# ---------- top “tabs” ----------
selected = option_menu(
    None,
    ["Live Feed", "Ops Analytics", "Batch Scoring", "Single Transaction", "Cases"],
    icons=["activity", "bar-chart", "cloud-upload", "sliders", "folder-check"],
    orientation="horizontal",
    default_index=0,
)

# =========================================================
# ======================== LIVE FEED ======================
# =========================================================
if selected == "Live Feed":
    st_autorefresh(interval=2000, key="live_tick")

    st.subheader("Live Events")
    ensure_live_log_schema()
    trim_csv(LIVE_LOG, MAX_LIVE_ROWS, "event_id")

    if LIVE_LOG.exists():
        try:
            feed = pd.read_csv(LIVE_LOG)
            if not feed.empty:
                feed = feed.sort_values("event_id", ascending=False).head(300)

                df_disp = feed[["event_id", "ts", "proba", "decision"]].copy()
                st.table(style_livefeed(df_disp))

                with st.expander("🔎 Inspect payload JSON for a specific row"):
                    feed["_label"] = (
                        feed["event_id"].astype(str)
                        + " | "
                        + feed["ts"].astype(str)
                        + " | "
                        + feed["decision"].astype(str)
                        + " | "
                        + (feed["proba"] * 100).round(2).astype(str)
                        + "%"
                    )
                    idx = st.selectbox(
                        "Select a row",
                        options=list(feed.index),
                        format_func=lambda i: feed.loc[i, "_label"],
                    )
                    raw = feed.loc[idx, "payload"]
                    try:
                        payload_obj = json.loads(raw) if isinstance(raw, str) else raw
                    except Exception:
                        payload_obj = {"ERROR": {"message": "src property must be a valid json object"}}
                    st.json(payload_obj)
            else:
                st.info("Waiting for events… start the simulator.")
        except Exception as e:
            st.error("Failed to read live log.")
            st.exception(e)
    else:
        st.info("No live events yet. Start the scoring service and simulator.")

# =========================================================
# ===================== OPS ANALYTICS =====================
# =========================================================
elif selected == "Ops Analytics":
    st.subheader("Operational Analytics")
    ensure_live_log_schema()
    trim_csv(LIVE_LOG, MAX_LIVE_ROWS, "event_id")

    if not LIVE_LOG.exists():
        st.info("No live events yet. Start the scoring service and simulator.")
    else:
        try:
            df = pd.read_csv(LIVE_LOG)
            if df.empty:
                st.info("No events to analyse yet.")
            else:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                df = df.dropna(subset=["ts"])
                df["hour"] = df["ts"].dt.hour
                df["dow"] = df["ts"].dt.day_name()
                df["is_review"] = (
                    df["decision"].astype(str).str.upper() == "REVIEW"
                ).astype(int)

                # ---------- Event-level KPIs ----------
                col1, col2, col3, col4 = st.columns(4)
                last60 = df[df["ts"] >= df["ts"].max() - pd.Timedelta(minutes=60)]

                with col1:
                    st.metric("Tx / min (last 60m)", f"{len(last60) / 60:.1f}")

                with col2:
                    st.metric("Review rate (overall)", f"{df['is_review'].mean():.1%}")

                with col3:
                    st.metric(
                        "Avg proba (REVIEW)",
                        f"{df.loc[df.is_review == 1, 'proba'].mean():.2%}",
                    )

                with col4:
                    st.metric(
                        "p95 proba (REVIEW)",
                        f"{df.loc[df.is_review == 1, 'proba'].quantile(0.95):.2%}",
                    )

                # ---------- Case-level KPIs ----------
                cases = sync_cases_from_live()
                if not cases.empty:
                    cases["status"] = cases["status"].astype(str).str.upper()
                    total_cases = len(cases)
                    pending_cases = cases[cases["status"] == "PENDING"]
                    resolved = cases[cases["status"] != "PENDING"]
                    confirmed_fraud = cases[cases["status"] == "CONFIRMED_FRAUD"]
                    confirmed_legit = cases[cases["status"] == "CONFIRMED_LEGIT"]

                    resolution_rate = (
                        len(resolved) / total_cases if total_cases > 0 else 0.0
                    )
                    confirmed_fraud_rate = (
                        len(confirmed_fraud) / len(resolved)
                        if len(resolved) > 0
                        else 0.0
                    )
                    false_positive_rate = (
                        len(confirmed_legit) / len(resolved)
                        if len(resolved) > 0
                        else 0.0
                    )
                    backlog = len(pending_cases)

                    avg_res_minutes = None
                    if len(resolved) > 0 and "updated_at" in resolved.columns:
                        try:
                            ts_dt = pd.to_datetime(resolved["ts"], errors="coerce")
                            upd_dt = pd.to_datetime(
                                resolved["updated_at"], errors="coerce"
                            )
                            mask = ts_dt.notna() & upd_dt.notna()
                            if mask.any():
                                deltas = (
                                    upd_dt[mask] - ts_dt[mask]
                                ).dt.total_seconds()
                                avg_res_minutes = deltas.mean() / 60.0
                        except Exception:
                            avg_res_minutes = None

                    c5, c6, c7, c8 = st.columns(4)
                    with c5:
                        st.metric("Total cases", total_cases)
                    with c6:
                        st.metric("Case resolution rate", f"{resolution_rate:.1%}")
                    with c7:
                        st.metric(
                            "Confirmed fraud rate",
                            f"{confirmed_fraud_rate:.1%}",
                        )
                    with c8:
                        fp_text = f"{false_positive_rate:.1%}"
                        if backlog > 0:
                            fp_text += f"  |  Backlog: {backlog}"
                        st.metric("False positive rate", fp_text)

                    if avg_res_minutes is not None:
                        st.caption(
                            f"Average time to resolution: {avg_res_minutes:.1f} minutes "
                            f"(for resolved cases with timestamps)."
                        )
                    else:
                        st.caption(
                            "Average time to resolution: not enough resolved cases with timestamps yet."
                        )
                else:
                    st.info(
                        "Case queue not initialised yet. Let the simulator run to create some REVIEW transactions."
                    )

                # ---------- charts ----------
                c1, c2 = st.columns(2)
                ts_agg = (
                    df.set_index("ts")
                    .resample("1min")["is_review"]
                    .agg(["count", "mean"])
                    .fillna(0)
                )

                # Tx per minute (Plotly)
                with c1:
                    fig1 = go.Figure()
                    fig1.add_trace(
                        go.Scatter(
                            x=ts_agg.index,
                            y=ts_agg["count"],
                            mode="lines",
                            name="Tx / min",
                        )
                    )
                    fig1.add_trace(
                        go.Scatter(
                            x=ts_agg.index,
                            y=ts_agg["count"]
                            .rolling(5, min_periods=1)
                            .mean(),
                            mode="lines",
                            name="Rolling avg (5 min)",
                            line=dict(dash="dash"),
                        )
                    )
                    fig1.update_layout(
                        title="Transactions per minute (with rolling avg)",
                        xaxis_title="Time",
                        yaxis_title="Count",
                    )
                    fig1.update_xaxes(tickangle=-30)
                    st.plotly_chart(fig1, use_container_width=True)

                # Review rate per minute (Plotly)
                with c2:
                    fig2 = go.Figure()
                    fig2.add_trace(
                        go.Scatter(
                            x=ts_agg.index,
                            y=ts_agg["mean"],
                            mode="lines",
                            name="Review rate / min",
                        )
                    )
                    fig2.add_trace(
                        go.Scatter(
                            x=ts_agg.index,
                            y=ts_agg["mean"]
                            .rolling(5, min_periods=1)
                            .mean(),
                            mode="lines",
                            name="Rolling avg (5 min)",
                            line=dict(dash="dash"),
                        )
                    )
                    fig2.update_layout(
                        title="Review rate per minute (with rolling avg)",
                        xaxis_title="Time",
                        yaxis_title="Rate",
                    )
                    fig2.update_xaxes(tickangle=-30)
                    st.plotly_chart(fig2, use_container_width=True)

                c3, c4 = st.columns(2)

                # Distribution of predicted probability (Plotly)
                with c3:
                    fig3 = px.histogram(
                        df,
                        x="proba",
                        nbins=30,
                        title="Distribution of predicted probability",
                    )
                    fig3.update_layout(
                        xaxis_title="Probability",
                        yaxis_title="Frequency",
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                # ---------- Heatmap: Review rate by Day × Hour (Plotly) ----------
                with c4:
                    pivot = (
                        df.pivot_table(
                            index="dow",
                            columns="hour",
                            values="is_review",
                            aggfunc="mean",
                        )
                        .reindex(
                            [
                                "Monday",
                                "Tuesday",
                                "Wednesday",
                                "Thursday",
                                "Friday",
                                "Saturday",
                                "Sunday",
                            ]
                        )
                    )

                    # Ensure we always have columns 0..23, even if some hours missing
                    all_hours = list(range(24))
                    pivot = pivot.reindex(columns=all_hours)

                    # If there is no data at all, avoid crashing
                    if pivot.isna().all().all():
                        st.info(
                            "Not enough data to build the day × hour heatmap yet."
                        )
                    else:
                        fig4 = px.imshow(
                            pivot.values,
                            x=[str(h) for h in pivot.columns],  # 24 labels
                            y=list(pivot.index),                # 7 labels
                            color_continuous_scale="Plasma",
                            zmin=0,
                            zmax=0.30,
                            labels={"color": "Review rate"},
                            aspect="auto",
                        )
                        fig4.update_layout(
                            title="Review rate heatmap (Day of week × Hour)",
                            xaxis_title="Hour of Day",
                            yaxis_title="Day of Week",
                        )
                        # Annotate cells with percentages
                        text = pivot.applymap(
                            lambda v: f"{v:.0%}" if pd.notna(v) else ""
                        ).values
                        fig4.update_traces(
                            text=text,
                            texttemplate="%{text}",
                            textfont_size=9,
                        )

                        st.plotly_chart(fig4, use_container_width=True)

        except Exception as e:
            st.error("Failed to compute analytics.")
            st.exception(e)

# =========================================================
# ===================== BATCH SCORING =====================
# =========================================================
elif selected == "Batch Scoring":
    st.subheader("Upload CSV to score multiple transactions")
    st.markdown(
        "- Use this when you want to score **many transactions at once** "
        "(e.g., a daily extract).\n"
        "- Click **Download CSV Template**, keep the header names unchanged, "
        "fill rows in the same order, then upload the file.\n"
        "- The app returns a scored CSV with `fraud_proba` and a business "
        "decision (`APPROVE` / `REVIEW`) at the current threshold."
    )

    st.download_button(
        "Download CSV Template",
        data=make_empty_template().to_csv(index=False),
        file_name="fraud_scoring_template.csv",
        mime="text/csv",
    )

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df_in = pd.read_csv(file)
            missing = [c for c in feature_cols if c not in df_in.columns]
            extra = [c for c in df_in.columns if c not in feature_cols]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                if extra:
                    st.info(f"Ignoring extra columns: {extra}")
                X = df_in[feature_cols].copy()
                probs = rf.predict_proba(X)[:, 1]
                preds = (probs >= threshold).astype(int)

                out = df_in.copy()
                out["fraud_proba"] = probs
                out["decision"] = np.where(preds == 1, "REVIEW", "APPROVE")

                st.write("Preview of scored results:")
                st.dataframe(out.head(20), use_container_width=True)

                st.download_button(
                    "Download Scored CSV",
                    data=out.to_csv(index=False),
                    file_name="scored_transactions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error("Failed to score the uploaded file. Check columns and data format.")
            st.exception(e)

# =========================================================
# ========================= CASES =========================
# =========================================================
elif selected == "Cases":
    st.subheader("Cases / Alerts Queue")

    # Toggle for demo auto-resolution
    demo_auto = st.checkbox(
        "Demo: auto-resolve pending cases (simulate customer responses)",
        value=True,
        help="When checked, the app periodically auto-resolves some pending "
             "cases using a realistic ruleset (for demo only).",
    )

    cases = sync_cases_from_live()
    cases = auto_resolve_cases(cases, threshold=threshold, demo_enabled=demo_auto)

    if cases.empty:
        st.info(
            "No review cases yet. Once transactions are flagged as REVIEW, they will appear here."
        )
    else:
        total_cases = len(cases)
        cases["status"] = cases["status"].astype(str).str.upper()
        pending = cases[cases["status"] == "PENDING"]
        confirmed_fraud = cases[cases["status"] == "CONFIRMED_FRAUD"]
        confirmed_legit = cases[cases["status"] == "CONFIRMED_LEGIT"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total cases", total_cases)
        with c2:
            st.metric("Pending", len(pending))
        with c3:
            st.metric("Confirmed fraud", len(confirmed_fraud))
        with c4:
            st.metric("Confirmed legit", len(confirmed_legit))

        st.markdown("### All cases")
        display_df = cases.copy()
        if "updated_at" in display_df.columns and display_df["updated_at"].notna().any():
            display_df = display_df.sort_values(
                ["updated_at", "event_id"], ascending=[False, False]
            )
        else:
            display_df = display_df.sort_values("event_id", ascending=False)

        if "proba" in display_df.columns:
            display_df["proba"] = display_df["proba"].apply(
                lambda v: f"{float(v) * 100:.2f}%" if pd.notna(v) else ""
            )

        st.dataframe(display_df, use_container_width=True)

        # Download audit log
        st.markdown("### Audit log")
        if AUDIT_PATH.exists():
            with open(AUDIT_PATH, "rb") as f:
                st.download_button(
                    "Download case audit log (CSV)",
                    data=f,
                    file_name="case_audit_log.csv",
                    mime="text/csv",
                )
        else:
            st.info("Audit log will appear once some cases are auto-resolved.")

# =========================================================
# =================== SINGLE TRANSACTION ==================
# =========================================================
else:  # "Single Transaction"
    st.subheader("Manually test one transaction")
    st.caption("Useful for debugging and seeing top contributing factors (SHAP).")

    if st.button("🎲 Prefill from random sample"):
        if samples is not None and not samples.empty:
            row = samples.sample(1).iloc[0]
            set_defaults_from_series(row)
            st.rerun()
        else:
            st.warning(
                "No demo samples found. Run train_and_export.py to recreate artifacts."
            )

    cols_to_show = [c for c in feature_cols if c.lower() != "id"]
    inputs = {}
    cols = st.columns(3)
    for i, col in enumerate(cols_to_show):
        stats = meta[col]
        m, lo, hi = float(stats["mean"]), float(stats["min"]), float(stats["max"])
        default_val = get_default_value(col, float(np.clip(m, lo, hi)))
        with cols[i % 3]:
            inputs[col] = st.number_input(
                col,
                value=default_val,
                min_value=lo,
                max_value=hi,
                step=0.01,
                format="%.4f",
            )
    if "id" in feature_cols:
        inputs["id"] = get_default_value("id", 0.0)

    if st.button("Score Transaction", type="primary"):
        x = pd.DataFrame([inputs])[feature_cols]
        proba = float(rf.predict_proba(x)[:, 1][0])
        label = int(proba >= threshold)

        st.metric(label="Fraud probability", value=f"{proba:.2%}")

        if label == 1:
            st.markdown(
                """
                <div style="background-color:#5a0e0e; padding:15px; border-radius:8px; color:#fff;">
                    <b>Decision:</b> 🚨 <span style="color:#ff4b4b;">Review (Possible Fraud)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background-color:#0e5a2b; padding:15px; border-radius:8px; color:#fff;">
                    <b>Decision:</b> ✅ <span style="color:#9eff9e;">Approved (Legit Transaction)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        try:
            shap_vals = shap_for_binary(rf, x)
            contrib = (
                pd.Series(shap_vals, index=feature_cols)
                .sort_values(key=np.abs, ascending=False)
                .head(10)
            )

            st.write("Top factors influencing this decision:")
            st.dataframe(
                pd.DataFrame({"feature": contrib.index, "contribution": contrib.values})
            )

            fig, ax = plt.subplots()
            contrib.iloc[::-1].plot(kind="barh", ax=ax)
            ax.set_title("SHAP contributions (top 10)")
            ax.set_xlabel("Contribution to fraud risk (±)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, clear_figure=True)

            st.info(
                f"Auto-summary: driven primarily by {', '.join(contrib.index[:3])}."
            )
        except Exception as e:
            st.warning("SHAP explanation could not be generated for this input.")
            st.exception(e)
