
# app.py
import os
import json
import sqlite3
from typing import Optional, Tuple, Any

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# SQLite path (shared by FastAPI + Streamlit in your single container)
DB_PATH = "artifacts/app.db"

# FastAPI base URL (single-container: localhost)
SERVICE_URL = os.getenv("SERVICE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Adaptive Fraud Detection Engine", layout="wide")


# -------------------- DB helpers --------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def qdf(sql: str, params: Tuple = ()) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def current_run_id() -> int:
    with get_conn() as conn:
        try:
            row = conn.execute(
                "SELECT value FROM app_state WHERE key='current_run_id' LIMIT 1;"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row["value"]) if row else 0


# -------------------- Data reads --------------------
def read_live_events(run_id: int, limit: int = 2000) -> pd.DataFrame:
    # Include new SQL-friendly columns if present
    with get_conn() as conn:
        # Try with extended columns first
        try:
            rows = conn.execute(
                """
                SELECT event_id, ts, decision, proba, amount, channel, country, customer_id, payload_json
                FROM live_events
                WHERE run_id=?
                ORDER BY event_id DESC
                LIMIT ?;
                """,
                (run_id, int(limit)),
            ).fetchall()
        except sqlite3.OperationalError:
            # Backward compatibility if columns don't exist
            rows = conn.execute(
                """
                SELECT event_id, ts, decision, proba, payload_json
                FROM live_events
                WHERE run_id=?
                ORDER BY event_id DESC
                LIMIT ?;
                """,
                (run_id, int(limit)),
            ).fetchall()



    df = pd.DataFrame([dict(r) for r in rows])
    if not df.empty and "payload_json" in df.columns:
        def _safe_json(s):
            if not isinstance(s, str):
                return s
            try:
                return json.loads(s)
            except Exception:
                return None

        df["payload"] = df["payload_json"].apply(_safe_json)
    return df

# -------------------- Live feed normalization (UI schema lock) --------------------
CANON_LIVE_COLS = [
    "event_id",
    "ts",
    "decision",
    "proba",
    "amount",
    "channel",
    "country",
    "customer_id",
    "risk_band",
]

def _risk_band(p):
    try:
        if p is None or (isinstance(p, float) and pd.isna(p)):
            return "NA"
        p = float(p)
        if p >= 0.90:
            return "HIGH"
        if p >= 0.50:
            return "MED"
        return "LOW"
    except Exception:
        return "NA"

def _extract_amount(payload):
    try:
        if payload is None:
            return None
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload, dict):
            val = payload.get("Amount") or payload.get("amount")
            return float(val) if val is not None else None
    except Exception:
        return None
    return None

def normalize_live_feed_ui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forces a consistent Live Feed UI for every run_id.
    Prevents schema flipping after 'New Run ID'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=CANON_LIVE_COLS)

    out = df.copy()

    # Amount (fallback from payload if needed)
    if "amount" not in out.columns or out["amount"].isna().all():
        if "payload" in out.columns:
            out["amount"] = out["payload"].apply(_extract_amount)
        else:
            out["amount"] = None

    # Stable defaults (UI must never flip)
    if "channel" not in out.columns:
        out["channel"] = "—"
    else:
        out["channel"] = out["channel"].fillna("—")

    if "country" not in out.columns:
        out["country"] = "—"
    else:
        out["country"] = out["country"].fillna("—")

    if "customer_id" not in out.columns:
        out["customer_id"] = "—"
    else:
        out["customer_id"] = out["customer_id"].fillna("—")


    # Risk band derived from proba
    if "proba" in out.columns:
        out["risk_band"] = out["proba"].apply(_risk_band)
    else:
        out["proba"] = None
        out["risk_band"] = "NA"

    # Ensure all canonical columns exist
    for c in CANON_LIVE_COLS:
        if c not in out.columns:
            out[c] = None

    return out[CANON_LIVE_COLS]


def read_cases(run_id: int, limit: int = 2000, status: Optional[str] = None) -> pd.DataFrame:
    where = "WHERE c.run_id=?"
    params = [run_id]
    if status:
        where += " AND c.status=?"
        params.append(status)

    q1 = f"""
    SELECT
      c.event_id,
      e.ts AS event_ts,
      e.proba,
      e.decision,
      c.status,
      c.customer_response,
      c.resolution_source,
      c.created_at,
      c.updated_at
    FROM cases c
    JOIN live_events e ON e.event_id = c.event_id AND e.run_id = c.run_id
    {where}
    ORDER BY COALESCE(c.updated_at, c.created_at) DESC, c.event_id DESC
    LIMIT ?;
    """

    q2 = f"""
    SELECT
      c.event_id,
      e.ts AS event_ts,
      e.proba,
      e.decision,
      c.status,
      c.customer_response,
      NULL AS resolution_source,
      c.created_at,
      c.updated_at
    FROM cases c
    JOIN live_events e ON e.event_id = c.event_id AND e.run_id = c.run_id
    {where}
    ORDER BY COALESCE(c.updated_at, c.created_at) DESC, c.event_id DESC
    LIMIT ?;
    """

    params2 = params + [int(limit)]

    with get_conn() as conn:
        try:
            rows = conn.execute(q1, params2).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(q2, params2).fetchall()

    return pd.DataFrame([dict(r) for r in rows])


def read_audit(run_id: int, limit: int = 2000) -> pd.DataFrame:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, event_id, old_status, new_status, source, proba, ts, logged_at
            FROM case_audit_log
            WHERE run_id=?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (run_id, int(limit)),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


# -------------------- API helpers (FastAPI writes only) --------------------
def api_post(path: str, payload: dict):
    url = f"{SERVICE_URL}{path}"
    r = requests.post(url, json=payload, timeout=10)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()


# -------------------- KPI snapshot --------------------
def analytics_snapshot(run_id: int) -> dict:
    with get_conn() as conn:
        ev = conn.execute(
            """
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN decision='REVIEW' THEN 1 ELSE 0 END) AS n_review,
              SUM(CASE WHEN decision='APPROVE' THEN 1 ELSE 0 END) AS n_approve,
              AVG(proba) AS avg_proba
            FROM live_events
            WHERE run_id=?;
            """,
            (run_id,),
        ).fetchone()

        cs = conn.execute(
            """
            SELECT
              COUNT(*) AS n_cases,
              SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) AS n_pending,
              SUM(CASE WHEN status='CONFIRMED_FRAUD' THEN 1 ELSE 0 END) AS n_fraud,
              SUM(CASE WHEN status='CONFIRMED_LEGIT' THEN 1 ELSE 0 END) AS n_legit,
              SUM(CASE WHEN status='DISMISSED' THEN 1 ELSE 0 END) AS n_dismissed
            FROM cases
            WHERE run_id=?;
            """,
            (run_id,),
        ).fetchone()

    events = int(ev["n"] or 0)
    approve = int(ev["n_approve"] or 0)
    review = int(ev["n_review"] or 0)
    pending = int(cs["n_pending"] or 0)

    review_rate = (review / events * 100.0) if events else 0.0

    return {
        "events": events,
        "approve": approve,
        "review": review,
        "review_rate": review_rate,
        "avg_proba": float(ev["avg_proba"] or 0.0),
        "cases": int(cs["n_cases"] or 0),
        "pending": pending,
        "fraud": int(cs["n_fraud"] or 0),
        "legit": int(cs["n_legit"] or 0),
        "dismissed": int(cs["n_dismissed"] or 0),
    }


# ---------------- UI ----------------
st.title("Adaptive Fraud Detection Engine (Production-Grade Demo SaaS)")

# ---------- Pill-style tab CSS (COSMETIC ONLY) ----------
st.markdown(
    """
    <style>
    /* Tab container */
    div[data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 0 !important;
        padding: 8px 6px 12px 6px;
    }

    /* Tab buttons */
    button[data-baseweb="tab"] {
        border: 0 !important;
        background: rgba(255,255,255,0.06) !important;
        color: rgba(255,255,255,0.85) !important;
        padding: 10px 18px !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        line-height: 1 !important;
        transition: all 0.15s ease-in-out;
    }

    /* Hover effect */
    button[data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.12) !important;
        transform: translateY(-1px);
    }

    /* Active pill */
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #ff4b4b !important;
        color: #111 !important;
        box-shadow: 0 8px 24px rgba(255, 75, 75, 0.25) !important;
    }

    /* Remove underline indicator */
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Prevent full-width tabs */
    div[data-baseweb="tab-list"] button {
        width: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- KPI header ----------
run_id = current_run_id()
snap = analytics_snapshot(run_id)

top = st.columns([1.1, 1.1, 1.1, 1.2, 1.5, 1.3])
top[0].metric("Run ID", run_id)
top[1].metric("Live Events", snap["events"])
top[2].metric("Review Rate", f"{snap['review_rate']:.2f}%")
top[3].metric("Review Queue", snap["pending"])
top[4].metric("Approve / Review", f"{snap['approve']} / {snap['review']}")
top[5].metric("Avg Risk", f"{snap['avg_proba']:.4f}")

st.divider()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls (FastAPI writes)")
    st.caption(f"SERVICE_URL: {SERVICE_URL}")

    if st.button("Reset Demo (new run_id)", use_container_width=True):
        out = api_post("/admin/reset", {"label": "demo reset"})
        st.success(f"Reset OK → new_run_id={out['new_run_id']}")
        st.rerun()

    st.subheader("Auto-resolve (demo)")
    auto_on = st.toggle("Enable auto-resolve calls", value=True)
    high = st.slider("High fraud threshold", 0.50, 0.999, 0.95, 0.001)
    low = st.slider("Low legit threshold", 0.0, 0.20, 0.02, 0.001)
    max_per = st.number_input("Max per call", min_value=1, max_value=2000, value=200, step=50)

    if st.button("Run auto-resolve once", use_container_width=True):
        out = api_post(
            "/demo/auto_resolve",
            {"high_fraud": high, "low_legit": low, "max_per_call": int(max_per)},
        )
        st.info(f"Resolved {out['resolved']} of {out['checked']} checked")
        st.rerun()

    if auto_on:
        st_autorefresh(interval=2500, key="auto_resolve_tick")
        try:
            _ = api_post(
                "/demo/auto_resolve",
                {"high_fraud": high, "low_legit": low, "max_per_call": int(max_per)},
            )
        except Exception as e:
            st.warning(f"Auto-resolve call failed: {e}")

    st.subheader("Refresh")
    if st.button("Refresh now", use_container_width=True):
        st.rerun()

# ---------- Tabs (pill styled via CSS above) ----------
tabs = st.tabs(["Live Feed", "Cases", "Analytics", "Audit Log"])


# ---------------- Live Feed ----------------
with tabs[0]:
    st.subheader("Live Feed")
    with st.expander("What this is", expanded=False):
        st.markdown(
            """
            **Real-time transaction decision stream.**

            This view shows every transaction as it is scored by the fraud model and 
            written to the system of record.

            - Each row represents a single transaction decision.
            - Risk scores reflect the model’s confidence at decision time.
            - Most traffic is low risk; high-risk events are intentionally rare.
            - Decisions flow here first before any downstream review or case handling.
            """
        )

    with st.expander("Data lineage", expanded=False):
        st.markdown(
        """
        - Transactions are written by FastAPI into an immutable live event stream.
        - Streamlit only reads and visualises.
        """
        )

    df = read_live_events(run_id, limit=700)

    if df.empty:
        st.info("No live events yet. Warming up...")
    else:
        # 🔒 Force canonical UI schema (no Run ID UI drift)
        show = normalize_live_feed_ui(df)

        def style_rows(row):
            styles = [""] * len(row)

            # decision colour
            if "decision" in row.index:
                i = list(row.index).index("decision")
                if row["decision"] == "APPROVE":
                    styles[i] = "background-color: rgba(0, 200, 0, 0.20); font-weight: 700;"
                else:
                    styles[i] = "background-color: rgba(255, 140, 0, 0.25); font-weight: 700;"

            # proba heat
            if "proba" in row.index:
                val = row["proba"]
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return styles
                p = float(val)
                j = list(row.index).index("proba")

                if p >= 0.90:
                    styles[j] = "background-color: rgba(255, 0, 0, 0.35); font-weight: 700;"
                elif p >= 0.50:
                    styles[j] = "background-color: rgba(255, 140, 0, 0.25);"
                else:
                    styles[j] = "background-color: rgba(255,255,255,0.03); color: rgba(255,255,255,0.70);"

            return styles

        st.dataframe(show.style.apply(style_rows, axis=1), use_container_width=True)


# ---------------- Cases ----------------
with tabs[1]:
    st.subheader("Cases")
    with st.expander("What this is", expanded=False):
        st.markdown(
            """
            **Investigation workflow for flagged transactions.**

            This section represents the operational case management layer 
            used by fraud teams.

            - Only transactions flagged for review appear here.
            - Most cases are automatically resolved using risk thresholds.
            - Manual resolution exists as an exception path for edge cases.
            - Every state change is auditable and traceable.
            """
        )

    with st.expander("Data lineage", expanded=False):
        st.markdown(
            """
            - Cases are created only from events with decision = REVIEW.
            - cases.event_id references live_events.event_id (no independent cases).
            - All status changes are written by FastAPI and recorded in the audit log.
            """
        )

    status_filter = st.selectbox(
        "Filter",
        ["ALL", "PENDING", "CONFIRMED_FRAUD", "CONFIRMED_LEGIT", "DISMISSED"],
        index=0,
        key="cases_filter",
    )
    status = None if status_filter == "ALL" else status_filter

    cdf = read_cases(run_id, limit=700, status=status)
    if cdf.empty:
        st.info("No cases yet (cases are created only when decision=REVIEW).")
    else:
        st.dataframe(cdf, use_container_width=True)

    with st.expander("Analyst override (optional)", expanded=False):
        st.caption(
            "Auto-resolution is the default in this demo. "
            "This manual override exists for exceptions and human-in-the-loop realism. "
            "All overrides are logged in the audit trail."
        )

        colA, colB, colC, colD = st.columns([1.0, 1.2, 1.8, 1.0])
        event_id = colA.number_input("event_id", min_value=0, value=0, step=1)
        new_status = colB.selectbox("new_status", ["CONFIRMED_FRAUD", "CONFIRMED_LEGIT", "DISMISSED"])
        customer_response = colC.text_input("customer_response (optional)", value="")

        if colD.button("Resolve (override)", use_container_width=True):
            if event_id <= 0:
                st.error("Enter a valid event_id")
            else:
                out = api_post(
                    f"/cases/{int(event_id)}/resolve",
                    {"new_status": new_status, "customer_response": customer_response or None, "source": "manual"},
                )
                st.success(f"Resolved event_id={out['event_id']} → {out['new_status']}")
                st.rerun()



# ---------------- Analytics (Banker-grade, SQL derived) ----------------
with tabs[2]:
    st.subheader("Analytics")
    with st.expander("What this is", expanded=False):
        st.markdown(
            """
            **Operational and risk performance analytics.**

            This dashboard provides an aggregated view of fraud activity,
            model behaviour, and operational health.

            - Metrics are derived directly from live transactions and cases.
            - No pre-aggregated or cached analytics are used.
            - Designed for fraud managers and risk stakeholders.
            - Suitable for monitoring effectiveness, drift, and workload.
            """
        )

    with st.expander("Data lineage", expanded=False):
        st.markdown(
            """
            - All metrics are computed via SQL at read-time.
            - No cached analytics tables or derived state.
            """
        )

    # --- Summary cards ---
    snap = analytics_snapshot(run_id)

    # Confirmed fraud rate = confirmed_fraud / total events (lower bound)
    fraud_rate = (snap["fraud"] / snap["events"] * 100.0) if snap["events"] else 0.0
    
    # False Positive Rate (banking logic)
    # Of all alerts (cases raised), how many were actually legit?
    false_positives = snap["legit"] + snap["dismissed"]
    total_alerts = snap["cases"]

    false_pos_rate = (false_positives / total_alerts * 100.0) if total_alerts else 0.0

    # Resolution timing (median/p90 in minutes)
    res_time = qdf(
        """
        SELECT
        (julianday(updated_at) - julianday(created_at)) * 24.0 * 60.0 AS minutes
        FROM cases
        WHERE run_id=?
        AND updated_at IS NOT NULL
        AND created_at IS NOT NULL
        AND (julianday(updated_at) - julianday(created_at)) IS NOT NULL
        """,
        (run_id,),
    )

    if not res_time.empty:
        med_min = float(res_time["minutes"].median())
        p90_min = float(res_time["minutes"].quantile(0.90))
    else:
        med_min, p90_min = 0.0, 0.0

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Total Tx", snap["events"])
    c2.metric("Review Rate", f"{snap['review_rate']:.2f}%")
    c3.metric("Confirmed Fraud Rate", f"{fraud_rate:.3f}%")
    c4.metric("False Positive Rate", f"{false_pos_rate:.1f}%")
    c5.metric("Avg Risk Score", f"{snap['avg_proba']:.4f}")
    c6.metric("Queue Backlog (PENDING)", snap["pending"])
    c7.metric("Resolution Time (p50 / p90)", f"{med_min:.1f}m / {p90_min:.1f}m")

    st.caption("All metrics below are computed via SQL at read-time (no cached analytics state).")

    st.divider()

    # --- Risk distribution ---
    left, right = st.columns(2)

    with left:
        st.markdown("#### Risk score distribution (sanity check)")
        dist = qdf(
            """
            SELECT
              CASE
                WHEN proba >= 0.90 THEN 'HIGH'
                WHEN proba >= 0.50 THEN 'MED'
                ELSE 'LOW'
              END AS band,
              COUNT(*) AS n
            FROM live_events
            WHERE run_id=?
            GROUP BY band;
            """,
            (run_id,),
        )
        if dist.empty:
            st.info("No data yet.")
        else:
            st.bar_chart(dist.set_index("band"))

    with right:
        st.markdown("#### Approve vs Review trend (by minute)")
        trend = qdf(
            """
            SELECT
              substr(ts, 1, 16) AS minute,
              SUM(CASE WHEN decision='APPROVE' THEN 1 ELSE 0 END) AS approve,
              SUM(CASE WHEN decision='REVIEW' THEN 1 ELSE 0 END) AS review
            FROM live_events
            WHERE run_id=?
            GROUP BY minute
            ORDER BY minute ASC
            LIMIT 240;
            """,
            (run_id,),
        )
        if trend.empty:
            st.info("No data yet.")
        else:
            trend = trend.set_index("minute")
            st.line_chart(trend[["approve", "review"]])

    st.divider()

    # --- Fraud patterns: by hour, by amount bucket ---
    left2, right2 = st.columns(2)

    with left2:
        st.markdown("#### Confirmed fraud by hour (UTC)")
        by_hour = qdf(
            """
            SELECT
              substr(e.ts, 12, 2) AS hour_utc,
              SUM(CASE WHEN c.status='CONFIRMED_FRAUD' THEN 1 ELSE 0 END) AS confirmed_fraud,
              COUNT(*) AS total,
              AVG(e.proba) AS avg_risk
            FROM live_events e
            LEFT JOIN cases c ON c.event_id=e.event_id AND c.run_id=e.run_id
            WHERE e.run_id=?
            GROUP BY hour_utc
            ORDER BY hour_utc ASC;
            """,
            (run_id,),
        )
        if by_hour.empty:
            st.info("No data yet.")
        else:
            by_hour = by_hour.set_index("hour_utc")
            st.line_chart(by_hour[["confirmed_fraud", "total"]])

    with right2:
        st.markdown("#### Fraud by amount bucket")
        # Only works if amount column exists; safe fallback handled
        try:
            amt = qdf(
                """
                SELECT
                  CASE
                    WHEN e.amount IS NULL THEN 'UNKNOWN'
                    WHEN e.amount < 10 THEN '<£10'
                    WHEN e.amount < 50 THEN '£10-£49'
                    WHEN e.amount < 200 THEN '£50-£199'
                    WHEN e.amount < 1000 THEN '£200-£999'
                    ELSE '£1000+'
                  END AS bucket,
                  COUNT(*) AS total,
                  SUM(CASE WHEN c.status='CONFIRMED_FRAUD' THEN 1 ELSE 0 END) AS confirmed_fraud,
                  SUM(CASE WHEN e.decision='REVIEW' THEN 1 ELSE 0 END) AS reviewed
                FROM live_events e
                LEFT JOIN cases c ON c.event_id=e.event_id AND c.run_id=e.run_id
                WHERE e.run_id=?
                GROUP BY bucket
                ORDER BY total DESC;
                """,
                (run_id,),
            )
            if amt.empty:
                st.info("No data yet.")
            else:
                amt = amt.set_index("bucket")
                st.bar_chart(amt[["total", "reviewed", "confirmed_fraud"]])
        except Exception:
            st.info("Amount analytics not available (live_events.amount column not present).")

    st.divider()

    # --- Ops health: status breakdown + auto vs manual + backlog over time ---
    l3, r3 = st.columns(2)

    with l3:
        st.markdown("#### Case outcomes (workflow effectiveness)")
        outcomes = qdf(
            """
            SELECT status, COUNT(*) AS n
            FROM cases
            WHERE run_id=?
            GROUP BY status
            ORDER BY n DESC;
            """,
            (run_id,),
        )
        if outcomes.empty:
            st.info("No cases yet.")
        else:
            st.bar_chart(outcomes.set_index("status"))

    with r3:
        st.markdown("#### Resolution source (auto vs manual)")
        try:
            src = qdf(
                """
                SELECT
                COALESCE(resolution_source, 'UNKNOWN') AS source,
                COUNT(*) AS n
                FROM cases
                WHERE run_id=? AND status != 'PENDING'
                GROUP BY source
                ORDER BY n DESC;
                """,
                (run_id,),
            )
            if src.empty:
                st.info("No resolved cases yet.")
            else:
                st.bar_chart(src.set_index("source"))
        except Exception:
            st.info("Resolution source analytics not available (cases.resolution_source column not present).")

    st.markdown("#### Review queue health (PENDING by minute)")
    backlog = qdf(
        """
        SELECT
          substr(created_at, 1, 16) AS minute,
          SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) AS pending_created
        FROM cases
        WHERE run_id=?
        GROUP BY minute
        ORDER BY minute ASC
        LIMIT 240;
        """,
        (run_id,),
    )
    if backlog.empty:
        st.info("No cases yet.")
    else:
        st.line_chart(backlog.set_index("minute")[["pending_created"]])


# ---------------- Audit Log ----------------
with tabs[3]:
    st.subheader("Audit Log")
    with st.expander("What this is", expanded=False):
        st.markdown(
            """
            **Compliance and audit trail for fraud decisions.**

            This log records every meaningful state transition in the system.

            - Includes automated and manual resolutions.
            - Captures old vs new status with timestamps.
            - Preserves the risk score at the time of decision.
            - Supports regulatory review, internal audits, and post-incident analysis.
            """
        )


    # Simple filters (UI-only; DB still authoritative)
    f1, f2, f3 = st.columns([1.2, 1.2, 2.0])
    source_filter = f1.selectbox("Source", ["ALL", "system", "auto", "manual"], index=0, key="audit_source")
    status_filter = f2.selectbox("New Status", ["ALL", "PENDING", "CONFIRMED_FRAUD", "CONFIRMED_LEGIT", "DISMISSED"], index=0, key="audit_status")
    limit = int(f3.slider("Rows", 100, 2000, 500, 100))

    where = "WHERE run_id=?"
    params = [run_id]

    if source_filter != "ALL":
        where += " AND source=?"
        params.append(source_filter)

    if status_filter != "ALL":
        where += " AND new_status=?"
        params.append(status_filter)

    adf = qdf(
        f"""
        SELECT id, event_id, old_status, new_status, source, proba, ts, logged_at
        FROM case_audit_log
        {where}
        ORDER BY id DESC
        LIMIT ?;
        """,
        tuple(params + [limit]),
    )

    if adf.empty:
        st.info("No audit entries yet.")
    else:
        st.dataframe(adf, use_container_width=True)
 
