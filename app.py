# app.py
import os
import json
import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
import streamlit as st

DB_PATH = "artifacts/app.db"
SERVICE_URL = os.getenv("SERVICE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Adaptive Fraud Detection Engine", layout="wide")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def current_run_id() -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key='current_run_id' LIMIT 1;").fetchone()
    return int(row["value"]) if row else 0


def read_live_events(run_id: int, limit: int = 2000) -> pd.DataFrame:
    with get_conn() as conn:
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
    if not df.empty:
        df["payload"] = df["payload_json"].apply(lambda s: json.loads(s) if isinstance(s, str) else s)
    return df


def read_cases(run_id: int, limit: int = 2000, status: Optional[str] = None) -> pd.DataFrame:
    where = "WHERE c.run_id=?"
    params = [run_id]
    if status:
        where += " AND c.status=?"
        params.append(status)

    q = f"""
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
    JOIN live_events e ON e.event_id = c.event_id
    {where}
    ORDER BY COALESCE(c.updated_at, c.created_at) DESC, c.event_id DESC
    LIMIT ?;
    """
    params.append(int(limit))

    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()
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

    return {
        "events": int(ev["n"] or 0),
        "approve": int(ev["n_approve"] or 0),
        "review": int(ev["n_review"] or 0),
        "avg_proba": float(ev["avg_proba"] or 0.0),
        "cases": int(cs["n_cases"] or 0),
        "pending": int(cs["n_pending"] or 0),
        "fraud": int(cs["n_fraud"] or 0),
        "legit": int(cs["n_legit"] or 0),
        "dismissed": int(cs["n_dismissed"] or 0),
    }


def api_post(path: str, payload: dict):
    url = f"{SERVICE_URL}{path}"
    r = requests.post(url, json=payload, timeout=10)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()


# ---------------- UI ----------------
st.title("Adaptive Fraud Detection Engine (Demo SaaS)")

run_id = current_run_id()

top = st.columns([1.2, 1.2, 1.2, 1.2, 2.2])
snap = analytics_snapshot(run_id)

top[0].metric("Run ID", run_id)
top[1].metric("Live Events", snap["events"])
top[2].metric("Review Queue", snap["pending"])
top[3].metric("Approve / Review", f'{snap["approve"]} / {snap["review"]}')
top[4].metric("Avg Risk", f'{snap["avg_proba"]:.4f}')

st.divider()

with st.sidebar:
    st.header("Controls (via FastAPI)")
    st.caption(f"SERVICE_URL: {SERVICE_URL}")

    if st.button("Reset Demo (new run_id)", use_container_width=True):
        out = api_post("/admin/reset", {"label": "demo reset"})
        st.success(f"Reset OK → new_run_id={out['new_run_id']}")
        st.rerun()

    st.subheader("Auto-resolve (demo)")
    auto_on = st.toggle("Enable auto-resolve calls", value=False)
    high = st.slider("High fraud threshold", 0.50, 0.999, 0.95, 0.001)
    low = st.slider("Low legit threshold", 0.0, 0.20, 0.02, 0.001)
    max_per = st.number_input("Max per call", min_value=1, max_value=2000, value=200, step=50)

    if st.button("Run auto-resolve once", use_container_width=True):
        out = api_post("/demo/auto_resolve", {"high_fraud": high, "low_legit": low, "max_per_call": int(max_per)})
        st.info(f"Resolved {out['resolved']} of {out['checked']} checked")
        st.rerun()

    if auto_on:
        try:
            _ = api_post("/demo/auto_resolve", {"high_fraud": high, "low_legit": low, "max_per_call": int(max_per)})
        except Exception as e:
            st.warning(f"Auto-resolve call failed: {e}")

    st.subheader("Refresh")
    refresh = st.button("Refresh now", use_container_width=True)
    if refresh:
        st.rerun()


tabs = st.tabs(["Live Feed", "Cases", "Analytics", "Audit Log"])

with tabs[0]:
    st.subheader("Live Feed (SQLite: live_events)")
    df = read_live_events(run_id, limit=500)
    if df.empty:
        st.info("No live events yet. Start the simulator.")
    else:
        st.dataframe(df.drop(columns=["payload_json"], errors="ignore"), use_container_width=True)

with tabs[1]:
    st.subheader("Cases (SQLite: cases JOIN live_events)")
    status_filter = st.selectbox("Filter", ["ALL", "PENDING", "CONFIRMED_FRAUD", "CONFIRMED_LEGIT", "DISMISSED"], index=0)
    status = None if status_filter == "ALL" else status_filter

    cdf = read_cases(run_id, limit=500, status=status)
    if cdf.empty:
        st.info("No cases yet (cases are created only when decision=REVIEW).")
    else:
        st.dataframe(cdf, use_container_width=True)

    st.markdown("### Resolve a case (FastAPI writes, Streamlit triggers)")
    colA, colB, colC, colD = st.columns([1.0, 1.2, 1.8, 1.0])
    event_id = colA.number_input("event_id", min_value=0, value=0, step=1)
    new_status = colB.selectbox("new_status", ["CONFIRMED_FRAUD", "CONFIRMED_LEGIT", "DISMISSED"])
    customer_response = colC.text_input("customer_response (optional)", value="")
    if colD.button("Resolve", use_container_width=True):
        if event_id <= 0:
            st.error("Enter a valid event_id")
        else:
            out = api_post(
                f"/cases/{int(event_id)}/resolve",
                {
                    "new_status": new_status,
                    "customer_response": customer_response or None,
                    "source": "manual",
                },
            )
            st.success(f"Resolved event_id={out['event_id']} → {out['new_status']}")
            st.rerun()

with tabs[2]:
    st.subheader("Analytics (derived from live_events + cases)")
    snap = analytics_snapshot(run_id)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Events", snap["events"])
    a2.metric("Cases", snap["cases"])
    a3.metric("Pending", snap["pending"])
    a4.metric("Confirmed Fraud", snap["fraud"])

    st.caption("All analytics are derived by querying SQLite. No analytics state is written.")

with tabs[3]:
    st.subheader("Audit Log (SQLite: case_audit_log)")
    adf = read_audit(run_id, limit=500)
    if adf.empty:
        st.info("No audit entries yet.")
    else:
        st.dataframe(adf, use_container_width=True)
