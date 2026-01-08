# service.py
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from storage import (
    init_db,
    get_conn,
    get_current_run_id,
    start_new_run,
    insert_live_event,
    ensure_case_for_event,
    resolve_case,
    prune_run,
    reset_demo,
)

ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ART / "rf_model.pkl"          # fallback
FEATS_PATH = ART / "feature_cols.json"
THRESH_PATH = ART / "threshold.json"       # fallback
MODEL_META_PATH = ART / "model_meta.json"  # preferred selector

MAX_EVENTS = 100_000  # cap per run

app = FastAPI(title="Fraud Scoring Service")

# Globals
MODEL = None
THRESHOLD = None
FEATURES = None
LOAD_ERROR = None


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_model_and_threshold():
    """
    Load active model + threshold from model_meta.json (preferred).
    Fallback to rf_model.pkl + threshold.json.
    """
    if MODEL_META_PATH.exists():
        mm = json.loads(MODEL_META_PATH.read_text())
        active = mm.get("active_model", "random_forest")
        models = mm.get("models", {}) or {}
        conf = models.get(active)
        if conf and "path" in conf and "best_threshold" in conf:
            model_file = ART / conf["path"]
            if model_file.exists():
                model = joblib.load(model_file)
                thr = float(conf["best_threshold"])
                print(f"[service] Loaded active model '{active}' from {model_file} thr={thr:.6f}")
                return model, thr

    model = joblib.load(MODEL_PATH)
    if THRESH_PATH.exists():
        thr_conf = json.loads(THRESH_PATH.read_text())
        thr = float(thr_conf.get("threshold", 0.025))
    else:
        thr = 0.025

    print(f"[service] Loaded fallback RF model {MODEL_PATH} thr={thr:.6f}")
    return model, thr


@app.on_event("startup")
def _startup():
    global MODEL, THRESHOLD, FEATURES, LOAD_ERROR

    # Always init DB (health should work even if artifacts broken)
    init_db()

    # Ensure current run exists
    _ = get_current_run_id()

    try:
        FEATURES = json.loads(FEATS_PATH.read_text())
        MODEL, THRESHOLD = _load_model_and_threshold()
        LOAD_ERROR = None
        print("[service] Startup OK.")
    except Exception as e:
        MODEL = None
        THRESHOLD = None
        FEATURES = None
        LOAD_ERROR = str(e)
        print(f"[service] Startup DEGRADED: {LOAD_ERROR}")


@app.get("/health")
def health():
    if LOAD_ERROR:
        return {"status": "degraded", "error": LOAD_ERROR}
    return {"status": "ok"}


class Tx(BaseModel):
    data: dict  # expects feature names


class ResolveCaseIn(BaseModel):
    new_status: str = Field(..., description="CONFIRMED_FRAUD | CONFIRMED_LEGIT | DISMISSED")
    customer_response: Optional[str] = None
    source: str = Field(default="manual")


class AutoResolveIn(BaseModel):
    high_fraud: float = 0.95
    low_legit: float = 0.02
    max_per_call: int = 200


class ResetIn(BaseModel):
    label: str = "demo reset"


@app.post("/score")
def score(tx: Tx):
    if LOAD_ERROR or MODEL is None or FEATURES is None or THRESHOLD is None:
        raise HTTPException(status_code=503, detail="Service not ready (missing artifacts/model).")

    # Validate and order features
    try:
        payload = dict(tx.data)

        # Backward compatible: if old artifacts expected "id", neutralise it.
        if "id" in FEATURES:
            payload["id"] = 0.0
        else:
            payload.pop("id", None)

        x = pd.DataFrame([payload])[FEATURES]
    except Exception:
        raise HTTPException(status_code=400, detail="Payload missing required feature columns")

    proba = float(MODEL.predict_proba(x)[:, 1][0])
    decision = "REVIEW" if proba >= THRESHOLD else "APPROVE"

    run_id = get_current_run_id()
    ts = _utc_iso()

    # 1) Always write live event
    event_id = insert_live_event(
        run_id=run_id,
        ts=ts,
        decision=decision,
        proba=proba,
        payload=tx.data,
    )

    # 2) If REVIEW, create case row referencing the live event
    if decision == "REVIEW":
        ensure_case_for_event(run_id=run_id, event_id=event_id, created_at=ts)

    # 3) Prune safely (FK cascade handles cases/audit)
    prune_run(run_id=run_id, max_events=MAX_EVENTS)

    return {
        "run_id": run_id,
        "event_id": event_id,
        "decision": decision,
        "proba": proba,
        "threshold": THRESHOLD,
    }


@app.get("/cases")
def list_cases(status: Optional[str] = None, limit: int = 200):
    """
    Convenience endpoint (optional). Streamlit can also read SQLite directly.
    """
    run_id = get_current_run_id()
    limit = max(1, min(int(limit), 2000))

    where = "WHERE c.run_id=?"
    params: List[Any] = [run_id]
    if status:
        where += " AND c.status=?"
        params.append(status)

    q = f"""
    SELECT
      c.event_id,
      c.run_id,
      c.status,
      c.customer_response,
      c.resolution_source,
      c.created_at,
      c.updated_at,
      e.ts AS event_ts,
      e.decision,
      e.proba
    FROM cases c
    JOIN live_events e ON e.event_id = c.event_id
    {where}
    ORDER BY COALESCE(c.updated_at, c.created_at) DESC, c.event_id DESC
    LIMIT ?;
    """
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(q, params).fetchall()

    return {"run_id": run_id, "cases": [dict(r) for r in rows]}


@app.post("/cases/{event_id}/resolve")
def resolve_case_endpoint(event_id: int, body: ResolveCaseIn):
    run_id = get_current_run_id()
    now = _utc_iso()

    allowed = {"CONFIRMED_FRAUD", "CONFIRMED_LEGIT", "DISMISSED"}
    if body.new_status not in allowed:
        raise HTTPException(status_code=400, detail=f"new_status must be one of {sorted(list(allowed))}")

    # Grab proba + ts for audit
    with get_conn() as conn:
        ev = conn.execute(
            "SELECT proba, ts FROM live_events WHERE run_id=? AND event_id=? LIMIT 1;",
            (run_id, int(event_id)),
        ).fetchone()
    if not ev:
        raise HTTPException(status_code=404, detail="Live event not found for this run_id")

    resolve_case(
        run_id=run_id,
        event_id=int(event_id),
        new_status=body.new_status,
        customer_response=body.customer_response,
        source=body.source,
        proba=float(ev["proba"]) if ev["proba"] is not None else None,
        ts=str(ev["ts"]) if ev["ts"] is not None else None,
        logged_at=now,
    )

    return {"ok": True, "run_id": run_id, "event_id": event_id, "new_status": body.new_status}


@app.post("/demo/auto_resolve")
def auto_resolve(body: AutoResolveIn):
    """
    Demo automation: resolves a subset of PENDING cases based on proba bands.
    - proba >= high_fraud => CONFIRMED_FRAUD
    - proba <= low_legit  => CONFIRMED_LEGIT
    Everything else stays PENDING.
    """
    run_id = get_current_run_id()
    now = _utc_iso()

    high = float(body.high_fraud)
    low = float(body.low_legit)
    max_per = max(1, min(int(body.max_per_call), 2000))

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.event_id, e.proba, e.ts
            FROM cases c
            JOIN live_events e ON e.event_id = c.event_id
            WHERE c.run_id=? AND c.status='PENDING'
            ORDER BY c.event_id ASC
            LIMIT ?;
            """,
            (run_id, max_per),
        ).fetchall()

    changed = 0
    for r in rows:
        event_id = int(r["event_id"])
        proba = float(r["proba"])
        ts = str(r["ts"])

        if proba >= high:
            new_status = "CONFIRMED_FRAUD"
        elif proba <= low:
            new_status = "CONFIRMED_LEGIT"
        else:
            continue

        resolve_case(
            run_id=run_id,
            event_id=event_id,
            new_status=new_status,
            customer_response=None,
            source="auto",
            proba=proba,
            ts=ts,
            logged_at=now,
        )
        changed += 1

    return {"ok": True, "run_id": run_id, "resolved": changed, "checked": len(rows), "high_fraud": high, "low_legit": low}


@app.post("/admin/reset")
def admin_reset(body: ResetIn):
    """
    Soft reset: starts a new run_id. No historical confusion in UI if Streamlit scopes to current run_id.
    """
    new_run = reset_demo(label=body.label)
    return {"ok": True, "new_run_id": new_run}
