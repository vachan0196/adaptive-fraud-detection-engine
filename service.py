# service.py
import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from storage import init_db, insert_live_event, get_conn

ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ART / "rf_model.pkl"          # fallback
FEATS_PATH = ART / "feature_cols.json"
THRESH_PATH = ART / "threshold.json"       # fallback
MODEL_META_PATH = ART / "model_meta.json"  # preferred selector

MAX_EVENTS = 100_000  # hard cap

app = FastAPI(title="Fraud Scoring Service")

# Globals (populated at startup)
MODEL = None
THRESHOLD = None
FEATURES = None
LOAD_ERROR = None


def _load_model_and_threshold():
    """
    Load the active model + its threshold from model_meta.json if present.
    Fallback to RF + threshold.json otherwise.
    """
    # Preferred path: model_meta.json
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

    # Fallback: RF + threshold.json
    model = joblib.load(MODEL_PATH)
    if THRESH_PATH.exists():
        thr_conf = json.loads(THRESH_PATH.read_text())
        thr = float(thr_conf.get("threshold", 0.025))
    else:
        thr = 0.025

    print(f"[service] Loaded fallback RF model {MODEL_PATH} thr={thr:.6f}")
    return model, thr


def _trim_live_events_db():
    """Keep only the most recent MAX_EVENTS rows in SQLite."""
    try:
        with get_conn() as conn:
            conn.execute(
                """
                DELETE FROM live_events
                WHERE event_id NOT IN (
                    SELECT event_id FROM live_events
                    ORDER BY event_id DESC
                    LIMIT ?
                )
                """,
                (int(MAX_EVENTS),),
            )
    except Exception as e:
        print(f"[WARN] Failed to trim live_events table: {e}")


@app.on_event("startup")
def _startup():
    global MODEL, THRESHOLD, FEATURES, LOAD_ERROR

    # DB always initialised (so /health works even if artifacts are broken)
    init_db()

    try:
        FEATURES = json.loads(FEATS_PATH.read_text())
        MODEL, THRESHOLD = _load_model_and_threshold()
        LOAD_ERROR = None
        print("[service] Startup OK.")
    except Exception as e:
        # Do NOT crash the service; keep it up and report degraded health.
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
    data: dict  # expects exact feature names


@app.post("/score")
def score(tx: Tx):
    if LOAD_ERROR or MODEL is None or FEATURES is None or THRESHOLD is None:
        raise HTTPException(status_code=503, detail="Service not ready (missing artifacts/model).")

    # Validate and order features
    try:
        payload = dict(tx.data)

        # Backward compatible: if older artifacts still expect "id", neutralise it.
        if "id" in FEATURES:
            payload["id"] = 0.0
        else:
            payload.pop("id", None)

        x = pd.DataFrame([payload])[FEATURES]
    except Exception:
        raise HTTPException(status_code=400, detail="Payload missing required feature columns")

    proba = float(MODEL.predict_proba(x)[:, 1][0])
    decision = "REVIEW" if proba >= THRESHOLD else "APPROVE"

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    event_id = insert_live_event(
        ts=ts,
        decision=decision,
        proba=proba,
        payload=tx.data,
    )

    _trim_live_events_db()

    return {
        "event_id": event_id,
        "decision": decision,
        "proba": proba,
        "threshold": THRESHOLD,
    }
