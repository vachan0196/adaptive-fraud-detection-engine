# service.py
import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# SQLite storage helpers (you must have storage.py in the same folder)
from storage import init_db, insert_live_event, get_conn


# ---------------- Paths & constants ----------------
ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ART / "rf_model.pkl"          # fallback
FEATS_PATH = ART / "feature_cols.json"
THRESH_PATH = ART / "threshold.json"       # fallback
MODEL_META_PATH = ART / "model_meta.json"  # preferred selector

MAX_EVENTS = 100_000  # keep at most this many events in the DB table


# ---------------- Load features ----------------
features = json.loads(FEATS_PATH.read_text())


def _load_model_and_threshold():
    """
    Load the active model + its threshold from model_meta.json if present.
    Fallback to RF + threshold.json otherwise.
    """
    # Preferred path: model_meta.json
    if MODEL_META_PATH.exists():
        try:
            mm = json.loads(MODEL_META_PATH.read_text())
            active = mm.get("active_model", "random_forest")
            models = mm.get("models", {})
            conf = models.get(active)
            if conf and "path" in conf and "best_threshold" in conf:
                model_file = ART / conf["path"]
                if model_file.exists():
                    model = joblib.load(model_file)
                    thr = float(conf["best_threshold"])
                    print(
                        f"[service] Loaded active model '{active}' "
                        f"from {model_file} with threshold {thr:.6f}"
                    )
                    return model, thr
        except Exception as e:
            print(f"[service] Failed to read model_meta.json, falling back. Error: {e}")

    # Fallback: RF + threshold.json
    model = joblib.load(MODEL_PATH)
    if THRESH_PATH.exists():
        thr_conf = json.loads(THRESH_PATH.read_text())
        thr = float(thr_conf.get("threshold", 0.025))
    else:
        thr = 0.025

    print(f"[service] Loaded fallback RF model {MODEL_PATH} with threshold {thr:.6f}")
    return model, thr


model, THRESHOLD = _load_model_and_threshold()


def _trim_live_events_db():
    """
    Keep only the most recent MAX_EVENTS rows in SQLite.
    This is safe and avoids the DB growing forever.
    """
    try:
        with get_conn() as conn:
            # Delete anything older than the last MAX_EVENTS event_ids
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
        # Non-fatal
        print(f"[WARN] Failed to trim live_events table: {e}")


# ---------------- FastAPI app ----------------
app = FastAPI(title="Fraud Scoring Service")

# Ensure DB/tables exist (safe to call on every startup)
init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


class Tx(BaseModel):
    data: dict  # expects exact feature names


@app.post("/score")
def score(tx: Tx):
    # Validate and order features
    try:
        payload = dict(tx.data)

        # Backward compatible: if older artifacts still expect "id", neutralize it.
        if "id" in features:
            payload["id"] = 0.0
        else:
            payload.pop("id", None)

        x = pd.DataFrame([payload])[features]

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Payload missing required feature columns",
        )

    # Model prediction (RF or other, depending on active model)
    proba = float(model.predict_proba(x)[:, 1][0])
    decision = "REVIEW" if proba >= THRESHOLD else "APPROVE"

    # Write event atomically to SQLite (no CSV corruption possible)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    event_id = insert_live_event(
        ts=ts,
        decision=decision,
        proba=proba,
        payload=tx.data,   # store original payload
    )

    # Keep DB size bounded
    _trim_live_events_db()

    # Return result
    return {
        "event_id": event_id,
        "decision": decision,
        "proba": proba,
        "threshold": THRESHOLD,
    }
