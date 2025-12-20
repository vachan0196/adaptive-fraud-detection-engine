# service.py
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Paths & constants ----------------
ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ART / "rf_model.pkl"          # fallback
FEATS_PATH = ART / "feature_cols.json"
THRESH_PATH = ART / "threshold.json"       # fallback
MODEL_META_PATH = ART / "model_meta.json"  # NEW
LOG_PATH = ART / "live_events.csv"

MAX_EVENTS = 100_000  # keep at most this many events in the log

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
    print(
        f"[service] Loaded fallback RF model {MODEL_PATH} with threshold {thr:.6f}"
    )
    return model, thr


model, THRESHOLD = _load_model_and_threshold()

# ---------------- Init live_events.csv & event_id ----------------
def _init_log_and_counter() -> int:
    """
    Ensure live_events.csv exists with the new schema and
    return the next event_id to use.
    """
    if not LOG_PATH.exists():
        pd.DataFrame(
            columns=["event_id", "ts", "decision", "proba", "payload"]
        ).to_csv(LOG_PATH, index=False)
        return 1

    try:
        df = pd.read_csv(LOG_PATH, usecols=["event_id"])
        if df.empty:
            return 1
        return int(df["event_id"].max()) + 1
    except Exception:
        # Old / incompatible file: reset it
        print("Resetting live_events.csv to new schema.")
        pd.DataFrame(
            columns=["event_id", "ts", "decision", "proba", "payload"]
        ).to_csv(LOG_PATH, index=False)
        return 1


NEXT_EVENT_ID = _init_log_and_counter()


def _get_next_event_id() -> int:
    global NEXT_EVENT_ID
    eid = NEXT_EVENT_ID
    NEXT_EVENT_ID += 1
    return eid


def _trim_live_log():
    """Keep only the most recent MAX_EVENTS rows in live_events.csv."""
    try:
        df = pd.read_csv(LOG_PATH)
        if len(df) > MAX_EVENTS:
            df = df.tail(MAX_EVENTS)
            df.to_csv(LOG_PATH, index=False)
    except Exception as e:
        # Non-fatal; log to console
        print(f"[WARN] Failed to trim live_events.csv: {e}")


# ---------------- FastAPI app ----------------
app = FastAPI(title="Fraud Scoring Service")


class Tx(BaseModel):
    data: dict  # expects exact feature names


@app.post("/score")
def score(tx: Tx):
    # Validate and order features
    try:
        x = pd.DataFrame([tx.data])[features]
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Payload missing required feature columns",
        )

    # Model prediction (RF or LightGBM, depending on active model)
    proba = float(model.predict_proba(x)[:, 1][0])
    decision = "REVIEW" if proba >= THRESHOLD else "APPROVE"

    # Create log record
    event_id = _get_next_event_id()
    rec = {
        "event_id": event_id,
        "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "decision": decision,
        "proba": proba,
        "payload": json.dumps(tx.data),
    }

    # Append to log
    pd.DataFrame([rec]).to_csv(LOG_PATH, mode="a", index=False, header=False)

    # Enforce rolling window
    _trim_live_log()

    # Return result (event_id is useful for debugging / joining)
    return {
        "event_id": event_id,
        "decision": decision,
        "proba": proba,
        "threshold": THRESHOLD,
    }
