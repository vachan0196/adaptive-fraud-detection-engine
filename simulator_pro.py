"""
Lightweight transaction simulator for the Adaptive Fraud Detection Engine.

- Sends ~0.5 events/second (1 tx / 2 seconds) to the FastAPI scoring service
- Uses artifacts/sim_pool.parquet as the source transaction pool
- Uses artifacts/feature_cols.json to build the feature payload
"""

import json
import logging
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------
# Config
# ---------------------------

ART_DIR = Path("artifacts")
SIM_POOL_PATH = ART_DIR / "sim_pool.parquet"
FEATURE_COLS_PATH = ART_DIR / "feature_cols.json"

# Target events per second (you asked for ~1 tx every 2 sec)
EVENTS_PER_SECOND = 0.5          # 0.5 events/sec = 1 tx per 2 seconds
SLEEP_BETWEEN_EVENTS = 1.0 / EVENTS_PER_SECOND

# FastAPI scoring endpoint inside the same container
# If running via Docker Compose with a separate service container,
# you can override this with SERVICE_URL="http://service:8000/score"
SERVICE_URL = os.getenv("SERVICE_URL", "http://127.0.0.1:8000/score")

# How many rows from sim_pool to keep in memory (for small EC2 types)
MAX_ROWS_IN_MEMORY = 50_000

# How often to print stats (seconds)
LOG_INTERVAL = 30


# ---------------------------
# Logging
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[sim] %(message)s",
)
log = logging.getLogger("simulator")


# ---------------------------
# Helpers
# ---------------------------

def load_artifacts():
    """Load sim_pool + feature columns in a memory-friendly way."""
    if not SIM_POOL_PATH.exists():
        raise FileNotFoundError(f"sim_pool parquet not found at: {SIM_POOL_PATH}")

    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(f"feature_cols.json not found at: {FEATURE_COLS_PATH}")

    log.info("Loading artifacts...")

    # Load feature columns
    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)

    # Load parquet and downsample if needed
    sim_df = pd.read_parquet(SIM_POOL_PATH)

    n_rows = len(sim_df)
    if n_rows > MAX_ROWS_IN_MEMORY:
        sim_df = sim_df.sample(
            n=MAX_ROWS_IN_MEMORY,
            random_state=42,
            replace=False,
        )
        log.info(
            "Downsampled sim_pool to %d rows for performance (from %d)",
            len(sim_df),
            n_rows,
        )
    else:
        log.info("Loaded sim_pool with %d rows", n_rows)

    # Shuffle once up-front
    sim_df = sim_df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    # Optional: identify probability columns just for logging
    proba_col = None
    if "rf_proba" in sim_df.columns:
        proba_col = "rf_proba"
    elif "lgbm_proba" in sim_df.columns:
        proba_col = "lgbm_proba"

    if proba_col:
        log.info("Found probability column '%s' in sim_pool", proba_col)
    else:
        log.info("No 'rf_proba' or 'lgbm_proba' column found – will still stream events normally.")

    log.info("Artifacts loaded successfully.")
    return sim_df, feature_cols, proba_col


def build_payload(row, feature_cols):
    """
    Build the JSON payload expected by the /score endpoint.

    service.py defines:

        class Tx(BaseModel):
            data: dict

    So the body must be:
        {"data": {feature_name: value, ...}}
    """
    features = row[feature_cols].to_dict()
    if "id" in features:
        features["id"] = 0.0

    payload = {"data": features}
    return payload


def send_event(session, payload):
    """Send a single event to the scoring service."""
    try:
        resp = session.post(SERVICE_URL, json=payload, timeout=3)
        resp.raise_for_status()
        out = resp.json()
        log.debug("Scored event: %s", out)
        return True
    except Exception as e:
        log.warning("Failed to call service %s: %s", SERVICE_URL, e)
        return False


# ---------------------------
# Main streaming loop
# ---------------------------

def main():
    sim_df, feature_cols, proba_col = load_artifacts()

    total_sent = 0
    total_failed = 0
    t_start = time.time()
    t_last_log = t_start

    log.info(
        "Starting realistic transaction stream at ~%.2f events/sec "
        "(~1 tx every %.1f seconds)",
        EVENTS_PER_SECOND,
        1.0 / EVENTS_PER_SECOND,
    )
    log.info("Posting to %s", SERVICE_URL)

    session = requests.Session()
    n = len(sim_df)
    idx = 0

    while True:
        row = sim_df.iloc[idx]
        idx = (idx + 1) % n  # loop through the pool

        payload = build_payload(row, feature_cols)
        ok = send_event(session, payload)

        total_sent += 1
        if not ok:
            total_failed += 1

        now = time.time()
        if now - t_last_log >= LOG_INTERVAL:
            elapsed = now - t_start
            eps = total_sent / elapsed if elapsed > 0 else 0.0

            msg = f"Stats | sent={total_sent}, failed={total_failed}, avg_events/sec={eps:.2f}"
            if proba_col is not None:
                msg += f", last_{proba_col}={float(row[proba_col]):.4f}"
            log.info(msg)

            t_last_log = now

        # throttle to target events/sec
        time.sleep(SLEEP_BETWEEN_EVENTS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Simulator stopped by user.")
