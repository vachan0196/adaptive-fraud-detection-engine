"""
Realistic transaction simulator.

Runs forever by design, but must be log-safe and not spam warnings if the API is down.
"""

import json
import logging
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests

ART_DIR = Path("artifacts")
SIM_POOL_PATH = ART_DIR / "sim_pool.parquet"
FEATURE_COLS_PATH = ART_DIR / "feature_cols.json"

EVENTS_PER_SECOND = 0.5
SLEEP_BETWEEN_EVENTS = 1.0 / EVENTS_PER_SECOND

SERVICE_URL = os.getenv("SERVICE_URL", "http://127.0.0.1:8000/score")

MAX_ROWS_IN_MEMORY = 50_000
LOG_INTERVAL = 30

logging.basicConfig(level=logging.INFO, format="[sim] %(message)s")
log = logging.getLogger("simulator")

# --- realistic context fields for Live Feed ---
CHANNELS = ["POS", "ECOM", "ATM", "MOTO", "TRANSFER"]
COUNTRIES = ["GB", "US", "IN", "FR", "DE", "AE"]


def load_artifacts():
    if not SIM_POOL_PATH.exists():
        raise FileNotFoundError(f"sim_pool parquet not found at: {SIM_POOL_PATH}")
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(f"feature_cols.json not found at: {FEATURE_COLS_PATH}")

    log.info("Loading artifacts...")
    feature_cols = json.loads(FEATURE_COLS_PATH.read_text())

    sim_df = pd.read_parquet(SIM_POOL_PATH)
    n_rows = len(sim_df)
    if n_rows > MAX_ROWS_IN_MEMORY:
        sim_df = sim_df.sample(n=MAX_ROWS_IN_MEMORY, random_state=42, replace=False)
        log.info("Downsampled sim_pool to %d rows (from %d)", len(sim_df), n_rows)
    else:
        log.info("Loaded sim_pool with %d rows", n_rows)

    sim_df = sim_df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    proba_col = "rf_proba" if "rf_proba" in sim_df.columns else ("lgbm_proba" if "lgbm_proba" in sim_df.columns else None)
    if proba_col:
        log.info("Found probability column '%s' in sim_pool", proba_col)

    return sim_df, feature_cols, proba_col


def _gen_amount_gbp() -> float:
    """
    Generate a realistic-ish transaction amount in GBP for demo:
    log-normal-ish distribution with hard caps.
    """
    # random.lognormvariate(mu, sigma) ~ exp(N(mu, sigma^2))
    # mu=3.5 => median ~ exp(3.5)=33.1
    # sigma=1.2 gives long tail
    amt = float(random.lognormvariate(3.5, 1.2))
    amt = max(1.0, min(amt, 5000.0))
    return round(amt, 2)


def build_payload(row, feature_cols):
    feats = row[feature_cols].to_dict()
    if "id" in feats:
        feats["id"] = 0.0

    amount = _gen_amount_gbp()

    channel = random.choices(
        population=CHANNELS,
        weights=[0.55, 0.25, 0.10, 0.03, 0.07],
        k=1,
    )[0]

    country = random.choices(
        population=COUNTRIES,
        weights=[0.70, 0.10, 0.08, 0.05, 0.04, 0.03],
        k=1,
    )[0]

    customer_id = f"CUST-{random.randint(100000, 999999)}"

    # Keep features under "data" because service.py expects Tx.data
    return {
        "data": feats,
        "amount": amount,
        "channel": channel,
        "country": country,
        "customer_id": customer_id,
    }


def main():
    sim_df, feature_cols, proba_col = load_artifacts()

    session = requests.Session()
    n = len(sim_df)
    idx = 0

    total_sent = 0
    total_failed = 0
    t_start = time.time()
    t_last_log = t_start

    # Backoff control (prevents spam if API is down)
    backoff = 1.0
    max_backoff = 60.0
    last_fail_log = 0.0

    log.info("Starting stream at ~%.2f events/sec, posting to %s", EVENTS_PER_SECOND, SERVICE_URL)

    while True:
        row = sim_df.iloc[idx]
        idx = (idx + 1) % n

        payload = build_payload(row, feature_cols)

        ok = False
        try:
            resp = session.post(SERVICE_URL, json=payload, timeout=3)
            resp.raise_for_status()
            ok = True
            backoff = 1.0  # reset on success
        except Exception as e:
            total_failed += 1
            now = time.time()
            if now - last_fail_log >= 15:
                log.warning("API not ready (%s). Backing off %.1fs", str(e).splitlines()[0], backoff)
                last_fail_log = now
            time.sleep(backoff)
            backoff = min(max_backoff, backoff * 2)

        total_sent += 1

        now = time.time()
        if now - t_last_log >= LOG_INTERVAL:
            elapsed = now - t_start
            eps = total_sent / elapsed if elapsed > 0 else 0.0
            msg = f"Stats | sent={total_sent}, failed={total_failed}, avg_events/sec={eps:.2f}"
            if proba_col is not None:
                msg += f", last_{proba_col}={float(row[proba_col]):.4f}"
            log.info(msg)
            t_last_log = now

        if ok:
            time.sleep(SLEEP_BETWEEN_EVENTS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Simulator stopped by user.")
