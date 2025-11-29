"""
Realistic transaction simulator for the fraud demo.

- Uses a pre-scored pool (sim_pool.parquet) to get realistic probability distributions.
- Time-of-day patterns: more risky traffic at night / evenings.
- Card-level bursts: occasional "attack" periods for a single card_id.
- Sends payloads to the FastAPI scoring service at /score.

Run (in a separate terminal, after starting the service):

    uvicorn service:app --reload --port 8000
    python simulator_pro.py
"""

import time
import random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import json

# --------------------------------------------------------------------
# Paths and config
# --------------------------------------------------------------------
ART_DIR = Path("artifacts")
FEATS_PATH = ART_DIR / "feature_cols.json"
SIM_POOL_PATH = ART_DIR / "sim_pool.parquet"   # created by train_and_export.py

SCORE_URL = "http://127.0.0.1:8000/score"

# Average transactions per second (Poisson around this)
TX_PER_SEC_BASE = 12  # tweak if you want more / less volume

# Number of synthetic cards in the population
N_CARDS = 500

# Duration of a card "attack" burst, in seconds
ATTACK_MIN_SEC = 20
ATTACK_MAX_SEC = 60

# --------------------------------------------------------------------
# Load feature cols + sim_pool
# --------------------------------------------------------------------
print("[sim] Loading artifacts...")

if not FEATS_PATH.exists():
    raise FileNotFoundError("feature_cols.json not found. Run train_and_export.py first.")

if not SIM_POOL_PATH.exists():
    raise FileNotFoundError(
        f"sim_pool.parquet not found at {SIM_POOL_PATH}. "
        "Run train_and_export.py to create it."
    )

# Correct way: feature_cols.json is a simple JSON list
feature_cols = json.loads(FEATS_PATH.read_text())

sim_pool_raw = pd.read_parquet(SIM_POOL_PATH)

if "rf_proba" not in sim_pool_raw.columns:
    raise ValueError(
        "sim_pool.parquet does not contain 'rf_proba'. "
        "Make sure you are using the updated train_and_export.py."
    )

# Only keep model feature columns + rf_proba (+Class for reference)
sim_pool = sim_pool_raw.copy()
missing = [c for c in feature_cols if c not in sim_pool.columns]
if missing:
    raise ValueError(f"sim_pool.parquet is missing feature columns: {missing}")

sim_pool = sim_pool[feature_cols + ["rf_proba", "Class"]]  # Class not required but nice to keep

# Optional downsample if sim_pool is huge
MAX_SAMPLES = 50_000
if len(sim_pool) > MAX_SAMPLES:
    sim_pool = sim_pool.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)
    print(f"[sim] Downsampled sim_pool to {MAX_SAMPLES} rows for performance.")

print("[sim] Building probability buckets from rf_proba...")
probas = sim_pool["rf_proba"].values

# Buckets by risk level using rf_proba
bucket_low = sim_pool[probas < 0.01]                       # almost certainly legit
bucket_med = sim_pool[(probas >= 0.01) & (probas < 0.10)]
bucket_high = sim_pool[(probas >= 0.10) & (probas < 0.50)]
bucket_vhigh = sim_pool[probas >= 0.50]

def _safe_bucket(df_fallback: pd.DataFrame, df_bucket: pd.DataFrame) -> pd.Series:
    """Return a random row from bucket, or fallback if bucket is empty."""
    if len(df_bucket) == 0:
        return df_fallback.sample(1).iloc[0]
    return df_bucket.sample(1).iloc[0]

bucket_all = sim_pool

print(
    f"[sim] Bucket sizes -> low={len(bucket_low)}, "
    f"med={len(bucket_med)}, high={len(bucket_high)}, very_high={len(bucket_vhigh)}"
)

# --------------------------------------------------------------------
# Card population + attack state
# --------------------------------------------------------------------
CARDS = [f"C{str(i).zfill(6)}" for i in range(N_CARDS)]
ATTACK_CARD = None
ATTACK_END: datetime | None = None

def pick_card_and_intensity(now: datetime) -> tuple[str, str]:
    """
    Returns (card_id, intensity) where intensity ∈ {"normal", "attack"}.
    Occasionally starts an attack burst on a single card.
    """
    global ATTACK_CARD, ATTACK_END

    if ATTACK_CARD is not None and ATTACK_END is not None and now < ATTACK_END:
        return ATTACK_CARD, "attack"

    if ATTACK_CARD is not None and ATTACK_END is not None and now >= ATTACK_END:
        ATTACK_CARD = None
        ATTACK_END = None

    # Small chance to start a new attack
    if ATTACK_CARD is None and random.random() < 0.003:
        ATTACK_CARD = random.choice(CARDS)
        dur = random.randint(ATTACK_MIN_SEC, ATTACK_MAX_SEC)
        ATTACK_END = now + timedelta(seconds=dur)
        print(f"[sim] Starting attack on card {ATTACK_CARD} for ~{dur}s")
        return ATTACK_CARD, "attack"

    return random.choice(CARDS), "normal"

# --------------------------------------------------------------------
# Sampling logic (time-of-day + intensity)
# --------------------------------------------------------------------
NIGHT_HOURS = set(range(0, 6))      # 00:00–05:59
BUSINESS_HOURS = set(range(8, 18))  # 08:00–17:59
EVENING_HOURS = set(range(18, 24))  # 18:00–23:59

def pick_sample(now: datetime, intensity: str) -> pd.Series:
    """
    Choose one row from sim_pool according to:
    - Time of day
    - Whether the card is under attack
    This shapes the resulting score distribution while still using the pre-scored rf_proba.
    """
    hour = now.hour
    r = random.random()

    if intensity == "attack":
        # During an attack, traffic from that card is mostly high-risk
        weights = (0.05, 0.15, 0.50, 0.30)
    else:
        if hour in NIGHT_HOURS:
            weights = (0.50, 0.25, 0.20, 0.05)   # more fraud at night
        elif hour in BUSINESS_HOURS:
            weights = (0.75, 0.15, 0.08, 0.02)   # mostly clean traffic
        elif hour in EVENING_HOURS:
            weights = (0.60, 0.20, 0.15, 0.05)   # evenings medium risk
        else:
            weights = (0.70, 0.20, 0.08, 0.02)   # fallback

    w_low, w_med, w_high, w_vhigh = weights

    if r < w_low:
        return _safe_bucket(bucket_all, bucket_low)
    elif r < w_low + w_med:
        return _safe_bucket(bucket_all, bucket_med)
    elif r < w_low + w_med + w_high:
        return _safe_bucket(bucket_all, bucket_high)
    else:
        return _safe_bucket(bucket_all, bucket_vhigh)

# --------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------
def main():
    print("[sim] Starting realistic transaction stream...")
    print(f"[sim] Posting to {SCORE_URL}")
    print(f"[sim] Avg. transactions/sec ≈ {TX_PER_SEC_BASE}")

    while True:
        now = datetime.utcnow()

        # Poisson number of transactions in this 1-second batch
        batch_size = np.random.poisson(TX_PER_SEC_BASE)
        batch_size = max(1, int(batch_size))

        for _ in range(batch_size):
            now = datetime.utcnow()
            card_id, intensity = pick_card_and_intensity(now)
            row = pick_sample(now, intensity)

            payload = {col: float(row[col]) for col in feature_cols}
            # Extra fields not used by the model but kept in the payload for realism
            payload["card_id"] = card_id
            payload["channel"] = random.choice(["POS", "E_COM", "ATM"])
            payload["country"] = random.choice(["UK", "US", "IN", "FR", "DE"])

            try:
                resp = requests.post(
                    SCORE_URL,
                    json={"data": payload},
                    timeout=3,
                )
                if resp.status_code != 200:
                    print("[sim] Error from service:", resp.status_code, resp.text[:200])
            except Exception as e:
                print("[sim] Failed to call service:", e)
                time.sleep(2)
                break

        time.sleep(1.0)


if __name__ == "__main__":
    main()
