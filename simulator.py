# simulator.py
import time, random
from pathlib import Path
import requests
import pandas as pd

SAMPLES = Path("artifacts/demo_samples.parquet")
URL = "http://127.0.0.1:8000/score"

samples = pd.read_parquet(SAMPLES)
features = [c for c in samples.columns if c != "Class"]

print(f"Loaded {len(samples)} demo rows. Streaming to {URL} ... Ctrl+C to stop.")
while True:
    row = samples.sample(1).iloc[0][features].to_dict()
    row["id"] = 0.0  # avoid leakage in demo
    try:
        r = requests.post(URL, json={"data": row}, timeout=5)
        print(r.json())
    except Exception as e:
        print("Request failed:", e)
    time.sleep(random.uniform(0.8, 3.0))
