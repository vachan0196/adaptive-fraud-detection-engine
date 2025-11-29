# train_and_export.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import lightgbm as lgb  # NEW

# ==== CONFIG ====

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data_raw" / "creditcard_2023.csv"   # ✅ instead of the D:\... path
TARGET = "Class"
ART_DIR = PROJECT_ROOT / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# business cost ratio: FN much worse than FP
C_FN, C_FP = 100, 1


def cost_at_threshold(y_true, scores, t, C_FN=100, C_FP=1):
    yhat = (scores >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
    return fn * C_FN + fp * C_FP, {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def sweep_best_threshold(y_true, scores, thresholds, C_FN=100, C_FP=1):
    best_t, best_cost, best_cm = None, 1e18, None
    for t in thresholds:
        c, cm = cost_at_threshold(y_true, scores, t, C_FN, C_FP)
        if c < best_cost:
            best_cost, best_t, best_cm = c, float(t), cm
    return best_t, best_cost, best_cm


# ==== LOAD ====
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==== TRAIN RF ====
rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

# probabilities on test set (used for threshold + sim pool)
proba_rf_te = rf.predict_proba(X_te)[:, 1]
auc_rf = roc_auc_score(y_te, proba_rf_te)
print(f"RF ROC AUC: {auc_rf:.6f}")

# ==== THRESHOLD SWEEP (focus 0..0.2) for RF ====
thresholds = np.linspace(0.0, 0.2, 2001)
best_t_rf, best_cost_rf, best_cm_rf = sweep_best_threshold(
    y_te, proba_rf_te, thresholds, C_FN, C_FP
)
print(
    f"[RF] Best threshold = {best_t_rf:.6f} | "
    f"Expected cost = {best_cost_rf} | CM = {best_cm_rf}"
)

# ==== TRAIN LightGBM ====
lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
lgbm.fit(X_tr, y_tr)

proba_lgbm_te = lgbm.predict_proba(X_te)[:, 1]
auc_lgbm = roc_auc_score(y_te, proba_lgbm_te)
print(f"LightGBM ROC AUC: {auc_lgbm:.6f}")

best_t_lgbm, best_cost_lgbm, best_cm_lgbm = sweep_best_threshold(
    y_te, proba_lgbm_te, thresholds, C_FN, C_FP
)
print(
    f"[LightGBM] Best threshold = {best_t_lgbm:.6f} | "
    f"Expected cost = {best_cost_lgbm} | CM = {best_cm_lgbm}"
)

# ==== DECIDE ACTIVE MODEL (by expected cost) ====
if best_cost_lgbm < best_cost_rf:
    active_model = "lightgbm"
    active_threshold = best_t_lgbm
else:
    active_model = "random_forest"
    active_threshold = best_t_rf

print(
    f"Active model selected: {active_model} "
    f"| threshold = {active_threshold:.6f}"
)

# ==== SAVE ARTIFACTS ====
# models
joblib.dump(rf, ART_DIR / "rf_model.pkl")
joblib.dump(lgbm, ART_DIR / "lgbm_model.pkl")

# feature columns
with open(ART_DIR / "feature_cols.json", "w") as f:
    json.dump(list(X.columns), f, indent=2)

# basic stats for UI bounds
desc = X_tr.describe().T
meta = {}
for col in X.columns:
    m = float(desc.loc[col, "mean"]) if col in desc.index else 0.0
    sd = float(desc.loc[col, "std"]) if col in desc.index else 1.0
    lo = float(desc.loc[col, "min"]) if col in desc.index else -20.0
    hi = float(desc.loc[col, "max"]) if col in desc.index else 20.0
    if col.lower() == "amount":
        lo, hi = max(0.0, lo), max(hi, 5000.0)
    meta[col] = {"mean": m, "std": sd, "min": lo, "max": hi}

with open(ART_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# threshold + cost ratio config (for *active* model)
with open(ART_DIR / "threshold.json", "w") as f:
    json.dump(
        {"threshold": float(active_threshold), "C_FN": int(C_FN), "C_FP": int(C_FP)},
        f,
        indent=2,
    )

# ---------------------------------------------------------
# model registry / meta (both models) — JSON SAFE
# ---------------------------------------------------------
rf_summary = {
    "path": "rf_model.pkl",
    "best_threshold": float(best_t_rf),
    "expected_cost": int(best_cost_rf),
    "auc": float(auc_rf),
    "confusion_matrix": {k: int(v) for k, v in best_cm_rf.items()},
}

lgbm_summary = {
    "path": "lgbm_model.pkl",
    "best_threshold": float(best_t_lgbm),
    "expected_cost": int(best_cost_lgbm),
    "auc": float(auc_lgbm),
    "confusion_matrix": {k: int(v) for k, v in best_cm_lgbm.items()},
}

model_meta = {
    "active_model": str(active_model),
    "models": {
        "random_forest": rf_summary,
        "lightgbm": lgbm_summary,
    },
}

with open(ART_DIR / "model_meta.json", "w") as f:
    json.dump(model_meta, f, indent=2)

# ---------------------------------------------------------
# demo sample set (for Random button & simple simulator)
# ---------------------------------------------------------
demo = X_te.copy()
demo["Class"] = y_te.values
demo.sample(n=min(1000, len(demo)), random_state=42).to_parquet(
    ART_DIR / "demo_samples.parquet", index=False
)

# ---------------------------------------------------------
# sim_pool: pre-scored pool for realistic simulator
# ---------------------------------------------------------
sim_pool = X_te.copy()
sim_pool["rf_proba"] = proba_rf_te
sim_pool["lgbm_proba"] = proba_lgbm_te
sim_pool["Class"] = y_te.values
sim_pool.to_parquet(ART_DIR / "sim_pool.parquet", index=False)
print("Saved sim_pool.parquet for realistic simulator.")

# ---------------------------------------------------------
# init live log with event_id column (aligns with service.py)
# ---------------------------------------------------------
live_log = ART_DIR / "live_events.csv"
if not live_log.exists():
    pd.DataFrame(
        columns=["event_id", "ts", "decision", "proba", "payload"]
    ).to_csv(live_log, index=False)
    print("Initialised live_events.csv with event_id column.")

print("Artifacts saved in ./artifacts")
