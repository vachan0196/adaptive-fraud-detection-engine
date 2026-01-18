#!/usr/bin/env bash
set -euo pipefail

ART_DIR="/app/artifacts"
SEED_DIR="/seed/artifacts"

echo "[boot] Starting container..."
mkdir -p "${ART_DIR}"

# Seed only the ML artifacts needed to run scoring/UI.
# Never seed app.db and never create CSVs.
required_files=(
  "feature_cols.json"
  "metadata.json"
  "threshold.json"
  "rf_model.pkl"
)

for f in "${required_files[@]}"; do
  if [ ! -f "${ART_DIR}/${f}" ] && [ -f "${SEED_DIR}/${f}" ]; then
    echo "[boot] Seeding missing ${f} from image seed..."
    cp -n "${SEED_DIR}/${f}" "${ART_DIR}/${f}"
  fi
done

optional_files=(
  "model_meta.json"
  "demo_samples.parquet"
  "sim_pool.parquet"
  "lgbm_model.pkl"
)

for f in "${optional_files[@]}"; do
  if [ ! -f "${ART_DIR}/${f}" ] && [ -f "${SEED_DIR}/${f}" ]; then
    echo "[boot] Seeding optional ${f} from image seed..."
    cp -n "${SEED_DIR}/${f}" "${ART_DIR}/${f}"
  fi
done

echo "[boot] Starting FastAPI (uvicorn) on :8000..."
uvicorn service:app --host 0.0.0.0 --port 8000 --workers 1 &
API_PID=$!

echo "[boot] Waiting for FastAPI /health..."
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    echo "[boot] FastAPI is up."
    break
  fi
  sleep 1
  if [ "$i" -eq 60 ]; then
    echo "[boot] ERROR: FastAPI did not become healthy within 60s."
    kill "${API_PID}" >/dev/null 2>&1 || true
    exit 1
  fi
done

# --- Start simulator automatically ---
echo "[boot] Starting Simulator (simulator_pro.py)..."
python -u /app/simulator_pro.py &
SIM_PID=$!

echo "[boot] Starting Streamlit on :8501..."
streamlit run /app/app.py --server.port=8501 --server.address=0.0.0.0 &
ST_PID=$!

term_handler() {
  echo "[boot] Caught termination signal. Stopping..."
  kill "${ST_PID}" >/dev/null 2>&1 || true
  kill "${SIM_PID}" >/dev/null 2>&1 || true
  kill "${API_PID}" >/dev/null 2>&1 || true
  wait "${ST_PID}" >/dev/null 2>&1 || true
  wait "${SIM_PID}" >/dev/null 2>&1 || true
  wait "${API_PID}" >/dev/null 2>&1 || true
}

trap term_handler SIGTERM SIGINT

echo "[boot] Container ready. PIDs: uvicorn=${API_PID}, sim=${SIM_PID}, streamlit=${ST_PID}"
wait -n "${API_PID}" "${SIM_PID}" "${ST_PID}"
echo "[boot] One process exited; shutting down the others..."
term_handler
exit 1
