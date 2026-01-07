#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/app"
ART_DIR="/app/artifacts"
SEED_DIR="/seed/artifacts"

echo "[boot] Starting container..."
mkdir -p "${ART_DIR}"

# If running with: -v $(pwd)/artifacts:/app/artifacts
# the mount might be empty and would hide baked artifacts. Seed required files safely.
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

# Optional seeds (do not fail if missing)
optional_files=(
  "model_meta.json"
  "demo_samples.parquet"
  "sim_pool.parquet"
)

for f in "${optional_files[@]}"; do
  if [ ! -f "${ART_DIR}/${f}" ] && [ -f "${SEED_DIR}/${f}" ]; then
    echo "[boot] Seeding optional ${f} from image seed..."
    cp -n "${SEED_DIR}/${f}" "${ART_DIR}/${f}"
  fi
done

# Create CSVs that Streamlit expects (best-effort; Streamlit also self-heals)
if [ ! -f "${ART_DIR}/cases.csv" ]; then
  echo "event_id,ts,proba,status,customer_response,resolution_source,updated_at" > "${ART_DIR}/cases.csv"
fi

# Ensure SQLite file path exists (tables created by init_db in service/app)
touch "${ART_DIR}/app.db" || true

# Start FastAPI (background)
echo "[boot] Starting FastAPI (uvicorn) on :8000..."
uvicorn service:app --host 0.0.0.0 --port 8000 --workers 1 &
API_PID=$!

# Wait for /health to respond (so Streamlit + manual simulator don’t race)
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

# Start Streamlit (background)
echo "[boot] Starting Streamlit on :8501..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
ST_PID=$!

# Stop everything cleanly on SIGTERM/SIGINT (docker stop, EC2 reboot, etc.)
term_handler() {
  echo "[boot] Caught termination signal. Stopping..."
  kill "${ST_PID}" >/dev/null 2>&1 || true
  kill "${API_PID}" >/dev/null 2>&1 || true
  wait "${ST_PID}" >/dev/null 2>&1 || true
  wait "${API_PID}" >/dev/null 2>&1 || true
}
trap term_handler SIGTERM SIGINT

# If either process exits, kill the other and fail (restart policy can bring it back)
echo "[boot] Container ready. PIDs: uvicorn=${API_PID}, streamlit=${ST_PID}"
wait -n "${API_PID}" "${ST_PID}"
echo "[boot] One process exited; shutting down the other..."
term_handler
exit 1
