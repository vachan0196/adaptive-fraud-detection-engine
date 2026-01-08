#!/usr/bin/env bash
set -e

mkdir -p artifacts

# Ensure DB file exists (SQLite will create tables via service startup)
touch artifacts/app.db

# Start FastAPI (writer)
uvicorn service:app --host 0.0.0.0 --port 8000 &

# Start Streamlit (reader)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
