FROM python:3.11-slim

WORKDIR /app

# System deps (LightGBM + general builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py service.py simulator_pro.py train_and_export.py ./

# Copy artifacts (must exist in repo for AWS runtime)
COPY artifacts ./artifacts

# (Optional) if you keep dataset on EC2 and want training in-container later
# COPY data_raw ./data_raw

EXPOSE 8000
EXPOSE 8501

CMD ["bash", "-c", "\
    uvicorn service:app --host 0.0.0.0 --port 8000 & \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0 & \
    python simulator_pro.py \
"]
