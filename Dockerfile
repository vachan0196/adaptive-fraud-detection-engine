FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (code + artifacts, if present)
COPY . .

# (Optional but recommended) – build artifacts inside the image
# Comment this out if you prefer to bake artifacts locally instead.

# Expose ports:
EXPOSE 8000
EXPOSE 8501

# Start:
# 1) FastAPI scoring service (uvicorn)
# 2) Streamlit dashboard
# 3) Simulator to generate live events
CMD ["bash", "-c", "\
    uvicorn service:app --host 0.0.0.0 --port 8000 & \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0 & \
    python simulator_pro.py \
"]
