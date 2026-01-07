FROM python:3.11-slim

WORKDIR /app

# System deps: build tools, LightGBM runtime, plus curl+sqlite3 for health/debug
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    bash \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Seed artifacts baked into image (used if EC2 volume mount is empty)
# We keep them separate from /app/artifacts so a bind-mount won't hide them.
RUN mkdir -p /seed/artifacts && \
    if [ -d "/app/artifacts" ]; then cp -a /app/artifacts/. /seed/artifacts/; fi

# Ensure scripts are executable
RUN chmod +x /app/start.sh || true

EXPOSE 8000
EXPOSE 8501

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

ENTRYPOINT ["/app/start.sh"]
