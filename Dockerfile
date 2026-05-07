FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PYTHONPATH=/app     TRADAR_CONFIG=/app/config/tradar.yaml     TRADAR_PROFILE=paper     TRADAR_DATA_DIR=/data     TRADAR_DB_PATH=/data/tradarbot.db

WORKDIR /app

RUN apt-get update     && apt-get install -y --no-install-recommends         build-essential         sqlite3         curl     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip     && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=10s --retries=3     CMD python deploy/healthcheck.py --config config/tradar.yaml || exit 1

CMD ["python", "run.py", "--config", "config/tradar.yaml", "--profile", "paper"]
