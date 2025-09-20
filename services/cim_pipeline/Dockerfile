# Multi-stage Dockerfile for Clinical Image Masking Pipeline
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements-core.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements-core.txt

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd --create-home --shell /bin/bash appuser

COPY requirements-core.txt requirements-core.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-core.txt

COPY src/ /app/src/
COPY services/cim_pipeline/start.sh /app/start.sh
RUN chmod +x /app/start.sh && chown -R appuser:appuser /app
USER appuser
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5000/health || exit 1
CMD ["/app/start.sh"]
