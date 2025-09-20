# Multi-stage Dockerfile for PHI Detection Service
# Stage 1: build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements specific to PHI service if present,
# otherwise fallback to core requirements
COPY requirements-presidio.txt requirements-presidio.txt
COPY requirements-core.txt requirements-core.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements-presidio.txt || pip install --no-cache-dir -r requirements-core.txt

# Stage 2: application image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder - using pip install again into final image for simplicity
COPY requirements-presidio.txt requirements-presidio.txt
COPY requirements-core.txt requirements-core.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-presidio.txt || pip install --no-cache-dir -r requirements-core.txt

# Copy service code
COPY src/ /app/src/
COPY services/phi_service/app.py /app/app.py
COPY config/ /app/config/

# Ensure permissions
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Healthcheck - simple HTTP GET to /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app", "--workers", "2"]
