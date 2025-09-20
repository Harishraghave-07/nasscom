#!/bin/bash
# Start script for API Gateway
set -euo pipefail

# Allow optional environment variables to override host/port
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-2}

# Run the FastAPI server module
exec uvicorn src.api.server:app --host "$HOST" --port "$PORT" --workers "$WORKERS" --proxy-headers
