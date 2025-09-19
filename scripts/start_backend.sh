#!/usr/bin/env bash
set -euo pipefail

# Start the FastAPI backend (uvicorn) for masking pipeline
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "Virtualenv python not found at $VENV_PY"
  echo "Activate your venv or modify this script to point to the correct python." >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR"

echo "Starting uvicorn on http://127.0.0.1:8000"
"$VENV_PY" -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000

# Example curl (run in another terminal):
# curl -F "file=@/path/to/Highlighted PHI Details Emily_Dawson.pdf" http://127.0.0.1:8000/process
