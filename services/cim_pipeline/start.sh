#!/bin/bash
set -euo pipefail
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-5000}

# Example: Run pipeline as a long-lived service exposing a minimal health endpoint
# If the pipeline is not designed as a server, this will run periodically or listen for a queue.

# Start a tiny health server in background
python - <<'PY'
from fastapi import FastAPI
import uvicorn
app = FastAPI()

@app.get('/health')
def health():
    return {'status':'ok'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int('''$PORT'''))
PY
