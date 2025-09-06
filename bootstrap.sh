#!/usr/bin/env bash
set -euo pipefail

# Create venv and install dependencies
python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi
if [ -f dev-requirements.txt ]; then
  pip install -r dev-requirements.txt
fi
# Install pre-commit and set up hooks
pip install pre-commit
pre-commit install

echo "Bootstrap complete. Activate the venv with: source .venv/bin/activate"
