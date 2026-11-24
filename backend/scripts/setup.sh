#!/usr/bin/env bash
# Create the backend virtualenv and install everything.
#   ./scripts/setup.sh          core + neural models (torch, demucs, crepe)
#   ./scripts/setup.sh --core   core only (DSP separation fallback, pYIN)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
$PY -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install --no-deps "basic-pitch==0.4.0"
if [[ "${1:-}" != "--core" ]]; then
  if [[ "$(uname)" == "Darwin" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    pip install -r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
  else
    pip install -r requirements-ml.txt
  fi
fi
python scripts/download_models.py
echo "Done. Start the API with:  . .venv/bin/activate && uvicorn app.main:app --port 8000"
