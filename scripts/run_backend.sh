#!/usr/bin/env bash
# Start the SciAgent backend (development mode) on the GPU host.
#
#   bash scripts/run_backend.sh [port]
#
# The LLM endpoint is configured through LLM_BASE_URL (see .env.example).
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/sciagent}"
BACKEND_DIR="${BACKEND_DIR:-$HOME/sciagent/backend}"
PORT="${1:-9000}"
HOST="${HOST:-0.0.0.0}"

cd "$BACKEND_DIR"
export PYTHONPATH="$BACKEND_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MPLBACKEND=Agg

exec "$ENV_DIR/bin/python" -m uvicorn app.main:app \
    --host "$HOST" --port "$PORT" \
    --workers 1 --timeout-keep-alive 120 \
    --log-level info
