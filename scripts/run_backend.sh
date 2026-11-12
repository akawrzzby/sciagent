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

# The retrieval models are pre-downloaded to ~/models by
# scripts/download_models.py.  Prefer the local directory: the HF mirror is
# unreachable from many networks and a failed download would silently degrade
# the dense channel.
MODEL_ROOT="${MODEL_ROOT:-$HOME/models}"
if [ -d "$MODEL_ROOT/bge-m3" ]; then
  export EMBEDDING_MODEL="${EMBEDDING_MODEL:-$MODEL_ROOT/bge-m3}"
fi
if [ -d "$MODEL_ROOT/bge-reranker-v2-m3" ]; then
  export RERANKER_MODEL="${RERANKER_MODEL:-$MODEL_ROOT/bge-reranker-v2-m3}"
fi
# 192 threads on a 192-core host costs a 13x throughput loss; 64 is the sweet
# spot for bge-m3 (see docs/technical-report.md, section 4.8).
export EMBEDDING_THREADS="${EMBEDDING_THREADS:-64}"

# Serve the built SPA from the same process when the build output is present.
# This keeps the public demo behind a single URL (no separate reverse proxy).
STATIC_CANDIDATE="${STATIC_CANDIDATE:-$HOME/sciagent/frontend/dist}"
if [ -f "$STATIC_CANDIDATE/index.html" ]; then
  export STATIC_DIR="${STATIC_DIR:-$STATIC_CANDIDATE}"
  echo "==> serving the frontend from $STATIC_DIR"
fi

exec "$ENV_DIR/bin/python" -m uvicorn app.main:app \
    --host "$HOST" --port "$PORT" \
    --workers 1 --timeout-keep-alive 120 \
    --log-level info
