#!/usr/bin/env bash
# A/B measurement of the metadata aware retrieval.
#
# The three switches (METADATA_AWARE_EMBEDDING, METADATA_AWARE_RERANK,
# METADATA_BOOST) exist so that the effect of the metadata prefix can be
# measured instead of asserted.  The embedding switch only takes effect when
# the vectors are rebuilt, so the "before" case has to reindex as well.
#
#   bash scripts/eval_retrieval_ab.sh
#
# The result is written to data/eval_retrieval_before.json and
# data/eval_retrieval_after.json; the backend is left in the "after" state.
set -uo pipefail

HOME_DIR="${HOME_DIR:-$HOME}"
ROOT="$HOME_DIR/sciagent"
PY="$HOME_DIR/envs/sciagent/bin/python"
PORT="${PORT:-9000}"

run_case () {
    local label="$1" embed="$2" rerank="$3" boost="$4"
    echo ""
    echo "=============================================================="
    echo "case $label : embedding=$embed rerank=$rerank boost=$boost"
    echo "=============================================================="

    pkill -f "uvicorn app.main" 2>/dev/null || true
    sleep 3

    cd "$ROOT/backend"
    setsid nohup env \
        METADATA_AWARE_EMBEDDING="$embed" \
        METADATA_AWARE_RERANK="$rerank" \
        METADATA_BOOST="$boost" \
        bash "$ROOT/scripts/run_backend.sh" "$PORT" \
        > "$ROOT/backend.log" 2>&1 < /dev/null &
    sleep 15

    echo "==> rebuilding the vectors"
    curl -s --max-time 2400 -X POST "http://127.0.0.1:${PORT}/api/admin/reindex"
    echo ""

    cd "$ROOT"
    "$PY" scripts/eval_retrieval.py \
        --base "http://127.0.0.1:${PORT}" \
        --out "$ROOT/data/eval_retrieval_${label}.json" \
        --label "$label"
}

run_case before false false 0.0
run_case after true true 0.30

echo ""
echo "==> done"
