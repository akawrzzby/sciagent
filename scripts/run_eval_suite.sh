#!/usr/bin/env bash
# Run the remaining RAG evaluation suite on the GPU host.
#
#   bash ~/sciagent/scripts/run_eval_suite.sh
#
# 1. the negative control: three questions the library cannot answer
# 2. two more full runs of the competition question set, so the numbers in
#    the technical document can be reported as a mean over three runs
#    instead of a single lucky run
set -uo pipefail

cd "$HOME/sciagent"
PY="$HOME/envs/sciagent/bin/python"
DATA="$HOME/sciagent/data"

echo "==> [1/3] negative control (questions outside the library)"
"$PY" scripts/eval_rag_control.py > "$DATA/eval_control.log" 2>&1
echo "    control exit=$?"

for run in 5 6; do
    echo "==> [2/3] eval run v$run"
    "$PY" scripts/eval_rag.py --base http://127.0.0.1:9000 \
        --out "$DATA/eval_rag_v$run.json" > "$DATA/eval_rag_v$run.log" 2>&1
    echo "    v$run exit=$?"
done

echo "==> all done"
for f in "$DATA"/eval_rag_v*.json; do
    echo "--- $f"
    "$PY" - "$f" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
s = data.get("summary", {})
print("   ", json.dumps(s, ensure_ascii=False))
PYEOF
done
