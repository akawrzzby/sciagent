#!/usr/bin/env bash
# Restart the SciAgent backend on the GPU host.
set -uo pipefail
PORT="${1:-9000}"
LOG="${LOG:-$HOME/sciagent/backend.log}"

echo "==> stopping any running instance"
pkill -f "uvicorn app.main" 2>/dev/null || true
sleep 2

echo "==> starting on port $PORT"
cd "$HOME/sciagent/backend"
setsid nohup bash "$HOME/sciagent/scripts/run_backend.sh" "$PORT" > "$LOG" 2>&1 < /dev/null &
sleep 12

if curl -s --max-time 15 "http://127.0.0.1:${PORT}/api/health" > /tmp/health.json 2>/dev/null; then
  echo "==> health:"
  python3 -c "import json;d=json.load(open('/tmp/health.json'));print(json.dumps(d,ensure_ascii=False,indent=1)[:900])" 2>/dev/null || cat /tmp/health.json
else
  echo "==> not healthy, log tail:"
  tail -30 "$LOG"
fi
