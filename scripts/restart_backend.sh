#!/usr/bin/env bash
# Restart the SciAgent backend on the GPU host.
#
# A plain ``pkill`` is not enough.  Uvicorn shuts down gracefully and waits for
# the in-flight requests; a long LLM call (or an open SSE stream) can keep the
# old process alive for minutes.  Two servers then end up sharing the port
# through SO_REUSEPORT, and the traffic is split between a stale build and the
# new one -- which is exactly how a "the fix is deployed but the old behaviour
# is still there" bug happens.
set -uo pipefail
PORT="${1:-9000}"
LOG="${LOG:-$HOME/sciagent/backend.log}"

echo "==> stopping any running instance"
pkill -f "uvicorn app.main" 2>/dev/null || true
for _ in $(seq 1 10); do
  sleep 1
  pgrep -f "uvicorn app.main" >/dev/null 2>&1 || break
done
if pgrep -f "uvicorn app.main" >/dev/null 2>&1; then
  echo "    the old process is still draining requests - forcing it"
  pkill -9 -f "uvicorn app.main" 2>/dev/null || true
  sleep 2
fi

echo "==> starting on port $PORT"
cd "$HOME/sciagent/backend"
setsid nohup bash "$HOME/sciagent/scripts/run_backend.sh" "$PORT" > "$LOG" 2>&1 < /dev/null &
sleep 12

echo "==> instances"
pgrep -af "uvicorn app.main" || echo "    (none)"

if curl -s --max-time 15 "http://127.0.0.1:${PORT}/api/health" > /tmp/health.json 2>/dev/null; then
  echo "==> health:"
  cat /tmp/health.json
  echo ""
else
  echo "==> not healthy, log tail:"
  tail -30 "$LOG"
fi
