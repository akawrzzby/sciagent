#!/usr/bin/env bash
# Start a Cloudflare quick tunnel in front of the local SciAgent instance.
#
#   bash scripts/start_tunnel.sh [local-port] [log-file]
#
# The script writes the public URL to "$HOME/sciagent/tunnel.url" and keeps the
# tunnel alive with a supervisor loop (a quick tunnel dies when the link drops).
set -uo pipefail

PORT="${1:-9000}"
BIN="$HOME/sciagent/bin/cloudflared"
LOG="${2:-$HOME/sciagent/tunnel.log}"
URL_FILE="$HOME/sciagent/tunnel.url"

if [ ! -x "$BIN" ]; then
  echo "!!! cloudflared not found at $BIN" >&2
  exit 1
fi

# Stop a previous tunnel.
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1

echo "==> starting tunnel to 127.0.0.1:$PORT"
: > "$LOG"

# Supervisor: restart the tunnel if it dies, and refresh the URL file.
(
  while true; do
    "$BIN" tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" >> "$LOG" 2>&1
    echo "--- tunnel exited, restarting in 5s" >> "$LOG"
    sleep 5
  done
) &
echo $! > "$HOME/sciagent/tunnel.pid"

# Wait for the URL to appear in the log.
for i in $(seq 1 60); do
  sleep 2
  url=$(grep -ao "https://[a-z0-9-]*\.trycloudflare\.com" "$LOG" | head -1)
  if [ -n "$url" ]; then
    echo "$url" > "$URL_FILE"
    echo "==> public URL: $url"
    exit 0
  fi
done

echo "!!! the tunnel did not report a URL in time; log tail:" >&2
tail -20 "$LOG" >&2
exit 1
