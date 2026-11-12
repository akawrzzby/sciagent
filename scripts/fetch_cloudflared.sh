#!/usr/bin/env bash
# Probe mirrors for a complete cloudflared linux-amd64 binary.
#
# The earlier attempt on this host failed because the GitHub release asset was
# rate limited to ~7 KB/s, so the binary was truncated and segfaulted.  This
# probe measures the throughput and the Content-Length of every candidate so
# that a *complete* download can be picked.
set -uo pipefail

TARGET="${TARGET:-$HOME/bin/cloudflared}"
mkdir -p "$(dirname "$TARGET")"

CANDIDATES=(
  "https://registry.npmmirror.com/-/binary/cloudflared/v2024.12.2/cloudflared-linux-amd64"
  "https://registry.npmmirror.com/-/binary/cloudflared/latest/cloudflared-linux-amd64"
  "https://npmmirror.com/mirrors/cloudflared/latest/cloudflared-linux-amd64"
  "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
  "https://dl.cloudflare.com/client/v4/cloudflared-linux-amd64"
  "https://pkg.cloudflare.com/cloudflared/linux-amd64"
)

for url in "${CANDIDATES[@]}"; do
  echo "--- $url"
  # First a cheap HEAD request to learn the size and the redirect target.
  head=$(curl -sIL --max-time 20 "$url" 2>/dev/null | tr -d '\r')
  code=$(printf '%s' "$head" | awk 'toupper($1) ~ /^HTTP/ {c=$2} END {print c+0}')
  len=$(printf '%s' "$head" | awk 'tolower($1)=="content-length:" {v=$2} END {print v+0}')
  echo "    http=$code content-length=$len"
  if [ "$code" != "200" ] || [ "$len" -lt 10000000 ]; then
    echo "    -> skipped (not a complete binary)"
    continue
  fi
  echo "    downloading..."
  curl -sL --max-time 600 --retry 2 -o "$TARGET" "$url"
  got=$(stat -c %s "$TARGET" 2>/dev/null || echo 0)
  echo "    got $got bytes"
  if [ "$got" -ne "$len" ]; then
    echo "    -> size mismatch, discarding"
    rm -f "$TARGET"
    continue
  fi
  chmod +x "$TARGET"
  echo "    running the version check:"
  if "$TARGET" --version 2>&1 | head -3; then
    echo "    -> SUCCESS: $TARGET"
    exit 0
  fi
  echo "    -> binary does not run on this host"
  rm -f "$TARGET"
done

echo "no usable cloudflared binary found"
exit 1
