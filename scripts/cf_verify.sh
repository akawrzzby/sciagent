#!/usr/bin/env bash
# Verify the cloudflared binary on the host.
BIN="$HOME/sciagent/bin/cloudflared"
echo "--- file"
ls -la "$BIN"
echo "--- magic"
head -c 4 "$BIN" | od -An -tx1
echo "--- version"
chmod +x "$BIN" 2>/dev/null
"$BIN" --version 2>&1 | head -3
echo "exit=$?"
