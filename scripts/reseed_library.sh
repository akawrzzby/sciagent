#!/usr/bin/env bash
# Wipe the library and re-seed it with the official test dataset.
#
#   bash scripts/reseed_library.sh [base-url] [data-dir]
#
# Used after a parsing fix: the stored artefacts were produced by the old
# code path, so they are deleted and every document is parsed again.
set -uo pipefail

BASE="${1:-http://127.0.0.1:9000}"
DATA_DIR="${2:-$HOME/ra_testdata}"

echo "==> current library"
curl -s --max-time 30 "$BASE/api/papers" | grep -o '"id": *"[^"]*"' | sed 's/.*: *"//;s/"//' > /tmp/paper_ids.txt
n=$(wc -l < /tmp/paper_ids.txt)
echo "    $n documents"

if [ "$n" -gt 0 ]; then
  while read -r pid; do
    [ -z "$pid" ] && continue
    code=$(curl -s --max-time 30 -o /dev/null -w "%{http_code}" -X DELETE "$BASE/api/papers/$pid")
    echo "    delete $pid -> $code"
  done < /tmp/paper_ids.txt
fi

echo "==> after wipe"
curl -s --max-time 30 "$BASE/api/papers/stats"
echo ""
