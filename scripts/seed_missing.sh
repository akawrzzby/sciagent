#!/usr/bin/env bash
# Re-upload the documents that did not make it into the library yet.
#
#   bash scripts/seed_missing.sh [base-url] [data-dir]
#
# The library is inspected first and only the missing files are uploaded, so
# the script is safe to re-run.
set -uo pipefail

BASE="${1:-http://127.0.0.1:9000}"
DATA_DIR="${2:-$HOME/ra_testdata}"

FILES=(
  "1706.03762v1.pdf"
  "1706.03762v7.pdf"
  "1910.02054v3.pdf"
  "1910.10683v4.pdf"
  "2005.14165v4.pdf"
  "1607.06450v1.pdf"
  "1707.06347v2.pdf"
  "2206.07682v2.pdf"
  "2107.03374v2.pdf"
  "2212.04356v1.pdf"
  "2305.14314v1.pdf"
  "2307.09288v2.pdf"
  "1706.03762v1_img.pdf"
)
EXPECTED="${EXPECTED:-13}"

# What is already in the library?
have=$(curl -s --max-time 60 "$BASE/api/papers" | grep -o '"filename": *"[^"]*"' | wc -l)
echo "==> library currently holds $have documents"

args=()
count=0
for f in "${FILES[@]}"; do
  if [ ! -f "$DATA_DIR/$f" ]; then
    echo "    skip (missing on disk): $f"
    continue
  fi
  if curl -s --max-time 60 "$BASE/api/papers" | grep -q "\"filename\": \"$f\""; then
    echo "    already present: $f"
    continue
  fi
  args+=(-F "files=@$DATA_DIR/$f")
  count=$((count + 1))
done

if [ "$count" -eq 0 ]; then
  echo "==> nothing to upload"
  exit 0
fi

echo "==> uploading $count document(s)"
curl -s --max-time 900 -X POST "$BASE/api/papers/upload" \
  "${args[@]}" -F "allow_duplicate=true" | head -c 500
echo ""

echo "==> waiting"
prev=-1
stagnant=0
for i in $(seq 1 240); do
  sleep 10
  stats=$(curl -s --max-time 30 "$BASE/api/papers/stats")
  total=$(echo "$stats" | sed -n 's/.*"papers": *\([0-9]*\).*/\1/p')
  echo "    t=$((i * 10))s  $stats"
  if [ "$total" = "$EXPECTED" ]; then
    echo "    complete"
    break
  fi
  if [ "$total" = "$prev" ]; then
    stagnant=$((stagnant + 1))
    if [ "$stagnant" -ge 18 ]; then
      echo "    stalled at $total"
      break
    fi
  else
    stagnant=0
  fi
  prev="$total"
done

echo "==> final"
curl -s --max-time 30 "$BASE/api/papers/stats"
echo ""
