#!/usr/bin/env bash
# Upload the official test dataset into the running SciAgent instance.
#
#   bash scripts/seed_library.sh [base-url] [data-dir]
#
# Notes:
#   * The upload endpoint parses in the background, so the script uploads a
#     batch and then polls until the expected number of papers has landed.
#   * ``allow_duplicate=true`` is required: the dataset contains two revisions
#     of the same arXiv paper (1706.03762 v1 / v7) and the version-diff
#     feature needs both revisions in the library.
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

echo "==> uploading to $BASE"
present=()
for f in "${FILES[@]}"; do
  if [ -f "$DATA_DIR/$f" ]; then
    present+=("$DATA_DIR/$f")
  else
    echo "    skip (missing): $f"
  fi
done

if [ ${#present[@]} -eq 0 ]; then
  echo "!!! nothing to upload - is $DATA_DIR correct?" >&2
  exit 1
fi

args=()
for f in "${present[@]}"; do
  args+=(-F "files=@$f")
done

resp=$(curl -s --max-time 900 -X POST "$BASE/api/papers/upload" \
        "${args[@]}" -F "allow_duplicate=true")
echo "    $resp" | head -c 700
echo ""

echo "==> waiting for the background parsing to finish (expect $EXPECTED papers)"
prev=-1
for i in $(seq 1 240); do
  sleep 10
  stats=$(curl -s --max-time 30 "$BASE/api/papers/stats")
  total=$(echo "$stats" | sed -n 's/.*"papers": *\([0-9]*\).*/\1/p')
  echo "    t=$((i * 10))s  $stats"
  if [ "$total" = "$EXPECTED" ]; then
    echo "    reached the expected count"
    break
  fi
  # Bail out if nothing changed for 12 consecutive polls (2 minutes).
  if [ "$total" = "$prev" ]; then
    stagnant=$((stagnant + 1))
    if [ "$stagnant" -ge 12 ]; then
      echo "    stalled at $total papers"
      break
    fi
  else
    stagnant=0
  fi
  prev="$total"
done

echo ""
echo "==> final state"
curl -s --max-time 30 "$BASE/api/papers/stats"
echo ""
