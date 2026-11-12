#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Parallel chunked downloader for the retrieval models.
#
# The HF mirror in this lab stalls at 0 B/s for large blobs, so we fetch from
# ModelScope instead and split every file into 16 MB ranges that are pulled by
# a pool of curl workers.  The result is a plain HuggingFace directory that
# sentence-transformers can load directly.
#
# usage: download_hf_model.sh <model-id> <target-dir> [connections]
#   e.g. download_hf_model.sh BAAI/bge-m3 ~/models/bge-m3 16
# ---------------------------------------------------------------------------
set -uo pipefail

MODEL_ID="${1:?model id required}"
TARGET="${2:?target dir required}"
CONN="${3:-16}"
CHUNK=$((16 * 1024 * 1024))

MS="https://www.modelscope.cn/api/v1/models/${MODEL_ID}/repo?Revision=master&FilePath="

mkdir -p "$TARGET"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo ">>> model: $MODEL_ID -> $TARGET (conn=$CONN)"

# ---------------------------------------------------------------- file list
# The ModelScope tree API returns JSON; only "blob" entries are files, and the
# heavy auxiliary trees (onnx/, imgs/) are skipped on purpose.
FILES=$(python3 - "$MODEL_ID" <<'PY'
import json, sys, urllib.request

model = sys.argv[1]
url = f"https://www.modelscope.cn/api/v1/models/{model}/repo/files?Revision=master"
with urllib.request.urlopen(url, timeout=60) as fh:
    data = json.load(fh)

skip_prefix = ("onnx/", "imgs/", "onnx_int8/", "openvino/")
skip_names = {".gitattributes", "README.md", "README_zh.md", "config_sentence_transformers.json"}

out = []
for node in data.get("Data", {}).get("Files", []):
    if node.get("Type") != "blob":
        continue
    path = node.get("Path", "")
    if path in skip_names or path.startswith(skip_prefix):
        continue
    out.append((path, int(node.get("Size") or 0)))

for path, size in sorted(out):
    print(f"{path}\t{size}")
PY
)

if [ -z "$FILES" ]; then
  echo "!!! could not list files for $MODEL_ID" >&2
  exit 1
fi
echo ">>> $(echo "$FILES" | wc -l) files"

# --------------------------------------------------------------- one file
fetch_one() {
  local path="$1"
  local size_hint="$2"
  local out="$TARGET/$path"
  if [ -s "$out" ]; then echo "  skip  $path"; return 0; fi
  mkdir -p "$(dirname "$out")"

  local size
  size=$(curl -sIL --max-time 60 "${MS}${path}" | tr -d '\r' \
        | awk 'tolower($1)=="content-length:"{v=$2} END{print v+0}')
  [ -z "$size" ] && size=0
  [ "$size" -eq 0 ] && size="$size_hint"

  if [ "$size" -le "$CHUNK" ]; then
    curl -sL --max-time 900 --retry 3 -o "$out" "${MS}${path}" \
      && echo "  ok    $path (${size}B)" \
      || { echo "  FAIL  $path"; rm -f "$out"; return 1; }
    return 0
  fi

  local n=$(( (size + CHUNK - 1) / CHUNK ))
  echo "  ..    $path (${size}B, ${n} chunks)"
  local safe="${path//\//_}"
  local i
  for i in $(seq 0 $((n - 1))); do
    local start=$(( i * CHUNK ))
    local end=$(( start + CHUNK - 1 ))
    [ "$end" -ge "$size" ] && end=$(( size - 1 ))
    echo "$start $end $i"
  done | xargs -P "$CONN" -I{} -d '\n' bash -c '
      set -- {}
      start=$1; end=$2; idx=$3
      part="'"$WORK"'/${safe}.part${idx}"
      curl -sL --max-time 900 --retry 3 -r "${start}-${end}" -o "$part" "'"${MS}"'${path}" \
        && [ -s "$part" ] || { echo "  chunk FAIL $path #$idx"; exit 1; }
    '

  : > "$out"
  for i in $(seq 0 $((n - 1))); do cat "$WORK/${safe}.part${i}" >> "$out"; done
  local got
  got=$(stat -c %s "$out")
  if [ "$got" -eq "$size" ]; then
    echo "  ok    $path (${size}B)"
  else
    echo "  SIZE MISMATCH $path: ${got} != ${size}"
    rm -f "$out"
    return 1
  fi
}

export -f fetch_one
export WORK TARGET MS CHUNK

echo "$FILES" | tr '\t' ' ' | xargs -P 4 -I{} bash -c 'fetch_one "$@"' _ {}

echo ">>> done: $TARGET"
du -sh "$TARGET"
find "$TARGET" -maxdepth 2 -type f -printf '  %10s %p\n' | sort -k2
