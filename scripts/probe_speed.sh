#!/usr/bin/env bash
# Measure the achievable download rate for the retrieval models.
# The HF mirror may be slow; ModelScope is probed as an alternative source.
set -uo pipefail

HUB="$HOME/.cache/huggingface/hub/models--BAAI--bge-m3/blobs"

echo "=== current hf-mirror progress ==="
f=$(ls -1 "$HUB"/*.incomplete 2>/dev/null | head -1)
if [ -n "$f" ]; then
  s1=$(stat -c %s "$f")
  sleep 20
  s2=$(stat -c %s "$f")
  rate=$(( (s2 - s1) / 20 ))
  echo "file : $f"
  echo "size : $s1 -> $s2 bytes"
  echo "rate : ${rate} B/s"
  echo "eta  : $(( (2270769664 - s2) / (rate > 0 ? rate : 1) / 60 )) min"
else
  echo "no incomplete blob (download may have finished)"
fi

echo
echo "=== ModelScope probe (BAAI/bge-m3) ==="
curl -sL -o /dev/null \
  -w '  config.json : %{speed_download} B/s http=%{http_code} size=%{size_download}\n' \
  --max-time 30 \
  'https://www.modelscope.cn/api/v1/models/BAAI/bge-m3/repo?Revision=master&FilePath=config.json'

echo "=== ModelScope probe (8 MB range of pytorch_model.bin) ==="
curl -sL -o /dev/null \
  -w '  range       : %{speed_download} B/s http=%{http_code} size=%{size_download}\n' \
  --max-time 30 -r 0-8388608 \
  'https://www.modelscope.cn/api/v1/models/BAAI/bge-m3/repo?Revision=master&FilePath=pytorch_model.bin'

echo
echo "=== hf-mirror probe (8 MB range of pytorch_model.bin) ==="
curl -sL -o /dev/null \
  -w '  range       : %{speed_download} B/s http=%{http_code} size=%{size_download}\n' \
  --max-time 30 -r 0-8388608 \
  'https://hf-mirror.com/BAAI/bge-m3/resolve/main/pytorch_model.bin'

echo
echo "=== mirror list check ==="
for host in mirrors.tuna.tsinghua.edu.cn hf-mirror.com modelscope.cn; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "https://$host/" 2>/dev/null)
  echo "  ${host}: http=${code}"
done
