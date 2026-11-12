#!/usr/bin/env bash
# Install the embedding stack (torch is downloaded separately for speed).
set -euo pipefail
ENV_DIR="${ENV_DIR:-$HOME/envs/sciagent}"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_TELEMETRY=1

echo "==> installing torch from the local wheel"
"$ENV_DIR/bin/pip" install -q /tmp/wheels/torch-*.whl -i "$PIP_MIRROR" \
    --trusted-host pypi.tuna.tsinghua.edu.cn 2>&1 | tail -3

echo "==> installing sentence-transformers"
"$ENV_DIR/bin/pip" install -q "sentence-transformers>=3.0.0" -i "$PIP_MIRROR" \
    --trusted-host pypi.tuna.tsinghua.edu.cn 2>&1 | tail -3

echo "==> torch check"
"$ENV_DIR/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "==> downloading models through $HF_ENDPOINT"
"$ENV_DIR/bin/python" - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
for name in ("BAAI/bge-m3",):
    try:
        print("  downloading", name, flush=True)
        SentenceTransformer(name, trust_remote_code=True)
        print("    ok:", name, flush=True)
    except Exception as exc:
        print("    FAILED:", name, repr(exc)[:300], flush=True)
try:
    print("  downloading BAAI/bge-reranker-v2-m3", flush=True)
    CrossEncoder("BAAI/bge-reranker-v2-m3", trust_remote_code=True)
    print("    ok: reranker", flush=True)
except Exception as exc:
    print("    FAILED: reranker", repr(exc)[:300], flush=True)
PY
echo "==> done"
