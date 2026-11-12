#!/usr/bin/env bash
# Create the SciAgent runtime environment.
#
# Strategy: the GPU host already runs a vLLM conda environment that ships a
# CUDA build of PyTorch.  We create a *separate* virtual environment that
# inherits those site-packages (`--system-site-packages`) so that the dense
# retriever can run on the L40 GPUs without downloading a multi-hundred
# megabyte wheel.  Everything else is installed from the Tsinghua mirror.
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/envs/sciagent}"
BASE_PY="$HOME/miniconda3/envs/qwen3.8-vllm/bin/python"
BACKEND_DIR="${BACKEND_DIR:-$HOME/sciagent/backend}"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "==> creating $ENV_DIR (python 3.12, inheriting the CUDA torch)"
rm -rf "$ENV_DIR"
"$BASE_PY" -m venv --system-site-packages "$ENV_DIR"

echo "==> installing backend dependencies"
"$ENV_DIR/bin/pip" install -q --upgrade pip -i "$PIP_MIRROR" \
    --trusted-host pypi.tuna.tsinghua.edu.cn
"$ENV_DIR/bin/pip" install -q -r "$BACKEND_DIR/requirements.txt" -i "$PIP_MIRROR" \
    --trusted-host pypi.tuna.tsinghua.edu.cn

echo "==> installing sentence-transformers"
"$ENV_DIR/bin/pip" install -q "sentence-transformers>=3.0.0" -i "$PIP_MIRROR" \
    --trusted-host pypi.tuna.tsinghua.edu.cn

echo "==> verification"
"$ENV_DIR/bin/python" - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "devices", torch.cuda.device_count())
for m in ("fastapi", "uvicorn", "pydantic", "sqlalchemy", "aiosqlite", "httpx",
          "pymupdf", "numpy", "networkx", "sklearn", "matplotlib",
          "sentence_transformers", "transformers"):
    try:
        importlib.import_module(m)
        print("  OK      ", m)
    except Exception as exc:
        print("  MISSING ", m, repr(exc)[:120])
PY
echo "==> ready"
