# syntax=docker/dockerfile:1
#
# SciAgent 后端镜像 (FastAPI + uvicorn)
# 构建上下文为 ./backend，因此 COPY 路径相对于 backend/。
#
#   docker build -t sciagent-backend ./backend
#
FROM python:3.11-slim

# pip 走国内镜像：官方 PyPI 在境内经常超时，拉不动 torch/scikit-learn 这类大包
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL}

# build-essential : 少数依赖没有预编译 wheel 时需要本地编译 (Cython 等)
# libgl1/libglib2.0-0 : matplotlib 运行时依赖 (import matplotlib.pyplot)
# curl             : HEALTHCHECK 探针
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只拷贝依赖清单再安装：源码变动不会让这一层缓存失效，重建镜像快得多
COPY requirements.txt ./
RUN pip install --no-cache-dir \
        -i "${PIP_INDEX_URL}" \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        -r requirements.txt

# 稠密检索 / 重排是可选增强：requirements.txt 里不含 sentence-transformers，
# 缺失时后端会自动退化为纯 BM25（app/rag/embeddings.py 捕获 ImportError）。
# 需要开启时构建： docker build --build-arg INSTALL_EMBEDDINGS=true ./backend
ARG INSTALL_EMBEDDINGS=false
RUN if [ "${INSTALL_EMBEDDINGS}" = "true" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.3.0" && \
        pip install --no-cache-dir -i "${PIP_INDEX_URL}" "sentence-transformers>=3.0.0"; \
    fi

COPY app ./app
COPY pyproject.toml ./

# 非 root 运行；data/ 与 models/ 会在 compose 里挂具名卷
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/data /app/models \
    && chown -R app:app /app
USER app

# 运行时目录/缓存位置，与 docker-compose.yml 里的具名卷保持一致
ENV DATA_DIR=/app/data \
    INDEX_DIR=/app/data/index \
    AUDIT_DIR=/app/data/audit \
    HF_HOME=/app/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/models/sentence-transformers

EXPOSE 8000

# /api/health 由 app/api/routers/health.py 提供，是真正的就绪探针
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
