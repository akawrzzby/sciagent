#!/usr/bin/env bash
#
# 前端容器启动脚本
#   1. 等待后端 /api/health 就绪，避免用户一打开页面就看到 502
#   2. 拉起 Nginx（前台运行，作为容器主进程）
#
# 由 frontend/Dockerfile 作为 ENTRYPOINT 使用。
set -euo pipefail

# 后端地址：compose 里的服务名，走 Docker 内置 DNS 解析
BACKEND_URL="${BACKEND_URL:-http://backend:8000}"
HEALTH_URL="${BACKEND_URL}/api/health"
# 最长等待时间（秒）
MAX_WAIT="${MAX_WAIT_SECONDS:-120}"
# 每次探测间隔（秒）
INTERVAL="${RETRY_INTERVAL_SECONDS:-2}"

echo "[entrypoint] 等待后端就绪: ${HEALTH_URL}（最长 ${MAX_WAIT}s）"

attempt=0
elapsed=0
until curl -fsS --max-time 5 "${HEALTH_URL}" >/dev/null 2>&1; do
    attempt=$(( attempt + 1 ))
    elapsed=$(( elapsed + INTERVAL ))

    if [ "${elapsed}" -ge "${MAX_WAIT}" ]; then
        # 超过等待上限不直接退出：让 Nginx 起来，前端至少能加载静态资源，
        # 后端恢复后前端刷新即可正常工作。
        echo "[entrypoint] 后端 ${MAX_WAIT}s 内未就绪（已重试 ${attempt} 次），仍继续启动 Nginx" >&2
        break
    fi

    echo "[entrypoint] 后端未就绪，第 ${attempt} 次重试（${elapsed}s/${MAX_WAIT}s）"
    sleep "${INTERVAL}"
done

if [ "${attempt}" -gt 0 ]; then
    echo "[entrypoint] 后端已就绪（重试 ${attempt} 次，用时 ${elapsed}s）"
else
    echo "[entrypoint] 后端立即就绪"
fi

echo "[entrypoint] 启动 Nginx"
exec nginx -g 'daemon off;'
