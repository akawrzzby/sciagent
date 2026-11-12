# syntax=docker/dockerfile:1
#
# SciAgent 前端镜像 (Vite 构建 -> Nginx 托管)
#
# 构建上下文必须是仓库根目录，因为运行阶段还要用到 deploy/nginx.conf：
#   docker build -f frontend/Dockerfile -t sciagent-frontend .
#
# ---------------------------------------------------------------- 阶段 1：构建
FROM node:20-alpine AS builder

WORKDIR /build

# npm 国内镜像（与 pip 的清华源同理）
ARG NPM_REGISTRY=https://registry.npmmirror.com

# 先只拷贝依赖清单再安装：源码变动不会让这一层缓存失效
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --registry=${NPM_REGISTRY}

COPY frontend/ ./

# 前端走同源 /api（frontend/src/lib/api.ts: BASE = VITE_API_BASE ?? '/api'），
# 生产环境由 Nginx 反代到 backend:8000，因此这里保持 /api 不带域名。
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=${VITE_API_BASE}

RUN npm run build

# ---------------------------------------------------------------- 阶段 2：运行
FROM nginx:alpine AS runtime

# bash : entrypoint.sh 使用 `#!/usr/bin/env bash` + `set -euo pipefail`
#        (busybox ash 不支持 pipefail)
# curl  : entrypoint.sh 轮询后端 /api/health
RUN apk add --no-cache bash curl

# SPA history 路由 + /api 反向代理（含 SSE 必需的无缓冲配置）
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
# 启动脚本：等后端就绪后再拉起 Nginx
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

COPY --from=builder /build/dist /usr/share/nginx/html

EXPOSE 80

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
