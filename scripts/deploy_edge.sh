#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Deploy the SciAgent edge layer on the Docker-capable host.
#
#   bash scripts/deploy_edge.sh <api-upstream> [frontend-dist-dir]
#
#   api-upstream       e.g. http://10.184.17.179:9000
#   frontend-dist-dir  directory holding the built Vite output (default
#                      ~/sciagent/frontend/dist)
#
# Two containers are started on a dedicated docker network:
#
#   sciagent-nginx    serves the SPA and reverse-proxies /api (SSE aware)
#   sciagent-tunnel   exposes the nginx container through a Cloudflare
#                      quick tunnel, which prints a public https:// URL
#
# The quick tunnel URL changes on every restart.  For a stable address use a
# named tunnel instead (see docs/demo-script.md).
# ---------------------------------------------------------------------------
set -uo pipefail

API_UPSTREAM="${1:?usage: deploy_edge.sh <api-upstream> [dist-dir]}"
DIST_DIR="${2:-$HOME/sciagent/frontend/dist}"
WORK_DIR="${WORK_DIR:-$HOME/sciagent/edge}"
NETWORK="${NETWORK:-sciagent-edge}"
NGINX_NAME="${NGINX_NAME:-sciagent-nginx}"
TUNNEL_NAME="${TUNNEL_NAME:-sciagent-tunnel}"

echo "==> api upstream : $API_UPSTREAM"
echo "==> frontend dist: $DIST_DIR"

if [ ! -f "$DIST_DIR/index.html" ]; then
  echo "!!! $DIST_DIR/index.html not found - build the frontend first" >&2
  exit 1
fi

mkdir -p "$WORK_DIR/templates"
cp "$DIST_DIR"/* "$WORK_DIR/" 2>/dev/null || true

# The template lives next to this script in the repository; fall back to a
# copy that was placed on the host manually.
TEMPLATE_SRC="${TEMPLATE_SRC:-$(dirname "$0")/../deploy/nginx.edge.conf.template}"
if [ ! -f "$TEMPLATE_SRC" ]; then
  echo "!!! nginx template not found at $TEMPLATE_SRC" >&2
  exit 1
fi
cp "$TEMPLATE_SRC" "$WORK_DIR/templates/default.conf.template"

echo "==> restarting containers"
docker rm -f "$TUNNEL_NAME" "$NGINX_NAME" >/dev/null 2>&1 || true
docker network rm "$NETWORK" >/dev/null 2>&1 || true
docker network create "$NETWORK" >/dev/null

echo "==> starting nginx"
docker run -d --name "$NGINX_NAME" --restart unless-stopped \
  --network "$NETWORK" \
  -v "$WORK_DIR":/usr/share/nginx/html:ro \
  -v "$WORK_DIR/templates":/etc/nginx/templates:ro \
  -e API_UPSTREAM="$API_UPSTREAM" \
  nginx:alpine >/dev/null

sleep 3
echo "==> nginx status: $(docker inspect -f '{{.State.Status}}' "$NGINX_NAME" 2>/dev/null)"

# Wait for nginx to answer on the container network.
for i in $(seq 1 20); do
  if docker run --rm --network "$NETWORK" curlimages/curl:latest -fsS \
       "http://$NGINX_NAME/api/health" >/dev/null 2>&1; then
    echo "==> nginx proxy to the API is healthy (after ${i}s)"
    break
  fi
  sleep 1
done

echo "==> starting the cloudflared quick tunnel"
docker rm -f "$TUNNEL_NAME" >/dev/null 2>&1 || true
docker run -d --name "$TUNNEL_NAME" --restart unless-stopped \
  --network "$NETWORK" \
  cloudflare/cloudflared:latest \
  tunnel --no-autoupdate --url "http://$NGINX_NAME:80" >/dev/null

echo "==> waiting for the public URL"
URL=""
for i in $(seq 1 40); do
  URL=$(docker logs "$TUNNEL_NAME" 2>&1 | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1)
  if [ -n "$URL" ]; then
    echo ""
    echo "================================================================"
    echo "  PUBLIC URL: $URL"
    echo "================================================================"
    break
  fi
  sleep 2
done

if [ -z "$URL" ]; then
  echo "!!! the tunnel did not print a URL - check 'docker logs $TUNNEL_NAME'" >&2
  exit 1
fi

echo ""
echo "==> local check"
curl -s -o /dev/null -w "  nginx  -> HTTP %{http_code}\n" "http://127.0.0.1" 2>/dev/null || true
curl -s -o /dev/null -w "  api    -> HTTP %{http_code}\n" "$API_UPSTREAM/api/health" 2>/dev/null || true
echo ""
echo "Keep the containers alive with:"
echo "  docker logs -f $TUNNEL_NAME"
