#!/usr/bin/env bash
# Smoke-test a running SciAgent deployment.
#
#   bash scripts/verify_deployment.sh [base-url]
#
# Checks that the SPA, the API, the static assets and the SSE endpoint all
# answer correctly.  Designed to run against the public demo URL as well as
# against http://127.0.0.1:9000.
set -uo pipefail

BASE="${1:-http://127.0.0.1:9000}"
BASE="${BASE%/}"

fail=0

check() {
  local label="$1" url="$2" expect="$3"
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$url" 2>/dev/null)
  if [ "$code" = "$expect" ]; then
    printf '  [ok]   %-34s %s -> %s\n' "$label" "$url" "$code"
  else
    printf '  [FAIL] %-34s %s -> %s (expected %s)\n' "$label" "$url" "$code" "$expect"
    fail=1
  fi
}

echo "==> verifying $BASE"

check "SPA root"            "$BASE/"                 200
check "SPA deep link"       "$BASE/graph"            200
check "API health"          "$BASE/api/health"       200
check "API papers list"     "$BASE/api/papers"       200
check "API stats"           "$BASE/api/papers/stats" 200
check "API openapi"         "$BASE/api/openapi.json" 200

# The hashed asset name changes with every build, so take it from index.html.
asset=$(curl -s --max-time 20 "$BASE/" | grep -o '/assets/[^"]*\.js' | head -1)
if [ -n "$asset" ]; then
  check "hashed JS asset"    "$BASE$asset"            200
else
  echo "  [warn] no hashed asset found in index.html"
fi

echo "==> health payload"
curl -s --max-time 20 "$BASE/api/health" | head -c 600
echo ""
echo "==> SSE endpoint (expect an event stream, 5 s window)"
timeout 5 curl -sN --max-time 5 -X POST "$BASE/api/rag/ask" \
  -H 'Content-Type: application/json' \
  -d '{"question":"health probe","paper_ids":[]}' 2>/dev/null | head -5
echo ""
echo "==> done"
exit $fail
