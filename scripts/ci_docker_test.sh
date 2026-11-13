#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the backend image and prove that it actually contains the current
# source -- then run the test suite *inside* that image.
#
# Why this exists
# ---------------
# During the pre-submission build we hit a real defect that no "docker build
# succeeded" check would ever catch: two builds were running at the same time.
# The one that started EARLIER (15:21:01, 24m15s) finished LATER (15:45) than
# the one started at 15:36, and it wrote the `latest` tag with a context
# snapshot taken six minutes before a bug fix was committed.  The running
# container was fine; the tag pointed at stale code.  A test case caught it.
#
# So this script enforces two things, in this order:
#   1. the image's /app/app must be byte-identical to the working tree
#      (a `diff -r` against a bind mount) -- otherwise the build is rejected;
#   2. the test suite must pass inside that exact image.
#
# Usage (Linux / macOS / WSL):
#   bash scripts/ci_docker_test.sh
#
# Requires: docker, and network access for `pip install pytest` inside the
# container (the image itself ships no test dependencies).
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-sciagent-backend:ci}"

echo "==> 1/3  build $IMAGE from $ROOT/backend"
docker build -t "$IMAGE" "$ROOT/backend"

echo
echo "==> 2/3  verify the image contains the current source (anti stale-tag guard)"
# /src/app is the working tree, /app/app is what the image shipped.
docker run --rm -v "$ROOT/backend/app:/src/app:ro" "$IMAGE" sh -c '
    if diff -r /app/app /src/app; then
        echo "    OK: image code == working tree"
    else
        echo "    FAIL: the image does not match the working tree." >&2
        echo "          This is the stale-image race documented in docs/technical-report.md 6.6." >&2
        exit 1
    fi
'

echo
echo "==> 3/3  run the test suite inside $IMAGE"
docker run --rm \
    -v "$ROOT/backend/tests:/app/tests:ro" \
    -v "$ROOT/scripts:/scripts:ro" \
    "$IMAGE" sh -c '
        if ! python -c "import pytest" 2>/dev/null; then
            pip install -q pytest >/dev/null 2>&1 || {
                echo "    FAIL: cannot install pytest inside the image (no network?)" >&2
                exit 1
            }
        fi
        python -m pytest tests/ -q -rs -p no:cacheprovider
    '

echo
echo "==> done: image verified and tests passed"
