#!/usr/bin/env bash
# Re-check the graph and the contradiction detection on the CURRENT library.
cd "$HOME/sciagent" || exit 1
PY="$HOME/envs/sciagent/bin/python"

echo "=== library size ==="
curl -s --max-time 30 http://127.0.0.1:9000/api/papers > /tmp/papers.json
"$PY" - <<'PYEOF'
import json
d = json.load(open("/tmp/papers.json", encoding="utf-8"))
print("papers:", len(d.get("papers", [])))
PYEOF

echo
echo "=== contradictions ==="
curl -s --max-time 180 -X POST http://127.0.0.1:9000/api/graph/contradictions \
    -H 'Content-Type: application/json' -d '{"paper_ids": []}' > /tmp/contra.json 2>/dev/null
"$PY" - <<'PYEOF'
import json
try:
    d = json.load(open("/tmp/contra.json", encoding="utf-8"))
except Exception as e:
    print("read failed:", e)
    raise SystemExit
if isinstance(d, dict):
    for k, v in d.items():
        if isinstance(v, list):
            print("  %-22s %d item(s)" % (k, len(v)))
        elif not isinstance(v, (dict,)):
            print("  %-22s %s" % (k, v))
    items = d.get("contradictions") or d.get("items") or []
    for c in items[:10]:
        print("   -", json.dumps(c, ensure_ascii=False)[:400])
else:
    print(str(d)[:600])
PYEOF
