#!/usr/bin/env python3
"""Compare the two retrieval A/B cases, row by row."""
import json, os

DATA = os.path.expanduser("~/sciagent/data")
data = {}
for label in ("before", "after"):
    p = os.path.join(DATA, "eval_retrieval_%s.json" % label)
    data[label] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

for label, d in data.items():
    if not d:
        continue
    print("=== %s ===" % label)
    for k in ("label", "queries", "top_k", "hit@5", "mrr", "mean_metadata_affinity"):
        print("   %-24s %s" % (k, d.get(k)))
    print()

b = data.get("before") or {}
a = data.get("after") or {}
rb = {r.get("query"): r for r in b.get("rows", [])}
ra = {r.get("query"): r for r in a.get("rows", [])}
print("=== per query ===")
for q in rb:
    x, y = rb[q], ra.get(q, {})
    mark = "  <== 提升" if x.get("rank") != y.get("rank") else ""
    print("  before rank=%-4s after rank=%-4s | %s%s" % (
        x.get("rank"), y.get("rank"), (q or "")[:56], mark))
