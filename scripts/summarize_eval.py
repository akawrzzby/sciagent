#!/usr/bin/env python3
"""Aggregate the three evaluation rounds into one table."""
import json, os

DATA = os.path.expanduser("~/sciagent/data")
runs = {}
for v in ("v4", "v5", "v6"):
    p = os.path.join(DATA, "eval_rag_%s.json" % v)
    if os.path.exists(p):
        runs[v] = json.load(open(p, encoding="utf-8"))

def rows(d):
    out = []
    for q in d.get("results", []):
        m = q.get("metrics", {})
        out.append({
            "kind": q.get("kind", "?"),
            "label": q.get("label", ""),
            "f": m.get("faithfulness"),
            "t": m.get("citation_traceability"),
            "p": m.get("context_precision"),
            "claims": m.get("claim_count"),
            "sup": m.get("supported_claims"),
            "cc": q.get("citation_count"),
            "ret": m.get("retrieved_chunks"),
            "used": m.get("used_chunks"),
            "rr": q.get("refine_rounds"),
        })
    return out

print("=== per question, three runs (v4 / v5 / v6) ===")
agg = {}
for v in ("v4", "v5", "v6"):
    if v not in runs:
        continue
    for r in rows(runs[v]):
        agg.setdefault((r["kind"], r["label"]), []).append(r)

for (kind, label), vals in agg.items():
    fs = [v["f"] for v in vals if v["f"] is not None]
    ps = [v["p"] for v in vals if v["p"] is not None]
    cs = [v["claims"] for v in vals if v["claims"] is not None]
    ss = [v["sup"] for v in vals if v["sup"] is not None]
    print("  %-6s %-34s faith %s  mean %.4f | prec %s mean %.4f | claims %s sup %s | refine %s" % (
        kind, label,
        "/".join("%.2f" % x for x in fs), sum(fs) / len(fs),
        "/".join("%.2f" % x for x in ps), sum(ps) / len(ps),
        "/".join(str(x) for x in cs), "/".join(str(x) for x in ss),
        "/".join(str(v["rr"]) for v in vals)))

print()
print("=== overall ===")
fs, ts, ps = [], [], []
for v in ("v4", "v5", "v6"):
    if v not in runs:
        continue
    s = runs[v]["summary"]
    fs.append(s["faithfulness"]); ts.append(s["citation_traceability"]); ps.append(s["context_precision"])
    print("  %s: faith=%.4f trace=%.4f prec=%.4f claims=%d sup=%d refine=%d" % (
        v, s["faithfulness"], s["citation_traceability"], s["context_precision"],
        s["claims"], s["supported_claims"], s["refine_rounds"]))
print("  MEAN  faith=%.4f trace=%.4f prec=%.4f" % (sum(fs)/len(fs), sum(ts)/len(ts), sum(ps)/len(ps)))
print("  RANGE faith=[%.4f, %.4f]  trace=[%.4f, %.4f]  prec=[%.4f, %.4f]" % (
    min(fs), max(fs), min(ts), max(ts), min(ps), max(ps)))
print("  STD   faith=%.4f" % (sum((x - sum(fs)/len(fs))**2 for x in fs) / len(fs))**0.5)

print()
print("=== retrieved vs used chunks (v4) ===")
for r in rows(runs["v4"]):
    print("  %-6s %-34s retrieved=%s used=%s" % (r["kind"], r["label"], r["ret"], r["used"]))

ctl = os.path.join(DATA, "eval_control.json")
if os.path.exists(ctl):
    c = json.load(open(ctl, encoding="utf-8"))
    print()
    print("=== control ===")
    print(json.dumps(c.get("summary", {}), ensure_ascii=False))
    for r in c.get("results", []):
        m = r.get("metrics", {})
        print("  %-46s faith=%s claims=%s unsupported=%s | %s" % (
            r.get("question", "")[:46], m.get("faithfulness"), m.get("claim_count"),
            len(m.get("unsupported", [])), str(r.get("answer", ""))[:60]))
else:
    print("NO control json (check eval_control.log)")
