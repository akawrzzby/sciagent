"""The negative control of the RAG metrics.

Three questions that the library cannot answer.  A metric that reports
"supported" for a question nobody in the library answers is not measuring
anything.  The agent is expected to abstain; the interesting number is not
the faithfulness of the refusal (a refusal has no claim to support) but the
*rate of honest refusals*: how many of the three questions are answered with
"the documents do not contain this" instead of an invented number.
"""

import json
import os
import sqlite3
import sys

sys.path.insert(0, "/home/wrz/sciagent/backend")

import httpx

from app.rag.evaluation import evaluate_answer

BASE = os.environ.get("SCIAGENT_BASE", "http://127.0.0.1:9000")
DB = os.environ.get(
    "SCIAGENT_DB", os.path.expanduser("~/sciagent/data/sciagent.db")
)

QUESTIONS = [
    "What is the validation loss of the Transformer on CIFAR-10 after 100 epochs?",
    "Which conference did this paper win the best paper award at?",
    "What GPU budget in USD was spent on training the big Transformer?",
]

# Phrasings that count as an honest refusal.  The agent is free to phrase it
# in its own words; what matters is that it does not invent a number.
REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "not stated",
    "no information",
    "not mentioned",
    "cannot be determined",
    "not specified",
    "not available",
)


def chunk_texts(ids):
    ids = [i for i in ids if i]
    if not ids:
        return {}
    out = {}
    conn = sqlite3.connect(DB)
    try:
        marks = ",".join("?" * len(ids))
        for cid, content in conn.execute(
                "select id, content from chunks where id in (%s)" % marks, ids):
            out[cid] = content or ""
    finally:
        conn.close()
    return out


def is_refusal(answer):
    low = (answer or "").lower()
    return any(m in low for m in REFUSAL_MARKERS)


results = []
for question in QUESTIONS:
    print("=" * 72)
    print("Q:", question)
    payload = httpx.post(BASE + "/api/rag/ask",
                         json={"question": question, "paper_ids": [], "stream": False},
                         timeout=900).json()
    answer = payload.get("answer", "")
    citations = payload.get("citations", []) or []
    texts = chunk_texts({str(c.get("chunk_id") or "") for c in citations})
    enriched = [{**c, "text": texts.get(str(c.get("chunk_id") or ""), c.get("quote", ""))}
                for c in citations]
    report = evaluate_answer(answer, enriched)
    refusal = is_refusal(answer)
    print("ANSWER:", answer[:600].replace("\n", " "))
    print("--- faith=%.2f trace=%.2f prec=%.2f claims=%d unsupported=%d refine=%s refusal=%s"
          % (report.faithfulness, report.citation_traceability, report.context_precision,
             len(report.claims), len(report.unsupported), payload.get("refine_rounds"), refusal))
    for claim in report.claims:
        print("   [%s] cov=%.2f crit=%.2f :: %s"
              % ("OK " if claim.supported else "BAD", claim.coverage, claim.critical,
                 claim.text[:120]))
    results.append({
        "question": question,
        "answer": answer,
        "refused": refusal,
        "refine_rounds": payload.get("refine_rounds"),
        "citation_count": len(citations),
        "metrics": {
            "faithfulness": report.faithfulness,
            "citation_traceability": report.citation_traceability,
            "context_precision": report.context_precision,
            "claim_count": len(report.claims),
            "supported_claims": len(report.claims) - len(report.unsupported),
            "unsupported": [c.text for c in report.unsupported],
        },
    })

summary = {
    "questions": len(results),
    "honest_refusals": sum(1 for r in results if r["refused"]),
    "refusal_rate": round(sum(1 for r in results if r["refused"]) / len(results), 4) if results else 0.0,
    "fabricated_answers": sum(1 for r in results if not r["refused"]),
    "mean_faithfulness": round(sum(r["metrics"]["faithfulness"] for r in results) / len(results), 4) if results else 0.0,
}
print()
print("=" * 72)
print("SUMMARY:", json.dumps(summary, ensure_ascii=False))

out = os.environ.get(
    "SCIAGENT_OUT", os.path.expanduser("~/sciagent/data/eval_control.json")
)
if len(sys.argv) > 1:
    out = sys.argv[1]
with open(out, "w", encoding="utf-8") as fh:
    json.dump({"summary": summary, "results": results}, fh, ensure_ascii=False, indent=2)
print("written to", out)
