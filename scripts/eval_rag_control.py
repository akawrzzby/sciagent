"""The negative control of the RAG metrics.

Two questions that the library cannot answer.  A metric that reports
"supported" for a question nobody in the library answers is not measuring
anything.  The agent is expected to abstain or to answer with the
"not stated" phrasing -- and if it does answer, the faithfulness must drop.
"""

import json
import os
import sqlite3
import sys

sys.path.insert(0, "/home/wrz/sciagent/backend")

import httpx

from app.rag.evaluation import evaluate_answer

BASE = "http://127.0.0.1:9000"
DB = "/home/wrz/sciagent/data/sciagent.db"

QUESTIONS = [
    "What is the validation loss of the Transformer on CIFAR-10 after 100 epochs?",
    "Which conference did this paper win the best paper award at?",
    "What GPU budget in USD was spent on training the big Transformer?",
]


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
    print("ANSWER:", answer[:600].replace("\n", " "))
    print("--- faith=%.2f trace=%.2f prec=%.2f claims=%d unsupported=%d refine=%s"
          % (report.faithfulness, report.citation_traceability, report.context_precision,
             len(report.claims), len(report.unsupported), payload.get("refine_rounds")))
    for claim in report.claims:
        print("   [%s] cov=%.2f crit=%.2f :: %s"
              % ("OK " if claim.supported else "BAD", claim.coverage, claim.critical,
                 claim.text[:120]))
