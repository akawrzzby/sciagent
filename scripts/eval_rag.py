"""Evaluate the grounded QA pipeline on the organiser's question set.

The question set mirrors the two scenarios the brief names explicitly:

* single paper -- author and affiliation, the beta_1/beta_2 hyper-parameters,
  the training GPU models, the BLEU score of a task, and the full reference
  of a cited work;
* cross paper -- how BERT relates to the Transformer encoder, the parameter
  growth of GPT-1/2/3, the repositioning of layer normalisation, and the
  growth of the context window.

Three rule based metrics are computed for every question (see
``app/rag/evaluation.py``):

``faithfulness``          the share of the answer's claims that its own
                          citations actually support
``citation_traceability`` the share of the citation tags that resolve to a
                          retrieved passage
``context_precision``     the share of the retrieved passages the answer used

A claim counts as supported when the cited passage covers enough of its
wording *and* states every critical fact of it (numbers, model names and
acronyms).  The check runs against the whole passage, not against the short
quote that the interface shows, so the passages are read back from the
database by their chunk id.

Run on the server:
    ~/envs/sciagent/bin/python ~/sciagent/scripts/eval_rag.py \
        --base http://127.0.0.1:9000 --out ~/sciagent/data/eval_rag.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.expanduser("~/sciagent/backend"))

import httpx  # noqa: E402

from app.rag.evaluation import evaluate_answer  # noqa: E402

# --------------------------------------------------------------------- questions
SINGLE_QUESTIONS = [
    ("Who is the 6th author of the paper and which institution is he "
     "affiliated with?", "6th author + institution"),
    ("What are the values of the hyper-parameters beta_1 and beta_2 used "
     "in the Adam optimizer?", "beta_1 / beta_2"),
    ("Which GPU models were used for training, and how many of them?", "GPU models + count"),
    ("What BLEU score did the model obtain on the WMT 2014 English-to-German "
     "translation task?", "BLEU score"),
    ("What is the full reference (authors, year, title) of the work that "
     "introduced layer normalization, as cited in this paper?",
     "full reference of the cited work"),
]

MULTI_QUESTIONS = [
    ("Compare BERT with the Transformer encoder: which parts did BERT inherit "
     "and which did it discard?", "BERT vs Transformer encoder"),
    ("How did the parameter count evolve from GPT-1 to GPT-2 to GPT-3?",
     "GPT-1/2/3 parameter growth"),
    ("Where is layer normalisation placed in the Transformer blocks, and how "
     "does that differ from the placement in the original BERT?",
     "LN placement details"),
    ("How does the context window grow across the models discussed here?",
     "context window growth"),
]

# The paper the single-paper questions are about.
TARGET_TITLE = "Attention Is All You Need"


def _find_paper(base: str, title: str, exact: bool = True) -> str | None:
    """Return the id of the library paper whose title matches."""
    papers = httpx.get(f"{base}/api/papers", timeout=60).json()
    items = papers.get("papers", papers) if isinstance(papers, dict) else papers
    for p in items:
        t = (p.get("title") or "").strip()
        if (t == title) if exact else (title.lower() in t.lower()):
            return p.get("id")
    return None


def _ask(base: str, question: str, paper_ids: list[str]) -> dict:
    resp = httpx.post(
        f"{base}/api/rag/ask",
        json={"question": question, "paper_ids": paper_ids, "stream": False},
        timeout=900,
    )
    resp.raise_for_status()
    return resp.json()


def _chunk_texts(db_path: str, chunk_ids: set[str]) -> dict[str, str]:
    """The full text of the passages that were cited."""
    ids = [c for c in chunk_ids if c]
    if not ids or not os.path.exists(db_path):
        return {}
    out: dict[str, str] = {}
    conn = sqlite3.connect(db_path)
    try:
        for start in range(0, len(ids), 400):
            batch = ids[start:start + 400]
            marks = ",".join("?" * len(batch))
            sql = f"select id, content from chunks where id in ({marks})"
            for cid, content in conn.execute(sql, batch):
                out[cid] = content or ""
    finally:
        conn.close()
    return out


def run_question(base: str, question: str, kind: str, label: str,
                 paper_ids: list[str], db_path: str) -> dict:
    print(f"  [{kind}] {label} ...", flush=True)
    try:
        payload = _ask(base, question, paper_ids)
    except Exception as exc:  # pragma: no cover - reported, not raised
        print(f"      FAILED: {exc}", flush=True)
        return {"kind": kind, "label": label, "question": question, "error": str(exc)}

    answer = payload.get("answer", "")
    citations = payload.get("citations", []) or []
    texts = _chunk_texts(db_path, {str(c.get("chunk_id") or "") for c in citations})
    enriched = [
        {**c, "text": texts.get(str(c.get("chunk_id") or ""), str(c.get("quote", "")))}
        for c in citations
    ]
    report = evaluate_answer(answer, enriched)
    print(f"      faithfulness={report.faithfulness:.2f} "
          f"traceability={report.citation_traceability:.2f} "
          f"precision={report.context_precision:.2f} "
          f"({len(report.claims)} claims, "
          f"{len(report.unsupported)} unsupported, "
          f"refine_rounds={payload.get('refine_rounds', 0)})", flush=True)
    for claim in report.unsupported:
        print(f"        ! {claim.reason} :: {claim.text[:110]}", flush=True)
    return {
        "kind": kind,
        "label": label,
        "question": question,
        "answer": answer,
        "citation_count": len(citations),
        # The full citation list is kept so that the numbers can be recomputed
        # from the JSON alone, without asking the model again.
        "citations": [
            {
                "index": c.get("index"),
                "chunk_id": c.get("chunk_id"),
                "paper_id": c.get("paper_id"),
                "paper_title": c.get("paper_title"),
                "page_no": c.get("page_no"),
                "section_title": c.get("section_title"),
                "quote": c.get("quote"),
                "text": c.get("text"),
            }
            for c in enriched
        ],
        "refine_rounds": payload.get("refine_rounds", 0),
        "metrics": report.to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:9000")
    ap.add_argument("--out", default=os.path.expanduser("~/sciagent/data/eval_rag.json"))
    ap.add_argument("--db", default=os.path.expanduser("~/sciagent/data/sciagent.db"))
    args = ap.parse_args()
    base = args.base.rstrip("/")

    target = _find_paper(base, TARGET_TITLE)
    if not target:
        raise SystemExit(f"library does not contain a paper titled {TARGET_TITLE!r}")
    print(f"target paper: {target}")

    results: list[dict] = []
    print("single paper questions:")
    for question, label in SINGLE_QUESTIONS:
        results.append(run_question(base, question, "single", label, [target], args.db))

    print("cross paper questions:")
    for question, label in MULTI_QUESTIONS:
        results.append(run_question(base, question, "multi", label, [], args.db))

    ok = [r for r in results if "metrics" in r]
    summary = {
        "questions": len(results),
        "answered": len(ok),
        "faithfulness": round(sum(r["metrics"]["faithfulness"] for r in ok) / max(1, len(ok)), 4),
        "citation_traceability": round(
            sum(r["metrics"]["citation_traceability"] for r in ok) / max(1, len(ok)), 4),
        "context_precision": round(
            sum(r["metrics"]["context_precision"] for r in ok) / max(1, len(ok)), 4),
        "claims": sum(r["metrics"]["claim_count"] for r in ok),
        "supported_claims": sum(r["metrics"]["supported_claims"] for r in ok),
        "refine_rounds": sum(int(r.get("refine_rounds") or 0) for r in ok),
    }
    payload = {"summary": summary, "results": results}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print(f"{'kind':<8}{'label':<32}{'faith':>7}{'trace':>7}{'prec':>7}{'claims':>8}")
    for r in results:
        if "metrics" not in r:
            print(f"{r['kind']:<8}{r['label']:<32}{'ERROR':>7}")
            continue
        m = r["metrics"]
        print(f"{r['kind']:<8}{r['label']:<32}{m['faithfulness']:>7.2f}"
              f"{m['citation_traceability']:>7.2f}{m['context_precision']:>7.2f}"
              f"{m['claim_count']:>8}")
    print("-" * 72)
    for key, value in summary.items():
        print(f"{key:<24}{value}")
    print()
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
