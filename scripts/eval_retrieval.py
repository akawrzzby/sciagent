"""Measure the retrieval quality of the library, with and without the
metadata aware boost.

The question this script answers is a narrow one: *when the query points at a
paper -- by its title, by its authors or by its year -- does the retriever put
that paper's passages in front?*

It talks to ``POST /api/rag/search``, which is the plain retrieval endpoint
(no LLM in the loop), so the numbers measure retrieval and nothing else.

Run on the server::

    ~/envs/sciagent/bin/python ~/sciagent/scripts/eval_retrieval.py \
        --base http://127.0.0.1:9000 --out ~/sciagent/data/eval_retrieval.json

Add ``--label`` to mark the run (e.g. ``before`` / ``after``) so that two
runs can be compared.
"""

from __future__ import annotations

import argparse
import json
import os

import httpx

# (query, expected paper title fragment)
QUERIES: list[tuple[str, str]] = [
    # --- questions that name the paper only through its content -------------
    ("What BLEU score did the model obtain on the WMT 2014 English-to-German "
     "translation task?", "Attention Is All You Need"),
    ("What are the values of the hyper-parameters beta_1 and beta_2 used in "
     "the Adam optimizer?", "Attention Is All You Need"),
    ("Which GPU models were used for training, and how many of them?",
     "Attention Is All You Need"),
    ("How does layer normalization differ from batch normalization?",
     "Layer Normalization"),
    ("What is the memory overhead of the optimizer state, and how is it "
     "partitioned across devices?", "ZeRO"),
    ("What is the maximum sequence length that Llama 2 was trained on?",
     "Llama 2"),
    ("How is the model adapted to a task with a small number of examples?",
     "Language Models are Few-Shot Learners"),
    ("What is the effect of quantizing the frozen weights during "
     "fine-tuning?", "QLORA"),
    ("How is the training data transcribed and segmented?",
     "Robust Speech Recognition"),
    ("What abilities appear only above a certain model scale?",
     "Emergent Abilities"),
    # --- questions that name the paper through its metadata -----------------
    ("What did Vaswani et al. report in 2017?", "Attention Is All You Need"),
    ("Attention Is All You Need: which GPUs were used?", "Attention Is All You Need"),
    ("What did Dettmers and Pagnoni report about quantized fine-tuning?",
     "QLORA"),
    ("What did Raffel and Shazeer report in the T5 paper?",
     "Exploring the Limits of Transfer Learning"),
]

TOP_K = 5


def _papers(base: str) -> list[dict]:
    resp = httpx.get(f"{base}/api/papers", timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    items = payload.get("papers", payload) if isinstance(payload, dict) else payload
    return list(items)


def _search(base: str, query: str, top_k: int) -> list[dict]:
    resp = httpx.post(f"{base}/api/rag/search",
                      json={"query": query, "top_k": top_k, "stream": False},
                      timeout=300)
    resp.raise_for_status()
    payload = resp.json()
    return list(payload.get("passages") or [])


def run(base: str, top_k: int) -> dict:
    papers = _papers(base)
    titles = {p.get("id"): (p.get("title") or "") for p in papers}

    rows: list[dict] = []
    for query, expected in QUERIES:
        passages = _search(base, query, top_k)
        rank = 0
        hit_title = ""
        for i, p in enumerate(passages, start=1):
            title = titles.get(p.get("paper_id"), p.get("paper_title", ""))
            if expected.lower() in (title or "").lower():
                rank = i
                hit_title = title
                break
        affinity = max((float(p.get("metadata_affinity") or 0.0) for p in passages),
                       default=0.0)
        rows.append({
            "query": query,
            "expected": expected,
            "rank": rank,
            "hit": bool(rank),
            "hit_title": hit_title,
            "returned": len(passages),
            "max_metadata_affinity": round(affinity, 4),
            "top_titles": [titles.get(p.get("paper_id"), p.get("paper_title", ""))
                           for p in passages[:3]],
        })

    hits = sum(1 for r in rows if r["hit"])
    mrr = sum(1.0 / r["rank"] for r in rows if r["hit"]) / max(1, len(rows))
    mean_affinity = sum(r["max_metadata_affinity"] for r in rows) / max(1, len(rows))
    return {
        "queries": len(rows),
        f"hit@{top_k}": round(hits / max(1, len(rows)), 4),
        "mrr": round(mrr, 4),
        "mean_metadata_affinity": round(mean_affinity, 4),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:9000")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/sciagent/data/eval_retrieval.json"))
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    result = run(base, args.top_k)
    result["label"] = args.label
    result["top_k"] = args.top_k

    print(f"{'query':<64}{'rank':>6}")
    print("-" * 70)
    for r in result["rows"]:
        mark = "ok" if r["hit"] else "--"
        print(f"{r['query'][:62]:<64}{mark:>3}{r['rank'] or 0:>3}")
    print("-" * 70)
    print(f"{'queries':<24}{result['queries']}")
    print(f"{'hit@' + str(args.top_k):<24}{result[f'hit@{args.top_k}']}")
    print(f"{'mrr':<24}{result['mrr']}")
    print(f"{'mean metadata affinity':<24}{result['mean_metadata_affinity']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
