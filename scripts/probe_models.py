"""Smoke-test the local retrieval models.

Checks that the dense encoder and the cross-encoder can be loaded and that
they behave sensibly (related texts are closer than unrelated ones).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(os.environ.get("SCIAGENT_MODEL_DIR", Path.home() / "models"))
DENSE = MODEL_ROOT / "bge-m3"
RERANK = MODEL_ROOT / "bge-reranker-v2-m3"


def probe_dense() -> bool:
    if not DENSE.is_dir():
        print(f"[skip] {DENSE} not found")
        return True
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(str(DENSE), device="cpu")
    print(f"[ok]   dense loaded in {time.time() - t0:.1f}s "
          f"(dim={model.get_sentence_embedding_dimension()})")

    sentences = [
        "multi head attention in the transformer encoder",
        "the transformer uses multi-head attention",
        "photosynthesis takes place in the chloroplasts of plants",
    ]
    t0 = time.time()
    vecs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
    dt = time.time() - t0
    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])
    print(f"[ok]   encoded 3 sentences in {dt:.2f}s "
          f"({3 / max(dt, 1e-6):.1f} sentences/s)")
    print(f"       sim(related)   = {sim_related:.4f}")
    print(f"       sim(unrelated) = {sim_unrelated:.4f}")
    if sim_related <= sim_unrelated:
        print("[FAIL] related texts are not closer than unrelated ones")
        return False
    print("[ok]   dense channel behaves correctly")
    return True


def probe_reranker() -> bool:
    if not RERANK.is_dir():
        print(f"[skip] {RERANK} not found")
        return True
    from sentence_transformers import CrossEncoder

    t0 = time.time()
    model = CrossEncoder(str(RERANK), device="cpu")
    print(f"[ok]   reranker loaded in {time.time() - t0:.1f}s")

    query = "how many GPUs were used for training the base model?"
    passages = [
        "We trained our models on 8 NVIDIA P100 GPUs for 12.5 days.",
        "We trained our models on 8 P100 GPUs for 12.5 days.",
        "The dataset contains 3 million English-German sentence pairs.",
    ]
    t0 = time.time()
    scores = model.predict([(query, p) for p in passages], show_progress_bar=False)
    print(f"[ok]   reranked 3 passages in {time.time() - t0:.2f}s")
    for p, s in zip(passages, scores):
        print(f"       {float(s):+.4f}  {p[:60]}")
    if float(scores[0]) <= float(scores[2]):
        print("[FAIL] reranker did not prefer the on-topic passage")
        return False
    print("[ok]   reranker channel behaves correctly")
    return True


def main() -> int:
    ok = True
    ok = probe_dense() and ok
    ok = probe_reranker() and ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
