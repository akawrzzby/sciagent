"""Module 2 — hybrid retrieval: BM25, RRF fusion, chunking, traceability.

These tests cover the *mechanical* parts of the retrieval pipeline.  The dense
channel and the cross-encoder are optional (they need the downloaded models),
so they are exercised only when the weights are present.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.rag.chunker import chunk_document, chunk_stats, split_sentences
from app.rag.sparse import BM25Index, content_tokens, reciprocal_rank_fusion, tokenize


# ------------------------------------------------------------------- BM25
def test_bm25_ranks_the_matching_document_first() -> None:
    index = BM25Index({})
    index.add("transformer", "attention is all you need self attention multi head attention")
    index.add("bert", "bidirectional transformer pretraining masked language model")
    index.add("gpt", "autoregressive language models few shot learners scale")

    hits = index.search("multi head attention", top_k=3)
    assert hits
    assert hits[0][0] == "transformer"


def test_bm25_ignores_stop_words_only_queries() -> None:
    index = BM25Index({"a": "the quick brown fox"})
    assert index.search("the of and") == []


def test_bm25_returns_empty_for_unknown_terms() -> None:
    index = BM25Index({"a": "attention transformer"})
    assert index.search("photosynthesis chlorophyll") == []


def test_tokenizer_drops_short_function_words() -> None:
    """``tokenize`` keeps every word; ``content_tokens`` drops stop words."""
    assert "is" in tokenize("Attention is all you need")
    assert "is" not in content_tokens("Attention is all you need")
    assert "attention" in content_tokens("Attention is all you need")


# -------------------------------------------------------------------- RRF
def test_rrf_promotes_documents_present_in_both_rankings() -> None:
    sparse = [("a", 9.0), ("b", 8.0), ("c", 7.0)]
    dense = [("c", 0.95), ("a", 0.90), ("b", 0.10)]

    fused = reciprocal_rank_fusion([sparse, dense])
    scores = dict(fused)
    # "a" is 1st in sparse and 2nd in dense -> should outrank "c" (3rd / 1st).
    assert scores["a"] > scores["c"]
    assert fused[0][0] == "a"


def test_rrf_is_scale_free() -> None:
    """A document ranked first in one list beats one ranked first in another
    only marginally - RRF does not let a huge raw score dominate."""
    a = reciprocal_rank_fusion([[("x", 1e9), ("y", 1.0)]])
    b = reciprocal_rank_fusion([[("y", 1e-9), ("x", 1.0)]])
    assert a[0][0] == "x"
    assert b[0][0] == "y"


def test_rrf_accepts_weights() -> None:
    fused = reciprocal_rank_fusion([[("a", 1.0)], [("b", 1.0)]], weights=[1.0, 5.0])
    scores = dict(fused)
    assert scores["b"] > scores["a"]


def test_rrf_with_no_rankings_is_empty() -> None:
    assert reciprocal_rank_fusion([]) == []


# --------------------------------------------------------------- chunking
def test_split_sentences() -> None:
    text = "This is the first sentence. This is the second one! And a third?"
    assert len(split_sentences(text)) == 3


def test_chunk_document_produces_traceable_chunks() -> None:
    body = (
        "We propose a new simple network architecture, the Transformer, based solely on "
        "attention mechanisms, dispensing with recurrence and convolutions entirely. "
    ) * 12
    sections = [
        {"id": "s1", "title": "Introduction", "number": "1", "level": 1,
         "page_no": 1, "content": body},
        {"id": "s2", "title": "Future Work", "number": "6", "level": 1,
         "page_no": 3, "content": body},
    ]
    blocks = [{"content": body, "page_no": 1}, {"content": body, "page_no": 3}]

    chunks = chunk_document(full_text=body, sections=sections, blocks=blocks)

    assert chunks
    # Every chunk must carry a page number and a character range so that the
    # UI can jump back to the exact source location.
    for c in chunks:
        assert c.page_no >= 0
        assert c.char_start >= 0
        assert c.char_end >= c.char_start
        assert c.content.strip()
    # The Future Work section is tagged so that the opportunity miner can find it.
    kinds = {c.section_title for c in chunks}
    assert any(k.endswith("Future Work") for k in kinds)
    assert any(c.kind == "future_work" or "future" in c.section_title.lower()
               for c in chunks)


def test_chunk_stats_sums_tokens() -> None:
    body = "word " * 400
    sections = [{"id": "s1", "title": "S", "number": "1", "level": 1,
                 "page_no": 1, "content": body}]
    chunks = chunk_document(full_text=body, sections=sections, blocks=[{"content": body, "page_no": 1}])
    stats = chunk_stats(chunks)
    assert stats["count"] == len(chunks)
    assert stats["total_chars"] > 0
    assert stats["avg_tokens"] > 0


# ------------------------------------------------- optional dense channel
def _local_model_dir(name: str) -> Path | None:
    candidate = Path(os.environ.get("SCIAGENT_MODEL_DIR", Path.home() / "models")) / name
    return candidate if candidate.is_dir() else None


@pytest.mark.skipif(_local_model_dir("bge-m3") is None,
                    reason="bge-m3 weights are not available locally")
def test_dense_channel_produces_similar_vectors_for_related_text() -> None:
    from app.rag.embeddings import EmbeddingService

    service = EmbeddingService()
    vecs = service.encode(["multi head attention", "multi-headed attention mechanism"])
    assert vecs is not None
    assert vecs.shape[0] == 2
    assert float(vecs[0] @ vecs[1]) > 0.7


@pytest.mark.skipif(_local_model_dir("bge-m3") is None,
                    reason="bge-m3 weights are not available locally")
def test_dense_channel_distinguishes_unrelated_text() -> None:
    from app.rag.embeddings import EmbeddingService

    service = EmbeddingService()
    vecs = service.encode(
        ["multi head attention in transformers", "photosynthesis in chloroplasts of plants"]
    )
    assert vecs is not None
    assert float(vecs[0] @ vecs[1]) < 0.5
