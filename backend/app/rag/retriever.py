"""Hybrid retrieval: BM25 + dense vectors → RRF fusion → cross-encoder rerank.

The retriever is the heart of the anti-hallucination story: an answer may only
cite a passage that came out of this pipeline, and every returned passage
carries the provenance needed to jump back into the PDF.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag import meta_aware, sparse
from app.rag.embeddings import VectorStore, get_embedding_service

logger = get_logger(__name__)


@dataclass(slots=True)
class Passage:
    """A retrieved passage with full provenance."""

    chunk_id: str
    paper_id: str
    ordinal: int
    content: str
    page_no: int
    section_title: str = ""
    char_start: int = 0
    char_end: int = 0
    kind: str = "body"
    score: float = 0.0
    sparse_rank: int = 0
    dense_rank: int = 0
    rerank_score: float = 0.0
    paper_title: str = ""
    metadata_affinity: float = 0.0
    metadata_match: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "ordinal": self.ordinal,
            "content": self.content,
            "page_no": self.page_no,
            "section_title": self.section_title,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "kind": self.kind,
            "score": round(self.score, 6),
            "sparse_rank": self.sparse_rank,
            "dense_rank": self.dense_rank,
            "rerank_score": round(self.rerank_score, 6),
            "metadata_affinity": round(self.metadata_affinity, 4),
            "metadata_match": self.metadata_match,
        }

    def citation(self, index: int) -> dict[str, Any]:
        """Compact citation payload used by the API / UI."""
        return {
            "index": index,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "page_no": self.page_no,
            "section_title": self.section_title,
            "chunk_id": self.chunk_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "quote": self.content[:320],
            "score": round(self.score, 6),
            "metadata_affinity": round(self.metadata_affinity, 4),
        }


@dataclass
class RetrievalResult:
    """The outcome of one retrieval round."""

    passages: list[Passage] = field(default_factory=list)
    query: str = ""
    strategy: str = "hybrid"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "passages": [p.to_dict() for p in self.passages],
            "diagnostics": self.diagnostics,
        }


class HybridRetriever:
    """Combines the sparse and dense channels over the chunk store."""

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.embeddings = get_embedding_service()
        self.store = VectorStore()
        self._bm25: sparse.BM25Index | None = None
        self._bm25_signature: tuple[int, float] | None = None

    # ------------------------------------------------------------------ index
    async def rebuild_index(self, chunks: Sequence[dict[str, Any]]) -> None:
        """(Re)build the sparse index from the database chunks."""
        docs = {c["id"]: c["content"] for c in chunks}
        index = sparse.BM25Index(documents=docs)
        self._bm25 = index
        self._bm25_signature = (len(docs), float(sum(len(v) for v in docs.values())))

    def _ensure_index(self, chunks: Sequence[dict[str, Any]]) -> sparse.BM25Index:
        signature = (len(chunks), float(sum(len(c["content"]) for c in chunks)))
        if self._bm25 is None or self._bm25_signature != signature:
            self._bm25 = sparse.BM25Index(documents={c["id"]: c["content"] for c in chunks})
            self._bm25_signature = signature
        return self._bm25

    # ------------------------------------------------------------------ search
    async def search(
        self,
        query: str,
        chunks: Sequence[dict[str, Any]],
        *,
        top_k: int | None = None,
        paper_ids: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
        rerank: bool = True,
    ) -> RetrievalResult:
        """Run the hybrid retrieval pipeline.

        ``chunks`` is the candidate pool (already filtered by the caller when a
        specific subset of the library should be searched).
        """
        settings = self.settings
        top_k = top_k or settings.retrieve_top_k_final
        diagnostics: dict[str, Any] = {"candidates": len(chunks), "channels": []}

        if not chunks:
            return RetrievalResult(query=query, strategy="empty", diagnostics=diagnostics)

        by_id = {c["id"]: c for c in chunks}
        index = self._ensure_index(chunks)

        # ---- channel 1: sparse ------------------------------------------
        sparse_ranking = index.search(query, top_k=settings.retrieve_top_k_sparse)
        diagnostics["channels"].append({"name": "bm25", "hits": len(sparse_ranking)})

        # ---- channel 2: dense --------------------------------------------
        dense_ranking: list[tuple[str, float]] = []
        query_vec = await asyncio.to_thread(self.embeddings.encode_query, query)
        if query_vec is not None:
            # Group the chunks per paper so that only the needed memmaps are
            # touched (memory friendly).
            by_paper: dict[str, list[dict[str, Any]]] = {}
            for c in chunks:
                by_paper.setdefault(c["paper_id"], []).append(c)

            for paper_id, paper_chunks in by_paper.items():
                matrix = self.store.load(paper_id)
                if matrix is None or matrix.size == 0:
                    continue
                rows = [c.get("embedding_row", -1) for c in paper_chunks]
                valid = [(c, r) for c, r in zip(paper_chunks, rows) if 0 <= r < matrix.shape[0]]
                if not valid:
                    continue
                sub = matrix[[r for _, r in valid]]
                sims = VectorStore.cosine(np.asarray(sub), query_vec)
                for (chunk, _row), sim in zip(valid, sims):
                    dense_ranking.append((chunk["id"], float(sim)))
            dense_ranking.sort(key=lambda kv: -kv[1])
            dense_ranking = dense_ranking[: settings.retrieve_top_k_dense]
            diagnostics["channels"].append({"name": "dense", "hits": len(dense_ranking)})
        else:
            diagnostics["channels"].append({"name": "dense", "hits": 0,
                                            "note": "dense model unavailable"})

        # ---- fusion -------------------------------------------------------
        strategy = "hybrid" if dense_ranking else "sparse"
        fused = sparse.reciprocal_rank_fusion([sparse_ranking, dense_ranking],
                                              weights=[1.0, 1.2])
        diagnostics["fused"] = len(fused)

        sparse_rank = {key: i + 1 for i, (key, _) in enumerate(sparse_ranking)}
        dense_rank = {key: i + 1 for i, (key, _) in enumerate(dense_ranking)}

        candidates = [by_id[key] for key, _ in fused if key in by_id][:40]
        if not candidates:
            return RetrievalResult(query=query, strategy=strategy, diagnostics=diagnostics)

        # ---- reranking ------------------------------------------------------
        passages: list[Passage] = []
        rerank_scores: list[float] = []
        if rerank and dense_ranking:
            # The cross-encoder sees the same metadata conditioned text as the
            # dense channel.  Without the prefix the last ranking step would
            # throw the metadata away again: a query that names a paper would
            # still be answered by whichever paper happens to share the
            # wording.
            if getattr(settings, "metadata_aware_rerank", True):
                rerank_docs = [meta_aware.embedding_text(c) for c in candidates]
            else:
                rerank_docs = [c["content"] for c in candidates]
            rerank_scores = await asyncio.to_thread(
                self.embeddings.rerank, query, rerank_docs,
            )
        if rerank_scores:
            for chunk, rs in zip(candidates, rerank_scores):
                passages.append(self._to_passage(chunk, score=rs, rerank_score=rs))
            passages.sort(key=lambda p: -p.rerank_score)
            diagnostics["reranked"] = len(passages)
        else:
            fused_scores = dict(fused)
            for chunk in candidates:
                passages.append(self._to_passage(chunk, score=fused_scores.get(chunk["id"], 0.0)))

        # ---- filtering + top-k ---------------------------------------------
        if kinds:
            wanted = set(kinds)
            passages = [p for p in passages if p.kind in wanted]

        # ---- metadata aware boost ------------------------------------------
        # A query that names the paper ("what did Vaswani report in 2017")
        # lifts that paper's passages; a query that names nothing lifts
        # nothing.  The affinity is reported so that the UI can explain the
        # retrieval.
        affinity = self._metadata_affinity(query, chunks)
        boost = float(getattr(settings, "metadata_boost", 0.0) or 0.0)
        if affinity and boost > 0:
            for p in passages:
                info = affinity.get(p.paper_id) or {}
                p.metadata_affinity = float(info.get("affinity", 0.0))
                p.metadata_match = info
                p.score = p.score * (1.0 + boost * p.metadata_affinity)
            passages.sort(key=lambda p: -p.score)
            diagnostics["metadata_boost"] = {
                "weight": boost,
                "matched_papers": [pid for pid, info in affinity.items()
                                   if info.get("affinity", 0) > 0],
            }

        passages = passages[:top_k]
        for p in passages:
            p.sparse_rank = sparse_rank.get(p.chunk_id, 0)
            p.dense_rank = dense_rank.get(p.chunk_id, 0)

        diagnostics["returned"] = len(passages)
        return RetrievalResult(query=query, strategy=strategy, passages=passages,
                               diagnostics=diagnostics)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _metadata_affinity(query: str,
                            chunks: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """How well the metadata of every paper in the pool matches the query."""
        out: dict[str, dict[str, Any]] = {}
        for c in chunks:
            pid = c.get("paper_id", "")
            if not pid or pid in out:
                continue
            out[pid] = meta_aware.explain_affinity(
                query,
                c.get("paper_title", "") or "",
                c.get("paper_authors"),
                c.get("paper_year"),
            )
        return out

    @staticmethod
    def _to_passage(chunk: dict[str, Any], *, score: float, rerank_score: float = 0.0) -> Passage:
        return Passage(
            chunk_id=chunk["id"],
            paper_id=chunk.get("paper_id", ""),
            ordinal=chunk.get("ordinal", 0),
            content=chunk.get("content", ""),
            page_no=chunk.get("page_no", 0),
            section_title=chunk.get("section_title", "") or "",
            char_start=chunk.get("char_start", 0),
            char_end=chunk.get("char_end", 0),
            kind=chunk.get("kind", "body"),
            score=score,
            rerank_score=rerank_score,
            paper_title=chunk.get("paper_title", "") or "",
        )


async def embed_chunks(
    chunks: Sequence[dict[str, Any]],
    *,
    service=None,
    batch_size: int = 8,
) -> np.ndarray | None:
    """Embed a list of chunks (``{"content": str}``) into a dense matrix.

    The embedded text carries the paper's title, authors and year in front of
    the content, so that the vector is aware of the metadata the chunk came
    from (see :mod:`app.rag.meta_aware`).
    """
    service = service or get_embedding_service()
    if getattr(get_settings(), "metadata_aware_embedding", True):
        texts = [meta_aware.embedding_text(c) for c in chunks]
    else:
        texts = [c.get("content", "") for c in chunks]
    if not texts:
        return None
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vecs = await asyncio.to_thread(service.encode, batch)
        if vecs is None:
            return None
        vectors.append(np.asarray(vecs, dtype=np.float32))
    if not vectors:
        return None
    return np.vstack(vectors)
