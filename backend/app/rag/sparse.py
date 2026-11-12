"""Sparse retrieval (BM25) over the local chunk store.

We implement BM25 directly instead of relying on SQLite FTS5 because we need
the *raw term statistics* to fuse with the dense channel via Reciprocal Rank
Fusion, and because the implementation must work identically inside and outside
the Docker container (no extension loading issues).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

# Token pattern: latin words, numbers and CJK single characters.
_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]")

# Stop words that add noise to BM25 scoring.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "that", "this", "these", "those", "we", "our", "it", "its", "can", "will",
    "would", "which", "than", "then", "there", "their", "they", "has", "have",
    "had", "not", "no", "also", "such", "using", "use", "used", "via", "et",
    "al", "fig", "figure", "table", "eq", "equation", "section",
}


def tokenize(text: str) -> list[str]:
    """Tokenise text for BM25 (latin words + individual CJK characters)."""
    return [t.lower() for t in _TOKEN.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    """Tokenise and drop stop words."""
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1]


@dataclass(slots=True)
class BM25Index:
    """A BM25 index over a set of documents.

    ``documents`` maps a document key to its raw text.  The index is rebuilt
    lazily and cached; adding a document invalidates the cache.
    """

    documents: dict[str, str]
    k1: float = 1.5
    b: float = 0.75

    _tokens: dict[str, list[str]] | None = None
    _df: Counter | None = None
    _avgdl: float = 0.0

    # ------------------------------------------------------------------ build
    def invalidate(self) -> None:
        self._tokens = None
        self._df = None

    def add(self, key: str, text: str) -> None:
        self.documents[key] = text
        self.invalidate()

    def _build(self) -> None:
        if self._tokens is not None:
            return
        tokens: dict[str, list[str]] = {}
        df: Counter = Counter()
        total_len = 0
        for key, text in self.documents.items():
            toks = content_tokens(text)
            tokens[key] = toks
            total_len += len(toks)
            df.update(set(toks))
        self._tokens = tokens
        self._df = df
        self._avgdl = (total_len / len(tokens)) if tokens else 0.0

    @property
    def size(self) -> int:
        return len(self.documents)

    # ------------------------------------------------------------------ query
    def search(self, query: str, top_k: int = 40) -> list[tuple[str, float]]:
        """Return ``[(document_key, score)]`` sorted by descending score."""
        self._build()
        assert self._tokens is not None and self._df is not None
        if not self._tokens:
            return []

        q_terms = content_tokens(query)
        if not q_terms:
            return []

        n_docs = len(self._tokens)
        idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in self._df.items()
        }

        scores: dict[str, float] = {}
        for key, toks in self._tokens.items():
            if not toks:
                continue
            counts = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                score += idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scores[key] = score

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:top_k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked lists with Reciprocal Rank Fusion.

    RRF is scale free, which makes it the right tool to combine a BM25 score
    (unbounded) with a cosine similarity (0..1) without normalisation hacks.
    """
    if not rankings:
        return []
    weights = list(weights) if weights else [1.0] * len(rankings)

    fused: dict[str, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, (key, _score) in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + w / (k + rank)

    return sorted(fused.items(), key=lambda kv: -kv[1])


def bm25_idf_weights(index: BM25Index, query: str) -> dict[str, float]:
    """Return the IDF of every query term (used to explain a retrieval)."""
    index._build()  # noqa: SLF001 - intentional internal use
    assert index._df is not None
    n_docs = max(1, index.size)
    out: dict[str, float] = {}
    for term in content_tokens(query):
        df = index._df.get(term, 0)  # noqa: SLF001
        out[term] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    return out
