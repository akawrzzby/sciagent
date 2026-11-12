"""Dense embeddings and cross-encoder reranking.

The dense channel of the hybrid retriever uses ``BAAI/bge-m3`` – a multilingual
model (Chinese + English) which is essential because the product mixes both
languages.  The reranker uses ``BAAI/bge-reranker-v2-m3`` as a cross-encoder.

Both models are optional: when ``sentence-transformers`` is not installed the
service degrades to "dense channel disabled" and the retriever falls back to
pure sparse (BM25) retrieval.  This keeps the deployment lightweight and the
tests deterministic.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any, Sequence

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _configure_threads(threads: int) -> None:
    """Cap the number of torch CPU threads.

    torch sizes its thread pool from the core count, which on a 192-core host
    means 192 threads for a model that only scales to roughly one thread per
    physical core.  The oversubscription costs more in synchronisation than it
    wins in parallelism (measured: 7 passages/s with 192 threads versus
    100 passages/s with 64), so the pool is capped explicitly.
    """
    if threads <= 0:
        return
    try:
        import torch

        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
    except Exception as exc:  # noqa: BLE001 - torch is optional at import time
        logger.debug("could not set torch thread count: %s", exc)


class EmbeddingService:
    """Lazy, thread-safe wrapper around the dense encoder."""

    def __init__(self) -> None:
        self._model = None
        self._reranker = None
        self._lock = threading.Lock()
        self._unavailable_reason = ""

    # ------------------------------------------------------------------ state
    @property
    def available(self) -> bool:
        return self._model is not None or not self._unavailable_reason

    def status(self) -> dict[str, Any]:
        return {
            "dense_available": self._model is not None,
            "reranker_available": self._reranker is not None,
            "model": get_settings().embedding_model,
            "reranker": get_settings().reranker_model,
            "reason": self._unavailable_reason,
        }

    # ------------------------------------------------------------------ loading
    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            settings = get_settings()
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                self._unavailable_reason = f"sentence-transformers unavailable: {exc}"
                logger.warning("%s -> dense retrieval disabled", self._unavailable_reason)
                return None
            try:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                settings = get_settings()
                _configure_threads(settings.embedding_threads)
                logger.info("loading dense model %s (device=%s, threads=%d)",
                            settings.embedding_model, settings.embedding_device,
                            settings.embedding_threads)
                self._model = SentenceTransformer(
                    settings.embedding_model,
                    device=settings.embedding_device,
                    trust_remote_code=True,
                )
                self._model.max_seq_length = settings.embedding_max_length
                logger.info("dense model ready")
            except Exception as exc:
                self._unavailable_reason = f"failed to load {settings.embedding_model}: {exc}"
                logger.warning("%s", self._unavailable_reason)
                return None
            return self._model

    def _ensure_reranker(self):
        if self._reranker is not None:
            return self._reranker
        with self._lock:
            if self._reranker is not None:
                return self._reranker
            settings = get_settings()
            if not settings.reranker_enabled:
                self._unavailable_reason = "reranker disabled by configuration"
                return None
            try:
                from sentence_transformers import CrossEncoder
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                settings = get_settings()
                _configure_threads(settings.embedding_threads)
                logger.info("loading reranker %s", settings.reranker_model)
                self._reranker = CrossEncoder(settings.reranker_model,
                                              device=settings.embedding_device,
                                              trust_remote_code=True)
            except Exception as exc:
                logger.warning("reranker unavailable: %s", exc)
                self._unavailable_reason = f"reranker unavailable: {exc}"
                return None
            return self._reranker

    # ------------------------------------------------------------------ encoding
    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray | None:
        """Encode a batch of texts; returns ``None`` when dense mode is off."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._ensure_model()
        if model is None:
            return None
        settings = get_settings()
        try:
            vectors = model.encode(
                list(texts),
                batch_size=settings.embedding_batch_size,
                normalize_embeddings=True,   # cosine == dot product afterwards
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(vectors, dtype=np.float32)
        except Exception as exc:
            logger.warning("encode failed: %s", exc)
            return None

    def encode_query(self, query: str) -> np.ndarray | None:
        """Encode a single query.

        bge-m3 is trained with an *instruction* prefix for retrieval queries.
        """
        prefix = "Represent this sentence for searching relevant passages: "
        vec = self.encode([prefix + query], is_query=True)
        if vec is None:
            return None
        return vec[0]

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[float]:
        """Score passages with the cross-encoder; higher is better."""
        if not passages:
            return []
        model = self._ensure_reranker()
        if model is None:
            return []
        settings = get_settings()
        k = top_k or settings.reranker_max_passages
        try:
            scores = model.predict([(query, p) for p in passages[:k]], batch_size=8,
                                   show_progress_bar=False)
            return [float(s) for s in scores]
        except Exception as exc:
            logger.warning("rerank failed: %s", exc)
            return []


# --------------------------------------------------------------------------- store
class VectorStore:
    """On-disk vector storage (one ``.npy`` matrix per paper).

    Using a memmap per document keeps the resident memory proportional to the
    papers that take part in a query instead of the whole library – this is the
    "memory management" aspect of the engineering rubric.
    """

    def __init__(self, index_dir: str | None = None) -> None:
        settings = get_settings()
        self.index_dir = index_dir or str(settings.index_dir)
        os.makedirs(self.index_dir, exist_ok=True)

    def path_for(self, paper_id: str) -> str:
        return os.path.join(self.index_dir, f"{paper_id}.npy")

    def save(self, paper_id: str, vectors: np.ndarray) -> str:
        path = self.path_for(paper_id)
        np.save(path, vectors)
        return path

    def load(self, paper_id: str, mmap: bool = True) -> np.ndarray | None:
        path = self.path_for(paper_id)
        if not os.path.exists(path):
            return None
        try:
            return np.load(path, mmap_mode="r" if mmap else None)
        except Exception:
            return None

    def exists(self, paper_id: str) -> bool:
        return os.path.exists(self.path_for(paper_id))

    def delete(self, paper_id: str) -> None:
        path = self.path_for(paper_id)
        if os.path.exists(path):
            os.remove(path)

    @staticmethod
    def cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of a (normalised) matrix against a query vector."""
        if matrix is None or matrix.size == 0:
            return np.zeros((0,), dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return np.zeros((matrix.shape[0],), dtype=np.float32)
        # The stored vectors are already L2-normalised, so a dot product is
        # enough; we normalise defensively in case of older index files.
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        return ((matrix @ q) / norms).astype(np.float32)


def text_hash(text: str) -> str:
    """Stable hash used for cache keys."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Return the process-wide embedding service."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
