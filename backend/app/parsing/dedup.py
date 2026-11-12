"""Content based duplicate & version detection.

The organisers' test case uploads ``1706.03762v7.pdf`` and then
``1706.03762v1.pdf`` and expects the agent to notice that both are the *same
paper* (Attention Is All You Need) in different revisions.

We combine four independent signals so that the decision does not rely on the
file name:

===========================  ==================================================
signal                       purpose
===========================  ==================================================
``file_hash``                byte identical upload
``content_hash``             identical normalised text, different container
arXiv id + version           explicit version of the same preprint
semantic similarity          cosine over paragraph embeddings + SimHash
===========================  ==================================================
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


# --------------------------------------------------------------------------- hashes
def file_hash(path: str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of the raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def normalise_text(text: str) -> str:
    """Collapse whitespace and lowercase so that trivial formatting differences
    do not change the hash."""
    t = re.sub(r"\s+", " ", text or "").strip().lower()
    return t


def content_hash(text: str) -> str:
    """SHA-256 of the normalised text."""
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- simhash
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalise_text(text))


def simhash(text: str, bits: int = 64) -> int:
    """Classic SimHash over word 3-grams (fast near-duplicate signal)."""
    tokens = _tokens(text)
    if len(tokens) < 3:
        tokens = tokens or ["empty"]
    shingles = [" ".join(tokens[i : i + 3]) for i in range(max(1, len(tokens) - 2))]
    vector = np.zeros(bits, dtype=np.float64)
    for shingle in shingles:
        h = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(bits):
            vector[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if vector[i] > 0:
            out |= 1 << i
    return out


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two SimHashes."""
    return bin(a ^ b).count("1")


def simhash_similarity(a: int, b: int, bits: int = 64) -> float:
    """1.0 == identical, 0.0 == completely different."""
    return 1.0 - hamming_distance(a, b) / bits


# --------------------------------------------------------------------------- records
@dataclass(slots=True)
class PaperFingerprint:
    """The fingerprint of one parsed paper, used for duplicate detection."""

    paper_id: str
    title: str = ""
    arxiv_id: str = ""
    version_label: str = ""
    file_hash: str = ""
    content_hash: str = ""
    simhash: int = 0
    char_count: int = 0
    page_count: int = 0
    abstract: str = ""
    embedding: np.ndarray | None = None
    year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "arxiv_id": self.arxiv_id,
            "version_label": self.version_label,
            "file_hash": self.file_hash,
            "content_hash": self.content_hash,
            "simhash": hex(self.simhash),
            "char_count": self.char_count,
            "page_count": self.page_count,
            "year": self.year,
        }


@dataclass(slots=True)
class DuplicateCandidate:
    """A suspected duplicate together with the evidence that supports it."""

    fingerprint: PaperFingerprint
    score: float
    reasons: list[str] = field(default_factory=list)
    same_paper: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.fingerprint.paper_id,
            "title": self.fingerprint.title,
            "arxiv_id": self.fingerprint.arxiv_id,
            "version_label": self.fingerprint.version_label,
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "same_paper": self.same_paper,
        }


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare_fingerprints(
    incoming: PaperFingerprint,
    existing: Iterable[PaperFingerprint],
    *,
    semantic_threshold: float = 0.93,
    simhash_threshold: float = 0.90,
) -> list[DuplicateCandidate]:
    """Score the incoming paper against every paper already in the library."""
    out: list[DuplicateCandidate] = []

    for cand in existing:
        reasons: list[str] = []
        score = 0.0
        same_paper = False

        if incoming.file_hash and incoming.file_hash == cand.file_hash:
            score = 1.0
            reasons.append("文件字节完全一致 (SHA-256 相同)")
            same_paper = True
        elif incoming.content_hash and incoming.content_hash == cand.content_hash:
            score = 0.99
            reasons.append("正文内容完全一致 (规范化文本哈希相同)")
            same_paper = True
        else:
            if incoming.arxiv_id and incoming.arxiv_id == cand.arxiv_id:
                same_paper = True
                reasons.append(
                    f"arXiv 编号相同 ({incoming.arxiv_id})，版本 "
                    f"{cand.version_label or '?'} → {incoming.version_label or '?'}"
                )
                score = max(score, 0.97)

            sim = simhash_similarity(incoming.simhash, cand.simhash)
            if sim >= simhash_threshold:
                reasons.append(f"SimHash 近似度 {sim:.3f} (≥ {simhash_threshold})")
                score = max(score, sim)

            sem = _cosine(incoming.embedding, cand.embedding)
            if sem >= semantic_threshold:
                reasons.append(f"语义相似度 {sem:.4f} (≥ {semantic_threshold})")
                score = max(score, sem)
                same_paper = same_paper or sem >= 0.985

            if incoming.title and cand.title and _title_similarity(incoming.title,
                                                                   cand.title) >= 0.9:
                reasons.append("标题高度相似")
                score = max(score, 0.9)

        if reasons:
            out.append(DuplicateCandidate(fingerprint=cand, score=score,
                                          reasons=reasons, same_paper=same_paper))

    out.sort(key=lambda c: -c.score)
    return out


def _title_similarity(a: str, b: str) -> float:
    """Token Jaccard similarity between two titles."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def version_rank(version_label: str) -> int:
    """Numeric rank of a version label (``v7`` -> 7, ``''`` -> 0)."""
    m = re.search(r"v(\d+)", version_label or "")
    return int(m.group(1)) if m else 0


def recommend_version(
    incoming: PaperFingerprint,
    existing: DuplicateCandidate,
) -> dict[str, Any]:
    """Decide which version to keep.

    The agent must "suggest keeping the newer version" – we compare the arXiv
    version numbers and, as a fallback, the document length (newer revisions
    are usually longer because they add appendices).
    """
    in_rank = version_rank(incoming.version_label)
    ex_rank = version_rank(existing.fingerprint.version_label)

    if in_rank and ex_rank:
        keep_incoming = in_rank > ex_rank
        reason = (f"上传版本 {incoming.version_label or '无版本号'} "
                  f"{'>' if keep_incoming else '<'} 库中版本 "
                  f"{existing.fingerprint.version_label or '无版本号'}")
    elif in_rank:
        keep_incoming = True
        reason = f"上传文件带版本号 {incoming.version_label}，库中版本无版本号信息"
    else:
        keep_incoming = False
        reason = "库中版本带有明确的 arXiv 版本号，判定为更新的修订版"

    # Length is only a tie breaker.
    if incoming.char_count and existing.fingerprint.char_count:
        delta = incoming.char_count - existing.fingerprint.char_count
        if abs(delta) > max(2000, 0.08 * max(incoming.char_count,
                                              existing.fingerprint.char_count)):
            keep_incoming = delta > 0
            reason += f"；正文长度差异 {delta:+d} 字符"

    return {
        "recommendation": "keep_incoming" if keep_incoming else "keep_existing",
        "keep": "uploaded" if keep_incoming else "library",
        "reason": reason,
        "uploaded_version": incoming.version_label,
        "library_version": existing.fingerprint.version_label,
        "similarity": round(existing.score, 4),
        "reasons": existing.reasons,
    }


def diff_versions(old_text: str, new_text: str, *, context: int = 1) -> list[dict[str, Any]]:
    """Line level diff between two revisions of the same paper.

    Returns a list of ``{"op": "equal"|"insert"|"delete", "old": str,
    "new": str}`` entries suitable for rendering in the UI.
    """
    import difflib

    old_lines = [l.strip() for l in old_text.splitlines() if l.strip()]
    new_lines = [l.strip() for l in new_text.splitlines() if l.strip()]
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)

    out: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.append({"op": "equal", "old": "\n".join(old_lines[i1:i2]),
                        "new": "\n".join(new_lines[j1:j2])})
        elif tag == "delete":
            out.append({"op": "delete", "old": "\n".join(old_lines[i1:i2]), "new": ""})
        elif tag == "insert":
            out.append({"op": "insert", "old": "", "new": "\n".join(new_lines[j1:j2])})
        else:
            out.append({"op": "replace", "old": "\n".join(old_lines[i1:i2]),
                        "new": "\n".join(new_lines[j1:j2])})
    return out
