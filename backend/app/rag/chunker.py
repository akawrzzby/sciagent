"""Chunking: turning a parsed document into retrievable, traceable passages.

The chunker is *hierarchical*: section → paragraph → sentence, packed into
passages of roughly ``TARGET_TOKENS`` with a small overlap.  Every chunk keeps
the page number and the character offsets into the document's plain text, which
is what powers the "click the citation → jump to the exact place in the PDF"
interaction (source tracing).

Two special chunk kinds are produced in addition to the body text:

``future_work``  the "Future Work" / "Limitations" paragraph of each paper –
                 the raw material for the horizon-scanning feature.
``abstract``     the abstract, which is retrieved preferentially for
                 high-level questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Average characters per token for English scientific text.
CHARS_PER_TOKEN = 4.0
TARGET_TOKENS = 380
OVERLAP_TOKENS = 60
MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = int(TARGET_TOKENS * CHARS_PER_TOKEN)

# Sentence boundary: terminator followed by whitespace and a capital/quote.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[ \t]+(?=[\"'(\[]?[A-Z0-9])")

# Section titles whose content is mined for research opportunities.
FUTURE_WORK_TITLES = (
    "future work", "future directions", "future research", "outlook",
    "limitations", "limitation", "conclusion and future work",
    "conclusions and future work", "discussion and future work",
    "open problems", "open questions",
)


@dataclass(slots=True)
class Chunk:
    """A retrievable passage."""

    ordinal: int
    content: str
    page_no: int
    section_id: str | None = None
    section_title: str = ""
    char_start: int = 0
    char_end: int = 0
    kind: str = "body"

    @property
    def token_estimate(self) -> int:
        return int(len(self.content) / CHARS_PER_TOKEN)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "content": self.content,
            "page_no": self.page_no,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "kind": self.kind,
        }


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences (regex based, language agnostic enough)."""
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or [text.strip()]


def _is_future_work(title: str) -> bool:
    t = (title or "").lower().strip()
    return any(k in t for k in FUTURE_WORK_TITLES)


def chunk_document(
    *,
    full_text: str,
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
    abstract: str = "",
) -> list[Chunk]:
    """Split a parsed document into retrieval chunks.

    ``sections`` come from the heading tree, ``blocks`` are the positioned text
    blocks (used to resolve the page number of a chunk).
    """
    chunks: list[Chunk] = []
    ordinal = 0

    # --- page lookup ----------------------------------------------------
    block_pages: list[tuple[int, int, str]] = []  # (char_start, page_no, text)
    cursor = 0
    for b in blocks:
        text = b.get("content", "")
        if not text:
            continue
        block_pages.append((cursor, b.get("page_no", 0), text))
        cursor += len(text) + 1  # +1 for the joining newline

    def page_for(char_pos: int) -> int:
        page = 0
        for start, pno, _ in block_pages:
            if start <= char_pos:
                page = pno
            else:
                break
        return page

    # --- abstract -------------------------------------------------------
    if abstract and len(abstract) >= MIN_CHUNK_CHARS:
        chunks.append(Chunk(ordinal=ordinal, content=abstract.strip(), page_no=0,
                             section_title="Abstract", kind="abstract"))
        ordinal += 1

    # --- body sections --------------------------------------------------
    for sec in sections:
        title = sec.get("title", "") or ""
        number = sec.get("number", "") or ""
        content = (sec.get("content") or "").strip()
        if not content:
            continue

        heading_label = f"{number} {title}".strip()
        kind = "future_work" if _is_future_work(title) else "body"

        # Locate the section inside the full text so that we can record exact
        # character offsets (needed for the source anchors).
        section_start = _find_offset(full_text, content, sec)

        for piece in _pack(content):
            if len(piece) < MIN_CHUNK_CHARS and kind == "body":
                continue
            start = section_start + content.find(piece) if section_start >= 0 else 0
            chunks.append(Chunk(
                ordinal=ordinal,
                content=piece,
                page_no=sec.get("page_no", 0) or page_for(start),
                section_title=heading_label,
                char_start=start,
                char_end=start + len(piece),
                kind=kind,
            ))
            ordinal += 1

    # --- fallback: no sections at all -----------------------------------
    if not chunks and full_text:
        for piece in _pack(full_text):
            chunks.append(Chunk(ordinal=ordinal, content=piece, page_no=0,
                                 section_title="", kind="body"))
            ordinal += 1

    return chunks


def _find_offset(haystack: str, needle: str, section: dict[str, Any]) -> int:
    """Best-effort location of a section inside the document text."""
    if not needle:
        return -1
    # Use the first 60 characters as the probe (sections are often preceded by
    # the heading text which is not part of ``content``).
    probe = needle[:60]
    idx = haystack.find(probe)
    if idx >= 0:
        return idx
    probe = needle[:30]
    return haystack.find(probe)


def _pack(text: str) -> list[str]:
    """Pack sentences into chunks of at most ``MAX_CHUNK_CHARS`` characters."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    out: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 > MAX_CHUNK_CHARS and current:
            out.append(" ".join(current))
            # Overlap: keep the tail of the previous chunk.
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > OVERLAP_TOKENS * CHARS_PER_TOKEN:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current = list(tail)
            current_len = tail_len
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        out.append(" ".join(current))
    return [c for c in out if c.strip()]


def chunk_stats(chunks: Iterable[Chunk]) -> dict[str, Any]:
    """Summary statistics used by the test report and the UI."""
    chunks = list(chunks)
    if not chunks:
        return {"count": 0, "total_chars": 0, "avg_chars": 0, "avg_tokens": 0}
    total = sum(len(c.content) for c in chunks)
    return {
        "count": len(chunks),
        "total_chars": total,
        "avg_chars": total // len(chunks),
        "avg_tokens": sum(c.token_estimate for c in chunks) // len(chunks),
        "kinds": {kind: sum(1 for c in chunks if c.kind == kind)
                  for kind in {c.kind for c in chunks}},
    }
