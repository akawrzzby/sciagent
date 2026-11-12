"""Bibliography parsing.

Two things matter for the competition:

1. **Structured entries** – authors / title / year / venue / DOI / arXiv id, so
   the graph builder can align a citation with a paper in the local library.
2. **Bidirectional anchors** – the ``[n]`` markers that appear in the body
   text must be resolvable to the reference entry, and vice versa.  We store
   the marker index and the raw text, and the API exposes a resolver.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.parsing.layout import Block

# --------------------------------------------------------------------------- regexes
_REF_SECTION = re.compile(r"^\s*(references|bibliography|reference)\b", re.IGNORECASE)
# "[1] Author, Title. Venue, 2017."
_BRACKET_ENTRY = re.compile(r"^\s*\[(\d{1,3})\]\s*(.+)$")
# "1. Author, Title..." (some publishers number without brackets)
_DOT_ENTRY = re.compile(r"^\s*(\d{1,3})\.\s+(.+)$")

_ARXIV_IN_REF = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5})(?:v(\d+))?", re.IGNORECASE)
_DOI_IN_REF = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
_YEAR_IN_REF = re.compile(r"\b(19|20)\d{2}\b")
# In-text citation markers: [1], [1, 2], [1-3], [1, 2, 4]
_INTEXT = re.compile(r"\[(\d{1,3}(?:\s*[,\-–]\s*\d{1,3})*)\]")


@dataclass(slots=True)
class Reference:
    """A single bibliography entry."""

    index_in_doc: int
    raw_text: str
    authors: list[str] = field(default_factory=list)
    title: str = ""
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    page_no: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_in_doc": self.index_in_doc,
            "raw_text": self.raw_text,
            "authors": self.authors,
            "title": self.title,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "page_no": self.page_no,
        }


def find_reference_section(blocks: list[Block]) -> tuple[int, int]:
    """Return ``(start_block_index, end_block_index)`` of the bibliography.

    The section starts at the heading that matches ``References`` and ends
    before an ``Appendix`` / ``Acknowledgments`` heading or the end of the
    document.
    """
    start = -1
    for i, b in enumerate(blocks):
        text = b.text.strip()
        if len(text) > 40:
            continue
        if _REF_SECTION.match(text):
            start = i
            break
    if start < 0:
        return -1, -1

    end = len(blocks)
    for j in range(start + 1, len(blocks)):
        text = blocks[j].text.strip()
        if len(text) > 40:
            continue
        if re.match(r"^\s*(appendix|appendix\s+[A-Z]|acknowledg)", text, re.IGNORECASE):
            end = j
            break
    return start, end


def parse_references(blocks: list[Block]) -> list[Reference]:
    """Parse the bibliography into structured entries."""
    start, end = find_reference_section(blocks)
    if start < 0:
        return []

    refs: list[Reference] = []
    current_marker: int | None = None
    buffer: list[str] = []
    current_page = blocks[start].page_no

    def flush() -> None:
        nonlocal current_marker, buffer
        if current_marker is not None and buffer:
            text = " ".join(buffer).strip()
            if len(text) > 20:
                refs.append(_finalise(current_marker, text, current_page))
        buffer = []
        current_marker = None

    for block in blocks[start + 1 : end]:
        text = re.sub(r"\s+", " ", block.text).strip()
        if not text:
            continue

        m = _BRACKET_ENTRY.match(text)
        if m:
            flush()
            current_marker = int(m.group(1))
            buffer = [m.group(2)]
            current_page = block.page_no
            continue

        m = _DOT_ENTRY.match(text)
        if m and len(text) > 12:
            flush()
            current_marker = int(m.group(1))
            buffer = [m.group(2)]
            current_page = block.page_no
            continue

        # Continuation line of the previous entry.
        if current_marker is not None:
            # A new entry that does not use a marker (rare) is separated by a
            # blank line; we approximate with a capitalised start.
            buffer.append(text)
            if len(buffer) > 6:
                flush()
        else:
            current_marker = len(refs) + 1
            buffer = [text]

    flush()

    # Renumber sequentially when the source numbering is inconsistent.
    for i, ref in enumerate(refs, start=1):
        if ref.index_in_doc != i:
            ref.index_in_doc = i
    return refs


def _finalise(marker: int, text: str, page_no: int) -> Reference:
    """Extract the structured fields of one reference."""
    ref = Reference(index_in_doc=marker, raw_text=text, page_no=page_no)

    m = _ARXIV_IN_REF.search(text)
    if m:
        ref.arxiv_id = m.group(1)

    d = _DOI_IN_REF.search(text)
    if d:
        ref.doi = d.group(1).rstrip(".")

    y = _YEAR_IN_REF.search(text)
    if y:
        ref.year = int(y.group(0))

    # Split "Authors. Title. Venue, Year."
    parts = re.split(r"\.\s+", text)
    if parts:
        first = parts[0].strip()
        # Authors segment: everything before the first period that follows a
        # capitalised title-ish chunk.
        if len(first) < 400:
            ref.authors = [a.strip() for a in re.split(r",\s*|\s+and\s+", first) if a.strip()]
        if len(parts) > 1:
            ref.title = parts[1].strip()
        if len(parts) > 2:
            ref.venue = parts[2].strip().rstrip(",")
    if not ref.title:
        ref.title = text[:300]

    # Year may sit at the end of the string.
    if not ref.year:
        tail = re.search(r"\b(19|20)\d{2}\b", text)
        if tail:
            ref.year = int(tail.group(0))
    return ref


def find_intext_citations(text: str) -> list[tuple[int, list[int]]]:
    """Find in-text citation markers.

    Returns a list of ``(character_offset, [reference_numbers])`` tuples.  The
    offsets drive the "click [11] -> jump to the reference" interaction.
    """
    out: list[tuple[int, list[int]]] = []
    for m in _INTEXT.finditer(text):
        spec = m.group(1)
        numbers: list[int] = []
        for part in re.split(r"\s*,\s*", spec):
            rng = re.match(r"^(\d{1,3})\s*[\-–]\s*(\d{1,3})$", part)
            if rng:
                a, b = int(rng.group(1)), int(rng.group(2))
                if a <= b and b - a < 40:
                    numbers.extend(range(a, b + 1))
            elif part.isdigit():
                numbers.append(int(part))
        if numbers:
            out.append((m.start(), numbers))
    return out


def citation_context(text: str, offset: int, window: int = 260) -> str:
    """Return the sentence surrounding an in-text citation.

    Used to judge whether a citation is *substantive* or a mere "perfunctory"
    reference (the "citation gold content" filter of module 4).
    """
    start = max(0, offset - window)
    end = min(len(text), offset + window)
    snippet = text[start:end]
    # Trim to sentence boundaries.
    left = snippet.rfind(". ", 0, offset - start)
    if left > 0:
        snippet = snippet[left + 2 :]
    right = snippet.find(". ", offset - start)
    if right > 0:
        snippet = snippet[: right + 1]
    return re.sub(r"\s+", " ", snippet).strip()
