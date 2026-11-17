"""Assemble one paper into a flat, renderable document.

The parser keeps a paper in five separate streams (content blocks, sections,
tables, figures, formulas) and the translator keeps the bilingual pairs as a
separate segment list keyed by content block.  The reading view needs a single
stream: every element of the paper in the order a human reads it, with its
translation already attached to the element it belongs to.

This module joins those streams into a :class:`~app.render.model.Document`.

It knows nothing about HTTP (see ``app/api/routers/document.py``) and nothing
about table layout (see ``app.render.spans``).
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select

from app.core.logging import get_logger
from app.render.model import Block, Cell, Document, Sentence
from app.services import papers as paper_service

# The translator's own splitter is reused instead of a local one: the segments
# were produced with it, so the reader pairs the same sentences the translator
# saw.  A second splitter would drift and break the one-to-one alignment.
from app.services.translation import split_sentences
from app.storage.db import session_scope
from app.storage.models import Translation

logger = get_logger(__name__)

#: Content-block types that carry running prose.  ``header`` and
#: ``figure_label`` are page furniture (a folio, a label inside a picture) and
#: ``formula`` is rendered from the formula row, which carries the LaTeX and
#: the cropped image.
_PARAGRAPH_TYPES = frozenset({"body", "text", "caption"})

#: ``\tag{3}`` -- the explicit equation number of a LaTeX source.
_TAG_NUMBER = re.compile(r"\\tag\*?\{([^}]{1,24})\}")

#: ``(3)`` / ``(A.1)`` -- the number the paper prints at the right margin.
_PAREN_NUMBER = re.compile(r"\(([A-Z]?\.?\d{1,3}(?:\.\d{1,3})?)\)\s*$")


# --------------------------------------------------------------------------- helpers
def _normalise(text: str) -> str:
    """Collapse whitespace, so a caption matches its translated segment."""
    return " ".join(str(text or "").split()).strip()


def _bbox(value: Any) -> list[float]:
    """Coerce a stored bbox into a list of floats (empty when unusable)."""
    try:
        return [float(v) for v in value][:4]
    except (TypeError, ValueError):
        return []


def _page(entry: dict[str, Any]) -> int:
    """The page of a stored section / block / table / figure / formula."""
    try:
        return int(entry.get("page_no") or 0)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------- translations
class _Segments:
    """The translation segments of one paper, indexed for the lookups below.

    An empty index is a valid state: a paper that was never translated still
    gets a document, every ``target`` is simply an empty string.
    """

    def __init__(self, raw: list[dict[str, Any]]) -> None:
        self._by_block: dict[str, list[dict[str, Any]]] = {}
        self._by_text: dict[str, str] = {}
        ordered = sorted(raw, key=lambda seg: seg.get("index") or 0)
        for seg in ordered:
            source = _normalise(seg.get("source") or "")
            target = _normalise(seg.get("translation") or "")
            block_id = str(seg.get("block_id") or "")
            if block_id:
                self._by_block.setdefault(block_id, []).append(seg)
            if source and target:
                # First hit wins: the segments are in document order, so the
                # first translation of a repeated sentence is the right one.
                self._by_text.setdefault(source, target)

    def for_block(self, block_id: str) -> list[dict[str, Any]]:
        """The segments that were cut from one content block, in order."""
        return self._by_block.get(block_id, [])

    def target(self, text: str) -> str:
        """The translation of ``text``, or ``""`` when there is none.

        A heading, a caption or the abstract is often longer or shorter than
        one segment, so a direct hit is tried first and the sentence-by-
        sentence join second.  Anything else stays empty on purpose: a
        half-translated paragraph would claim a translation that does not
        exist.
        """
        wanted = _normalise(text)
        if not wanted:
            return ""
        direct = self._by_text.get(wanted)
        if direct:
            return direct
        pieces = [self._by_text.get(_normalise(s)) for s in split_sentences(wanted)]
        if pieces and all(pieces):
            return " ".join(piece for piece in pieces if piece)
        return ""


def _sentences(block_id: str, text: str, segments: _Segments) -> list[Sentence]:
    """The aligned sentences of one paragraph."""
    stored = segments.for_block(block_id)
    if stored:
        out: list[Sentence] = []
        for i, seg in enumerate(stored):
            index = seg.get("index")
            out.append(Sentence(
                id=str(index) if index is not None else f"{block_id}:{i}",
                source=_normalise(seg.get("source") or ""),
                target=_normalise(seg.get("translation") or ""),
                block_id=block_id,
            ))
        return out
    # The block was never translated (or the translation is older than the
    # block): the prose still has to be readable, only the target is missing.
    return [
        Sentence(id=f"{block_id}:{i}", source=sentence,
                 target=segments.target(sentence), block_id=block_id)
        for i, sentence in enumerate(split_sentences(text))
    ]


# ---------------------------------------------------------------------------- blocks
def _heading_blocks(sections: list[dict[str, Any]], segments: _Segments) -> list[Block]:
    """One heading block per node of the section tree."""
    out: list[Block] = []
    for section in sections:
        title = _normalise(section.get("title") or "")
        if not title:
            continue
        number = _normalise(section.get("number") or "")
        # The reading view numbers its headings from 2 (1 is the paper title),
        # so a top level section is 2 and each level below adds one.
        level = int(section.get("level") or 1) + 1
        out.append(Block(
            kind="heading",
            page_no=_page(section),
            bbox=_bbox(section.get("bbox")),
            id=str(section.get("id") or ""),
            level=max(2, level),
            text=f"{number} {title}".strip(),
            target=segments.target(title),
        ))
    return out


def _paragraph_blocks(blocks: list[dict[str, Any]], segments: _Segments,
                      attached_captions: set[str]) -> list[Block]:
    """The prose of the paper, one block per paragraph."""
    out: list[Block] = []
    for block in blocks:
        block_type = str(block.get("type") or block.get("block_type") or "").lower()
        if block_type not in _PARAGRAPH_TYPES:
            continue
        text = _normalise(block.get("content") or "")
        if not text:
            continue
        # A caption is rendered from the table or figure it belongs to; the
        # caption also exists as a content block, and keeping both would print
        # every caption twice.
        if block_type == "caption" and text.lower() in attached_captions:
            continue
        block_id = str(block.get("id") or "")
        sentences = _sentences(block_id, text, segments)
        out.append(Block(
            kind="paragraph",
            page_no=_page(block),
            bbox=_bbox(block.get("bbox")),
            id=block_id,
            text=text,
            target=" ".join(s.target for s in sentences if s.target),
            sentences=sentences,
        ))
    return out


def _parse_markdown(markdown: str) -> tuple[list[list[Cell]], int]:
    """Parse a stored table into ``(rows, header_rows)``.

    ``spans`` owns the table vocabulary (merged cells, alignment rows), so this
    module never inspects the Markdown itself.  The import is local because it
    is the only dependency of this module on the table layer and a paper
    without tables should not need it at all.
    """
    from app.render.spans import parse_markdown

    try:
        return parse_markdown(markdown)
    except Exception as exc:  # noqa: BLE001 - one bad table must not break the paper
        # The table still has its caption and its cropped image, so the reader
        # degrades to "look at the picture" instead of failing the whole view.
        logger.warning("table markdown could not be parsed: %s", exc)
        return [], 0


def _table_block(table: dict[str, Any], paper_id: str, segments: _Segments) -> Block:
    """One extracted table."""
    table_id = str(table.get("id") or "")
    rows, header_rows = _parse_markdown(str(table.get("markdown") or ""))
    caption = _normalise(table.get("caption") or "")
    return Block(
        kind="table",
        page_no=_page(table),
        bbox=_bbox(table.get("bbox")),
        id=table_id,
        caption=caption,
        caption_target=segments.target(caption),
        header_rows=int(header_rows or 0),
        rows=rows,
        image_url=f"/api/reader/{paper_id}/tables/{table_id}/image",
    )


def _figure_block(figure: dict[str, Any], paper_id: str, segments: _Segments) -> Block:
    """One extracted figure."""
    figure_id = str(figure.get("id") or "")
    caption = _normalise(figure.get("caption") or "")
    return Block(
        kind="figure",
        page_no=_page(figure),
        bbox=_bbox(figure.get("bbox")),
        id=figure_id,
        caption=caption,
        caption_target=segments.target(caption),
        image_url=f"/api/reader/{paper_id}/figures/{figure_id}/image",
    )


def _formula_number(latex: str, raw_text: str) -> str:
    """The printed number of a display equation (``""`` when it has none).

    The transcription keeps ``\tag{3}`` when the source had it; a PDF that was
    only transcribed from its glyphs leaves the number at the end of the line
    ("... (3)").
    """
    for text in (latex, raw_text):
        if not text:
            continue
        tag = _TAG_NUMBER.search(text)
        if tag:
            return _normalise(tag.group(1))
    for text in (raw_text, latex):
        if not text:
            continue
        paren = _PAREN_NUMBER.search(text)
        if paren:
            return _normalise(paren.group(1))
    return ""


def _formula_block(formula: dict[str, Any], paper_id: str) -> Block:
    """One display equation."""
    formula_id = str(formula.get("id") or "")
    latex = str(formula.get("latex") or "").strip()
    raw_text = str(formula.get("raw_text") or "").strip()
    return Block(
        kind="formula",
        page_no=_page(formula),
        bbox=_bbox(formula.get("bbox")),
        id=formula_id,
        # The flattened text is the fallback: it is ugly, but an equation the
        # VLM could not transcribe is still better than an empty block.
        latex=latex or raw_text,
        number=_formula_number(latex, raw_text),
        image_url=f"/api/reader/{paper_id}/formulas/{formula_id}/image",
    )


def _reading_key(block: Block) -> tuple[int, float, float]:
    """Reading position: page, then the top of the block, then its left edge.

    A block without a bbox has no position on the page, so it goes to the end
    of its page instead of pretending to stand at the top.
    """
    if len(block.bbox) >= 4:
        return (block.page_no, block.bbox[1], block.bbox[0])
    return (block.page_no, float("inf"), float("inf"))


def _author_names(value: Any) -> list[str]:
    """The author names, from the ``{"name": ...}`` records the parser stores."""
    out: list[str] = []
    for item in value or []:
        name = _normalise((item or {}).get("name") if isinstance(item, dict) else item)
        if name:
            out.append(name)
    return out


def _counts(blocks: list[Block]) -> dict[str, int]:
    """How many blocks of each kind the document has."""
    counts: dict[str, int] = {}
    sentences = 0
    translated = 0
    for block in blocks:
        counts[block.kind] = counts.get(block.kind, 0) + 1
        for sentence in block.sentences:
            sentences += 1
            translated += 1 if sentence.target else 0
    counts["total"] = len(blocks)
    counts["sentences"] = sentences
    counts["translated"] = translated
    return counts


# ----------------------------------------------------------------------------- build
async def _load_segments(paper_id: str, lang: str) -> list[dict[str, Any]]:
    """The cached translation segments of one paper (empty when untranslated)."""
    async with session_scope() as db:
        row = (await db.execute(
            select(Translation).where(Translation.paper_id == paper_id,
                                      Translation.locale == lang))).scalars().first()
        return list(row.segments or []) if row is not None else []


def _centre_inside(inner: list[float], outer: list[float], pad: float = 2.0) -> bool:
    """True when the centre of ``inner`` falls inside ``outer``."""
    if len(inner) < 4 or len(outer) < 4:
        return False
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return (outer[0] - pad <= cx <= outer[2] + pad
            and outer[1] - pad <= cy <= outer[3] + pad)


def _drop_shadowed(blocks: list[Block]) -> list[Block]:
    """Remove the blocks that would print the same content twice.

    Three kinds of duplication reach this point:

    * a heading is also emitted as a body block with the same text;
    * the text layer of a table or a figure arrives as body blocks, flattened
      (``"Average System MNLI-(m/mm) QQP ..."``), because a PDF has no cell
      structure in its text layer;
    * an axis of a chart is detected as a display formula (``"200 400 600"``).

    The object itself is rendered as a table or an image, so the shadow blocks
    are dropped rather than shown next to it.
    """
    media = [b for b in blocks if b.kind in ("table", "figure")]
    figures = [b for b in media if b.kind == "figure"]
    headings = {
        (_normalise(b.text).lower(), b.page_no) for b in blocks if b.kind == "heading"
    }

    kept: list[Block] = []
    for block in blocks:
        if block.kind in ("table", "figure", "heading"):
            kept.append(block)
            continue
        if block.kind == "paragraph" and (
                (_normalise(block.text).lower(), block.page_no) in headings):
            continue
        if block.kind == "formula":
            if any(_centre_inside(block.bbox, figure.bbox) for figure in figures):
                continue
            kept.append(block)
            continue
        if any(_centre_inside(block.bbox, item.bbox) for item in media):
            continue
        kept.append(block)
    return kept


async def build_document(paper_id: str, lang: str = "zh") -> Document | None:
    """Re-lay out one paper for the reading view.

    Every element of the paper (heading, paragraph, table, figure, formula) is
    one block of the returned document, ordered by page and then by its
    position on the page, so the reader can render the paper as a continuous
    document instead of as a set of pages.

    The translation is attached when it exists: a sentence keeps the target of
    the segment it came from, a heading and a caption are matched on their
    text.  A paper that was never translated still yields a document -- its
    targets are empty strings, never an error.

    Returns ``None`` when the paper does not exist.
    """
    paper = await paper_service.get_paper(paper_id)
    if paper is None:
        return None
    structure = await paper_service.get_paper_structure(paper_id)
    segments = _Segments(await _load_segments(paper_id, lang))

    tables = list(structure.get("tables") or [])
    figures = list(structure.get("figures") or [])
    formulas = list(structure.get("formulas") or [])

    # A caption is stored twice: in its own table/figure row and as a content
    # block of the prose.  The row owns the caption, the block is dropped.
    attached_captions = {
        _normalise(item.get("caption") or "").lower()
        for item in (*tables, *figures)
    } - {""}

    blocks: list[Block] = []
    blocks.extend(_heading_blocks(list(structure.get("sections") or []), segments))
    blocks.extend(_paragraph_blocks(list(structure.get("blocks") or []),
                                    segments, attached_captions))
    blocks.extend(_table_block(table, paper_id, segments) for table in tables)
    blocks.extend(_figure_block(figure, paper_id, segments) for figure in figures)
    blocks.extend(_formula_block(formula, paper_id) for formula in formulas)
    blocks = _drop_shadowed(blocks)
    # Python's sort is stable, so elements that share a position keep the
    # order they were collected in.
    blocks.sort(key=_reading_key)

    title = _normalise(paper.get("title") or "") or _normalise(structure.get("title") or "")
    abstract = _normalise(paper.get("abstract") or "")
    document = Document(
        paper_id=paper_id,
        title=title,
        title_target=segments.target(title),
        authors=_author_names(paper.get("authors")),
        abstract=abstract,
        abstract_target=segments.target(abstract),
        lang=lang,
        blocks=blocks,
        counts=_counts(blocks),
    )
    logger.info("built document for %s: %s blocks (%s translated of %s sentences)",
                paper_id, len(blocks), document.counts.get("translated", 0),
                document.counts.get("sentences", 0))
    return document
