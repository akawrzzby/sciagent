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

#: A heading that is nothing but an equation number: ``(1)``, ``(2a)``.
_EQUATION_LABEL = re.compile(r"\(?\d{1,3}[a-z]?\)?")

#: A heading that starts with a section number: ``3``, ``3.2``, ``A.1``.
_SECTION_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)+)")

#: A chapter of the appendix: ``A Evaluation Datasets``.
_LETTERED_HEADING = re.compile(r"^[A-Z]\s+\S")


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


def _order(entry: dict[str, Any]) -> float:
    """The parser's column-aware reading position of a block.

    Zero means "unknown": the block is placed by :func:`_place_media` instead.
    """
    try:
        return float(entry.get("order_index") or 0)
    except (TypeError, ValueError):
        return 0.0


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
def _heading_blocks(sections: list[dict[str, Any]], segments: _Segments,
                    blocks: list[dict[str, Any]] | None = None,
                    paper_title: str = "") -> list[Block]:
    """One heading block per node of the section tree.

    A section row carries no reading position, but the same title also arrives
    as a ``text`` block, which does.  That block is the one the order is taken
    from; without it a heading would sort to the top of its page.
    """
    positions: dict[tuple[str, int], tuple[float, list[float]]] = {}
    for block in blocks or []:
        key = (_normalise(block.get("content") or "").lower(), _page(block))
        order = _order(block)
        if order:
            positions.setdefault(key, (order, _bbox(block.get("bbox"))))

    out: list[Block] = []
    candidates: list[tuple[Block, str]] = []
    for section in sections:
        title = _normalise(section.get("title") or "")
        if not title:
            continue
        number = _normalise(section.get("number") or "")
        # The reading view numbers its headings from 2 (1 is the paper title),
        # so a top level section is 2 and each level below adds one.
        level = int(section.get("level") or 1) + 1
        label = f"{number} {title}".strip()
        page = _page(section)
        # Section rows number themselves on their own scale (0, 1, 2 …), which
        # cannot be compared with the block positions; only the text block that
        # carries the same title is on the right scale.
        order, bbox = positions.get((label.lower(), page), (0.0, []))
        candidates.append((Block(
            kind="heading",
            page_no=page,
            bbox=_bbox(section.get("bbox")) or bbox,
            id=str(section.get("id") or ""),
            level=max(2, level),
            text=label,
            target=segments.target(title),
            order=order,
        ), title.lower()))

    # The running head printed at the top of every page is parsed as a section
    # whose title is the paper's own title.  A title that shows up on three or
    # more pages is that running head too, not a section: keeping it turns the
    # chapter numbering into one fake chapter per page.
    title_key = _normalise(paper_title).lower()
    pages_by_title: dict[str, set[int]] = {}
    for block, bare in candidates:
        pages_by_title.setdefault(bare, set()).add(block.page_no)

    first_page = min((block.page_no for block, _ in candidates), default=1)
    seen_abstract = False
    seen_appendix = False

    for block, bare in candidates:
        text = _normalise(block.text)
        if bare == "abstract":
            seen_abstract = True
        # The running head carries the paper's own title.  A *numbered* section
        # may legitimately repeat the title (``3 Layer normalization`` in the
        # paper "Layer Normalization"), so only the unnumbered rows are
        # dropped here; a numbered running head is caught by the repeat rule.
        if title_key and bare == title_key and not _SECTION_NUMBER.match(text):
            continue
        if len(pages_by_title[bare]) >= 3:
            continue
        # An equation number is set like a heading ("(1)") and is picked up as
        # one, but it has no words in it.
        if _EQUATION_LABEL.fullmatch(text):
            continue
        # Everything above the abstract on the first page is front matter: the
        # title, the authors, their affiliations.  The masthead already shows
        # the first two, and an author line is never a section.
        if (not seen_abstract and block.page_no == first_page
                and not _SECTION_NUMBER.match(text)):
            continue
        # An appendix is lettered (A, B, C …).  A numbered list inside one is a
        # set of instructions, not a chapter, but its items are parsed as
        # chapters numbered 1, 2, 3 ….
        if block.level <= 2 and _LETTERED_HEADING.match(text):
            seen_appendix = True
        elif seen_appendix and block.level <= 2 and text[:1].isdigit():
            continue
        out.append(block)
    return out


def _paragraph_blocks(blocks: list[dict[str, Any]], segments: _Segments,
                      attached_captions: set[str]) -> list[Block]:
    """The prose of the paper, one block per paragraph."""
    out: list[Block] = []
    for block in blocks:
        block_type = str(block.get("type") or block.get("block_type") or "").lower()
        text = _normalise(block.get("content") or "")
        if not text:
            continue
        # A paper set entirely in a math font can have its running prose
        # classified as a formula (ZeRO: 158 formula blocks, 18 body blocks).
        # Prose is recovered here; a page number is not.
        if block_type == "formula":
            if _looks_like_prose(text):
                block_type = "body"
            else:
                continue
        if block_type not in _PARAGRAPH_TYPES:
            continue
        if len(text) <= 4 and text.isdigit():
            continue  # a folio
        # A word or two on its own is what the parser makes of the labels
        # printed inside a figure ("governments", "American"); it is not a
        # paragraph, and printing it between two paragraphs is noise.
        if _is_fragment(text):
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
            order=_order(block),
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


def _reading_key(block: Block) -> tuple[int, float, float, float]:
    """Kept for the tests: page, then the top of the block, then its left edge."""
    y = block.bbox[1] if len(block.bbox) >= 4 else float("inf")
    x = block.bbox[0] if len(block.bbox) >= 4 else float("inf")
    return (block.page_no, block.order, y, x)


def _sort_by_layout(blocks: list[Block]) -> list[Block]:
    """Put the blocks of a paper in reading order, from geometry alone.

    The parser also numbers the blocks, but its numbering is only column aware
    on the pages where its own column detection succeeded: on a page where it
    failed, the number is a plain top-to-bottom sweep and a right-column
    heading ends up before the left-column text that precedes it.

    So the order is rebuilt here:

    * a block wider than 55% of the text area spans the columns (a figure, a
      full-width table, a caption); those split the page into bands,
    * inside a band the left column is read to its bottom, then the right one,
    * within a column the blocks go top to bottom.

    A band starts with the spanning block that opens it, so ``(band, column)``
    is enough: spanning blocks take column 0 and the columns below them take
    1 and 2.
    """
    by_page: dict[int, list[Block]] = {}
    for block in blocks:
        by_page.setdefault(block.page_no, []).append(block)

    ordered: list[Block] = []
    for page in sorted(by_page):
        items = by_page[page]
        boxes = [b.bbox for b in items if len(b.bbox) >= 4]
        if not boxes:
            ordered.extend(sorted(items, key=lambda b: (b.kind, b.id)))
            continue
        left_edge = min(b[0] for b in boxes)
        right_edge = max(b[2] for b in boxes)
        width = max(right_edge - left_edge, 1.0)
        middle = (left_edge + right_edge) / 2

        spanning = [b for b in items
                    if len(b.bbox) >= 4
                    and (b.bbox[2] - b.bbox[0]) > width * 0.55
                    # A wide *heading* is just a long title.  Only an object
                    # can span the columns and split the page into bands; a
                    # text block has to cover almost the whole width to.
                    and (b.kind in ("figure", "table", "formula")
                         or (b.bbox[2] - b.bbox[0]) > width * 0.8)]
        spanning.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        band_of: dict[int, int] = {id(b): index for index, b in enumerate(spanning)}

        def key(block: Block) -> tuple[int, int, int, float, float]:
            if len(block.bbox) < 4:
                return (10 ** 6, 0, 0, 0.0, 0.0)
            y = block.bbox[1]
            band = band_of.get(id(block))
            if band is not None:
                # A spanning object closes the band above it, so it sorts after
                # the blocks that start above its own top edge.
                return (band, 1, 0, y, block.bbox[0])
            # everything below a spanning block belongs to the next band
            band = sum(1 for b in spanning if b.bbox[3] <= y)
            column = 0 if (block.bbox[0] + block.bbox[2]) / 2 < middle else 1
            return (band, 0, column, y, block.bbox[0])

        ordered.extend(sorted(items, key=key))

    for index, block in enumerate(ordered):
        block.order = float(index)
    return ordered


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


#: A block classified as a formula that is really a sentence.
def _looks_like_prose(text: str) -> bool:
    """True for a "formula" block that is an ordinary paragraph.

    The discriminator is length: an equation is a handful of tokens, a
    paragraph is a dozen words or more.  Symbols are not a reliable signal,
    because a paper set in a math font has ordinary words in it too.
    """
    words = text.split()
    return len(words) >= 12 and sum(1 for w in words if w.isalpha()) >= 6


#: A label printed inside a figure, picked up as a paragraph of its own.
def _is_fragment(text: str) -> bool:
    """True for a standalone word or two, which is never a paragraph.

    A real short paragraph is a list item ("• 16 layers"), a sentence
    ("Limitations."), or a table cell kept as text; those are kept.  Punctuation
    on its own — the parser emits "." and "-" from the rules inside a figure —
    is always a fragment.
    """
    if not any(ch.isalnum() for ch in text):
        return True
    if text[:1] in "•-–—*" or text.endswith((".", ":", "?", "!")):
        return False
    return len(text.split()) <= 3 and len(text) <= 40


#: A caption starts with the object it belongs to: ``Figure 3:``, ``Table E.1:``.
_CAPTION_START = re.compile(
    r"^(?:figure|table)\s+[A-Z]?\.?\d+(?:\.\d+)*\s*[:.]", re.IGNORECASE)

#: How far a caption may sit from its object and still belong to it, in points.
#: A figure at the top of a column can have its caption at the bottom of the
#: same column, so the reach is a whole column's height, not a line or two.
_CAPTION_REACH = 420.0


def _fill_missing_captions(blocks: list[Block], raw: list[dict[str, Any]],
                           segments: _Segments) -> set[str]:
    """Attach a caption to the figures and tables the parse left without one.

    The caption is a content block of its own, and pairing it with its object
    is a separate step upstream that does not always succeed.  Three cases are
    handled here, all of them layout facts rather than guesses:

    * the caption sits an inch away from the object, not next to it;
    * a figure is made of several panels, and the parse recorded one row per
      panel with the caption attached to only one of them — the panels then
      share the caption of the group;
    * the caption is the last line of the previous page.

    Returns the texts that were consumed, so they are not printed a second time
    as paragraphs.
    """
    captions = [b for b in raw
                if str(b.get("type") or "").lower() == "caption"
                or (str(b.get("type") or "").lower() == "body"
                    and _CAPTION_START.match(_normalise(b.get("content") or "")))]
    by_page: dict[int, list[tuple[float, str, dict[str, Any]]]] = {}
    for caption in captions:
        text = _normalise(caption.get("content") or "")
        if not _CAPTION_START.match(text):
            continue
        box = _bbox(caption.get("bbox"))
        if len(box) < 4:
            continue
        by_page.setdefault(_page(caption), []).append((box[1], text, caption))

    used: set[str] = set()
    for block in blocks:
        if block.kind not in ("figure", "table") or block.caption.strip():
            continue
        if len(block.bbox) < 4:
            continue
        kind = "table" if block.kind == "table" else "figure"
        # A caption is looked for on the object's own page first, then on the
        # page that holds its top or bottom edge.
        pages = [block.page_no]
        if block.bbox[1] < 300:
            pages.append(block.page_no - 1)
        if block.bbox[3] > 500:
            pages.append(block.page_no + 1)

        best: tuple[str, dict[str, Any]] | None = None
        best_gap = _CAPTION_REACH
        for page in pages:
            for y0, text, caption in by_page.get(page, []):
                if kind == "table" and not text.lower().startswith("table"):
                    continue
                if kind == "figure" and not text.lower().startswith("figure"):
                    continue
                gap = min(abs(y0 - block.bbox[3]), abs(block.bbox[1] - y0))
                if page != block.page_no:
                    gap += 80.0  # a caption on another page is a fallback
                if gap < best_gap:
                    best, best_gap = (text, caption), gap
        if best is not None:
            text, _ = best
            block.caption = text
            block.caption_target = segments.target(text)
            used.add(text.lower())
    return used


def _drop_front_matter(blocks: list[Block], title: str, authors: list[str]) -> list[Block]:
    """Remove the title and the author block from the body of the paper.

    The masthead already prints both, and the parser also emits them as body
    paragraphs: the title once, then one line per author with the affiliation
    and the e-mail address attached.
    """
    title_key = _normalise(title).lower()
    names = [name.strip().lower() for name in authors if name.strip()]
    kept: list[Block] = []
    for block in blocks:
        if block.kind != "paragraph":
            kept.append(block)
            continue
        text = _normalise(block.text)
        lowered = text.lower()
        if title_key and lowered == title_key:
            continue
        if "@" in text and len(text) < 200:
            continue  # an author line carries the e-mail address
        if names and len(text) < 200 and any(lowered.startswith(n) for n in names):
            continue
        kept.append(block)
    return kept


def _drop_page_images(blocks: list[Block]) -> list[Block]:
    """Remove the "figures" and "formulas" that are really whole pages.

    A scanned paper is parsed page by page, and the rasteriser records each
    page as a figure: the box starts at the origin and covers the whole sheet,
    and there is no caption because the page is not a figure.  A page of
    running text set in a math font can also be recorded as one enormous
    formula.  Shown as objects they would print the page twice.
    """
    kept: list[Block] = []
    for block in blocks:
        page_sized = (len(block.bbox) >= 4
                      and (block.bbox[2] - block.bbox[0]) >= 480
                      and (block.bbox[3] - block.bbox[1]) >= 600)
        if page_sized and block.kind == "formula":
            continue
        if (page_sized and block.kind == "figure" and not block.caption.strip()
                and block.bbox[0] <= 2 and block.bbox[1] <= 2):
            continue
        kept.append(block)
    return kept


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
        (_normalise(b.text).lower(), b.page_no): b for b in blocks if b.kind == "heading"
    }

    def inside_object(block: Block) -> bool:
        """True when this text sits inside a table or a figure *of its page*.

        The page has to be compared: a figure at the top of page 5 and a
        paragraph at the top of page 4 share the same coordinates, and
        ignoring that used to delete a third of the paper.
        """
        return any(
            item.page_no == block.page_no and _centre_inside(block.bbox, item.bbox)
            for item in media
        )

    kept: list[Block] = []
    for block in blocks:
        if block.kind in ("table", "figure", "heading"):
            kept.append(block)
            continue
        if block.kind == "paragraph" and (
                (_normalise(block.text).lower(), block.page_no) in headings):
            # The heading owns this text, but the body block carries the
            # reading position, so it is handed over before the duplicate goes.
            heading = headings[(_normalise(block.text).lower(), block.page_no)]
            if not heading.order:
                heading.order = block.order
            if len(heading.bbox) < 4:
                heading.bbox = block.bbox
            continue
        if block.kind == "formula":
            if any(item.page_no == block.page_no
                   and _centre_inside(block.bbox, item.bbox) for item in figures):
                continue
            kept.append(block)
            continue
        if inside_object(block):
            continue
        kept.append(block)
    return kept


def _looks_like_reference(text: str) -> bool:
    """True for a line that could be a bibliography entry.

    The reference extractor also collects numbered lists, figure labels and
    instructions ("Remove any phrases between matching brackets"), which would
    otherwise be printed as if they were cited works.  These three tests reject
    the obvious cases without guessing at the citation style.
    """
    stripped = text.strip()
    if not stripped or stripped[0] in "•-/–—*":
        return False
    words = stripped.split()
    if len(words) < 5:
        return False
    digits = sum(1 for ch in stripped if ch.isdigit())
    if digits > len(stripped) * 0.45:
        return False  # a row of numbers copied out of a table or a chart
    return True


def _reference_blocks(references: list[dict[str, Any]], segments: _Segments) -> list[Block]:
    """The bibliography, as its own section at the end of the document.

    The reading view used to end with the last paragraph: the paper cites
    ``[11]`` but the list it points at was never rendered, so nothing could be
    jumped to.  The printed entry is kept as it appears in the paper; the parsed
    fields are not always reliable enough to rebuild it.
    """
    if not references:
        return []
    heading = Block(kind="heading", id="references", level=2, text="参考文献",
                    target="参考文献")
    out = [heading]
    for item in references:
        number = str(item.get("index_in_doc") or "").strip()
        text = _normalise(item.get("raw_text") or "")
        if not text or not _looks_like_reference(text):
            continue
        out.append(Block(
            kind="reference",
            id=f"ref-{number}",
            number=number,
            text=text,
            target=segments.target(text),
        ))
    return out if len(out) > 1 else []


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
    blocks.extend(_heading_blocks(list(structure.get("sections") or []), segments,
                                  list(structure.get("blocks") or []),
                                  str(paper.get("title") or structure.get("title") or "")))
    blocks.extend(_paragraph_blocks(list(structure.get("blocks") or []),
                                    segments, attached_captions))
    blocks.extend(_table_block(table, paper_id, segments) for table in tables)
    blocks.extend(_figure_block(figure, paper_id, segments) for figure in figures)
    blocks.extend(_formula_block(formula, paper_id) for formula in formulas)
    borrowed = _fill_missing_captions(blocks, list(structure.get("blocks") or []), segments)
    if borrowed:
        blocks = [b for b in blocks
                  if not (b.kind == "paragraph" and _normalise(b.text).lower() in borrowed)]

    title = _normalise(paper.get("title") or "") or _normalise(structure.get("title") or "")
    blocks = _drop_page_images(blocks)
    blocks = _drop_front_matter(blocks, title, _author_names(paper.get("authors")))
    blocks = _drop_shadowed(blocks)
    blocks = _sort_by_layout(blocks)
    blocks.extend(_reference_blocks(list(structure.get("references") or []), segments))

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
