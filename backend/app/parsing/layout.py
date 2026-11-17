"""Low level PDF layout analysis.

Responsibilities
----------------
* Extract positioned text blocks with font metadata from PyMuPDF.
* Detect the column layout of a page (single / double column) so that the
  reading order and the rendered text match the printed paper.
* Provide helpers to convert between PDF points and the normalised
  ``[x0, y0, x1, y1]`` bounding boxes stored in the database.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

import pymupdf

# --------------------------------------------------------------------------- constants
# Horizontal gap (in PDF points) that separates two text columns.
COLUMN_GAP = 28.0
# A block must cover at least this fraction of the page height to count as a
# "full column" block when clustering columns.
MIN_COLUMN_COVERAGE = 0.25

# Patterns that identify a page that is mostly one wide table.
_HORIZONTAL_RULE = re.compile(r"[-_]{3,}")

# --------------------------------------------------------------------------- block types
#
# Every text block carries a type so that the reader can tell body text from
# captions, figure labels, footnotes and running headers.  The values are
# stored in ``content_blocks.block_type`` (a column that already exists, it
# used to be the constant "text").
BLOCK_BODY = "body"
BLOCK_CAPTION = "caption"
BLOCK_FIGURE_LABEL = "figure_label"
BLOCK_FOOTNOTE = "footnote"
BLOCK_HEADER = "header"
BLOCK_TEXT = "text"
BLOCK_FORMULA = "formula"

#: Every value ``block_type`` may take.
BLOCK_TYPES = frozenset({
    BLOCK_BODY, BLOCK_CAPTION, BLOCK_FIGURE_LABEL,
    BLOCK_FOOTNOTE, BLOCK_HEADER, BLOCK_TEXT, BLOCK_FORMULA,
})

#: Blocks that must never become part of a section's body text.
EXCLUDED_BLOCK_TYPES = frozenset({BLOCK_FIGURE_LABEL, BLOCK_FOOTNOTE, BLOCK_HEADER})

#: Blocks that make up the body of a section.
CONTENT_BLOCK_TYPES = frozenset({BLOCK_BODY, BLOCK_CAPTION})

# --------------------------------------------------------------------------- thresholds
#: A block whose top edge lies below this fraction of the page height sits in
#: the footnote zone (the bottom of the page).
FOOTNOTE_ZONE = 0.72
#: A footnote is rendered noticeably smaller than the body text of its page.
FOOTNOTE_SIZE_RATIO = 0.93
#: The top band of a page, where a running header lives.
HEADER_ZONE = 0.12
#: The margin band at the very top and the very bottom of a page, where the
#: page number (the "folio") is printed.
FOLIO_ZONE = 0.90
#: A block that covers at least this fraction of a figure belongs to the figure.
FIGURE_OVERLAP_RATIO = 0.5

# "Figure 3: ..." / "Fig. 3 ..." / "Table 2: ..." (the caption may be wrapped
# over several lines, hence the whitespace normalisation before matching).
_CAPTION_RE = re.compile(r"^(figure|fig\.?|table|tab\.?)\s*\d+", re.IGNORECASE)
# A footnote starts with its marker ("9 The GLUE data set ...", "10https://...").
_FOOTNOTE_MARKER_RE = re.compile(
    r"^\s*(?:[\u00b9\u00b2\u00b3\u2070-\u209f]|\d{1,3})\s*(?=[A-Za-z(\[])"
)
# The rotated arXiv stamp in the page margin.
_ARXIV_STAMP_RE = re.compile(r"^\s*arxiv:\s*\d{4}\.\d{4,5}", re.IGNORECASE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
# A page number standing on its own: "12", "iv".
_FOLIO_RE = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdm]{1,7})\s*$", re.IGNORECASE)

# Text extraction flags for the structured ("dict") reader.  Ligatures are
# expanded back to ASCII on purpose -- see ``extract_blocks``.
_DICT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES


@dataclass(slots=True)
class Span:
    """A run of text sharing one font."""

    text: str
    bbox: tuple[float, float, float, float]
    size: float
    flags: int
    font: str

    @property
    def is_bold(self) -> bool:
        # PyMuPDF flag bit 4 (16) == bold, bit 1 (2) == italic.
        return bool(self.flags & 16) or "bold" in self.font.lower() or "black" in self.font.lower()

    @property
    def is_italic(self) -> bool:
        return bool(self.flags & 2) or "italic" in self.font.lower() or "oblique" in self.font.lower()

    @property
    def is_math_font(self) -> bool:
        """True for Computer Modern / math fonts used by LaTeX output."""
        f = self.font.lower()
        return any(k in f for k in ("cmmi", "cmsy", "cmex", "cmr", "msam", "msbm", "math"))


@dataclass(slots=True)
class Block:
    """A positioned block of text (paragraph-ish)."""

    text: str
    bbox: tuple[float, float, float, float]
    spans: list[Span] = field(default_factory=list)
    page_no: int = 0
    column: int = 0
    block_index: int = 0
    # True when the block was split out of a paragraph because it is an
    # equation (see :func:`split_equation_lines`).
    is_equation: bool = False
    # One of ``BLOCK_BODY`` / ``BLOCK_CAPTION`` / ``BLOCK_FIGURE_LABEL`` /
    # ``BLOCK_FOOTNOTE`` / ``BLOCK_HEADER`` / ``BLOCK_TEXT``.
    block_type: str = BLOCK_BODY

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def max_size(self) -> float:
        return max((s.size for s in self.spans), default=0.0)

    @property
    def font_size(self) -> float:
        """Dominant (mode-like) font size of the block."""
        if not self.spans:
            return 0.0
        sizes = [round(s.size, 1) for s in self.spans if s.text.strip()]
        return statistics.mode(sizes) if sizes else 0.0

    @property
    def is_bold(self) -> bool:
        chars = sum(len(s.text.strip()) for s in self.spans)
        bold = sum(len(s.text.strip()) for s in self.spans if s.is_bold)
        return chars > 0 and bold / chars > 0.6

    @property
    def is_math(self) -> bool:
        math_chars = sum(len(s.text.strip()) for s in self.spans if s.is_math_font)
        total = sum(len(s.text.strip()) for s in self.spans) or 1
        return math_chars / total > 0.5


def _line_is_equation(line_spans: list[Span]) -> bool:
    """True when a printed line is an equation rather than prose.

    Two signals are used: most of the characters come from a math font
    (``cmmi``/``cmsy``/``cmex``/``msam``/``msbm``), or the line carries an
    equation sign and little else.  The number of an equation -- "(1)" -- is
    part of the equation as well.
    """
    text = "".join(s.text for s in line_spans).strip()
    if not text:
        return False
    if _EQUATION_NUMBER.fullmatch(text):
        return True
    math_chars = sum(len(s.text.strip()) for s in line_spans if s.is_math_font)
    total = sum(len(s.text.strip()) for s in line_spans) or 1
    if math_chars / total > 0.4:
        return True
    # A line such as "h0 = UWe + Wp" is typeset in a mix of fonts.
    if "=" in text and math_chars / total > 0.15 and len(text.split()) <= 12:
        return True
    return False


# --------------------------------------------------------------------------- printed rows
#
# PyMuPDF cuts a text line wherever a character changes the vertical extent of
# the line: a subscript, a superscript or a big operator with its limit
# underneath each become a line of their own.  Equation (1) of the BERT paper,
#
#     L1(U) = sum_i log P(ui | ui-k, ..., ui-1; Theta)      (1)
#
# therefore arrives as three pieces -- "L1(U) =", the sum glyph, and
# "i log P(...)" -- and the reader would show the equation in pieces and the
# limit under the sum in the wrong place.

#: Two lines belong to the same printed row when their vertical extents
#: overlap by at least this fraction of the shorter one.
_ROW_OVERLAP_RATIO = 0.5
#: Two lines belong to the same row only when they also touch horizontally.
#: A run-in heading ("Auxiliary training objectives" set in front of the
#: paragraph it introduces) shares the row of the paragraph but stands a few
#: points away from it.
_ROW_GAP = 6.0
#: A span smaller than this fraction of the row's base size is a script.
_SCRIPT_SIZE_RATIO = 0.85
#: The vertical offset (in PDF points) that separates a script from its base.
_SCRIPT_OFFSET = 1.0
#: How far below a big operator its limit may sit and still belong to it.
_LIMIT_GAP = 12.0


def _band(spans: list[Span]) -> tuple[float, float]:
    """The vertical extent of a group of spans."""
    return (min(s.bbox[1] for s in spans), max(s.bbox[3] for s in spans))


def _band_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """How much two vertical bands overlap, as a fraction of the shorter one."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi <= lo:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    return (hi - lo) / shorter if shorter > 0 else 0.0


def _row_gap(a: list[Span], b: list[Span]) -> float:
    """The horizontal distance between two groups of spans (0 when they meet)."""
    a0 = min(s.bbox[0] for s in a)
    a1 = max(s.bbox[2] for s in a)
    b0 = min(s.bbox[0] for s in b)
    b1 = max(s.bbox[2] for s in b)
    if b0 > a1:
        return b0 - a1
    if a0 > b1:
        return a0 - b1
    return 0.0


def _sort_row(row: list[Span]) -> list[Span]:
    """Order the spans of a row left to right.

    Spans that start at the same place are ordered top to bottom, so that
    the limit under an operator follows the operator instead of preceding it.
    """
    ordered = sorted(row, key=lambda s: s.bbox[0])
    result: list[Span] = []
    group: list[Span] = []
    for span in ordered:
        if group and span.bbox[0] - group[0].bbox[0] > 2.0:
            result.extend(sorted(group, key=lambda s: s.bbox[1]))
            group = []
        group.append(span)
    result.extend(sorted(group, key=lambda s: s.bbox[1]))
    return result


def _is_extension_font(span: Span) -> bool:
    """True for a span set in the extension font, where LaTeX puts the big
    operators (the summation sign, the product sign, the large brackets).

    Only such a glyph can carry a limit underneath it, and that is what
    separates a limit from any other small character that happens to sit
    below something.
    """
    return "cmex" in span.font.lower()


def _is_lone_equation_number(group: list[Span]) -> bool:
    """True for a group that holds nothing but the number of an equation."""
    text = "".join(s.text for s in group).strip()
    return bool(_EQUATION_NUMBER.fullmatch(text))


def _attach_limits(rows: list[list[Span]]) -> list[list[Span]]:
    """Move the limit under a big operator into the operator's row.

    "sum_i" is printed on two rows: the operator on one, the limit underneath
    it on the next.  Read as two rows the limit becomes a line of its own and
    the formula reads as noise.
    """
    if len(rows) < 2:
        return rows
    sizes = [round(s.size, 1) for row in rows for s in row if s.text.strip()]
    base_size = statistics.mode(sizes) if sizes else 0.0
    if base_size <= 0:
        return rows
    moves: list[tuple[int, int, Span]] = []
    for index, row in enumerate(rows):
        for span in row:
            if span.size > base_size * _SCRIPT_SIZE_RATIO:
                continue
            centre = (span.bbox[0] + span.bbox[2]) / 2.0
            target: int | None = None
            for above in range(index):
                for anchor in rows[above]:
                    if not _is_extension_font(anchor):
                        continue
                    if anchor.bbox[3] > span.bbox[1]:
                        continue
                    if span.bbox[1] - anchor.bbox[3] > _LIMIT_GAP:
                        continue
                    if not (anchor.bbox[0] - 2.0 <= centre <= anchor.bbox[2] + 2.0):
                        continue
                    target = above
                    break
                if target is not None:
                    break
            if target is not None:
                moves.append((index, target, span))
    if not moves:
        return rows
    for source, destination, span in moves:
        rows[source] = [s for s in rows[source] if s is not span]
        rows[destination].append(span)
        # The extension font has no character map, so the big operator comes
        # out as a letter.  A limit underneath it means it is a summation.
        for anchor in rows[destination]:
            if _is_extension_font(anchor) and anchor.text.strip() == "X":
                anchor.text = "∑"
    rows = [row for row in rows if row]
    for row in rows:
        row[:] = _sort_row(row)
    rows.sort(key=lambda row: (min(s.bbox[1] for s in row),
                                min(s.bbox[0] for s in row)))
    return rows


def _visual_rows(line_groups: list[list[Span]]) -> list[list[Span]]:
    """Merge the PDF lines that share a printed row.

    The spans of a row are sorted left to right, so that a character and its
    script keep their order even when PyMuPDF reported them in separate
    lines.
    """
    rows: list[list[Span]] = []
    bands: list[tuple[float, float]] = []
    for group in line_groups:
        if not group:
            continue
        band = _band(group)
        target: int | None = None
        for i, other in enumerate(bands):
            if _band_overlap(band, other) < _ROW_OVERLAP_RATIO:
                continue
            if (_row_gap(rows[i], group) > _ROW_GAP
                    and not _is_lone_equation_number(group)):
                continue
            target = i
            break
        if target is None:
            rows.append(list(group))
            bands.append(band)
            continue
        rows[target].extend(group)
        bands[target] = (min(bands[target][0], band[0]),
                         max(bands[target][1], band[1]))
    rows.sort(key=lambda row: (min(s.bbox[1] for s in row),
                                min(s.bbox[0] for s in row)))
    for row in rows:
        row[:] = _sort_row(row)
    return _attach_limits(rows)


def _span_bands(spans: list[Span]) -> list[tuple[float, float]]:
    """The printed rows of a set of spans, as vertical bands."""
    bands: list[tuple[float, float]] = []
    for span in sorted(spans, key=lambda s: s.bbox[1]):
        band = (span.bbox[1], span.bbox[3])
        for i, other in enumerate(bands):
            if _band_overlap(band, other) >= _ROW_OVERLAP_RATIO:
                bands[i] = (min(other[0], band[0]), max(other[1], band[1]))
                break
        else:
            bands.append(band)
    return bands


def _x_overlap(a: Span, b: Span) -> float:
    """How much two spans overlap horizontally (negative when they are apart)."""
    return min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])


def _pick_anchor(script: Span, anchors: list[Span]) -> Span | None:
    """The character a script belongs to.

    Usually that is the character printed just before it.  The limit under a
    big operator is the exception: it is printed on the row below and sits
    directly under the operator, so the operator is its anchor.  Only a
    character set in a math font can be an anchor -- the digits of a
    superscript citation are part of the sentence, not of a formula.
    """
    candidates = [a for a in anchors if a.is_math_font]
    if not candidates:
        return None
    prev = candidates[-1]
    if _x_overlap(prev, script) >= -2.0:
        return prev
    centre = (script.bbox[0] + script.bbox[2]) / 2.0
    for anchor in reversed(candidates):
        if anchor.bbox[0] - 1.0 <= centre <= anchor.bbox[2] + 1.0:
            return anchor
    return None


def _row_texts(rows: list[list[Span]]) -> list[str]:
    """The text of every printed row, with scripts written as LaTeX markup.

    A subscript or a superscript is a smaller span that sits lower or higher
    than the character it belongs to.  Read as plain characters the formula
    falls apart -- "W" and "e" become two separate letters, "x" and "1" become
    "x1" -- and the reader shows "the token embedding matrix We" instead of
    "W_e".
    """
    sizes = [round(s.size, 1) for row in rows for s in row if s.text.strip()]
    base_size = statistics.mode(sizes) if sizes else 0.0
    anchors: list[Span] = []
    texts: list[str] = []

    for row in rows:
        parts: list[str] = []
        script: list[str] = []
        script_kind = ""
        current_anchor: Span | None = None

        def flush() -> None:
            nonlocal script, script_kind
            if script:
                mark = "_" if script_kind == "sub" else "^"
                parts.append(f"{mark}{{{''.join(script)}}}")
            script = []
            script_kind = ""

        for span in row:
            text = span.text
            if not text.strip():
                continue
            anchor = None
            if base_size > 0 and span.size <= base_size * _SCRIPT_SIZE_RATIO:
                anchor = _pick_anchor(span, anchors) or current_anchor
            if anchor is not None:
                centre = (span.bbox[1] + span.bbox[3]) / 2.0
                base_centre = (anchor.bbox[1] + anchor.bbox[3]) / 2.0
                if centre > base_centre + _SCRIPT_OFFSET:
                    kind = "sub"
                elif centre < base_centre - _SCRIPT_OFFSET:
                    kind = "sup"
                else:
                    kind = ""
                if kind:
                    if kind != script_kind:
                        flush()
                        script_kind = kind
                    current_anchor = anchor
                    script.append(text.strip())
                    continue
            flush()
            script_kind = ""
            current_anchor = None
            parts.append(text)
            if base_size <= 0 or span.size >= base_size * _SCRIPT_SIZE_RATIO:
                anchors.append(span)
        flush()
        texts.append("".join(parts))
    return texts


def split_equation_lines(
    page_no: int,
    rows: list[list[Span]],
    texts: list[str] | None = None,
) -> list[Block]:
    """Split the printed rows of one PDF block into prose and equation runs.

    A LaTeX PDF often puts the sentence that introduces an equation and the
    equation itself into the same text block:

        "over target tokens: h0 = UWe + Wp hl = transformer_block(hl-1) ..."

    As one paragraph the equation reads as flattened noise ("hl−1∀i ∈[1, n]").
    Splitting the block at the equation rows lets the reader show the
    typeset equation as an image and keeps the sentence around it readable.
    """
    if len(rows) < 2:
        return []
    if texts is None:
        texts = _row_texts(rows)
    kinds = [_line_is_equation(row) for row in rows]
    if not any(kinds) or all(kinds):
        return []

    runs: list[tuple[bool, list[int]]] = []
    for index, is_equation in enumerate(kinds):
        if runs and runs[-1][0] == is_equation:
            runs[-1][1].append(index)
        else:
            runs.append((is_equation, [index]))

    blocks: list[Block] = []
    for is_equation, indexes in runs:
        spans = [span for i in indexes for span in rows[i]]
        text = "\n".join(texts[i] for i in indexes).strip()
        if not text:
            continue
        x0 = min(s.bbox[0] for s in spans)
        y0 = min(s.bbox[1] for s in spans)
        x1 = max(s.bbox[2] for s in spans)
        y1 = max(s.bbox[3] for s in spans)
        blocks.append(Block(
            text=text, bbox=(x0, y0, x1, y1), spans=spans, page_no=page_no,
            block_type=BLOCK_FORMULA if is_equation else BLOCK_BODY,
            is_equation=is_equation,
        ))
    return blocks


def bbox_to_list(bbox: Iterable[float]) -> list[float]:
    """Convert a bbox tuple into a JSON friendly list rounded to 2 decimals."""
    return [round(float(v), 2) for v in bbox]


#: The first bytes of a UTF-8 sequence, as they look after a latin-1 decode.
#: They are the signature of a text that was encoded twice.
_MOJIBAKE_LEADS = ("Ã", "Â", "â", "Î", "Ï", "Ð", "Ê", "Ñ", "ð", "Þ", "ã", "Å")

#: The number printed at the right margin of an equation.
_EQUATION_NUMBER = re.compile(r"\(\d{1,2}[a-z]?\)")


def repair_mojibake(text: str) -> str:
    """Repair text that was decoded as latin-1 instead of utf-8.

    Some PDF fonts ship without a usable ``ToUnicode`` map, and the text then
    arrives as its raw UTF-8 bytes read one byte at a time: the minus sign
    becomes "âˆ′" and the Greek capital theta becomes "Î".  The bytes are
    still there, so the string can be decoded again.

    The repair is only accepted when it actually cleans the string: a text
    that is already correct, or that uses a Latin-1 letter which happens to
    look like one of the markers, is left untouched.
    """
    if not text or not any(mark in text for mark in _MOJIBAKE_LEADS):
        return text
    for encoding in ("cp1252", "latin-1"):
        try:
            raw = text.encode(encoding)
        except UnicodeEncodeError:
            continue
        try:
            fixed = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        before = sum(text.count(mark) for mark in _MOJIBAKE_LEADS)
        after = sum(fixed.count(mark) for mark in _MOJIBAKE_LEADS)
        if after < before:
            return fixed
        return text
    return text


def _join_equation_fragments(a: Block, b: Block) -> bool:
    """True when two blocks are fragments of the same display equation.

    LaTeX typesets a display equation as several PDF text blocks when it
    holds a tall operator: the operator glyph and the limit underneath it
    have their own bounding box and PyMuPDF cuts the block there.  The two
    fragments are recognised by three things at once -- both are set in a
    math font, they cover a common printed row, and they overlap
    horizontally.  The horizontal overlap is what keeps two columns of a
    two-column page apart.
    """
    if not (a.is_equation and b.is_equation):
        return False
    if not (a.is_math and b.is_math):
        return False
    if b.x0 > a.x1 or a.x0 > b.x1:
        return False
    bands_a = _span_bands(a.spans)
    bands_b = _span_bands(b.spans)
    if not bands_a or not bands_b:
        return False
    return any(
        _band_overlap(x, y) >= _ROW_OVERLAP_RATIO
        for x in bands_a for y in bands_b
    )


def _rows_from_spans(spans: list[Span]) -> list[list[Span]]:
    """Group a set of spans into the printed rows they belong to."""
    rows: list[list[Span]] = []
    bands: list[tuple[float, float]] = []
    for span in sorted(spans, key=lambda s: (s.bbox[1], s.bbox[0])):
        band = (span.bbox[1], span.bbox[3])
        for i, other in enumerate(bands):
            if _band_overlap(band, other) >= _ROW_OVERLAP_RATIO:
                bands[i] = (min(other[0], band[0]), max(other[1], band[1]))
                rows[i].append(span)
                break
        else:
            rows.append([span])
            bands.append(band)
    rows.sort(key=lambda row: (min(s.bbox[1] for s in row),
                                min(s.bbox[0] for s in row)))
    for row in rows:
        row[:] = _sort_row(row)
    return rows


def _merge_equation_fragments(page_no: int, blocks: list[Block]) -> list[Block]:
    """Merge the blocks that are fragments of one display equation.

    Without this the reader shows equation (1) of the BERT paper in two
    halves -- "L1(U) = sum" on one card and "i log P(...) (1)" on the next --
    and the cropped image of each half is cut in the wrong place.
    """
    merged: list[Block] = []
    for block in blocks:
        if merged and _join_equation_fragments(merged[-1], block):
            prev = merged[-1]
            prev.spans = prev.spans + block.spans
            prev.bbox = (min(prev.x0, block.x0), min(prev.y0, block.y0),
                         max(prev.x1, block.x1), max(prev.y1, block.y1))
            prev.is_equation = True
            prev.block_type = BLOCK_FORMULA
            # The two halves form one row again, so the text is rebuilt from
            # the merged spans: the limit under the operator belongs to it.
            rows = _attach_limits(_rows_from_spans(prev.spans))
            prev.text = "\n".join(_row_texts(rows)).strip()
            continue
        merged.append(block)
    for index, block in enumerate(merged):
        block.block_index = index
    return merged


def extract_blocks(page: pymupdf.Page, page_no: int) -> list[Block]:
    """Extract positioned text blocks from a page.

    PyMuPDF's ``dict`` extraction keeps the original block/line/span hierarchy
    which is exactly what we need to rebuild the document structure.
    """
    blocks: list[Block] = []
    # ``TEXTFLAGS_DICT`` keeps ligatures, and a LaTeX PDF that renders "Jeff
    # Dean" with the ``ff`` ligature then yields "JeﬀDean" -- the space
    # behind the ligature is swallowed and the name is no longer a name.
    # Dropping ``TEXT_PRESERVE_LIGATURES`` expands the ligature back to
    # ASCII and restores the missing space.
    data = page.get_text("dict", flags=_DICT_FLAGS)
    for b in data.get("blocks", []):
        if b.get("type") != 0:  # 0 == text, 1 == image
            continue
        line_groups: list[list[Span]] = []
        for line in b.get("lines", []):
            line_spans: list[Span] = []
            for sp in line.get("spans", []):
                line_spans.append(Span(
                    text=repair_mojibake(sp.get("text", "")),
                    bbox=tuple(sp.get("bbox", (0, 0, 0, 0))),
                    size=float(sp.get("size", 0)),
                    flags=int(sp.get("flags", 0)),
                    font=str(sp.get("font", "")),
                ))
            if any(s.text.strip() for s in line_spans):
                line_groups.append(line_spans)
        if not line_groups:
            continue

        # A line that PyMuPDF cut because of a script is put back together, so
        # that "h" with its subscript and superscript is one row again.
        rows = _visual_rows(line_groups)
        texts = _row_texts(rows)

        # A block that mixes prose and an equation is split into its parts.
        pieces = split_equation_lines(page_no, rows, texts)
        if not pieces:
            spans = [span for row in rows for span in row]
            text = "\n".join(texts).strip()
            if not text:
                continue
            pieces = [Block(text=text,
                            bbox=(min(s.bbox[0] for s in spans),
                                  min(s.bbox[1] for s in spans),
                                  max(s.bbox[2] for s in spans),
                                  max(s.bbox[3] for s in spans)),
                            spans=spans, page_no=page_no,
                            is_equation=bool(rows) and all(
                                _line_is_equation(row) for row in rows))]

        blocks.extend(pieces)

    # An equation that PyMuPDF cut into two blocks is put back together.
    blocks = _merge_equation_fragments(page_no, blocks)
    for index, block in enumerate(blocks):
        block.block_index = index
    return blocks


# --------------------------------------------------------------------------- classification
def normalise_block_text(text: str) -> str:
    """Collapse the line breaks of a block into single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def is_caption_text(text: str) -> bool:
    """True for ``Figure 3: ...`` / ``Fig. 3 ...`` / ``Table 2: ...`` captions."""
    return bool(_CAPTION_RE.match(normalise_block_text(text)))


def inside_figure(bbox: tuple[float, float, float, float],
                  figures: list[tuple[float, float, float, float]]) -> bool:
    """True when a block belongs to one of the figure regions.

    Two signals are used: the centre of the block lies inside the figure, or
    the figure covers more than half of the block.
    """
    x0, y0, x1, y1 = bbox
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    for fx0, fy0, fx1, fy1 in figures:
        ix0, iy0 = max(x0, fx0), max(y0, fy0)
        ix1, iy1 = min(x1, fx1), min(y1, fy1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        if (ix1 - ix0) * (iy1 - iy0) / area >= FIGURE_OVERLAP_RATIO:
            return True
        if fx0 <= cx <= fx1 and fy0 <= cy <= fy1:
            return True
    return False


def _page_body_size(blocks: list[Block], page_height: float, fallback: float) -> float:
    """Median font size of the blocks that are *not* in the footnote zone."""
    sizes = [b.font_size for b in blocks
             if b.font_size > 0 and b.y0 < page_height * FOOTNOTE_ZONE]
    if not sizes:
        sizes = [b.font_size for b in blocks if b.font_size > 0]
    if not sizes:
        return fallback
    return statistics.median(sizes)


def _is_footnote(block: Block, page_height: float, body_size: float) -> bool:
    """A footnote sits in the bottom of the page and is printed smaller."""
    if page_height <= 0 or body_size <= 0:
        return False
    if block.y0 < page_height * FOOTNOTE_ZONE:
        return False
    size = block.font_size
    if size <= 0 or size > body_size * FOOTNOTE_SIZE_RATIO:
        return False
    text = normalise_block_text(block.text)
    # A footnote carries its marker ("9 The GLUE data set ...") or a URL.
    return bool(_FOOTNOTE_MARKER_RE.match(text) or _URL_RE.search(text))


def _is_folio(block: Block, page_height: float) -> bool:
    """True for a page number printed in the top or bottom margin.

    The folio is set in the body type, so the footnote rule -- which expects a
    smaller type -- does not catch it.  Left alone it ends up in the prose and,
    because it is a short line standing on its own, the heading detector reads
    it as a section title.
    """
    if page_height <= 0:
        return False
    if not _FOLIO_RE.match(normalise_block_text(block.text)):
        return False
    # Only a single line can be a page number.
    if block.height > page_height * 0.03:
        return False
    ratio = block.y0 / page_height
    return ratio >= FOLIO_ZONE or ratio <= HEADER_ZONE


def _is_header(block: Block, page_width: float) -> bool:
    """A running header hugs the left or the right page margin.

    The rotated arXiv stamp of the NeurIPS style is the typical example: it
    is a narrow block that sticks to the page edge.
    """
    if page_width <= 0:
        return False
    if block.width >= page_width * 0.12:
        return False
    if len(normalise_block_text(block.text)) > 120:
        return False
    return block.x0 <= page_width * 0.08 or block.x1 >= page_width * 0.92


def _is_prose(text: str) -> bool:
    """True for something that reads like a sentence rather than a fragment."""
    words = text.split()
    if len(words) >= 6:
        return True
    if len(words) >= 2 and text.endswith((".", "!", "?", "。", "！", "？")):
        return True
    return False


def classify_block(
    block: Block,
    *,
    page_width: float,
    page_height: float,
    body_size: float,
    figures: list[tuple[float, float, float, float]] | None = None,
    formulas: list[tuple[float, float, float, float]] | None = None,
) -> str:
    """Return the type of a single block."""
    text = normalise_block_text(block.text)
    if not text:
        return BLOCK_TEXT
    if _CAPTION_RE.match(text):
        return BLOCK_CAPTION
    if block.is_equation:
        # A display equation.  Its text is a flattened formula ("log P(u|u-k)")
        # and reads as noise in a paragraph, so the reader shows the cropped
        # image of the equation instead.
        return BLOCK_FORMULA
    if formulas and inside_figure(block.bbox, formulas):
        # An equation that already was a block of its own.
        return BLOCK_FORMULA
    if figures and inside_figure(block.bbox, figures):
        return BLOCK_FIGURE_LABEL
    if _is_folio(block, page_height):
        return BLOCK_HEADER
    if _is_footnote(block, page_height, body_size):
        return BLOCK_FOOTNOTE
    if _is_header(block, page_width):
        return BLOCK_HEADER
    if _is_prose(text):
        return BLOCK_BODY
    return BLOCK_TEXT


def classify_blocks(
    per_page_blocks: dict[int, list[Block]],
    *,
    page_sizes: dict[int, tuple[float, float]] | None = None,
    figure_bboxes: dict[int, list[tuple[float, float, float, float]]] | None = None,
    formula_bboxes: dict[int, list[tuple[float, float, float, float]]] | None = None,
    body_size: float = 0.0,
) -> None:
    """Tag every block of the document with its type (in place).

    ``figure_bboxes`` carries the figure regions per page -- they are
    produced by :func:`app.parsing.extractors.extract_figures`, so the figure
    pass has to run *before* this function.  ``formula_bboxes`` works the same
    way for the display equations.
    """
    page_sizes = page_sizes or {}
    figure_bboxes = figure_bboxes or {}
    formula_bboxes = formula_bboxes or {}
    for page_no, blocks in per_page_blocks.items():
        width, height = page_sizes.get(page_no, (612.0, 792.0))
        page_body = _page_body_size(blocks, height, body_size)
        for b in blocks:
            b.block_type = classify_block(
                b, page_width=width, page_height=height, body_size=page_body,
                figures=figure_bboxes.get(page_no, []),
                formulas=formula_bboxes.get(page_no, []),
            )


def detect_columns(blocks: list[Block], page_width: float) -> int:
    """Return the number of text columns on a page (1 or 2).

    Strategy: cluster the left edges of the blocks.  Two-column academic
    layouts produce two well separated clusters of left edges; single column
    layouts produce one dominant cluster.  We additionally require that the
    clusters are balanced enough to be real columns and that the gap is
    significant compared to the page width.
    """
    if len(blocks) < 4:
        return 1

    lefts = sorted(b.x0 for b in blocks if b.width > 5)
    if not lefts:
        return 1

    # Simple 1-D clustering with a tolerance of 18pt.
    clusters: list[list[float]] = [[lefts[0]]]
    for x in lefts[1:]:
        if x - clusters[-1][-1] <= 18:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    # Keep clusters that carry a meaningful amount of text.
    weighted = [(c, len(c)) for c in clusters]
    weighted.sort(key=lambda t: -t[1])
    if len(weighted) < 2:
        return 1

    left_cluster, right_cluster = weighted[0][0], weighted[1][0]
    left_edge = statistics.median(left_cluster)
    right_edge = statistics.median(right_cluster)
    gap = right_edge - left_edge

    if gap < COLUMN_GAP:
        return 1
    # The right column must start beyond the middle of the text area.
    if right_edge < page_width * 0.45:
        return 1
    # Both columns need at least 20% of the blocks, otherwise it is likely a
    # table or a figure caption rather than a second column.
    total = len(lefts)
    if len(left_cluster) / total < 0.2 or len(right_cluster) / total < 0.2:
        return 1
    return 2


def assign_columns(blocks: list[Block], page_width: float, n_columns: int) -> None:
    """Tag every block with its column index using a vertical gap model."""
    if n_columns <= 1:
        for b in blocks:
            b.column = 0
        return

    # The split point is the middle of the largest empty vertical band.
    mid = page_width / 2
    candidates = sorted(b.x0 for b in blocks)
    split = mid
    best_gap = 0.0
    for a, b in zip(candidates, candidates[1:]):
        if b - a > best_gap and a < mid < b:
            best_gap = b - a
            split = (a + b) / 2
    for blk in blocks:
        blk.column = 0 if (blk.x0 + blk.x1) / 2 < split else 1


def reading_order(blocks: list[Block]) -> list[Block]:
    """Sort blocks into human reading order.

    For single column pages this is plain top-to-bottom.  For double column
    pages we read the left column completely before the right one, which is how
    a human reads a paper.
    """
    if not blocks:
        return []
    n_columns = max(b.column for b in blocks) + 1
    if n_columns <= 1:
        return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))

    ordered: list[Block] = []
    for col in range(n_columns):
        col_blocks = [b for b in blocks if b.column == col]
        col_blocks.sort(key=lambda b: (round(b.y0, 1), b.x0))
        ordered.extend(col_blocks)
    return ordered


def page_body_font_size(blocks: list[Block]) -> float:
    """Estimate the dominant body-text font size of a document.

    Used to decide whether a heading is "larger" than the surrounding text.
    """
    sizes = [b.font_size for b in blocks if b.text and len(b.text) > 40 and b.font_size > 0]
    if not sizes:
        sizes = [b.font_size for b in blocks if b.font_size > 0]
    if not sizes:
        return 10.0
    return statistics.median(sizes)


def join_blocks(blocks: list[Block], separator: str = " ") -> str:
    """Concatenate blocks into a single plain-text string."""
    return separator.join(b.text.replace("\n", " ").strip() for b in blocks if b.text.strip())


def detect_page_is_scanned(page: pymupdf.Page) -> bool:
    """A page without a usable text layer is treated as a scan."""
    text = page.get_text("text").strip()
    if len(text) > 200:
        return False
    # A page with only a couple of glyphs (page numbers) is still "scanned".
    images = page.get_images(full=True)
    drawings = page.get_drawings()
    return len(text) < 80 and (len(images) > 0 or len(drawings) > 3)


#: The resolution at which the page images of the library are rendered.  A
#: bbox in PDF points maps to the page image by ``PAGE_IMAGE_DPI / 72``.
PAGE_IMAGE_DPI = 130


def render_page_png(page: pymupdf.Page, dpi: int = 150) -> bytes:
    """Rasterise a page to PNG bytes (used by the VLM for OCR / formula reading)."""
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return pix.tobytes("png")


def crop_page_png(page: pymupdf.Page, bbox: tuple[float, float, float, float],
                  dpi: int = 220) -> bytes:
    """Rasterise a rectangular region of a page (formula / figure crops)."""
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    rect = pymupdf.Rect(*bbox)
    # Keep a small margin so that superscripts are not clipped.
    rect = pymupdf.Rect(rect.x0 - 3, rect.y0 - 3, rect.x1 + 3, rect.y1 + 3) & page.rect
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    return pix.tobytes("png")


def page_metadata(page: pymupdf.Page) -> dict[str, Any]:
    """Return basic geometry of a page."""
    rect = page.rect
    return {"width": rect.width, "height": rect.height, "rotation": page.rotation}
