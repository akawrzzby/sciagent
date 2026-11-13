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


def bbox_to_list(bbox: Iterable[float]) -> list[float]:
    """Convert a bbox tuple into a JSON friendly list rounded to 2 decimals."""
    return [round(float(v), 2) for v in bbox]


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
        spans: list[Span] = []
        lines: list[str] = []
        for line in b.get("lines", []):
            line_spans: list[Span] = []
            for sp in line.get("spans", []):
                line_spans.append(Span(
                    text=sp.get("text", ""),
                    bbox=tuple(sp.get("bbox", (0, 0, 0, 0))),
                    size=float(sp.get("size", 0)),
                    flags=int(sp.get("flags", 0)),
                    font=str(sp.get("font", "")),
                ))
            spans.extend(line_spans)
            lines.append("".join(s.text for s in line_spans))
        text = "\n".join(lines).strip()
        if not text:
            continue
        blocks.append(Block(text=text, bbox=tuple(b.get("bbox", (0, 0, 0, 0))),
                            spans=spans, page_no=page_no,
                            block_index=len(blocks)))
    return blocks


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
