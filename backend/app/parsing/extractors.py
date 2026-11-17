"""Table, figure and formula extraction.

Tables are recovered structurally with PyMuPDF's table finder.
Figures are recovered as cropped page regions (rasterised) so that even vector
drawings (matplotlib plots, TikZ pictures) can be shown in the reader.
Formulas are *detected* geometrically and transcribed to LaTeX by the
vision-language model — the model is a genuine multimodal model, so it can read
the rendered equation directly.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from typing import Any

import pymupdf

from app.parsing.layout import Block, PAGE_IMAGE_DPI, crop_page_png, render_page_png

# --------------------------------------------------------------------------- caption patterns
_FIGURE_CAPTION = re.compile(r"^\s*(figure|fig\.?)\s*(\d+)\s*[.:：]?\s*(.*)$", re.IGNORECASE)
_TABLE_CAPTION = re.compile(r"^\s*(table|tab\.?)\s*(\d+)\s*[.:：]?\s*(.*)$", re.IGNORECASE)
_EQUATION_CAPTION = re.compile(r"^\s*\(?\d+\)?\s*$")

# --------------------------------------------------------------------------- drawing geometry
# Smallest side (in PDF points) a drawing must have to be considered at all.
# Hair lines - the rules of a table or the rule under a running header - are
# below it and are therefore ignored.
_MIN_DRAWING_SIDE = 3.0
# A drawing that is already this big is a figure on its own.
_BIG_RECT_WIDTH = 70.0
_BIG_RECT_HEIGHT = 55.0
# A "box" of a TikZ style picture.
_BOX_MIN_SIDE = 8.0
_BOX_MIN_AREA = 150.0
# A graphic region needs at least this many boxes ...
_REGION_MIN_BOXES = 3
# ... that span at least this fraction of the page width ...
_REGION_MIN_WIDTH_RATIO = 0.25
# ... and are at least this tall ...
_REGION_MIN_HEIGHT = 30.0
# ... and do not swallow the whole page.
_REGION_MAX_AREA_RATIO = 0.45
# Maximum gap between two boxes of the same picture.
_REGION_MAX_GAP = 60.0


@dataclass(slots=True)
class ExtractedTable:
    """A table with its caption and serialised content."""

    page_no: int
    bbox: list[float]
    markdown: str = ""
    csv_text: str = ""
    caption: str = ""
    n_rows: int = 0
    n_cols: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"page_no": self.page_no, "bbox": self.bbox, "markdown": self.markdown,
                "csv": self.csv_text, "caption": self.caption,
                "n_rows": self.n_rows, "n_cols": self.n_cols}


@dataclass(slots=True)
class ExtractedFigure:
    """A figure / chart region."""

    page_no: int
    bbox: list[float]
    image_path: str = ""
    caption: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"page_no": self.page_no, "bbox": self.bbox, "image_path": self.image_path,
                "caption": self.caption, "description": self.description}


@dataclass(slots=True)
class ExtractedFormula:
    """A formula region (LaTeX is filled in by the VLM transcriber)."""

    page_no: int
    bbox: list[float]
    raw_text: str = ""
    latex: str = ""
    image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"page_no": self.page_no, "bbox": self.bbox, "raw_text": self.raw_text,
                "latex": self.latex, "image_path": self.image_path}


# --------------------------------------------------------------------------- tables
#: The rules of a table are horizontal lines.  In a ``booktabs`` table they are
#: drawn as thin filled bars, which the line strategy of PyMuPDF does not see.
_RULE_MIN_WIDTH = 40.0
_RULE_MAX_THICKNESS = 2.0


def _horizontal_rules(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    """The horizontal rules of a page, as ``(y, x0, x1)``.

    Both the stroked lines and the thin filled bars are collected: the second
    kind is what a ``booktabs`` table uses for its rules.
    """
    rules: set[tuple[float, float, float]] = set()
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    for drawing in drawings:
        for item in drawing.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) <= 1.0 and abs(p2.x - p1.x) >= _RULE_MIN_WIDTH:
                    rules.add((round(p1.y, 1), round(min(p1.x, p2.x), 1),
                               round(max(p1.x, p2.x), 1)))
            elif item[0] == "re":
                rect = item[1]
                if rect.height <= _RULE_MAX_THICKNESS and rect.width >= _RULE_MIN_WIDTH:
                    rules.add((round(rect.y0, 1), round(rect.x0, 1),
                               round(rect.x1, 1)))
    return sorted(rules)


def _rule_regions(
    page: pymupdf.Page,
    captions: list[tuple[str, str, tuple[float, float, float, float]]],
) -> list[tuple[float, float, float, float]]:
    """The regions of the page that are bounded by horizontal rules.

    The rules that share their horizontal extent belong to the same table.  A
    ``Table N:`` caption that stands between two rules starts a new table.
    """
    groups: dict[tuple[float, float], list[float]] = {}
    for y, x0, x1 in _horizontal_rules(page):
        key = next((k for k in groups
                    if abs(k[0] - x0) <= 5 and abs(k[1] - x1) <= 5), None)
        if key is None:
            key = (x0, x1)
            groups[key] = []
        groups[key].append(y)

    page_height = page.rect.height
    regions: list[tuple[float, float, float, float]] = []
    for (x0, x1), ys in groups.items():
        ys = sorted(ys)
        if len(ys) < 2:
            continue
        # A table caption between two rules starts a new table.
        cuts = [0]
        for i in range(len(ys) - 1):
            low, high = ys[i], ys[i + 1]
            if any(kind == "table" and low < bbox[1] < high
                   for kind, _text, bbox in captions):
                cuts.append(i + 1)
        cuts.append(len(ys))
        for start, end in zip(cuts, cuts[1:]):
            chunk = ys[start:end]
            if len(chunk) < 2:
                continue
            top, bottom = chunk[0], chunk[-1]
            # The rule that separates the footnote area from the text is not a
            # table, and a region that covers the whole page is an artefact.
            if top >= page_height * 0.72 or bottom - top > page_height * 0.8:
                continue
            regions.append((x0, top, x1, bottom))
    return regions


def _find_tables(page: pymupdf.Page) -> tuple[list[Any], str]:
    """Find the tables of a page that are ruled on all sides.

    The ``text`` strategy is not used here: it infers the columns from the
    words, so a page of running text looks like a table with one column per
    word.  A table without vertical rules is found by :func:`_rule_regions`
    instead, which looks at the rules of the page.
    """
    try:
        finder = page.find_tables()
        tables = list(getattr(finder, "tables", []) or [])
    except Exception:
        tables = []
    return tables, "lines"


def _looks_like_a_table(rows: list[list[str]]) -> bool:
    """Guard for the text strategy: prose must not be mistaken for a table.

    The text strategy splits a paragraph into one column per word, so a page
    of running text looks like a table with twenty columns.  A real table is
    a small grid whose cells are short and mostly filled.
    """
    width = max(len(row) for row in rows)
    if width < 2 or width > 24:
        return False
    cells = [c for row in rows for c in row]
    filled = [c for c in cells if c]
    if len(filled) < 4:
        return False
    # A table is a grid: most of its cells carry a value.
    if len(filled) / max(1, len(cells)) < 0.15:
        return False
    # ... and the cells are short (numbers, labels), not paragraphs.
    mean_len = sum(len(c) for c in filled) / len(filled)
    return mean_len <= 45


def _read_grid(page: pymupdf.Page,
               bbox: tuple[float, float, float, float]) -> list[list[str]]:
    """Read the cells of the table that lives in ``bbox``.

    The columns are inferred from the *gutters*: the x-ranges that no word
    crosses anywhere in the table.  A cell whose text wraps stays inside its
    own column, so the gutter between two columns stays empty, while a wrapped
    word never opens a gutter of its own.

    ``find_tables(strategy="text")`` was used before and cannot do this: it
    treats the gap between two words as a column border, so "per Layer" came
    back as "p | er Layer" and "Sequential Maximum Path" as
    "Sequential Maxim | um Path".
    """
    rect = pymupdf.Rect(bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2) & page.rect
    try:
        words = page.get_text("words", clip=rect)
    except Exception:
        return []
    words = [w for w in words if (w[4] or "").strip()]
    if len(words) < 4:
        return []

    # ---- columns: maximal x-ranges with no word on any line ----------------
    spans = sorted((float(w[0]), float(w[2])) for w in words)
    merged: list[list[float]] = []
    for x0, x1 in spans:
        if merged and x0 <= merged[-1][1] + 1.5:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    left, right = merged[0][0], merged[-1][1]
    # a gutter narrower than this is word spacing, not a column border
    min_gutter = float(os.environ.get("SCIAGENT_MIN_GUTTER", "0")) or max(5.0, (right - left) * 0.018)
    edges = [left]
    for prev, cur in zip(merged, merged[1:]):
        if cur[0] - prev[1] >= min_gutter:
            edges.append((prev[1] + cur[0]) / 2)
    edges.append(right)
    if len(edges) < 2:
        return []
    width = len(edges) - 1

    # ---- lines: cluster the words by their vertical centre ------------------
    # The tolerance is generous on purpose: a formula is set with superscripts
    # and subscripts whose own boxes sit on a different baseline, and they
    # belong to the same line as the base text.
    lines: list[list[tuple[float, float, str]]] = []
    current: list[tuple[float, float, str]] = []
    last_y = None
    for w in sorted(words, key=lambda w: (round((w[1] + w[3]) / 2, 1), w[0])):
        centre = (float(w[1]) + float(w[3])) / 2
        height = max(1.0, float(w[3]) - float(w[1]))
        if last_y is not None and abs(centre - last_y) > height * 0.85:
            lines.append(sorted(current, key=lambda item: item[0]))
            current = []
        current.append((float(w[0]), float(w[2]), (w[4] or "").strip()))
        last_y = centre
    if current:
        lines.append(sorted(current, key=lambda item: item[0]))

    # ---- fill the grid ------------------------------------------------------
    grid: list[list[str]] = []
    for line in lines:
        cells = [""] * width
        for x0, x1, text in line:
            centre = (x0 + x1) / 2
            for index in range(width):
                if edges[index] <= centre <= edges[index + 1]:
                    cells[index] = (cells[index] + " " + text).strip()
                    break
        grid.append(cells)

    # ---- merge the wrapped continuation of a cell into the row above --------
    merged_rows: list[list[str]] = []
    for row in grid:
        filled = [i for i, c in enumerate(row) if c]
        if merged_rows and len(filled) == 1:
            above = merged_rows[-1]
            index = filled[0]
            if above[index]:
                above[index] = (above[index] + " " + row[index]).strip()
                continue
        merged_rows.append(row)
    return [r for r in merged_rows if any(c for c in r)]


def _bbox_overlap(a: tuple[float, float, float, float],
                  b: tuple[float, float, float, float]) -> float:
    """Intersection over the smaller area of two boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def extract_tables(page: pymupdf.Page, page_no: int,
                   blocks: list[Block] | None = None,
                   exclude: list[tuple[float, float, float, float]] | None = None
                   ) -> list[ExtractedTable]:
    """Detect the tables of a page and serialise them to Markdown + CSV.

    ``exclude`` holds the regions of the page that hold a picture: the grid of
    a chart and the frame of a legend look like a table to the ruling-line
    strategy, and a table that sits inside a picture is not a table.
    """
    out: list[ExtractedTable] = []
    captions = _collect_captions(blocks or [])
    excluded = list(exclude or [])
    taken: list[tuple[float, float, float, float]] = []

    # (a) tables that are ruled on all sides
    ruled, _strategy = _find_tables(page)
    for tbl in ruled:
        bbox = tuple(float(v) for v in tbl.bbox)
        if any(_overlap_ratio(bbox, region) > 0.5 for region in excluded):
            continue
        try:
            data = tbl.extract()
        except Exception:
            continue
        rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in data]
        rows = [r for r in rows if any(c for c in r)]
        if len(rows) < 2:
            continue
        width = max(len(r) for r in rows)
        if width < 1 or width > 40:
            continue
        caption, _ = _match_caption(captions, bbox, kind="table", above=True)
        out.append(ExtractedTable(page_no=page_no, bbox=[round(v, 2) for v in bbox],
                                  markdown=_rows_to_markdown(rows),
                                  csv_text=_rows_to_csv(rows), caption=caption,
                                  n_rows=len(rows), n_cols=width))
        taken.append(bbox)

    # (b) booktabs tables: the horizontal rules tell where they are
    for bbox in _rule_regions(page, captions):
        if any(_overlap_ratio(bbox, region) > 0.5 for region in excluded):
            continue
        if any(_bbox_overlap(bbox, other) > 0.3 for other in taken):
            continue
        rows = _read_grid(page, bbox)
        if len(rows) < 2 or not _looks_like_a_table(rows):
            continue
        width = max(len(r) for r in rows)
        caption, _ = _match_caption(captions, bbox, kind="table", above=True)
        out.append(ExtractedTable(page_no=page_no, bbox=[round(v, 2) for v in bbox],
                                  markdown=_rows_to_markdown(rows),
                                  csv_text=_rows_to_csv(rows), caption=caption,
                                  n_rows=len(rows), n_cols=width))
        taken.append(bbox)
    return out


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Render a table as GitHub flavoured Markdown."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    header = padded[0]
    body = padded[1:]
    lines = ["| " + " | ".join(_md_escape(c) for c in header) + " |",
             "| " + " | ".join("---" for _ in range(width)) + " |"]
    for row in body:
        lines.append("| " + " | ".join(_md_escape(c) for c in row) + " |")
    return "\n".join(lines)


def _md_escape(cell: str) -> str:
    return cell.replace("|", "\\|").replace("\n", " ").strip()


def _rows_to_csv(rows: list[list[str]]) -> str:
    import csv
    import io as _io

    buf = _io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


# --------------------------------------------------------------------------- figures
def figure_regions(page: pymupdf.Page,
                   blocks: list[Block]) -> list[tuple[float, float, float, float]]:
    """The regions of a page that hold a picture.

    Two kinds are recognised: a raster image that is embedded in the page and
    a cluster of vector drawings.  The regions are needed before the tables
    are read, because the grid of a chart and the frame of a legend look
    exactly like a table to the ruling-line strategy.
    """
    page_area = page.rect.width * page.rect.height
    regions: list[tuple[float, float, float, float]] = []
    seen_rects: list[tuple[float, float, float, float]] = []

    # (a) embedded raster images -----------------------------------------
    try:
        images = page.get_images(full=True)
    except Exception:
        images = []
    for info in images:
        xref = info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            if rect.width < 60 or rect.height < 50:
                continue
            if rect.get_area() < 4000:
                continue
            # A picture that covers most of the page is the page background,
            # not a figure.
            if rect.get_area() > 0.6 * page_area:
                continue
            seen_rects.append((rect.x0, rect.y0, rect.x1, rect.y1))

    # (b) vector drawing clusters ----------------------------------------
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    cluster = _cluster_drawings(drawings, page.rect.width, page.rect.height)
    for rect in cluster:
        if any(_overlap_ratio(rect, r) > 0.6 for r in seen_rects):
            continue
        seen_rects.append(rect)

    regions.extend(seen_rects)
    return regions


def extract_figures(
    page: pymupdf.Page,
    page_no: int,
    blocks: list[Block],
    out_dir: str,
    paper_id: str,
    *,
    max_figures: int = 12,
    exclude: list[tuple[float, float, float, float]] | None = None,
) -> list[ExtractedFigure]:
    """Extract figure regions.

    We look at (a) raster images embedded in the page and (b) vector drawing
    clusters, then attach the nearest ``Figure N:`` caption.
    """
    figures: list[ExtractedFigure] = []
    captions = _collect_captions(blocks)
    excluded = list(exclude or [])

    seen_rects = [r for r in figure_regions(page, blocks)
                  if not _hits_excluded(r, excluded)]

    # A figure that is drawn as several panels (a "left" and a "right" half, or
    # a grid of plots) is several regions on the page but one figure in the
    # paper.  The caption tells them apart from two unrelated figures that
    # stand side by side.
    page_area = page.rect.width * page.rect.height
    panels: list[tuple[float, float, float, float]] = []
    panel_captions: list[str] = []
    for rect in seen_rects[:max_figures]:
        caption, _no = _match_caption(captions, rect, kind="figure")
        merged = False
        for i, existing in enumerate(panels):
            if not caption or panel_captions[i] != caption:
                continue
            union = (min(existing[0], rect[0]), min(existing[1], rect[1]),
                     max(existing[2], rect[2]), max(existing[3], rect[3]))
            if (union[2] - union[0]) * (union[3] - union[1]) > 0.5 * page_area:
                continue
            panels[i] = union
            merged = True
            break
        if not merged:
            panels.append(rect)
            panel_captions.append(caption)

    for i, bbox in enumerate(panels):
        try:
            png = crop_page_png(page, bbox, dpi=170)
        except Exception:
            continue
        fname = f"{paper_id}_p{page_no}_fig{i}.png"
        path = f"{out_dir}/{fname}"
        try:
            with open(path, "wb") as fh:
                fh.write(png)
        except OSError:
            path = ""
        caption, caption_no = _match_caption(captions, bbox, kind="figure")
        figures.append(ExtractedFigure(page_no=page_no,
                                       bbox=[round(v, 2) for v in bbox],
                                       image_path=path,
                                       caption=caption))
    return figures


def _same_band(a: tuple[float, float, float, float],
               b: tuple[float, float, float, float]) -> bool:
    """True when two regions stand on the same line, side by side."""
    return abs(a[1] - b[1]) <= 30 and abs(a[3] - b[3]) <= 30

def _hits_excluded(rect: tuple[float, float, float, float],
                   excluded: list[tuple[float, float, float, float]]) -> bool:
    """True when a region overlaps one of the excluded areas (a table)."""
    return any(_overlap_ratio(rect, other) > 0.25 for other in excluded)


def _cluster_drawings(drawings: list[Any], page_width: float = 595.0,
                      page_height: float = 842.0) -> list[tuple[float, float, float, float]]:
    """Group vector drawings into figure-sized bounding boxes.

    Two kinds of regions are produced:

    * **big rectangles** – a drawing that is already figure sized (a plot
      frame, a box around a picture).  Overlapping ones are merged.
    * **graphic regions** – a picture that is drawn as many small boxes and
      arrows (a TikZ diagram) has no big rectangle at all.  Such a cluster is
      recognised by the *shape* of the region: at least three box-like
      drawings that together form a wide band.  Hair lines (the rules of a
      table or the rule under a running header) are ignored, so a text page
      never becomes a figure.
    """
    page_area = max(1.0, page_width * page_height)
    rects: list[tuple[float, float, float, float]] = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        if r.width < _MIN_DRAWING_SIDE or r.height < _MIN_DRAWING_SIDE:
            continue
        if r.get_area() > 0.85 * page_area:
            continue
        rects.append((r.x0, r.y0, r.x1, r.y1))

    big = [r for r in rects
           if (r[2] - r[0]) >= _BIG_RECT_WIDTH and (r[3] - r[1]) >= _BIG_RECT_HEIGHT]
    small = [r for r in rects if r not in big]

    clusters = _merge_overlapping(big)
    regions = _graphic_regions(small, page_width, page_height)

    out: list[tuple[float, float, float, float]] = []
    for rect in clusters + regions:
        if any(_overlap_ratio(rect, existing) > 0.5 for existing in out):
            continue
        out.append(rect)
    return out


def _merge_overlapping(rects: list[tuple[float, float, float, float]]
                       ) -> list[tuple[float, float, float, float]]:
    """Merge the rectangles that overlap each other."""
    merged: list[list[float]] = []
    for rect in rects:
        placed = False
        for m in merged:
            if _overlap_ratio((m[0], m[1], m[2], m[3]), rect) > 0.25:
                m[0] = min(m[0], rect[0]); m[1] = min(m[1], rect[1])
                m[2] = max(m[2], rect[2]); m[3] = max(m[3], rect[3])
                placed = True
                break
        if not placed:
            merged.append(list(rect))
    return [(m[0], m[1], m[2], m[3]) for m in merged]


def _graphic_regions(rects: list[tuple[float, float, float, float]],
                     page_width: float, page_height: float
                     ) -> list[tuple[float, float, float, float]]:
    """Merge small drawings into a figure-sized band (TikZ style pictures)."""
    boxes = [r for r in rects
             if min(r[2] - r[0], r[3] - r[1]) >= _BOX_MIN_SIDE
             and (r[2] - r[0]) * (r[3] - r[1]) >= _BOX_MIN_AREA]
    if len(boxes) < _REGION_MIN_BOXES:
        return []

    clusters: list[list[float]] = []
    for rect in boxes:
        for c in clusters:
            if _close_enough((c[0], c[1], c[2], c[3]), rect):
                c[0] = min(c[0], rect[0]); c[1] = min(c[1], rect[1])
                c[2] = max(c[2], rect[2]); c[3] = max(c[3], rect[3])
                c[4] += 1
                break
        else:
            clusters.append([rect[0], rect[1], rect[2], rect[3], 1])

    out: list[tuple[float, float, float, float]] = []
    for x0, y0, x1, y1, count in clusters:
        width, height = x1 - x0, y1 - y0
        if count < _REGION_MIN_BOXES:
            continue
        if width < page_width * _REGION_MIN_WIDTH_RATIO:
            continue
        if height < _REGION_MIN_HEIGHT:
            continue
        if width * height > _REGION_MAX_AREA_RATIO * page_width * page_height:
            continue
        out.append((x0, y0, x1, y1))
    return out


def _close_enough(a: tuple[float, float, float, float],
                  b: tuple[float, float, float, float]) -> bool:
    """True when two drawings belong to the same picture."""
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return dx <= _REGION_MAX_GAP and dy <= _REGION_MAX_GAP


def _overlap_ratio(a: tuple[float, float, float, float],
                   b: tuple[float, float, float, float]) -> float:
    """Intersection over the smaller area."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return inter / min(area_a, area_b)


def _collect_captions(blocks: list[Block]) -> list[tuple[str, str, tuple[float, float, float, float]]]:
    """Collect ``Figure N:`` / ``Table N:`` captions with their position."""
    out: list[tuple[str, str, tuple[float, float, float, float]]] = []
    for b in blocks:
        text = re.sub(r"\s+", " ", b.text).strip()
        m = _FIGURE_CAPTION.match(text)
        if m:
            out.append(("figure", text, b.bbox))
            continue
        m = _TABLE_CAPTION.match(text)
        if m:
            out.append(("table", text, b.bbox))
    return out


def _match_caption(captions, bbox, *, kind: str, above: bool = False) -> tuple[str, int]:
    """Find the caption that belongs to a figure/table region.

    A table is captioned *above* it and a figure *below* it (LaTeX
    convention).  The other side is only searched when the expected side has
    no caption at all: a page can hold two tables, and the caption of the
    second one is closer to the bottom of the first one than its own caption
    is to its top.
    """
    def search(want_above: bool) -> tuple[str, int]:
        best_text, best_no, best_dist = "", 0, 1e9
        for ckind, ctext, cbbox in captions:
            if ckind != kind:
                continue
            if want_above:
                up = bbox[1] - cbbox[3]
                dist = abs(up) if -30 <= up <= 220 else None
            else:
                dy = cbbox[1] - bbox[3]
                dist = abs(dy) if -30 <= dy <= 220 else None
            if dist is None or dist >= best_dist:
                continue
            best_dist = dist
            best_text = ctext
            m = re.match(r"^\s*(?:figure|fig\.?|table|tab\.?)\s*(\d+)", ctext, re.IGNORECASE)
            best_no = int(m.group(1)) if m else 0
        return best_text, best_no

    first_above = above
    text, no = search(first_above)
    if text:
        return text, no
    return search(not first_above)


# --------------------------------------------------------------------------- formulas
def detect_formulas(page: pymupdf.Page, page_no: int, blocks: list[Block]) -> list[ExtractedFormula]:
    """Detect formula regions.

    Formulas are recognised by three signals:
    * the presence of math fonts (Computer Modern ``cmmi``/``cmsy``/``msam``),
    * a high density of mathematical symbols,
    * an isolated line that is not a normal sentence.
    """
    out: list[ExtractedFormula] = []
    for b in blocks:
        text = b.text.strip()
        if not text or len(text) < 3:
            continue
        math_like = b.is_math
        symbol_ratio = _symbol_ratio(text)
        isolated = len(text.split()) <= 40

        # A block that was split out of a paragraph because it is an equation
        # is an equation by construction.
        if not (b.is_equation or math_like or symbol_ratio > 0.32):
            continue
        if not b.is_equation and not isolated and not math_like:
            continue
        # Skip pure punctuation / page numbers.
        if re.fullmatch(r"[\W\d\s]{0,6}", text):
            continue
        out.append(ExtractedFormula(page_no=page_no,
                                    bbox=[round(v, 2) for v in b.bbox],
                                    raw_text=text))
    return out


def _symbol_ratio(text: str) -> float:
    """Fraction of characters that are mathematical operators."""
    math_chars = set("=+−-×÷±∑∏∫∮√∞≈≠≤≥∂∇∈∉⊂⊆∪∩→←↔⇒⇔∀∃∅αβγδεθλμπρστφψωΓΔΘΛΞΠΦΨΩ")
    letters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if not text:
        return 0.0
    ops = sum(1 for c in text if c in math_chars)
    alnum = sum(1 for c in text if c in letters or c.isdigit())
    denom = ops + alnum
    return ops / denom if denom else 0.0


def render_formula_image(page: pymupdf.Page, formula: ExtractedFormula,
                          out_dir: str, paper_id: str, index: int) -> str:
    """Rasterise a formula region so the VLM can read it."""
    fname = f"{paper_id}_p{formula.page_no}_eq{index}.png"
    path = f"{out_dir}/{fname}"
    try:
        png = crop_page_png(page, tuple(formula.bbox), dpi=260)
        with open(path, "wb") as fh:
            fh.write(png)
        formula.image_path = path
    except Exception:
        pass
    return path


def save_page_image(page: pymupdf.Page, page_no: int, out_dir: str, paper_id: str,
                    dpi: int = PAGE_IMAGE_DPI) -> str:
    """Persist a rendered page (used by the reader and the OCR pipeline)."""
    fname = f"{paper_id}_page{page_no:04d}.png"
    path = f"{out_dir}/{fname}"
    try:
        with open(path, "wb") as fh:
            fh.write(render_page_png(page, dpi=dpi))
    except Exception:
        return ""
    return path


def figure_description_prompt(caption: str) -> str:
    """Prompt used by the VLM to describe a figure."""
    return (
        "Describe this academic figure in one dense paragraph. "
        "State: (1) what the axes show, (2) the key trend or comparison, "
        "(3) any numbers that are legible. "
        "Be precise and factual; do not invent values that are not visible."
        + (f"\nThe paper's caption is: {caption}" if caption else "")
    )
