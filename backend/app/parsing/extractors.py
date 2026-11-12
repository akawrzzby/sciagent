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
import re
from dataclasses import dataclass, field
from typing import Any

import pymupdf

from app.parsing.layout import Block, crop_page_png, render_page_png

# --------------------------------------------------------------------------- caption patterns
_FIGURE_CAPTION = re.compile(r"^\s*(figure|fig\.?)\s*(\d+)\s*[.:：]?\s*(.*)$", re.IGNORECASE)
_TABLE_CAPTION = re.compile(r"^\s*(table|tab\.?)\s*(\d+)\s*[.:：]?\s*(.*)$", re.IGNORECASE)
_EQUATION_CAPTION = re.compile(r"^\s*\(?\d+\)?\s*$")


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
def extract_tables(page: pymupdf.Page, page_no: int) -> list[ExtractedTable]:
    """Detect tables on a page and serialise them to Markdown + CSV."""
    out: list[ExtractedTable] = []
    try:
        finder = page.find_tables()
    except Exception:
        return out

    for tbl in getattr(finder, "tables", []):
        try:
            data = tbl.extract()
        except Exception:
            continue
        if not data or len(data) < 2:
            continue
        # Very wide tables are usually layout artefacts.
        width = max(len(r) for r in data)
        if width < 2 or width > 40:
            continue
        rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in data]
        rows = [r for r in rows if any(c for c in r)]
        if len(rows) < 2:
            continue

        markdown = _rows_to_markdown(rows)
        csv_text = _rows_to_csv(rows)
        bbox = [round(v, 2) for v in list(tbl.bbox)]
        out.append(ExtractedTable(page_no=page_no, bbox=bbox, markdown=markdown,
                                  csv_text=csv_text,
                                  n_rows=len(rows), n_cols=width))
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
def extract_figures(
    page: pymupdf.Page,
    page_no: int,
    blocks: list[Block],
    out_dir: str,
    paper_id: str,
    *,
    max_figures: int = 12,
) -> list[ExtractedFigure]:
    """Extract figure regions.

    We look at (a) raster images embedded in the page and (b) vector drawing
    clusters, then attach the nearest ``Figure N:`` caption.
    """
    figures: list[ExtractedFigure] = []
    captions = _collect_captions(blocks)

    # (a) embedded raster images -----------------------------------------
    seen_rects: list[tuple[float, float, float, float]] = []
    for info in page.get_images(full=True):
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
            seen_rects.append((rect.x0, rect.y0, rect.x1, rect.y1))

    # (b) vector drawing clusters ----------------------------------------
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    cluster = _cluster_drawings(drawings)
    for rect in cluster:
        if any(_overlap_ratio(rect, r) > 0.6 for r in seen_rects):
            continue
        seen_rects.append(rect)

    for i, (x0, y0, x1, y1) in enumerate(seen_rects[:max_figures]):
        bbox = (x0, y0, x1, y1)
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


def _cluster_drawings(drawings: list[Any]) -> list[tuple[float, float, float, float]]:
    """Group vector drawings into figure-sized bounding boxes."""
    rects: list[tuple[float, float, float, float]] = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        if r.width < 70 or r.height < 55:
            continue
        if r.get_area() > 0.85 * (page_area := 595 * 842):
            continue
        rects.append((r.x0, r.y0, r.x1, r.y1))
    # Merge overlapping rectangles.
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


def _match_caption(captions, bbox, *, kind: str) -> tuple[str, int]:
    """Find the caption that belongs to a figure/table region."""
    best_text, best_no, best_dist = "", 0, 1e9
    for ckind, ctext, cbbox in captions:
        if ckind != kind:
            continue
        dy = cbbox[1] - bbox[3]          # caption below the figure
        if dy < -30 or dy > 220:
            continue
        dist = abs(dy)
        if dist < best_dist:
            best_dist = dist
            best_text = ctext
            m = re.match(r"^\s*(?:figure|fig\.?|table|tab\.?)\s*(\d+)", ctext, re.IGNORECASE)
            best_no = int(m.group(1)) if m else 0
    return best_text, best_no


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

        if not (math_like or symbol_ratio > 0.32):
            continue
        if not isolated and not math_like:
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
                    dpi: int = 130) -> str:
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
