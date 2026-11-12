"""The document parsing pipeline.

``parse_document`` turns a PDF file into the structured representation used by
every other module:

```
PDF ──▶ page blocks ──▶ column detection ──▶ reading order
                                    │
                                    ├─▶ heading tree (level 1/2/3)
                                    ├─▶ metadata (title/authors/affiliations/abstract)
                                    ├─▶ tables (Markdown + CSV)
                                    ├─▶ figures (cropped PNG + caption)
                                    ├─▶ formulas (region + VLM → LaTeX)
                                    ├─▶ references (structured entries)
                                    └─▶ chunks (for hybrid retrieval)
```

Scanned documents (no text layer) take a detour: every page is rasterised and
transcribed by the multimodal model, then the Markdown result is re-parsed with
the same heading heuristics.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import pymupdf

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.client import get_llm_client
from app.parsing import dedup, extractors, headings, layout, metadata, ocr, references

logger = get_logger(__name__)


@dataclass(slots=True)
class ParsedDocument:
    """Everything the parser learned about one PDF."""

    metadata: metadata.PaperMetadata
    sections: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    formulas: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)
    pages: list[dict[str, Any]] = field(default_factory=list)
    full_text: str = ""
    is_scanned: bool = False
    ocr_pages: int = 0
    page_count: int = 0
    warnings: list[str] = field(default_factory=list)
    parse_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "sections": self.sections,
            "blocks": self.blocks,
            "tables": self.tables,
            "figures": self.figures,
            "formulas": self.formulas,
            "references": self.references,
            "pages": self.pages,
            "is_scanned": self.is_scanned,
            "ocr_pages": self.ocr_pages,
            "page_count": self.page_count,
            "warnings": self.warnings,
            "parse_seconds": round(self.parse_seconds, 2),
        }


ProgressFn = Callable[[float, str], Any]


async def parse_document(
    path: str,
    *,
    paper_id: str = "",
    out_dir: str = "",
    do_ocr: bool = True,
    do_figures: bool = True,
    do_formulas: bool = True,
    transcribe_formulas: bool = True,
    progress: ProgressFn | None = None,
    session_id: str = "",
) -> ParsedDocument:
    """Parse ``path`` and return the structured document.

    ``progress(fraction, message)`` is called during the run so the UI can show
    a live progress bar.
    """
    settings = get_settings()
    out_dir = out_dir or str(settings.data_dir / "pages")
    os.makedirs(out_dir, exist_ok=True)
    paper_id = paper_id or os.path.splitext(os.path.basename(path))[0]

    t0 = time.perf_counter()
    doc = pymupdf.open(path)
    try:
        parsed = await _parse_text_document(
            doc, paper_id=paper_id, out_dir=out_dir, do_figures=do_figures,
            do_formulas=do_formulas, progress=progress,
        )

        # A document without a usable text layer is a scan -> OCR it.
        if parsed.is_scanned and do_ocr:
            if progress:
                await progress(0.15, "检测到扫描件，启动多模态 OCR …")
            parsed = await _parse_scanned_document(
                doc, paper_id=paper_id, out_dir=out_dir, progress=progress,
                session_id=session_id, do_figures=do_figures,
            )

        if transcribe_formulas and parsed.formulas:
            await _transcribe_formulas(parsed, path, paper_id, out_dir, progress)

        parsed.parse_seconds = time.perf_counter() - t0
        return parsed
    finally:
        doc.close()


# --------------------------------------------------------------------------- text path
async def _parse_text_document(
    doc: pymupdf.Document,
    *,
    paper_id: str,
    out_dir: str,
    do_figures: bool,
    do_formulas: bool,
    progress: ProgressFn | None,
) -> ParsedDocument:
    """Parse a PDF that has a text layer."""
    settings = get_settings()
    parsed = ParsedDocument(metadata=metadata.PaperMetadata())
    parsed.page_count = doc.page_count

    all_blocks: list[layout.Block] = []
    per_page_blocks: dict[int, list[layout.Block]] = {}

    total_pages = doc.page_count
    max_pages = settings.parse_max_pages or total_pages

    for index in range(min(total_pages, max_pages)):
        page = doc[index]
        page_no = index + 1
        blocks = layout.extract_blocks(page, page_no)
        n_cols = layout.detect_columns(blocks, page.rect.width)
        layout.assign_columns(blocks, page.rect.width, n_cols)
        per_page_blocks[page_no] = blocks
        all_blocks.extend(blocks)

        if progress and (index % 5 == 0 or index == min(total_pages, max_pages) - 1):
            await progress(0.05 + 0.35 * (index / max(1, min(total_pages, max_pages))),
                           f"解析页面 {page_no}/{min(total_pages, max_pages)}")

    if not all_blocks:
        parsed.is_scanned = True
        return parsed

    # A document with almost no text but with images is a scan.
    total_chars = sum(len(b.text) for b in all_blocks)
    if total_chars < 60 * len(per_page_blocks) and any(
            len(doc[i].get_images(full=True)) for i in range(min(3, doc.page_count))):
        parsed.is_scanned = True
        return parsed

    body_size = layout.page_body_font_size(all_blocks)

    # ---- metadata ------------------------------------------------------
    if progress:
        await progress(0.45, "抽取标题 / 作者 / 机构 / 摘要 …")
    first_page_no = min((b.page_no for b in all_blocks), default=1)
    first_page_text = "\n".join(b.text for b in all_blocks if b.page_no == first_page_no)
    page_width = doc[first_page_no - 1].rect.width if doc.page_count else 612.0
    meta = metadata.extract_metadata(
        all_blocks, body_size=body_size,
        filename=os.path.basename(doc.name),
        raw_text="\n".join(b.text for b in all_blocks),
        page_width=page_width,
    )
    metadata.infer_venue(meta, first_page_text)
    parsed.metadata = meta

    # ---- reading order + headings --------------------------------------
    if progress:
        await progress(0.5, "构建标题层级树 …")
    ordered_blocks: list[layout.Block] = []
    for page_no in sorted(per_page_blocks):
        ordered_blocks.extend(layout.reading_order(per_page_blocks[page_no]))

    candidates = headings.detect_headings(ordered_blocks, body_size)
    tree = headings.build_tree(candidates)
    flat = headings.flatten(tree)
    headings.assign_content(flat, ordered_blocks)

    parsed.sections = [
        {
            "title": n.title, "number": n.number, "level": n.level,
            "page_no": n.page_no, "bbox": n.bbox, "content": n.content,
            "order_index": n.order_index,
        }
        for n in flat
    ]

    # ---- content blocks -------------------------------------------------
    body_texts = {id(b) for b in ordered_blocks}
    parsed.blocks = [
        {
            "page_no": b.page_no, "block_type": "text",
            "bbox": layout.bbox_to_list(b.bbox), "content": b.text,
            "order_index": b.block_index,
        }
        for b in ordered_blocks
    ]

    # ---- tables / figures / formulas ------------------------------------
    for index in range(min(total_pages, max_pages)):
        page = doc[index]
        page_no = index + 1
        if do_figures:
            try:
                for fig in extractors.extract_figures(page, page_no, per_page_blocks[page_no],
                                                       out_dir, paper_id):
                    parsed.figures.append(fig.to_dict())
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("figure extraction failed on page %s: %s", page_no, exc)
        try:
            for tbl in extractors.extract_tables(page, page_no):
                parsed.tables.append(tbl.to_dict())
        except Exception as exc:
            logger.debug("table extraction failed on page %s: %s", page_no, exc)
        if do_formulas:
            try:
                for f in extractors.detect_formulas(page, page_no, per_page_blocks[page_no]):
                    parsed.formulas.append(f.to_dict())
            except Exception as exc:
                logger.debug("formula detection failed on page %s: %s", page_no, exc)

        # Render the page image for the reader (and later for the VLM).
        try:
            img = extractors.save_page_image(page, page_no, out_dir, paper_id, dpi=130)
            parsed.pages.append({"page_no": page_no, "image_path": img,
                                 "width": page.rect.width, "height": page.rect.height})
        except Exception:
            parsed.pages.append({"page_no": page_no, "image_path": ""})

    # ---- references -----------------------------------------------------
    if progress:
        await progress(0.7, "解析参考文献列表 …")
    refs = references.parse_references(ordered_blocks)
    parsed.references = [r.to_dict() for r in refs]

    # ---- full text ------------------------------------------------------
    parsed.full_text = "\n".join(b.text for b in ordered_blocks)
    return parsed


# --------------------------------------------------------------------------- scan path
async def _parse_scanned_document(
    doc: pymupdf.Document,
    *,
    paper_id: str,
    out_dir: str,
    progress: ProgressFn | None,
    session_id: str,
    do_figures: bool,
) -> ParsedDocument:
    """OCR a scanned document page by page and rebuild the structure."""
    settings = get_settings()
    client = get_llm_client()

    parsed = ParsedDocument(metadata=metadata.PaperMetadata())
    parsed.page_count = doc.page_count
    parsed.is_scanned = True

    # 1) rasterise every page --------------------------------------------
    png_pages: list[tuple[int, bytes]] = []
    for index in range(doc.page_count):
        page = doc[index]
        page_no = index + 1
        try:
            png = layout.render_page_png(page, dpi=settings.render_dpi)
        except Exception:
            continue
        png_pages.append((page_no, png))
        try:
            img_path = extractors.save_page_image(page, page_no, out_dir, paper_id, dpi=130)
            parsed.pages.append({"page_no": page_no, "image_path": img_path,
                                 "width": page.rect.width, "height": page.rect.height})
        except Exception:
            parsed.pages.append({"page_no": page_no, "image_path": ""})

    # 2) transcribe -------------------------------------------------------
    ocr_results = await ocr.ocr_document(png_pages, client=client, session_id=session_id)
    parsed.ocr_pages = sum(1 for r in ocr_results if r.ok)

    # 3) rebuild blocks / headings from the Markdown -----------------------
    #    The OCR output carries an explicit Markdown structure, so the heading
    #    tree is rebuilt from it instead of being re-inferred from typography
    #    (which a scan does not have).
    ordered_blocks: list[layout.Block] = []
    heading_candidates: list[headings.HeadingCandidate] = []
    y = 0.0
    for res in ocr_results:
        if not res.ok:
            parsed.warnings.append(f"第 {res.page_no} 页 OCR 失败: {res.error[:120]}")
            continue
        md = res.markdown
        if not md.strip():
            continue
        page_blocks, page_candidates = ocr.markdown_to_blocks(
            md, page_no=res.page_no, start_index=len(ordered_blocks), y_top=y,
        )
        ordered_blocks.extend(page_blocks)
        heading_candidates.extend(page_candidates)
        if page_blocks:
            last = page_blocks[-1]
            y = last.bbox[3] + 4

    # 4) headings ---------------------------------------------------------
    body_size = 10.0
    tree = headings.build_tree(heading_candidates)
    flat = headings.flatten(tree)
    headings.assign_content(flat, ordered_blocks)
    parsed.sections = [
        {"title": n.title, "number": n.number, "level": n.level, "page_no": n.page_no,
         "bbox": n.bbox, "content": n.content, "order_index": n.order_index}
        for n in flat
    ]

    parsed.blocks = [
        {"page_no": b.page_no, "block_type": "text",
         "bbox": layout.bbox_to_list(b.bbox), "content": b.text,
         "order_index": b.block_index}
        for b in ordered_blocks
    ]

    # 5) metadata from the OCR text --------------------------------------
    page_markdowns = [r.markdown for r in ocr_results if r.ok]
    full_text = "\n\n".join(page_markdowns)
    parsed.full_text = full_text
    meta = metadata.extract_metadata_from_markdown(page_markdowns, filename=doc.name)
    parsed.metadata = meta

    # 6) references -------------------------------------------------------
    refs = references.parse_references(ordered_blocks)
    parsed.references = [r.to_dict() for r in refs]

    # 7) figures (raster pages) ------------------------------------------
    if do_figures:
        for page_info in parsed.pages:
            path = page_info.get("image_path")
            if path:
                parsed.figures.append({
                    "page_no": page_info["page_no"], "bbox": [0, 0, 612, 792],
                    "image_path": path, "caption": "", "description": "",
                })
    return parsed


# --------------------------------------------------------------------------- formulas
async def _transcribe_formulas(
    parsed: ParsedDocument,
    path: str,
    paper_id: str,
    out_dir: str,
    progress: ProgressFn | None,
) -> None:
    """Ask the multimodal model to convert formula regions into LaTeX."""
    if not parsed.formulas:
        return
    client = get_llm_client()
    doc = pymupdf.open(path)
    try:
        limit = min(len(parsed.formulas), 40)
        for i, fdict in enumerate(parsed.formulas[:limit]):
            page_no = fdict["page_no"]
            page = doc[page_no - 1]
            formula = extractors.ExtractedFormula(page_no=page_no,
                                                   bbox=fdict["bbox"],
                                                   raw_text=fdict.get("raw_text", ""))
            extractors.render_formula_image(page, formula, out_dir, paper_id, i)
            if not formula.image_path:
                continue
            png = open(formula.image_path, "rb").read()
            messages = [
                {"role": "system", "content":
                    "You transcribe mathematical expressions from images into LaTeX. "
                    "Reply with the LaTeX only, wrapped in $...$. "
                    "If the image is not a formula, reply with an empty string."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Transcribe this formula into LaTeX."},
                    {"type": "image_url", "image_url": {"url": ocr.encode_image(png)}},
                ]},
            ]
            resp = await client.chat(messages, temperature=0.0, max_tokens=512,
                                     purpose="formula:latex")
            latex = resp.content.strip()
            latex = latex.strip("`").strip()
            if latex.startswith("$$") and latex.endswith("$$"):
                latex = latex[2:-2]
            elif latex.startswith("$") and latex.endswith("$"):
                latex = latex[1:-1]
            fdict["latex"] = latex.strip()
            fdict["image_path"] = formula.image_path
            if progress and i % 5 == 0:
                await progress(0.85 + 0.1 * (i / limit), f"公式 LaTeX 转写 {i + 1}/{limit}")
    finally:
        doc.close()


# --------------------------------------------------------------------------- fingerprint
def build_fingerprint(
    parsed: ParsedDocument,
    *,
    paper_id: str,
    path: str,
    embedding: Iterable[float] | None = None,
) -> dedup.PaperFingerprint:
    """Build the fingerprint used for duplicate detection."""
    import numpy as np

    fp = dedup.PaperFingerprint(
        paper_id=paper_id,
        title=parsed.metadata.title,
        arxiv_id=parsed.metadata.arxiv_id,
        version_label=parsed.metadata.version_label,
        file_hash=dedup.file_hash(path) if os.path.exists(path) else "",
        content_hash=dedup.content_hash(parsed.full_text),
        simhash=dedup.simhash(parsed.full_text),
        char_count=len(parsed.full_text),
        page_count=parsed.page_count,
        abstract=parsed.metadata.abstract,
        year=parsed.metadata.year,
    )
    if embedding is not None:
        fp.embedding = np.asarray(list(embedding), dtype=np.float32)
    return fp
