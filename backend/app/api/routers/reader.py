"""Module 3 – layout-aware reading, citation navigation and bilingual translation."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.config import get_settings
from app.parsing import layout
from app.parsing.references import citation_context, find_intext_citations
from app.services import papers as paper_service
from app.services import translation as translation_service
from app.storage.db import session_scope
from app.storage.models import (
    Chunk,
    PaperFigure,
    PaperFormula,
    PaperReference,
    PaperTable,
)

router = APIRouter(prefix="/reader", tags=["reader"])


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    target_lang: str = "zh"


# --------------------------------------------------------------------------- pages
@router.get("/{paper_id}/pages")
async def pages(paper_id: str) -> dict:
    """The rendered page images of a paper."""
    structure = await paper_service.get_paper_structure(paper_id)
    if not structure:
        raise HTTPException(status_code=404, detail="paper not found")

    settings = get_settings()
    page_dir = settings.data_dir / "pages"
    result = []
    for i in range(1, structure.get("page_count", 0) + 1):
        prefix = f"{paper_id}_page"
        found = ""
        if page_dir.exists():
            for f in page_dir.glob(f"{prefix}{i:04d}.png"):
                found = f"/api/reader/{paper_id}/page/{i}/image"
                break
        result.append({"page_no": i, "image_url": found})
    return {"paper_id": paper_id, "pages": result,
            "page_count": structure.get("page_count", 0)}


@router.get("/{paper_id}/page/{page_no}/image")
async def page_image(paper_id: str, page_no: int):
    """Serve one rendered page as PNG."""
    settings = get_settings()
    pattern = settings.data_dir / "pages" / f"{paper_id}_page{page_no:04d}.png"
    if not _safe_under(settings.data_dir, pattern):
        raise HTTPException(status_code=404, detail="page image not found")
    if not pattern.exists():
        raise HTTPException(status_code=404, detail="page image not found")
    return FileResponse(pattern, media_type="image/png")


# --------------------------------------------------------------------------- images
def _safe_under(root: Path, candidate: Path) -> bool:
    """True when ``candidate`` really is inside ``root`` (no ``..`` escape)."""
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
    except OSError:
        return False
    return root_resolved == candidate_resolved or root_resolved in candidate_resolved.parents


@router.get("/{paper_id}/figures/{figure_id}/image")
async def figure_image(paper_id: str, figure_id: str):
    """Serve the cropped image of one figure."""
    settings = get_settings()
    async with session_scope() as db:
        figure = await db.get(PaperFigure, figure_id)
        if figure is None or figure.paper_id != paper_id:
            raise HTTPException(status_code=404, detail="figure not found")
        path = Path(figure.image_path or "")

    if not path.name or not _safe_under(settings.data_dir, path):
        raise HTTPException(status_code=400, detail="invalid figure path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="figure image not found")
    return FileResponse(path, media_type="image/png")


@router.get("/{paper_id}/tables/{table_id}/image")
async def table_image(paper_id: str, table_id: str):
    """Serve a table as an image, cropped from the rendered page."""
    settings = get_settings()
    async with session_scope() as db:
        table = await db.get(PaperTable, table_id)
        if table is None or table.paper_id != paper_id:
            raise HTTPException(status_code=404, detail="table not found")
        page_no = table.page_no
        bbox = table.bbox

    page_path = settings.data_dir / "pages" / f"{paper_id}_page{page_no:04d}.png"
    if not page_path.exists() or not bbox or len(bbox) < 4:
        raise HTTPException(status_code=404, detail="table image not available")

    try:
        from PIL import Image

        with Image.open(page_path) as image:
            # The page image is rendered at a fixed resolution, so a bbox in
            # PDF points maps to it by that factor -- no matter how wide the
            # page is.
            scale = layout.PAGE_IMAGE_DPI / 72.0
            x0, y0, x1, y1 = (float(v) for v in bbox[:4])
            crop = image.crop((
                max(0, int(x0 * scale)), max(0, int(y0 * scale)),
                min(image.width, int(x1 * scale)), min(image.height, int(y1 * scale)),
            ))
            if crop.width < 4 or crop.height < 4:
                raise HTTPException(status_code=404, detail="table image not available")
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            buffer.seek(0)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"table image failed: {exc}") from exc
    return StreamingResponse(buffer, media_type="image/png")


@router.get("/{paper_id}/formulas/{formula_id}/image")
async def formula_image(paper_id: str, formula_id: str):
    """Serve a display equation as an image, cropped from the rendered page.

    The text of an equation extracted from a PDF is a flattened formula
    ("log P(u|u-k, . . . , u-1; Θ)") whose subscripts and symbols are lost.
    The cropped image keeps the typeset equation as the reader shows it.
    """
    settings = get_settings()
    async with session_scope() as db:
        formula = await db.get(PaperFormula, formula_id)
        if formula is None or formula.paper_id != paper_id:
            raise HTTPException(status_code=404, detail="formula not found")
        page_no = formula.page_no
        bbox = formula.bbox

    page_path = settings.data_dir / "pages" / f"{paper_id}_page{page_no:04d}.png"
    if not page_path.exists() or not bbox or len(bbox) < 4:
        raise HTTPException(status_code=404, detail="formula image not available")

    try:
        from PIL import Image

        with Image.open(page_path) as image:
            scale = layout.PAGE_IMAGE_DPI / 72.0
            x0, y0, x1, y1 = (float(v) for v in bbox[:4])
            crop = image.crop((
                max(0, int(x0 * scale)), max(0, int(y0 * scale)),
                min(image.width, int(x1 * scale)), min(image.height, int(y1 * scale)),
            ))
            if crop.width < 4 or crop.height < 4:
                raise HTTPException(status_code=404, detail="formula image not available")
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            buffer.seek(0)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"formula image failed: {exc}") from exc
    return StreamingResponse(buffer, media_type="image/png")


# --------------------------------------------------------------------------- citations
@router.get("/{paper_id}/citations")
async def citation_map(paper_id: str) -> dict:
    """Map every in-text citation marker to its bibliography entry.

    This is what powers the "click [11] -> jump to the reference" and
    "click the reference -> jump back to the sentence" interaction.
    """
    async with session_scope() as db:
        refs = (await db.execute(
            select(PaperReference).where(PaperReference.paper_id == paper_id)
            .order_by(PaperReference.index_in_doc))).scalars().all()
        chunks = (await db.execute(
            select(Chunk).where(Chunk.paper_id == paper_id)
            .order_by(Chunk.ordinal))).scalars().all()

    references = {r.index_in_doc: {
        "index_in_doc": r.index_in_doc, "raw_text": r.raw_text, "title": r.title,
        "authors": r.authors or [], "year": r.year, "venue": r.venue, "doi": r.doi,
        "arxiv_id": r.arxiv_id, "linked_paper_id": r.linked_paper_id,
    } for r in refs}

    markers: list[dict[str, Any]] = []
    for chunk in chunks:
        found = find_intext_citations(chunk.content)
        if not found:
            continue
        for offset, numbers in found:
            markers.append({
                "chunk_id": chunk.id,
                "page_no": chunk.page_no,
                "char_start": chunk.char_start + offset,
                "char_end": chunk.char_start + offset + 40,
                "numbers": numbers,
                "context": citation_context(chunk.content, offset),
            })

    return {
        "paper_id": paper_id,
        "references": list(references.values()),
        "markers": markers,
        "count": len(markers),
    }


@router.get("/{paper_id}/reference/{index}")
async def reference(paper_id: str, index: int) -> dict:
    """One bibliography entry (the target of a citation jump)."""
    async with session_scope() as db:
        ref = (await db.execute(
            select(PaperReference).where(
                PaperReference.paper_id == paper_id,
                PaperReference.index_in_doc == index))).scalars().first()
        if ref is None:
            raise HTTPException(status_code=404, detail="reference not found")
        # Find where this reference is cited (the "jump back" anchor).
        chunks = (await db.execute(
            select(Chunk).where(Chunk.paper_id == paper_id))).scalars().all()
        back = []
        for chunk in chunks:
            for offset, numbers in find_intext_citations(chunk.content):
                if index in numbers:
                    back.append({"chunk_id": chunk.id, "page_no": chunk.page_no,
                                 "char_start": chunk.char_start + offset,
                                 "context": citation_context(chunk.content, offset)})
                    break
        return {
            "index_in_doc": ref.index_in_doc, "raw_text": ref.raw_text,
            "title": ref.title, "authors": ref.authors or [], "year": ref.year,
            "venue": ref.venue, "doi": ref.doi, "arxiv_id": ref.arxiv_id,
            "linked_paper_id": ref.linked_paper_id, "cited_at": back,
        }


# --------------------------------------------------------------------------- translation
@router.post("/translate")
async def translate(req: TranslateRequest) -> dict:
    """Translate a selection (paragraph, sentence or hovered phrase)."""
    result = await translation_service.translate_text(req.text, req.target_lang)
    return result


# The whole-document translation lives in ``app/api/routers/translation.py``:
# it runs as a background job so the reader can show progress instead of
# waiting for a request that takes minutes.
