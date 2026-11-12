"""Module 3 – layout-aware reading, citation navigation and bilingual translation."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.config import get_settings
from app.parsing.references import citation_context, find_intext_citations
from app.services import papers as paper_service
from app.services import translation as translation_service
from app.storage.db import session_scope
from app.storage.models import Chunk, PaperReference, Translation

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
    if not pattern.exists():
        raise HTTPException(status_code=404, detail="page image not found")
    return FileResponse(pattern, media_type="image/png")


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


@router.get("/{paper_id}/translation")
async def document_translation(paper_id: str, target_lang: str = "zh") -> dict:
    """The (cached) aligned translation of a whole paper."""
    async with session_scope() as db:
        cached = (await db.execute(
            select(Translation).where(Translation.paper_id == paper_id,
                                      Translation.locale == target_lang))).scalars().first()
        if cached is not None:
            return {"paper_id": paper_id, "locale": target_lang,
                    "markdown": cached.markdown,
                    "segments": cached.segments or [], "cached": True}

    structure = await paper_service.get_paper_structure(paper_id)
    if not structure:
        raise HTTPException(status_code=404, detail="paper not found")

    async with session_scope() as db:
        blocks = (await db.execute(
            select(Chunk).where(Chunk.paper_id == paper_id)
            .order_by(Chunk.ordinal))).scalars().all()
    block_dicts = [{"content": c.content, "page_no": c.page_no} for c in blocks]

    segments = await translation_service.build_document_segments(
        structure["sections"], block_dicts
    )
    segments = await translation_service.translate_document(segments, target_lang)

    markdown = "\n\n".join(
        (f"### {s.source}\n\n{s.translation}" if s.kind == "heading" else s.translation)
        for s in segments
    )
    async with session_scope() as db:
        db.add(Translation(id=uuid4(), paper_id=paper_id, locale=target_lang,
                           markdown=markdown,
                           segments=[s.to_dict() for s in segments]))
    return {"paper_id": paper_id, "locale": target_lang, "markdown": markdown,
            "segments": [s.to_dict() for s in segments], "cached": False}


def uuid4() -> str:
    import uuid

    return uuid.uuid4().hex
