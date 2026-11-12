"""Module 1 – PDF library management (upload, list, structure, versions)."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.events import get_bus, paper_channel
from app.core.logging import get_logger
from app.parsing import dedup as dedup_mod
from app.services import papers as paper_service
from app.storage.db import session_scope
from app.storage.models import Paper

logger = get_logger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])

ALLOWED_SUFFIXES = {".pdf"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB

# How many documents may be parsed at the same time.
#
# FastAPI's ``BackgroundTasks`` runs its tasks **sequentially**, so a batch
# upload of N PDFs would be parsed one after another.  The competition asks for
# a robust "multi-document concurrent analysis", so the ingestion jobs are
# dispatched onto the event loop instead and guarded by this semaphore.  The
# value is deliberately small: every parse is CPU bound (PyMuPDF extraction +
# page rendering) and the host also serves the LLM.
_PARSE_CONCURRENCY = 3
_parse_semaphore: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _parse_semaphore
    if _parse_semaphore is None:
        _parse_semaphore = asyncio.Semaphore(_PARSE_CONCURRENCY)
    return _parse_semaphore


async def _ingest_guarded(path: str, filename: str, paper_id: str,
                          allow_duplicate: bool) -> None:
    """Run one ingestion job, bounded by the global parse concurrency."""
    async with _semaphore():
        try:
            await paper_service.ingest_pdf(
                path, filename=filename, paper_id=paper_id,
                allow_duplicate=allow_duplicate,
            )
        except Exception as exc:  # noqa: BLE001 - a background job must not
            # kill the whole batch: record the failure on the paper row.
            logger.exception("ingestion failed for %s (%s)", filename, paper_id)
            async with session_scope() as db:
                paper = await db.get(Paper, paper_id)
                if paper is not None:
                    paper.status = "failed"
                    paper.parse_error = str(exc)[:500]


@router.get("")
async def list_papers(q: str = "", limit: int = 200) -> dict:
    """List the library."""
    items = await paper_service.list_papers(q, limit)
    return {"papers": items, "count": len(items)}


@router.get("/stats")
async def stats() -> dict:
    """Aggregate library statistics."""
    return await paper_service.library_stats()


@router.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    allow_duplicate: bool = Form(False),
) -> dict:
    """Upload one or more PDFs.

    The PDFs are stored immediately and parsed in the background; the client
    subscribes to ``/api/papers/{paper_id}/events`` for progress.
    """
    settings = get_settings()
    uploads_dir = settings.data_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, Any]] = []
    for upload_file in files:
        name = upload_file.filename or "upload.pdf"
        suffix = os.path.splitext(name)[1].lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400,
                                detail=f"仅支持 PDF 文件，收到: {name}")

        paper_id = uuid.uuid4().hex
        target = uploads_dir / f"{paper_id}_{os.path.basename(name)}"
        size = 0
        with open(target, "wb") as fh:
            while chunk := await upload_file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    fh.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="文件超过 200MB 限制")
                fh.write(chunk)

        accepted.append({"paper_id": paper_id, "filename": name, "path": str(target),
                         "size": size})

    # Kick off the parsing jobs.  They run concurrently (see
    # ``_PARSE_CONCURRENCY``) so that a batch upload is not processed one
    # document after another.
    for item in accepted:
        asyncio.create_task(_ingest_guarded(
            item["path"], item["filename"], item["paper_id"], allow_duplicate,
        ))

    return {"accepted": accepted, "count": len(accepted),
            "message": "文件已接收，正在后台解析。请订阅事件流查看进度。"}


@router.get("/{paper_id}")
async def get_paper(paper_id: str) -> dict:
    paper = await paper_service.get_paper(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="paper not found")
    return paper


@router.get("/{paper_id}/structure")
async def structure(paper_id: str) -> dict:
    """Heading tree, blocks, tables, figures, formulas and references."""
    data = await paper_service.get_paper_structure(paper_id)
    if not data:
        raise HTTPException(status_code=404, detail="paper not found")
    return data


@router.get("/{paper_id}/sections")
async def sections(paper_id: str) -> dict:
    """Only the heading tree (used by the navigation sidebar)."""
    data = await paper_service.get_paper_structure(paper_id)
    if not data:
        raise HTTPException(status_code=404, detail="paper not found")
    return {"sections": data["sections"]}


@router.get("/{paper_id}/references")
async def references(paper_id: str) -> dict:
    data = await paper_service.get_paper_structure(paper_id)
    if not data:
        raise HTTPException(status_code=404, detail="paper not found")
    return {"references": data["references"]}


@router.get("/{paper_id}/figures")
async def figures(paper_id: str) -> dict:
    data = await paper_service.get_paper_structure(paper_id)
    if not data:
        raise HTTPException(status_code=404, detail="paper not found")
    return {"figures": data["figures"]}


@router.get("/{paper_id}/tables")
async def tables(paper_id: str) -> dict:
    data = await paper_service.get_paper_structure(paper_id)
    if not data:
        raise HTTPException(status_code=404, detail="paper not found")
    return {"tables": data["tables"]}


@router.get("/{paper_id}/formulas")
async def formulas(paper_id: str) -> dict:
    data = await paper_service.get_paper_structure(paper_id)
    if not data:
        raise HTTPException(status_code=404, detail="paper not found")
    return {"formulas": data["formulas"]}


@router.get("/{paper_id}/versions")
async def versions(paper_id: str) -> dict:
    """The version history of one paper."""
    from sqlalchemy import select

    from app.storage.models import PaperVersion

    async with session_scope() as db:
        rows = (await db.execute(
            select(PaperVersion).where(PaperVersion.paper_id == paper_id)
            .order_by(PaperVersion.created_at))).scalars().all()
        return {"versions": [
            {"id": v.id, "version_label": v.version_label, "filename": v.filename,
             "page_count": v.page_count, "char_count": v.char_count,
             "is_current": v.is_current, "diff_summary": v.diff_summary,
             "created_at": v.created_at.isoformat() if v.created_at else ""}
            for v in rows
        ]}


@router.get("/{paper_id}/versions/diff")
async def version_diff(paper_id: str, other_paper_id: str) -> dict:
    """Diff two revisions of the same paper."""
    from sqlalchemy import select

    from app.storage.models import Chunk

    async with session_scope() as db:
        a = (await db.execute(
            select(Chunk).where(Chunk.paper_id == paper_id)
            .order_by(Chunk.ordinal))).scalars().all()
        b = (await db.execute(
            select(Chunk).where(Chunk.paper_id == other_paper_id)
            .order_by(Chunk.ordinal))).scalars().all()
        text_a = "\n".join(c.content for c in a)
        text_b = "\n".join(c.content for c in b)
    diff = dedup_mod.diff_versions(text_a, text_b)
    changed = sum(1 for d in diff if d["op"] != "equal")
    return {"diff": diff, "changed_blocks": changed, "total_blocks": len(diff)}


@router.delete("/{paper_id}")
async def delete(paper_id: str) -> dict:
    ok = await paper_service.delete_paper(paper_id)
    if not ok:
        raise HTTPException(status_code=404, detail="paper not found")
    return {"deleted": paper_id}


@router.get("/{paper_id}/file")
async def download_pdf(paper_id: str):
    """Serve the original PDF."""
    async with session_scope() as db:
        p = await db.get(Paper, paper_id)
        if p is None or not p.source_path or not os.path.exists(p.source_path):
            raise HTTPException(status_code=404, detail="file not found")
        path, filename = p.source_path, p.filename
    return FileResponse(path, media_type="application/pdf", filename=filename)
