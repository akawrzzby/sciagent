"""Paper service: upload → parse → de-duplicate → index → persist.

This is the module-1 business logic.  The upload path is asynchronous: the
HTTP request returns immediately with a job id and the client follows the
progress on the SSE channel.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.core.config import get_settings
from app.core.events import get_bus, paper_channel
from app.core.logging import get_logger
from app.parsing import dedup as dedup_mod
from app.parsing import pipeline
from app.rag import chunker
from app.rag.embeddings import VectorStore, get_embedding_service
from app.storage.db import session_scope
from app.storage.models import (
    Chunk,
    ContentBlock,
    Paper,
    PaperFigure,
    PaperFormula,
    PaperReference,
    PaperTable,
    PaperVersion,
    Section,
)

logger = get_logger(__name__)


@dataclass
class UploadOutcome:
    """The result of processing one uploaded PDF."""

    paper_id: str
    status: str
    duplicate: bool = False
    duplicate_of: str | None = None
    recommendation: dict[str, Any] | None = None
    diff: list[dict[str, Any]] | None = None
    diff_summary: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id, "status": self.status, "duplicate": self.duplicate,
            "duplicate_of": self.duplicate_of, "recommendation": self.recommendation,
            "diff": self.diff, "diff_summary": self.diff_summary, "message": self.message,
        }


# --------------------------------------------------------------------------- helpers
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(name: str) -> str:
    return os.path.basename(name).replace("..", "_").strip() or "upload.pdf"


async def _emit(channel: str, event_type: str, **data: Any) -> None:
    await get_bus().publish(channel, event_type, **data)


# --------------------------------------------------------------------------- fingerprinting
async def _load_fingerprints() -> list[dedup_mod.PaperFingerprint]:
    """Load the fingerprints of every paper in the library."""
    store = VectorStore()
    service = get_embedding_service()
    out: list[dedup_mod.PaperFingerprint] = []

    async with session_scope() as db:
        rows = (await db.execute(select(Paper))).scalars().all()
        for p in rows:
            fp = dedup_mod.PaperFingerprint(
                paper_id=p.id,
                title=p.title,
                arxiv_id=p.arxiv_id,
                version_label=p.version_label,
                file_hash=p.file_hash,
                content_hash=p.content_hash,
                char_count=0,
                page_count=p.page_count,
                abstract=p.abstract,
                year=p.year,
            )
            # Recompute the simhash from the stored text is expensive; instead
            # store the character count which is a good proxy for the diff.
            fp.char_count = getattr(p, "char_count", 0) or 0
            matrix = store.load(p.id, mmap=False)
            if matrix is not None and matrix.size:
                fp.embedding = matrix[0]
            out.append(fp)
    return out


# --------------------------------------------------------------------------- ingest
async def ingest_pdf(
    source_path: str,
    *,
    filename: str = "",
    paper_id: str = "",
    session_id: str = "",
    allow_duplicate: bool = False,
) -> UploadOutcome:
    """Parse a PDF and add it to the library.

    When a duplicate is detected the paper is *not* stored silently: the caller
    receives a recommendation and can decide to overwrite or to keep the
    existing version.
    """
    settings = get_settings()
    paper_id = paper_id or uuid.uuid4().hex
    filename = filename or os.path.basename(source_path)
    channel = paper_channel(paper_id)

    await _emit(channel, "status", status="parsing", message="正在解析 PDF …", paper_id=paper_id)

    async def progress(fraction: float, message: str) -> None:
        await _emit(channel, "progress", progress=round(fraction, 4), message=message,
                    paper_id=paper_id)

    parsed = await pipeline.parse_document(
        source_path, paper_id=paper_id, out_dir=str(settings.data_dir / "pages"),
        progress=progress, session_id=session_id,
    )

    # ---- duplicate detection -------------------------------------------
    fingerprint = pipeline.build_fingerprint(
        parsed, paper_id=paper_id, path=source_path,
        embedding=await _document_embedding(parsed),
    )
    existing = await _load_fingerprints()
    candidates = dedup_mod.compare_fingerprints(fingerprint, existing)
    top = candidates[0] if candidates else None

    if top and top.same_paper and not allow_duplicate:
        recommendation = dedup_mod.recommend_version(fingerprint, top)
        diff = await _version_diff(top.fingerprint.paper_id, parsed.full_text)
        await _emit(channel, "duplicate", duplicate_of=top.fingerprint.paper_id,
                    recommendation=recommendation, reasons=top.reasons)
        return UploadOutcome(
            paper_id=paper_id, status="duplicate", duplicate=True,
            duplicate_of=top.fingerprint.paper_id, recommendation=recommendation,
            diff=diff, message="检测到内容高度重合的文献，已阻止入库并给出版本建议",
        )

    # ---- persist ---------------------------------------------------------
    await _emit(channel, "status", status="storing", message="正在写入索引 …", paper_id=paper_id)
    await _persist(paper_id, parsed, source_path, filename)

    await _emit(channel, "status", status="done", message="解析完成", paper_id=paper_id,
                page_count=parsed.page_count, is_scanned=parsed.is_scanned)
    return UploadOutcome(paper_id=paper_id, status="ok", duplicate=False)


async def _document_embedding(parsed: pipeline.ParsedDocument) -> list[float] | None:
    """Embed the first paragraph of the document (used for dedup)."""
    service = get_embedding_service()
    probe = parsed.metadata.abstract or parsed.full_text[:1200]
    if not probe:
        return None
    vecs = await asyncio.to_thread(service.encode, [probe])
    if vecs is None or not len(vecs):
        return None
    return list(vecs[0])


async def _version_diff(existing_paper_id: str, new_text: str) -> list[dict[str, Any]]:
    """Diff the new upload against the text of the existing paper."""
    async with session_scope() as db:
        chunks = (await db.execute(
            select(Chunk).where(Chunk.paper_id == existing_paper_id)
            .order_by(Chunk.ordinal))).scalars().all()
        old_text = "\n".join(c.content for c in chunks)
    return dedup_mod.diff_versions(old_text, new_text)


async def _persist(
    paper_id: str,
    parsed: pipeline.ParsedDocument,
    source_path: str,
    filename: str,
) -> None:
    """Write the parsed document into the database and the vector index."""
    settings = get_settings()
    meta = parsed.metadata

    # Copy the source file into the managed upload directory.
    uploads = settings.data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    managed_path = uploads / f"{paper_id}_{_safe_filename(filename)}"
    if os.path.abspath(source_path) != os.path.abspath(str(managed_path)):
        shutil.copy2(source_path, managed_path)

    # ---- chunks ----------------------------------------------------------
    chunk_list = chunker.chunk_document(
        full_text=parsed.full_text,
        sections=parsed.sections,
        blocks=parsed.blocks,
        abstract=meta.abstract,
    )

    async with session_scope() as db:
        # Remove any previous content of this paper (idempotent re-upload).
        for model in (Chunk, ContentBlock, PaperTable, PaperFigure, PaperFormula,
                      PaperReference, Section):
            await db.execute(model.__table__.delete().where(model.__table__.c.paper_id == paper_id))

        paper = await db.get(Paper, paper_id)
        if paper is None:
            paper = Paper(id=paper_id)
            db.add(paper)

        paper.title = meta.title
        paper.authors = [{"name": a.name, "affiliations": a.affiliation_indices}
                         for a in meta.authors]
        paper.affiliations = meta.affiliations
        paper.abstract = meta.abstract
        paper.year = meta.year
        paper.venue = meta.venue
        paper.doi = meta.doi
        paper.arxiv_id = meta.arxiv_id
        paper.version_label = meta.version_label
        paper.filename = filename
        paper.source_path = str(managed_path)
        paper.file_hash = dedup_mod.file_hash(str(managed_path))
        paper.content_hash = dedup_mod.content_hash(parsed.full_text)
        paper.file_size = os.path.getsize(str(managed_path))
        paper.page_count = parsed.page_count
        paper.is_scanned = parsed.is_scanned
        paper.ocr_pages = parsed.ocr_pages
        paper.status = "ready"
        paper.parse_error = ""
        paper.metadata_json = {
            "keywords": meta.keywords,
            "warnings": parsed.warnings,
            "parse_seconds": round(parsed.parse_seconds, 2),
            "chunk_count": len(chunk_list),
            "columns": "two-column" if _is_two_column(parsed) else "single-column",
        }

        # Sections
        section_ids: dict[str, str] = {}
        for sec in parsed.sections:
            sid = uuid.uuid4().hex
            section_ids[f"{sec.get('page_no')}:{sec.get('order_index')}"] = sid
            db.add(Section(
                id=sid, paper_id=paper_id, parent_id=None, level=sec.get("level", 1),
                number=sec.get("number", ""), title=sec.get("title", ""),
                page_start=sec.get("page_no", 0), page_end=sec.get("page_no", 0),
                bbox=sec.get("bbox", []), content=sec.get("content", ""),
                order_index=sec.get("order_index", 0),
            ))

        # Blocks
        for i, b in enumerate(parsed.blocks):
            db.add(ContentBlock(
                id=uuid.uuid4().hex, paper_id=paper_id, section_id=None,
                page_no=b.get("page_no", 0), block_type=b.get("block_type", "text"),
                bbox=b.get("bbox", []), content=b.get("content", ""), order_index=i,
            ))

        for t in parsed.tables:
            db.add(PaperTable(id=uuid.uuid4().hex, paper_id=paper_id,
                              page_no=t.get("page_no", 0), caption=t.get("caption", ""),
                              markdown=t.get("markdown", ""), csv_text=t.get("csv", ""),
                              bbox=t.get("bbox", [])))
        for f in parsed.figures:
            db.add(PaperFigure(id=uuid.uuid4().hex, paper_id=paper_id,
                               page_no=f.get("page_no", 0), caption=f.get("caption", ""),
                               image_path=f.get("image_path", ""), bbox=f.get("bbox", []),
                               description=f.get("description", "")))
        for fx in parsed.formulas:
            db.add(PaperFormula(id=uuid.uuid4().hex, paper_id=paper_id,
                                page_no=fx.get("page_no", 0), latex=fx.get("latex", ""),
                                raw_text=fx.get("raw_text", ""), bbox=fx.get("bbox", [])))
        for r in parsed.references:
            db.add(PaperReference(
                id=uuid.uuid4().hex, paper_id=paper_id,
                index_in_doc=r.get("index_in_doc", 0), raw_text=r.get("raw_text", ""),
                authors=r.get("authors", []), title=r.get("title", ""),
                year=r.get("year"), venue=r.get("venue", ""), doi=r.get("doi", ""),
                arxiv_id=r.get("arxiv_id", ""),
            ))

        # Chunks
        chunk_rows: list[Chunk] = []
        for c in chunk_list:
            row = Chunk(
                id=uuid.uuid4().hex, paper_id=paper_id, section_id=None,
                ordinal=c.ordinal, page_no=c.page_no, char_start=c.char_start,
                char_end=c.char_end, content=c.content,
                token_count=c.token_estimate, kind=c.kind,
                embedding_ref="", embedding_row=-1,
            )
            db.add(row)
            chunk_rows.append(row)

        await db.flush()

        # Version history
        db.add(PaperVersion(
            id=uuid.uuid4().hex, paper_id=paper_id, version_label=meta.version_label,
            filename=filename, file_hash=paper.file_hash, content_hash=paper.content_hash,
            page_count=parsed.page_count, char_count=len(parsed.full_text),
            is_current=True,
        ))

    # ---- dense index -----------------------------------------------------
    await _index_chunks(paper_id, chunk_list)


def _is_two_column(parsed: pipeline.ParsedDocument) -> bool:
    """Heuristic: does the document look like a two-column paper?"""
    from app.parsing.layout import detect_columns

    # Re-derive from the stored blocks (cheap approximation).
    per_page: dict[int, list] = {}
    for b in parsed.blocks:
        per_page.setdefault(b.get("page_no", 0), []).append(b)
    votes = 0
    for page_blocks in list(per_page.values())[:5]:
        if len(page_blocks) < 6:
            continue
        xs = sorted(b["bbox"][0] for b in page_blocks if b.get("bbox"))
        if len(xs) < 6:
            continue
        left = [x for x in xs if x < 300]
        right = [x for x in xs if x >= 300]
        if len(left) >= 3 and len(right) >= 3:
            votes += 1
    return votes >= 2


async def _index_chunks(paper_id: str, chunk_list: list[chunker.Chunk]) -> None:
    """Embed the chunks of a paper and write the vector matrix."""
    if not chunk_list:
        return
    service = get_embedding_service()
    store = VectorStore()

    vectors: list[Any] = []
    batch = 8
    for start in range(0, len(chunk_list), batch):
        piece = chunk_list[start : start + batch]
        vecs = await asyncio.to_thread(service.encode, [c.content for c in piece])
        if vecs is None:
            break
        vectors.append(vecs)
    if not vectors:
        logger.info("dense index skipped for %s (no embedding model)", paper_id)
        return

    import numpy as np

    matrix = np.vstack(vectors).astype("float32")
    path = store.save(paper_id, matrix)

    async with session_scope() as db:
        rows = (await db.execute(
            select(Chunk).where(Chunk.paper_id == paper_id)
            .order_by(Chunk.ordinal))).scalars().all()
        for i, row in enumerate(rows):
            if i < matrix.shape[0]:
                row.embedding_row = i
                row.embedding_ref = path


# --------------------------------------------------------------------------- queries
async def list_papers(query: str = "", limit: int = 200) -> list[dict[str, Any]]:
    """Return the library, optionally filtered by a text query."""
    async with session_scope() as db:
        stmt = select(Paper).order_by(Paper.created_at.desc()).limit(limit)
        rows = (await db.execute(stmt)).scalars().all()
        out = []
        for p in rows:
            if query:
                haystack = f"{p.title} {p.arxiv_id} {' '.join(p.affiliations or [])}".lower()
                if query.lower() not in haystack:
                    continue
            out.append({
                "id": p.id, "title": p.title, "authors": p.authors or [],
                "affiliations": p.affiliations or [], "year": p.year,
                "arxiv_id": p.arxiv_id, "version_label": p.version_label,
                "page_count": p.page_count, "is_scanned": p.is_scanned,
                "ocr_pages": p.ocr_pages, "status": p.status,
                "created_at": p.created_at.isoformat() if p.created_at else "",
                "metadata": p.metadata_json or {},
            })
        return out


async def get_paper(paper_id: str) -> dict[str, Any] | None:
    async with session_scope() as db:
        p = await db.get(Paper, paper_id)
        if p is None:
            return None
        return {
            "id": p.id, "title": p.title, "authors": p.authors or [],
            "affiliations": p.affiliations or [], "abstract": p.abstract,
            "year": p.year, "venue": p.venue, "doi": p.doi, "arxiv_id": p.arxiv_id,
            "version_label": p.version_label, "page_count": p.page_count,
            "is_scanned": p.is_scanned, "ocr_pages": p.ocr_pages,
            "file_hash": p.file_hash, "content_hash": p.content_hash,
            "filename": p.filename, "file_size": p.file_size,
            "metadata": p.metadata_json or {},
            "created_at": p.created_at.isoformat() if p.created_at else "",
        }


async def get_paper_structure(paper_id: str) -> dict[str, Any]:
    """Heading tree + blocks + tables + figures + formulas + references."""
    async with session_scope() as db:
        p = await db.get(Paper, paper_id)
        if p is None:
            return {}
        sections = (await db.execute(
            select(Section).where(Section.paper_id == paper_id)
            .order_by(Section.order_index))).scalars().all()
        tables = (await db.execute(
            select(PaperTable).where(PaperTable.paper_id == paper_id))).scalars().all()
        figures = (await db.execute(
            select(PaperFigure).where(PaperFigure.paper_id == paper_id))).scalars().all()
        formulas = (await db.execute(
            select(PaperFormula).where(PaperFormula.paper_id == paper_id))).scalars().all()
        refs = (await db.execute(
            select(PaperReference).where(PaperReference.paper_id == paper_id)
            .order_by(PaperReference.index_in_doc))).scalars().all()

        return {
            "paper_id": paper_id,
            "title": p.title,
            "sections": [{"id": s.id, "title": s.title, "number": s.number,
                          "level": s.level, "page_no": s.page_start,
                          "bbox": s.bbox, "content": s.content,
                          "order_index": s.order_index} for s in sections],
            "tables": [{"id": t.id, "page_no": t.page_no, "caption": t.caption,
                        "markdown": t.markdown, "csv": t.csv_text, "bbox": t.bbox}
                       for t in tables],
            "figures": [{"id": f.id, "page_no": f.page_no, "caption": f.caption,
                         "image_path": f.image_path, "bbox": f.bbox,
                         "description": f.description} for f in figures],
            "formulas": [{"id": x.id, "page_no": x.page_no, "latex": x.latex,
                          "raw_text": x.raw_text, "bbox": x.bbox} for x in formulas],
            "references": [{"id": r.id, "index_in_doc": r.index_in_doc,
                            "raw_text": r.raw_text, "authors": r.authors,
                            "title": r.title, "year": r.year, "venue": r.venue,
                            "doi": r.doi, "arxiv_id": r.arxiv_id,
                            "linked_paper_id": r.linked_paper_id} for r in refs],
        }


async def delete_paper(paper_id: str) -> bool:
    VectorStore().delete(paper_id)
    async with session_scope() as db:
        p = await db.get(Paper, paper_id)
        if p is None:
            return False
        path = p.source_path
        await db.delete(p)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    return True


async def library_stats() -> dict[str, Any]:
    """Aggregate statistics of the library (shown on the dashboard)."""
    from sqlalchemy import func

    async with session_scope() as db:
        total = (await db.execute(select(func.count(Paper.id)))).scalar() or 0
        pages = (await db.execute(select(func.coalesce(func.sum(Paper.page_count), 0)))).scalar() or 0
        chunks = (await db.execute(select(func.count(Chunk.id)))).scalar() or 0
        scanned = (await db.execute(
            select(func.count(Paper.id)).where(Paper.is_scanned == True))).scalar() or 0
        refs = (await db.execute(select(func.count(PaperReference.id)))).scalar() or 0
        tables = (await db.execute(select(func.count(PaperTable.id)))).scalar() or 0
        figures = (await db.execute(select(func.count(PaperFigure.id)))).scalar() or 0
        formulas = (await db.execute(select(func.count(PaperFormula.id)))).scalar() or 0
    return {
        "papers": total, "pages": pages, "chunks": chunks, "scanned_papers": scanned,
        "references": refs, "tables": tables, "figures": figures, "formulas": formulas,
    }
