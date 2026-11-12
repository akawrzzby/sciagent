"""Administrative endpoints: prompt audit trail, statistics and index management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select

from app.storage.db import session_scope
from app.storage.models import AuditRecord, Job, Paper

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/audit")
async def audit(limit: int = 50, purpose: str = "", session_id: str = "") -> dict:
    """The prompt audit trail.

    Every LLM call is recorded with its system prompt, user prompt, tools,
    response, token usage and latency.  This is the artefact requested for the
    final round ("Prompt 审计记录").
    """
    async with session_scope() as db:
        stmt = select(AuditRecord).order_by(AuditRecord.created_at.desc()).limit(limit)
        if purpose:
            stmt = stmt.where(AuditRecord.purpose.like(f"%{purpose}%"))
        if session_id:
            stmt = stmt.where(AuditRecord.session_id == session_id)
        rows = (await db.execute(stmt)).scalars().all()
        return {
            "records": [
                {"id": r.id, "session_id": r.session_id, "purpose": r.purpose,
                 "system_prompt": r.system_prompt, "user_prompt": r.user_prompt[:4000],
                 "tools": [t.get("function", {}).get("name") for t in (r.tools or [])],
                 "response": (r.response or "")[:4000],
                 "reasoning": (r.reasoning or "")[:2000],
                 "tool_calls": r.tool_calls or [],
                 "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                 "latency_ms": r.latency_ms, "ok": r.ok,
                 "created_at": r.created_at.isoformat() if r.created_at else ""}
                for r in rows
            ],
            "count": len(rows),
        }


@router.get("/audit/stats")
async def audit_stats() -> dict:
    """Aggregated usage statistics of the LLM calls."""
    async with session_scope() as db:
        rows = (await db.execute(select(AuditRecord))).scalars().all()
    by_purpose: dict[str, dict[str, Any]] = {}
    for r in rows:
        entry = by_purpose.setdefault(r.purpose or "unknown", {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0,
            "errors": 0,
        })
        entry["calls"] += 1
        entry["prompt_tokens"] += r.prompt_tokens
        entry["completion_tokens"] += r.completion_tokens
        entry["latency_ms"] += r.latency_ms
        if not r.ok:
            entry["errors"] += 1
    for entry in by_purpose.values():
        entry["avg_latency_ms"] = entry["latency_ms"] // max(1, entry["calls"])
    return {"total_calls": len(rows), "by_purpose": by_purpose}


@router.get("/jobs")
async def jobs(limit: int = 50) -> dict:
    async with session_scope() as db:
        rows = (await db.execute(
            select(Job).order_by(Job.created_at.desc()).limit(limit))).scalars().all()
        return {"jobs": [
            {"id": j.id, "kind": j.kind, "target_id": j.target_id, "status": j.status,
             "progress": j.progress, "message": j.message, "error": j.error,
             "created_at": j.created_at.isoformat() if j.created_at else ""}
            for j in rows
        ]}


@router.post("/reindex")
async def reindex() -> dict:
    """Rebuild the dense vector index of the whole library."""
    from app.rag.embeddings import VectorStore, get_embedding_service
    from app.rag.retriever import embed_chunks
    from app.storage.models import Chunk as ChunkModel

    service = get_embedding_service()
    store = VectorStore()

    async with session_scope() as db:
        papers = (await db.execute(select(Paper))).scalars().all()
        report = []
        for p in papers:
            chunks = (await db.execute(
                select(ChunkModel).where(ChunkModel.paper_id == p.id)
                .order_by(ChunkModel.ordinal))).scalars().all()
            if not chunks:
                continue
            payload = [{"content": c.content} for c in chunks]
            matrix = await embed_chunks(payload, service=service)
            if matrix is None:
                report.append({"paper_id": p.id, "status": "skipped (no dense model)"})
                continue
            store.save(p.id, matrix)
            for i, c in enumerate(chunks):
                c.embedding_row = i
                c.embedding_ref = store.path_for(p.id)
            report.append({"paper_id": p.id, "chunks": len(chunks), "status": "ok"})
    return {"report": report}
