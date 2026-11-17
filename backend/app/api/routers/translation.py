"""Full-document translation: start a job, poll its progress, read the result.

Translating a paper is a long operation — a 16 page paper is roughly six
hundred sentences.  Doing it inside the request would either time out or
leave the reader staring at a spinner for minutes with no way to tell
whether anything is happening.

So the work runs as a background job:

    POST /api/reader/{paper_id}/translation   -> {"status": "started", "job_id": ...}
    GET  /api/reader/{paper_id}/translation   -> {"status": "running"|"ready"|"error", ...}

A failed translation is **never** cached: the reader shows the error and the
user can retry.  (The previous implementation replaced every failed segment
with its English source and stored that as the cache, so a single LLM
outage silently turned the whole bilingual view into a copy of the
original — and kept doing so for every later request.)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.services import papers as paper_service
from app.services import translation as translation_service
from app.storage.db import session_scope
from app.storage.models import ContentBlock, Job, Paper, Translation

router = APIRouter(prefix="/reader", tags=["reader"])

#: Only one translation per paper may run at a time.
_RUNNING = {"running", "pending"}


def _job_id() -> str:
    return uuid.uuid4().hex


#: A job that has not reported progress for this long is considered dead.
_STALE_AFTER_SECONDS = 900


async def fail_running_jobs() -> None:
    """Mark the translation jobs that a restart interrupted.

    The worker lives in the process that started it, so a restart (or a
    deployment) kills it silently.  Without this the reader would keep
    polling a job that can never finish.
    """
    async with session_scope() as db:
        rows = (await db.execute(
            select(Job).where(Job.kind == "translation", Job.status.in_(_RUNNING))
        )).scalars().all()
        for job in rows:
            job.status = "failed"
            job.error = "服务重启，翻译中断，请重新开始"
            db.add(job)


def _is_stale(job: Job) -> bool:
    """True when the worker behind this job is gone (e.g. the server restarted)."""
    stamp = getattr(job, "updated_at", None) or getattr(job, "created_at", None)
    if stamp is None:
        return False
    # SQLite keeps the timestamp naive; treat it as UTC.
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    return age > _STALE_AFTER_SECONDS


# --------------------------------------------------------------------------- helpers
async def _load_blocks(paper_id: str) -> list[dict]:
    """The document stream used for translation: the parsed content blocks."""
    async with session_scope() as db:
        rows = (await db.execute(
            select(ContentBlock)
            .where(ContentBlock.paper_id == paper_id)
            .order_by(ContentBlock.order_index))).scalars().all()
    return [{"id": b.id, "page_no": b.page_no, "type": b.block_type or "",
             "content": b.content or ""} for b in rows]


async def _update_job(job_id: str, **fields) -> None:
    async with session_scope() as db:
        job = await db.get(Job, job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        db.add(job)


async def _run_translation(paper_id: str, target_lang: str, job_id: str) -> None:
    """Background worker: build the segments, translate them, cache the result."""
    try:
        structure = await paper_service.get_paper_structure(paper_id)
        if not structure:
            raise HTTPException(status_code=404, detail="paper not found")

        blocks = await _load_blocks(paper_id)
        segments = await translation_service.build_document_segments(
            structure.get("sections", []), blocks)
        await _update_job(job_id, status="running", progress=0.0,
                          message=f"共 {len(segments)} 段，开始翻译")

        async def progress(fraction: float) -> None:
            await _update_job(job_id, progress=round(fraction, 4),
                              message=f"已翻译 {int(fraction * len(segments))} / {len(segments)} 段")

        outcome = await translation_service.translate_document(
            segments, target_lang, progress_cb=progress)

        if not outcome.ok and not outcome.done:
            await _update_job(job_id, status="failed", error="; ".join(outcome.errors[:5]),
                              result=outcome.to_dict())
            return

        markdown = "\n\n".join(
            (f"### {s.source}\n\n{s.translation}" if s.kind == "heading" else s.translation)
            for s in segments
        )
        async with session_scope() as db:
            paper = await db.get(Paper, paper_id)
            if paper is None:
                await _update_job(job_id, status="failed", error="paper not found")
                return
            existing = (await db.execute(
                select(Translation).where(Translation.paper_id == paper_id,
                                          Translation.locale == target_lang))).scalars().first()
            if existing is None:
                db.add(Translation(id=_job_id(), paper=paper, locale=target_lang,
                                   markdown=markdown,
                                   segments=[s.to_dict() for s in segments]))
        await _update_job(job_id, status="done", progress=1.0,
                          message=(f"完成，共 {len(segments)} 段" if outcome.ok
                                   else f"完成 {outcome.done} 段，{len(outcome.failed)} 段未翻译"),
                          result=outcome.to_dict())
    except Exception as exc:  # pragma: no cover - defensive
        await _update_job(job_id, status="failed", error=str(exc)[:400])


# --------------------------------------------------------------------------- routes
@router.post("/{paper_id}/translation")
async def start_translation(paper_id: str, target_lang: str = "zh") -> dict:
    """Start (or join) the translation of a whole paper."""
    async with session_scope() as db:
        paper = await db.get(Paper, paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="paper not found")

        cached = (await db.execute(
            select(Translation).where(Translation.paper_id == paper_id,
                                      Translation.locale == target_lang))).scalars().first()
        if cached is not None:
            segments = cached.segments or []
            failed = sum(1 for s in segments if not (s.get("translation") or "").strip())
            return {"status": "ready", "cached": True,
                    "segments": segments, "markdown": cached.markdown,
                    "failed": failed}

        running = (await db.execute(
            select(Job).where(Job.kind == "translation",
                              Job.target_id == paper_id,
                              Job.status.in_(_RUNNING))
            .order_by(Job.created_at.desc()))).scalars().first()
        if running is not None and not _is_stale(running):
            return {"status": running.status, "job_id": running.id,
                    "progress": running.progress, "message": running.message,
                    "cached": False}
        if running is not None:
            # The worker died with the process that hosted it; do not let the
            # stale row block a new attempt.
            running.status = "failed"
            running.error = "任务已失效，请重新开始"
            db.add(running)

        job = Job(id=_job_id(), kind="translation", target_id=paper_id,
                  status="running", progress=0.0, message="正在准备段落")
        db.add(job)

    asyncio.create_task(_run_translation(paper_id, target_lang, job.id))
    return {"status": "started", "job_id": job.id, "progress": 0.0, "cached": False}


@router.get("/{paper_id}/translation")
async def translation_status(paper_id: str, target_lang: str = "zh") -> dict:
    """The translation of a paper: ready, still running, or failed."""
    async with session_scope() as db:
        cached = (await db.execute(
            select(Translation).where(Translation.paper_id == paper_id,
                                      Translation.locale == target_lang))).scalars().first()
        if cached is not None:
            segments = cached.segments or []
            # A sentence that could not be translated keeps its English source.
            failed = sum(1 for s in segments if not (s.get("translation") or "").strip())
            return {"status": "ready", "cached": True,
                    "segments": segments, "markdown": cached.markdown,
                    "failed": failed}

        job = (await db.execute(
            select(Job).where(Job.kind == "translation",
                              Job.target_id == paper_id)
            .order_by(Job.created_at.desc()))).scalars().first()

    if job is None:
        return {"status": "idle", "segments": [], "markdown": "", "cached": False}
    if job.status in _RUNNING:
        return {"status": "running", "job_id": job.id, "progress": job.progress,
                "message": job.message, "segments": [], "cached": False}
    if job.status == "done":
        # The job finished but the cache row is missing — ask the client to retry.
        return {"status": "idle", "message": job.message, "segments": [], "cached": False}
    return {"status": "error", "error": job.error or "翻译失败",
            "message": job.message, "segments": [], "cached": False}
