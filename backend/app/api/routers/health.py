"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import func, select

from app.core.config import get_settings
from app.llm.client import get_llm_client
from app.rag.embeddings import get_embedding_service
from app.storage.db import session_scope
from app.storage.models import Paper

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness + dependency status."""
    settings = get_settings()
    llm = await get_llm_client().health()
    embeddings = get_embedding_service().status()

    async with session_scope() as db:
        papers = (await db.execute(select(func.count(Paper.id)))).scalar() or 0

    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "database": {"ok": True, "url": settings.sqlite_url, "papers": papers},
        "llm": llm,
        "embeddings": embeddings,
    }


@router.get("/ready")
async def ready() -> dict:
    """Readiness probe: the LLM backend must be reachable."""
    llm = await get_llm_client().health()
    return {"ready": bool(llm.get("ok")), "llm": llm}


@router.get("/config")
async def public_config() -> dict:
    """Expose the non-secret parts of the configuration to the UI."""
    settings = get_settings()
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "agent_max_steps": settings.agent_max_steps,
        "retrieve_top_k_final": settings.retrieve_top_k_final,
        "render_dpi": settings.render_dpi,
    }
