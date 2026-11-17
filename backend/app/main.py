"""SciAgent backend application entry point.

The service exposes:

* ``/api/*``          – REST + SSE endpoints for the six competition modules
* ``/api/health``     – readiness probe (also reports the LLM backend status)
* ``/docs``           – auto-generated OpenAPI documentation

Everything heavy (PDF parsing, embedding, retrieval) runs inside asyncio tasks
or a process pool so the event loop stays responsive for streaming answers.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import get_settings
from app.core.events import get_bus
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: set up logging, the database and shared clients."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    settings.ensure_dirs()
    logger.info("starting %s v%s (env=%s)", settings.app_name, settings.app_version,
                settings.environment)

    from app.storage.db import dispose_db, init_db

    await init_db()

    # A restart kills the background translation workers.  Mark their jobs as
    # failed so the reader can start a fresh one instead of polling a worker
    # that no longer exists.
    try:
        from app.api.routers.translation import fail_running_jobs

        await fail_running_jobs()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("could not clear the stale translation jobs: %s", exc)

    # Warm up the retrieval models in the background.  Loading bge-m3 takes a
    # couple of seconds; doing it here keeps the first user request fast and
    # makes ``/api/health`` report the *real* capability of the dense and
    # rerank channels instead of "not loaded yet".
    _warm_up_retrieval_models()

    yield

    await dispose_db()
    logger.info("shutdown complete")


def _warm_up_retrieval_models() -> None:
    """Load the embedding / reranker models off the event loop."""

    def _load() -> None:
        from app.rag.embeddings import get_embedding_service

        service = get_embedding_service()
        try:
            service.encode(["warm up"], is_query=False)
            service.rerank("warm up", ["warm up passage"])
        except Exception as exc:  # noqa: BLE001 - the service degrades on its own
            logger.warning("retrieval warm-up failed: %s", exc)
        status = service.status()
        logger.info("retrieval models ready: dense=%s reranker=%s",
                    status.get("dense_available"), status.get("reranker_available"))

    threading.Thread(target=_load, name="sciagent-warmup", daemon=True).start()


def create_app() -> FastAPI:
    """Build the FastAPI application (factory keeps tests isolated)."""
    settings = get_settings()

    app = FastAPI(
        title="SciAgent API",
        description="科研小助手 Agent — 全流程科研智能体后端",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------- routers
    from app.api.routers import (
        admin,
        graph,
        health,
        models,
        papers,
        rag,
        reader,
        search,
        translation,
        viz,
        writing,
    )

    # ``translation`` is registered before ``reader`` on purpose: both expose
    # ``/api/reader/{paper_id}/translation`` and the first match wins.
    for module in (health, models, papers, rag, translation, reader, search, graph, writing,
                   viz, admin):
        app.include_router(module.router, prefix="/api")
    from app.api.routers import document
    app.include_router(document.router, prefix="/api")

    # ------------------------------------------------------------- errors
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "internal error", "error": str(exc)[:400]})

    @app.get("/api", include_in_schema=False)
    async def _root() -> dict[str, str]:
        return {"name": settings.app_name, "version": settings.app_version,
                "docs": "/api/docs"}

    # ------------------------------------------------------- static frontend
    _mount_frontend(app, settings.static_dir)

    return app


def _mount_frontend(app: FastAPI, static_dir: str) -> None:
    """Serve the built SPA from the same process when ``STATIC_DIR`` is set.

    The API routers are registered first, so ``/api/*`` always wins; anything
    else falls back to ``index.html`` so that the React Router deep links
    (``/graph``, ``/reader/<id>`` …) survive a page reload.
    """
    if not static_dir:
        return
    root = Path(static_dir).expanduser().resolve()
    index = root / "index.html"
    if not index.is_file():
        logger.warning("STATIC_DIR=%s has no index.html - the SPA is not served",
                       static_dir)
        return

    logger.info("serving the frontend from %s", root)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: str) -> FileResponse:
        if full_path:
            candidate = (root / full_path).resolve()
            # Never serve anything outside the static root.
            if candidate.is_file() and root in candidate.parents:
                return FileResponse(candidate)
        return FileResponse(index)


app = create_app()
