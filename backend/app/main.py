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
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

    yield

    await dispose_db()
    logger.info("shutdown complete")


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
        papers,
        rag,
        reader,
        search,
        viz,
        writing,
    )

    for module in (health, papers, rag, reader, search, graph, writing, viz, admin):
        app.include_router(module.router, prefix="/api")

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

    return app


app = create_app()
