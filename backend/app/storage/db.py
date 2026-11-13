"""Async database engine and session management (SQLAlchemy 2.0)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.models import Base

logger = get_logger(__name__)

_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    """Lazily create the async engine (one per process)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.sqlite_url,
            echo=False,
            future=True,
            # SQLite + aiosqlite: disable cross-thread checks, use WAL for
            # concurrent readers while a writer is active.
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        # WAL mode gives us "one writer + many readers" concurrency for free.
        @event.listens_for(_engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _record):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(), expire_on_commit=False, class_=AsyncSession
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope: commits on success, rolls back on error."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create every table (idempotent) and bring an old database up to date."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)
    logger.info("database initialised (%s)", engine.url)


# Columns that were added to the schema after the first release.  SQLite has
# no "ALTER TABLE ... ADD COLUMN IF NOT EXISTS", so the columns are checked
# and added one by one.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("chunks", "section_title", "TEXT DEFAULT ''"),
)


def _add_missing_columns(connection) -> None:  # pragma: no cover - driver hook
    for table, column, ddl in _ADDED_COLUMNS:
        rows = connection.execute(
            text(f"PRAGMA table_info({table})")
        ).fetchall()
        if not rows:
            continue
        if any(r[1] == column for r in rows):
            continue
        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
        logger.info("added missing column %s.%s", table, column)


async def dispose_db() -> None:
    """Release pooled connections on shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
