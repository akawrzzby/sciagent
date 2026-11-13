"""The data access layer that feeds the agent's retrieval tools.

The search tools read their passages straight from the database.  A field that
the service reads but the schema does not have makes *every* search fail, and
the agent then answers every question with a refusal -- silently, because the
tool error is swallowed into the trace.  ``chunk.section_title`` used to be
exactly that field: the model had no such column, so the whole retrieval path
was dead while the unit tests still passed.

These tests therefore talk to a real (temporary) SQLite database instead of
mocking the models.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text

from app.core.config import get_settings
from app.storage.db import dispose_db, init_db, session_scope
from app.storage.models import Chunk, Paper
from app.services import agent as agent_service


def _db_path() -> Path:
    url = get_settings().sqlite_url
    return Path(url.split("///", 1)[-1])


async def _reset_db() -> None:
    """Empty the tables so that each test starts from a known state."""
    async with session_scope() as db:
        await db.execute(text("DELETE FROM chunks"))
        await db.execute(text("DELETE FROM papers"))


def test_load_chunks_returns_the_section_title() -> None:
    """The retrieval tools need the heading a chunk was cut from."""

    async def scenario() -> list[dict]:
        await init_db()
        try:
            await _reset_db()
            async with session_scope() as db:
                db.add(Paper(id="paper1", title="Attention Is All You Need", year=2017))
                db.add(Chunk(
                    id="chunk1", paper_id="paper1", ordinal=0, page_no=3,
                    content="The model uses multi-head attention.",
                    section_title="3.2.1 Attention", kind="body"))
            return await agent_service.load_chunks()
        finally:
            await dispose_db()

    rows = asyncio.run(scenario())
    assert len(rows) == 1
    assert rows[0]["section_title"] == "3.2.1 Attention"
    assert rows[0]["paper_title"] == "Attention Is All You Need"
    assert rows[0]["content"].startswith("The model uses multi-head")


def test_load_chunks_can_be_restricted_to_a_paper() -> None:
    async def scenario() -> list[dict]:
        await init_db()
        try:
            await _reset_db()
            async with session_scope() as db:
                db.add(Paper(id="paper1", title="A", year=2017))
                db.add(Paper(id="paper2", title="B", year=2018))
                db.add(Chunk(id="c1", paper_id="paper1", ordinal=0, content="one",
                             section_title="1 Intro"))
                db.add(Chunk(id="c2", paper_id="paper2", ordinal=0, content="two",
                             section_title="2 Method"))
            return await agent_service.load_chunks(["paper2"])
        finally:
            await dispose_db()

    rows = asyncio.run(scenario())
    assert [r["id"] for r in rows] == ["c2"]
    assert rows[0]["section_title"] == "2 Method"


def test_init_db_brings_an_old_database_up_to_date() -> None:
    """A database created before a column was added must be migrated."""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    # An old chunks table: everything except section_title.
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, paper_id TEXT)")
    con.commit()
    con.close()

    async def scenario() -> None:
        await init_db()
        await dispose_db()

    asyncio.run(scenario())

    con = sqlite3.connect(path)
    cols = [r[1] for r in con.execute("PRAGMA table_info(chunks)")]
    con.close()
    assert "section_title" in cols


@pytest.mark.parametrize("field", ["section_title", "content", "page_no"])
def test_every_field_the_retrieval_reads_exists(field: str) -> None:
    """Guard against the service reading a column the model does not have."""
    from app.storage.models import Base

    assert hasattr(Chunk, field), f"Chunk.{field} is read by the retriever"
    assert field in {c.name for c in Base.metadata.tables["chunks"].columns}
