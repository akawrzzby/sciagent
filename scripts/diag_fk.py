"""Diagnose the ``content_blocks`` -> ``papers`` foreign-key failure.

Every variant runs against the *same* database but with its own ``paper_id``,
so the row counts are directly comparable.  The "echo" variant additionally
prints the raw SQL statement order that SQLAlchemy emits, which is the
decisive evidence for the root cause.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.expanduser("~/sciagent/backend"))

TMP = tempfile.mkdtemp(prefix="fkdiag-")
os.environ["DATA_DIR"] = TMP
os.environ["INDEX_DIR"] = os.path.join(TMP, "index")
os.environ["AUDIT_DIR"] = os.path.join(TMP, "audit")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{os.path.join(TMP, 'diag.db')}"

from sqlalchemy import func, select  # noqa: E402

from app.storage.db import get_engine, init_db, session_scope  # noqa: E402
from app.storage.models import (  # noqa: E402
    Chunk,
    ContentBlock,
    Paper,
    PaperFigure,
    PaperFormula,
    PaperReference,
    PaperTable,
    Section,
)

MODELS = (Paper, Section, ContentBlock, Chunk, PaperReference)


async def _counts(db, pid: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for model in MODELS:
        col = model.__table__.c.id if model is Paper else model.__table__.c.paper_id
        out[model.__tablename__] = (
            await db.execute(
                select(func.count()).select_from(model).where(col == pid)
            )
        ).scalar_one()
    return out


async def _run(tag: str, build, echo: bool = False) -> None:
    pid = uuid.uuid4().hex
    engine = get_engine()
    engine.echo = echo
    try:
        async with session_scope() as db:
            await build(db, pid)
            await db.flush()
        async with session_scope() as db:
            print(f"[{tag}] OK   {await _counts(db, pid)}")
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic
        print(f"[{tag}] FAIL {type(exc).__name__}: {str(exc).splitlines()[0]}")
    finally:
        engine.echo = False


# --------------------------------------------------------------------------- variants
async def v_baseline(db, pid):
    """The pre-fix code path: bare FK columns, no ORM relationship."""
    paper = Paper(id=pid)
    db.add(paper)
    paper.title = "t"
    paper.status = "ready"
    db.add(ContentBlock(id=uuid.uuid4().hex, paper_id=pid, section_id=None,
                        page_no=1, block_type="text", bbox=[], content="c",
                        order_index=0))


async def v_flush_parent_first(db, pid):
    """Materialise the parent row before the children are queued."""
    paper = Paper(id=pid)
    db.add(paper)
    paper.title = "t"
    paper.status = "ready"
    await db.flush()
    db.add(ContentBlock(id=uuid.uuid4().hex, paper_id=pid, section_id=None,
                        page_no=1, block_type="text", bbox=[], content="c",
                        order_index=0))


async def v_models_with_relationship(db, pid):
    """Only models that DO declare an ORM ``paper`` relationship."""
    paper = Paper(id=pid)
    db.add(paper)
    paper.title = "t"
    paper.status = "ready"
    db.add(Section(id=uuid.uuid4().hex, paper=paper, parent_id=None, level=1,
                   number="1", title="Intro", page_start=1, page_end=1,
                   bbox=[], content="x", order_index=0))
    db.add(Chunk(id=uuid.uuid4().hex, paper=paper, section_id=None, ordinal=0,
                 page_no=1, char_start=0, char_end=1, content="c", token_count=1,
                 kind="body", embedding_ref="", embedding_row=-1))
    db.add(PaperReference(id=uuid.uuid4().hex, paper=paper, index_in_doc=1,
                          raw_text="r", authors=[], title="t", year=None,
                          venue="", doi="", arxiv_id=""))


async def v_relationship_attribute(db, pid):
    """The fixed code path: assign the ORM relationship object."""
    paper = Paper(id=pid)
    db.add(paper)
    paper.title = "t"
    paper.status = "ready"
    db.add(ContentBlock(id=uuid.uuid4().hex, paper=paper, section_id=None,
                        page_no=1, block_type="text", bbox=[], content="c",
                        order_index=0))
    db.add(PaperTable(id=uuid.uuid4().hex, paper=paper, page_no=1, caption="c",
                      markdown="m", csv_text="c", bbox=[]))
    db.add(PaperFigure(id=uuid.uuid4().hex, paper=paper, page_no=1, caption="c",
                       image_path="", bbox=[], description=""))
    db.add(PaperFormula(id=uuid.uuid4().hex, paper=paper, page_no=1, latex="x",
                        raw_text="x", bbox=[]))


async def main() -> None:
    await init_db()
    await _run("echo-fixed-path", v_relationship_attribute, echo=True)
    await _run("old-path-bare-fk-column", v_baseline)
    await _run("flush-parent-first", v_flush_parent_first)
    await _run("models-with-relationship-only", v_models_with_relationship)
    await _run("fixed-path-relationship-object", v_relationship_attribute)


asyncio.run(main())
