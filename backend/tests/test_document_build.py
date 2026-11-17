"""The reading view: reading order, sentence alignment and the empty fallback.

``build_document`` joins five stored streams (sections, content blocks, tables,
figures, formulas) and the cached translation into one flat document.  The
tests below pin down the three properties the reader depends on:

* the blocks are in reading order, so a table stands where it stands on the
  page and not at the end of the document;
* a sentence keeps the translation of the segment it came from, and a heading
  or a caption finds its translation by its text;
* a paper that was never translated still yields a document, every ``target``
  is simply empty.

Everything runs on the throw-away SQLite of ``conftest`` with hand-built
structures, so no GPU, no LLM and no real PDF are involved.  ``app.render.spans``
is stubbed: it is the table-cell vocabulary, not the subject of these tests.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
import uuid
from typing import Any

import pytest
from fastapi import HTTPException

from app.render.document import build_document
from app.render.model import Cell, Document
from app.storage.db import dispose_db, init_db, session_scope
from app.storage.models import (
    ContentBlock,
    Paper,
    PaperFigure,
    PaperFormula,
    PaperTable,
    Section,
    Translation,
)

#: A table as the parser stores it: a Markdown pipe table with a header row.
TABLE_MARKDOWN = "| Model | F1 |\n| --- | --- |\n| BERT | 82.1 |"


def _fake_parse_markdown(markdown: str) -> tuple[list[list[Cell]], int]:
    """A minimal stand-in for ``app.render.spans.parse_markdown``."""
    rows: list[list[Cell]] = []
    header_rows = 0
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= set("|-: "):
            continue
        cells = [Cell(text=part.strip())
                 for part in stripped.strip("|").split("|")]
        if not rows:
            header_rows = 1
        rows.append(cells)
    return rows, header_rows


@pytest.fixture
def stub_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the stub, whether or not the real module exists yet."""
    module = types.ModuleType("app.render.spans")
    module.parse_markdown = _fake_parse_markdown  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.render.spans", module)


def _paper_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


def _run(paper: Paper, rows: list[Any], paper_id: str) -> Document | None:
    """Persist ``rows`` with ``paper`` and build the document of ``paper_id``."""

    async def scenario() -> Document | None:
        await init_db()
        try:
            async with session_scope() as db:
                db.add(paper)
                # The parent row is flushed first: a child INSERT that runs
                # before its paper exists is rejected by SQLite.
                await db.flush()
                for row in rows:
                    db.add(row)
            return await build_document(paper_id)
        finally:
            await dispose_db()

    return asyncio.run(scenario())


# ------------------------------------------------------------------- ordering
def test_blocks_come_back_in_reading_order(stub_spans: None) -> None:
    """Page first, then the top of the block, then its left edge."""
    paper_id = _paper_id("ord")
    paper = Paper(id=paper_id, title="Ordering", abstract="")
    intro = Section(id=f"{paper_id}s1", paper=paper, level=1, number="1",
                    title="Introduction", page_start=1, bbox=[72, 100, 300, 112],
                    order_index=0)
    setup = Section(id=f"{paper_id}s2", paper=paper, level=2, number="1.1",
                    title="Setup", page_start=1, bbox=[72, 150, 300, 162],
                    order_index=1)
    rows: list[Any] = [
        intro, setup,
        ContentBlock(id=f"{paper_id}b1", paper=paper, page_no=1, block_type="body",
                     bbox=[72, 200, 500, 220],
                     content="First paragraph. Second sentence.", order_index=0),
        ContentBlock(id=f"{paper_id}b2", paper=paper, page_no=1, block_type="body",
                     bbox=[72, 400, 500, 420],
                     content="Last paragraph of page one.", order_index=1),
        ContentBlock(id=f"{paper_id}b3", paper=paper, page_no=2, block_type="body",
                     bbox=[300, 120, 560, 140],
                     content="Right column paragraph.", order_index=2),
        ContentBlock(id=f"{paper_id}b4", paper=paper, page_no=2, block_type="body",
                     bbox=[72, 120, 280, 140],
                     content="Left column paragraph.", order_index=3),
        # The caption of the table also exists as a content block; it must not
        # show up as a second paragraph.
        ContentBlock(id=f"{paper_id}bc", paper=paper, page_no=1,
                     block_type="caption", bbox=[72, 300, 400, 315],
                     content="Table 1: Results.", order_index=4),
        # An equation is stored as a content block *and* as a formula row; only
        # the row (LaTeX + cropped image) is rendered.
        ContentBlock(id=f"{paper_id}bq", paper=paper, page_no=1,
                     block_type="formula", bbox=[200, 250, 400, 270],
                     content="y = x", order_index=5),
        PaperTable(id=f"{paper_id}t1", paper=paper, page_no=1,
                   caption="Table 1: Results.", markdown=TABLE_MARKDOWN,
                   bbox=[72, 300, 400, 340]),
        PaperFormula(id=f"{paper_id}x1", paper=paper, page_no=1,
                     latex="y = x \\tag{3}", raw_text="y = x (3)",
                     bbox=[200, 250, 400, 270]),
        PaperFigure(id=f"{paper_id}f1", paper=paper, page_no=1,
                    caption="Figure 1: The model.", image_path="",
                    bbox=[72, 450, 400, 560]),
    ]

    document = _run(paper, rows, paper_id)

    assert document is not None
    assert [(block.kind, block.id) for block in document.blocks] == [
        ("heading", intro.id),
        ("heading", setup.id),
        ("paragraph", f"{paper_id}b1"),
        ("formula", f"{paper_id}x1"),
        ("table", f"{paper_id}t1"),
        ("paragraph", f"{paper_id}b2"),
        ("figure", f"{paper_id}f1"),
        # Same height on page two: the left column is read first.
        ("paragraph", f"{paper_id}b4"),
        ("paragraph", f"{paper_id}b3"),
    ]


def test_the_counts_and_the_heading_levels_match_the_blocks(stub_spans: None) -> None:
    """A top level section is level 2, its child is level 3."""
    paper_id = _paper_id("lv")
    paper = Paper(id=paper_id, title="Levels", abstract="")
    intro = Section(id=f"{paper_id}s1", paper=paper, level=1, number="2",
                    title="Related Work", page_start=1, bbox=[72, 90, 300, 102],
                    order_index=0)
    child = Section(id=f"{paper_id}s2", paper=paper, level=2, number="2.1",
                    title="Attention", page_start=1, bbox=[72, 120, 300, 132],
                    order_index=1)
    grandchild = Section(id=f"{paper_id}s3", paper=paper, level=3, number="2.1.1",
                         title="Scaled Dot Product", page_start=1,
                         bbox=[72, 150, 300, 162], order_index=2)
    rows: list[Any] = [
        intro, child, grandchild,
        ContentBlock(id=f"{paper_id}b1", paper=paper, page_no=1, block_type="body",
                     bbox=[72, 200, 500, 220], content="A paragraph.", order_index=0),
        PaperTable(id=f"{paper_id}t1", paper=paper, page_no=1, caption="",
                   markdown=TABLE_MARKDOWN, bbox=[72, 300, 400, 340]),
    ]

    document = _run(paper, rows, paper_id)

    assert document is not None
    headings = [block for block in document.blocks if block.kind == "heading"]
    assert [block.level for block in headings] == [2, 3, 4]
    assert [block.text for block in headings] == [
        "2 Related Work", "2.1 Attention", "2.1.1 Scaled Dot Product",
    ]
    assert document.counts["heading"] == 3
    assert document.counts["paragraph"] == 1
    assert document.counts["table"] == 1
    assert document.counts["total"] == len(document.blocks) == 5

    table = next(block for block in document.blocks if block.kind == "table")
    assert table.header_rows == 1
    assert [[cell.text for cell in row] for row in table.rows] == [
        ["Model", "F1"], ["BERT", "82.1"],
    ]
    assert table.image_url == f"/api/reader/{paper_id}/tables/{paper_id}t1/image"


def test_a_broken_table_does_not_break_the_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """A table whose cells cannot be read still shows its caption and image."""
    def _explode(markdown: str) -> tuple[list[list[Cell]], int]:
        raise ValueError("ragged row")

    module = types.ModuleType("app.render.spans")
    module.parse_markdown = _explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.render.spans", module)

    paper_id = _paper_id("bad")
    paper = Paper(id=paper_id, title="Broken table", abstract="")
    rows: list[Any] = [
        PaperTable(id=f"{paper_id}t1", paper=paper, page_no=1,
                   caption="Table 1: Results.", markdown=TABLE_MARKDOWN,
                   bbox=[72, 300, 400, 340]),
    ]

    document = _run(paper, rows, paper_id)

    assert document is not None
    table = next(block for block in document.blocks if block.kind == "table")
    assert table.rows == []
    assert table.header_rows == 0
    assert table.caption == "Table 1: Results."
    assert table.image_url == f"/api/reader/{paper_id}/tables/{paper_id}t1/image"


# ---------------------------------------------------------------- translation
def _bilingual_paper(paper_id: str) -> Paper:
    return Paper(id=paper_id, title="Attention For All",
                 authors=[{"name": "Ada Lovelace", "affiliations": [0]},
                          {"name": "Bo Chen", "affiliations": [1]}],
                 abstract="We study attention. It works well.")


def _bilingual_rows(paper_id: str, paper: Paper,
                    segments: list[dict[str, Any]] | None = None) -> list[Any]:
    """The rows of the bilingual fixture; ``segments=None`` means untranslated.

    The translation row is only created when it is wanted: attaching it to
    ``paper`` also attaches it to ``Paper.translations``, and the cascade would
    persist it even if it is left out of the list.
    """
    rows: list[Any] = [
        Section(id=f"{paper_id}s1", paper=paper, level=1, number="1",
                title="Introduction", page_start=1, bbox=[72, 90, 300, 102],
                order_index=0),
        ContentBlock(id=f"{paper_id}b1", paper=paper, page_no=1, block_type="body",
                     bbox=[72, 200, 500, 220],
                     content="We study attention. It works well.", order_index=0),
        PaperTable(id=f"{paper_id}t1", paper=paper, page_no=1,
                   caption="Table 1: Results.", markdown=TABLE_MARKDOWN,
                   bbox=[72, 300, 400, 340]),
    ]
    if segments is None:
        return rows
    rows.append(Translation(id=f"{paper_id}tr", paper=paper, locale="zh",
                            markdown="", segments=segments))
    return rows


#: The translation of the fixture paper, in the order the translator stored it.
_SEGMENTS: list[dict[str, Any]] = [
    {"index": 1, "source": "Attention For All",
     "translation": "面向所有人的注意力", "kind": "title",
     "block_id": "", "page_no": 1},
    {"index": 2, "source": "Introduction", "translation": "引言",
     "kind": "heading", "block_id": "", "page_no": 1},
    {"index": 3, "source": "We study attention.",
     "translation": "我们研究注意力。", "kind": "paragraph",
     "block_id": "BLOCK", "page_no": 1},
    {"index": 4, "source": "It works well.", "translation": "效果很好。",
     "kind": "paragraph", "block_id": "BLOCK", "page_no": 1},
    {"index": 5, "source": "Table 1: Results.", "translation": "表 1：结果。",
     "kind": "caption", "block_id": "BLOCKCAPTION", "page_no": 1},
]


def _segments_for(paper_id: str) -> list[dict[str, Any]]:
    """``_SEGMENTS`` with the block ids resolved for this paper."""
    out: list[dict[str, Any]] = []
    for segment in _SEGMENTS:
        resolved = dict(segment)
        if resolved["block_id"] == "BLOCK":
            resolved["block_id"] = f"{paper_id}b1"
        elif resolved["block_id"] == "BLOCKCAPTION":
            resolved["block_id"] = f"{paper_id}bc"
        out.append(resolved)
    return out


def test_a_translation_reaches_the_sentence_it_belongs_to(stub_spans: None) -> None:
    """Sentences pair by block id, headings and captions by their text."""
    paper_id = _paper_id("bi")
    paper = _bilingual_paper(paper_id)

    document = _run(paper, _bilingual_rows(paper_id, paper, _segments_for(paper_id)),
                    paper_id)

    assert document is not None
    assert document.title_target == "面向所有人的注意力"
    assert document.authors == ["Ada Lovelace", "Bo Chen"]
    # No segment holds the whole abstract: it is joined from its sentences.
    assert document.abstract_target == "我们研究注意力。 效果很好。"

    heading = next(block for block in document.blocks if block.kind == "heading")
    assert heading.text == "1 Introduction"
    assert heading.target == "引言"

    paragraph = next(block for block in document.blocks if block.kind == "paragraph")
    assert [sentence.source for sentence in paragraph.sentences] == [
        "We study attention.", "It works well.",
    ]
    assert [sentence.target for sentence in paragraph.sentences] == [
        "我们研究注意力。", "效果很好。",
    ]
    assert [sentence.block_id for sentence in paragraph.sentences] == [f"{paper_id}b1"] * 2
    assert paragraph.target == "我们研究注意力。 效果很好。"
    assert document.counts["sentences"] == 2
    assert document.counts["translated"] == 2

    table = next(block for block in document.blocks if block.kind == "table")
    assert table.caption == "Table 1: Results."
    assert table.caption_target == "表 1：结果。"


def test_one_untranslated_sentence_keeps_an_empty_target(stub_spans: None) -> None:
    """A missing translation is an empty string, never the English source."""
    paper_id = _paper_id("half")
    paper = _bilingual_paper(paper_id)
    rows = _bilingual_rows(paper_id, paper, [
        {"index": 1, "source": "We study attention.",
         "translation": "我们研究注意力。", "kind": "paragraph",
         "block_id": f"{paper_id}b1", "page_no": 1},
        {"index": 2, "source": "It works well.", "translation": "",
         "kind": "paragraph", "block_id": f"{paper_id}b1", "page_no": 1},
    ])

    document = _run(paper, rows, paper_id)

    assert document is not None
    paragraph = next(block for block in document.blocks if block.kind == "paragraph")
    assert [sentence.target for sentence in paragraph.sentences] == [
        "我们研究注意力。", "",
    ]
    assert paragraph.target == "我们研究注意力。"
    assert document.title_target == ""
    assert document.counts["translated"] == 1


def test_without_a_translation_every_target_is_empty(stub_spans: None) -> None:
    """An untranslated paper still yields a readable document."""
    paper_id = _paper_id("raw")
    paper = _bilingual_paper(paper_id)
    rows = _bilingual_rows(paper_id, paper)

    document = _run(paper, rows, paper_id)

    assert document is not None
    assert document.lang == "zh"
    assert document.title_target == ""
    assert document.abstract_target == ""
    heading = next(block for block in document.blocks if block.kind == "heading")
    assert (heading.text, heading.target) == ("1 Introduction", "")
    paragraph = next(block for block in document.blocks if block.kind == "paragraph")
    # The prose is still split into sentences, so the reader can highlight a
    # sentence and ask for its translation later.
    assert [sentence.source for sentence in paragraph.sentences] == [
        "We study attention.", "It works well.",
    ]
    assert [sentence.target for sentence in paragraph.sentences] == ["", ""]
    assert paragraph.target == ""
    table = next(block for block in document.blocks if block.kind == "table")
    assert table.caption_target == ""
    assert document.counts["translated"] == 0
    assert document.counts["sentences"] == 2


# ------------------------------------------------------------------- formulas
def test_a_formula_carries_its_number_and_its_image(stub_spans: None) -> None:
    """The number comes from ``\\tag{}`` or from the printed ``(3)``."""
    paper_id = _paper_id("fx")
    paper = Paper(id=paper_id, title="Formulas", abstract="")
    rows: list[Any] = [
        PaperFormula(id=f"{paper_id}x1", paper=paper, page_no=1,
                     latex="L = -\\sum_i \\log P(u_i) \\tag{3}",
                     raw_text="L = -sum i log P(ui) (3)", bbox=[100, 200, 500, 220]),
        PaperFormula(id=f"{paper_id}x2", paper=paper, page_no=1, latex="",
                     raw_text="y = a x + b (7)", bbox=[100, 300, 500, 320]),
    ]

    document = _run(paper, rows, paper_id)

    assert document is not None
    formulas = [block for block in document.blocks if block.kind == "formula"]
    assert [block.number for block in formulas] == ["3", "7"]
    assert formulas[0].latex.startswith("L = -\\sum")
    # Without a LaTeX transcription the flattened text is still readable.
    assert formulas[1].latex == "y = a x + b (7)"
    assert formulas[0].image_url == f"/api/reader/{paper_id}/formulas/{paper_id}x1/image"
    assert document.counts["formula"] == 2


# --------------------------------------------------------------------- router
def test_the_document_route_is_registered_and_404s_for_an_unknown_paper() -> None:
    """The frontend URL contract, and the "paper not found" status."""
    from app.api.routers import document as document_router

    paths = {route.path for route in document_router.router.routes}
    assert "/reader/{paper_id}/document" in paths

    async def scenario() -> HTTPException:
        await init_db()
        try:
            with pytest.raises(HTTPException) as caught:
                await document_router.get_document("does-not-exist")
            return caught.value
        finally:
            await dispose_db()

    error = asyncio.run(scenario())
    assert error.status_code == 404


# ------------------------------------------------------- the model contract
def test_to_dict_serialises_paragraphs_and_tables(stub_spans: None) -> None:
    """The route returns ``document.to_dict()``; that must be JSON-able."""
    paper_id = _paper_id("ser")
    paper = _bilingual_paper(paper_id)

    document = _run(paper, _bilingual_rows(paper_id, paper, _segments_for(paper_id)),
                    paper_id)

    assert document is not None
    payload = document.to_dict()
    assert payload["paper_id"] == paper_id
    assert payload["title_target"] == "面向所有人的注意力"
    paragraph = next(block for block in payload["blocks"] if block["kind"] == "paragraph")
    assert paragraph["sentences"][0] == {"id": "3", "source": "We study attention.",
                                         "target": "我们研究注意力。",
                                         "block_id": f"{paper_id}b1"}
    table = next(block for block in payload["blocks"] if block["kind"] == "table")
    assert table["rows"][0][0] == {"text": "Model", "target": "", "rowspan": 1,
                                   "colspan": 1, "header": False}
    # The whole payload has to survive the JSON encoder of FastAPI.
    assert json.dumps(payload, ensure_ascii=False)
