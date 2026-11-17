"""Block classification, the reader contract and the image routes.

The reader showed three real defects that all come from the same root: the
parser had no idea *what* a text block is.  A label inside a figure, a
footnote at the bottom of a page and a running header were all glued into the
body text of a section, and the figures had no URL the browser could load.

The tests below pin the behaviour down:

* a label inside a figure is ``figure_label`` and never reaches the section
* a small text at the bottom of the page is ``footnote`` and never reaches the
  section either
* ordinary running text is *not* mistaken for one of them
* the structure endpoint exposes the page count and the typed blocks
* the figure image route serves a PNG and refuses a path that escapes the
  data directory
"""

from __future__ import annotations

import asyncio
import uuid

import pymupdf
import pytest
from fastapi import HTTPException

from app.parsing import extractors, layout
from app.parsing.headings import HeadingNode, assign_content
from tests.conftest import make_block

PAGE_W, PAGE_H = 612.0, 792.0


def _node(bbox: tuple[float, float, float, float] = (50, 60, 300, 80)) -> HeadingNode:
    return HeadingNode(title="Related Work", number="2", level=1, page_no=1,
                       bbox=list(bbox))


# --------------------------------------------------------------------- captions
def test_caption_blocks_are_recognised() -> None:
    """``Figure 1: ...`` / ``Table 2: ...`` are captions, not body text."""
    figure_caption = make_block("Figure 1: Overall architecture of the pre-training "
                                "and fine-tuning procedure.", size=9.0, y0=300, y1=320)
    table_caption = make_block("Table 2: Results on the GLUE benchmark.", size=9.0,
                               y0=340, y1=356)
    blocks = [figure_caption, table_caption]
    layout.classify_blocks({1: blocks}, page_sizes={1: (PAGE_W, PAGE_H)},
                           body_size=10.0)

    assert figure_caption.block_type == layout.BLOCK_CAPTION
    assert table_caption.block_type == layout.BLOCK_CAPTION


def test_a_caption_stays_in_the_section() -> None:
    """A caption is part of the prose of the section it belongs to."""
    caption = make_block("Figure 1: Overall architecture of the pre-training.", size=9.0,
                         y0=300, y1=320)
    blocks = [caption]
    layout.classify_blocks({1: blocks}, page_sizes={1: (PAGE_W, PAGE_H)}, body_size=10.0)

    node = _node()
    assign_content([node], blocks)
    assert "Overall architecture" in node.content


# --------------------------------------------------------------------- formulas
def test_an_equation_is_split_out_of_its_paragraph() -> None:
    """A PDF block that holds a sentence *and* the equation it introduces
    is split, so the equation can be shown as an image instead of as
    flattened text inside the paragraph.
    """
    prose = layout.Span(text="over target tokens:", bbox=(50, 100, 300, 112),
                        size=10.0, flags=0, font="Helvetica")
    math = layout.Span(text="h0 = UWe + Wp", bbox=(120, 114, 300, 126),
                       size=10.0, flags=0, font="CMMI10")
    math2 = layout.Span(text="hl = transformer_block(hl−1)", bbox=(120, 126, 320, 138),
                        size=10.0, flags=0, font="CMMI10")

    pieces = layout.split_equation_lines(3, [[prose], [math], [math2]])

    assert len(pieces) == 2
    assert pieces[0].block_type == layout.BLOCK_BODY
    assert pieces[0].text == "over target tokens:"
    assert pieces[1].block_type == layout.BLOCK_FORMULA
    assert pieces[1].is_equation is True
    assert pieces[1].text.startswith("h0 = UWe + Wp")
    assert pieces[1].bbox[1] > pieces[0].bbox[3]


def test_a_plain_paragraph_is_not_split() -> None:
    """Prose that merely contains a symbol stays one block."""
    line = layout.Span(text="where U = (u−k, . . . , u−1) is the context vector.",
                       bbox=(50, 100, 400, 112), size=10.0, flags=0, font="Helvetica")
    assert layout.split_equation_lines(3, [[line]]) == []


# ----------------------------------------------------------- printed rows
def _span(text: str, x0: float, y0: float, x1: float, y1: float,
          size: float = 10.0, font: str = "CMMI10") -> layout.Span:
    return layout.Span(text=text, bbox=(x0, y0, x1, y1), size=size, flags=0, font=font)


def test_a_subscript_is_written_as_a_script() -> None:
    """``W`` and ``e`` are one formula, not two letters.

    The subscript is a smaller span that sits lower than the character it
    belongs to; read as plain text the reader shows "We" where the paper
    prints "W_e".
    """
    row = [
        _span("W", 441.1, 463.0, 452.8, 473.0),
        _span("e", 452.8, 466.9, 458.0, 473.8, size=7.0, font="CMMI7"),
    ]
    assert layout._row_texts([row]) == ["W_{e}"]


def test_a_superscript_is_written_as_a_script() -> None:
    """``x`` with the exponent 1 is ``x^{1}``, not ``x1``."""
    row = [
        _span("x", 247.2, 576.1, 252.9, 586.1),
        _span("1", 252.9, 574.3, 257.3, 581.3, size=7.0, font="CMR7"),
    ]
    assert layout._row_texts([row]) == ["x^{1}"]


def test_a_digit_of_the_sentence_is_not_a_script() -> None:
    """A superscript citation is part of the sentence, not of a formula."""
    row = [
        layout.Span(text="challenging", bbox=(100, 550, 160, 560), size=10.0,
                    flags=0, font="Helvetica"),
        layout.Span(text="1", bbox=(160, 545, 164, 552), size=7.0, flags=0,
                    font="CMR7"),
    ]
    assert layout._row_texts([row]) == ["challenging1"]


def test_a_run_in_heading_is_not_pulled_into_the_paragraph() -> None:
    """A heading set in front of its paragraph shares the row but not the text.

    The two lines overlap vertically, yet the heading stands a few points
    away from the paragraph; joining them would glue "Auxiliary training
    objectives" into the sentence.
    """
    heading = layout.Span(text="Auxiliary training objectives",
                          bbox=(108, 118.4, 229.1, 128.4), size=10.0,
                          flags=0, font="Helvetica-Bold")
    body = layout.Span(text="Adding auxiliary unsupervised training objectives",
                       bbox=(239, 118.5, 504, 128.5), size=10.0, flags=0,
                       font="Helvetica")
    rows = layout._visual_rows([[heading], [body]])
    assert len(rows) == 2


def test_the_limit_under_an_operator_belongs_to_the_operator() -> None:
    """``sum_i`` is printed on two rows and has to be read as one.

    The limit underneath the summation sign is a small span in the row below;
    left there it becomes a line of its own.
    """
    operator = _span("X", 260.2, 323.8, 274.6, 333.7, size=10.0, font="CMEX10")
    limit = _span("i", 266.0, 340.1, 268.8, 347.1, size=7.0, font="CMMI7")
    rows = layout._attach_limits([[operator], [limit]])
    assert len(rows) == 1
    assert rows[0] == [operator, limit]


def test_a_small_character_below_the_text_is_not_a_limit() -> None:
    """Only a big operator carries a limit.

    A subscript that happens to sit under a character of the line above is
    part of that line, and the merge must not pull it out of it.
    """
    letter = _span("C", 247.8, 528.4, 256.5, 538.3, size=10.0, font="CMSY10")
    digit = _span("1", 252.9, 574.3, 257.3, 581.3, size=7.0, font="CMR7")
    rows = layout._attach_limits([[letter], [digit]])
    assert len(rows) == 2


def test_the_two_halves_of_a_display_equation_are_joined() -> None:
    """LaTeX cuts a display equation where the big operator stands.

    Equation (1) of the BERT paper arrives as "L1(U) = sum" and
    "i log P(...) (1)".  Left alone the reader shows the equation in two
    halves and crops each of them in the wrong place.
    """
    left = layout.Block(
        text="L1(U) =",
        bbox=(220.7, 323.8, 274.6, 336.9),
        spans=[_span("L", 220.7, 326.1, 226.7, 336.0),
               _span("X", 260.2, 323.8, 274.6, 333.7, font="CMEX10")],
        page_no=3,
        is_equation=True,
    )
    right = layout.Block(
        text="i log P(u|u−k) (1)",
        bbox=(266.0, 325.9, 504.0, 347.1),
        spans=[_span("i", 266.0, 340.1, 268.8, 347.1, size=7.0, font="CMMI7"),
               _span("log", 276.3, 326.1, 289.0, 336.0, font="CMR10"),
               _span("(1)", 492.4, 326.3, 504.0, 336.2, font="NimbusRomNo9L")],
        page_no=3,
        is_equation=True,
    )

    merged = layout._merge_equation_fragments(3, [left, right])

    assert len(merged) == 1
    assert merged[0].bbox == (220.7, 323.8, 504.0, 347.1)
    assert "∑_{i}" in merged[0].text
    assert "(1)" in merged[0].text


def test_two_columns_are_not_joined_into_one_equation() -> None:
    """Two equations side by side in a two-column page stay separate."""
    left = layout.Block(
        text="a = b",
        bbox=(60, 300, 280, 312),
        spans=[_span("a", 60, 300, 80, 312)],
        page_no=1,
        is_equation=True,
    )
    right = layout.Block(
        text="c = d",
        bbox=(306, 300, 500, 312),
        spans=[_span("c", 306, 300, 326, 312)],
        page_no=1,
        is_equation=True,
    )

    merged = layout._merge_equation_fragments(1, [left, right])

    assert len(merged) == 2


def test_a_display_equation_is_not_body_text() -> None:
    """The flattened text of an equation must not become a paragraph.

    The reader shows the cropped image of the equation instead; the extracted
    text ("log P(u|u-k, . . . , u-1; Θ)") reads as noise in the prose.
    """
    equation = make_block("log P(u|u−k, . . . , u−1; Θ)", size=10.0,
                          y0=300, y1=320, x0=200, x1=420)
    prose = make_block("The model is trained with the objective below.", size=10.0,
                        y0=200, y1=220, x1=450)
    blocks = [prose, equation]

    layout.classify_blocks({1: blocks}, page_sizes={1: (PAGE_W, PAGE_H)},
                           formula_bboxes={1: [(200.0, 300.0, 420.0, 320.0)]},
                           body_size=10.0)

    assert equation.block_type == layout.BLOCK_FORMULA
    assert prose.block_type == layout.BLOCK_BODY


def test_a_formula_is_kept_out_of_the_section_text() -> None:
    """An equation is not part of the running text of its section."""
    equation = make_block("∑ log P(u|u−k)", size=10.0, y0=300, y1=320)
    prose = make_block("We minimise the negative log likelihood of the corpus.",
                        size=10.0, y0=200, y1=220, x1=450)
    blocks = [prose, equation]
    layout.classify_blocks({1: blocks}, page_sizes={1: (PAGE_W, PAGE_H)},
                           formula_bboxes={1: [(50.0, 300.0, 300.0, 320.0)]},
                           body_size=10.0)

    node = _node()
    assign_content([node], blocks)
    assert "negative log likelihood" in node.content
    assert "∑" not in node.content


# --------------------------------------------------------------------- mojibake
def test_repair_mojibake_restores_utf8_math_symbols() -> None:
    """A PDF font without a ToUnicode map yields UTF-8 bytes read as latin-1.

    The minus sign arrives as "â\\x88\\x92" and the greek theta as "Î\\x98";
    the bytes are still there, so the string can be decoded once more.
    """
    broken = "log P(u|uâ\x88\x92k, . . . , uâ\x88\x921; Î\x98)"
    assert layout.repair_mojibake(broken) == "log P(u|u−k, . . . , u−1; Θ)"


def test_repair_mojibake_leaves_correct_text_alone() -> None:
    """A clean text must not be touched, not even with latin letters."""
    for text in ("We show 8.9% absolute improvement.",
                 "naïve café — dash",
                 "αβγ δεζ",
                 "plain ascii"):
        assert layout.repair_mojibake(text) == text


# ---------------------------------------------------------------- figure labels
def test_a_label_inside_a_figure_never_reaches_the_section() -> None:
    """The text of a TikZ picture is not body text."""
    heading = make_block("2 Related Work", size=12.0, bold=True, y0=60, y1=80)
    body = make_block("The transformer architecture relies on multi-head self attention "
                      "over the whole sequence of the input.", size=10.0,
                      y0=120, y1=150, x1=450)
    label = make_block("E[CLS] E1 E[SEP] Tok N", size=8.0, y0=200, y1=215,
                       x0=100, x1=250)
    blocks = [heading, body, label]

    layout.classify_blocks({1: blocks}, page_sizes={1: (PAGE_W, PAGE_H)},
                           figure_bboxes={1: [(90.0, 190.0, 260.0, 220.0)]},
                           body_size=10.0)

    assert label.block_type == layout.BLOCK_FIGURE_LABEL
    assert body.block_type == layout.BLOCK_BODY

    node = _node()
    assign_content([node], blocks)
    assert "multi-head self attention" in node.content
    assert "E[CLS]" not in node.content


# --------------------------------------------------------------------- footnotes
def test_a_small_text_at_the_bottom_of_the_page_is_a_footnote() -> None:
    """The BERT style footnote must not end up between two columns of text."""
    body = make_block("The GLUE benchmark contains nine sentence level tasks and three "
                      "different subsets of the data.", size=10.0, y0=300, y1=330)
    footnote = make_block("10https://gluebenchmark.com/leaderboard", size=8.0,
                          y0=700, y1=712)
    blocks = [body, footnote]

    layout.classify_blocks({1: blocks}, page_sizes={1: (PAGE_W, PAGE_H)}, body_size=10.0)

    assert footnote.block_type == layout.BLOCK_FOOTNOTE

    node = _node()
    assign_content([node], blocks)
    assert "gluebenchmark" not in node.content
    assert "GLUE benchmark contains nine" in node.content


def test_running_text_is_not_mistaken_for_a_footnote_or_a_label() -> None:
    """The counter example: ordinary prose stays body text."""
    paragraph = make_block(
        "We propose a simple and effective approach for the joint learning of "
        "representations. The model is trained on eleven different tasks and "
        "evaluated on the GLUE benchmark.", size=10.0, y0=300, y1=340, x1=460)
    bottom_paragraph = make_block(
        "This last paragraph reaches the bottom of the page but it is printed with "
        "the same font size as the rest of the body text.", size=10.0,
        y0=700, y1=740, x1=460)
    blocks = [paragraph, bottom_paragraph]

    layout.classify_blocks({1: blocks}, page_sizes={1: (PAGE_W, PAGE_H)},
                           figure_bboxes={1: [(60.0, 90.0, 200.0, 140.0)]},
                           body_size=10.0)

    assert paragraph.block_type == layout.BLOCK_BODY
    assert bottom_paragraph.block_type == layout.BLOCK_BODY


def test_a_narrow_block_at_the_page_edge_is_a_header() -> None:
    """The rotated arXiv stamp in the margin is a header."""
    stamp = make_block("arXiv:1706.03762v7 [cs.CL] 2 Aug 2023", size=8.0,
                       x0=28, y0=200, x1=44, y1=430)
    blocks = [stamp]

    layout.classify_blocks({1: blocks}, page_sizes={1: (PAGE_W, PAGE_H)}, body_size=10.0)

    assert stamp.block_type == layout.BLOCK_HEADER


# ------------------------------------------------------------------ page number
def test_the_page_number_is_not_body_text() -> None:
    """The folio is set in the body type and stands alone at the bottom.

    Left alone it ends up in the prose and, because it is a short line, the
    heading detector reads it as a section title -- which is how the outline
    of a paper ended up with an entry per page.
    """
    body = make_block("The model is trained on eleven different tasks and evaluated "
                      "on the GLUE benchmark.", size=10.0, y0=300, y1=340, x1=460)
    folio = make_block("12", size=10.0, x0=301, y0=742, x1=311, y1=752)
    blocks = [body, folio]

    layout.classify_blocks({1: blocks}, page_sizes={1: (PAGE_W, PAGE_H)}, body_size=10.0)

    assert folio.block_type == layout.BLOCK_HEADER
    assert body.block_type == layout.BLOCK_BODY

    node = _node()
    assign_content([node], blocks)
    assert "12" not in node.content


def test_a_number_in_the_text_is_still_body_text() -> None:
    """The counter example: "12" in the middle of the page is content."""
    number = make_block("12", size=10.0, x0=120, y0=300, x1=128, y1=312)
    blocks = [number]

    layout.classify_blocks({1: blocks}, page_sizes={1: (PAGE_W, PAGE_H)}, body_size=10.0)

    assert number.block_type != layout.BLOCK_HEADER


# --------------------------------------------------------------------- headings
def test_a_page_number_is_not_a_heading() -> None:
    """A folio must not open a section."""
    from app.parsing.headings import detect_headings

    folio = make_block("12", size=11.0, bold=True, x0=301, y0=742, x1=311, y1=752)
    candidates = detect_headings([folio], body_size=10.0)

    assert candidates == []


def test_a_displayed_equation_is_not_a_heading() -> None:
    """``L1(U) = X`` is a formula, not a section title."""
    from app.parsing.headings import detect_headings

    equation = make_block("L1(U) = X", size=11.0, bold=True, x0=220, y0=323,
                          x1=275, y1=337)
    candidates = detect_headings([equation], body_size=10.0)

    assert candidates == []


def test_a_line_that_introduces_a_formula_is_not_a_heading() -> None:
    """``This gives us the following objective to maximize:`` ends with a colon."""
    from app.parsing.headings import detect_headings

    lead_in = make_block("This gives us the following objective to maximize:",
                         size=11.0, bold=True, x0=108, y0=595, x1=309, y1=605)
    candidates = detect_headings([lead_in], body_size=10.0)

    assert candidates == []


def test_a_real_heading_is_still_a_heading() -> None:
    """The counter example: a numbered title keeps its place in the tree."""
    from app.parsing.headings import detect_headings

    heading = make_block("2.1\nUnsupervised pre-training", size=11.0, bold=True,
                         x0=108, y0=279, x1=243, y1=289)
    candidates = detect_headings([heading], body_size=10.0)

    assert len(candidates) == 1
    assert candidates[0].title == "Unsupervised pre-training"
    assert candidates[0].number == "2.1"


# ------------------------------------------------------------ section ownership
def test_a_block_belongs_to_the_section_that_is_open_at_its_position() -> None:
    """The blocks carry the section they belong to.

    The reader groups the paragraphs by section; without the link it has to
    fall back to the page number, which puts the abstract under the heading of
    the introduction.
    """
    from app.parsing.headings import HeadingNode, section_keys_for_blocks

    abstract = HeadingNode(title="Abstract", number="", level=1, page_no=1,
                           bbox=[283, 242, 328, 254], order_index=1)
    introduction = HeadingNode(title="Introduction", number="1", level=1, page_no=1,
                                bbox=[108, 465, 191, 477], order_index=2)
    related = HeadingNode(title="Related Work", number="2", level=1, page_no=2,
                          bbox=[108, 376, 197, 388], order_index=3)
    blocks = [
        make_block("The abstract of the paper.", page_no=1, y0=270, y1=290),
        make_block("The first paragraph of the introduction.", page_no=1,
                   y0=500, y1=520),
        make_block("The last paragraph of page one.", page_no=1, y0=600, y1=620),
        make_block("A paragraph of the related work.", page_no=2, y0=420, y1=440),
    ]

    keys = section_keys_for_blocks([abstract, introduction, related], blocks)

    assert keys == ["1:1", "1:2", "1:2", "2:3"]


def test_a_block_before_the_first_heading_has_no_section() -> None:
    """The text that opens the paper belongs to no section."""
    from app.parsing.headings import HeadingNode, section_keys_for_blocks

    introduction = HeadingNode(title="Introduction", number="1", level=1, page_no=1,
                                bbox=[108, 465, 191, 477], order_index=0)
    blocks = [make_block("The title and the authors of the paper.", page_no=1,
                         y0=100, y1=140)]

    assert section_keys_for_blocks([introduction], blocks) == [None]


# -------------------------------------------------------------- structure API
def test_paper_structure_exposes_page_count_and_typed_blocks() -> None:
    """``GET /api/papers/{id}/structure`` carries the blocks and the page count."""
    from app.services import papers as paper_service
    from app.storage.db import dispose_db, init_db, session_scope
    from app.storage.models import ContentBlock, Paper

    paper_id = f"struct{uuid.uuid4().hex[:8]}"

    async def scenario() -> dict:
        await init_db()
        try:
            async with session_scope() as db:
                db.add(Paper(id=paper_id, title="A Paper", page_count=16))
                db.add(ContentBlock(id=f"{paper_id}b1", paper_id=paper_id, page_no=3,
                                    block_type="body", bbox=[1, 2, 3, 4],
                                    content="The body of the paper.", order_index=0))
                db.add(ContentBlock(id=f"{paper_id}b2", paper_id=paper_id, page_no=3,
                                    block_type="figure_label", bbox=[5, 6, 7, 8],
                                    content="E[CLS]", order_index=1))
            return await paper_service.get_paper_structure(paper_id)
        finally:
            await dispose_db()

    data = asyncio.run(scenario())

    assert data["page_count"] == 16
    assert len(data["blocks"]) == 2
    assert [b["order_index"] for b in data["blocks"]] == [0, 1]
    assert {b["type"] for b in data["blocks"]} <= layout.BLOCK_TYPES
    assert data["blocks"][1]["type"] == "figure_label"
    assert data["blocks"][0]["content"].startswith("The body")


def test_the_reader_page_list_uses_the_page_count() -> None:
    """``GET /api/reader/{id}/pages`` used to return an empty list."""
    from app.api.routers import reader
    from app.storage.db import dispose_db, init_db, session_scope
    from app.storage.models import Paper

    paper_id = f"pages{uuid.uuid4().hex[:8]}"

    async def scenario() -> dict:
        await init_db()
        try:
            async with session_scope() as db:
                db.add(Paper(id=paper_id, title="A Paper", page_count=3))
            return await reader.pages(paper_id)
        finally:
            await dispose_db()

    data = asyncio.run(scenario())

    assert data["page_count"] == 3
    assert [p["page_no"] for p in data["pages"]] == [1, 2, 3]


# --------------------------------------------------------------- image routes
def test_the_figure_image_route_serves_a_png_and_refuses_traversal(
        tmp_path, monkeypatch) -> None:
    """The browser needs a real URL for a figure - and only inside the data dir."""
    from app.api.routers import reader
    from app.core.config import get_settings
    from app.storage.db import dispose_db, init_db, session_scope
    from app.storage.models import Paper, PaperFigure

    paper_id = f"fig{uuid.uuid4().hex[:8]}"
    figure_id = f"{paper_id}f0"
    evil_id = f"{paper_id}f1"
    settings = get_settings()
    # The figure image has to live inside the data directory - the test uses
    # a throw-away one instead of the real data dir.
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    png_path = pages_dir / f"{paper_id}_p3_fig0.png"
    png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

    async def scenario() -> tuple:
        await init_db()
        try:
            async with session_scope() as db:
                db.add(Paper(id=paper_id, title="A Paper", page_count=3))
                db.add(PaperFigure(id=figure_id, paper_id=paper_id, page_no=3,
                                   caption="Figure 1", image_path=str(png_path),
                                   bbox=[0, 0, 10, 10]))
                # A path that tries to escape the data directory.
                db.add(PaperFigure(id=evil_id, paper_id=paper_id, page_no=3,
                                   caption="evil", image_path="../../etc/passwd",
                                   bbox=[0, 0, 10, 10]))
            good = await reader.figure_image(paper_id, figure_id)
            with pytest.raises(HTTPException) as exc:
                await reader.figure_image(paper_id, evil_id)
            return good, exc.value
        finally:
            await dispose_db()

    response, error = asyncio.run(scenario())

    assert response.status_code == 200
    assert response.media_type == "image/png"
    assert error.status_code in (400, 404)
    assert "passwd" not in str(error.detail).lower()


def test_the_reader_exposes_the_image_routes() -> None:
    """The URL contract the frontend relies on."""
    from app.api.routers import reader

    paths = {route.path for route in reader.router.routes}
    assert "/reader/{paper_id}/figures/{figure_id}/image" in paths
    assert "/reader/{paper_id}/tables/{table_id}/image" in paths


# --------------------------------------------------------------------- tables
def _ruled_table_page(caption: str = "Table 1: Comparison of the models.",
                       caption_above: bool = True) -> pymupdf.Document:
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    x0, y0, x1, y1 = 70.0, 140.0, 320.0, 210.0
    for x in (x0, (x0 + x1) / 2, x1):
        page.draw_line((x, y0), (x, y1))
    for y in (y0, (y0 + y1) / 2, y1):
        page.draw_line((x0, y), (x1, y))
    page.insert_text((x0 + 6, y0 + 18), "Model", fontsize=9)
    page.insert_text((x0 + 96, y0 + 18), "F1", fontsize=9)
    page.insert_text((x0 + 6, y0 + 48), "BERT", fontsize=9)
    page.insert_text((x0 + 96, y0 + 48), "82.1", fontsize=9)
    if caption_above:
        page.insert_text((x0, y0 - 12), caption, fontsize=9)
    else:
        page.insert_text((x0, y1 + 20), caption, fontsize=9)
    return doc


def test_a_table_gets_its_caption() -> None:
    """``paper_tables.caption`` used to be empty for every table."""
    doc = _ruled_table_page()
    try:
        page = doc[0]
        blocks = layout.extract_blocks(page, 1)
        tables = extractors.extract_tables(page, 1, blocks)
    finally:
        doc.close()

    assert tables, "the ruled table was not detected"
    assert tables[0].caption.startswith("Table 1")
    assert tables[0].n_rows >= 2


def test_a_booktabs_table_is_found_by_the_text_strategy() -> None:
    """A table without vertical rules is invisible to the line strategy."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    # booktabs: only horizontal rules (they are thin filled rectangles).
    for y in (150.0, 175.0, 205.0):
        page.draw_rect(pymupdf.Rect(70, y, 330, y + 0.8), color=(0, 0, 0),
                       fill=(0, 0, 0))
    for i, (name, score) in enumerate([("Model", "F1"), ("BERT", "82.1"),
                                       ("RoBERTa", "83.4")]):
        y = 165 + i * 25
        page.insert_text((75, y), name, fontsize=9)
        page.insert_text((160, y), score, fontsize=9)
    page.insert_text((70, 130), "Table 1: Results of the models.", fontsize=9)
    try:
        blocks = layout.extract_blocks(page, 1)
        tables = extractors.extract_tables(page, 1, blocks)
    finally:
        doc.close()

    assert tables, "the booktabs table was not detected"
    assert tables[0].n_cols >= 2
    assert tables[0].n_rows >= 2
    assert "82.1" in tables[0].markdown
    assert tables[0].caption.startswith("Table 1")


def test_running_text_is_not_mistaken_for_a_table() -> None:
    """The text strategy sees one column per word - that is not a table."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    for i in range(8):
        page.insert_text((72, 100 + i * 30),
                         "We propose a simple and effective approach for the joint "
                         "learning of representations of words and phrases.",
                         fontsize=10)
    try:
        blocks = layout.extract_blocks(page, 1)
        tables = extractors.extract_tables(page, 1, blocks)
    finally:
        doc.close()

    assert tables == []


def test_a_chart_is_not_a_table() -> None:
    """The grid of a chart and the frame of a legend look like a table.

    The ruling-line strategy reads them as one, and the figure is then
    dropped because it overlaps the "table".  The regions of the pictures are
    therefore read first, and a table inside a picture is not a table.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    # A plot with its axes and a legend box.
    for i in range(6):
        page.draw_line((100, 120 + i * 20), (300, 120 + i * 20))
    for i in range(6):
        page.draw_line((100 + i * 40, 120), (100 + i * 40, 220))
    page.draw_rect(pymupdf.Rect(320, 120, 430, 180), color=(0, 0, 0), width=1)
    try:
        blocks = layout.extract_blocks(page, 1)
        regions = extractors.figure_regions(page, blocks)
        assert regions, "the chart was not recognised as a picture"
        tables = extractors.extract_tables(page, 1, blocks, exclude=regions)
    finally:
        doc.close()

    assert tables == []


def test_the_content_of_a_table_is_not_a_heading() -> None:
    """The numbers of a results table stand in their own column.

    ``10 0`` is a row of a table, not a section title.
    """
    from app.parsing.headings import detect_headings

    cell = make_block("10 0", size=11.0, bold=True, x0=140, y0=400, x1=160, y1=412)
    candidates = detect_headings(
        [cell], body_size=10.0,
        excluded_regions={1: [(120.0, 390.0, 200.0, 420.0)]},
    )

    assert candidates == []


# -------------------------------------------------------------------- figures
def test_a_tikz_style_picture_becomes_one_figure(tmp_path) -> None:
    """A picture drawn as many small boxes has no big rectangle of its own."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    for i in range(4):
        x = 70 + i * 120
        page.draw_rect(pymupdf.Rect(x, 90, x + 90, 140), color=(0, 0, 0), width=1)
        page.insert_text((x + 10, 120), f"Box {i}", fontsize=9)
    path = tmp_path / "tikz.pdf"
    doc.save(str(path))
    doc.close()

    doc = pymupdf.open(str(path))
    try:
        page = doc[0]
        blocks = layout.extract_blocks(page, 1)
        figures = extractors.extract_figures(page, 1, blocks, str(tmp_path), "tikz")
    finally:
        doc.close()

    assert len(figures) == 1
    x0, y0, x1, y1 = figures[0].bbox
    assert x0 <= 70 and x1 >= 480
    assert y0 <= 90 and y1 >= 140


def test_a_plain_text_page_is_not_a_figure(tmp_path) -> None:
    """A page of running text (and a header rule) must not become a figure."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.draw_line((72, 60), (540, 60))  # the rule under a running header
    for i in range(12):
        page.insert_text((72, 100 + i * 22),
                         "An ordinary paragraph of running text without any drawing.",
                         fontsize=10)
    path = tmp_path / "text.pdf"
    doc.save(str(path))
    doc.close()

    doc = pymupdf.open(str(path))
    try:
        page = doc[0]
        blocks = layout.extract_blocks(page, 1)
        figures = extractors.extract_figures(page, 1, blocks, str(tmp_path), "text")
    finally:
        doc.close()

    assert figures == []


# ---------------------------------------------------------------- end to end
def _mini_paper(path) -> None:
    """A one page paper with a heading, a figure, a caption and a footnote."""
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((72, 80), "1 Introduction", fontsize=12)
    page.insert_text((72, 110),
                     "We study the architecture of a transformer model and show that "
                     "the pre-training objective drives the representation quality.",
                     fontsize=10)
    for i in range(4):
        x = 70 + i * 120
        page.draw_rect(pymupdf.Rect(x, 200, x + 90, 250), color=(0, 0, 0), width=1)
        page.insert_text((x + 10, 230), f"E{i}", fontsize=8)
    page.insert_text((72, 275), "Figure 1: The architecture of the model.", fontsize=9)
    page.insert_text((72, 740), "1https://example.org/footnote", fontsize=8)
    doc.save(str(path))
    doc.close()


def test_the_pipeline_keeps_figure_labels_and_footnotes_out_of_the_text(tmp_path) -> None:
    """The whole chain: parse -> figure -> classify -> section -> blocks."""
    from app.parsing.pipeline import parse_document

    pdf_path = tmp_path / "mini.pdf"
    _mini_paper(pdf_path)

    parsed = asyncio.run(parse_document(str(pdf_path), paper_id="mini",
                                        out_dir=str(tmp_path), do_ocr=False,
                                        transcribe_formulas=False))

    types = {b["block_type"] for b in parsed.blocks}
    assert layout.BLOCK_FIGURE_LABEL in types, types
    assert layout.BLOCK_FOOTNOTE in types, types
    assert layout.BLOCK_CAPTION in types, types
    assert layout.BLOCK_BODY in types, types

    content = "\n".join(section["content"] for section in parsed.sections)
    assert "pre-training objective" in content
    assert "E0" not in content
    assert "example.org/footnote" not in content

    # The plain text of the paper (and therefore the retrieval corpus) is
    # free of the footnotes and of the labels inside the figure.
    assert "example.org/footnote" not in parsed.full_text
    assert "E0" not in parsed.full_text

    # The figure was extracted and carries the caption.
    assert parsed.figures
    assert parsed.figures[0]["caption"].startswith("Figure 1")
