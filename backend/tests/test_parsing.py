"""Module 1 — parsing engine: heading tree, columns, references, metadata."""

from __future__ import annotations

import pytest

from app.parsing.headings import build_tree, detect_headings, flatten
from app.parsing.layout import Block


# ----------------------------------------------------------------- headings
def _heading_blocks() -> list[Block]:
    from tests.conftest import make_block

    return [
        make_block("1 Introduction", size=12.0, bold=True),
        make_block("2 Background", size=12.0, bold=True),
        make_block("2.1 Model Architecture", size=11.0, bold=True),
        make_block("2.1.1 Multi-Head Attention", size=10.5, bold=True),
        make_block("This paragraph is ordinary body text that should never be "
                   "detected as a heading candidate at all.", size=10.0),
        make_block("arXiv:1706.03762v1 [cs.CL] 12 Jun 2017", size=8.0),
        make_block("3 Conclusion", size=12.0, bold=True),
    ]


def test_detect_headings_builds_numbered_levels() -> None:
    candidates = detect_headings(_heading_blocks(), body_size=10.0)
    numbers = {c.number: c.level for c in candidates}
    assert numbers["1"] == 1
    assert numbers["2"] == 1
    assert numbers["2.1"] == 2
    assert numbers["2.1.1"] == 3


def test_arxiv_stamp_is_not_a_heading() -> None:
    """Regression: the rotated arXiv stamp used to be parsed as a heading."""
    candidates = detect_headings(_heading_blocks(), body_size=10.0)
    assert not any("arXiv" in c.title for c in candidates)


def test_body_paragraph_is_not_a_heading() -> None:
    candidates = detect_headings(_heading_blocks(), body_size=10.0)
    assert not any("ordinary body text" in c.title for c in candidates)


def test_heading_tree_nesting() -> None:
    candidates = detect_headings(_heading_blocks(), body_size=10.0)
    tree = build_tree(candidates)
    flat = flatten(tree)

    by_number = {n.number: n for n in flat}
    assert "2.1.1" in by_number
    assert by_number["2.1.1"].level == 3
    # A level-3 heading must live under a level-2 ancestor.
    parent = by_number["2.1.1"].parent
    assert parent is not None
    assert parent.number == "2.1"


def test_heading_tree_preserves_order() -> None:
    candidates = detect_headings(_heading_blocks(), body_size=10.0)
    flat = flatten(build_tree(candidates))
    assert flat[0].number == "1"
    assert flat[-1].number == "3"


# --------------------------------------------------------------- real paper
def _parse(path) -> object:
    """Run the (async) parsing pipeline outside of an event loop."""
    import asyncio

    from app.parsing.pipeline import parse_document

    return asyncio.run(parse_document(str(path), paper_id="pytest", do_ocr=False,
                                      transcribe_formulas=False))


def test_real_pdf_heading_tree(attention_pdf) -> None:
    """The 26-level heading tree of the real paper contains 3.2.2."""
    if attention_pdf is None:
        pytest.skip("1706.03762v1.pdf not available locally")
    doc = _parse(attention_pdf)
    numbers = {s["number"] for s in doc.sections}
    assert "3.2.2" in numbers
    assert doc.metadata.title == "Attention Is All You Need"


def test_real_pdf_references_are_structured(attention_pdf) -> None:
    if attention_pdf is None:
        pytest.skip("1706.03762v1.pdf not available locally")
    doc = _parse(attention_pdf)
    assert len(doc.references) >= 70
    sample = doc.references[0]
    assert sample["title"]
    assert sample["authors"]


def test_real_pdf_authors_include_affiliated_researcher(attention_pdf) -> None:
    """The 6th author of the paper is the answer to test question #1."""
    if attention_pdf is None:
        pytest.skip("1706.03762v1.pdf not available locally")
    doc = _parse(attention_pdf)
    names = " ".join(a.name for a in doc.metadata.authors)
    assert "Gomez" in names
    assert len(doc.metadata.authors) >= 8


def test_real_pdf_tables_and_formulas_are_extracted(attention_pdf) -> None:
    if attention_pdf is None:
        pytest.skip("1706.03762v1.pdf not available locally")
    doc = _parse(attention_pdf)
    assert len(doc.tables) >= 3
    assert len(doc.figures) >= 5
    assert len(doc.formulas) >= 5


def test_real_pdf_abstract_and_arxiv_id(attention_pdf) -> None:
    if attention_pdf is None:
        pytest.skip("1706.03762v1.pdf not available locally")
    doc = _parse(attention_pdf)
    assert len(doc.metadata.abstract) > 500
    assert doc.metadata.arxiv_id == "1706.03762"
    assert doc.metadata.version_label == "v1"
