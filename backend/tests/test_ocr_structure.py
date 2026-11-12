"""Module 1 — reconstructing the structure of a scanned document.

A scanned PDF has no text layer and no typography, so the heading tree and the
front matter have to be rebuilt from the Markdown that the vision model
produced.  These tests use a verbatim excerpt of the real OCR output for
``1706.03762v1_img.pdf`` so they run without a live model.
"""

from __future__ import annotations

import pytest

from app.parsing.headings import build_tree, flatten
from app.parsing.metadata import extract_metadata_from_markdown
from app.parsing.ocr import markdown_to_blocks


# A verbatim excerpt of the OCR output for page 1 of the scanned paper.
OCR_PAGE_1 = """arXiv:1706.03762v1 [cs.CL] 12 Jun 2017

# Attention Is All You Need

Ashish Vaswani$^{*}$
Google Brain
avaswani@google.com

Noam Shazeer$^{*}$
Google Brain
noam@google.com

Aidan N. Gomez$^{\\dagger}$
University of Toronto
aidan@cs.toronto.edu

## Abstract

The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks in an encoder-decoder configuration.

## 1 Introduction

Recurrent neural networks, long short-term memory [12] and gated recurrent [7]
neural networks in particular, have been firmly established as state of the art.

$^{*}$Equal contribution. Listing order is random.
"""

OCR_PAGE_3 = """## 3.2 Attention

An attention function can be described as mapping a query and a set of
key-value pairs to an output.

### 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2).

$$
\\mathrm{Attention}(Q, K, V) = \\mathrm{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V \\tag{1}
$$
"""

OCR_REFERENCES_PAGE = """## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, and Jakob Uszkoreit. Attention
is all you need. In *Advances in Neural Information Processing Systems*, 2017.

[2] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization.
In *ICLR*, 2015.
"""


# ------------------------------------------------------------------ blocks
def test_markdown_to_blocks_extracts_heading_levels() -> None:
    blocks, candidates = markdown_to_blocks(OCR_PAGE_1, page_no=1)
    assert candidates
    by_title = {c.title: c for c in candidates}
    assert "Attention Is All You Need" in by_title
    assert by_title["Attention Is All You Need"].level == 1
    assert by_title["Abstract"].level == 2
    assert by_title["1 Introduction"].number == "1"


def test_markdown_to_blocks_uses_printed_number_for_the_level() -> None:
    _blocks, candidates = markdown_to_blocks(OCR_PAGE_3, page_no=3)
    by_title = {c.title: c for c in candidates}
    # "### 3.2.1 ..." is a level-3 section even though the Markdown says "###".
    assert by_title["3.2.1 Scaled Dot-Product Attention"].level == 3
    assert by_title["3.2.1 Scaled Dot-Product Attention"].number == "3.2.1"
    assert by_title["3.2 Attention"].level == 2


def test_markdown_to_blocks_strips_the_hash_prefix() -> None:
    """The '#' must not survive into the block text, otherwise the reference
    section cannot be located (``References`` vs ``## References``)."""
    blocks, _candidates = markdown_to_blocks(OCR_REFERENCES_PAGE, page_no=11)
    texts = [b.text for b in blocks]
    assert "References" in texts
    assert not any(t.startswith("#") for t in texts)


def test_markdown_to_blocks_keeps_body_text() -> None:
    blocks, _candidates = markdown_to_blocks(OCR_PAGE_1, page_no=1)
    joined = "\n".join(b.text for b in blocks)
    assert "Recurrent neural networks" in joined
    assert "avaswani@google.com" in joined


def test_markdown_to_blocks_builds_a_nested_tree() -> None:
    blocks, candidates = markdown_to_blocks(OCR_PAGE_1 + "\n" + OCR_PAGE_3, page_no=1)
    flat = flatten(build_tree(candidates))
    numbers = {n.number for n in flat}
    assert "1" in numbers
    assert "3.2" in numbers
    assert "3.2.1" in numbers
    # The level-3 section is nested below the level-2 one.
    by_number = {n.number: n for n in flat}
    assert by_number["3.2.1"].parent is not None
    assert by_number["3.2.1"].parent.number == "3.2"


# ---------------------------------------------------------------- metadata
def test_metadata_from_markdown_reads_the_title() -> None:
    meta = extract_metadata_from_markdown([OCR_PAGE_1])
    assert meta.title == "Attention Is All You Need"


def test_metadata_from_markdown_reads_the_authors() -> None:
    meta = extract_metadata_from_markdown([OCR_PAGE_1])
    names = [a.name for a in meta.authors]
    assert "Ashish Vaswani" in names
    assert "Noam Shazeer" in names
    assert "Aidan N. Gomez" in names


def test_metadata_from_markdown_reads_the_affiliations() -> None:
    meta = extract_metadata_from_markdown([OCR_PAGE_1])
    joined = " ".join(meta.affiliations)
    assert "Google Brain" in joined
    assert "University of Toronto" in joined


def test_metadata_from_markdown_reads_the_abstract() -> None:
    meta = extract_metadata_from_markdown([OCR_PAGE_1])
    assert "dominant sequence transduction models" in meta.abstract
    assert len(meta.abstract) > 100


def test_metadata_from_markdown_reads_the_arxiv_id_and_year() -> None:
    meta = extract_metadata_from_markdown([OCR_PAGE_1])
    assert meta.arxiv_id == "1706.03762"
    assert meta.version_label == "v1"
    assert meta.year == 2017


def test_metadata_from_empty_input_is_empty_but_safe() -> None:
    meta = extract_metadata_from_markdown([])
    assert meta.title == ""
    assert meta.authors == []


def test_metadata_falls_back_to_the_filename() -> None:
    meta = extract_metadata_from_markdown([""], filename="Some-Paper.pdf")
    assert meta.title == "Some-Paper"


# -------------------------------------------------------------- references
def test_reference_section_is_found_in_ocr_blocks() -> None:
    from app.parsing.references import find_reference_section

    blocks, _candidates = markdown_to_blocks(OCR_REFERENCES_PAGE, page_no=11)
    start, end = find_reference_section(blocks)
    assert start >= 0
    assert end > start


def test_references_are_parsed_from_ocr_blocks() -> None:
    from app.parsing.references import parse_references

    blocks, _candidates = markdown_to_blocks(OCR_REFERENCES_PAGE, page_no=11)
    refs = parse_references(blocks)
    assert len(refs) == 2
    assert refs[0].title
    assert refs[0].authors


@pytest.mark.parametrize("markdown", ["", "# ", "no headings here"])
def test_markdown_to_blocks_tolerates_degenerate_input(markdown: str) -> None:
    blocks, candidates = markdown_to_blocks(markdown, page_no=1)
    assert isinstance(blocks, list)
    assert isinstance(candidates, list)
