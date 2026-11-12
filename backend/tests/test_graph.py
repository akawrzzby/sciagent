"""Module 5 — citation graph, foundation mining and citation quality."""

from __future__ import annotations

import pytest

from app.graph.builder import (
    build_graph,
    compute_citation_relevance,
    filter_substantive_references,
)


# ------------------------------------------------------------------- graph
def test_graph_builds_edges_only_between_library_papers(
    sample_papers: list[dict], sample_references: dict[str, list[dict]]
) -> None:
    graph = build_graph(sample_papers, sample_references)
    ids = {p["paper_id"] for p in sample_papers}
    for e in graph.edges:
        assert e["source"] in ids
        assert e["target"] in ids
        assert e["source"] != e["target"]


def test_pagerank_is_a_probability_distribution(
    sample_papers: list[dict], sample_references: dict[str, list[dict]]
) -> None:
    graph = build_graph(sample_papers, sample_references)
    assert graph.nodes
    assert sum(n.pagerank for n in graph.nodes) == pytest.approx(1.0, abs=1e-3)


def test_most_cited_paper_has_the_highest_pagerank(
    sample_papers: list[dict], sample_references: dict[str, list[dict]]
) -> None:
    """Both BERT and GPT cite the Transformer paper, so it must lead."""
    graph = build_graph(sample_papers, sample_references)
    top = max(graph.nodes, key=lambda n: n.pagerank)
    assert top.paper_id == "pA"
    assert top.in_degree == 2


def test_k_core_is_monotonic_with_the_number_of_edges(
    sample_papers: list[dict], sample_references: dict[str, list[dict]]
) -> None:
    graph = build_graph(sample_papers, sample_references)
    by_id = {n.paper_id: n for n in graph.nodes}
    # pA is in the 2-core of the chain B -> A <- C.
    assert by_id["pA"].k_core >= by_id["pB"].k_core


def test_foundation_papers_are_returned_first(
    sample_papers: list[dict], sample_references: dict[str, list[dict]]
) -> None:
    graph = build_graph(sample_papers, sample_references)
    foundation = graph.foundation_papers()
    assert foundation
    assert foundation[0].paper_id == "pA"
    assert foundation[0].role == "foundation"


def test_derivative_papers_are_the_ones_that_cite(
    sample_papers: list[dict], sample_references: dict[str, list[dict]]
) -> None:
    graph = build_graph(sample_papers, sample_references)
    derivative_ids = {n.paper_id for n in graph.derivative_papers()}
    assert "pB" in derivative_ids


def test_graph_serialises_to_plain_json(
    sample_papers: list[dict], sample_references: dict[str, list[dict]]
) -> None:
    import json

    graph = build_graph(sample_papers, sample_references)
    payload = graph.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["nodes"]
    assert payload["edges"]


def test_graph_with_a_single_paper_has_no_edges(sample_papers: list[dict]) -> None:
    graph = build_graph(sample_papers[:1], {})
    assert graph.edges == []
    assert len(graph.nodes) == 1


# -------------------------------------------------------- citation quality
def test_substantive_citation_outscores_a_bare_mention() -> None:
    contextual = (
        "Our encoder follows the multi-head attention mechanism of Vaswani et al. [12], "
        "and outperforms the previous state-of-the-art by 2.3 BLEU on WMT 2014."
    )
    bare = "see [12]"
    assert compute_citation_relevance(contextual, "Attention Is All You Need") > \
        compute_citation_relevance(bare, "Attention Is All You Need")


def test_filter_splits_substantive_and_perfunctory_references() -> None:
    refs = [
        {"title": "Attention Is All You Need", "context":
         "We use the multi-head attention of Vaswani et al. [12] and improve BLEU by 2.3.",
         "authors": ["Vaswani"], "year": 2017},
        {"title": "Some Other Work", "context": "see [13]", "authors": ["X"], "year": 2010},
    ]
    result = filter_substantive_references(refs)
    assert result["kept"]
    assert result["dropped"]
    assert result["kept"][0]["title"] == "Attention Is All You Need"
    assert all(k["relevance"] >= result["threshold"] for k in result["kept"])


def test_filter_is_sorted_by_relevance() -> None:
    refs = [
        {"title": "A", "context": "short", "relevance": 0.5},
        {"title": "B", "context": "long", "relevance": 0.95},
        {"title": "C", "context": "x", "relevance": 0.6},
    ]
    result = filter_substantive_references(refs)
    relevances = [r["relevance"] for r in result["kept"]]
    assert relevances == sorted(relevances, reverse=True)


def test_empty_reference_list_yields_empty_result() -> None:
    result = filter_substantive_references([])
    assert result["kept"] == []
    assert result["dropped"] == []
