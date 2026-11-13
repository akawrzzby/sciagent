"""Metadata aware retrieval: the affinity rules and the embedded prefix.

The point of these tests is that the boost stays *explainable* and stays
*conservative*: a query that names a paper lifts that paper, a query that
names nothing lifts nothing.
"""

from __future__ import annotations

from app.rag import meta_aware


AUTHORS = [{"name": "Ashish Vaswani", "affiliations": []},
           {"name": "Noam Shazeer", "affiliations": []},
           {"name": "Niki Parmar", "affiliations": []}]


def test_the_authors_are_read_from_every_storage_shape() -> None:
    assert meta_aware.author_names(AUTHORS) == ["Ashish Vaswani", "Noam Shazeer",
                                                 "Niki Parmar"]
    assert meta_aware.author_names(["Ashish Vaswani"]) == ["Ashish Vaswani"]
    assert meta_aware.author_names('["Ashish Vaswani"]') == ["Ashish Vaswani"]
    assert meta_aware.author_names(None) == []


def test_the_surname_ignores_the_name_particles() -> None:
    assert meta_aware.surname("Ashish Vaswani") == "Vaswani"
    assert meta_aware.surname("Henrique Ponde de Oliveira Pinto") == "Pinto"
    assert meta_aware.surname("Ludwig van Beethoven") == "Beethoven"


def test_the_embedded_prefix_carries_title_authors_and_year() -> None:
    header = meta_aware.metadata_header(
        "Attention Is All You Need", AUTHORS, 2017)
    assert "Attention Is All You Need" in header
    assert "Vaswani" in header
    assert "2017" in header
    # Three or more authors are abbreviated.
    assert "et al." in header


def test_the_stored_content_is_never_modified() -> None:
    """Only the vector sees the metadata; the citation quotes the original."""
    text = meta_aware.embedding_text({
        "content": "The model reaches 28.4 BLEU.",
        "paper_title": "Attention Is All You Need",
        "paper_authors": AUTHORS,
        "paper_year": 2017,
    })
    assert text.endswith("The model reaches 28.4 BLEU.")
    assert "Attention Is All You Need" in text


def test_a_query_that_names_the_paper_scores_higher() -> None:
    named = meta_aware.paper_affinity(
        "What did Vaswani report in the Attention Is All You Need paper?",
        "Attention Is All You Need", AUTHORS, 2017)
    anonymous = meta_aware.paper_affinity(
        "What BLEU score did the model reach on WMT 2014?",
        "Attention Is All You Need", AUTHORS, 2017)
    assert named > anonymous
    assert named > 0.5


def test_a_query_that_names_nothing_does_not_boost() -> None:
    assert meta_aware.paper_affinity(
        "What BLEU score did the model reach?",
        "Attention Is All You Need", AUTHORS, 2017) == 0.0


def test_the_year_alone_is_a_weak_signal() -> None:
    affinity = meta_aware.paper_affinity(
        "which paper from 2017 used attention?", "Attention Is All You Need",
        AUTHORS, 2017)
    assert 0.0 < affinity < 0.3


def test_the_affinity_is_explained_part_by_part() -> None:
    info = meta_aware.explain_affinity(
        "Vaswani 2017", "Attention Is All You Need", AUTHORS, 2017)
    assert info["title"] == 0.0
    assert info["authors"] > 0
    assert info["year"] == 1.0
    assert 0 < info["affinity"] <= 1.0


def test_the_boost_never_exceeds_one() -> None:
    affinity = meta_aware.paper_affinity(
        "Attention Is All You Need Vaswani 2017", "Attention Is All You Need",
        AUTHORS, 2017)
    assert affinity <= 1.0


def test_the_matched_authors_are_reported() -> None:
    assert meta_aware.matched_authors("What did Vaswani publish?", AUTHORS) == [
        "Ashish Vaswani"]
    assert meta_aware.matched_authors("nothing here", AUTHORS) == []
