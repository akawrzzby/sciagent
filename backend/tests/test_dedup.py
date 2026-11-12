"""Module 1/5 — semantic duplicate detection and version comparison.

The competition dataset deliberately contains the *same* paper in two arXiv
versions (``1706.03762v1`` and ``1706.03762v7``).  The system must recognise
them as one work, recommend the newer version, and be able to show the diff.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.parsing.dedup import (
    PaperFingerprint,
    compare_fingerprints,
    content_hash,
    diff_versions,
    hamming_distance,
    recommend_version,
    simhash,
    simhash_similarity,
    version_rank,
)


def _fp(**kw) -> PaperFingerprint:
    base = dict(
        paper_id="x",
        title="Attention Is All You Need",
        arxiv_id="1706.03762",
        version_label="v1",
        file_hash="",
        content_hash="",
        simhash=0,
        char_count=1000,
        page_count=10,
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    )
    base.update(kw)
    return PaperFingerprint(**base)


# ------------------------------------------------------------------ hashes
def test_content_hash_is_whitespace_insensitive() -> None:
    a = content_hash("Attention  is\nall you need")
    b = content_hash("attention is all you need")
    assert a == b


def test_content_hash_differs_for_different_text() -> None:
    assert content_hash("attention") != content_hash("transformer")


def test_simhash_similarity_of_identical_text_is_one() -> None:
    text = "The dominant sequence transduction models are based on recurrent networks."
    assert simhash_similarity(simhash(text), simhash(text)) == pytest.approx(1.0)


def test_simhash_detects_near_duplicates() -> None:
    base = "The dominant sequence transduction models are based on recurrent networks."
    variant = "The dominant sequence transduction models are based on recurrent neural networks."
    assert simhash_similarity(simhash(base), simhash(variant)) > 0.8


def test_simhash_detects_unrelated_text() -> None:
    a = "The dominant sequence transduction models are based on recurrent networks."
    b = "A survey of graph neural networks for molecular property prediction."
    assert simhash_similarity(simhash(a), simhash(b)) < 0.7


def test_hamming_distance_of_equal_hashes_is_zero() -> None:
    assert hamming_distance(simhash("abc def"), simhash("abc def")) == 0


# -------------------------------------------------------------- comparison
def test_same_arxiv_id_is_flagged_as_same_paper() -> None:
    incoming = _fp(version_label="v7")
    existing = _fp(paper_id="old", version_label="v1")
    candidates = compare_fingerprints(incoming, [existing])
    assert candidates
    assert candidates[0].same_paper is True
    assert candidates[0].score >= 0.9


def test_identical_file_hash_scores_one() -> None:
    incoming = _fp(file_hash="deadbeef")
    existing = _fp(paper_id="old", file_hash="deadbeef")
    candidates = compare_fingerprints(incoming, [existing])
    assert candidates[0].score == 1.0
    assert candidates[0].same_paper is True


def test_semantic_embedding_similarity_detects_duplicate() -> None:
    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    incoming = _fp(paper_id="new", embedding=vec)
    existing = _fp(paper_id="old", embedding=vec * 1.0)
    candidates = compare_fingerprints(incoming, [existing])
    assert candidates[0].same_paper is True
    assert any("语义相似度" in r for r in candidates[0].reasons)


def test_unrelated_papers_are_not_duplicates() -> None:
    incoming = _fp(paper_id="new", title="Graph Neural Networks for Molecules",
                   arxiv_id="1801.00001", simhash=simhash("gnn molecules prediction"))
    existing = _fp(paper_id="old", title="BERT Pre-training",
                   arxiv_id="1810.04805", simhash=simhash("bert pretraining language model"))
    candidates = compare_fingerprints(incoming, [existing])
    assert not candidates or candidates[0].same_paper is False


# ---------------------------------------------------------------- versions
def test_version_rank_parses_v_labels() -> None:
    assert version_rank("v7") == 7
    assert version_rank("") == 0
    assert version_rank("v12") == 12


def test_recommend_version_prefers_the_newer_one() -> None:
    incoming = _fp(paper_id="new", version_label="v7", char_count=120_000)
    existing_candidate = compare_fingerprints(
        incoming, [_fp(paper_id="old", version_label="v1", char_count=100_000)]
    )[0]
    decision = recommend_version(incoming, existing_candidate)
    assert decision["recommendation"] == "keep_incoming"
    assert decision["keep"] == "uploaded"
    assert decision["reason"]


def test_recommend_version_keeps_the_newer_when_it_is_the_existing_one() -> None:
    incoming = _fp(paper_id="new", version_label="v1", char_count=100_000)
    existing_candidate = compare_fingerprints(
        incoming, [_fp(paper_id="old", version_label="v7", char_count=120_000)]
    )[0]
    decision = recommend_version(incoming, existing_candidate)
    assert decision["recommendation"] == "keep_existing"
    assert decision["keep"] == "library"


# ------------------------------------------------------------------- diff
def test_diff_versions_reports_changed_lines() -> None:
    old = "The model uses 8 GPUs.\nThe batch size is 64.\nThe dropout rate is 0.1."
    new = "The model uses 8 GPUs.\nThe batch size is 256.\nThe dropout rate is 0.1."
    diff = diff_versions(old, new)
    assert diff
    changed = [d for d in diff if d.get("op") in {"replace", "insert", "delete"}]
    assert changed


def test_diff_versions_of_identical_text_has_no_change() -> None:
    text = "a\nb\nc"
    diff = diff_versions(text, text)
    assert all(d["op"] == "equal" for d in diff)
