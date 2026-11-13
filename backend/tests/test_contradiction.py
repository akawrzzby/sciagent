"""Contradiction detection.

The rules are deliberately strict: a metric alone is not a contradiction, the
two passages have to talk about the same thing as well.
"""

from __future__ import annotations

from app.graph import contradiction


def _chunk(paper: str, title: str, text: str, page: int = 3, chunk_id: str = "") -> dict:
    return {"paper_id": paper, "paper_title": title, "content": text,
            "page_no": page, "id": chunk_id or f"{paper}:{page}"}


def test_a_metric_and_its_value_are_extracted() -> None:
    claims = contradiction.extract_number_claims(
        "The big Transformer model achieves a BLEU score of 41.0 on the "
        "WMT 2014 English-to-German translation task.")
    assert len(claims) == 1
    assert claims[0].metric == "BLEU"
    assert claims[0].value == 41.0


def test_a_value_that_stands_before_its_metric_is_extracted() -> None:
    claims = contradiction.extract_number_claims(
        "The model reaches 28.4 BLEU on the WMT 2014 English-to-German task.")
    assert claims and claims[0].metric == "BLEU"
    assert claims[0].value == 28.4


def test_two_values_of_the_same_metric_in_the_same_context_conflict() -> None:
    chunks = [
        _chunk("p1", "Attention v1",
               "The big Transformer model achieves a BLEU score of 41.0 on the "
               "WMT 2014 English-to-German translation task."),
        _chunk("p2", "Attention v7",
               "The big Transformer model achieves a BLEU score of 41.8 on the "
               "WMT 2014 English-to-German translation task."),
    ]
    conflicts = contradiction.detect_numeric_conflicts(chunks)
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.metric == "BLEU"
    assert c.left["value"] == 41.0
    assert c.right["value"] == 41.8
    assert c.delta == 0.8
    # Both sides can be quoted.
    assert "41.0" in c.left["text"]
    assert "41.8" in c.right["text"]


def test_two_values_of_the_same_metric_in_different_contexts_do_not_conflict() -> None:
    """BLEU 41.0 on English-to-German and BLEU 28.4 on English-to-French are
    two results, not a contradiction."""
    chunks = [
        _chunk("p1", "Paper A",
               "The model achieves a BLEU score of 41.0 on the WMT 2014 "
               "English-to-German translation task."),
        _chunk("p2", "Paper B",
               "The model achieves a BLEU score of 28.4 on the WMT 2014 "
               "English-to-French translation task."),
    ]
    assert contradiction.detect_numeric_conflicts(chunks) == []


def test_the_same_number_twice_is_not_a_contradiction() -> None:
    chunks = [
        _chunk("p1", "Paper A", "The model reaches a BLEU score of 41.0 on WMT."),
        _chunk("p2", "Paper B", "The model reaches a BLEU score of 41.0 on WMT."),
    ]
    assert contradiction.detect_numeric_conflicts(chunks) == []


def test_the_same_paper_reporting_two_values_is_not_a_contradiction() -> None:
    chunks = [
        _chunk("p1", "Paper A", "The model reaches a BLEU score of 41.0 on WMT."),
        _chunk("p1", "Paper A", "The model reaches a BLEU score of 28.4 on WMT."),
    ]
    assert contradiction.detect_numeric_conflicts(chunks) == []


def test_an_opposite_claim_is_detected() -> None:
    chunks = [
        _chunk("p1", "Paper A",
               "Layer normalization improves the training speed of the "
               "transformer encoder considerably."),
        _chunk("p2", "Paper B",
               "Layer normalization does not improve the training speed of the "
               "transformer encoder in practice."),
    ]
    conflicts = contradiction.detect_claim_conflicts(chunks)
    assert len(conflicts) == 1
    assert conflicts[0].kind == "claim"


def test_the_report_carries_both_quotes() -> None:
    chunks = [
        _chunk("p1", "Attention v1",
               "The big Transformer model achieves a BLEU score of 41.0 on the "
               "WMT 2014 English-to-German translation task."),
        _chunk("p2", "Attention v7",
               "The big Transformer model achieves a BLEU score of 41.8 on the "
               "WMT 2014 English-to-German translation task."),
    ]
    result = contradiction.detect_contradictions(chunks)
    assert result["count"] == 1
    assert result["numeric_count"] == 1
    assert len(result["papers_involved"]) == 2
    md = contradiction.conflicts_to_markdown(result)
    assert "41.0" in md and "41.8" in md


def test_an_empty_library_reports_no_conflict() -> None:
    result = contradiction.detect_contradictions([])
    assert result["count"] == 0
    assert "未在当前文献库中检出矛盾" in contradiction.conflicts_to_markdown(result)


# --- the loose version of the claim detector, and why it is not used ---------
#
# "we observe an improvement" against "we did not observe an improvement" is
# the classic false positive of a pattern based contradiction detector.  The
# detector only accepts a subject that the two passages actually share, and
# only over the words that few papers of the library use.

def test_a_generic_subject_does_not_conflict() -> None:
    """Two papers that both talk about "performance" do not contradict."""
    chunks = [
        _chunk("p1", "Paper A",
               "While we have not found them necessary to achieve good performance, "
               "it is possible that the results could be further improved by "
               "incorporating this."),
        _chunk("p2", "Paper B",
               "The only caveat is the performance drop between C4 and C5 for "
               "60B parameter model."),
    ]
    # More papers that use the same generic words.
    for i in range(6):
        chunks.append(_chunk(f"f{i}", f"Filler {i}",
                             "We report the model performance and the training "
                             "cost of the setup."))
    assert contradiction.detect_claim_conflicts(chunks) == []


def test_a_table_is_not_read_as_a_claim() -> None:
    """A hyper-parameter table is a list of numbers, not a statement."""
    table = ("Dataset GLUE (Acc.) Super-NaturalInstructions (RougeL) Model "
             "RoBERTa-large T5-80M T5-250M T5-780M T5-3B T5-11B BF16 88.6 40.1 "
             "42.1 48.0 54.3 62.0 BF16 replication 88.6 40.0 42.2 47.3 54.9 "
             "LoRA BF16 88.8 40.5 42.6 47.1 55.4 60.7 QLoRA Int8 88.8 40.4 42.9 "
             "45.4 56.5 60.7")
    chunks = [
        _chunk("p1", "Paper A",
               "However, this not only restricts the possible solutions, but "
               "also reduces the degrees of separation between the specification "
               "and the code required to be produced."),
        _chunk("p2", "Paper B", table),
    ]
    assert contradiction.detect_claim_conflicts(chunks) == []
