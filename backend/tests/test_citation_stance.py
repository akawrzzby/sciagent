"""Citation stance: supporting / contrasting / mentioning.

The taxonomy follows the Smart Citations scheme (Nicholson et al., 2021),
but the labels are derived from explicit discourse markers instead of a
trained classifier, so every decision can be explained to a reviewer.
"""

from __future__ import annotations

from app.graph import builder


def test_a_citation_that_backs_the_claim_is_supporting() -> None:
    stance = builder.classify_citation_context(
        "Vaswani et al. showed that the Transformer generalises well to other tasks.",
        "Attention Is All You Need",
    )
    assert stance.stance == builder.CITATION_STANCE_SUPPORTING
    assert stance.confidence > 0
    assert stance.reason


def test_a_citation_used_against_another_line_of_work_is_contrasting() -> None:
    stance = builder.classify_citation_context(
        "However, Vaswani et al. showed no improvement over the recurrent baseline.",
        "Attention Is All You Need",
    )
    assert stance.stance == builder.CITATION_STANCE_CONTRASTING


def test_a_citation_without_a_stance_signal_is_only_mentioned() -> None:
    stance = builder.classify_citation_context(
        "Several related studies have explored this topic, among them Vaswani et al.",
        "Attention Is All You Need",
    )
    assert stance.stance == builder.CITATION_STANCE_MENTIONING


def test_a_bare_mention_carries_no_confidence() -> None:
    stance = builder.classify_citation_context("", "Attention Is All You Need")
    assert stance.stance == builder.CITATION_STANCE_MENTIONING
    assert stance.confidence == 0.0
    assert stance.reason


def test_a_contrastive_marker_wins_over_a_supportive_one() -> None:
    """'however, [12] showed ...' is a contrast even though [12] gives evidence."""
    stance = builder.classify_citation_context(
        "However, Vaswani et al. showed a different result on the same benchmark.",
        "Attention Is All You Need",
    )
    assert stance.stance == builder.CITATION_STANCE_CONTRASTING


def test_every_stance_has_a_chinese_label() -> None:
    for key in (builder.CITATION_STANCE_SUPPORTING,
                builder.CITATION_STANCE_CONTRASTING,
                builder.CITATION_STANCE_MENTIONING):
        assert builder.CITATION_STANCE_LABEL[key]
