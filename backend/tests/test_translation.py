"""Tests for the bilingual reader's translation pipeline.

Three things are pinned down here:

1. the prose clean-up (PDF line breaks and hyphenation) and the sentence
   splitter — the reader shows one Chinese sentence per English sentence,
   so a wrong boundary becomes a visibly wrong translation;
2. the one-to-one English/Chinese pairing after a translation round;
3. the failure contract: a segment that cannot be translated is **reported**,
   never silently replaced by its English source (that bug turned the whole
   bilingual view into a copy of the original and cached it).
"""

from __future__ import annotations

import asyncio

import pytest

from app.services import translation as translation_service
from app.services.translation import (
    Segment,
    build_document_segments,
    merge_short_segments,
    normalise_prose,
    split_sentences,
    translate_document,
)


# --------------------------------------------------------------------------- prose
def test_normalise_prose_joins_hard_wrapped_lines() -> None:
    raw = "BERT alleviates the previously mentioned\nuni-directionality constraint by using a\nmasked language model."
    assert normalise_prose(raw) == (
        "BERT alleviates the previously mentioned uni-directionality constraint "
        "by using a masked language model."
    )


def test_normalise_prose_repairs_hyphenated_line_breaks() -> None:
    # "apply-\ning" is one word split across a line, not a hyphenated word.
    assert normalise_prose("to apply-\ning pre-trained models") == "to applying pre-trained models"


def test_normalise_prose_keeps_real_hyphens() -> None:
    assert normalise_prose("a state-of-the-art model") == "a state-of-the-art model"


# --------------------------------------------------------------------------- sentences
def test_split_sentences_finds_boundaries() -> None:
    sentences = split_sentences(
        "We propose a new model. It is trained on eleven tasks. "
        "The results are strong."
    )
    assert sentences == [
        "We propose a new model.",
        "It is trained on eleven tasks.",
        "The results are strong.",
    ]


def test_split_sentences_does_not_split_on_abbreviations() -> None:
    sentences = split_sentences(
        "Prior work uses word embeddings (e.g. Brown et al., 1992). "
        "We take a different route."
    )
    assert len(sentences) == 2
    assert sentences[0].startswith("Prior work uses word embeddings")
    assert sentences[1].startswith("We take a different route.")


def test_split_sentences_does_not_split_on_initials() -> None:
    sentences = split_sentences("A. Vaswani proposed the Transformer. It became standard.")
    assert len(sentences) == 2


def test_split_sentences_keeps_decimals_intact() -> None:
    # "8.7%" must stay inside one sentence: a split here leaves the bilingual
    # reader hovering over the fragment "achieved 8." instead of the sentence.
    sentences = split_sentences(
        "For example, we achieved 8.7% absolute improvement on the test set. "
        "The gain holds across all tasks."
    )
    assert len(sentences) == 2
    assert sentences[0] == (
        "For example, we achieved 8.7% absolute improvement on the test set."
    )
    assert sentences[1] == "The gain holds across all tasks."


def test_split_sentences_keeps_spaced_decimals_intact() -> None:
    # The PDF text of BERT carries "8. 9%" (the number is broken by the line
    # break of the original layout); after the line join it is one number.
    sentences = split_sentences(
        "For instance, we achieve absolute improvements of 8. 9% on commonsense "
        "reasoning (Stories Cloze Test), 5. 7% on question answering (RACE)."
    )
    assert len(sentences) == 1


def test_split_sentences_keeps_scientific_notation_intact() -> None:
    sentences = split_sentences(
        "We used the Adam optimization scheme [27] with a max learning rate of "
        "2.5e-4. The schedule was tuned on the dev set."
    )
    assert len(sentences) == 2
    assert sentences[0].endswith("2.5e-4.")
    assert sentences[1] == "The schedule was tuned on the dev set."


def test_split_sentences_still_splits_before_a_word() -> None:
    # A number that starts the next sentence must not be swallowed: the
    # period after "3" ends the sentence, "5 more items" opens the next one.
    sentences = split_sentences(
        "The corpus contains 3. 5 more items than the previous release. "
        "We keep the ordering."
    )
    assert sentences == [
        "The corpus contains 3.",
        "5 more items than the previous release.",
        "We keep the ordering.",
    ]


def test_split_sentences_keeps_broken_numbers_with_a_unit_intact() -> None:
    # Some layouts break a number into "8." and "9%"; the unit shows that the
    # two halves belong together.
    sentences = split_sentences(
        "We achieve absolute improvements of 8. 9% on commonsense reasoning. "
        "The gain holds for all tasks."
    )
    assert len(sentences) == 2
    assert sentences[0].endswith("commonsense reasoning.")
    assert "8. 9%" in sentences[0]


def test_split_sentences_keeps_section_numbers_intact() -> None:
    sentences = split_sentences(
        "As shown in Section 3.2 the encoder keeps a constant depth. "
        "This matters for long inputs."
    )
    assert len(sentences) == 2
    assert sentences[0].endswith("constant depth.")
    assert "Section 3.2" in sentences[0]


def test_merge_short_segments_glues_fragments() -> None:
    merged = merge_short_segments(["A full sentence that is long enough.", "However."])
    assert merged == ["A full sentence that is long enough. However."]


# --------------------------------------------------------------------------- segments
@pytest.mark.asyncio
async def test_build_document_segments_is_sentence_aligned() -> None:
    sections = [{"title": "Introduction", "page_no": 1}]
    blocks = [
        {"id": "b1", "page_no": 1, "type": "body",
         "content": "We propose a new model. It is trained on eleven tasks. "
                    "The results are strong."},
        {"id": "b2", "page_no": 1, "type": "footnote",
         "content": "9 A footnote that must never be translated as body text."},
    ]
    segments = await build_document_segments(sections, blocks)

    assert [s.kind for s in segments] == ["heading", "paragraph", "paragraph", "paragraph"]
    assert segments[1].source == "We propose a new model."
    assert segments[1].block_id == "b1"
    # The footnote block is not part of the prose.
    assert all("footnote" not in s.source for s in segments)


# --------------------------------------------------------------------------- translation
class _FakeLLM:
    """A model double that echoes a numbered line per sentence."""

    def __init__(self, *, fail_indices: set[int] | None = None, fail_all: bool = False) -> None:
        self.fail_indices = fail_indices or set()
        self.fail_all = fail_all
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0

    async def chat(self, messages, **_kwargs):
        self.calls += 1
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        await asyncio.sleep(0.01)
        self.concurrent -= 1

        user = messages[-1]["content"]
        if self.fail_all:
            from app.llm.client import LLMResponse

            return LLMResponse(content="", finish_reason="error", raw={"error": "boom"})

        lines = []
        for line in user.splitlines():
            if not line or line[0].isdigit() is False or ": " not in line:
                continue
            index, _, text = line.partition(": ")
            index = index.strip()
            if not index.isdigit():
                continue
            if int(index) in self.fail_indices:
                continue
            lines.append(f"{index}: 译文-{text[:12]}")
        from app.llm.client import LLMResponse

        return LLMResponse(content="\n".join(lines), finish_reason="stop")


@pytest.mark.asyncio
async def test_translate_document_pairs_every_segment(monkeypatch) -> None:
    fake = _FakeLLM()
    monkeypatch.setattr(translation_service, "get_llm_client", lambda: fake)

    segments = [
        Segment(index=i + 1, source=f"Sentence number {i} is long enough to translate.")
        for i in range(6)
    ]
    outcome = await translate_document(segments, batch_size=3, concurrency=3)

    assert outcome.ok
    assert all(s.translation for s in segments)
    assert all(s.translation.startswith("译文-") for s in segments)
    # Six segments, three per batch -> two batch calls.
    assert fake.calls == 2


@pytest.mark.asyncio
async def test_translate_document_runs_batches_concurrently(monkeypatch) -> None:
    fake = _FakeLLM()
    monkeypatch.setattr(translation_service, "get_llm_client", lambda: fake)

    segments = [
        Segment(index=i + 1, source=f"Sentence number {i} is long enough to translate.")
        for i in range(8)
    ]
    await translate_document(segments, batch_size=2, concurrency=4)

    assert fake.max_concurrent > 1


@pytest.mark.asyncio
async def test_translate_document_reports_failures_instead_of_falling_back(monkeypatch) -> None:
    """A failed segment must not be silently replaced by its English source."""
    fake = _FakeLLM(fail_all=True)
    monkeypatch.setattr(translation_service, "get_llm_client", lambda: fake)

    segments = [
        Segment(index=i + 1, source=f"Sentence number {i} is long enough to translate.")
        for i in range(2)
    ]
    outcome = await translate_document(segments, batch_size=2, concurrency=1)

    assert not outcome.ok
    assert outcome.failed
    # The English source is NOT copied over the failed translation.
    assert all(not s.translation for s in segments)


@pytest.mark.asyncio
async def test_translate_document_reports_progress(monkeypatch) -> None:
    fake = _FakeLLM()
    monkeypatch.setattr(translation_service, "get_llm_client", lambda: fake)

    seen: list[float] = []

    async def progress(fraction: float) -> None:
        seen.append(fraction)

    segments = [
        Segment(index=i + 1, source=f"Sentence number {i} is long enough to translate.")
        for i in range(4)
    ]
    await translate_document(segments, batch_size=2, concurrency=2, progress_cb=progress)

    assert seen
    assert seen[-1] == pytest.approx(1.0)
