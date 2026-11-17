"""Module 3 – academic translation (EN <-> ZH) with segment alignment.

Two capabilities are needed:

* **Segment level** – the user selects a paragraph (or hovers a sentence) and
  gets a translation.
* **Document level** – the whole paper is translated into Markdown and stored
  **aligned with the original paragraphs**.  The alignment is what powers the
  bilingual reading mode: the frontend scrolls the two columns together.

Code, formulas, citations and numbers are protected: they are replaced by
placeholders before the request and restored afterwards, so they are never
mangled by the model.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.llm import prompts
from app.llm.client import get_llm_client

logger = get_logger(__name__)

# --------------------------------------------------------------------------- protection
_PLACEHOLDER = "␀{i}␁"

_PROTECT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("math", re.compile(r"\$\$.+?\$\$", re.DOTALL)),
    ("math", re.compile(r"\$[^$\n]{1,200}\$")),
    ("latex", re.compile(r"\\[a-zA-Z]+\{[^}]{0,200}\}")),
    ("latex", re.compile(r"\\(?:frac|sum|int|prod|sqrt|alpha|beta|gamma|delta|epsilon|"
                         r"theta|lambda|mu|pi|sigma|phi|omega|left|right|mathbb|mathcal|"
                         r"mathbf|text|mathrm|begin|end)\b[^%\n]{0,120}")),
    ("code", re.compile(r"`[^`\n]{1,200}`")),
    ("code", re.compile(r"```.*?```", re.DOTALL)),
    ("url", re.compile(r"https?://\S+")),
    ("citation", re.compile(r"\[\s*\d+(?:\s*[,;-]\s*\d+)*\s*\]")),
    ("author_year", re.compile(r"\([A-Z][A-Za-z\u00c0-\u024f'.-]+(?: et al\.)?,? "
                               r"\d{4}[a-z]?\)")),
    ("doi", re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")),
    ("arxiv", re.compile(r"\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\b")),
]


@dataclass
class Segment:
    """One aligned (source, translation) pair."""

    index: int
    source: str
    translation: str = ""
    kind: str = "paragraph"   # paragraph | heading | caption | code | formula
    page_no: int = 0
    section_title: str = ""
    block_id: str = ""        # the content block this segment came from
    block_order: int = -1     # the block's index in the document stream

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "source": self.source, "translation": self.translation,
                "kind": self.kind, "page_no": self.page_no,
                "section_title": self.section_title, "block_id": self.block_id,
                "block_order": self.block_order}


def protect(text: str) -> tuple[str, list[str]]:
    """Replace protected spans with placeholders.

    Returns ``(masked_text, originals)``.
    """
    originals: list[str] = []
    masked = text
    for _kind, pattern in _PROTECT_PATTERNS:
        def _sub(match: re.Match[str]) -> str:
            originals.append(match.group(0))
            return _PLACEHOLDER.format(i=len(originals) - 1)

        masked = pattern.sub(_sub, masked)
    return masked, originals


def restore(text: str, originals: list[str]) -> str:
    """Put the protected spans back."""
    out = text
    for i, original in enumerate(originals):
        out = out.replace(_PLACEHOLDER.format(i=i), original)
        # Models sometimes insert spaces around the placeholder.
        out = out.replace(f" {_PLACEHOLDER.format(i=i)} ", original)
        out = out.replace(f" {_PLACEHOLDER.format(i=i)}", original)
        out = out.replace(f"{_PLACEHOLDER.format(i=i)} ", original)
    return out


# --------------------------------------------------------------------------- translation
async def translate_text(text: str, target_lang: str = "zh") -> dict[str, Any]:
    """Translate a piece of text, keeping code and formulas intact."""
    if not text or not text.strip():
        return {"translation": "", "target_lang": target_lang, "protected": 0}

    masked, originals = protect(text)
    if not masked.strip():
        return {"translation": text, "target_lang": target_lang, "protected": len(originals)}

    llm = get_llm_client()
    direction = "English to Simplified Chinese" if target_lang == "zh" else "Simplified Chinese to English"
    resp = await llm.chat(
        [
            {"role": "system", "content": prompts.SYSTEM_TRANSLATE},
            {"role": "user", "content":
                f"Translate the following text ({direction}). "
                f"Keep the placeholder markers (␀0␁, ␀1␁, ...) exactly as they are.\n\n"
                f"{masked}"},
        ],
        temperature=0.1,
        purpose="translate",
        # Translation needs no reasoning: with the thinking channel on, the
        # model spends the whole token budget on a chain of thought and the
        # translation itself gets truncated (measured: 4096 completion tokens
        # and 190 s per call, with no usable output).
        thinking=False,
    )
    translation = restore(resp.content.strip(), originals)
    return {"translation": translation, "target_lang": target_lang,
            "protected": len(originals), "source": text}


# --------------------------------------------------------------------------- prose
# PDF text arrives with hard line breaks and hyphenation at the line end
# ("apply-\ning pre-trained").  Left alone, the reader shows the same
# "broken words" the PDF has, and the sentence splitter cannot find a
# boundary.  These helpers turn a raw block into readable prose.

_HYPHEN_BREAK = re.compile(r"(?<![-\w])([A-Za-z]{2,})-\n([a-z]{2,})")
_LINE_BREAK = re.compile(r"\s*\n\s*")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

#: Abbreviations that end with a period but do not end a sentence.
_ABBREVIATIONS = {
    "e.g", "i.e", "et al", "vs", "etc", "cf", "al", "resp", "approx",
    "fig", "eq", "eqs", "sec", "secs", "tab", "tabs", "ref", "refs",
    "app", "no", "vol", "pp", "mr", "mrs", "ms", "dr", "prof", "st",
}

#: A number, with an optional fraction and an optional exponent.
_DECIMAL_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

#: The exponent of a scientific notation ("5e-4").
_EXPONENT = re.compile(r"[eE][-+]?\d")

#: What may follow a number that was split off by the layout ("8. 9%").
_UNIT_SUFFIXES = {"%", "‰", "°", ")", "]", ",", ".", ":", "、"}


def normalise_prose(text: str) -> str:
    """Join hard-wrapped lines and repair hyphenated line breaks."""
    if not text:
        return ""
    out = text.replace("\u00a0", " ").replace("\u2018", "'").replace("\u2019", "'")
    out = _HYPHEN_BREAK.sub(r"\1\2", out)
    out = _LINE_BREAK.sub(" ", out)
    out = _MULTI_SPACE.sub(" ", out)
    return out.strip()


def _ends_with_abbreviation(text: str, start: int, dot: int) -> bool:
    """True when the word ending at ``dot`` is an abbreviation or an initial."""
    window = text[start:dot + 1]
    match = re.search(r"([A-Za-z][A-Za-z.]*)\.$", window)
    if not match:
        return False
    word = match.group(1).lower().rstrip(".")
    if word in _ABBREVIATIONS:
        return True
    # Single letter initials: "A. B. Smith".
    return len(word) == 1


def _starts_a_sentence(ch: str) -> bool:
    return ch.isupper() or ch.isdigit() or ch in "\"'“‘([“"


def _is_decimal_point(text: str, dot: int) -> bool:
    """True when the period sits inside a number.

    "8.9% absolute improvement" is one sentence, not two: without this the
    splitter cuts after "8." because a digit may start a sentence, and the
    bilingual reader then shows a broken English fragment on hover.

    A period between two digits ("3.2") is always a decimal point.  A period
    followed by whitespace and then a number ("8. 9%") is only accepted when
    the number carries a unit or an exponent, so that "3. 5 more items" still
    ends its sentence.
    """
    if dot == 0 or not text[dot - 1].isdigit():
        return False
    j = dot + 1
    while j < len(text) and text[j] in "\"'”’)]":
        j += 1
    if j < len(text) and text[j].isdigit():
        return True
    k = j
    while k < len(text) and text[k].isspace():
        k += 1
    if k >= len(text) or not text[k].isdigit():
        return False
    number = _DECIMAL_NUMBER.match(text, k)
    if number is None:
        return False
    if _EXPONENT.search(number.group(0)):
        return True
    return text[number.end():number.end() + 1] in _UNIT_SUFFIXES


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences.

    The splitter is deliberately conservative: a boundary is only accepted
    when the next character can begin a sentence and the previous word is
    not a known abbreviation.  Everything the model translates is therefore
    still a complete sentence, which is what the bilingual reader needs for
    a one-to-one English/Chinese pairing.
    """
    text = normalise_prose(text)
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in ".!?":
            i += 1
            continue
        j = i + 1
        while j < n and text[j] in "\"'”’)]":
            j += 1
        if ch == "." and (_ends_with_abbreviation(text, start, i)
                          or _is_decimal_point(text, i)):
            i = j
            continue
        k = j
        while k < n and text[k].isspace():
            k += 1
        if k >= n:
            # The text ends here, so the sentence is complete.
            sentence = text[start:].strip()
            if sentence:
                sentences.append(sentence)
            start = n
            break
        if not _starts_a_sentence(text[k]):
            i = j
            continue
        sentence = text[start:k].strip()
        if sentence:
            sentences.append(sentence)
        start = k
        i = k
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def merge_short_segments(sentences: list[str], min_len: int = 20) -> list[str]:
    """Glue very short sentences to their neighbour.

    A standalone "However." is not a translatable unit and looks broken in
    the reader, so it is merged into the sentence before it.  The threshold is
    deliberately low: a real sentence of twenty characters still deserves
    its own Chinese counterpart.
    """
    merged: list[str] = []
    for sentence in sentences:
        if merged and len(sentence) < min_len:
            merged[-1] = f"{merged[-1]} {sentence}".strip()
            continue
        merged.append(sentence)
    return merged


# --------------------------------------------------------------------------- document
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_HEADING = re.compile(r"^\s*(#{1,4})\s+(.*)$")

#: Block types that carry running prose worth translating.
_TRANSLATABLE = {"body", "text", "caption", ""}


async def build_document_segments(
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> list[Segment]:
    """Split a parsed document into aligned segments for the bilingual reader.

    The unit is a **sentence**: one English sentence becomes one segment so
    the reader can pair it with exactly one Chinese sentence and reveal the
    source on hover.  Headings are kept as their own segments.
    """
    segments: list[Segment] = []
    idx = 0

    # Headings first so the reader can show a bilingual outline.
    for sec in sections:
        title = (sec.get("title") or "").strip()
        if not title:
            continue
        idx += 1
        segments.append(Segment(index=idx, source=title, kind="heading",
                                page_no=sec.get("page_no", 0),
                                section_title=title))

    # Body paragraphs, split into sentences.
    for order, block in enumerate(blocks):
        block_type = (block.get("type") or block.get("block_type") or "").lower()
        if block_type and block_type not in _TRANSLATABLE:
            continue
        text = (block.get("content") or "").strip()
        if not text:
            continue
        kind = "caption" if block_type == "caption" else "paragraph"
        for para in _PARAGRAPH_SPLIT.split(text):
            para = para.strip()
            if not para:
                continue
            for sentence in merge_short_segments(split_sentences(para)):
                if len(sentence) < 12:
                    continue
                idx += 1
                segments.append(Segment(
                    index=idx, source=sentence, kind=kind,
                    page_no=block.get("page_no", 0),
                    block_id=str(block.get("id") or ""),
                    block_order=order,
                ))
    return segments


@dataclass
class TranslateOutcome:
    """The result of translating a whole document."""

    segments: list[Segment]
    total: int = 0
    done: int = 0
    failed: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "done": self.done, "ok": self.ok,
                "failed": self.failed[:50], "errors": self.errors[:10]}


#: ``0: 译文`` — one line per sentence, the index is echoed back.
_NUMBERED_LINE = re.compile(r"^\s*\[?(\d{1,4})\]?\s*[:：]\s*(.*)$")

_TRANSLATE_SYSTEM = (
    "You are a professional academic translator. Translate English academic prose "
    "into precise, formal Simplified Chinese. Preserve every technical term, "
    "number, unit, citation marker and placeholder exactly. Never add, drop or "
    "reorder clauses."
)


async def _translate_batch(
    llm: Any,
    batch: list[Segment],
) -> tuple[dict[int, str], list[str]]:
    """Translate one batch; returns ``{index_in_batch: translation}`` and errors."""
    masked_sources: list[str] = []
    originals: list[list[str]] = []
    for seg in batch:
        masked, orig = protect(seg.source)
        seg._protected = orig  # type: ignore[attr-defined]
        masked_sources.append(masked)
        originals.append(orig)

    numbered = "\n".join(f"{i}: {text}" for i, text in enumerate(masked_sources))
    user = (
        f"Translate the following {len(batch)} numbered English sentences into "
        f"Simplified Chinese.\n"
        f"Rules:\n"
        f"- Output exactly {len(batch)} lines, one per sentence, in the same order.\n"
        f"- Every line starts with its original number, then a colon and a space "
        f"(for example \"0: ...\").\n"
        f"- Do not merge, split, reorder or omit a sentence.\n"
        f"- Keep the placeholder markers, numbers, units, citations and formulas "
        f"exactly as they are.\n\n"
        f"{numbered}"
    )
    resp = await llm.chat(
        [{"role": "system", "content": _TRANSLATE_SYSTEM},
         {"role": "user", "content": user}],
        temperature=0.1,
        purpose="translate:doc",
        # No reasoning channel: it eats the token budget and the translation
        # never makes it out (measured: 4096 tokens / 190 s per batch).
        thinking=False,
        max_tokens=2048,
    )
    if resp.finish_reason == "error" or not resp.content.strip():
        return {}, [str(resp.raw.get("error", "empty response"))[:200] or "empty response"]

    parsed: dict[int, str] = {}
    for line in resp.content.splitlines():
        match = _NUMBERED_LINE.match(line)
        if not match:
            continue
        try:
            index = int(match.group(1))
        except ValueError:
            continue
        if 0 <= index < len(batch) and index not in parsed:
            parsed[index] = match.group(2).strip()

    # The model did not answer for every sentence: retry those one by one so
    # the English/Chinese pairing stays one-to-one.
    errors: list[str] = []
    for index in range(len(batch)):
        if index in parsed and parsed[index]:
            continue
        single = await _translate_single(llm, masked_sources[index])
        if single:
            parsed[index] = single
        else:
            errors.append(f"segment {batch[index].index} could not be translated")
    return parsed, errors


async def _translate_single(llm: Any, masked: str) -> str:
    resp = await llm.chat(
        [{"role": "system", "content": _TRANSLATE_SYSTEM},
         {"role": "user", "content":
          f"Translate the following English sentence into Simplified Chinese. "
          f"Output only the translation.\n\n{masked}"}],
        temperature=0.1,
        purpose="translate:doc",
        thinking=False,
        max_tokens=1024,
    )
    if resp.finish_reason == "error" or not resp.content.strip():
        return ""
    return resp.content.strip()


async def translate_document(
    segments: list[Segment],
    target_lang: str = "zh",
    *,
    batch_size: int = 8,
    concurrency: int = 6,
    progress_cb=None,
) -> TranslateOutcome:
    """Translate every segment, keeping the original order (parallel alignment).

    The batches run **concurrently**: a paper has several hundred sentences
    and the model server handles parallel requests far better than a long
    serial chain.  A failed batch is retried per sentence, and a segment
    that still cannot be translated is reported instead of being silently
    replaced by its English source.
    """
    llm = get_llm_client()
    total = len(segments)
    outcome = TranslateOutcome(segments=segments, total=total)
    if not total:
        return outcome

    batches = [segments[start:start + batch_size] for start in range(0, total, batch_size)]
    semaphore = asyncio.Semaphore(max(1, concurrency))
    done = 0
    lock = asyncio.Lock()

    async def run_batch(batch: list[Segment]) -> None:
        nonlocal done
        async with semaphore:
            parsed, errors = await _translate_batch(llm, batch)
        async with lock:
            for index, seg in enumerate(batch):
                translation = parsed.get(index, "")
                if translation:
                    seg.translation = restore(translation, getattr(seg, "_protected", []))
                else:
                    outcome.failed.append(seg.index)
            outcome.errors.extend(errors)
            done += len(batch)
            outcome.done = done
            if progress_cb is not None:
                await progress_cb(done / max(1, total))

    await asyncio.gather(*(run_batch(batch) for batch in batches))
    return outcome
