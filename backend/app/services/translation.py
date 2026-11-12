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

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "source": self.source, "translation": self.translation,
                "kind": self.kind, "page_no": self.page_no,
                "section_title": self.section_title}


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
    )
    translation = restore(resp.content.strip(), originals)
    return {"translation": translation, "target_lang": target_lang,
            "protected": len(originals), "source": text}


# --------------------------------------------------------------------------- document
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_HEADING = re.compile(r"^\s*(#{1,4})\s+(.*)$")


async def build_document_segments(
    sections: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> list[Segment]:
    """Split a parsed document into aligned segments for the bilingual reader."""
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

    # Body paragraphs (from the blocks, in reading order).
    for block in blocks:
        text = (block.get("content") or "").strip()
        if not text:
            continue
        for para in _PARAGRAPH_SPLIT.split(text):
            para = para.strip()
            if len(para) < 40:
                continue
            idx += 1
            segments.append(Segment(index=idx, source=para, kind="paragraph",
                                    page_no=block.get("page_no", 0)))
    return segments


async def translate_document(
    segments: list[Segment],
    target_lang: str = "zh",
    *,
    batch_size: int = 4,
    progress_cb=None,
) -> list[Segment]:
    """Translate every segment, keeping the original order (parallel alignment)."""
    llm = get_llm_client()
    total = len(segments)
    done = 0

    for start in range(0, total, batch_size):
        batch = segments[start : start + batch_size]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.SYSTEM_TRANSLATE},
        ]
        for i, seg in enumerate(batch):
            masked, originals = protect(seg.source)
            seg._protected = originals  # type: ignore[attr-defined]
            messages.append({
                "role": "user",
                "content": f"--- SEGMENT {i} ---\n{masked}",
            })
        messages.append({
            "role": "user",
            "content": "Translate every SEGMENT above. Reply with the translations in "
                       "the same order, separated by a line containing only '---'.",
        })

        resp = await llm.chat(messages, temperature=0.1, purpose="translate:doc")
        parts = re.split(r"^---\s*$", resp.content, flags=re.MULTILINE)

        # The reply may contain the segment markers; split on them instead.
        marked = re.split(r"---+\s*SEGMENT\s*(\d+)\s*-+", resp.content, flags=re.IGNORECASE)
        if len(marked) > 1:
            for j in range(1, len(marked) - 1, 2):
                try:
                    idx_in_batch = int(marked[j])
                except ValueError:
                    continue
                if 0 <= idx_in_batch < len(batch):
                    seg = batch[idx_in_batch]
                    originals = getattr(seg, "_protected", [])
                    seg.translation = restore(marked[j + 1].strip(), originals)
        else:
            for i, seg in enumerate(batch):
                if i < len(parts):
                    originals = getattr(seg, "_protected", [])
                    seg.translation = restore(parts[i].strip(), originals)

        for seg in batch:
            if not seg.translation:
                seg.translation = seg.source  # fall back to the original
            done += 1
            if progress_cb is not None:
                await progress_cb(done / max(1, total))
    return segments
