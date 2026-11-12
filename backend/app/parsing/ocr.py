"""OCR for scanned documents using the multimodal LLM.

The competition dataset contains ``1706.03762v1_img.pdf`` — a rasterised
version of *Attention Is All You Need* without a text layer.  Classical OCR
(tesseract) would destroy the two-column layout and the mathematics.

Our model (``Qwen3.8-27B``) is a **vision-language model**; we render the page
at 150 dpi and ask it to transcribe the page into structured Markdown that
preserves headings, paragraphs, table stubs and formula placeholders.

The OCR prompt is deliberately strict so the output can be fed straight into
the normal (text based) parsing pipeline afterwards.
"""

from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.llm.client import LLMClient, get_llm_client

logger = get_logger(__name__)

OCR_SYSTEM_PROMPT = (
    "You are an OCR engine for academic papers. You transcribe a single page image "
    "into faithful Markdown.\n"
    "Rules:\n"
    "1. Output ONLY the transcription. No preamble, no commentary.\n"
    "2. Preserve the reading order of the printed page (for two-column papers read "
    "the left column top-to-bottom, then the right column).\n"
    "3. Keep headings as Markdown headings (#, ##, ###) matching their level.\n"
    "4. Keep tables as Markdown tables; keep formulas as LaTeX inside $...$ or $$...$$.\n"
    "5. Keep author names, affiliations and reference lists verbatim.\n"
    "6. Never invent text that is not visible. If a region is illegible write [illegible]."
)


@dataclass(slots=True)
class OCRPage:
    """OCR output for one page."""

    page_no: int
    markdown: str
    raw_response: str = ""
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"page_no": self.page_no, "markdown": self.markdown,
                "ok": self.ok, "error": self.error}


def encode_image(png_bytes: bytes) -> str:
    """Base64-encode a PNG for the OpenAI-style ``image_url`` payload."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


async def ocr_page(
    client: LLMClient,
    png_bytes: bytes,
    page_no: int,
    *,
    language: str = "en",
    session_id: str = "",
) -> OCRPage:
    """Transcribe a single rendered page with the VLM."""
    user = (
        "Transcribe the page image below into Markdown."
        + (f" The page language is {language}." if language != "en" else "")
    )
    messages = [
        {"role": "system", "content": OCR_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": encode_image(png_bytes)}},
        ]},
    ]
    try:
        resp = await client.chat(
            messages,
            max_tokens=4096,
            temperature=0.0,
            session_id=session_id,
            purpose=f"ocr:page{page_no}",
        )
    except Exception as exc:
        return OCRPage(page_no=page_no, markdown="", ok=False, error=str(exc)[:300])

    if resp.finish_reason == "error":
        return OCRPage(page_no=page_no, markdown="", raw_response=resp.content,
                       ok=False, error=str(resp.raw.get("error", "llm error"))[:300])

    text = _strip_preamble(resp.content)
    return OCRPage(page_no=page_no, markdown=text, raw_response=resp.content)


def _strip_preamble(text: str) -> str:
    """Remove chatty preambles the model sometimes adds."""
    if not text:
        return ""
    out = text.strip()
    out = re.sub(r"^```(?:markdown)?\s*", "", out)
    out = re.sub(r"```\s*$", "", out).strip()
    for marker in ("Here is the transcription", "Here is the Markdown transcription",
                   "Transcription:", "Markdown:"):
        idx = out.find(marker)
        if 0 <= idx < 60:
            out = out[idx + len(marker) :].lstrip(" :\n")
    return out


async def ocr_document(
    png_pages: list[tuple[int, bytes]],
    *,
    client: LLMClient | None = None,
    session_id: str = "",
    progress_cb=None,
    concurrency: int = 4,
) -> list[OCRPage]:
    """OCR a whole scanned document.

    Pages are transcribed **concurrently** (bounded by ``concurrency``): the
    vLLM server batches the requests internally, so a 15 page scan finishes in
    a fraction of the time a sequential loop would need.  Results are returned
    in page order regardless of completion order.
    """
    client = client or get_llm_client()
    total = max(1, len(png_pages))
    semaphore = asyncio.Semaphore(max(1, concurrency))
    done = 0
    done_lock = asyncio.Lock()

    async def _one(page_no: int, png: bytes) -> OCRPage:
        nonlocal done
        async with semaphore:
            result = await ocr_page(client, png, page_no, session_id=session_id)
        async with done_lock:
            done += 1
            if progress_cb is not None:
                await progress_cb(page_no, done / total, result)
        return result

    tasks = [_one(page_no, png) for page_no, png in png_pages]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out: list[OCRPage] = []
    for (page_no, _png), res in zip(png_pages, results):
        if isinstance(res, OCRPage):
            out.append(res)
        else:
            out.append(OCRPage(page_no=page_no, markdown="", ok=False,
                               error=str(res)[:300]))
    out.sort(key=lambda r: r.page_no)
    return out


def ocr_markdown_to_sections(markdown: str) -> list[tuple[int, str, str]]:
    """Parse the OCR Markdown into ``(level, number, title)`` triples."""
    out: list[tuple[int, str, str]] = []
    for line in markdown.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)$", line.strip())
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        num = ""
        nm = re.match(r"^((?:\d{1,2}(?:\.\d{1,2}){0,3})\.)\s+(.*)$", title)
        if nm:
            num = nm.group(1).rstrip(".")
            title = nm.group(2)
        out.append((level, num, title))
    return out


# --------------------------------------------------------------------------- structure
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_NUMBER = re.compile(r"^((?:\d{1,2}(?:\.\d{1,2}){0,4})\.?)\s+(.*)$")

# Font size used for a heading of each Markdown level.  The OCR output has no
# typography, so the level has to be re-expressed as a size for the blocks to
# look like the ones produced by the text based parser.
_HEADING_SIZES = {1: 16.0, 2: 13.5, 3: 12.0, 4: 11.0, 5: 10.5, 6: 10.0}

_BODY_SIZE = 10.0


def clean_heading_text(text: str) -> str:
    """Strip Markdown decoration and inline LaTeX from a heading."""
    out = text.strip()
    out = re.sub(r"\*\*(.+?)\*\*", r"\1", out)
    out = re.sub(r"__(.+?)__", r"\1", out)
    out = re.sub(r"\*(.+?)\*", r"\1", out)
    out = re.sub(r"\$[^$]{0,60}\$", " ", out)      # inline LaTeX markers
    out = out.replace("\\*", "*").replace("\\_", "_")
    return re.sub(r"\s+", " ", out).strip().rstrip("#").strip()


def markdown_to_blocks(
    markdown: str,
    *,
    page_no: int,
    start_index: int = 0,
    y_top: float = 0.0,
) -> tuple[list[Any], list[Any]]:
    """Turn an OCR Markdown page into layout blocks and heading candidates.

    Unlike the text based parser - which infers headings from font size and
    numbering - the scanned path has an explicit Markdown structure.  We use it
    directly: the ``#`` prefix gives the level, and a printed section number
    such as ``3.2.1`` refines it further.

    Returns ``(blocks, heading_candidates)``.
    """
    from app.parsing import headings as headings_mod
    from app.parsing import layout as layout_mod

    blocks: list[Any] = []
    candidates: list[Any] = []
    y = y_top
    index = start_index

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            y += 12
            continue

        m = _MD_HEADING.match(line.strip())
        if m:
            md_level = len(m.group(1))
            title = clean_heading_text(m.group(2))
            number = ""
            nm = _MD_NUMBER.match(title)
            if nm:
                number = nm.group(1).rstrip(".")
                title = nm.group(2).strip()
            # The depth of the printed number is more reliable than the
            # Markdown depth (the model may emit "##" for a level-3 section).
            level = number.count(".") + 1 if number else min(md_level, 4)
            size = _HEADING_SIZES.get(md_level, _BODY_SIZE)
            height = 14 + 10 * (len(title) // 90)
            bbox = (60.0, y, 540.0, y + height)
            span = layout_mod.Span(text=title, bbox=bbox, size=size, flags=16, font="OCR-Bold")
            block = layout_mod.Block(text=title, bbox=bbox, spans=[span],
                                     page_no=page_no, block_index=index)
            blocks.append(block)
            if title:
                candidates.append(
                    headings_mod.HeadingCandidate(block=block, title=title, number=number,
                                                  level=level, score=1.0))
            index += 1
            y += height + 6
            continue

        text = line.strip()
        height = 12 + 10 * (len(text) // 90)
        bbox = (60.0, y, 540.0, y + height)
        span = layout_mod.Span(text=text, bbox=bbox, size=_BODY_SIZE, flags=0, font="OCR")
        blocks.append(layout_mod.Block(text=text, bbox=bbox, spans=[span],
                                       page_no=page_no, block_index=index))
        index += 1
        y += height + 4

    return blocks, candidates

