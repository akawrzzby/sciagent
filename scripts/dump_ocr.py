"""Dump the raw OCR Markdown for the first pages of the scanned test paper.

The scanned-paper pipeline has to rebuild the heading tree, the front matter
and the bibliography from the model's transcription.  This script prints the
transcription so the reconstruction rules can be written against the real
output instead of a guess.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.expanduser("~/sciagent/backend"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import pymupdf  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.llm.client import get_llm_client  # noqa: E402
from app.parsing import layout, ocr  # noqa: E402

DATA = os.path.expanduser("~/ra_testdata")
PDF = os.path.join(DATA, "1706.03762v1_img.pdf")
PAGES = int(os.environ.get("OCR_DUMP_PAGES", "3"))


async def main() -> None:
    settings = get_settings()
    client = get_llm_client()
    doc = pymupdf.open(PDF)

    png_pages = []
    for index in range(min(PAGES, doc.page_count)):
        page = doc[index]
        png_pages.append((index + 1, layout.render_page_png(page, dpi=settings.render_dpi)))

    results = await ocr.ocr_document(png_pages, client=client, session_id="dump")

    for res in results:
        print("=" * 78)
        print(f"PAGE {res.page_no}  ok={res.ok}")
        print("=" * 78)
        print(res.markdown)
    doc.close()


asyncio.run(main())
