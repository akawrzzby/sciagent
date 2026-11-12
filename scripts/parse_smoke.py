"""Smoke test for the parsing pipeline against the official test dataset.

Run on the GPU host::

    ~/envs/sciagent/bin/python ~/sciagent/scripts/parse_smoke.py

It exercises the three official parsing use-cases:

* ``1706.03762v1.pdf``   – standard two-column paper
* ``1706.03762v1_img.pdf`` – scanned paper (VLM OCR)
* ``UnderstandingDeepLearning_*.pdf`` – 541 page book
"""

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/sciagent/backend"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from app.parsing import pipeline  # noqa: E402

DATA = os.path.expanduser("~/ra_testdata")


async def run_one(path: str, *, do_ocr: bool = True, limit_pages: int = 0) -> dict:
    name = os.path.basename(path)
    t0 = time.time()

    async def progress(frac, msg):
        if frac >= 0.95 or frac < 0.05:
            print("    %5.1f%%  %s" % (frac * 100, msg), flush=True)

    parsed = await pipeline.parse_document(
        path, paper_id="smoke", out_dir="/tmp/smoke_pages",
        do_ocr=do_ocr, transcribe_formulas=False, progress=progress,
    )
    meta = parsed.metadata
    return {
        "file": name,
        "seconds": round(time.time() - t0, 1),
        "pages": parsed.page_count,
        "scanned": parsed.is_scanned,
        "ocr_pages": parsed.ocr_pages,
        "title": meta.title,
        "authors": [a.name for a in meta.authors][:8],
        "affiliations": meta.affiliations[:4],
        "abstract_chars": len(meta.abstract),
        "abstract_head": meta.abstract[:180],
        "year": meta.year,
        "arxiv_id": meta.arxiv_id,
        "version": meta.version_label,
        "venue": meta.venue,
        "n_sections": len(parsed.sections),
        "sections_head": [
            {"n": s["number"], "t": s["title"][:60], "lvl": s["level"], "p": s["page_no"]}
            for s in parsed.sections[:12]
        ],
        "n_tables": len(parsed.tables),
        "n_figures": len(parsed.figures),
        "n_formulas": len(parsed.formulas),
        "n_references": len(parsed.references),
        "ref_head": [r["raw_text"][:90] for r in parsed.references[:3]],
        "text_chars": len(parsed.full_text),
        "warnings": parsed.warnings[:3],
    }


async def main() -> None:
    targets = [
        (os.path.join(DATA, "1706.03762v1.pdf"), True),
        (os.path.join(DATA, "1706.03762v7.pdf"), False),
    ]
    if "--scan" in sys.argv:
        targets.append((os.path.join(DATA, "1706.03762v1_img.pdf"), True))
    if "--book" in sys.argv:
        targets.append((os.path.join(DATA, "UnderstandingDeepLearning_02_09_26_C.pdf"), False))

    results = []
    for path, ocr in targets:
        if not os.path.exists(path):
            print("SKIP (missing):", path)
            continue
        print("=" * 78)
        print("PARSING", os.path.basename(path), flush=True)
        res = await run_one(path, do_ocr=ocr)
        results.append(res)
        print(json.dumps(res, ensure_ascii=False, indent=1)[:2600], flush=True)

    print("=" * 78)
    print("SUMMARY")
    for r in results:
        print("  %-44s %6.1fs pages=%-4s secs=%-4s refs=%-4s tables=%-3s figs=%-3s"
              % (r["file"][:44], r["seconds"], r["pages"], r["n_sections"],
                 r["n_references"], r["n_tables"], r["n_figures"]))


asyncio.run(main())
