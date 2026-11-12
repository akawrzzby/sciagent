"""Check whether PyMuPDF's ``find_tables`` is safe to call from several threads.

The hypothesis: ``Page.find_tables()`` mutates global PyMuPDF state, so calling
it concurrently on pages of *different* document handles loses results.

Run on the server:
    ~/envs/sciagent/bin/python ~/sciagent/scripts/check_table_threads.py
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pymupdf

PATH = os.path.expanduser("~/ra_testdata/1706.03762v1.pdf")


def count_tables(doc: pymupdf.Document, page_no: int) -> int:
    try:
        finder = doc[page_no].find_tables()
    except Exception:
        return 0
    return len(getattr(finder, "tables", []))


def main() -> None:
    if not os.path.exists(PATH):
        print("SKIP (missing):", PATH)
        return

    # --- reference: one document, serial ----------------------------------
    doc = pymupdf.open(PATH)
    try:
        serial = [count_tables(doc, i) for i in range(doc.page_count)]
    finally:
        doc.close()
    print("serial,   shared handle :", serial, "total =", sum(serial))

    # --- serial, separate handles ----------------------------------------
    def one(i: int) -> int:
        d = pymupdf.open(PATH)
        try:
            return count_tables(d, i)
        finally:
            d.close()

    separate = [one(i) for i in range(len(serial))]
    print("serial,   own handle    :", separate, "total =", sum(separate))

    # --- concurrent, separate handles ------------------------------------
    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(one, range(len(serial))))
    print("parallel, own handle    :", concurrent, "total =", sum(concurrent))

    # --- concurrent, shared handle (what the pipeline does now) ----------
    shared = pymupdf.open(PATH)

    def one_shared(i: int) -> int:
        return count_tables(shared, i)

    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent_shared = list(pool.map(one_shared, range(shared.page_count)))
    shared.close()
    print("parallel, shared handle :", concurrent_shared, "total =", sum(concurrent_shared))


if __name__ == "__main__":
    main()
