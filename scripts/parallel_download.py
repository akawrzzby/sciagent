#!/usr/bin/env python3
"""Download a large file with parallel HTTP range requests.

Large release assets are frequently rate limited to a few KB/s on a single
connection from academic networks.  Splitting the file into ranges and
fetching them concurrently usually bypasses that limit - the same trick that
makes ``scripts/download_models.py`` able to pull a 2.3 GB model in four
minutes.

Usage::

    python scripts/parallel_download.py <url> <output-file> [--workers 16]
                                        [--chunk-mb 16]

Only the standard library is used, so it runs on Windows, Linux and macOS.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK_DEFAULT = 16 * 1024 * 1024


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def content_length(url: str) -> int:
    """Ask the server for the size of the resource."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "sciagent/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers.get("Content-Length") or 0)


def fetch_range(url: str, start: int, end: int, dest: Path, retries: int = 5) -> int:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "sciagent/1.0", "Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1024 * 1024)
            got = dest.stat().st_size
            expect = end - start + 1
            if got == expect:
                return got
            last = RuntimeError(f"short read {got} != {expect}")
        except Exception as exc:  # noqa: BLE001 - retry on anything network-ish
            last = exc
        time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"range {start}-{end} failed: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url")
    ap.add_argument("output")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk_mb * 1024 * 1024

    total = content_length(args.url)
    if total <= 0:
        print("the server did not report a Content-Length", file=sys.stderr)
        return 2
    print(f"==> {human(total)} in {total // chunk + 1} chunks, {args.workers} workers")

    parts_dir = out.parent / f".{out.name}.parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    bounds = []
    index = 0
    while index < total:
        start = index * chunk
        end = min(start + chunk - 1, total - 1)
        bounds.append((start, end, index))
        index += 1

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_range, args.url, s, e, parts_dir / f"part{i:05d}"): (s, e, i)
            for s, e, i in bounds
        }
        for fut in futures:
            fut.result()
            done += 1
            elapsed = time.time() - started
            rate = done * chunk / max(elapsed, 1e-6)
            print(f"    {done}/{len(bounds)} chunks  {human(rate)}/s", flush=True)

    with out.open("wb") as fh:
        for i in range(len(bounds)):
            with (parts_dir / f"part{i:05d}").open("rb") as src:
                shutil.copyfileobj(src, fh, length=1024 * 1024)
    shutil.rmtree(parts_dir, ignore_errors=True)

    got = out.stat().st_size
    if got != total:
        print(f"size mismatch: {got} != {total}", file=sys.stderr)
        return 1
    print(f"==> {out}  ({human(got)}) in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
