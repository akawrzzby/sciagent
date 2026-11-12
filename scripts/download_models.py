#!/usr/bin/env python3
"""Download the retrieval models from ModelScope with parallel range requests.

Why ModelScope instead of HuggingFace?  In this lab network the HF mirror
stalls at 0 B/s on the 2.2 GB weight blob, while ModelScope serves the same
file over plain HTTPS and honours ``Range`` requests.  Splitting every file
into 16 MB ranges and pulling them with a thread pool turns a stalled
transfer into a fast one.

The result is a plain HuggingFace directory, so it can be loaded with

    SentenceTransformer("/home/<user>/models/bge-m3")

Usage::

    python download_models.py BAAI/bge-m3 ~/models/bge-m3 --workers 16
    python download_models.py BAAI/bge-reranker-v2-m3 ~/models/bge-reranker-v2-m3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API_TMPL = "https://www.modelscope.cn/api/v1/models/{model}/repo/files?Revision=master"
# The "resolve" endpoint serves both top-level blobs and files inside
# sub-directories (e.g. ``1_Pooling/config.json``), which the tree API does
# not enumerate.  It is therefore the primary download URL.
RESOLVE_TMPL = "https://modelscope.cn/models/{model}/resolve/master/{path}"
REPO_TMPL = (
    "https://www.modelscope.cn/api/v1/models/{model}/repo"
    "?Revision=master&FilePath={path}"
)
CHUNK = 16 * 1024 * 1024

# Auxiliary trees that only bloat the download.
SKIP_PREFIXES = ("onnx/", "imgs/", "onnx_int8/", "openvino/", "colbert/")
SKIP_NAMES = {
    ".gitattributes",
    "README.md",
    "README_zh.md",
    "config_sentence_transformers.json",
    "colbert_linear.pt",
    "sparse_linear.pt",
}

# sentence-transformers keeps the pooling / normalisation configuration in
# sub-directories that the tree API reports as opaque "tree" nodes.  They are
# small, so we simply probe for them.
EXTRA_PATHS = (
    "1_Pooling/config.json",
    "2_Normalize/config.json",
    "2_Dense/config.json",
    "sentence_bert_config.json",
)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def list_files(model: str) -> list[tuple[str, int]]:
    """Return ``[(path, size)]`` for every blob in the repository."""
    url = API_TMPL.format(model=model)
    req = urllib.request.Request(url, headers={"User-Agent": "sciagent/1.0"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        data = json.load(fh)

    files: list[tuple[str, int]] = []
    for node in data.get("Data", {}).get("Files", []):
        if node.get("Type") != "blob":
            continue
        path = node.get("Path", "")
        if path in SKIP_NAMES or path.startswith(SKIP_PREFIXES):
            continue
        files.append((path, int(node.get("Size") or 0)))
    return sorted(files)


def _fetch_range(url: str, start: int, end: int, dest: Path, retries: int = 4) -> int:
    """Download ``[start, end]`` into ``dest``; returns the byte count."""
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
        time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"range {start}-{end} failed: {last}")


def _download_small(model: str, path: str, out: Path) -> bool:
    """Download a small file, trying the resolve endpoint first."""
    quoted = urllib.parse.quote(path)
    for tmpl in (RESOLVE_TMPL, REPO_TMPL):
        url = tmpl.format(model=model, path=quoted)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sciagent/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if not data:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            return True
        except Exception:  # noqa: BLE001 - try the next endpoint
            continue
    return False


def download_file(model: str, path: str, size: int, target: Path, workers: int) -> bool:
    out = target / path
    if out.exists() and out.stat().st_size == size and size > 0:
        print(f"  skip  {path} ({human(size)})")
        return True
    if out.exists() and size == 0 and out.stat().st_size > 0:
        print(f"  skip  {path}")
        return True

    out.parent.mkdir(parents=True, exist_ok=True)
    url = RESOLVE_TMPL.format(model=model, path=urllib.parse.quote(path))

    if size <= CHUNK:
        tmp = out.with_suffix(out.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sciagent/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh, length=1024 * 1024)
            if tmp.stat().st_size == 0:
                tmp.unlink(missing_ok=True)
                if _download_small(model, path, out):
                    print(f"  ok    {path} (via fallback endpoint)")
                    return True
                print(f"  FAIL  {path}: empty response")
                return False
            tmp.replace(out)
            print(f"  ok    {path} ({human(size)})")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {path}: {exc}")
            tmp.unlink(missing_ok=True)
            if _download_small(model, path, out):
                print(f"  ok    {path} (via fallback endpoint)")
                return True
            return False

    n_chunks = (size + CHUNK - 1) // CHUNK
    print(f"  ..    {path} ({human(size)}, {n_chunks} chunks, {workers} workers)")
    parts_dir = out.parent / f".{out.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    bounds: list[tuple[int, int, int]] = []
    for i in range(n_chunks):
        start = i * CHUNK
        end = min(start + CHUNK - 1, size - 1)
        bounds.append((start, end, i))

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_range, url, s, e, parts_dir / f"part{i:05d}"): (s, e, i)
            for s, e, i in bounds
        }
        for fut in futures:
            try:
                fut.result()
                done += 1
                if done % 8 == 0 or done == n_chunks:
                    print(f"       {done}/{n_chunks} chunks")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL  {path} chunk: {exc}")
                shutil.rmtree(parts_dir, ignore_errors=True)
                return False

    with out.open("wb") as fh:
        for i in range(n_chunks):
            part = parts_dir / f"part{i:05d}"
            with part.open("rb") as src:
                shutil.copyfileobj(src, fh, length=1024 * 1024)
    shutil.rmtree(parts_dir, ignore_errors=True)

    got = out.stat().st_size
    if got != size:
        print(f"  SIZE MISMATCH {path}: {got} != {size}")
        out.unlink(missing_ok=True)
        return False
    print(f"  ok    {path} ({human(size)})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="model id, e.g. BAAI/bge-m3")
    ap.add_argument("target", help="local directory to write the model into")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--file-workers", type=int, default=4)
    args = ap.parse_args()

    target = Path(args.target).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    print(f">>> {args.model} -> {target}")
    try:
        files = list_files(args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"!!! cannot list {args.model}: {exc}", file=sys.stderr)
        return 2
    print(f">>> {len(files)} files")

    total = sum(s for _, s in files)
    print(f">>> total {human(total)}")

    ok = True
    with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
        futures = [
            pool.submit(download_file, args.model, path, size, target, args.workers)
            for path, size in files
        ]
        for fut in futures:
            ok = fut.result() and ok

    # The tree API hides the sentence-transformers sub-configurations, so they
    # are probed explicitly (they are tiny and cost nothing to try).
    for extra in EXTRA_PATHS:
        dest = target / extra
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  skip  {extra}")
            continue
        if _download_small(args.model, extra, dest):
            print(f"  ok    {extra} (probed)")
        else:
            print(f"  --    {extra} not present in this repository")

    print(">>> done")
    for p in sorted(target.rglob("*")):
        if p.is_file():
            print(f"  {p.stat().st_size:>12} {p.relative_to(target)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
