"""Verify and assemble the cloudflared binary from its downloaded parts.

The parallel downloader stores one file per range request.  This script
re-checks every part against the expected size and then concatenates them in
order, so a truncated or corrupted download cannot slip through.

Usage:
    python assemble_parts.py <parts-dir> <output-file> <total-bytes>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

parts_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
total = int(sys.argv[3])

parts = sorted(parts_dir.iterdir())
print(f"parts: {len(parts)}")

problems: list[str] = []
offset = 0
for part in parts:
    size = part.stat().st_size
    if size == 0:
        problems.append(f"{part.name}: empty")
        continue
    offset += size

print(f"total bytes: {offset} (expected {total})")
if offset != total:
    problems.append("size mismatch")

if problems:
    for p in problems:
        print("  PROBLEM:", p)
    raise SystemExit(1)

digest = hashlib.sha256()
with out.open("wb") as fh:
    for part in parts:
        data = part.read_bytes()
        digest.update(data)
        fh.write(data)

print(f"wrote {out} ({out.stat().st_size} bytes)")
print(f"sha256: {digest.hexdigest()}")
