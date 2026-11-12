import sys, os, json
sys.path.insert(0, os.path.expanduser("~/sciagent/backend"))
import pymupdf
from app.parsing import layout

path = os.path.expanduser("~/ra_testdata/1706.03762v1.pdf")
doc = pymupdf.open(path)
page = doc[0]
blocks = layout.extract_blocks(page, 1)
n_cols = layout.detect_columns(blocks, page.rect.width)
layout.assign_columns(blocks, page.rect.width, n_cols)
print("page size", page.rect, "cols:", n_cols)
print("=" * 100)
for b in blocks:
    print(f"[x0={b.x0:6.1f} y0={b.y0:6.1f} x1={b.x1:6.1f} y1={b.y1:6.1f} "
          f"size={b.font_size:5.1f} bold={b.is_bold} col={b.column}] {b.text[:110]!r}")
print("=" * 100)
print("spans of first 6 blocks:")
for b in blocks[:6]:
    print(f"-- block y0={b.y0:.1f} size={b.font_size}")
    for s in b.spans[:8]:
        print(f"     size={s.size:5.2f} flags={s.flags:3d} font={s.font[:28]:28s} {s.text[:60]!r}")
