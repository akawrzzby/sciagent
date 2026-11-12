"""Print row counts for every table of the live SciAgent database.

Run on the server:
    ~/envs/sciagent/bin/python ~/sciagent/scripts/db_inspect.py [db-path]
"""

from __future__ import annotations

import os
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/sciagent/data/sciagent.db")

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = con.cursor()

tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

print(f"db: {DB}")
for t in tables:
    try:
        n = cur.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        n = f"ERR {exc}"
    print(f"  {t:24s} {n}")

print("\n-- papers")
try:
    for row in cur.execute(
        "SELECT id, substr(title,1,60), status, page_count FROM papers ORDER BY created_at"
    ):
        print(" ", row)
except Exception as exc:  # noqa: BLE001
    print("  ERR", exc)

print("\n-- jobs")
try:
    for row in cur.execute("SELECT kind, status, progress, substr(message,1,70) FROM jobs"):
        print(" ", row)
except Exception as exc:  # noqa: BLE001
    print("  ERR", exc)

con.close()
