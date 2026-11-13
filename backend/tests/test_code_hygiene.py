"""Static hygiene of the backend package.

Two production bugs slipped through the unit tests because they lived on
paths that no test touched: ``chunk.section_title`` (a column the model did
not have) and ``os`` (a module that was never imported).  Both blow up at
*call* time, not at import time, so a test that only imports the module
cannot see them.

This test runs an AST based scope check over every module of the package: a
name that is read but never bound, imported or passed in is a bug waiting to
happen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
CHECKER = REPO_ROOT / "scripts" / "check_undefined.py"


def _load_checker():
    if not CHECKER.is_file():
        pytest.skip(f"{CHECKER} is not available")
    spec = importlib.util.spec_from_file_location("_undefined_check", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_undefined_check"] = module
    spec.loader.exec_module(module)
    return module


def test_no_module_uses_a_name_it_never_binds() -> None:
    checker = _load_checker()
    findings: list[str] = []
    for path in sorted(checker.iter_files(str(BACKEND_DIR / "app"))):
        for lineno, name, kind in checker.check_file(path):
            rel = Path(path).relative_to(REPO_ROOT)
            findings.append(f"{rel}:{lineno}: {kind}: {name}")
    assert not findings, "undefined names found:\n" + "\n".join(findings)


def test_the_checker_itself_finds_a_missing_import(tmp_path: Path) -> None:
    """A checker that never reports anything is worthless."""
    checker = _load_checker()
    bad = tmp_path / "bad.py"
    bad.write_text("def f():\n    return os.path.basename('x')\n", encoding="utf-8")
    good = tmp_path / "good.py"
    good.write_text("import os\n\ndef f():\n    return os.path.basename('x')\n",
                    encoding="utf-8")
    assert checker.check_file(str(bad)), "the checker missed a missing import"
    assert not checker.check_file(str(good)), "the checker reported a false positive"
