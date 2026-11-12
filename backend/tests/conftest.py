"""Shared pytest fixtures.

The test suite must run **without** a live LLM and without network access, so
every fixture here is either synthetic or points at an optional local PDF.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make ``app`` importable when pytest is invoked from the repository root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.parsing.layout import Block, Span  # noqa: E402


# --------------------------------------------------------------------- env
@pytest.fixture(scope="session", autouse=True)
def _isolated_env(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point every runtime directory at a throw-away location.

    The settings singleton is created lazily, so overriding the environment
    before the first ``get_settings()`` call is enough to keep the developer's
    real ``data/`` directory untouched.
    """
    root = tmp_path_factory.mktemp("sciagent-test")
    os.environ.setdefault("DATA_DIR", str(root / "data"))
    os.environ.setdefault("INDEX_DIR", str(root / "index"))
    os.environ.setdefault("AUDIT_DIR", str(root / "audit"))
    os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{root / 'test.db'}")
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ.setdefault("RERANKER_ENABLED", "false")


# ------------------------------------------------------------------ blocks
def make_block(
    text: str,
    *,
    page_no: int = 1,
    size: float = 10.0,
    bold: bool = False,
    x0: float = 50.0,
    y0: float = 50.0,
    x1: float = 300.0,
    y1: float = 70.0,
    column: int = 0,
) -> Block:
    """Build a :class:`Block` with a single synthetic span."""
    flags = 16 if bold else 0
    span = Span(
        text=text,
        bbox=(x0, y0, x1, y1),
        size=size,
        flags=flags,
        font="Helvetica-Bold" if bold else "Helvetica",
    )
    return Block(
        text=text,
        bbox=(x0, y0, x1, y1),
        spans=[span],
        page_no=page_no,
        column=column,
    )


@pytest.fixture
def block_factory():
    return make_block


# --------------------------------------------------------------- papers
@pytest.fixture
def sample_papers() -> list[dict]:
    """Three papers that form a small citation chain B -> A, C -> A."""
    return [
        {"paper_id": "pA", "title": "Attention Is All You Need", "year": 2017,
         "arxiv_id": "1706.03762", "doi": "10.5555/3295222"},
        {"paper_id": "pB", "title": "BERT: Pre-training of Deep Bidirectional Transformers",
         "year": 2019, "arxiv_id": "1810.04805", "doi": ""},
        {"paper_id": "pC", "title": "Language Models are Few-Shot Learners", "year": 2020,
         "arxiv_id": "2005.14165", "doi": ""},
    ]


@pytest.fixture
def sample_references() -> dict[str, list[dict]]:
    """Bibliographies that reference the papers above by title."""
    return {
        "pB": [
            {"index_in_doc": 1, "title": "Attention Is All You Need",
             "authors": ["Ashish Vaswani"], "year": 2017, "venue": "NeurIPS",
             "arxiv_id": "1706.03762", "doi": "", "context": "",
             "relevance": 1.0},
            {"index_in_doc": 2, "title": "A Structured Language Model",
             "authors": ["Some One"], "year": 2015, "venue": "ACL",
             "arxiv_id": "", "doi": "", "context": "", "relevance": 0.9},
        ],
        "pC": [
            {"index_in_doc": 1, "title": "Attention Is All You Need",
             "authors": ["Ashish Vaswani"], "year": 2017, "venue": "NeurIPS",
             "arxiv_id": "1706.03762", "doi": "", "context": "",
             "relevance": 1.0},
        ],
    }


# ------------------------------------------------------------ real PDFs
def _find_pdf(name: str) -> Path | None:
    """Look for a real test PDF in the usual local locations."""
    candidates = [
        Path(os.environ.get("SCIAGENT_TEST_PDF_DIR", "")) / name,
        Path.home() / "ra_testdata" / name,
        Path(__file__).resolve().parents[2] / "科研小助手" / "pdf" / name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


@pytest.fixture(scope="session")
def attention_pdf() -> Path | None:
    """The Attention Is All You Need paper, when it is available locally."""
    return _find_pdf("1706.03762v1.pdf")


@pytest.fixture(scope="session")
def scanned_pdf() -> Path | None:
    """The image-only (scanned) variant of the same paper."""
    return _find_pdf("1706.03762v1_img.pdf")
