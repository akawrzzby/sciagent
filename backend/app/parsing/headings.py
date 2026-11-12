"""Heading tree extraction.

We combine three independent signals to decide whether a text block is a
heading and at which level:

1. **Numbering pattern** – ``3.2.2 Multi-Head Attention`` gives an explicit
   level (number of dots + 1).  This is the most reliable signal in
   LaTeX-generated papers.
2. **Font size / weight** – a heading is usually rendered larger and/or bold
   relative to the body text.
3. **Lexical patterns** – canonical section names (``Abstract``,
   ``Introduction``, ``Related Work``, ``Conclusion``, ``References``,
   ``Acknowledgments``, ``Appendix``) as well as appendix numbering
   (``A``, ``A.1``).

The result is a nested tree (parents/children) that powers the reader's
navigation sidebar and the "jump to 3.2.2" requirement of the task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.parsing.layout import Block

# --------------------------------------------------------------------- patterns
# "3.2.2  Multi-Head Attention" / "3.2.2. Multi-Head"
_NUMBERED = re.compile(r"^\s*((?:\d{1,2})(?:\.\d{1,2}){0,4})\.?\s+(\S.*)$")
# "A.1  Preliminaries" / "A.1. Something"
_APPENDIX_NUMBERED = re.compile(r"^\s*([A-Z](?:\.\d{1,2}){0,4})\.?\s+(\S.*)$")
# "Appendix A  Proofs"
_APPENDIX_WORD = re.compile(r"^\s*Appendix\s+([A-Z])\b[\s.:-]*(.*)$", re.IGNORECASE)

# Canonical top level section names (lowercase comparison).
_CANONICAL_L1 = {
    "abstract", "introduction", "background", "related work", "related works",
    "preliminaries", "preliminary", "method", "methods", "methodology",
    "approach", "model", "architecture", "problem statement", "problem definition",
    "experiments", "experimental setup", "experimental results", "results",
    "evaluation", "analysis", "ablation study", "ablation studies",
    "discussion", "conclusion", "conclusions", "conclusion and future work",
    "limitations", "acknowledgments", "acknowledgements", "acknowledgment",
    "references", "bibliography", "appendix", "appendices", "ethics statement",
    "reproducibility statement",
}

# Canonical level-2 names that frequently appear as "3 Method" or "3.1 Setup".
_CANONICAL_L2 = {
    "setup", "implementation details", "implementation", "datasets", "dataset",
    "baseline", "baselines", "metrics", "training", "fine-tuning", "finetuning",
    "data", "model details", "hyperparameters", "hyper-parameter", "hyper-parameters",
    "optimization", "architecture", "framework", "overview", "problem formulation",
    "notation", "task definition", "data statistics", "reproducibility",
}

# Lines that are clearly not headings even though they may look like one.
_NOT_HEADING = re.compile(
    r"^\s*(?:figure|fig\.?|table|tab\.?|algorithm|listing|equation|eq\.?)\s*\d+",
    re.IGNORECASE,
)


@dataclass(slots=True)
class HeadingCandidate:
    """A block that looks like a heading."""

    block: Block
    title: str
    number: str
    level: int
    score: float


@dataclass(slots=True)
class HeadingNode:
    """A node of the heading tree."""

    title: str
    number: str
    level: int
    page_no: int
    bbox: list[float]
    content: str = ""
    children: list["HeadingNode"] = field(default_factory=list)
    parent: "HeadingNode | None" = None
    order_index: int = 0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "number": self.number,
            "level": self.level,
            "page_no": self.page_no,
            "bbox": self.bbox,
            "children": [c.to_dict() for c in self.children],
        }


def _clean_title(text: str) -> str:
    """Normalise whitespace and strip trailing punctuation artefacts."""
    t = re.sub(r"\s+", " ", text).strip()
    return t.rstrip(" .:;")


def _looks_like_heading_text(block: Block) -> bool:
    """Reject captions, references and other non-heading lines."""
    text = block.text.strip()
    if not text or len(text) > 180:
        return False
    # The rotated arXiv stamp in the left margin is not a heading.
    if re.match(r"^\s*arXiv:\d{4}\.\d{4,5}", text, re.IGNORECASE):
        return False
    if _NOT_HEADING.match(text):
        return False
    # A heading rarely ends with a sentence period or contains many commas.
    if text.count(",") > 3:
        return False
    if text.endswith((".", ";")) and not re.match(r"^\s*(?:\d+\.?\d*|[A-Z])", text):
        return False
    return True


def detect_headings(
    blocks: list[Block],
    body_size: float,
    *,
    min_relative_size: float = 1.08,
    page_offset: int = 0,
) -> list[HeadingCandidate]:
    """Detect heading candidates in reading order.

    ``body_size`` is the dominant body font size of the document; a block whose
    font size exceeds ``body_size * min_relative_size`` (or that is bold and
    short) is a heading candidate.
    """
    candidates: list[HeadingCandidate] = []

    for block in blocks:
        text = block.text.strip()
        if not text or not _looks_like_heading_text(block):
            continue

        # A heading normally occupies a single line.
        if len(text.splitlines()) > 2:
            continue

        # 1) explicit numbering ------------------------------------------
        m = _NUMBERED.match(text)
        if m:
            number, title = m.group(1), _clean_title(m.group(2))
            level = number.count(".") + 1
            if level <= 4 and len(title) <= 140:
                candidates.append(HeadingCandidate(block, title, number, level, 1.0))
                continue

        m = _APPENDIX_WORD.match(text)
        if m:
            label = f"Appendix {m.group(1).upper()}"
            title = _clean_title(m.group(2)) or label
            candidates.append(HeadingCandidate(block, title, m.group(1).upper(), 1, 0.95))
            continue

        m = _APPENDIX_NUMBERED.match(text)
        if m and m.group(1)[0].isalpha() and m.group(1)[0].isupper():
            number, title = m.group(1), _clean_title(m.group(2))
            level = number.count(".") + 1
            if len(title) <= 140:
                candidates.append(HeadingCandidate(block, title, number, level, 0.9))
                continue

        # 2) canonical names ----------------------------------------------
        lower = text.lower().rstrip(":")
        size = block.font_size
        bigger = size >= body_size * min_relative_size
        bold = block.is_bold

        if lower in _CANONICAL_L1 and (bigger or bold or len(text) < 60):
            candidates.append(HeadingCandidate(block, _clean_title(text), "", 1, 0.9))
            continue
        if lower in _CANONICAL_L2 and (bigger or bold):
            candidates.append(HeadingCandidate(block, _clean_title(text), "", 2, 0.75))
            continue

        # 3) typography only ----------------------------------------------
        if bigger and bold and len(text) <= 120:
            candidates.append(HeadingCandidate(block, _clean_title(text), "", 1, 0.6))
        elif bigger and len(text) <= 100 and not text.endswith("."):
            candidates.append(HeadingCandidate(block, _clean_title(text), "", 2, 0.5))

    # Remove duplicated headings that appear on consecutive pages (running headers).
    return _dedupe_running_headers(candidates)


def _dedupe_running_headers(candidates: list[HeadingCandidate]) -> list[HeadingCandidate]:
    """Drop repeated short headings (journal running headers)."""
    seen: dict[str, int] = {}
    result: list[HeadingCandidate] = []
    for cand in candidates:
        key = f"{cand.level}:{cand.title.lower()}"
        count = seen.get(key, 0)
        seen[key] = count + 1
        # The same heading may legitimately repeat in a book chapter, but not
        # more than twice within the front matter.
        if count >= 2 and len(cand.title) < 40:
            continue
        result.append(cand)
    return result


def build_tree(candidates: list[HeadingCandidate]) -> list[HeadingNode]:
    """Build a nested heading tree from a flat, ordered candidate list."""
    roots: list[HeadingNode] = []
    stack: list[HeadingNode] = []

    for idx, cand in enumerate(candidates):
        node = HeadingNode(
            title=cand.title,
            number=cand.number,
            level=cand.level,
            page_no=cand.block.page_no,
            bbox=list(cand.block.bbox),
            order_index=idx,
        )
        # Pop until the top of the stack has a smaller level.
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    return roots


def flatten(tree: list[HeadingNode]) -> list[HeadingNode]:
    """Depth-first flattening of the heading tree."""
    out: list[HeadingNode] = []

    def walk(nodes: list[HeadingNode]) -> None:
        for n in nodes:
            out.append(n)
            walk(n.children)

    walk(tree)
    return out


def assign_content(nodes: list[HeadingNode], body_blocks: list[Block]) -> None:
    """Attach the body text that belongs to each heading node."""
    flat = flatten(nodes)
    if not flat:
        return

    # Index body blocks by page for a fast lookup.
    by_page: dict[int, list[Block]] = {}
    for b in body_blocks:
        by_page.setdefault(b.page_no, []).append(b)

    for i, node in enumerate(flat):
        end_page = flat[i + 1].page_no if i + 1 < len(flat) else node.page_no
        parts: list[str] = []
        for page_no in range(node.page_no, end_page + 1):
            for b in by_page.get(page_no, []):
                # Only blocks that appear *after* the heading on the same page.
                if page_no == node.page_no and b.y0 <= node.bbox[1]:
                    continue
                parts.append(b.text)
        node.content = "\n".join(parts).strip()
