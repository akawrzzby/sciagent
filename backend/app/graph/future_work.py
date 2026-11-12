"""Future-work mining: turning "we leave X for future work" into research leads.

The pipeline has four stages:

1. **Extraction** – pull the future-work / limitations / discussion paragraphs
   out of the parsed documents.
2. **Semantic de-duplication** – the same idea is often proposed by several
   papers in different words; we cluster the statements with the dense encoder
   and merge them.
3. **Recency check** – search arXiv for each surviving idea.  If a paper from
   the last months already addresses it, the idea is filtered out (this is the
   "do not propose ideas that have already been tried" requirement).
4. **Synthesis** – the LLM writes a structured opportunity list with the
   evidence quote and an explicit novelty argument.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.core.logging import get_logger
from app.llm import prompts
from app.llm.client import LLMClient, get_llm_client
from app.rag.embeddings import get_embedding_service
from app.tools import sources

logger = get_logger(__name__)

# Sentences that explicitly announce future work.
_FUTURE_MARKERS = (
    "future work", "future research", "in the future", "we plan to", "we will explore",
    "remains open", "remains an open", "leave .* for future", "we did not explore",
    "limitation", "does not address", "is beyond the scope", "further investigation",
    "interesting future", "promising direction", "尚未", "未来工作", "未来研究",
    "有待", "尚未解决",
)


@dataclass
class FutureStatement:
    """A single future-work statement extracted from a paper."""

    text: str
    paper_id: str
    paper_title: str = ""
    section_title: str = ""
    page_no: int = 0
    year: int | None = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "paper_id": self.paper_id, "paper_title": self.paper_title,
            "section_title": self.section_title, "page_no": self.page_no,
            "year": self.year, "score": round(self.score, 4),
        }


@dataclass
class Opportunity:
    """A research direction proposed by the agent."""

    idea: str
    evidence: str = ""
    source_papers: list[str] = field(default_factory=list)
    novelty_argument: str = ""
    difficulty: str = "medium"
    impact: str = "medium"
    solved_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea": self.idea, "evidence": self.evidence, "source_papers": self.source_papers,
            "novelty_argument": self.novelty_argument, "difficulty": self.difficulty,
            "impact": self.impact, "solved_by": self.solved_by,
        }


# --------------------------------------------------------------------------- extraction
_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+(?=[A-Z一-鿿])")


def extract_future_statements(
    chunks: Sequence[dict[str, Any]],
    papers: dict[str, dict[str, Any]] | None = None,
    *,
    max_per_paper: int = 6,
) -> list[FutureStatement]:
    """Find the future-work sentences inside the chunk store."""
    papers = papers or {}
    statements: list[FutureStatement] = []

    for chunk in chunks:
        kind = chunk.get("kind", "body")
        section = (chunk.get("section_title") or "").lower()
        is_fw_section = kind == "future_work" or any(
            k in section for k in ("future work", "limitation", "conclusion", "discussion",
                                   "outlook", "open problem"))
        if not is_fw_section:
            continue

        text = chunk.get("content", "")
        sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
        hits: list[str] = []
        for sent in sentences:
            low = sent.lower()
            if any(re.search(m, low) for m in _FUTURE_MARKERS):
                hits.append(sent)
            elif len(hits) < 2 and len(sent) > 40 and kind == "future_work":
                # In an explicit future-work section most sentences are relevant.
                hits.append(sent)
        if not hits:
            continue

        paper = papers.get(chunk.get("paper_id", ""), {})
        for sent in hits[:max_per_paper]:
            statements.append(FutureStatement(
                text=sent,
                paper_id=chunk.get("paper_id", ""),
                paper_title=paper.get("title", ""),
                section_title=chunk.get("section_title", "") or "",
                page_no=chunk.get("page_no", 0),
                year=paper.get("year"),
            ))
    return statements


# --------------------------------------------------------------------------- dedup
async def deduplicate_statements(
    statements: Sequence[FutureStatement],
    *,
    threshold: float = 0.86,
) -> list[FutureStatement]:
    """Merge semantically equivalent future-work statements.

    Statements that are near-duplicates keep the highest scoring representative
    and remember the other papers that proposed the same idea.
    """
    if not statements:
        return []

    service = get_embedding_service()
    vectors = await asyncio.to_thread(service.encode, [s.text for s in statements])
    if vectors is None or len(vectors) != len(statements):
        # Fall back to a lexical similarity.
        return _lexical_dedup(statements, threshold=0.9)

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    keep: list[int] = []
    groups: list[list[int]] = []
    for i in range(len(statements)):
        merged = False
        for gi, group in enumerate(groups):
            rep = keep[gi]
            sim = float(np.dot(matrix[i], matrix[rep]))
            if sim >= threshold:
                group.append(i)
                merged = True
                break
        if not merged:
            keep.append(i)
            groups.append([i])

    out: list[FutureStatement] = []
    for rep, group in zip(keep, groups):
        primary = statements[rep]
        # Record the co-proposing papers inside the text so the LLM can cite them.
        others = [statements[i].paper_title for i in group if i != rep]
        if others:
            primary.text = primary.text.rstrip()
        out.append(primary)
    return out


def _lexical_dedup(statements: Sequence[FutureStatement], threshold: float = 0.9) -> list[FutureStatement]:
    out: list[FutureStatement] = []
    for s in statements:
        toks = set(re.findall(r"[a-z]{4,}", s.text.lower()))
        if not toks:
            continue
        duplicate = False
        for existing in out:
            other = set(re.findall(r"[a-z]{4,}", existing.text.lower()))
            if not other:
                continue
            if len(toks & other) / len(toks | other) >= threshold:
                duplicate = True
                break
        if not duplicate:
            out.append(s)
    return out


# --------------------------------------------------------------------------- recency
async def check_recent_works(
    statements: Sequence[FutureStatement],
    *,
    max_checks: int = 6,
    per_query: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Search arXiv for recent work that may already address each idea."""
    results: dict[str, list[dict[str, Any]]] = {}
    service = get_embedding_service()

    for s in list(statements)[:max_checks]:
        keywords = _keywords_from_statement(s.text)
        if not keywords:
            continue
        query = " ".join(keywords)
        try:
            papers = await sources.arxiv_search(query, per_query)
        except Exception as exc:
            logger.warning("arXiv recency check failed: %s", exc)
            results[s.text] = []
            continue
        results[s.text] = [p.to_dict() for p in papers]
    return results


def _keywords_from_statement(text: str, limit: int = 8) -> list[str]:
    """Extract the technical keywords of a future-work statement."""
    stop = {"future", "work", "remain", "remains", "paper", "work", "study", "studies",
            "research", "direction", "directions", "approach", "approaches", "method",
            "methods", "we", "our", "this", "that", "these", "those", "with", "from",
            "into", "also", "more", "than", "such", "which", "their", "there", "have",
            "been", "will", "would", "could", "should", "can", "may", "not", "but",
            "using", "used", "use", "based", "via", "for", "and", "the", "are", "was"}
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
    counts: dict[str, int] = {}
    for t in tokens:
        tl = t.lower()
        if tl in stop or len(tl) < 3:
            continue
        counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[:limit]]


# --------------------------------------------------------------------------- synthesis
async def synthesize_opportunities(
    statements: Sequence[FutureStatement],
    recent: dict[str, list[dict[str, Any]]],
    *,
    llm: LLMClient | None = None,
    session_id: str = "",
) -> list[Opportunity]:
    """Ask the LLM to write the structured opportunity list."""
    if not statements:
        return []
    llm = llm or get_llm_client()

    stmt_block = "\n\n".join(
        f"- (paper: {s.paper_title or s.paper_id}) {s.text}" for s in statements
    )
    recent_block = "\n".join(
        f"- for statement '{st[:120]}…': " +
        ", ".join(f"{p['title']} ({p.get('published', '')[:10]})"
                  for p in papers[:5])
        for st, papers in recent.items()
    ) or "(no recent lookup performed)"

    payload, raw = await llm.complete_json(
        prompts.SYSTEM_FUTURE_WORK,
        f"FUTURE WORK STATEMENTS\n{stmt_block}\n\nRECENT PAPERS (2024-2026)\n{recent_block}",
        schema_hint=(
            '{"opportunities": [{"idea": "string", "source_papers": ["string"], '
            '"evidence": "string", "novelty_argument": "string", '
            '"difficulty": "low|medium|high", "impact": "low|medium|high"}]}'
        ),
        session_id=session_id,
        purpose="future_work",
    )
    if not payload:
        return [Opportunity(idea=s.text, evidence=s.text,
                            source_papers=[s.paper_title] if s.paper_title else [])
                for s in statements]

    out: list[Opportunity] = []
    for item in payload.get("opportunities", []) or []:
        if not isinstance(item, dict):
            continue
        out.append(Opportunity(
            idea=str(item.get("idea", "")).strip(),
            evidence=str(item.get("evidence", "")).strip(),
            source_papers=[str(x) for x in (item.get("source_papers") or [])],
            novelty_argument=str(item.get("novelty_argument", "")).strip(),
            difficulty=str(item.get("difficulty", "medium")).lower(),
            impact=str(item.get("impact", "medium")).lower(),
        ))
    out.sort(key=lambda o: ({"high": 0, "medium": 1, "low": 2}.get(o.impact, 1),
                             {"high": 0, "medium": 1, "low": 2}.get(o.difficulty, 1)))
    return out


async def mine_future_work(
    chunks: Sequence[dict[str, Any]],
    papers: dict[str, dict[str, Any]],
    *,
    session_id: str = "",
    check_recent: bool = True,
) -> dict[str, Any]:
    """Full future-work mining pipeline."""
    statements = extract_future_statements(chunks, papers)
    if not statements:
        return {"statements": [], "opportunities": [], "recent": {},
                "note": "库中没有找到 Future Work / Limitations 相关段落"}

    deduped = await deduplicate_statements(statements)
    recent = await check_recent_works(deduped) if check_recent else {}
    opportunities = await synthesize_opportunities(deduped, recent, session_id=session_id)

    return {
        "statements": [s.to_dict() for s in deduped],
        "raw_statement_count": len(statements),
        "opportunities": [o.to_dict() for o in opportunities],
        "recent": recent,
    }
