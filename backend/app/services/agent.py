"""Agent service: wires the tool registry to the real subsystems.

This module is the glue between the :class:`~app.agents.orchestrator.AgentOrchestrator`
and the concrete implementations (retrieval, graph mining, writing assistant,
visualisation).  Keeping it separate means the orchestrator stays testable and
the tools stay swappable.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Sequence

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph import builder as graph_builder
from app.graph import future_work as future_work_mod
from app.llm import prompts
from app.llm.client import get_llm_client
from app.rag.retriever import HybridRetriever, Passage, embed_chunks
from app.storage.db import session_scope
from app.storage.models import Chunk, Paper, PaperReference
from app.tools.registry import ToolRegistry, build_registry

logger = get_logger(__name__)

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Return the process-wide retriever."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# --------------------------------------------------------------------------- data access
async def load_chunks(paper_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Load the chunk pool (optionally restricted to a subset of papers)."""
    async with session_scope() as db:
        stmt = (select(Chunk, Paper.title, Paper.authors, Paper.year)
                .join(Paper, Chunk.paper_id == Paper.id))
        if paper_ids:
            stmt = stmt.where(Chunk.paper_id.in_(list(paper_ids)))
        rows = (await db.execute(stmt.order_by(Chunk.paper_id, Chunk.ordinal))).all()
        return [
            {
                "id": chunk.id, "paper_id": chunk.paper_id, "paper_title": title,
                "paper_authors": authors or [], "paper_year": year,
                "ordinal": chunk.ordinal, "content": chunk.content,
                "page_no": chunk.page_no, "section_title": chunk.section_title or "",
                "char_start": chunk.char_start, "char_end": chunk.char_end,
                "kind": chunk.kind, "embedding_row": chunk.embedding_row,
                "score": 0.0,
            }
            for chunk, title, authors, year in rows
        ]


async def load_papers(paper_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    async with session_scope() as db:
        stmt = select(Paper)
        if paper_ids:
            stmt = stmt.where(Paper.id.in_(list(paper_ids)))
        rows = (await db.execute(stmt)).scalars().all()
        return [
            {"paper_id": p.id, "title": p.title, "year": p.year, "arxiv_id": p.arxiv_id,
             "doi": p.doi, "authors": p.authors or [], "venue": p.venue,
             "abstract": p.abstract, "version_label": p.version_label or "",
             "is_scanned": bool(p.is_scanned)}
            for p in rows
        ]


async def load_references(paper_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Load the bibliography entries of the given papers with citation context."""
    from app.parsing.references import citation_context, find_intext_citations

    async with session_scope() as db:
        rows = (await db.execute(
            select(PaperReference).where(PaperReference.paper_id.in_(list(paper_ids)))
            .order_by(PaperReference.paper_id, PaperReference.index_in_doc))).scalars().all()
        refs = [
            {"paper_id": r.paper_id, "index_in_doc": r.index_in_doc, "title": r.title,
             "authors": r.authors or [], "year": r.year, "venue": r.venue,
             "doi": r.doi, "arxiv_id": r.arxiv_id, "raw_text": r.raw_text}
            for r in rows
        ]
        chunks = (await db.execute(
            select(Chunk).where(Chunk.paper_id.in_(list(paper_ids))))).scalars().all()

    # Attach the sentence in which each reference is cited -> citation relevance.
    by_paper: dict[str, str] = {}
    for c in chunks:
        by_paper[c.paper_id] = (by_paper.get(c.paper_id, "") + "\n" + c.content)

    for ref in refs:
        text = by_paper.get(ref["paper_id"], "")
        contexts: list[str] = []
        for offset, numbers in find_intext_citations(text):
            if ref["index_in_doc"] in numbers:
                contexts.append(citation_context(text, offset))
        ref["context"] = contexts[0] if contexts else ""
        ref["citation_count"] = len(contexts)
        ref["relevance"] = graph_builder.compute_citation_relevance(
            ref["context"], ref["title"]
        )
        stance = graph_builder.classify_citation_context(ref["context"], ref["title"])
        ref["stance"] = stance.stance
        ref["stance_label"] = graph_builder.CITATION_STANCE_LABEL.get(stance.stance, "")
        ref["stance_confidence"] = stance.confidence
        ref["stance_reason"] = stance.reason
    return refs


# --------------------------------------------------------------------------- tools
def _make_search_library(paper_ids_filter: list[str] | None):
    async def search_library(query: str, paper_ids: list[str] | None = None,
                            top_k: int = 8) -> dict[str, Any]:
        scope = paper_ids or paper_ids_filter
        chunks = await load_chunks(scope)
        if not chunks:
            return {"passages": [], "note": "文献库为空，请先上传论文"}
        retriever = get_retriever()
        result = await retriever.search(query, chunks, top_k=top_k)
        return {
            "passages": [p.to_dict() for p in result.passages],
            "strategy": result.strategy,
            "diagnostics": result.diagnostics,
        }

    return search_library


def _make_read_passages():
    async def read_passages(chunk_ids: list[str]) -> dict[str, Any]:
        if not chunk_ids:
            return {"passages": []}
        async with session_scope() as db:
            rows = (await db.execute(
                select(Chunk, Paper.title).join(Paper, Chunk.paper_id == Paper.id)
                .where(Chunk.id.in_(chunk_ids)))).all()
        return {"passages": [
            {"chunk_id": c.id, "paper_id": c.paper_id, "paper_title": title,
             "content": c.content, "page_no": c.page_no,
             "section_title": c.section_title or "", "char_start": c.char_start,
             "char_end": c.char_end, "kind": c.kind, "score": 1.0}
            for c, title in rows
        ]}

    return read_passages


def _make_compare_papers():
    async def compare_papers(paper_ids: list[str], aspect: str = "") -> dict[str, Any]:
        papers = await load_papers(paper_ids)
        if not papers:
            return {"papers": [], "note": "未找到指定论文"}
        chunks = await load_chunks(paper_ids)
        # One "profile" passage per paper: its abstract plus the best matching
        # passages for the requested aspect.
        by_paper: dict[str, list[dict[str, Any]]] = {}
        for c in chunks:
            by_paper.setdefault(c["paper_id"], []).append(c)

        retriever = get_retriever()
        aspect_query = aspect or "architecture method experiment results"
        selected: list[dict[str, Any]] = []
        for p in papers:
            pool = by_paper.get(p["paper_id"], [])
            if not pool:
                continue
            res = await retriever.search(aspect_query, pool, top_k=4)
            selected.extend([x.to_dict() for x in res.passages])

        return {"papers": papers, "aspect": aspect, "passages": selected,
                "passage_count": len(selected)}

    return compare_papers


def _make_build_graph():
    async def build_graph(paper_ids: list[str] | None = None) -> dict[str, Any]:
        papers = await load_papers(paper_ids)
        if len(papers) < 2:
            return {"nodes": [], "edges": [], "stats": {},
                    "note": "引用图谱至少需要两篇论文"}
        scope = [p["paper_id"] for p in papers]
        refs = await load_references(scope)
        refs_by_paper: dict[str, list[dict[str, Any]]] = {}
        for r in refs:
            refs_by_paper.setdefault(r["paper_id"], []).append(r)
        graph = graph_builder.build_graph(papers, refs_by_paper)
        return graph.to_dict()

    return build_graph


def _make_mine_future_work():
    async def mine_future_work(paper_ids: list[str]) -> dict[str, Any]:
        scope = paper_ids or None
        chunks = await load_chunks(scope)
        papers = {p["paper_id"]: p for p in await load_papers(scope)}
        result = await future_work_mod.mine_future_work(chunks, papers)
        return result

    return mine_future_work


def _make_generate_framework():
    async def generate_framework(idea: str) -> dict[str, Any]:
        papers = await load_papers()
        refs = await load_references([p["paper_id"] for p in papers]) if papers else []
        # Keep the most relevant references.
        refs = sorted(refs, key=lambda r: -r.get("relevance", 0))[:40]
        ref_block = "\n".join(
            f"- [{i + 1}] {r['title']} | {', '.join(r['authors'][:4])} | {r['year'] or 'n/a'} | {r['venue']}"
            for i, r in enumerate(refs)
        )
        llm = get_llm_client()
        payload, raw = await llm.complete_json(
            prompts.SYSTEM_FRAMEWORK,
            f"IDEA\n{idea}\n\nREFERENCE LIST (the ONLY references you may cite)\n{ref_block}",
            schema_hint=(
                '{"title": "string", "abstract": "string", '
                '"introduction": ["string"], '
                '"related_work": [{"topic": "string", "summary": "string", "citations": ["string"]}], '
                '"method": ["string"], "experiments": ["string"], '
                '"references": [{"key": "string", "title": "string", "authors": "string", '
                '"year": 2020, "venue": "string"}]}'
            ),
            purpose="framework",
            # A reasoning model spends thousands of tokens thinking before it
            # writes; with the old budget the JSON was cut off mid-object and
            # arrived as an unparseable fragment.
            thinking=False,
            max_tokens=6000,
        )
        if not payload:
            hint = ("模型输出被截断，JSON 不完整" if len(raw) > 2000
                    else "模型未返回合法 JSON")
            return {"error": hint, "raw": raw[:500]}

        # Guardrail: drop references that do not exist in the library.
        from app.llm.guardrails import validate_references

        check = validate_references(payload.get("references") or [], refs)
        payload["references"] = check.kept
        payload["rejected_references"] = check.rejected
        payload["reference_stats"] = check.to_dict()
        return payload

    return generate_framework


def _make_generate_diagram():
    async def generate_diagram(description: str,
                               diagram_type: str = "mermaid") -> dict[str, Any]:
        llm = get_llm_client()
        payload, raw = await llm.complete_json(
            prompts.SYSTEM_DIAGRAM,
            f"DESCRIPTION\n{description}\n\nPreferred diagram type: {diagram_type}",
            schema_hint='{"diagram_type": "string", "title": "string", "code": "string", '
                        '"description": "string"}',
            purpose="diagram",
            # the code field can be long, and reasoning would eat the budget
            thinking=False,
            max_tokens=4000,
        )
        if not payload:
            hint = ("模型输出被截断，JSON 不完整" if len(raw) > 1200
                    else "模型未返回合法 JSON")
            return {"error": hint, "raw": raw[:500]}
        return payload

    return generate_diagram


def _make_plot_csv():
    from app.tools.viz import plot_csv_data

    async def plot_csv(csv_content: str, chart_type: str = "line",
                       title: str = "", x_column: str = "",
                       y_columns: list[str] | None = None) -> dict[str, Any]:
        return await plot_csv_data(csv_content, chart_type=chart_type, title=title,
                                   x_column=x_column, y_columns=y_columns or [])

    return plot_csv


def _make_translate():
    async def translate(text: str, target_lang: str = "zh") -> dict[str, Any]:
        from app.services.translation import translate_text

        out = await translate_text(text, target_lang)
        return out

    return translate


def _make_library_state():
    """Handle the ``download_paper`` tool's side effect on the library."""

    async def library_state(action: str, path: str) -> dict[str, Any]:
        from app.services import papers as paper_service

        if action == "download":
            outcome = await paper_service.ingest_pdf(path, filename=os.path.basename(path))
            return outcome.to_dict()
        return {"status": "unknown"}

    return library_state


# --------------------------------------------------------------------------- factory
def build_agent_registry(paper_ids: Sequence[str] | None = None) -> ToolRegistry:
    """Create the tool registry for one agent session."""
    scope = list(paper_ids) if paper_ids else None
    return build_registry(
        search_library=_make_search_library(scope),
        read_passages=_make_read_passages(),
        compare_papers=_make_compare_papers(),
        build_graph=_make_build_graph(),
        mine_future_work=_make_mine_future_work(),
        generate_framework=_make_generate_framework(),
        generate_diagram=_make_generate_diagram(),
        plot_csv=_make_plot_csv(),
        translate=_make_translate(),
        library_state=_make_library_state(),
    )
