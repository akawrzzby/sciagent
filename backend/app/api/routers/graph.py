"""Module 5 – citation graph, foundation mining, review and future work."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.graph import builder as graph_builder
from app.graph import contradiction as contradiction_mod
from app.graph import future_work as future_work_mod
from app.services import agent as agent_service

router = APIRouter(prefix="/graph", tags=["graph"])


class GraphRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    # citation | co_citation | co_reference | all
    view: str = graph_builder.EDGE_KIND_CITATION


class ReviewRequest(BaseModel):
    paper_ids: list[str] = Field(default_factory=list)
    language: str = "zh"


@router.post("")
async def build_graph(req: GraphRequest) -> dict:
    """Build the citation graph and mine the foundation nodes.

    ``view`` selects which edges are returned:

    * ``citation``    -- the real citations (default);
    * ``co_citation`` -- papers that are cited by the same paper;
    * ``co_reference`` -- papers that cite the same reference;
    * ``all``         -- every edge, with its kind.
    """
    papers = await agent_service.load_papers(req.paper_ids or None)
    if len(papers) < 2:
        return {"nodes": [], "edges": [], "stats": {},
                "note": "引用图谱至少需要两篇论文，请先上传更多文献。"}
    scope = [p["paper_id"] for p in papers]
    refs = await agent_service.load_references(scope)
    refs_by_paper: dict[str, list[dict[str, Any]]] = {}
    for r in refs:
        refs_by_paper.setdefault(r["paper_id"], []).append(r)
    graph = graph_builder.build_graph(papers, refs_by_paper)

    view = (req.view or graph_builder.EDGE_KIND_CITATION).strip().lower()
    similarity = graph_builder.build_similarity_edges(papers, refs_by_paper,
                                                      graph.edges)
    graph.edges = graph_builder.filter_edges_by_view(
        list(graph.edges) + similarity, view)
    graph.stats["view"] = view
    graph.stats["similarity_edges"] = len(similarity)
    graph.stats["edge_kinds"] = {
        kind: sum(1 for e in graph.edges if e.get("kind") == kind)
        for kind in (graph_builder.EDGE_KIND_CITATION,
                     graph_builder.EDGE_KIND_CO_CITATION,
                     graph_builder.EDGE_KIND_CO_REFERENCE)
    }
    return graph.to_dict()


@router.get("/citations/quality")
async def citation_quality(paper_id: str) -> dict:
    """Split a bibliography into substantive and perfunctory citations."""
    refs = await agent_service.load_references([paper_id])
    result = graph_builder.filter_substantive_references(refs)
    stance_counts = {graph_builder.CITATION_STANCE_SUPPORTING: 0,
                     graph_builder.CITATION_STANCE_CONTRASTING: 0,
                     graph_builder.CITATION_STANCE_MENTIONING: 0}
    for ref in result["kept"] + result["dropped"]:
        stance = ref.get("stance")
        if stance in stance_counts:
            stance_counts[stance] += 1
    return {
        "paper_id": paper_id,
        "kept": result["kept"][:60],
        "dropped": result["dropped"][:60],
        "kept_count": len(result["kept"]),
        "dropped_count": len(result["dropped"]),
        "threshold": result["threshold"],
        "stance_counts": stance_counts,
    }


@router.post("/contradictions")
async def contradictions(req: GraphRequest) -> dict:
    """Detect contradictory claims between the papers of the library.

    Two passages that report the same metric in the same context with
    different values contradict each other; so do two passages that draw
    opposite conclusions about the same subject.  Both sides are quoted.
    """
    chunks = await agent_service.load_chunks(req.paper_ids or None)
    if not chunks:
        return {"conflicts": [], "count": 0, "note": "文献库为空"}
    result = contradiction_mod.detect_contradictions(chunks)
    result["markdown"] = contradiction_mod.conflicts_to_markdown(result)
    return result


@router.post("/future-work")
async def future_work(req: GraphRequest) -> dict:
    """Mine unexplored research directions from the future-work sections."""
    chunks = await agent_service.load_chunks(req.paper_ids or None)
    if not chunks:
        return {"statements": [], "opportunities": [], "note": "文献库为空"}
    papers = {p["paper_id"]: p for p in await agent_service.load_papers(req.paper_ids or None)}
    return await future_work_mod.mine_future_work(chunks, papers)


@router.post("/review")
async def review(req: ReviewRequest) -> dict:
    """Generate an automatic field review from the citation graph."""
    from app.llm import prompts
    from app.llm.client import get_llm_client

    papers = await agent_service.load_papers(req.paper_ids or None)
    if not papers:
        raise HTTPException(status_code=400, detail="文献库为空")

    scope = [p["paper_id"] for p in papers]
    refs = await agent_service.load_references(scope)
    refs_by_paper: dict[str, list[dict[str, Any]]] = {}
    for r in refs:
        refs_by_paper.setdefault(r["paper_id"], []).append(r)
    graph = graph_builder.build_graph(papers, refs_by_paper)

    foundation = graph.foundation_papers()[:8]
    evidence = []
    for node in foundation:
        paper = next((p for p in papers if p["paper_id"] == node.paper_id), {})
        evidence.append(
            f"[{node.paper_id}] {node.title} ({node.year or 'n/a'}) — "
            f"PageRank={node.pagerank:.5f}, Betweenness={node.betweenness:.5f}, "
            f"k-core={node.k_core}, in-degree={node.in_degree}\n"
            f"abstract: {(paper.get('abstract') or '')[:500]}"
        )

    metrics = "\n".join(
        f"- {n.title}: role={n.role} pagerank={n.pagerank:.5f} "
        f"betweenness={n.betweenness:.5f} kcore={n.k_core} in={n.in_degree} out={n.out_degree}"
        for n in graph.nodes
    )

    llm = get_llm_client()
    payload, raw = await llm.complete_json(
        prompts.SYSTEM_REVIEW,
        f"GRAPH METRICS\n{metrics}\n\nKEY PAPER EVIDENCE\n" + "\n\n".join(evidence),
        schema_hint=(
            '{"title": "string", "summary": "string", "core_findings": ["string"], '
            '"foundation_papers": [{"title": "string", "why_foundational": "string"}], '
            '"derivative_papers": [{"title": "string", "contribution": "string"}], '
            '"open_questions": ["string"]}'
        ),
        purpose="review",
    )
    if not payload:
        return {"error": "模型未返回合法 JSON", "raw": raw[:600], "graph": graph.to_dict()}

    payload["graph"] = graph.to_dict()
    return payload
