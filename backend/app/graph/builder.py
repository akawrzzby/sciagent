"""Citation graph construction and graph-theoretic mining.

The graph is built from the *parsed bibliographies* of the library papers: each
reference is aligned with a paper in the library (by arXiv id, DOI, or a fuzzy
title match) and becomes a directed edge ``citing -> cited``.

Once the graph exists we mine the "foundation" papers with classical graph
algorithms:

* **PageRank** – global importance (a paper cited by many important papers).
* **Betweenness centrality** – brokerage: papers that other papers rely on to
  connect sub-fields.
* **k-core decomposition** – the densely connected backbone of the field.
* **HITS** – separates hubs (many outgoing citations) from authorities.

A weighted *foundation score* combines the normalised metrics and drives the
"foundation paper vs. derivative paper" classification required by module 5.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import networkx as nx

from app.core.logging import get_logger

logger = get_logger(__name__)

# Weights of the foundation score.
W_PAGERANK = 0.35
W_BETWEENNESS = 0.25
W_KCORE = 0.20
W_INDEGREE = 0.20


def _normalize(values: dict[str, float]) -> dict[str, float]:
    """Scale a metric dictionary to [0, 1] (max normalisation)."""
    if not values:
        return {}
    hi = max(values.values())
    lo = min(values.values())
    if hi <= 0:
        return {k: 0.0 for k in values}
    if hi == lo:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


@dataclass
class GraphNode:
    """A paper in the citation graph."""

    paper_id: str
    title: str = ""
    year: int | None = None
    arxiv_id: str = ""
    in_degree: int = 0
    out_degree: int = 0
    pagerank: float = 0.0
    betweenness: float = 0.0
    k_core: int = 0
    hub_score: float = 0.0
    authority_score: float = 0.0
    foundation_score: float = 0.0
    role: str = "derivative"     # foundation | core | derivative
    cited_by: list[str] = field(default_factory=list)
    cites: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id, "title": self.title, "year": self.year,
            "arxiv_id": self.arxiv_id, "in_degree": self.in_degree,
            "out_degree": self.out_degree, "pagerank": round(self.pagerank, 6),
            "betweenness": round(self.betweenness, 6), "k_core": self.k_core,
            "hub_score": round(self.hub_score, 6),
            "authority_score": round(self.authority_score, 6),
            "foundation_score": round(self.foundation_score, 6), "role": self.role,
            "cited_by": self.cited_by, "cites": self.cites,
            "metadata": self.metadata,
        }


@dataclass
class CitationGraph:
    """The mined citation graph."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": self.edges,
            "stats": self.stats,
        }

    def foundation_papers(self) -> list[GraphNode]:
        return sorted([n for n in self.nodes if n.role in ("foundation", "core")],
                      key=lambda n: -n.foundation_score)

    def derivative_papers(self) -> list[GraphNode]:
        return sorted([n for n in self.nodes if n.role == "derivative"],
                      key=lambda n: -n.foundation_score)


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _title_key(title: str) -> str:
    """Reduce a title to its significant tokens for fuzzy matching."""
    stop = {"the", "a", "an", "of", "for", "and", "with", "on", "in", "to", "via",
            "using", "is", "are", "learning", "neural", "networks", "network"}
    toks = [t for t in _norm_title(title).split() if t not in stop]
    return " ".join(toks[:8])


def align_reference(
    reference: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Try to match a bibliography entry with a paper of the library.

    Matching order: arXiv id → DOI → fuzzy title similarity.
    """
    arxiv = (reference.get("arxiv_id") or "").strip().lower()
    doi = (reference.get("doi") or "").strip().lower()
    title = reference.get("title") or ""

    if arxiv:
        for cand in candidates:
            if (cand.get("arxiv_id") or "").strip().lower() == arxiv:
                return cand
    if doi:
        for cand in candidates:
            if (cand.get("doi") or "").strip().lower() == doi:
                return cand

    if not title:
        return None

    key = _title_key(title)
    if not key:
        return None
    key_tokens = set(key.split())

    best, best_score = None, 0.0
    for cand in candidates:
        ckey = _title_key(cand.get("title", ""))
        if not ckey:
            continue
        ctokens = set(ckey.split())
        if not ctokens:
            continue
        score = len(key_tokens & ctokens) / max(1, len(key_tokens | ctokens))
        # Reward containment (one title contained in the other).
        if key in ckey or ckey in key:
            score = max(score, 0.85)
        if score > best_score:
            best, best_score = cand, score

    if best is not None and best_score >= 0.62:
        return best
    return None


def build_graph(
    papers: Sequence[dict[str, Any]],
    references_by_paper: dict[str, list[dict[str, Any]]],
    *,
    relevance_threshold: float = 0.25,
) -> CitationGraph:
    """Build the citation graph and mine the foundation nodes.

    ``papers`` is a list of ``{"paper_id", "title", "year", "arxiv_id", "doi"}``.
    ``references_by_paper`` maps a paper id to its parsed bibliography entries.
    """
    graph = CitationGraph()
    by_id = {p["paper_id"]: p for p in papers}

    # --- edges ------------------------------------------------------------
    edges: list[dict[str, Any]] = []
    for src_id, refs in references_by_paper.items():
        if src_id not in by_id:
            continue
        for ref in refs:
            target = align_reference(ref, papers)
            if target is None:
                continue
            dst_id = target["paper_id"]
            if dst_id == src_id:
                continue
            relevance = float(ref.get("relevance", 1.0))
            if relevance < relevance_threshold:
                continue
            edges.append({
                "source": src_id,
                "target": dst_id,
                "relevance": round(relevance, 3),
                "context": (ref.get("context") or "")[:400],
                "reference_index": ref.get("index_in_doc", 0),
            })

    # De-duplicate (keep the strongest relevance per pair).
    best_edge: dict[tuple[str, str], dict[str, Any]] = {}
    for e in edges:
        key = (e["source"], e["target"])
        if key not in best_edge or e["relevance"] > best_edge[key]["relevance"]:
            best_edge[key] = e
    edges = list(best_edge.values())
    graph.edges = edges

    # --- networkx ---------------------------------------------------------
    G = nx.DiGraph()
    for p in papers:
        G.add_node(p["paper_id"], title=p.get("title", ""),
                   year=p.get("year"), arxiv_id=p.get("arxiv_id", ""))
    for e in edges:
        G.add_edge(e["source"], e["target"], relevance=e["relevance"])

    # --- metrics ----------------------------------------------------------
    n_nodes = G.number_of_nodes()
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6, weight=None) if n_nodes else {}
    betweenness = nx.betweenness_centrality(G, normalized=True) if n_nodes else {}

    try:
        core_numbers = nx.core_number(G) if n_nodes else {}
    except Exception:
        core_numbers = {}

    try:
        hubs, authorities = nx.hits(G, max_iter=100) if n_nodes else ({}, {})
    except Exception:
        hubs, authorities = ({}, {})

    n_pr = _normalize(pagerank)
    n_bc = _normalize(betweenness)
    n_core = _normalize({k: float(v) for k, v in core_numbers.items()})
    n_in = _normalize({n: float(d) for n, d in G.in_degree()})

    nodes: list[GraphNode] = []
    for pid in G.nodes():
        meta = by_id.get(pid, {})
        score = (W_PAGERANK * n_pr.get(pid, 0.0)
                 + W_BETWEENNESS * n_bc.get(pid, 0.0)
                 + W_KCORE * n_core.get(pid, 0.0)
                 + W_INDEGREE * n_in.get(pid, 0.0))
        nodes.append(GraphNode(
            paper_id=pid,
            title=meta.get("title", G.nodes[pid].get("title", "")),
            year=meta.get("year"),
            arxiv_id=meta.get("arxiv_id", ""),
            in_degree=G.in_degree(pid),
            out_degree=G.out_degree(pid),
            pagerank=pagerank.get(pid, 0.0),
            betweenness=betweenness.get(pid, 0.0),
            k_core=int(core_numbers.get(pid, 0)),
            hub_score=hubs.get(pid, 0.0),
            authority_score=authorities.get(pid, 0.0),
            foundation_score=score,
            cited_by=[u for u, v in G.in_edges(pid)],
            cites=[v for u, v in G.out_edges(pid)],
            metadata={k: meta.get(k) for k in ("authors", "venue", "abstract") if meta.get(k)},
        ))

    # --- classification ----------------------------------------------------
    # A paper is a *foundation* when it scores high AND is actually referenced
    # inside the graph; otherwise it is a *core* member or a *derivative* leaf.
    for node in nodes:
        if node.foundation_score >= 0.55 and node.in_degree > 0:
            node.role = "foundation"
        elif node.foundation_score >= 0.35:
            node.role = "core"
        else:
            node.role = "derivative"

    graph.nodes = nodes
    graph.stats = {
        "node_count": n_nodes,
        "edge_count": G.number_of_edges(),
        "density": round(nx.density(G), 5) if n_nodes else 0.0,
        "is_weakly_connected": nx.is_weakly_connected(G) if n_nodes else False,
        "foundation_count": sum(1 for n in nodes if n.role == "foundation"),
        "core_count": sum(1 for n in nodes if n.role == "core"),
        "derivative_count": sum(1 for n in nodes if n.role == "derivative"),
        "largest_component": (len(max(nx.weakly_connected_components(G), key=len))
                               if n_nodes else 0),
        "algorithms": ["PageRank", "Betweenness Centrality", "k-core", "HITS"],
    }
    return graph


def compute_citation_relevance(context: str, target_title: str) -> float:
    """Heuristic "citation gold content" score.

    A citation is *substantive* when the sentence that contains it shares
    vocabulary with the title of the cited work, or when the citation is
    attached to a claim (numbers, comparisons, method statements) rather than to
    a bare mention of prior art.
    """
    if not context:
        return 0.5
    ctx_tokens = set(re.findall(r"[a-z]{4,}", context.lower()))
    title_tokens = set(re.findall(r"[a-z]{4,}", (target_title or "").lower()))

    # A bare "see [12]" mention is short even when it shares no vocabulary with
    # the cited title, so the short-context penalty has to be applied before
    # the early return below - otherwise every bare mention would end up with
    # the neutral 0.5 and be classified as substantive.
    short = len(context.strip()) < 40
    if not ctx_tokens or not title_tokens:
        return 0.2 if short else 0.5

    overlap = len(ctx_tokens & title_tokens) / len(title_tokens)
    score = min(1.0, overlap * 2.0 + 0.2)

    # Boost citations attached to quantitative or comparative statements.
    if re.search(r"\d", context):
        score = min(1.0, score + 0.15)
    if re.search(r"\b(outperform|compare|state[- ]of[- ]the[- ]art|sota|better|"
                 r"improvement|unlike|whereas|contrast)\b", context, re.IGNORECASE):
        score = min(1.0, score + 0.2)
    # Demote bare "see [12]" style mentions.
    if short:
        score = max(0.05, score - 0.25)
    return round(score, 3)


def filter_substantive_references(
    references: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.45,
) -> dict[str, Any]:
    """Split a bibliography into substantive and perfunctory citations."""
    kept, dropped = [], []
    for ref in references:
        score = float(ref.get("relevance", compute_citation_relevance(
            ref.get("context", ""), ref.get("title", ""))))
        if score >= threshold:
            kept.append({**ref, "relevance": score})
        else:
            dropped.append({**ref, "relevance": score})
    kept.sort(key=lambda r: -r["relevance"])
    return {"kept": kept, "dropped": dropped, "threshold": threshold}
