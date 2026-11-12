"""The tool registry used by the agent's function-calling loop.

Every tool is an ``async`` callable with a JSON schema.  The executor hands the
schemas to the model, the model asks for a call, the registry executes it and
returns a JSON-serialisable result which is appended to the conversation.

Design decisions
----------------
* Tool errors are **caught and returned as data** (``{"error": "..."}``) so the
  model can recover instead of the whole turn crashing.
* Tools that touch the database receive a session factory; tools that only
  need the internet do not.
* Heavy tools (graph mining, review generation) are delegated to sub-agents
  which own their own prompts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.prompts import TOOL_SCHEMAS
from app.tools import sources

logger = get_logger(__name__)

ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class ToolSpec:
    """A callable tool plus its OpenAI-style schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolFn

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Holds the executable tools for one agent session."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------ register
    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def add(self, name: str, description: str, parameters: dict[str, Any],
            handler: ToolFn) -> None:
        self.register(ToolSpec(name=name, description=description,
                               parameters=parameters, handler=handler))

    # ------------------------------------------------------------------ query
    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def schemas(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        """Return the JSON schemas handed to the model."""
        if only:
            return [self._tools[n].schema() for n in only if n in self._tools]
        return [self._tools[n].schema() for n in sorted(self._tools)]

    def static_schemas(self) -> list[dict[str, Any]]:
        """The canonical schema list from :mod:`app.llm.prompts`."""
        return TOOL_SCHEMAS

    # ------------------------------------------------------------------ execute
    async def execute(self, name: str, arguments: dict[str, Any],
                      *, timeout: float = 180.0) -> dict[str, Any]:
        """Run a tool and normalise the outcome."""
        spec = self._tools.get(name)
        if spec is None:
            return {"error": f"unknown tool '{name}'", "available": self.names}
        try:
            result = await asyncio_wait(spec.handler, arguments, timeout)
        except TimeoutError:
            return {"error": f"tool '{name}' timed out after {timeout:.0f}s"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("tool %s failed", name)
            return {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(result, (dict, list)):
            return result  # type: ignore[return-value]
        return {"result": result}


async def asyncio_wait(handler: ToolFn, arguments: dict[str, Any], timeout: float) -> Any:
    """Run a coroutine with a timeout (thin wrapper so it can be mocked)."""
    import asyncio

    return await asyncio.wait_for(handler(**arguments), timeout=timeout)


# --------------------------------------------------------------------------- builders
def build_registry(
    *,
    search_library,
    read_passages,
    compare_papers,
    build_graph,
    mine_future_work,
    generate_framework,
    generate_diagram,
    plot_csv,
    translate,
    library_state,
) -> ToolRegistry:
    """Create the standard registry.

    The heavy callbacks are injected by the orchestrator so that the registry
    stays free of database / agent dependencies (and easy to unit test).
    """
    reg = ToolRegistry()
    settings = get_settings()

    reg.add(
        "search_library",
        "Semantic + keyword search over the papers in the user's library.",
        {"type": "object",
         "properties": {
             "query": {"type": "string"},
             "paper_ids": {"type": "array", "items": {"type": "string"}},
             "top_k": {"type": "integer", "default": 8},
         },
         "required": ["query"]},
        search_library,
    )
    reg.add(
        "read_passages",
        "Read the full text of specific retrieved passages by chunk id.",
        {"type": "object",
         "properties": {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
         "required": ["chunk_ids"]},
        read_passages,
    )
    reg.add(
        "compare_papers",
        "Compare several library papers on a specific aspect.",
        {"type": "object",
         "properties": {
             "paper_ids": {"type": "array", "items": {"type": "string"}},
             "aspect": {"type": "string"},
         },
         "required": ["paper_ids"]},
        compare_papers,
    )
    reg.add(
        "build_graph",
        "Build the citation graph and mine the foundation nodes with centrality algorithms.",
        {"type": "object",
         "properties": {"paper_ids": {"type": "array", "items": {"type": "string"}}}},
        build_graph,
    )
    reg.add(
        "mine_future_work",
        "Extract future-work statements and propose unexplored research directions.",
        {"type": "object",
         "properties": {"paper_ids": {"type": "array", "items": {"type": "string"}}},
         "required": ["paper_ids"]},
        mine_future_work,
    )
    reg.add(
        "search_arxiv",
        "Search arXiv for papers matching a query.",
        {"type": "object",
         "properties": {
             "query": {"type": "string"},
             "max_results": {"type": "integer", "default": 10},
         },
         "required": ["query"]},
        _make_arxiv_search(),
    )
    reg.add(
        "download_paper",
        "Download an arXiv paper by id and add it to the library.",
        {"type": "object",
         "properties": {"arxiv_id": {"type": "string"}},
         "required": ["arxiv_id"]},
        _make_download(library_state),
    )
    reg.add(
        "check_reproducibility",
        "Check whether a paper has public source code on GitHub.",
        {"type": "object",
         "properties": {"arxiv_id": {"type": "string"}, "title": {"type": "string"}}},
        _make_reproducibility(),
    )
    reg.add(
        "generate_framework",
        "Draft an academic paper framework grounded in the real references of the library.",
        {"type": "object",
         "properties": {"idea": {"type": "string"}},
         "required": ["idea"]},
        generate_framework,
    )
    reg.add(
        "generate_diagram",
        "Generate a TikZ / Mermaid / Graphviz architecture diagram.",
        {"type": "object",
         "properties": {
             "description": {"type": "string"},
             "diagram_type": {"type": "string", "enum": ["tikz", "mermaid", "graphviz"]},
         },
         "required": ["description"]},
        generate_diagram,
    )
    reg.add(
        "plot_csv",
        "Turn CSV data into an academic chart plus a figure caption.",
        {"type": "object",
         "properties": {
             "csv_content": {"type": "string"},
             "chart_type": {"type": "string", "enum": ["line", "bar", "table", "radar3d"]},
             "title": {"type": "string"},
             "x_column": {"type": "string"},
             "y_columns": {"type": "array", "items": {"type": "string"}},
         },
         "required": ["csv_content"]},
        plot_csv,
    )
    reg.add(
        "translate",
        "Translate academic text between English and Chinese.",
        {"type": "object",
         "properties": {
             "text": {"type": "string"},
             "target_lang": {"type": "string", "enum": ["zh", "en"]},
         },
         "required": ["text", "target_lang"]},
        translate,
    )
    return reg


# --------------------------------------------------------------------------- default tools
def _make_arxiv_search() -> ToolFn:
    async def search_arxiv(query: str, max_results: int = 10) -> dict[str, Any]:
        papers = await sources.arxiv_search(query, max_results)
        return {"count": len(papers), "papers": [p.to_dict() for p in papers]}

    return search_arxiv


def _make_download(library_state) -> ToolFn:
    async def download_paper(arxiv_id: str) -> dict[str, Any]:
        settings = get_settings()
        dest = os.path.join(str(settings.data_dir), "uploads")
        os.makedirs(dest, exist_ok=True)
        path = await sources.arxiv_download(arxiv_id, dest)
        if not path:
            return {"error": f"could not download arXiv:{arxiv_id}"}
        paper = await library_state("download", path)
        return {"ok": True, "path": path, "paper": paper}

    return download_paper


def _make_reproducibility() -> ToolFn:
    async def check_reproducibility(arxiv_id: str = "", title: str = "") -> dict[str, Any]:
        query = title.strip()
        if not query and arxiv_id:
            papers = await sources.arxiv_search(f"id:{arxiv_id}", 1)
            query = papers[0].title if papers else arxiv_id
        if not query:
            return {"error": "arxiv_id or title is required"}
        repos = await sources.github_repos(query)
        return {
            "query": query,
            "has_code": bool(repos),
            "repositories": repos,
            "best": repos[0] if repos else None,
        }

    return check_reproducibility
