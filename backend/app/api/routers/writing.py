"""Module 6 – academic writing assistant: paper framework, diagrams, artifacts."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.llm import prompts
from app.llm.client import get_llm_client
from app.services import agent as agent_service

router = APIRouter(prefix="/writing", tags=["writing"])


class FrameworkRequest(BaseModel):
    idea: str = Field(..., min_length=5, max_length=4000)


class DiagramRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=8000)
    diagram_type: str = "mermaid"


@router.post("/framework")
async def framework(req: FrameworkRequest) -> dict:
    """Generate a paper framework whose references are guaranteed to be real."""
    tool = agent_service.build_agent_registry().get("generate_framework")
    if tool is None:
        raise HTTPException(status_code=500, detail="tool not registered")
    result = await tool.handler(idea=req.idea)
    if "error" in result:
        raise HTTPException(status_code=502, detail=result["error"])
    return result


@router.post("/diagram")
async def diagram(req: DiagramRequest) -> dict:
    """Generate a TikZ / Mermaid / Graphviz architecture diagram."""
    llm = get_llm_client()
    payload, raw = await llm.complete_json(
        prompts.SYSTEM_DIAGRAM,
        f"DESCRIPTION\n{req.description}\n\nPreferred diagram type: {req.diagram_type}",
        schema_hint='{"diagram_type": "string", "title": "string", "code": "string", '
                    '"description": "string"}',
        purpose="diagram",
    )
    if not payload:
        raise HTTPException(status_code=502, detail="模型未返回合法 JSON")
    return payload


@router.get("/references")
async def available_references(limit: int = 100) -> dict:
    """The real references that the writing assistant may cite."""
    papers = await agent_service.load_papers()
    if not papers:
        return {"references": []}
    refs = await agent_service.load_references([p["paper_id"] for p in papers])
    refs.sort(key=lambda r: -r.get("relevance", 0))
    return {
        "references": [
            {"title": r["title"], "authors": r["authors"][:6], "year": r["year"],
             "venue": r["venue"], "arxiv_id": r["arxiv_id"], "doi": r["doi"],
             "relevance": r.get("relevance", 0), "paper_id": r["paper_id"]}
            for r in refs[:limit]
        ],
        "total": len(refs),
    }
