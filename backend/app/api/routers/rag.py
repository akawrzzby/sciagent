"""Module 2 – anti-hallucination RAG endpoints (agent chat, retrieval, tracing)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.agents.orchestrator import AgentContext, AgentOrchestrator
from app.core.events import get_bus, session_channel
from app.services import agent as agent_service
from app.services import papers as paper_service
from app.storage.db import session_scope
from app.storage.models import ChatMessage

router = APIRouter(prefix="/rag", tags=["rag"])


class AskRequest(BaseModel):
    """A question for the agent."""

    question: str = Field(..., min_length=1, max_length=8000)
    session_id: str = ""
    paper_ids: list[str] = Field(default_factory=list)
    stream: bool = True


class SearchRequest(BaseModel):
    """A direct (non-agentic) retrieval request."""

    query: str = Field(..., min_length=1)
    paper_ids: list[str] = Field(default_factory=list)
    top_k: int = 8


@router.post("/ask")
async def ask(req: AskRequest):
    """Ask the agent a question; returns a Server-Sent-Events stream."""
    session_id = req.session_id or uuid.uuid4().hex
    registry = agent_service.build_agent_registry(req.paper_ids or None)
    context = AgentContext(question=req.question, session_id=session_id,
                           paper_ids=req.paper_ids, stream=req.stream)
    orchestrator = AgentOrchestrator(context, registry=registry)

    if not req.stream:
        # Non streaming mode: collect the events and return the final answer.
        final: dict[str, Any] = {}
        async for event in orchestrator.run():
            if event.get("type") == "final":
                final = event
        await _persist_turn(session_id, req.question, final.get("answer", ""),
                            final.get("citations", []))
        return {"session_id": session_id, **final}

    async def event_stream():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        final: dict[str, Any] = {}
        try:
            async for event in orchestrator.run():
                if event.get("type") == "final":
                    final = event
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("agent stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)[:400]})}\n\n"
        await _persist_turn(session_id, req.question, final.get("answer", ""),
                            final.get("citations", []))
        yield "data: [DONE]\n\n"

    from app.core.logging import get_logger

    logger = get_logger(__name__)
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@router.post("/search")
async def search(req: SearchRequest) -> dict:
    """Direct hybrid retrieval (useful for debugging and for the UI preview)."""
    chunks = await agent_service.load_chunks(req.paper_ids or None)
    if not chunks:
        return {"passages": [], "note": "文献库为空"}
    retriever = agent_service.get_retriever()
    result = await retriever.search(req.query, chunks, top_k=req.top_k)
    return result.to_dict()


@router.get("/sessions/{session_id}/messages")
async def session_messages(session_id: str) -> dict:
    """The stored transcript of a session."""
    async with session_scope() as db:
        rows = (await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at))).scalars().all()
        return {"messages": [
            {"role": m.role, "content": m.content, "citations": m.citations or [],
             "created_at": m.created_at.isoformat() if m.created_at else ""}
            for m in rows
        ]}


@router.get("/sessions/{session_id}/events")
async def session_events(session_id: str):
    """Replay the event stream of a session (used to re-attach a client)."""
    bus = get_bus()

    async def generator():
        async for event in bus.subscribe(session_channel(session_id), replay=50):
            yield f"data: {json.dumps({'type': event.type, **event.data}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@router.get("/chunks/{paper_id}")
async def paper_chunks(paper_id: str, limit: int = 100, offset: int = 0) -> dict:
    """Inspect the chunks of a paper (transparency / debugging)."""
    chunks = await agent_service.load_chunks([paper_id])
    return {"paper_id": paper_id, "total": len(chunks),
            "chunks": chunks[offset : offset + limit]}


async def _persist_turn(session_id: str, question: str, answer: str,
                        citations: list[dict[str, Any]]) -> None:
    """Store a conversation turn."""
    try:
        async with session_scope() as db:
            db.add(ChatMessage(session_id=session_id, role="user", content=question))
            db.add(ChatMessage(session_id=session_id, role="assistant", content=answer,
                               citations=citations))
    except Exception as exc:  # pragma: no cover
        logger.debug("failed to persist chat turn: %s", exc)
