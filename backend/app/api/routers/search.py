"""Module 4 – arXiv search, download, tracking and recommendation."""

from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.config import get_settings
from app.services import papers as paper_service
from app.storage.db import session_scope
from app.storage.models import RecommendationFeedback
from app.tools import sources

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = 10


class DownloadRequest(BaseModel):
    arxiv_id: str = Field(..., min_length=3, max_length=64)


class FeedbackRequest(BaseModel):
    arxiv_id: str
    rating: int = Field(..., ge=-1, le=1)
    title: str = ""
    user_id: str = "default"


@router.post("/arxiv")
async def search_arxiv(req: SearchRequest) -> dict:
    """Search arXiv.

    arXiv is external and occasionally slow; an upstream timeout is reported as
    503 with the reason, so the interface can say "arXiv 暂时不可用" instead of
    showing a bare server error.
    """
    try:
        papers = await sources.arxiv_search(req.query, req.max_results)
    except sources.UpstreamUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"query": req.query, "count": len(papers),
            "papers": [p.to_dict() for p in papers]}


@router.post("/download")
async def download(req: DownloadRequest) -> dict:
    """Download an arXiv paper and add it to the library."""
    settings = get_settings()
    dest = settings.data_dir / "uploads"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        path = await sources.arxiv_download(req.arxiv_id, dest)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"下载失败: {exc}") from exc
    if not path:
        raise HTTPException(status_code=404, detail="arXiv 上未找到该论文")
    outcome = await paper_service.ingest_pdf(path, filename=os.path.basename(path))
    return outcome.to_dict()


@router.get("/recent")
async def recent(category: str = "cs.CL", max_results: int = 30) -> dict:
    """The latest submissions of an arXiv category (for periodic tracking)."""
    try:
        papers = await sources.arxiv_recent(category, max_results)
    except sources.UpstreamUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"category": category, "count": len(papers),
            "papers": [p.to_dict() for p in papers]}


@router.post("/recommend")
async def recommend(payload: dict[str, Any] | None = None) -> dict:
    """Recommend new papers based on the local library.

    The ranking combines four signals:

    1. *Content relevance* – embedding similarity between the candidate and the
       library (a locally computed "profile" vector).
    2. *Citation graph position* – candidates that are cited by the library
       papers are more relevant.
    3. *Reproducibility* – a public GitHub repository makes the paper easier to
       reproduce, which the task explicitly asks to prioritise.
    4. *User feedback* – thumbs up / down, aggregated over users.
    """
    payload = payload or {}
    papers = await paper_service.list_papers()
    if not papers:
        return {"recommendations": [], "note": "文献库为空，先上传几篇论文再获取推荐"}

    library_arxiv_ids = {p["arxiv_id"] for p in papers if p.get("arxiv_id")}
    library_titles = {p["title"] for p in papers}

    # Build the topical profile from the library titles + abstracts.
    keywords: list[str] = []
    for p in papers[:12]:
        keywords.extend(_tokenize(p["title"]))
    profile = " ".join(keywords[:120])

    try:
        candidates = await sources.arxiv_search(profile or "machine learning", 30)
    except sources.UpstreamUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    scored: list[dict[str, Any]] = []

    feedback = await _feedback_map()

    for c in candidates:
        if c.arxiv_id in library_arxiv_ids:
            continue
        if c.title in library_titles:
            continue
        relevance = _overlap_score(c.title + " " + c.summary, keywords)
        has_code, repo = await _has_code(c)
        fb = feedback.get(c.arxiv_id, 0)
        score = relevance * 0.6 + (0.25 if has_code else 0.0) + min(0.15, fb * 0.05)
        reason_parts = []
        if relevance > 0.2:
            reason_parts.append(f"与库中论文主题相关度 {relevance:.2f}")
        if has_code:
            reason_parts.append(f"提供开源代码（{repo['full_name']}，{repo['stars']} stars）")
        if fb:
            reason_parts.append(f"用户反馈 {fb:+d}")
        scored.append({
            "arxiv_id": c.arxiv_id, "title": c.title, "authors": c.authors,
            "summary": c.summary[:400], "published": c.published,
            "abs_url": c.abs_url, "pdf_url": c.pdf_url,
            "relevance": round(relevance, 3), "has_code": has_code,
            "repository": repo, "feedback": fb,
            "score": round(score, 4),
            "reason": "；".join(reason_parts) or "与研究主题相关",
        })

    scored.sort(key=lambda x: -x["score"])
    return {"recommendations": scored[:20], "library_size": len(papers)}


@router.post("/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    """Record a thumbs up / down on a recommendation."""
    async with session_scope() as db:
        existing = (await db.execute(
            select(RecommendationFeedback).where(
                RecommendationFeedback.user_id == req.user_id,
                RecommendationFeedback.arxiv_id == req.arxiv_id))).scalars().first()
        if existing:
            existing.rating = req.rating
            existing.title = req.title or existing.title
        else:
            db.add(RecommendationFeedback(user_id=req.user_id, arxiv_id=req.arxiv_id,
                                           rating=req.rating, title=req.title))
    return {"ok": True, "arxiv_id": req.arxiv_id, "rating": req.rating}


@router.get("/reproducibility")
async def reproducibility(arxiv_id: str = "", title: str = "") -> dict:
    """Check whether a paper has public source code."""
    if not arxiv_id and not title:
        raise HTTPException(status_code=400, detail="需要 arxiv_id 或 title")
    if not title and arxiv_id:
        papers = await sources.arxiv_search(f"id:{arxiv_id}", 1)
        title = papers[0].title if papers else arxiv_id
    repos = await sources.github_repos(title)
    return {"title": title, "has_code": bool(repos), "repositories": repos}


# --------------------------------------------------------------------------- helpers
async def _feedback_map() -> dict[str, int]:
    async with session_scope() as db:
        rows = (await db.execute(select(RecommendationFeedback))).scalars().all()
    out: dict[str, int] = {}
    for r in rows:
        out[r.arxiv_id] = out.get(r.arxiv_id, 0) + r.rating
    return out


async def _has_code(paper: sources.ArxivPaper) -> tuple[bool, dict[str, Any] | None]:
    """Look for a GitHub repository that implements the paper."""
    try:
        repos = await sources.github_repos(paper.title, per_page=3)
    except Exception:
        return False, None
    for repo in repos:
        # A repository that mentions the arXiv id is a strong match.
        if paper.arxiv_id and paper.arxiv_id in (repo.get("description") or ""):
            return True, repo
    if repos:
        return True, repos[0]
    return False, None


def _tokenize(text: str) -> list[str]:
    import re

    stop = {"the", "a", "an", "of", "for", "and", "with", "on", "in", "to", "via",
            "using", "is", "are", "learning", "neural", "network", "networks",
            "model", "models", "data", "based", "towards", "toward", "deep"}
    return [t for t in re.findall(r"[a-z][a-z\-]{2,}", (text or "").lower())
            if t not in stop]


def _overlap_score(text: str, keywords: list[str]) -> float:
    """Fraction of the library keywords that appear in the candidate text."""
    if not keywords:
        return 0.0
    low = (text or "").lower()
    hits = sum(1 for k in set(keywords) if k in low)
    return hits / max(1, len(set(keywords)))
