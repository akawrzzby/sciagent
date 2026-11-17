"""External data-source tools.

Three integrations are needed by the competition:

* **arXiv** – search, download and track new submissions (module 4).
* **OpenAlex** – bibliographic metadata and citation counts, used to judge the
  "citation gold content" of a reference and to resolve references to arXiv ids.
* **GitHub** – detect whether a paper has public source code, which is the
  "reproducibility" signal used by the recommender.

All of them degrade gracefully: when the network is unavailable the tool
returns an explanatory error instead of raising, so the agent can continue.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_ABS = "https://arxiv.org/abs/"
OPENALEX_API = "https://api.openalex.org/works"
GITHUB_SEARCH = "https://api.github.com/search/repositories"

_USER_AGENT = "SciAgent/0.1 (research assistant; +https://github.com/akawrzzby)"
_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass
class ArxivPaper:
    """A normalised arXiv record."""

    arxiv_id: str = ""
    version: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    summary: str = ""
    published: str = ""
    updated: str = ""
    categories: list[str] = field(default_factory=list)
    pdf_url: str = ""
    abs_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id, "version": self.version, "title": self.title,
            "authors": self.authors, "summary": self.summary[:1200],
            "published": self.published, "updated": self.updated,
            "categories": self.categories, "pdf_url": self.pdf_url, "abs_url": self.abs_url,
        }


class UpstreamUnavailable(RuntimeError):
    """arXiv or GitHub did not answer.

    A search box cannot do anything with a socket timeout, and letting it
    surface as a 500 tells the operator nothing.  This carries the upstream
    host so the route can answer with a sentence instead.
    """


async def _get(client: httpx.AsyncClient, url: str, params: dict | None = None) -> httpx.Response:
    host = url.split("/")[2] if "//" in url else url
    try:
        r = await client.get(url, params=params, headers={"User-Agent": _USER_AGENT})
        r.raise_for_status()
    except httpx.TimeoutException as exc:
        raise UpstreamUnavailable(f"{host} 响应超时") from exc
    except httpx.HTTPStatusError as exc:
        raise UpstreamUnavailable(f"{host} 返回 {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise UpstreamUnavailable(f"无法连接 {host}") from exc
    return r


async def arxiv_search(query: str, max_results: int = 10, *, timeout: int = 25) -> list[ArxivPaper]:
    """Search arXiv via its Atom API."""
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max(1, min(max_results, 50)),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await _get(client, ARXIV_API, params)

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        logger.warning("arXiv parse error: %s", exc)
        return []

    out: list[ArxivPaper] = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        m = re.search(r"abs/(.+)$", raw_id)
        full_id = m.group(1) if m else raw_id.rsplit("/", 1)[-1]
        base_id = re.sub(r"v\d+$", "", full_id)
        version = re.search(r"v(\d+)$", full_id)
        pdf_url = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
        out.append(ArxivPaper(
            arxiv_id=base_id,
            version=f"v{version.group(1)}" if version else "",
            title=_squash(entry.findtext(f"{_ATOM}title") or ""),
            authors=[_squash(a.findtext(f"{_ATOM}name") or "")
                     for a in entry.findall(f"{_ATOM}author")],
            summary=_squash(entry.findtext(f"{_ATOM}summary") or ""),
            published=entry.findtext(f"{_ATOM}published") or "",
            updated=entry.findtext(f"{_ATOM}updated") or "",
            categories=[c.get("term", "") for c in entry.findall(f"{_ATOM}category")],
            pdf_url=pdf_url,
            abs_url=f"{ARXIV_ABS}{full_id}",
        ))
    return out


async def arxiv_download(arxiv_id: str, dest_dir: str, *, timeout: int = 120) -> str | None:
    """Download the PDF of an arXiv paper.  Returns the file path."""
    base = re.sub(r"v\d+$", "", (arxiv_id or "").strip())
    if not base:
        return None
    url = f"https://arxiv.org/pdf/{base}"
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await _get(client, url)
    path = f"{dest_dir}/{base}.pdf"
    with open(path, "wb") as fh:
        fh.write(resp.content)
    return path


async def openalex_lookup(arxiv_id: str | None = None, title: str | None = None,
                          *, timeout: int = 20) -> dict[str, Any]:
    """Look up bibliographic metadata (citation count, venue, DOI) via OpenAlex."""
    params: dict[str, Any] = {"per-page": 1}
    if arxiv_id:
        params["filter"] = f"locations.source.id:S4306400194"  # placeholder, refined below
        params = {"filter": f"ids.openalex:doi:10.48550/arXiv.{arxiv_id}", "per-page": 1}
    elif title:
        params["filter"] = f"title.search:{title}"
    else:
        return {}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await _get(client, OPENALEX_API, params)
    data = resp.json()
    results = data.get("results") or []
    if not results:
        return {}
    w = results[0]
    return {
        "openalex_id": w.get("id", ""),
        "title": w.get("title", ""),
        "year": w.get("publication_year"),
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "cited_by_count": w.get("cited_by_count", 0),
        "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
        "type": w.get("type", ""),
        "authors": [a.get("author", {}).get("display_name", "")
                    for a in (w.get("authorships") or [])][:15],
    }


async def github_repos(query: str, *, timeout: int = 20, per_page: int = 5) -> list[dict[str, Any]]:
    """Search GitHub for repositories related to a paper."""
    q = quote_plus(query)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await _get(client, f"{GITHUB_SEARCH}?q={q}&sort=stars&order=desc&per_page={per_page}")
    data = resp.json()
    out: list[dict[str, Any]] = []
    for item in data.get("items", []):
        out.append({
            "full_name": item.get("full_name", ""),
            "html_url": item.get("html_url", ""),
            "description": item.get("description") or "",
            "stars": item.get("stargazers_count", 0),
            "forks": item.get("forks_count", 0),
            "language": item.get("language") or "",
            "pushed_at": item.get("pushed_at", ""),
            "license": ((item.get("license") or {}) or {}).get("spdx_id", ""),
        })
    return out


async def arxiv_recent(category: str = "cs.CL", max_results: int = 50,
                       *, timeout: int = 25) -> list[ArxivPaper]:
    """Fetch the most recent submissions of an arXiv category (for tracking)."""
    return await arxiv_search(f"cat:{category}", max_results=max_results, timeout=timeout)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
