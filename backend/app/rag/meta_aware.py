"""Metadata aware retrieval.

A chunk on its own is a wall of text: the sentence that answers "what BLEU
score did the model reach" looks the same no matter which paper it came from.
The paper it belongs to -- its title, its authors, its year -- is exactly the
part that tells two papers of the same field apart, and it is the part a
reader would use to narrow the search.

PaperQA2 (arXiv:2409.13740) calls this *metadata aware retrieval*: the
metadata of a document is carried along with its text instead of being left
behind in a database column.  This module implements the same idea twice:

* **at index time** the title, the year and the first authors are prefixed to
  the text that is embedded, so that a dense query such as "what did Vaswani
  et al. report in 2017" can match the chunk on its metadata and not only on
  its wording;
* **at query time** the same metadata is used to compute an *affinity* for
  every paper of the library, and the affinity raises the score of that
  paper's passages.

Both steps are transparent: the affinity is reported per passage, so the UI
can answer "why was this retrieved?".
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

# Words that carry no discriminating power inside a title.
_TITLE_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "with", "on", "in", "to",
    "via", "using", "is", "are", "was", "were", "be", "been", "from", "by",
    "at", "as", "that", "this", "it", "its", "into", "towards", "toward",
    "learning", "model", "models", "net", "networks", "network", "deep",
    "neural", "based", "using", "towards", "approach", "method", "methods",
}

_TOKEN = re.compile(r"[a-z0-9]+")

# The number of authors that go into the embedded prefix.  A paper with forty
# authors would otherwise bury its own content under a list of names.
MAX_PREFIX_AUTHORS = 6


def normalize(text: str) -> str:
    """Lower-case text with the punctuation removed."""
    return " " + " ".join(_TOKEN.findall((text or "").lower())) + " "


def content_words(text: str) -> list[str]:
    """The words of a text that can discriminate between documents."""
    return [w for w in _TOKEN.findall((text or "").lower())
            if len(w) > 2 and w not in _TITLE_STOPWORDS]


def author_names(authors: Any) -> list[str]:
    """The author names, no matter how the database stored them."""
    if not authors:
        return []
    if isinstance(authors, str):
        try:
            import json

            authors = json.loads(authors)
        except Exception:  # noqa: BLE001 - a plain string list is fine too
            return [a.strip() for a in authors.split(",") if a.strip()]
    out: list[str] = []
    for a in authors if isinstance(authors, (list, tuple)) else []:
        if isinstance(a, str):
            name = a
        elif isinstance(a, dict):
            name = str(a.get("name") or "")
        else:
            name = str(a)
        name = name.strip()
        if name:
            out.append(name)
    return out


def surname(name: str) -> str:
    """The family name of an author, ignoring the name particles."""
    particles = {"van", "von", "de", "del", "della", "di", "da", "dos", "du",
                 "la", "le", "el", "al", "bin", "ibn"}
    tokens = [t for t in re.split(r"\s+", name.strip()) if t]
    if not tokens:
        return ""
    family = tokens[-1]
    if family.lower().strip(".,") in particles and len(tokens) > 1:
        family = tokens[-2]
    return family.strip(".,")


def metadata_header(title: str, authors: Any, year: Any = None,
                    venue: str = "") -> str:
    """The one line prefix that is prepended to a chunk before embedding."""
    names = author_names(authors)[:MAX_PREFIX_AUTHORS]
    if names:
        surnames = [surname(n) for n in names]
        surnames = [s for s in surnames if s]
        if len(surnames) > 2:
            who = f"{surnames[0]} et al."
        else:
            who = ", ".join(surnames) or ", ".join(names)
    else:
        who = ""
    bits = [b for b in (str(title or "").strip(), who, str(year or "").strip()) if b]
    if not bits:
        return ""
    return " | ".join(bits)


def embed_text(content: str, title: str = "", authors: Any = None,
               year: Any = None) -> str:
    """The text that is embedded for one chunk of a paper."""
    header = metadata_header(title, authors, year)
    return f"{header}\n{content}" if header else content


def embedding_text(chunk: dict[str, Any]) -> str:
    """The text that is actually embedded for a chunk.

    The stored content is never modified -- the citation still quotes the
    original text.  Only the vector sees the metadata.
    """
    return embed_text(
        chunk.get("content", "") or "",
        chunk.get("paper_title", "") or chunk.get("title", ""),
        chunk.get("paper_authors") or chunk.get("authors"),
        chunk.get("paper_year") or chunk.get("year"),
    )


def title_affinity(query: str, title: str) -> float:
    """How much of the paper's title the query mentions (0..1)."""
    words = content_words(title)
    if not words:
        return 0.0
    q = normalize(query)
    hits = sum(1 for w in words if f" {w} " in q)
    return hits / len(words)


def author_affinity(query: str, authors: Any) -> float:
    """How many of the paper's authors the query names (0..1)."""
    names = author_names(authors)
    if not names:
        return 0.0
    q = normalize(query)
    hits = 0
    for name in names:
        fam = surname(name).lower()
        if fam and f" {fam} " in q:
            hits += 1
    if not hits:
        return 0.0
    # One named author is already a strong signal; three are a certainty.
    return min(1.0, hits / 3.0)


def year_affinity(query: str, year: Any) -> float:
    """Whether the query mentions the paper's year."""
    if not year:
        return 0.0
    return 1.0 if str(year) in (query or "") else 0.0


def paper_affinity(query: str, title: str, authors: Any = None,
                   year: Any = None) -> float:
    """The combined metadata affinity of a paper for a query (0..1).

    The weights mirror how a researcher would narrow a search: the title
    first, the authors next, the year last.
    """
    t = title_affinity(query, title)
    a = author_affinity(query, authors)
    y = year_affinity(query, year)
    score = 0.55 * t + 0.30 * a + 0.15 * y
    return max(0.0, min(1.0, score))


def explain_affinity(query: str, title: str, authors: Any = None,
                     year: Any = None) -> dict[str, Any]:
    """The affinity and its parts, for the UI and the diagnostics."""
    t = title_affinity(query, title)
    a = author_affinity(query, authors)
    y = year_affinity(query, year)
    return {
        "affinity": round(max(0.0, min(1.0, 0.55 * t + 0.30 * a + 0.15 * y)), 4),
        "title": round(t, 4),
        "authors": round(a, 4),
        "year": round(y, 4),
    }


def matched_authors(query: str, authors: Any) -> list[str]:
    """The authors of a paper that the query names."""
    q = normalize(query)
    out = []
    for name in author_names(authors):
        fam = surname(name)
        if fam and f" {fam.lower()} " in q:
            out.append(name)
    return out


def metadata_terms(title: str, authors: Any = None,
                   year: Any = None) -> list[str]:
    """The searchable metadata terms of a paper."""
    terms = content_words(title)
    for name in author_names(authors):
        fam = surname(name)
        if fam:
            terms.append(fam.lower())
    if year:
        terms.append(str(year))
    return terms


def paper_metadata(paper: dict[str, Any]) -> dict[str, Any]:
    """Normalise the metadata of a paper dict into the retrieval format."""
    return {
        "paper_title": paper.get("title", "") or "",
        "paper_authors": paper.get("authors") or [],
        "paper_year": paper.get("year") or None,
    }


def iter_metadata(chunks: Iterable[dict[str, Any]]) -> Sequence[dict[str, Any]]:
    """Return the chunks that carry usable metadata."""
    return [c for c in chunks if c.get("paper_title")]
