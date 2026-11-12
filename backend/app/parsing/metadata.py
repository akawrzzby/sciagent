"""Front-matter extraction: title, authors, affiliations, abstract, identifiers.

The heuristics are tuned for arXiv style preprints (the dataset provided by the
organisers) but degrade gracefully for other publishers.

The core idea is a **front-matter band**: on the first page the paper title is
followed by a band that contains the author blocks (name / affiliation /
e-mail) and then the abstract heading.  Locating that band lets us extract the
authors reliably – including the common "four authors side by side, each in
its own text block" layout of two-column preprints.

When a heuristic is unsure we fall back to the vision-language model, which
reads the rendered first page directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.parsing.layout import Block

# --------------------------------------------------------------------------- regexes
_ARXIV_ID = re.compile(r"\b(?:arXiv[:\s]*)?(\d{4}\.\d{4,5}(?:v\d+)?)\b")
_ARXIV_OLD_ID = re.compile(r"\b([a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\b")
_DOI = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
_ABSTRACT = re.compile(r"^\s*abstract\b[\s.:—-]*(.*)$", re.IGNORECASE | re.MULTILINE)
_KEYWORDS = re.compile(
    r"^\s*(?:keywords?|index terms)\b[\s.:—-]*(.*)$", re.IGNORECASE | re.MULTILINE)

# The rotated arXiv stamp in the left margin – never a title or a heading.
_ARXIV_STAMP = re.compile(r"^\s*arXiv:\d{4}\.\d{4,5}", re.IGNORECASE)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Words that indicate an institutional affiliation.
_AFFILIATION_MARKERS = (
    "university", "universit\u00e4t", "institute", "laborator", "college",
    "school of", "department", "dept", "research center", "research centre",
    "academy of sciences", "corporation", "inc.", "llc", "gmbh", "microsoft",
    "google", "openai", "meta ai", "deepmind", "nvidia", "adobe", "amazon",
    "technolog", "faculty of", "center for", "centre for", "hospital",
    "cn,", "china", "usa", "canada", "germany", "israel", "france", "japan",
    "korea", "singapore", "switzerland", "london", "cambridge", "toronto",
)

# Particles that may appear inside a personal name.
_NAME_PARTICLES = {"van", "von", "de", "del", "della", "di", "da", "dos", "du",
                   "la", "le", "el", "al", "bin", "ibn"}

# Footnote markers used by publishers (U+2217 is the LaTeX asterisk operator).
_MARKER_CHARS = "*\u2217\u2605\u2020\u2021\u00a7\u00b6\u2022\u00b7"

# Tokens that indicate an organisation rather than a person.
_ORG_WORDS = {
    "university", "universität", "institute", "laboratory", "lab", "labs",
    "college", "school", "faculty", "department", "dept", "academy",
    "corporation", "inc", "llc", "ltd", "gmbh", "plc", "ag", "bv",
    "center", "centre", "research", "technologies", "technology", "science",
    "sciences", "group", "team", "foundation", "ai", "researcher",
    "google", "brain", "microsoft", "meta", "deepmind", "nvidia", "openai",
    "amazon", "adobe", "facebook", "apple", "ibm", "baidu", "tencent",
    "alibaba", "huawei", "bytedance", "adobe", "samsung", "sony",
    "laboratories", "institut", "hochschule", "universite",
}

# Heading-like words that mark the end of the front-matter band.
_BAND_END_TITLES = ("abstract", "introduction", "1 introduction", "keywords",
                    "ccs concepts", "ACM Reference Format")


@dataclass(slots=True)
class Author:
    """A single author with optional affiliation markers."""

    name: str
    affiliation_indices: list[int] = field(default_factory=list)
    email: str = ""
    marker: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "affiliations": self.affiliation_indices,
                "email": self.email, "marker": self.marker}


@dataclass(slots=True)
class PaperMetadata:
    """Structured front matter of a paper."""

    title: str = ""
    authors: list[Author] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    version_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": [a.to_dict() for a in self.authors],
            "affiliations": self.affiliations,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "version_label": self.version_label,
        }


# --------------------------------------------------------------------------- helpers
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_arxiv_stamp(text: str) -> bool:
    return bool(_ARXIV_STAMP.match((text or "").strip()))


def _strip_markers(text: str) -> tuple[str, str]:
    """Remove footnote markers from a name; return ``(name, marker)``."""
    marker = ""
    m = re.match(rf"^\s*([{re.escape(_MARKER_CHARS)}]+)\s*(.*)$", text)
    if m:
        marker = m.group(1)
        text = m.group(2)
    else:
        text = re.sub(f"[{re.escape(_MARKER_CHARS)}]+", "", text)
    return _clean(text), marker


def _looks_like_person_name(text: str) -> bool:
    """A personal name looks like ``Firstname Lastname`` (2-4 tokens)."""
    text = _clean(text)
    if not text or len(text) > 60:
        return False
    if _EMAIL.search(text) or "@" in text:
        return False
    tokens = text.split()
    if not (2 <= len(tokens) <= 4):
        return False
    for tok in tokens:
        core = tok.strip(".,")
        if not core:
            return False
        # An organisation is never a person.
        if core.lower() in _ORG_WORDS:
            return False
        # Initials ("A.", "N.") or capitalised words.
        if not (core[0].isupper() or core.lower() in _NAME_PARTICLES):
            return False
        if not re.fullmatch(r"[A-Za-z\u00c0-\u024f.'\-\u0140-\u017f\u2217]+", core):
            return False
    return True


def _split_author_list(raw: str) -> list[tuple[str, str]]:
    """Split a comma separated author list into ``(name, marker)`` pairs."""
    raw = _clean(raw)
    if not raw:
        return []
    raw = re.sub(r",?\s+and\s+", ", ", raw, flags=re.IGNORECASE)
    out: list[tuple[str, str]] = []
    for part in (p.strip() for p in raw.split(",")):
        if not part or part.lower() in {"and", "&"}:
            continue
        name, marker = _strip_markers(part)
        if name:
            out.append((name, marker))
    return out


def _is_affiliation_line(text: str) -> bool:
    lower = text.lower()
    if len(text) > 300:
        return False
    return any(marker in lower for marker in _AFFILIATION_MARKERS)


# --------------------------------------------------------------------------- title
def _pick_title_block(first_page: list[Block], body_size: float,
                      page_width: float) -> Block | None:
    """Choose the block that most likely is the paper title."""
    best: Block | None = None
    best_score = -1e9

    for i, b in enumerate(first_page[:20]):
        text = _clean(b.text)
        if not text or len(text) < 8:
            continue
        if _is_arxiv_stamp(text):
            continue
        # The rotated arXiv stamp lives in the far-left margin.
        if b.x1 < 80 and b.height > 80:
            continue
        size = b.font_size
        if size < body_size * 1.12:
            continue
        if len(text.splitlines()) > 2:
            continue

        score = size - body_size
        # A title is usually wide and roughly centred.
        width_ratio = b.width / max(1.0, page_width)
        score += min(width_ratio, 0.6) * 12
        centre_offset = abs((b.x0 + b.x1) / 2 - page_width / 2)
        score += max(0.0, (150 - centre_offset)) / 60.0
        if b.is_bold:
            score += 1.5
        # Prefer blocks close to the top of the page.
        score += max(0.0, 60 - b.y0) / 30.0
        # A title is rarely a single short word.
        if len(text.split()) < 3:
            score -= 2.0
        if score > best_score:
            best_score = score
            best = b
    return best


def _find_band_end(first_page: list[Block], after_y: float) -> float:
    """Find the y coordinate where the front-matter band ends.

    The band ends at the first block that looks like the ``Abstract`` heading
    or the first numbered section heading.
    """
    for b in first_page:
        if b.y0 <= after_y:
            continue
        text = _clean(b.text).lower().rstrip(":")
        if not text or len(text) > 60:
            continue
        if text in _BAND_END_TITLES or text.startswith("abstract"):
            return b.y0
        if re.match(r"^\d+\s+[a-z]", text) and len(text) < 60:
            return b.y0
    return 1e9


# --------------------------------------------------------------------------- main
def extract_metadata(
    blocks: list[Block],
    *,
    body_size: float,
    filename: str = "",
    raw_text: str = "",
    page_width: float = 612.0,
) -> PaperMetadata:
    """Extract the front matter from the first page blocks."""
    meta = PaperMetadata()
    if not blocks:
        return meta

    first_page_no = min(b.page_no for b in blocks)
    first_page = sorted((b for b in blocks if b.page_no == first_page_no),
                        key=lambda b: (b.y0, b.x0))

    # ------------------------------------------------------------- title
    title_block = _pick_title_block(first_page, body_size, page_width)
    if title_block:
        meta.title = _clean(title_block.text)
    if not meta.title:
        meta.title = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)

    # ------------------------------------------------------------- authors
    band_top = title_block.y1 if title_block else 0.0
    band_bottom = _find_band_end(first_page, band_top)
    band = [b for b in first_page if band_top < b.y0 < band_bottom
            and not _is_arxiv_stamp(b.text)]

    authors: list[Author] = []
    affiliations: list[str] = []

    for block in band:
        lines = [ln.strip() for ln in block.text.splitlines() if ln.strip()]
        if not lines:
            continue
        head, marker = _strip_markers(lines[0])
        rest = lines[1:]

        email = ""
        for ln in lines:
            m = _EMAIL.search(ln)
            if m:
                email = m.group(0)
                break

        # --- case A: a block that is one author (name + affiliation + mail)
        if _looks_like_person_name(head):
            authors.append(Author(name=head, email=email, marker=marker))
            for ln in rest:
                if _EMAIL.search(ln):
                    continue
                if _is_affiliation_line(ln) and ln not in affiliations:
                    affiliations.append(ln)
            continue

        # --- case B: a block that lists several authors on one line
        pairs = _split_author_list(block.text.replace("\n", ", "))
        names = [(n, mk) for n, mk in pairs if _looks_like_person_name(n)]
        if names:
            for n, mk in names:
                authors.append(Author(name=n, email=email, marker=mk))
            continue

        # --- case C: an affiliation-only block
        for ln in lines:
            if _EMAIL.search(ln):
                continue
            if _is_affiliation_line(ln) and ln not in affiliations:
                affiliations.append(ln)

    # Deduplicate authors while preserving order.
    seen: set[str] = set()
    unique_authors: list[Author] = []
    for a in authors:
        key = a.name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_authors.append(a)
    meta.authors = unique_authors[:40]
    meta.affiliations = affiliations[:20]

    # ------------------------------------------------------------- abstract
    meta.abstract = _extract_abstract(first_page, raw_text)

    # ------------------------------------------------------------- keywords
    km = _KEYWORDS.search(raw_text)
    if km:
        meta.keywords = [k.strip() for k in re.split(r"[;,]", km.group(1)) if k.strip()][:12]

    # ------------------------------------------------------------- identifiers
    head_text = "\n".join(b.text for b in first_page[:16])
    m = _ARXIV_ID.search(head_text)
    if m:
        raw_id = m.group(1)
        vm = re.search(r"v(\d+)$", raw_id)
        meta.arxiv_id = re.sub(r"v\d+$", "", raw_id)
        meta.version_label = f"v{vm.group(1)}" if vm else ""
    else:
        m2 = _ARXIV_OLD_ID.search(head_text)
        if m2:
            raw_id = m2.group(1)
            vm = re.search(r"v(\d+)$", raw_id)
            meta.arxiv_id = re.sub(r"v\d+$", "", raw_id)
            meta.version_label = f"v{vm.group(1)}" if vm else ""

    d = _DOI.search(head_text)
    if d:
        meta.doi = d.group(1).rstrip(".")

    ym = re.search(r"\b(19|20)\d{2}\b", head_text)
    if ym:
        meta.year = int(ym.group(0))

    # The arXiv stamp carries the submission date: "arXiv:1706.03762v1 [cs.CL] 12 Jun 2017".
    for b in first_page:
        if _is_arxiv_stamp(b.text):
            dm = re.search(r"\b(19|20)\d{2}\b", b.text)
            if dm:
                meta.year = int(dm.group(0))
            break

    return meta


def _extract_abstract(first_page: list[Block], raw_text: str) -> str:
    """Extract the abstract paragraph."""
    for b in first_page:
        text = b.text.strip()
        m = _ABSTRACT.match(text)
        if m:
            rest = m.group(1).strip()
            if rest:
                return _clean(rest)
            idx = first_page.index(b)
            parts = [x.text.strip() for x in first_page[idx + 1 : idx + 4]]
            return _clean(" ".join(parts))[:4000]

    m = _ABSTRACT.search(raw_text)
    if m:
        tail = raw_text[m.end():]
        cut = re.search(r"\n\s*(?:1\s+)?(?:Introduction|I\.\s+INTRODUCTION|Keywords)\b",
                        tail, re.IGNORECASE)
        if cut:
            tail = tail[: cut.start()]
        return _clean(tail)[:4000]
    return ""


def infer_venue(meta: PaperMetadata, first_page_text: str) -> str:
    """Guess the publication venue from the *first page only*.

    Restricting the search to the first page avoids matching a venue that is
    merely mentioned inside a bibliography entry.
    """
    for pattern in (r"Proceedings of ([^\n]{5,120})",
                    r"In Proceedings ([^\n]{5,120})",
                    r"\b(Transactions on [A-Za-z ,-]{3,60})\b",
                    r"\b(Journal of [A-Za-z ,-]{3,60})\b",
                    r"\b(arXiv preprint arXiv:[0-9.]+)\b"):
        m = re.search(pattern, first_page_text, re.IGNORECASE)
        if m:
            candidate = _clean(m.group(1) if m.groups() else m.group(0))
            # A venue line mentions a year.
            if re.search(r"\b(19|20)\d{2}\b", candidate) or "arXiv" in candidate:
                meta.venue = candidate
                break
    return meta.venue


# ===========================================================================
# Scanned documents: metadata from OCR Markdown
# ===========================================================================
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# LaTeX footnote markers such as "$^{*}$" or "$^{\dagger}$".
_LATEX_MARKER = re.compile(r"\$\s*\^\s*(?:\{[^}]*\}|[^$\s])\s*\$")
_TRAILING_MARKERS = re.compile(r"[*†‡§¶•·]+\s*$")


def extract_metadata_from_markdown(
    pages: list[str],
    *,
    filename: str = "",
) -> PaperMetadata:
    """Extract the front matter from OCR Markdown.

    The scanned-paper path has no positioned blocks, so the metadata is read
    from the Markdown structure instead:

    * the first level-1 heading is the title;
    * the lines between the title and the next heading are the author list and
      the affiliations;
    * the paragraph under an ``Abstract`` heading is the abstract.
    """
    meta = PaperMetadata()
    if not pages:
        return meta

    first = pages[0]
    lines = first.splitlines()

    # ------------------------------------------------------------------ title
    headings: list[tuple[int, int, str]] = []
    title = ""
    title_idx = -1
    for i, line in enumerate(lines):
        m = _MD_HEADING_RE.match(line.strip())
        if not m:
            continue
        level = len(m.group(1))
        text = m.group(2).strip()
        headings.append((i, level, text))
        if not title and level == 1 and text:
            title = text
            title_idx = i
    meta.title = _clean(title) or re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)

    # ------------------------------------------------- authors / affiliations
    stop = len(lines)
    for i, _level, _text in headings:
        if i > title_idx:
            stop = i
            break
    band = lines[title_idx + 1: stop] if title_idx >= 0 else lines[:stop]

    authors: list[Author] = []
    affiliations: list[str] = []
    for raw in band:
        text = raw.strip()
        if not text:
            continue
        text = _LATEX_MARKER.sub(" ", text)
        text = _TRAILING_MARKERS.sub("", text).strip()
        if not text:
            continue

        # An e-mail line belongs to the author above it.
        if _EMAIL.search(text):
            continue

        head, marker = _strip_markers(text)
        if not head:
            continue

        if _looks_like_person_name(head):
            authors.append(Author(name=head, marker=marker))
            continue

        # Several names on one line, e.g. "Ashish Vaswani, Noam Shazeer".
        pairs = _split_author_list(head)
        names = [(n, mk) for n, mk in pairs if _looks_like_person_name(n)]
        if len(names) > 1:
            for n, mk in names:
                authors.append(Author(name=n, marker=mk))
            continue

        if _is_affiliation_line(head) and head not in affiliations:
            affiliations.append(head)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[Author] = []
    for a in authors:
        key = a.name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(a)
    meta.authors = unique[:40]
    meta.affiliations = affiliations[:20]

    # -------------------------------------------------------------- abstract
    for i, level, text in headings:
        if not re.match(r"^abstract\b", text.strip(), re.IGNORECASE):
            continue
        stop = len(lines)
        for j, _lvl, _t in headings:
            if j > i:
                stop = j
                break
        body = " ".join(l.strip() for l in lines[i + 1: stop] if l.strip())
        body = _LATEX_MARKER.sub(" ", body)
        meta.abstract = _clean(body)[:4000]
        break

    # ----------------------------------------------------------- identifiers
    head_text = "\n".join(lines[:40])
    m = _ARXIV_ID.search(head_text)
    if m:
        raw_id = m.group(1)
        vm = re.search(r"v(\d+)$", raw_id)
        meta.arxiv_id = re.sub(r"v\d+$", "", raw_id)
        meta.version_label = f"v{vm.group(1)}" if vm else ""
    else:
        m2 = _ARXIV_OLD_ID.search(head_text)
        if m2:
            raw_id = m2.group(1)
            vm = re.search(r"v(\d+)$", raw_id)
            meta.arxiv_id = re.sub(r"v\d+$", "", raw_id)
            meta.version_label = f"v{vm.group(1)}" if vm else ""

    d = _DOI.search(head_text)
    if d:
        meta.doi = d.group(1).rstrip(".")

    ym = re.search(r"\b(19|20)\d{2}\b", head_text)
    if ym:
        meta.year = int(ym.group(0))

    # The arXiv stamp carries the submission date.
    stamp = re.search(r"arXiv:\d{4}\.\d{4,5}v?\d*\s*\[?[^\n]*?((?:19|20)\d{2})", head_text)
    if stamp:
        meta.year = int(stamp.group(1))

    km = _KEYWORDS.search("\n".join(pages))
    if km:
        meta.keywords = [k.strip() for k in re.split(r"[;,]", km.group(1)) if k.strip()][:12]

    return meta
