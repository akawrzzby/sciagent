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

# Place names that must never be read as a person's name.  The single words
# are only those that cannot plausibly be a surname; multi word names are
# listed separately because every part of them is an ordinary word.
_PLACE_WORDS = {
    "california", "usa", "china", "canada", "germany", "israel", "france",
    "japan", "korea", "singapore", "switzerland", "london", "toronto",
    "beijing", "shanghai", "massachusetts", "seattle", "angeles", "austin",
    "boston", "maryland", "texas", "india", "italy", "spain", "sweden",
    "norway", "denmark", "finland", "poland", "australia", "brazil",
    "cambridge", "oxford", "york", "wales", "scotland", "ireland",
    "netherlands", "austria", "hungary", "romania", "greece", "portugal",
}
_PLACE_PHRASES = {
    "mountain view", "new york", "san francisco", "los angeles", "palo alto",
    "menlo park", "sunnyvale", "redmond", "bellevue", "hillsboro",
    "san diego", "san jose", "new jersey", "new zealand", "hong kong",
}


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
# Ligatures that LaTeX emits as a single code point.  A name such as
# "Raﬀel" has to pass the name test, which only accepts letters.
_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
    "\u00a0": " ",
}


def _clean(text: str) -> str:
    for src, dst in _LIGATURES.items():
        text = (text or "").replace(src, dst)
    return re.sub(r"\s+", " ", text).strip()


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


# A trailing affiliation index, e.g. the "1" in "Jason Wei 1".  The TMLR and
# NeurIPS templates print the institution index as a small superscript right
# after the author name.
_TRAILING_INDEX = re.compile(r"^(.*?)[\s]?(\d{1,2})$")


def _split_affiliation_index(text: str) -> tuple[str, str]:
    """Split a trailing affiliation index off a name.

    Returns ``(name, index)``.  The index is only accepted when the text that
    remains still looks like a personal name, so that names such as "GPT 3"
    are left alone.
    """
    text = _clean(text)
    m = _TRAILING_INDEX.match(text)
    if not m:
        return text, ""
    candidate, index = m.group(1).strip(), m.group(2)
    if candidate and _looks_like_person_name(candidate):
        return candidate, index
    return text, ""


def _affiliation_index(marker: str) -> int | None:
    """The affiliation index encoded in an author marker, if any."""
    m = re.search(r"\d{1,2}", marker or "")
    return int(m.group(0)) if m else None


def _looks_like_person_name(text: str) -> bool:
    """A personal name looks like ``Firstname Lastname`` (2-4 core tokens).

    Name particles ("de", "van", "von" ...) do not count towards the limit, so
    that a name such as "Henrique Ponde de Oliveira Pinto" is still accepted
    as one person rather than being split or dropped.
    """
    text = _clean(text)
    if not text or len(text) > 60:
        return False
    if _EMAIL.search(text) or "@" in text:
        return False
    if text.lower() in _PLACE_PHRASES:
        return False
    tokens = text.split()
    if not (2 <= len(tokens) <= 6):
        return False
    core = [t for t in tokens if t.strip(".,").lower() not in _NAME_PARTICLES]
    if not (2 <= len(core) <= 4):
        return False
    for tok in tokens:
        core_tok = tok.strip(".,")
        if not core_tok:
            return False
        # An organisation or a place is never a person.
        if core_tok.lower() in _ORG_WORDS or core_tok.lower() in _PLACE_WORDS:
            return False
        # Initials ("A.", "N.") or capitalised words.
        if not (core_tok[0].isupper() or core_tok.lower() in _NAME_PARTICLES):
            return False
        if not re.fullmatch(r"[A-Za-z\u00c0-\u024f.'\-\u0140-\u017f\u2217]+", core_tok):
            return False
    return True


def _split_author_list(raw: str) -> list[tuple[str, str]]:
    """Split an author list into ``(name, marker)`` pairs.

    Three layouts occur in practice:

    * ``Ashish Vaswani, Noam Shazeer, ...`` - comma separated.
    * ``Mark Chen * 1 Jerry Tworek * 1 Heewoo Jun * 1`` - a dense author band
      where the names are separated by whitespace and affiliation indices
      instead of by commas (used by the large-author papers, e.g. the 40+
      author Codex paper).  Splitting such a block on commas alone yields one
      very long token sequence that no longer looks like a person name, and
      the author list silently comes out empty.
    * ``Mark Chen * Jerry Tworek * ...`` - a dense band without indices.
    """
    raw = _clean(raw)
    if not raw:
        return []
    raw = re.sub(r",?\s+and\s+", ", ", raw, flags=re.IGNORECASE)

    # --- dense band: the affiliation index is the anchor -------------------
    # Every author is followed by its affiliation index, so the text between
    # two indices is exactly one author (plus an optional footnote marker).
    dense = _split_dense_author_band(raw)
    if len(dense) > 1 and all(_looks_like_person_name(n) for n, _ in dense):
        return dense

    # --- dense band without indices: cut at the footnote markers ----------
    marker_re = re.compile(rf"[{re.escape(_MARKER_CHARS)}]+")
    if marker_re.search(raw):
        out: list[tuple[str, str]] = []
        for piece in marker_re.split(raw):
            piece = piece.strip()
            if not piece:
                continue
            # The affiliation index sits *between* the marker and the next
            # name ("Chen * 1 Tworek"), so it has to be stripped as well.
            piece = re.sub(r"^\d{1,2}\s+", "", piece)
            name, marker = _strip_author_segment(piece)
            if name:
                out.append((name, marker))
        if len(out) > 1 and all(_looks_like_person_name(n) for n, _ in out):
            return out

    # --- comma separated ---------------------------------------------------
    out = []
    for part in (p.strip() for p in raw.split(",")):
        if not part or part.lower() in {"and", "&"}:
            continue
        name, marker = _strip_markers(part)
        name, index = _split_affiliation_index(name)
        if index:
            marker = f"{marker}{index}"
        name = _clean(name)
        if name:
            out.append((name, marker))
    return out


# An affiliation index is a small standalone integer, e.g. the "1" in
# "Mark Chen 1".  It is surrounded by whitespace on both sides.
_AFFIL_INDEX_TOKEN = re.compile(r"(?<!\S)(\d{1,2})(?!\S)")


def _strip_author_segment(text: str) -> tuple[str, str]:
    """Strip the footnote marker and affiliation index that trail a name."""
    text = text.strip()
    marker = ""
    m = re.search(rf"([{re.escape(_MARKER_CHARS)}]+)\s*$", text)
    if m:
        marker = m.group(1)
        text = text[: m.start()]
    text, index = _split_affiliation_index(text)
    if index:
        marker = f"{marker}{index}"
    return _clean(text), marker


def _split_dense_author_band(raw: str) -> list[tuple[str, str]]:
    """Split an author band that uses affiliation indices as separators.

    Large-author papers print their author list as one dense block per line::

        Mark Chen * 1 Jerry Tworek * 1 Heewoo Jun * 1

    The affiliation index is the reliable anchor: every author is followed by
    one, so the text between two indices is exactly one author plus an
    optional footnote marker.  The index is kept in the marker so that the
    author can be linked back to the institution it belongs to.
    """
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _AFFIL_INDEX_TOKEN.finditer(raw):
        name, marker = _strip_author_segment(raw[pos:m.start()])
        index = m.group(1)
        if name:
            out.append((name, f"{marker}{index}"))
        pos = m.end()
    name, marker = _strip_author_segment(raw[pos:])
    if name:
        out.append((name, marker))
    return out


# An affiliation line that carries its own index, e.g. "1Google Research".
# The index is printed *without* a space, which is what tells it apart from a
# numbered section heading such as "12 Introduction".
_AFFIL_INDEX_PREFIX = re.compile(r"^\s*(\d{1,2})[A-Z]")


def _is_affiliation_line(text: str) -> bool:
    lower = text.lower()
    if len(text) > 300:
        return False
    if any(marker in lower for marker in _AFFILIATION_MARKERS):
        return True
    # "3UNC Chapel Hill" -- an indexed affiliation whose name carries none of
    # the usual institutional words.
    return len(text) < 80 and bool(_AFFIL_INDEX_PREFIX.match(text.strip()))


# --------------------------------------------------------------------------- year
def _year_from_arxiv_id(arxiv_id: str) -> int | None:
    """Derive the submission year from an arXiv identifier.

    Modern arXiv identifiers encode the submission month as ``YYMM``: the first
    two digits are the last two digits of the year, the next two are the
    month.  This has to be preferred over scanning the page for a four digit
    number, because a naive scan happily matches the identifier itself
    ("1910.02054" -> 1910) and reports a year that never existed.
    """
    m = re.match(r"^(\d{2})\d{2}\.\d{4,5}", (arxiv_id or "").strip())
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 90 else 2000 + yy


# --------------------------------------------------------------------------- abstract
# Signals that the front matter has moved past the abstract.  An extracted
# abstract routinely carries the equal-contribution footnote, the per-author
# contribution list, the conference footer, the keyword list or the first
# section heading, because none of those is a heading of its own.
_ABSTRACT_STOP_PATTERNS = (
    r"[\*∗✻✦]\s*equal contribution",
    r"[\*∗✻✦†‡]\s*(?:work performed|second author|corresponding author)",
    r"author contributions",
    r"contributions for all the authors",
    r"listing order is random",
    r"correspondence to",
    r"\d{1,2}(?:st|nd|rd|th)\s+conference",
    r"\bkeywords?\s*:",
    r"\bindex terms\b",
    r"\b\d{1,2}\.?\s+introduction\b",
    r"\bpreprint\.\s*under review\b",
    r"\barxiv:\d{4}\.\d{4,5}",
)


def _trim_abstract(text: str) -> str:
    """Cut the front-matter tail that follows the abstract.

    Everything that sits *below* the abstract on the first page -- the
    equal-contribution footnote, the per-author contribution list, the
    conference footer, the keyword list, the first section heading -- ends up
    inside the extracted abstract unless it is removed explicitly.  Quoting
    that tail as if it were part of the abstract is worse than having no
    abstract at all.
    """
    text = text or ""
    kept: list[str] = []
    for line in text.splitlines():
        cut = len(line)
        for pattern in _ABSTRACT_STOP_PATTERNS:
            m = re.search(pattern, line, re.IGNORECASE)
            if m is not None and m.start() < cut:
                cut = m.start()
        if cut < len(line):
            kept.append(line[:cut])
            break
        kept.append(line)
    return _clean(" ".join(kept))


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
        head, _head_marker = _strip_markers(lines[0])
        head, _head_index = _split_affiliation_index(head)

        email = ""
        for ln in lines:
            m = _EMAIL.search(ln)
            if m:
                email = m.group(0)
                break

        # --- case A: the block holds one or more authors, one per line -----
        if _looks_like_person_name(head):
            # The TMLR and NeurIPS templates print the author, a superscript
            # affiliation index and the e-mail address next to each other, so
            # one PDF block can hold the whole author list.  The block is
            # therefore read line by line, and the e-mail that follows an
            # author belongs to that author.
            first = len(authors)
            for ln in lines:
                ln_head, ln_marker = _strip_markers(ln)
                ln_head, ln_index = _split_affiliation_index(ln_head)
                found = _EMAIL.search(ln)
                if found:
                    ln_head = _clean(ln_head.replace(found.group(0), " "))
                if _looks_like_person_name(ln_head):
                    authors.append(Author(
                        name=ln_head,
                        email=found.group(0) if found else "",
                        marker=ln_marker,
                        affiliation_indices=[int(ln_index)] if ln_index else [],
                    ))
                    continue
                if found:
                    # A bare e-mail line belongs to the author printed above it.
                    if len(authors) > first:
                        authors[-1].email = authors[-1].email or found.group(0)
                    elif not email:
                        email = found.group(0)
                    continue
                if _is_affiliation_line(ln) and ln not in affiliations:
                    affiliations.append(ln)
            continue

        # --- case B: a block that lists several authors on one line
        pairs = _split_author_list(block.text.replace("\n", ", "))
        names = [(n, mk) for n, mk in pairs if _looks_like_person_name(n)]
        if names:
            for n, mk in names:
                index = _affiliation_index(mk)
                authors.append(Author(
                    name=n, email=email, marker=mk,
                    affiliation_indices=[index] if index else [],
                ))
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

    # Link every author to the institution it was indexed with.  An index is
    # only kept when an affiliation line actually carries it, so that a stray
    # number in a name cannot invent an affiliation.
    known = set()
    for aff in meta.affiliations:
        m = _AFFIL_INDEX_PREFIX.match(aff.strip())
        if m:
            known.add(int(m.group(1)))
    for a in meta.authors:
        a.affiliation_indices = [i for i in a.affiliation_indices if i in known]

    # ------------------------------------------------------------- abstract
    meta.abstract = _extract_abstract(first_page, raw_text, body_size)

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

    # -------------------------------------------------------------- year
    # Priority matters here: a naive scan of the first page for a four digit
    # number matches the arXiv identifier itself ("1910.02054" -> 1910) and
    # reports a year that never existed.
    meta.year = _year_from_arxiv_id(meta.arxiv_id)
    if meta.year is None:
        # The arXiv stamp carries the submission date: "arXiv:1706.03762v1 [cs.CL] 12 Jun 2017".
        for b in first_page:
            if _is_arxiv_stamp(b.text):
                dm = re.search(r"\b(19|20)\d{2}\b", b.text)
                if dm:
                    meta.year = int(dm.group(0))
                break
    if meta.year is None:
        # Last resort: scan the page, but ignore the identifier and the DOI.
        scrubbed = _DOI.sub(" ", _ARXIV_ID.sub(" ", head_text))
        ym = re.search(r"\b(19|20)\d{2}\b", scrubbed)
        if ym:
            meta.year = int(ym.group(0))

    return meta


def _extract_abstract(first_page: list[Block], raw_text: str,
                      body_size: float = 0.0) -> str:
    """Extract the abstract paragraph.

    On a two-column page the ``Abstract`` heading and the column next to it
    share the same vertical position, so the abstract body cannot be found by
    simply taking "the next few blocks" -- that picks up the *other* column's
    section heading and the abstract comes out empty.  The body blocks are
    therefore collected from the heading's own column, up to the next
    heading-sized block.
    """
    for b in first_page:
        text = b.text.strip()
        m = _ABSTRACT.match(text)
        if not m:
            continue
        rest = m.group(1).strip()
        if rest:
            # The abstract may continue on the following lines of the block.
            block_lines = text.splitlines()
            if len(block_lines) > 1:
                rest = "\n".join([rest, *block_lines[1:]]).strip()
            return _trim_abstract(rest)[:4000]

        parts: list[str] = []
        for nxt in first_page:
            if nxt is b or nxt.column != b.column or nxt.y0 <= b.y1:
                continue
            if body_size and nxt.spans:
                size = max((s.size for s in nxt.spans), default=0)
                if size > body_size * 1.12:
                    break  # the next heading of the same column
            parts.append(nxt.text.strip())
            if len(parts) >= 3:
                break
        if parts:
            return _trim_abstract(" ".join(parts))[:4000]

    m = _ABSTRACT.search(raw_text)
    if m:
        tail = raw_text[m.end():]
        cut = re.search(r"\n\s*(?:1\s+)?(?:Introduction|I\.\s+INTRODUCTION|Keywords)\b",
                        tail, re.IGNORECASE)
        if cut:
            tail = tail[: cut.start()]
        return _trim_abstract(tail)[:4000]
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
        meta.abstract = _trim_abstract(body)[:4000]
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

    # Priority: the identifier itself, then the stamp, then a scrubbed scan.
    meta.year = _year_from_arxiv_id(meta.arxiv_id)
    if meta.year is None:
        stamp = re.search(r"arXiv:\d{4}\.\d{4,5}v?\d*\s*\[?[^\n]*?((?:19|20)\d{2})", head_text)
        if stamp:
            meta.year = int(stamp.group(1))
    if meta.year is None:
        scrubbed = _DOI.sub(" ", _ARXIV_ID.sub(" ", head_text))
        ym = re.search(r"\b(19|20)\d{2}\b", scrubbed)
        if ym:
            meta.year = int(ym.group(0))

    km = _KEYWORDS.search("\n".join(pages))
    if km:
        meta.keywords = [k.strip() for k in re.split(r"[;,]", km.group(1)) if k.strip()][:12]

    return meta
