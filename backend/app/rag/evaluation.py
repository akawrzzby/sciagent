"""Rule based evaluation of the grounded answering pipeline.

The three metrics follow the definitions of RAGAS (Es et al., 2023) and ARES
(Saad-Falcon et al., 2024), but they are computed **without an LLM judge**.

That is a deliberate choice.  Agent Laboratory (Schmidgall et al., 2025)
reports that LLM reviewers systematically over-rate AI generated work --
6.1/10 where human PhD reviewers give 3.8/10.  A rule based number is the one
that can actually be defended in a review, and it can be re-run by anyone.

* **Faithfulness** -- the share of the answer's claims whose content words are
  covered by the evidence that the claim itself cites.  This is the
  "is the answer supported by its own sources" question.
* **Citation traceability** -- the share of the answer's citation tags that
  resolve to a passage that was actually retrieved.  This is the "can the
  reader click through" question.
* **Context precision** -- the share of the retrieved passages that the answer
  actually used.  This separates "retrieved the wrong material" from
  "answered wrongly from the right material", which is the distinction the
  AWS Labs RAG evaluation guide insists on.

The module has no dependency on the LLM client, the database or the network,
so it can be unit tested and re-run offline.

Matching between a claim and its evidence is lexical, so both sides are put
through the same normalisation before they are compared:

* number words are folded onto digits (``eight`` -> ``8``), because papers
  write "8 GPUs" while answers tend to write "eight GPUs";
* common English inflections are stemmed (``training`` -> ``train``), so that
  a paraphrase of the evidence is not counted as a hallucination;
* citation tags are removed from the claim before it is tokenised, otherwise
  every ``[1]`` would count as an uncovered token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# The citation tag the agent is asked to emit: [1], [2], ...
_CITATION_TAG = re.compile(r"\[(\d{1,3})\]")

# The share of the claim's content words that the cited passage has to cover
# for the claim to count as supported.
FAITHFULNESS_THRESHOLD = 0.6

# A claim that cites several passages is judged against the *union* of them:
# a cross-paper comparison legitimately draws one fact from each paper, and
# RAGAS asks whether the claim is supported by the context it cites, not by
# any single passage.  The anchor keeps that from becoming a free pass -- the
# claim still has to be substantially present in at least one of the passages
# it cites, so "cite everything and hope" does not work.
FAITHFULNESS_ANCHOR = 0.3

# Content tokens: latin words (digits allowed inside, e.g. "P100"), numbers and
# CJK runs.  Ordered so that the longest form wins at a given position.
_CONTENT_TOKEN = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[.'\-][A-Za-z0-9]+)*"
    r"|\d+(?:\.\d+)?%?"
    r"|[\u4e00-\u9fff]{2,}"
)

_PURE_NUMBER = re.compile(r"^\d+(?:\.\d+)?%?$")

# Ordinals ("6th", "21st") state a position, not a value: the passage lists the
# authors in order and never writes "6".  They are dropped before the tokens
# are counted.
_ORDINAL = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)

# English number words -> digits.
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
    "thousand": "1000", "million": "1000000", "billion": "1000000000",
}

# Suffix -> replacement, applied in order.  The minimum stem length differs per
# suffix: dropping "s" needs less context than dropping "ing".
_SUFFIXES: tuple[tuple[str, str, int], ...] = (
    ("ies", "y", 3),
    ("ing", "", 4),
    ("ed", "", 4),
    ("es", "", 4),
    ("s", "", 3),
    ("e", "", 4),
)

# Words that carry no evidential value.
#
# Besides the usual function words, the list holds the *generic* nouns of
# academic prose ("the sixth author of the paper", "the model was trained
# ...").  Those words frame a claim, they do not assert anything: a passage
# that states "Vaswani ... Google Brain" supports the claim "the 6th author
# is Vaswani" even though it never uses the word "author".  The check is made
# after stemming, so the plural and the past form are covered as well.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "has", "have", "had", "not", "but", "you", "our", "its", "their", "they",
    "these", "those", "which", "when", "where", "while", "also", "into",
    "than", "then", "them", "such", "each", "both", "can", "could", "may",
    "one", "two", "three", "all", "any", "more", "most", "other", "some",
    "used", "using", "use", "based", "show", "shows", "shown", "propose",
    "proposed", "paper", "papers", "work", "works", "via", "per", "out",
    "over", "under", "between", "about", "after", "before", "because",
    "author", "authors", "model", "models", "method", "methods", "approach",
    "approaches", "result", "results", "section", "sections", "table",
    "tables", "figure", "figures", "experiment", "experiments", "value",
    "values", "number", "numbers", "setting", "settings", "performance",
    "perform", "performs", "present", "presents", "presented", "introduce",
    "introduces", "introduced", "describe", "describes", "described", "give",
    "gives", "given", "report", "reports", "reported", "compare", "compares",
    "compared", "comparison", "different", "different", "similar",
    # Function words and the generic verbs of academic prose.  They frame a
    # claim, they do not assert anything: "the 6th author is Aidan N. Gomez,
    # who is listed as affiliated with the University of Toronto" is backed by
    # a passage that lists the name and the institution, even though it uses
    # none of the words "6th", "who", "listed" or "affiliated".
    "who", "whom", "whose", "what", "when", "where", "whether", "while",
    "although", "though", "since", "unless", "within", "upon", "respectively",
    "according", "listed", "list", "affiliated", "affiliation", "associated",
    "obtained", "obtain", "reach", "achieve", "produce", "allow", "allows",
    "enable", "enables", "require", "requires", "include", "includes",
    "provide", "provides", "contain", "contains", "appear", "appears",
    "known", "called", "named", "titled", "state", "states", "stated",
    "indicate", "indicates", "indicated", "find", "finds", "found",
    "observe", "observed", "suggest", "suggests", "demonstrate",
    "demonstrates", "expect", "expected", "believe", "remain", "remains",
    "become", "becomes", "seem", "seems", "help", "helps", "need", "needs",
    "consider", "considered", "typically", "usually", "often", "always",
    "never", "currently", "recently", "previously", "later", "earlier",
    "overall", "additionally", "furthermore", "moreover", "thus", "hence",
    "whereas", "during", "through", "across", "along", "without", "among",
    "against", "throughout", "despite", "besides", "onto", "therefore",
    "的", "了", "和", "是", "在", "有", "与", "及", "对", "为", "中", "上",
    "下", "等", "并", "而", "其", "该", "这", "那", "我们", "他们", "可以",
    "以及", "因此", "但是", "如果", "因为", "所以", "通过", "进行", "使用",
}

# Abbreviations that end in a dot but do not end a sentence.
_ABBREVIATIONS = {
    "fig", "figs", "eq", "eqs", "ref", "refs", "sec", "secs", "tab", "tabs",
    "vs", "etc", "al", "no", "cf", "resp", "approx", "dr", "prof", "mr", "ms",
    "e.g", "i.e",
}

# Markdown furniture.
_HORIZONTAL_RULE = re.compile(r"^([-*_])(?:\s*\1){2,}$")
_LIST_MARKER = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+|>\s?)")


# The attribution of a claim -- "the paper notes that", "the table shows" --
# is not part of the assertion: no passage ever writes "the paper notes".
# The attribution is stripped before the claim is compared with the evidence.
_ATTRIBUTION = re.compile(
    r"^(?:the\s+|this\s+|these\s+|a\s+|an\s+)?"
    r"(?:paper|table|figure|fig|section|author|authors|study|text|document|"
    r"passage|evidence|footnote|note|result|work|method|dataset)"
    r"\b[^,.]{0,40}?\b(?:note|notes|noted|show|shows|shown|state|states|"
    r"stated|indicate|indicates|indicated|report|reports|reported|say|says|"
    r"said|list|lists|listed|give|gives|given|mention|mentions|mentioned|"
    r"record|records|recorded|observe|observes|observed|confirm|confirms|"
    r"argue|argues|suggest|suggests|suggested|claim|claims)"
    r"\s+(?:that\s+)?",
    re.IGNORECASE,
)


def strip_attribution(text: str) -> str:
    """The assertion of a claim, without the attribution it carries."""
    stripped = _ATTRIBUTION.sub("", text or "").strip()
    return stripped or (text or "")


@dataclass
class Claim:
    """A single assertion taken from the answer."""

    text: str
    citations: list[int] = field(default_factory=list)
    supported: bool = False
    coverage: float = 0.0
    critical: float = 1.0
    reason: str = ""
    core: str = ""          # the assertion without its attribution


@dataclass
class EvaluationReport:
    """The three metrics plus the evidence behind them."""

    faithfulness: float = 0.0
    citation_traceability: float = 0.0
    context_precision: float = 0.0
    claims: list[Claim] = field(default_factory=list)
    total_citations: int = 0
    resolvable_citations: int = 0
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    used_chunk_ids: list[str] = field(default_factory=list)
    unsupported: list[Claim] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "citation_traceability": round(self.citation_traceability, 4),
            "context_precision": round(self.context_precision, 4),
            "claim_count": len(self.claims),
            "supported_claims": sum(1 for c in self.claims if c.supported),
            "total_citations": self.total_citations,
            "resolvable_citations": self.resolvable_citations,
            "retrieved_chunks": len(self.retrieved_chunk_ids),
            "used_chunks": len(self.used_chunk_ids),
            "unsupported": [
                {"text": c.text[:200], "coverage": round(c.coverage, 3),
                 "critical": round(c.critical, 3), "reason": c.reason}
                for c in self.unsupported
            ],
        }


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _stem(token: str) -> str:
    """A deliberately small stemmer -- enough to tolerate paraphrasing."""
    if not token.isascii() or not token.isalpha() or len(token) < 4:
        return token
    for suffix, replacement, min_stem in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_stem:
            return token[: -len(suffix)] + replacement
    return token


def _trailing_word(text: str, dot_index: int) -> str:
    """The word that ends right before ``text[dot_index]``."""
    end = dot_index
    while end > 0 and (text[end - 1].isalnum() or text[end - 1] in ".'-"):
        end -= 1
    return text[end:dot_index]


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences, for Chinese and English.

    English full stops need care: they appear inside decimals ("28.4"), inside
    abbreviations ("Fig. 1") and inside initials ("U.S.").  Only a full stop
    that is followed by whitespace and a capital letter or a digit is treated
    as a sentence boundary.  Chinese punctuation needs no whitespace after it.
    """
    out: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)

        # Chinese sentence punctuation: a boundary on its own.
        if ch in "\u3002\uff01\uff1f\uff1b":
            out.append("".join(buf))
            buf = []
            i += 1
            continue

        # Western terminators: a boundary when whitespace follows.
        if ch in "!?;":
            j = i + 1
            if j >= n or text[j].isspace():
                out.append("".join(buf))
                buf = []
                i = j
                continue
            i += 1
            continue

        if ch == ".":
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < n else ""
            # Decimals and version numbers are not boundaries.
            if prev.isdigit() and nxt.isdigit():
                i += 1
                continue
            # A full stop right after a citation tag ends the sentence.
            if prev == "]":
                j = i + 1
                if j >= n or text[j].isspace():
                    out.append("".join(buf))
                    buf = []
                    i = j
                    continue
                i += 1
                continue
            word = _trailing_word(text, i)
            if word.lower() in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                i += 1
                continue
            j = i + 1
            if j < n and text[j].isspace():
                k = j
                while k < n and text[k].isspace():
                    k += 1
                if k >= n or text[k].isupper() or text[k].isdigit() or _is_cjk(text[k]):
                    out.append("".join(buf))
                    buf = []
                    i = j
                    continue
        i += 1

    if buf:
        out.append("".join(buf))
    return [s.strip() for s in out if s.strip()]


def _content_blocks(text: str) -> list[str]:
    """Split an answer into paragraphs of prose, dropping markdown furniture."""
    blocks: list[str] = []
    current: list[str] = []
    in_code = False
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not line:
            if current:
                blocks.append(" ".join(current))
                current = []
            continue
        if _HORIZONTAL_RULE.match(line):
            continue
        if line.startswith("#"):  # headings
            continue
        if line.startswith("|"):  # table rows
            continue
        line = _LIST_MARKER.sub("", line).strip()
        if not line:
            continue
        current.append(line)
    if current:
        blocks.append(" ".join(current))
    return blocks


def _split_table_row(line: str) -> list[str]:
    """Split one Markdown table row into its cells."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_separator(cells: list[str]) -> bool:
    """True for the ``|---|---|`` line that separates header and body."""
    filled = [c for c in cells if c]
    return bool(filled) and all(re.fullmatch(r":?-{2,}:?", c) for c in filled)


def _table_claims(text: str) -> list[str]:
    """Turn a Markdown table into claim texts.

    A cross-paper answer is a table, and the value in a cell *is* the claim;
    the citation tag inside the cell says which passage backs it.  The row
    label is prepended, so that "41.0" is verified as "BLEU: 41.0" and not as
    a bare number -- otherwise any passage containing the number would pass.
    """
    claims: list[str] = []
    lines = (text or "").splitlines()
    index = 0
    total = len(lines)
    while index < total:
        if not lines[index].strip().startswith("|"):
            index += 1
            continue
        rows: list[list[str]] = []
        while index < total and lines[index].strip().startswith("|"):
            rows.append(_split_table_row(lines[index]))
            index += 1
        if rows and _is_table_separator(rows[0]):
            rows = rows[1:]
        elif len(rows) >= 2 and _is_table_separator(rows[1]):
            rows = rows[2:]
        for row in rows:
            cells = [c for c in row if c]
            if not cells:
                continue
            if _CITATION_TAG.search(cells[0]):
                # The first cell carries the citation itself -- no row label.
                label, values = "", cells
            else:
                label, values = cells[0], cells[1:]
            for cell in values:
                if not _CITATION_TAG.search(cell):
                    continue
                claims.append(f"{label}: {cell}" if label else cell)
    return claims


def extract_claims(answer: str) -> list[Claim]:
    """Split the answer into atomic assertions with their citation tags."""
    claims: list[Claim] = []
    for block in _content_blocks(answer or ""):
        for raw in _split_sentences(block):
            text = raw.strip()
            if len(text) < 8:
                continue
            if text.endswith(":"):
                # "The Adam optimizer was used with:" introduces a list; it
                # does not assert anything on its own.
                continue
            tags = [int(m.group(1)) for m in _CITATION_TAG.finditer(text)]
            claims.append(Claim(text=text, citations=tags,
                                core=strip_attribution(text)))
    for text in _table_claims(answer or ""):
        tags = [int(m.group(1)) for m in _CITATION_TAG.finditer(text)]
        claims.append(Claim(text=text, citations=tags,
                            core=strip_attribution(text)))
    return claims


def content_tokens(text: str) -> list[str]:
    """The normalised tokens that a piece of evidence has to cover."""
    cleaned = _ORDINAL.sub(" ", _CITATION_TAG.sub(" ", text or ""))
    tokens: list[str] = []
    for m in _CONTENT_TOKEN.finditer(cleaned):
        tok = m.group(0).lower()
        if _PURE_NUMBER.match(tok):
            tokens.append(tok)
            continue
        if tok in _NUMBER_WORDS:
            tokens.append(_NUMBER_WORDS[tok])
            continue
        has_cjk = any(_is_cjk(ch) for ch in tok)
        if len(tok) < (2 if has_cjk else 3):
            continue
        stem = _stem(tok)
        # The stop word test runs on the raw token *and* on its stem, so that
        # "authors" is recognised once "author" is a stop word.
        if tok in _STOPWORDS or stem in _STOPWORDS:
            continue
        tokens.append(stem)
    return tokens


def _has_cjk(token: str) -> bool:
    return any(_is_cjk(ch) for ch in token)


def coverage(claim_text: str, evidence_text: str) -> float:
    """The share of the claim's content tokens that the evidence contains.

    Latin words and numbers are matched as tokens.  A Chinese run is matched
    on its characters instead: Chinese is written without spaces, so a run is
    compared as a set of characters and is covered when most of them occur in
    the evidence.  Without that, one extra character ("该模型在" against
    "模型在") would decide the whole claim.
    """
    tokens = content_tokens(claim_text)
    if not tokens:
        return 1.0
    evidence = set(content_tokens(evidence_text))
    if not evidence:
        return 0.0

    ev_chars: set[str] = set()
    for token in evidence:
        if _has_cjk(token):
            ev_chars.update(ch for ch in token if _is_cjk(ch))

    hits = 0
    for token in tokens:
        if _has_cjk(token):
            chars = [ch for ch in token if _is_cjk(ch)]
            if chars and sum(1 for ch in chars if ch in ev_chars) / len(chars) >= 0.7:
                hits += 1
        elif token in evidence:
            hits += 1
    return hits / len(tokens)


def _raw_tokens(text: str) -> set[str]:
    """Every word of the text, lower cased, without stemming or stop words."""
    out: set[str] = set()
    for m in _CONTENT_TOKEN.finditer(_CITATION_TAG.sub(" ", text or "")):
        tok = m.group(0).lower()
        out.add(_NUMBER_WORDS.get(tok, tok))
    return out


def critical_tokens(text: str) -> set[str]:
    """The tokens a passage must state for the claim to be supported.

    Numbers, model names ("A100", "GPT-3") and acronyms ("BLEU", "WMT") are
    the parts of a claim that a passage either states or does not state.  A
    paraphrase may change the wording around them, it may not change their
    value -- so they are checked apart from the rest of the claim, and a
    claim that names the wrong GPU or the wrong score is not supported.
    """
    out: set[str] = set()
    for sentence in _split_sentences(_ORDINAL.sub(" ", text or "")):
        for m in _CONTENT_TOKEN.finditer(_CITATION_TAG.sub(" ", sentence)):
            raw = m.group(0)
            tok = raw.lower()
            if _PURE_NUMBER.match(tok):
                out.add(tok)
                continue
            if len(tok) < 2:
                continue
            has_digit = any(ch.isdigit() for ch in tok)
            is_acronym = raw.isupper() and any(ch.isalpha() for ch in raw)
            if has_digit or is_acronym:
                out.add(tok)
    return out


def critical_coverage(claim_text: str, evidence_text: str) -> float:
    """The share of the claim's critical tokens that the evidence states."""
    critical = critical_tokens(claim_text)
    if not critical:
        return 1.0
    evidence = _raw_tokens(evidence_text)
    if not evidence:
        return 0.0
    hits = 0
    for tok in critical:
        if tok in evidence:
            hits += 1
            continue
        # "WMT-2014" is stated by a passage that writes "WMT 2014".
        parts = [p for p in re.split(r"[-./]", tok) if p]
        if len(parts) > 1 and all(p in evidence for p in parts):
            hits += 1
    return hits / len(critical)


def best_supporting_window(
    text: str,
    queries: list[str],
    limit: int = 320,
) -> str:
    """The part of a passage that best matches the claims citing it.

    A passage is a page of extracted text, not a sentence: the author list of
    a paper is one block of newlines, the caption of a figure is another.
    The unit that is searched is therefore the *line*, and the window that is
    returned is the best line with one neighbour on each side, so that a claim
    spanning two lines is still covered.
    """
    body = (text or "").strip()
    if not body:
        return ""
    if not queries:
        return body[:limit]

    wanted: set[str] = set()
    for query in queries:
        wanted.update(content_tokens(query))
    if not wanted:
        return body[:limit]

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return body[:limit]

    scored: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        overlap = len(wanted & set(content_tokens(line)))
        scored.append((overlap, index))
    # The earliest line wins a tie, so that the window points at the first
    # place the claim is stated.
    best_overlap, best_index = max(scored, key=lambda item: (item[0], -item[1]))

    if best_overlap == 0:
        # Nothing of the claim is in this passage -- show its beginning.
        return body[:limit]

    window = lines[max(0, best_index - 1): best_index + 2]
    out = "\n".join(window)
    if len(out) > limit:
        out = out[:limit]
    return out


def evaluate_answer(
    answer: str,
    citations: Iterable[dict[str, Any]],
    *,
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD,
) -> EvaluationReport:
    """Score a grounded answer against the evidence it cites.

    ``citations`` is the list produced by the orchestrator; each entry carries
    an ``index`` (the ``[n]`` tag), a ``chunk_id`` and a ``quote``.
    """
    cites = list(citations or [])
    by_index: dict[int, dict[str, Any]] = {}
    for c in cites:
        try:
            idx = int(c.get("index", 0))
        except (TypeError, ValueError):
            continue
        if idx:
            by_index[idx] = c

    report = EvaluationReport()

    # --- citation traceability -------------------------------------------
    tags = [int(m.group(1)) for m in _CITATION_TAG.finditer(answer or "")]
    report.total_citations = len(tags)
    report.resolvable_citations = sum(1 for t in tags if t in by_index)
    report.citation_traceability = (
        report.resolvable_citations / report.total_citations if report.total_citations else 0.0
    )

    retrieved = {str(c.get("chunk_id")) for c in cites if c.get("chunk_id")}
    report.retrieved_chunk_ids = sorted(retrieved)

    # --- claims -----------------------------------------------------------
    report.claims = extract_claims(answer)
    used: set[str] = set()
    supported = 0
    for claim in report.claims:
        quotes = [str(by_index[t].get("text") or by_index[t].get("quote", ""))
                  for t in claim.citations if t in by_index]
        for t in claim.citations:
            item = by_index.get(t)
            if item and item.get("chunk_id"):
                used.add(str(item["chunk_id"]))
        if not quotes:
            claim.supported = False
            claim.coverage = 0.0
            claim.reason = "该断言没有引用任何已检索到的证据"
            report.unsupported.append(claim)
            continue
        # The cited passages are judged together: a cross-paper comparison
        # draws one fact from each paper, and no single passage covers the
        # whole sentence.  The anchor is the share that at least one of the
        # cited passages has to cover on its own.
        union = "\n\n".join(quotes)
        union_coverage = coverage(claim.core, union)
        union_critical = critical_coverage(claim.core, union)
        anchor = max(coverage(claim.core, q) for q in quotes)
        claim.coverage = union_coverage
        claim.critical = union_critical
        # A claim is supported when the cited passages cover enough of its
        # wording *and* state every critical fact (numbers, model names,
        # acronyms).  The second test is what catches "eight A100 GPUs"
        # answered with a passage about "eight P100 GPUs".
        claim.supported = (union_coverage >= faithfulness_threshold
                           and union_critical >= 1.0
                           and anchor >= FAITHFULNESS_ANCHOR)
        claim.reason = f"引用内容覆盖率 {claim.coverage:.0%}"
        if claim.critical < 1.0:
            claim.reason += f"，关键事实缺失 {1 - claim.critical:.0%}"
        if anchor < FAITHFULNESS_ANCHOR:
            claim.reason += f"，且没有单段证据支撑（最高 {anchor:.0%}）"
        if claim.supported:
            supported += 1
        else:
            claim.reason += "，低于阈值"
            report.unsupported.append(claim)

    report.used_chunk_ids = sorted(used)
    report.faithfulness = supported / len(report.claims) if report.claims else 0.0
    report.context_precision = (
        len(used & retrieved) / len(retrieved) if retrieved else 0.0
    )
    return report
