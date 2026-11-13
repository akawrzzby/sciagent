"""Contradiction detection between the papers of the library.

PaperQA2 detects contradictions with ContraCrow, an LLM trained on thousands
of annotated contradiction pairs.  That needs a training corpus this project
does not have, and the library only holds a handful of citations between its
own papers -- too little material for that approach.

So the detection here takes the other route: **contradictory numbers**.  Two
passages that report the same metric, in the same context, with different
values, contradict each other -- and the contradiction can be shown, because
both passages can be quoted.  The classic case in the test library is the two
versions of "Attention Is All You Need": version 1 reports a BLEU score of
41.0 on WMT 2014 English-to-German, version 7 reports 41.8 for the same
task.

The rule has three parts, and all three have to hold:

1. the two passages report the **same metric** (BLEU, accuracy, F1, ...);
2. they talk about the **same thing** -- the surrounding context has to
   overlap, otherwise "BLEU 41.0 on English-to-German" and "BLEU 28.4 on
   English-to-French" are two results, not a contradiction;
3. the values **differ** by more than the tolerance.

A second, weaker detector looks for **opposite claims** about the same
subject: one passage says a method improves a metric, another says it does
not.  It is reported with a lower confidence because the negation is detected
by pattern, not by reading.

Everything here is rule based and explainable: every conflict carries the two
quotes, the numbers, the reason and a confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------- numbers
# The metric names that carry a number in the sentence.  "BLEU+4" and
# "ROUGE-L" are as common as plain "BLEU" or "F1".
_METRIC_NAMES = (
    r"BLEU(?:\+\d|\-\d)?"
    r"|ROUGE(?:-\w)?\d?"
    r"|F1(?:[- ]score)?"
    r"|accuracy"
    r"|precision"
    r"|recall"
    r"|perplexity"
    r"|PPL"
    r"|AUC"
    r"|top-1 accuracy"
    r"|pass@\d+"
    r"|MAE"
    r"|RMSE"
    r"|mAP"
)

# "<metric> ... <number>"  --  "a BLEU score of 41.0".
# The gap between the metric and its value may only hold a connector
# ("score of", "is", "reaches", ...).  A free gap would happily pair "BLEU"
# with a number that belongs to something else entirely ("BLEU on the WMT
# 2014 English-to-German task" -> 2014).
_METRIC_BEFORE = re.compile(
    rf"(?P<metric>{_METRIC_NAMES})"
    r"(?:\s+scores?)?"
    r"(?:\s*[,;:]\s*)?"
    r"(?:\s+(?:of|is|are|at|to|reaches|reach|achieves|achieve|obtains|obtain|"
    r"got|get|we|report|reported))?"
    r"\s*(?P<value>\d+(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent\b|percentage points?\b|points?\b)?",
    re.IGNORECASE,
)

# "<number> <metric>"  --  "28.4 BLEU"
_METRIC_AFTER = re.compile(
    rf"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|percent\b)?\s*"
    rf"(?P<metric>{_METRIC_NAMES})",
    re.IGNORECASE,
)

# The words that are too common to describe what a number refers to.
_CONTEXT_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "with", "on", "in", "to",
    "is", "are", "was", "were", "be", "been", "from", "by", "at", "as", "that",
    "this", "it", "its", "our", "we", "their", "model", "models", "score",
    "scores", "scored", "obtain", "obtains", "obtained", "obtaining",
    "achieve", "achieves", "achieved", "achieving", "reach", "reaches",
    "reached", "report", "reports", "reported", "get", "gets", "got",
    "result", "results", "baseline", "respectively", "task", "tasks",
    "experiment", "experiments", "dataset", "datasets", "data", "setting",
    "settings", "case", "cases", "state", "the-art", "art", "table",
    "figure", "fig", "section", "et", "al", "they", "higher", "lower",
    "best", "final", "also", "same", "such", "these", "than", "then",
    "when", "while", "where", "which", "both", "about", "over", "under",
    "between", "after", "before", "can", "could", "may", "one", "two",
    "three", "all", "any", "more", "most", "other",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Absolute and relative tolerance: two numbers that differ by less than this
# are the same number written twice.
_ABS_TOLERANCE = 0.05
_REL_TOLERANCE = 0.01


# --------------------------------------------------------------------------- model
@dataclass
class NumberClaim:
    """One reported number."""

    metric: str
    value: float
    unit: str
    text: str
    context: str
    paper_id: str = ""
    paper_title: str = ""
    page_no: int = 0
    chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "text": self.text,
            "context": self.context,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "page_no": self.page_no,
            "chunk_id": self.chunk_id,
        }


@dataclass
class Conflict:
    """A contradiction between two passages."""

    kind: str                      # "numeric" | "claim"
    metric: str = ""
    summary: str = ""
    left: dict[str, Any] = field(default_factory=dict)
    right: dict[str, Any] = field(default_factory=dict)
    delta: float | None = None
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "metric": self.metric,
            "summary": self.summary,
            "left": self.left,
            "right": self.right,
            "delta": self.delta,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- helpers
def _norm_metric(name: str) -> str:
    """A comparable key for a metric name."""
    m = name.lower().strip()
    m = re.sub(r"\s+", " ", m)
    aliases = {
        "f1": "F1", "f1 score": "F1", "f1-score": "F1",
        "bleu": "BLEU", "top-1 accuracy": "accuracy", "ppl": "perplexity",
        "mse": "RMSE",
    }
    return aliases.get(m, m.upper() if len(m) <= 6 else m)


def _context_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9\-\.]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _CONTEXT_STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


def _numbers_differ(a: float, b: float) -> bool:
    delta = abs(a - b)
    if delta <= _ABS_TOLERANCE:
        return False
    scale = max(abs(a), abs(b), 1e-9)
    return delta / scale > _REL_TOLERANCE


def _context_overlap(a: set[str], b: set[str]) -> float:
    """How much the two contexts talk about the same thing (0..1)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


# A table is a list of numbers, not a statement: it must not be read as a
# claim about a subject.
_NUMERIC_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def _is_tabular(content: str) -> bool:
    """Whether a passage is a table rather than prose."""
    text = content or ""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]*|\d+(?:\.\d+)?", text)
    if len(tokens) < 20:
        return False
    numbers = sum(1 for t in tokens if _NUMERIC_TOKEN.fullmatch(t))
    return numbers / len(tokens) >= 0.25


def _paper_frequencies(chunks: Sequence[dict[str, Any]]) -> dict[str, int]:
    """How many papers of the library mention each word.

    A word that almost every paper uses ("performance", "model", "training")
    does not name anything; a word that two papers share does.  The subject
    overlap of two claims is therefore measured over the rare words only --
    otherwise every "we observe an improvement" matches every "we do not
    observe an improvement", which is noise, not a finding.
    """
    df: dict[str, int] = {}
    seen: dict[str, set[str]] = {}
    for chunk in chunks:
        paper = str(chunk.get("paper_id") or chunk.get("paper_title") or "")
        words = _context_words(chunk.get("content", ""))
        if paper not in seen:
            seen[paper] = set()
        seen[paper].update(words)
    for words in seen.values():
        for word in words:
            df[word] = df.get(word, 0) + 1
    return df


def _specific_words(words: set[str], df: dict[str, int], papers: int) -> set[str]:
    """The words of a subject that actually name something."""
    limit = max(3, min(8, papers // 4))
    return {w for w in words if df.get(w, 0) <= limit}


# The capitalised words of a context: datasets, tasks, model families.
_CAPITALISED = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)\b")

# A capitalised word at the start of a sentence is usually just grammar.
_SENTENCE_START_WORDS = {
    "the", "a", "an", "in", "on", "we", "this", "these", "it", "as", "by",
    "for", "to", "when", "while", "our", "its", "both", "at", "but", "so",
    "then", "thus", "however", "table", "figure", "fig", "tab", "after",
    "before", "since", "if", "there", "their", "here", "also", "because",
    "although", "with", "from", "using", "based", "note", "overall",
    "specifically", "first", "second", "third", "such", "our", "their",
}


def _task_signature(context: str) -> set[str]:
    """The named entities of a context.

    Two passages that report the same metric but name a *different* task are
    not in conflict -- "BLEU 41.0 on English-to-German" and "BLEU 28.4 on
    English-to-French" are two results, not a contradiction.  The named
    entities are what tells the two apart, so they are compared separately
    from the plain word overlap.
    """
    sig: set[str] = set()
    for m in _CAPITALISED.finditer(context or ""):
        token = m.group(1).lower()
        if token in _CONTEXT_STOPWORDS or token in _SENTENCE_START_WORDS:
            continue
        sig.add(token)
    return sig


def _same_task(left: str, right: str, *, strict: bool = False) -> bool:
    """Whether the two contexts talk about the same named task.

    With ``strict`` the sameness has to be *established*: a passage that names
    a dataset, a task or a model family and a passage that names none are not
    known to be about the same thing, and a contradiction needs it.
    """
    a = _task_signature(left)
    b = _task_signature(right)
    if not a or not b:
        return not strict   # nothing to compare -- fall back to the word overlap
    return a == b


# --------------------------------------------------------------------------- extraction
def extract_number_claims(text: str, *, window: int = 220) -> list[NumberClaim]:
    """Every ``(metric, value)`` pair that a passage reports."""
    out: list[NumberClaim] = []
    for sentence in _split_sentences(text):
        found: list[tuple[str, float, str, int, int]] = []
        for m in _METRIC_BEFORE.finditer(sentence):
            found.append((m.group("metric"), float(m.group("value")),
                          (m.group("unit") or "").lower().strip(), m.start(), m.end()))
        for m in _METRIC_AFTER.finditer(sentence):
            found.append((m.group("metric"), float(m.group("value")),
                          (m.group("unit") or "").lower().strip(), m.start(), m.end()))
        for metric, value, unit, start, end in found:
            left = max(0, start - window // 2)
            right = min(len(sentence), end + window // 2)
            out.append(NumberClaim(
                metric=_norm_metric(metric),
                value=value,
                unit=unit,
                text=sentence[:400],
                context=sentence[left:right],
            ))
    return out


def _attach_source(claim: NumberClaim, chunk: dict[str, Any]) -> NumberClaim:
    claim.paper_id = chunk.get("paper_id", "") or ""
    claim.paper_title = chunk.get("paper_title", "") or ""
    claim.page_no = int(chunk.get("page_no", 0) or 0)
    claim.chunk_id = chunk.get("id", chunk.get("chunk_id", "")) or ""
    return claim


# --------------------------------------------------------------------------- numeric
def detect_numeric_conflicts(
    chunks: Sequence[dict[str, Any]],
    *,
    min_overlap: float = 0.34,
    limit_per_metric: int = 4,
) -> list[Conflict]:
    """Find passages that report the same metric with different values."""
    claims: list[NumberClaim] = []
    for chunk in chunks:
        for claim in extract_number_claims(chunk.get("content", "")):
            claims.append(_attach_source(claim, chunk))

    conflicts: list[Conflict] = []
    by_metric: dict[str, list[NumberClaim]] = {}
    for claim in claims:
        by_metric.setdefault(claim.metric, []).append(claim)

    for metric, group in by_metric.items():
        if len(group) < 2:
            continue
        found = 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                left, right = group[i], group[j]
                if left.chunk_id and left.chunk_id == right.chunk_id:
                    continue
                if left.paper_id and left.paper_id == right.paper_id:
                    # The same paper may legitimately report several values.
                    continue
                if not _numbers_differ(left.value, right.value):
                    continue
                if not _same_task(left.context, right.context):
                    continue
                # The metric the two sides share must not be counted as
                # evidence that they talk about the same thing.  Without this
                # exclusion, "humans achieve 95.6% accuracy" (GPT-3, few-shot
                # tasks) and "the final model achieves 97%-98% accuracy on NSP"
                # (BERT) look like the same measurement, because the only word
                # they share is the metric itself.  Two statements about the
                # same metric are only in conflict if they say something more
                # than "both mention the metric".
                metric_words = _context_words(metric)
                left_words = _context_words(left.context) - metric_words
                right_words = _context_words(right.context) - metric_words
                overlap = _context_overlap(left_words, right_words)
                if overlap < min_overlap:
                    continue
                delta = abs(left.value - right.value)
                rel = delta / max(abs(left.value), abs(right.value), 1e-9)
                confidence = min(0.98, 0.45 + 0.5 * overlap)
                conflicts.append(Conflict(
                    kind="numeric",
                    metric=metric,
                    summary=(f"同一指标 {metric} 在两处给出不同取值："
                             f"{left.value:g} vs {right.value:g}"),
                    left=left.to_dict(),
                    right=right.to_dict(),
                    delta=round(delta, 6),
                    confidence=confidence,
                    reason=(f"指标相同（{metric}）、上下文重合度 {overlap:.0%}、"
                            f"数值相差 {delta:g}（{rel:.1%}）"),
                ))
                found += 1
                if found >= limit_per_metric:
                    break
            if found >= limit_per_metric:
                break
    conflicts.sort(key=lambda c: -c.confidence)
    return conflicts


# --------------------------------------------------------------------------- claims
# A claim about a subject: "<subject> improves / reduces ..." and its negation.
# The subject is the handful of words in front of the direction verb, so that
# "layer normalization improves the training speed" and "layer normalization
# does not improve the training speed" can be lined up.
_CLAIM_DIRECTION = re.compile(
    r"(?P<before>[^.!?]{0,90}?)\b(?P<direction>"
    r"improv\w*|increas\w*|boost\w*|rais\w*|enhanc\w*|outperform\w*|"
    r"reduc\w*|decreas\w*|drop\w*|declin\w*|worsen\w*|hurt\w*|harm\w*|"
    r"underperform\w*|degrad\w*)\b",
    re.IGNORECASE,
)

_NEGATION = re.compile(
    r"\b(?:does not|do not|did not|is not|are not|was not|were not|not|no |"
    r"without|fails? to|fail(?:ed)? to|cannot|can't|never)\b",
    re.IGNORECASE,
)

_POSITIVE = ("improv", "increas", "boost", "rais", "enhanc", "outperform")
_NEGATIVE = ("reduc", "decreas", "drop", "declin", "worsen", "hurt", "harm",
             "underperform", "degrad")

# The words that turn a subject into noise.
_SUBJECT_STOPWORDS = _CONTEXT_STOPWORDS | {
    "significantly", "substantially", "considerably", "also", "still",
    "however", "therefore", "thus", "hence", "well", "much", "more", "less",
    "little", "surprisingly", "interestingly", "recently", "finally",
    "indeed", "overall", "generally", "typically", "usually", "often",
    "always", "never", "furthermore", "moreover", "hence", "because",
}

# The verbs that carry no subject.
_VERB_WORDS = {
    "observe", "observed", "observes", "observing", "show", "shows", "showed",
    "shown", "show", "find", "finds", "found", "report", "reports", "reported",
    "note", "notes", "noted", "see", "sees", "saw", "suggest", "suggests",
    "suggested", "indicate", "indicates", "indicated", "demonstrate",
    "demonstrates", "demonstrated", "reveal", "reveals", "revealed",
    "expect", "expects", "expected", "believe", "believes", "believe",
    "appear", "appears", "appeared", "seem", "seems", "seemed", "tend",
    "tends", "tended", "lead", "leads", "led", "cause", "causes", "caused",
    "result", "results", "resulted", "yield", "yields", "yielded",
}


def _direction_sign(direction: str) -> int:
    d = direction.lower()
    for prefix in _POSITIVE:
        if d.startswith(prefix):
            return 1
    for prefix in _NEGATIVE:
        if d.startswith(prefix):
            return -1
    return 1


@dataclass
class DirectionClaim:
    """One directional claim about a subject."""

    subject: str
    direction: str
    polarity: int          # +1 "better", -1 "worse"
    negated: bool
    text: str = ""
    context: str = ""
    paper_id: str = ""
    paper_title: str = ""
    page_no: int = 0
    chunk_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.subject,
            "value": 0.0,
            "unit": "",
            "text": self.text,
            "context": self.context,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "page_no": self.page_no,
            "chunk_id": self.chunk_id,
        }


def _subject_of(before: str) -> str:
    """The subject phrase in front of a direction verb.

    Only the content words are kept: the verbs of the sentence ("observe",
    "show") and the negation are noise, and a subject made of noise would
    make unrelated sentences look like they agree.
    """
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", before or "")
    # A word broken across a line ("Surpris- ingly") is one word.
    merged: list[str] = []
    for w in words:
        if merged and merged[-1].endswith("-"):
            merged[-1] = merged[-1][:-1] + w
            continue
        merged.append(w)
    kept: list[str] = []
    for w in merged:
        low = w.lower()
        if low in _SUBJECT_STOPWORDS or low in _VERB_WORDS:
            continue
        if _NEGATION.fullmatch(w) or low == "not":
            continue
        kept.append(w)
    if not kept:
        return ""
    return " ".join(kept[-5:]).strip()


def _direction_claims(text: str) -> list[DirectionClaim]:
    out: list[DirectionClaim] = []
    for sentence in _split_sentences(text):
        for m in _CLAIM_DIRECTION.finditer(sentence):
            before = m.group("before")
            negated = bool(_NEGATION.search(before))
            sign = _direction_sign(m.group("direction"))
            if negated:
                sign = -sign
            subject = _subject_of(before)
            if not subject:
                continue
            out.append(DirectionClaim(
                subject=subject,
                direction=m.group("direction"),
                polarity=sign,
                negated=negated,
                text=sentence[:400],
                context=sentence,
            ))
    return out


def detect_claim_conflicts(
    chunks: Sequence[dict[str, Any]],
    *,
    min_overlap: float = 0.6,
    limit: int = 6,
) -> list[Conflict]:
    """Find passages that make opposite claims about the same subject.

    The rule is deliberately strict.  Both sides must name a subject of at
    least two content words, the subjects must overlap, and the overlap is
    measured over the *discriminative* words only -- the ones that few papers
    of the library use.  A loose version of this detector reports every "we
    observe an increase" against every "we did not observe an improvement",
    which is noise, not a finding.
    """
    df = _paper_frequencies(chunks)
    papers = len({str(c.get("paper_id") or c.get("paper_title") or "")
                  for c in chunks}) or 1

    claims: list[DirectionClaim] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if _is_tabular(content):
            continue
        for claim in _direction_claims(content):
            claim.paper_id = chunk.get("paper_id", "") or ""
            claim.paper_title = chunk.get("paper_title", "") or ""
            claim.page_no = int(chunk.get("page_no", 0) or 0)
            claim.chunk_id = chunk.get("id", chunk.get("chunk_id", "")) or ""
            if len(_context_words(claim.subject)) < 2:
                continue
            claims.append(claim)

    conflicts: list[Conflict] = []
    for i in range(len(claims)):
        left = claims[i]
        left_words = _specific_words(_context_words(left.subject), df, papers)
        if not left_words:
            continue
        for j in range(i + 1, len(claims)):
            right = claims[j]
            if left.polarity == right.polarity:
                continue
            if left.chunk_id and left.chunk_id == right.chunk_id:
                continue
            if left.paper_id and left.paper_id == right.paper_id:
                continue
            right_words = _specific_words(_context_words(right.subject), df, papers)
            if not right_words:
                continue
            overlap = _context_overlap(left_words, right_words)
            if overlap < min_overlap:
                continue
            if not _same_task(left.context, right.context, strict=True):
                continue
            conflicts.append(Conflict(
                kind="claim",
                metric=left.subject,
                summary=(f"对「{left.subject}」的结论相反："
                         f"一处称{'不' if left.negated else ''}"
                         f"{left.direction}，另一处相反"),
                left=left.to_dict(),
                right=right.to_dict(),
                delta=None,
                confidence=min(0.8, 0.3 + 0.5 * overlap),
                reason=f"结论方向相反（{left.direction} vs {right.direction}），"
                       f"主语重合度 {overlap:.0%}",
            ))
            if len(conflicts) >= limit:
                return conflicts
    return conflicts


def _conflict_key(conflict: Conflict) -> tuple:
    """The identity of a conflict, so that the same finding is not counted
    twice when it appears in several copies of a passage."""
    def side(item: dict[str, Any]) -> tuple:
        text = re.sub(r"\s+", " ", (item.get("text") or "")).strip().lower()
        # The paper is not part of the identity: the same sentence is stored
        # in several chunks of the library, and finding it twice is not
        # finding it twice.
        return (item.get("metric", ""),
                round(float(item.get("value") or 0.0), 4), text[:90])

    left = side(conflict.left)
    right = side(conflict.right)
    if right < left:
        left, right = right, left
    return (conflict.kind, conflict.metric, left, right)


def _dedupe(conflicts: Sequence[Conflict]) -> list[Conflict]:
    seen: set[tuple] = set()
    out: list[Conflict] = []
    for c in conflicts:
        key = _conflict_key(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# --------------------------------------------------------------------------- entry
def detect_contradictions(
    chunks: Sequence[dict[str, Any]],
    *,
    include_claims: bool = True,
) -> dict[str, Any]:
    """Run both detectors and return the conflicts with a short summary."""
    numeric = detect_numeric_conflicts(chunks)
    claims = detect_claim_conflicts(chunks) if include_claims else []
    all_conflicts = _dedupe(numeric + claims)
    papers = sorted({c.left.get("paper_id", "") for c in all_conflicts
                     if c.left.get("paper_id")} |
                    {c.right.get("paper_id", "") for c in all_conflicts
                     if c.right.get("paper_id")})
    return {
        "conflicts": [c.to_dict() for c in all_conflicts],
        "count": len(all_conflicts),
        "numeric_count": len(_dedupe(numeric)),
        "claim_count": len(_dedupe(claims)),
        "papers_involved": papers,
        "scanned_chunks": len(chunks),
    }


def conflicts_to_markdown(result: dict[str, Any]) -> str:
    """A short report of the detected contradictions."""
    lines = ["# 矛盾检测报告", ""]
    conflicts = result.get("conflicts") or []
    if not conflicts:
        lines.append("未在当前文献库中检出矛盾。")
        return "\n".join(lines)
    lines.append(f"共检出 {result.get('count', 0)} 处矛盾"
                 f"（数值型 {result.get('numeric_count', 0)}、"
                 f"结论型 {result.get('claim_count', 0)}）。")
    lines.append("")
    for i, c in enumerate(conflicts, start=1):
        lines.append(f"## {i}. {c.get('summary', '')}")
        lines.append(f"- 置信度：{c.get('confidence', 0):.0%}｜依据：{c.get('reason', '')}")
        for side, label in (("left", "一侧"), ("right", "另一侧")):
            item = c.get(side) or {}
            lines.append(f"- {label}：{item.get('paper_title', '')}"
                         f"（p.{item.get('page_no', 0)}）"
                         f"｜{item.get('text', '')}")
        lines.append("")
    return "\n".join(lines)
