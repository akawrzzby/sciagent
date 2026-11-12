"""Guardrails: the academic-safety layer.

Five independent protections:

1. **Evidence binding** – a citation may only point at a passage that the
   retriever actually returned.  Tags that do not resolve are removed and the
   answer is marked as partially unsupported.
2. **Citation authenticity** – references produced by the writing assistant are
   checked against the real library; invented entries are dropped.
3. **Prompt-injection defence** – document text is treated as *data*, never as
   instructions; suspicious patterns are neutralised before they reach the LLM.
4. **Structured-output validation** – JSON answers are validated against a
   schema and re-requested once when they do not conform.
5. **Refusal / abstention** – when the evidence does not support an answer the
   agent must say so instead of guessing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.core.logging import get_logger

logger = get_logger(__name__)

# ===========================================================================
# 1. Prompt injection
# ===========================================================================
# Patterns that indicate an instruction embedded in a document which tries to
# hijack the agent.
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions",
    r"disregard\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions|rules)",
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"new\s+instructions?\s*:",
    r"system\s*prompt\s*:",
    r"<\s*/?\s*(?:system|assistant|instructions?)\s*>",
    r"请忽略(?:之前|以上|前面)的(?:所有)?指令",
    r"你现在是[一一个]",
    r"新的?指令[：:]",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Text that looks like an instruction to the model gets wrapped in a data
# delimiter so that the model treats it as content.
_INSTRUCTION_MARKER = re.compile(
    r"(?im)^\s*(?:instruction|instructions|note to (?:the )?(?:ai|model|assistant)|"
    r"重要说明|指令)\s*[:：].*$"
)


def sanitize_document_text(text: str) -> tuple[str, list[str]]:
    """Neutralise instruction-like content inside a document.

    Returns ``(clean_text, warnings)``.  The original wording is preserved –
    we only annotate it so that the model does not follow it.
    """
    warnings: list[str] = []
    if not text:
        return text, warnings

    if _INJECTION_RE.search(text):
        warnings.append("文档中检测到疑似提示注入内容，已做隔离处理")

    def _wrap(match: re.Match[str]) -> str:
        return f"[untrusted-content]{match.group(0)}[/untrusted-content]"

    cleaned = _INJECTION_RE.sub(_wrap, text)
    cleaned = _INSTRUCTION_MARKER.sub(
        lambda m: f"[untrusted-content]{m.group(0)}[/untrusted-content]", cleaned
    )
    return cleaned, warnings


def check_user_input(text: str) -> tuple[bool, str]:
    """Validate the *user's* own input (not document content)."""
    if not text or not text.strip():
        return False, "问题不能为空"
    if len(text) > 8000:
        return False, "问题过长，请拆分后提问"
    return True, ""


# ===========================================================================
# 2. Evidence binding
# ===========================================================================
_CITATION_TAG = re.compile(r"\[(\d{1,3})\]")


@dataclass
class GroundingReport:
    """The outcome of checking an answer against its evidence."""

    supported: bool
    used_citations: list[int] = field(default_factory=list)
    dropped_citations: list[int] = field(default_factory=list)
    unsupported_ratio: float = 0.0
    note: str = ""
    answer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "used_citations": self.used_citations,
            "dropped_citations": self.dropped_citations,
            "unsupported_ratio": round(self.unsupported_ratio, 3),
            "note": self.note,
        }


def bind_citations(answer: str, allowed_indices: Iterable[int]) -> tuple[str, GroundingReport]:
    """Remove citation tags that do not correspond to retrieved evidence.

    This is the mechanical anti-hallucination check: the model may only point
    at passages that the retriever actually produced.
    """
    allowed = set(allowed_indices)
    used: list[int] = []
    dropped: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if idx in allowed:
            used.append(idx)
            return match.group(0)
        dropped.append(idx)
        return ""

    cleaned = _CITATION_TAG.sub(_replace, answer)

    report = GroundingReport(
        supported=not dropped,
        used_citations=sorted(set(used)),
        dropped_citations=sorted(set(dropped)),
    )
    if dropped:
        report.note = (f"移除了 {len(dropped)} 个没有证据支撑的引用标记 "
                       f"{sorted(set(dropped))}")
    return cleaned, report


# ===========================================================================
# 3. Citation authenticity
# ===========================================================================
@dataclass
class ReferenceCheck:
    """Result of validating generated references against the real library."""

    kept: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"kept": self.kept, "rejected": self.rejected,
                "kept_count": len(self.kept), "rejected_count": len(self.rejected)}


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def validate_references(
    generated: Iterable[dict[str, Any]],
    real_references: Iterable[dict[str, Any]],
    *,
    fuzzy_threshold: float = 0.82,
) -> ReferenceCheck:
    """Keep only references that really exist in the library.

    ``real_references`` are the parsed bibliography entries (or the metadata of
    the papers in the library).  A generated reference survives when its title
    matches one of them closely enough, or when its arXiv id / DOI matches.
    """
    real = list(real_references)
    real_by_key: dict[str, dict[str, Any]] = {}
    normalized_titles: list[tuple[str, dict[str, Any]]] = []

    for ref in real:
        title = ref.get("title", "")
        arxiv = (ref.get("arxiv_id") or "").strip()
        doi = (ref.get("doi") or "").strip().lower()
        if arxiv:
            real_by_key[f"arxiv:{arxiv.lower()}"] = ref
        if doi:
            real_by_key[f"doi:{doi}"] = ref
        if title:
            normalized_titles.append((_normalize_title(title), ref))

    check = ReferenceCheck()
    for gen in generated:
        gen_title = _normalize_title(gen.get("title", ""))
        gen_arxiv = (gen.get("arxiv_id") or "").strip().lower()
        gen_doi = (gen.get("doi") or "").strip().lower()

        match: dict[str, Any] | None = None
        if gen_arxiv and f"arxiv:{gen_arxiv}" in real_by_key:
            match = real_by_key[f"arxiv:{gen_arxiv}"]
        elif gen_doi and f"doi:{gen_doi}" in real_by_key:
            match = real_by_key[f"doi:{gen_doi}"]
        else:
            best, best_score = 0.0, None
            for norm_title, ref in normalized_titles:
                score = _title_similarity(gen_title, norm_title)
                if score > best:
                    best, best_score = score, ref
            if best_score is not None and best >= fuzzy_threshold:
                match = best_score

        if match is not None:
            merged = dict(gen)
            merged["verified"] = True
            merged["matched_title"] = match.get("title", "")
            check.kept.append(merged)
        else:
            rejected = dict(gen)
            rejected["verified"] = False
            rejected["reason"] = "在本地文献库中未找到该文献，已判定为幻觉引用并移除"
            check.rejected.append(rejected)

    return check


def _title_similarity(a: str, b: str) -> float:
    """Token-level similarity with a small bonus for exact containment."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    jaccard = len(ta & tb) / len(ta | tb)
    if a in b or b in a:
        jaccard = max(jaccard, 0.9)
    return jaccard


# ===========================================================================
# 4. Structured output validation
# ===========================================================================
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "plan": ("intent", "subtasks"),
    "router": ("intent",),
    "framework": ("title", "abstract", "references"),
    "diagram": ("diagram_type", "code"),
    "caption": ("caption",),
    "future_work": ("opportunities",),
    "review": ("title", "summary"),
    "version_diff": ("summary", "changes"),
}


def validate_structure(payload: dict[str, Any] | None, kind: str) -> tuple[bool, str]:
    """Check that a JSON payload contains the fields we expect."""
    if payload is None:
        return False, "模型未返回可解析的 JSON"
    required = _REQUIRED_FIELDS.get(kind, ())
    missing = [f for f in required if f not in payload]
    if missing:
        return False, f"缺少必需字段: {', '.join(missing)}"
    return True, ""


def coerce_list(value: Any) -> list[Any]:
    """Return a list for whatever the model produced."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return [value]


def truncate(text: str, limit: int = 20000) -> str:
    """Keep payloads within a sane size."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def safe_json_loads(text: str) -> dict[str, Any] | None:
    """Parse JSON, tolerating trailing commas and code fences."""
    from app.llm.client import extract_json

    return extract_json(text)


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# ===========================================================================
# 5. Abstention
# ===========================================================================
NO_EVIDENCE_PHRASES = (
    "提供的文献中没有相关信息",
    "库中没有找到相关内容",
    "the provided documents do not contain",
)


def requires_abstention(passages: list[Any], min_score: float = 0.0) -> bool:
    """Decide whether the agent must abstain from answering."""
    if not passages:
        return True
    return all(getattr(p, "score", 0.0) < min_score for p in passages)


def build_abstention_message() -> str:
    """The canonical abstention reply."""
    return (
        "**提供的文献中没有找到能支撑该问题的证据。**\n\n"
        "为了避免产生幻觉，我不对无依据的内容作答。你可以：\n"
        "1. 换用更贴近原文措辞的检索词再试一次；\n"
        "2. 先把相关论文上传到文献库；\n"
        "3. 让我从 arXiv 检索并下载候选文献后再回答。"
    )
