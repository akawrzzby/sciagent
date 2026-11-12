"""The agent orchestrator.

SciAgent runs a **two-phase agent loop** per request:

Phase 1 — *Evidence gathering* (ReAct)
    The model receives the tool schemas and decides which tools to call.  Each
    tool call is executed by the :class:`~app.tools.registry.ToolRegistry` and
    the result is appended to the conversation.  The loop terminates when the
    model stops requesting tools or the step budget is exhausted.

Phase 2 — *Grounded generation*
    The gathered evidence is turned into a numbered ``EVIDENCE`` block and the
    model writes the final answer **token by token** (streamed to the browser).
    Every claim must carry a ``[n]`` citation; the guardrail layer then removes
    any tag that does not resolve to a retrieved passage.

On top of the loop sit the other three agent components:

``router``      classifies the intent (rule-based first, LLM as fallback)
``planner``     decomposes the request into an executable task DAG
``reflection``  verifies the final answer against the evidence
``memory``      keeps the conversation and the per-paper memory of a session

Every step is published on the session's event channel so the UI can render
the agent's thinking in real time.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from app.core.config import get_settings
from app.core.events import get_bus, session_channel
from app.core.logging import get_logger
from app.llm import guardrails, prompts
from app.llm.client import LLMClient, get_llm_client
from app.rag.retriever import Passage
from app.tools.registry import ToolRegistry

logger = get_logger(__name__)


# ===========================================================================
# Data structures
# ===========================================================================
@dataclass
class EvidenceItem:
    """One numbered piece of evidence available to the answer generator."""

    index: int
    text: str
    paper_id: str = ""
    paper_title: str = ""
    page_no: int = 0
    section_title: str = ""
    chunk_id: str = ""
    char_start: int = 0
    char_end: int = 0
    source: str = "search_library"   # which tool produced it
    score: float = 0.0

    def citation(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "page_no": self.page_no,
            "section_title": self.section_title,
            "chunk_id": self.chunk_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "quote": self.text[:320],
            "score": round(self.score, 5),
            "source": self.source,
        }

    def render(self) -> str:
        header = f"[{self.index}] "
        loc = f"({self.paper_title or self.paper_id}"
        if self.page_no:
            loc += f", p.{self.page_no}"
        if self.section_title:
            loc += f", {self.section_title}"
        loc += ")"
        return f"{header}{loc}\n{self.text}"


@dataclass
class AgentEvent:
    """An event emitted by the orchestrator."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}


@dataclass
class AgentContext:
    """Everything the orchestrator needs to answer one request."""

    question: str
    session_id: str = ""
    paper_ids: list[str] = field(default_factory=list)
    language: str = "zh"
    stream: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Orchestrator
# ===========================================================================
class AgentOrchestrator:
    """Runs one agent conversation."""

    def __init__(
        self,
        context: AgentContext,
        *,
        llm: LLMClient | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.context = context
        self.settings = get_settings()
        self.llm = llm or get_llm_client()
        self.registry = registry
        self.bus = get_bus()
        self.channel = session_channel(context.session_id or "default")

        self.evidence: list[EvidenceItem] = []
        self.trace: list[dict[str, Any]] = []
        self.plan: dict[str, Any] = {}
        self.intent: str = ""
        self.intent_confidence: float = 0.0
        self._pending: list[dict[str, Any]] = []
        self.final_answer: str = ""
        self.final_citations: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ publish
    async def _emit(self, event_type: str, **data: Any) -> None:
        """Publish an event to the session channel."""
        await self.bus.publish(self.channel, event_type, **data)
        if self.context.stream:
            self._pending.append({"type": event_type, **data})

    # ------------------------------------------------------------------ public
    async def run(self) -> AsyncIterator[dict[str, Any]]:
        """Execute the agent and yield SSE-ready events."""
        self._pending = []

        # --- input validation ------------------------------------------
        ok, error = guardrails.check_user_input(self.context.question)
        if not ok:
            yield {"type": "error", "message": error}
            return

        # --- 1. route ---------------------------------------------------
        await self._emit("stage", stage="routing", message="正在识别意图 …")
        await self._route()
        for _event in self._drain():
            yield _event
        await self._emit("route", intent=self.intent, confidence=self.intent_confidence,
                         plan=self.plan)

        # --- 2. plan ----------------------------------------------------
        await self._emit("stage", stage="planning", message="正在拆解任务 …")
        await self._plan()
        for _event in self._drain():
            yield _event
        await self._emit("plan", plan=self.plan)

        # --- 3. evidence gathering (ReAct) -------------------------------
        await self._emit("stage", stage="retrieving", message="正在检索证据 …")
        async for event in self._gather_evidence():
            yield event
        for _event in self._drain():
            yield _event

        # --- 4. grounded answer -----------------------------------------
        await self._emit("stage", stage="generating", message="正在基于证据生成回答 …")
        async for event in self._generate_answer():
            yield event
        answer, citations = self.final_answer, self.final_citations
        for _event in self._drain():
            yield _event

        # --- 5. reflection ----------------------------------------------
        report = self._reflect(answer, citations)
        await self._emit("guardrail", **report.to_dict())
        for _event in self._drain():
            yield _event

        final_answer = report.answer
        yield {"type": "final", "answer": final_answer, "citations": citations,
               "intent": self.intent, "plan": self.plan, "trace": self.trace,
               "guardrail": report.to_dict()}

    # ------------------------------------------------------------------ routing
    async def _route(self) -> None:
        """Classify the intent: cheap rules first, LLM as a fallback."""
        question = self.context.question
        rule_intent = rule_based_intent(question)
        if rule_intent:
            self.intent = rule_intent
            self.intent_confidence = 0.9
            return

        payload, _raw = await self.llm.complete_json(
            prompts.SYSTEM_ROUTER,
            f"User request: {question}",
            schema_hint='{"intent": "string", "confidence": "number", "slots": {}}',
            session_id=self.context.session_id,
            purpose="router",
        )
        if payload:
            self.intent = str(payload.get("intent", "single_qa"))
            try:
                self.intent_confidence = float(payload.get("confidence", 0.5))
            except (TypeError, ValueError):
                self.intent_confidence = 0.5
        else:
            self.intent = "single_qa"
            self.intent_confidence = 0.3

    # ------------------------------------------------------------------ planning
    async def _plan(self) -> None:
        """Turn the request into an executable task plan."""
        payload, _raw = await self.llm.complete_json(
            prompts.SYSTEM_PLANNER,
            f"User request: {self.context.question}\nDetected intent: {self.intent}\n"
            f"Selected paper ids: {self.context.paper_ids}",
            schema_hint=(
                '{"intent": "string", "reasoning": "string", '
                '"subtasks": [{"id": "t1", "goal": "string", "tool": "string", '
                '"depends_on": [], "requires_evidence": true}], '
                '"papers": [], "needs_clarification": false, "clarification_question": ""}'
            ),
            session_id=self.context.session_id,
            purpose="planner",
        )
        if payload:
            self.plan = payload
            # The planner may refine the intent.
            planned_intent = str(payload.get("intent", "") or "")
            if planned_intent and planned_intent != self.intent:
                self.plan["router_intent"] = self.intent
                self.intent = planned_intent
        else:
            self.plan = {
                "intent": self.intent,
                "reasoning": "fallback plan (planner did not return valid JSON)",
                "subtasks": [{"id": "t1", "goal": self.context.question,
                              "tool": "search_library", "depends_on": [],
                              "requires_evidence": True}],
                "papers": self.context.paper_ids,
                "needs_clarification": False,
            }

    # ------------------------------------------------------------------ ReAct loop
    async def _gather_evidence(self) -> AsyncIterator[dict[str, Any]]:
        """Phase 1: let the model call tools until it has enough evidence."""
        if self.registry is None or not self.registry.names:
            yield {"type": "tool_result", "name": "none",
                   "summary": "没有可用的工具，直接生成回答"}
            return

        tools = self.registry.schemas()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompts.SYSTEM_QA},
            {"role": "user", "content": self._initial_user_message()},
        ]

        max_steps = self.settings.agent_max_steps
        for step in range(max_steps):
            t0 = time.perf_counter()
            try:
                resp = await self.llm.chat(
                    messages, tools=tools, tool_choice="auto",
                    temperature=0.1, session_id=self.context.session_id,
                    purpose="agent:react",
                )
            except Exception as exc:
                logger.warning("agent step %s failed: %s", step, exc)
                break

            if resp.reasoning:
                await self._emit("thinking", text=resp.reasoning, step=step)
                for _event in self._drain():
                    yield _event

            if not resp.tool_calls:
                # The model decided it has enough evidence.
                self.trace.append({"step": step, "type": "stop",
                                   "latency_ms": resp.latency_ms})
                yield {"type": "agent_stop", "step": step, "reason": "no_more_tool_calls"}
                break

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in resp.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in resp.tool_calls:
                await self._emit("tool_call", name=tc.name, arguments=tc.arguments, step=step)
                for _event in self._drain():
                    yield _event

                result = await self.registry.execute(tc.name, tc.arguments)
                self._absorb_evidence(tc.name, result)

                summary = summarise_tool_result(tc.name, result)
                await self._emit("tool_result", name=tc.name, summary=summary,
                                 evidence_count=len(self.evidence), step=step)
                for _event in self._drain():
                    yield _event

                self.trace.append({
                    "step": step, "type": "tool_call", "name": tc.name,
                    "arguments": tc.arguments, "summary": summary,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                })

                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": guardrails.truncate(json.dumps(result, ensure_ascii=False)),
                })

            if len(self.evidence) >= 24:
                break

    def _initial_user_message(self) -> str:
        parts = [f"Question: {self.context.question}"]
        if self.context.paper_ids:
            parts.append(f"Restrict the search to these paper ids: {self.context.paper_ids}")
        plan = self.plan.get("subtasks") or []
        if plan:
            goals = "; ".join(str(t.get("goal", "")) for t in plan)
            parts.append(f"Your plan: {goals}")
        parts.append(
            "First call `search_library` (and `read_passages` / `compare_papers` when "
            "needed) to gather the evidence. Stop calling tools once you have the "
            "passages that answer the question."
        )
        return "\n".join(parts)

    def _absorb_evidence(self, tool_name: str, result: Any) -> None:
        """Turn a tool result into numbered evidence items."""
        if not isinstance(result, dict):
            return
        passages = result.get("passages") or result.get("results") or []
        for p in passages:
            if not isinstance(p, dict):
                continue
            text = p.get("content") or p.get("text") or ""
            if not text:
                continue
            item = EvidenceItem(
                index=len(self.evidence) + 1,
                text=text.strip(),
                paper_id=p.get("paper_id", ""),
                paper_title=p.get("paper_title", ""),
                page_no=p.get("page_no", 0),
                section_title=p.get("section_title", "") or "",
                chunk_id=p.get("chunk_id", ""),
                char_start=p.get("char_start", 0),
                char_end=p.get("char_end", 0),
                source=tool_name,
                score=float(p.get("score", 0.0)),
            )
            # Avoid duplicate evidence (same chunk retrieved twice).
            if any(e.chunk_id and e.chunk_id == item.chunk_id for e in self.evidence):
                continue
            self.evidence.append(item)

    # ------------------------------------------------------------------ generation
    async def _generate_answer(self) -> AsyncIterator[dict[str, Any]]:
        """Phase 2: write the grounded answer (streamed)."""
        evidence_block = self._render_evidence()
        system = prompts.SYSTEM_MULTI if self.intent == "multi_compare" else prompts.SYSTEM_QA
        if not self.evidence:
            message = guardrails.build_abstention_message()
            self.final_answer = message
            self.final_citations = []
            await self._emit("abstain", reason="no_evidence")
            return

        user = (
            f"Question: {self.context.question}\n\n"
            f"EVIDENCE\n{evidence_block}\n\n"
            "Write the final answer. Use [n] to cite the evidence blocks you rely on. "
            "Do not mention the phrase 'EVIDENCE' in your answer."
        )

        answer_parts: list[str] = []
        async for frame in self.llm.stream(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.1, session_id=self.context.session_id, purpose="agent:answer",
        ):
            kind = frame.get("type")
            if kind == "reasoning":
                text = frame.get("text", "")
                await self._emit("thinking", text=text, phase="answer")
                for _event in self._drain():
                    yield _event
            elif kind == "delta":
                text = frame.get("text", "")
                answer_parts.append(text)
                await self._emit("delta", text=text)
                for _event in self._drain():
                    yield _event
            elif kind == "done":
                resp = frame.get("response")
                if resp is not None and resp.content and not "".join(answer_parts).strip():
                    answer_parts = [resp.content]

        self.final_answer = "".join(answer_parts).strip()
        self.final_citations = [e.citation() for e in self.evidence]

    def _render_evidence(self) -> str:
        parts = []
        for e in self.evidence:
            parts.append(e.render())
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ reflection
    def _reflect(self, answer: str, citations: list[dict[str, Any]]) -> Any:
        """Verify the answer against the evidence and strip dangling citations."""
        allowed = {e.index for e in self.evidence}
        cleaned, report = guardrails.bind_citations(answer, allowed)
        report.answer = cleaned  # type: ignore[attr-defined]

        if not self.evidence:
            report.supported = False
            report.note = "没有检索到任何证据，已改为拒答"
        elif report.dropped_citations:
            report.note = (report.note or "") + "；这些引用没有对应的证据段落"
        return report

    # ------------------------------------------------------------------ helpers
    def _drain(self) -> list[dict[str, Any]]:
        out = self._pending
        self._pending = []
        return out


# ===========================================================================
# Helpers
# ===========================================================================
_RULES: list[tuple[tuple[str, ...], str]] = [
    (("比较", "对比", "区别", "差异", "vs", "versus"), "multi_compare"),
    (("翻译", "translate"), "translate"),
    (("去重", "重复", "duplicate"), "dedup_check"),
    (("图谱", "引用关系", "citation graph"), "graph"),
    (("综述", "review", "survey"), "review"),
    (("未来工作", "future work", "创新点", "研究方向"), "future_work"),
    (("推荐", "recommend"), "recommend"),
    (("检索", "下载", "arXiv", "arxiv", "查找论文"), "search"),
    (("论文框架", "提纲", "outline", "framework"), "write_paper"),
    (("作图", "绘图", "架构图", "diagram", "tikz", "mermaid"), "visualise"),
    (("公式", "formula", "equation"), "formula"),
    (("总结", "摘要", "summar", "概括"), "summarize"),
]


def rule_based_intent(question: str) -> str:
    """A cheap, deterministic intent guess (used before the LLM router)."""
    q = (question or "").lower()
    for keys, intent in _RULES:
        if any(k in q for k in keys):
            return intent
    return ""


def summarise_tool_result(tool_name: str, result: Any) -> str:
    """A one-line human readable summary of a tool result (for the UI trace)."""
    if not isinstance(result, dict):
        return str(result)[:200]
    if "error" in result:
        return f"错误: {result['error']}"
    if tool_name == "search_library":
        n = len(result.get("passages", []))
        papers = {p.get("paper_title", "") for p in result.get("passages", [])}
        return f"检索到 {n} 条证据，覆盖 {len(papers)} 篇文献"
    if tool_name == "read_passages":
        return f"读取了 {len(result.get('passages', []))} 个完整段落"
    if tool_name == "compare_papers":
        return f"完成 {len(result.get('papers', []))} 篇论文的结构化对比"
    if tool_name == "build_graph":
        s = result.get("stats", {})
        return (f"构建图谱：{s.get('node_count', 0)} 节点 / {s.get('edge_count', 0)} 条引用边，"
                f"基石论文 {s.get('foundation_count', 0)} 篇")
    if tool_name == "mine_future_work":
        return f"提炼出 {len(result.get('opportunities', []))} 个待探索方向"
    if tool_name == "search_arxiv":
        return f"arXiv 命中 {result.get('count', 0)} 篇"
    if tool_name == "download_paper":
        return "论文已下载并入库" if result.get("ok") else "下载失败"
    if tool_name == "check_reproducibility":
        return ("发现公开代码仓库" if result.get("has_code")
                else "未发现公开代码仓库")
    if tool_name == "generate_framework":
        return "已生成论文框架"
    if tool_name == "generate_diagram":
        return f"已生成 {result.get('diagram_type', '')} 架构图"
    if tool_name == "plot_csv":
        return "已生成图表与图注"
    if tool_name == "translate":
        return "翻译完成"
    return json.dumps(result, ensure_ascii=False)[:200]
