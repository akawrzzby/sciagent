"""The self-feedback loop of the agent.

When the answer is not covered by the evidence that it cites, the agent has to
look again.  These tests drive the orchestrator with a fake LLM and a fake
tool registry, so the loop is exercised without a model and without a
database.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.agents.orchestrator import AgentContext, AgentOrchestrator
from app.llm.client import ToolCall


class FakeResponse:
    def __init__(self, content: str = "", tool_calls=None, reasoning: str = "",
                 finish_reason: str = "stop") -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning = reasoning
        self.finish_reason = finish_reason
        self.latency_ms = 1
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.ok = True


class FakeLLM:
    """Returns a fixed sequence of answers."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls = 0
        self.stream_calls = 0
        self.kwargs: list[dict[str, Any]] = []

    async def complete_json(self, *args: Any, **kwargs: Any):
        return None, ""

    async def chat(self, messages, **kwargs: Any) -> FakeResponse:
        """The ReAct phase: ask for one tool call, then stop."""
        if self.calls == 0:
            self.calls += 1
            return FakeResponse(tool_calls=[
                ToolCall(id="t1", name="search_library",
                         arguments={"query": "BLEU score", "top_k": 4})])
        return FakeResponse()

    async def stream(self, messages, **kwargs: Any):
        self.stream_calls += 1
        self.kwargs.append(kwargs)
        text = self.answers[min(self.stream_calls - 1, len(self.answers) - 1)]
        yield {"type": "delta", "text": text}
        yield {"type": "done", "response": FakeResponse(content=text)}


class FakeRegistry:
    """A registry that hands out one passage per call."""

    def __init__(self, passages: list[dict[str, Any]]) -> None:
        self.passages = passages
        self.calls: list[dict[str, Any]] = []
        self.names = ["search_library"]

    def schemas(self):
        return []

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"name": name, "arguments": arguments})
        if name == "search_library" and self.passages:
            index = min(len(self.calls) - 1, len(self.passages) - 1)
            return {"passages": [self.passages[index]]}
        return {"passages": []}


class MultiRegistry(FakeRegistry):
    """A registry that returns every passage at once."""

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"name": name, "arguments": arguments})
        if name == "search_library":
            return {"passages": list(self.passages)}
        return {"passages": []}


class TruncatedLLM:
    """A model that burns its token budget on reasoning and writes nothing.

    This is what the cross-paper questions did in production: the completion
    tokens hit the cap, the finish reason is ``length`` instead of ``error``,
    the call is reported as successful and the answer body is empty.
    """

    def __init__(self, answers: list[str], fail_with: str = "") -> None:
        self.answers = list(answers)
        self.fail_with = fail_with
        self.calls = 0
        self.stream_calls = 0
        self.budgets: list[Any] = []

    async def complete_json(self, *args: Any, **kwargs: Any):
        return None, ""

    async def chat(self, messages, **kwargs: Any) -> FakeResponse:
        if self.calls == 0:
            self.calls += 1
            return FakeResponse(tool_calls=[
                ToolCall(id="t1", name="search_library",
                         arguments={"query": "BLEU score", "top_k": 4})])
        return FakeResponse()

    async def stream(self, messages, **kwargs: Any):
        self.stream_calls += 1
        self.budgets.append(kwargs.get("max_tokens"))
        if self.fail_with:
            yield {"type": "error", "message": self.fail_with}
            return
        text = self.answers[min(self.stream_calls - 1, len(self.answers) - 1)]
        if not text:
            # The budget went into the reasoning channel; nothing was written.
            yield {"type": "done",
                   "response": FakeResponse(content="", finish_reason="length")}
            return
        yield {"type": "delta", "text": text}
        yield {"type": "done", "response": FakeResponse(content=text)}


def _orchestrator(llm: FakeLLM, registry: FakeRegistry) -> AgentOrchestrator:
    orch = AgentOrchestrator(AgentContext(question="What BLEU score did it reach?",
                                           session_id="test"),
                              llm=llm, registry=registry)  # type: ignore[arg-type]
    return orch


def _run(orch: AgentOrchestrator) -> list[dict[str, Any]]:
    async def scenario() -> list[dict[str, Any]]:
        return [event async for event in orch.run()]

    return asyncio.run(scenario())


def test_an_unsupported_claim_triggers_a_second_lookup() -> None:
    """The first answer claims something its citation does not contain."""
    llm = FakeLLM([
        # The first draft cites [1] but the passage never mentions the score.
        "The model reached a BLEU score of 41.0 on the benchmark [1].",
        # The second draft is backed by the passage that was added -- so it has
        # to cite the *new* evidence block [2], not the old one.
        "The model reached a BLEU score of 28.4 on WMT 2014 [2].",
    ])
    registry = FakeRegistry([
        {"chunk_id": "c1", "paper_id": "p1", "paper_title": "Attention",
         "content": "We trained on the standard WMT 2014 dataset.", "page_no": 3},
        {"chunk_id": "c2", "paper_id": "p1", "paper_title": "Attention",
         "content": "The model reached a BLEU score of 28.4 on WMT 2014.",
         "page_no": 5},
    ])
    orch = _orchestrator(llm, registry)
    events = _run(orch)

    refine = [e for e in events if e.get("type") == "refine"]
    assert refine, "the agent did not notice the evidence gap"
    assert refine[0]["reason"] == "evidence_gap"
    assert refine[0]["round"] == 1

    # The gap query was turned into a retrieval call.
    assert any(c["name"] == "search_library" for c in registry.calls)

    # The answer was written a second time: one draft per generation.
    assert llm.stream_calls == 2
    assert orch.refine_rounds == 1

    final = [e for e in events if e.get("type") == "final"][0]
    assert final["refine_rounds"] == 1
    assert "28.4" in final["answer"]


def test_a_supported_answer_is_not_refined() -> None:
    llm = FakeLLM([
        "The model reached a BLEU score of 28.4 on WMT 2014 [1].",
    ])
    registry = FakeRegistry([
        {"chunk_id": "c1", "paper_id": "p1", "paper_title": "Attention",
         "content": "The model reached a BLEU score of 28.4 on WMT 2014.",
         "page_no": 5},
    ])
    orch = _orchestrator(llm, registry)
    events = _run(orch)

    assert not [e for e in events if e.get("type") == "refine"]
    assert llm.stream_calls == 1
    assert orch.refine_rounds == 0
    final = [e for e in events if e.get("type") == "final"][0]
    assert "[1]" in final["answer"]


def test_a_citation_that_points_at_the_wrong_block_is_repaired() -> None:
    """The model cites [1], but the author list is in [2]."""
    llm = FakeLLM([
        "The 6th author is Illia Polosukhin [1].",
    ])
    registry = MultiRegistry([
        {"chunk_id": "c1", "paper_id": "p1", "paper_title": "Attention",
         "content": "We trained on the standard WMT 2014 dataset.", "page_no": 3},
        {"chunk_id": "c2", "paper_id": "p1", "paper_title": "Attention",
         "content": "Illia Polosukhin*\nGoogle Research\nillia.polosukhin@gmail.com",
         "page_no": 1},
    ])
    orch = _orchestrator(llm, registry)
    events = _run(orch)

    final = [e for e in events if e.get("type") == "final"][0]
    assert "[2]" in final["answer"]
    assert "[1]" not in final["answer"]
    # The quote of the repaired citation shows the passage that backs it.
    quote = final["citations"][1]["quote"]
    assert "Illia Polosukhin" in quote


def test_two_claims_that_share_a_tag_are_repaired_separately() -> None:
    """One tag, two claims, two different passages -- they get two tags."""
    llm = FakeLLM([
        "The author list names Aidan N. Gomez [1]. "
        "A footnote notes that the work was performed at Facebook [1].",
    ])
    registry = MultiRegistry([
        {"chunk_id": "c1", "paper_id": "p1", "paper_title": "Attention",
         "content": "Ashish Vaswani*\nGoogle Brain\navaswani@google.com", "page_no": 1},
        {"chunk_id": "c2", "paper_id": "p1", "paper_title": "Attention",
         "content": "Aidan N. Gomez*\nUniversity of Toronto", "page_no": 1},
        {"chunk_id": "c3", "paper_id": "p1", "paper_title": "Attention",
         "content": "Work performed while at Facebook.", "page_no": 1},
    ])
    orch = _orchestrator(llm, registry)
    events = _run(orch)

    final = [e for e in events if e.get("type") == "final"][0]
    assert "Gomez [2]." in final["answer"]
    assert "Facebook [3]." in final["answer"]


def test_the_answer_step_uses_the_configured_thinking_mode() -> None:
    """The answer phase must forward the thinking switch to the model.

    Letting the model reason first is what used to eat the whole token budget
    before a single character of the answer was written.
    """
    from app.core.config import get_settings

    llm = FakeLLM([
        "The model reached a BLEU score of 28.4 on WMT 2014 [1].",
    ])
    registry = FakeRegistry([
        {"chunk_id": "c1", "paper_id": "p1", "paper_title": "Attention",
         "content": "The model reached a BLEU score of 28.4 on WMT 2014.",
         "page_no": 5},
    ])
    orch = _orchestrator(llm, registry)
    _run(orch)

    assert llm.kwargs, "the answer call recorded no parameters"
    expected = bool(get_settings().llm_answer_thinking)
    assert llm.kwargs[-1].get("thinking") == expected


def test_an_empty_answer_triggers_a_retry_instead_of_being_accepted() -> None:
    """The model wrote nothing: that is a failure, not an answer.

    In production the cross-paper questions came back with an empty body
    because the completion tokens hit the cap while the model was still
    reasoning -- the call was reported as successful, so the empty string was
    accepted as the answer.
    """
    llm = TruncatedLLM([
        "",  # the first draft: the whole budget went into the reasoning
        "The model reached a BLEU score of 28.4 on WMT 2014 [1].",
    ])
    registry = FakeRegistry([
        {"chunk_id": "c1", "paper_id": "p1", "paper_title": "Attention",
         "content": "The model reached a BLEU score of 28.4 on WMT 2014.",
         "page_no": 5},
    ])
    orch = _orchestrator(llm, registry)
    events = _run(orch)

    failed = [e for e in events if e.get("type") == "generation_failed"]
    assert failed, "the empty answer was not detected"
    assert "finish_reason=length" in failed[0]["reason"]

    refine = [e for e in events if e.get("type") == "refine"]
    assert refine, "the agent did not look again after the empty answer"
    assert refine[0]["reason"] == "empty_answer"

    # The retry gets a larger token budget than the attempt that failed.
    assert llm.budgets[1] is not None and llm.budgets[1] > llm.budgets[0]

    final = [e for e in events if e.get("type") == "final"][0]
    assert "28.4" in final["answer"]
    assert final["guardrail"]["supported"] is True


def test_an_answer_that_stays_empty_is_never_reported_as_supported() -> None:
    """No answer at all must not come back with a green guardrail badge."""
    llm = TruncatedLLM([""])
    registry = FakeRegistry([
        {"chunk_id": "c1", "paper_id": "p1", "paper_title": "Attention",
         "content": "The model reached a BLEU score of 28.4 on WMT 2014.",
         "page_no": 5},
    ])
    orch = _orchestrator(llm, registry)
    events = _run(orch)

    final = [e for e in events if e.get("type") == "final"][0]
    assert final["guardrail"]["supported"] is False
    assert "生成结果为空" in final["guardrail"]["note"]
    # An honest refusal beats an empty string.
    assert final["answer"].strip()
    assert [e for e in events if e.get("type") == "abstain"]


def test_a_failed_llm_call_is_reported_instead_of_being_swallowed() -> None:
    """An error frame from the model client must not end as an empty answer."""
    llm = TruncatedLLM([""], fail_with="ReadTimeout")
    registry = FakeRegistry([
        {"chunk_id": "c1", "paper_id": "p1", "paper_title": "Attention",
         "content": "The model reached a BLEU score of 28.4 on WMT 2014.",
         "page_no": 5},
    ])
    orch = _orchestrator(llm, registry)
    events = _run(orch)

    failed = [e for e in events if e.get("type") == "generation_failed"]
    assert failed, "the failed call was swallowed"
    assert "ReadTimeout" in failed[0]["reason"]

    final = [e for e in events if e.get("type") == "final"][0]
    assert final["guardrail"]["supported"] is False
    assert final["answer"].strip()
