"""Client for the OpenAI-compatible vLLM endpoint (Qwen3.8-27B).

Features that matter for the competition rubric:

* **Native function calling** – the model returns ``tool_calls`` and we execute
  them against the local tool registry (vLLM is started with
  ``--enable-auto-tool-choice``).
* **Reasoning passthrough** – Qwen3.8 emits a separate ``reasoning`` field; we
  surface it so the UI can visualise the agent's thinking process.
* **Streaming** – token level deltas are forwarded to the SSE bus.
* **Audit trail** – every call is persisted (prompt, tools, response, tokens,
  latency) so the final round can show a real "prompt audit record".
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- types
@dataclass(slots=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "ToolCall":
        args = raw.get("function", {}).get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        return cls(id=raw.get("id", uuid.uuid4().hex[:8]),
                   name=raw.get("function", {}).get("name", ""),
                   arguments=args or {})


@dataclass(slots=True)
class LLMResponse:
    """Normalised result of one chat completion."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# --------------------------------------------------------------------------- client
class LLMClient:
    """Thin async wrapper around the vLLM OpenAI-compatible API."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: int | None = None) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.timeout = timeout or settings.llm_timeout
        self._client: httpx.AsyncClient | None = None

    # -------------------------------------------------------------- lifecycle
    async def __aenter__(self) -> "LLMClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=10),
            headers=self._headers(),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10),
                headers=self._headers(),
            )
        return self._client

    # -------------------------------------------------------------- health
    async def health(self) -> dict[str, Any]:
        """Probe the endpoint; used by /api/health and the readiness probe."""
        try:
            r = await self.client.get("/models", timeout=10)
            r.raise_for_status()
            data = r.json()
            ids = [m.get("id") for m in data.get("data", [])]
            return {"ok": True, "models": ids, "base_url": self.base_url}
        except Exception as exc:  # pragma: no cover - network dependent
            return {"ok": False, "error": str(exc)[:300], "base_url": self.base_url}

    # -------------------------------------------------------------- chat
    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Iterable[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        session_id: str = "",
        purpose: str = "",
        thinking: bool | None = None,
    ) -> LLMResponse:
        """Run a non-streaming chat completion and return a normalised response."""
        settings = get_settings()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }
        if thinking is not None:
            # Qwen3-style chat templates switch the reasoning channel with
            # ``enable_thinking``; vLLM forwards it through extra body.
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = tool_choice or "auto"
        if stop:
            payload["stop"] = stop
        if response_format:
            payload["response_format"] = response_format

        t0 = time.perf_counter()
        try:
            r = await self.client.post("/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            logger.warning("LLM call failed purpose=%s err=%s", purpose, exc)
            return LLMResponse(content="", finish_reason="error",
                               latency_ms=int((time.perf_counter() - t0) * 1000),
                               raw={"error": str(exc)[:500]})

        latency = int((time.perf_counter() - t0) * 1000)
        msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
        usage = data.get("usage", {}) or {}
        resp = LLMResponse(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning") or msg.get("reasoning_content") or "",
            tool_calls=[ToolCall.from_raw(tc) for tc in (msg.get("tool_calls") or [])],
            finish_reason=(data.get("choices") or [{}])[0].get("finish_reason", ""),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency,
            raw=data,
        )
        await self._audit(session_id, purpose, messages, payload, resp)
        return resp

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Iterable[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str = "",
        purpose: str = "",
        thinking: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a chat completion.

        Yields normalised frames:
        ``{"type": "reasoning"|"delta"|"tool_call"|"done", ...}``.
        """
        settings = get_settings()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = tool_choice or "auto"

        t0 = time.perf_counter()
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        try:
            async with self.client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(body)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("usage"):
                        usage = chunk["usage"]

                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}

                    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        yield {"type": "reasoning", "text": reasoning}

                    content = delta.get("content")
                    if content:
                        content_parts.append(content)
                        yield {"type": "delta", "text": content}

                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                            yield {"type": "tool_call", "index": idx,
                                   "name": fn["name"], "arguments": ""}
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                            yield {"type": "tool_call", "index": idx,
                                   "name": slot["name"], "arguments": fn["arguments"]}
        except Exception as exc:
            logger.warning("LLM stream failed purpose=%s err=%s", purpose, exc)
            yield {"type": "error", "message": str(exc)[:400]}
            return

        calls = [ToolCall.from_raw({"id": v["id"], "function": {"name": v["name"],
                                                                 "arguments": v["arguments"]}})
                 for v in tool_acc.values() if v["name"]]
        final = LLMResponse(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=calls,
            finish_reason="tool_calls" if calls else "stop",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        await self._audit(session_id, purpose, messages, payload, final)
        yield {"type": "done", "response": final}

    # -------------------------------------------------------------- helpers
    async def complete_json(
        self,
        system: str,
        user: str,
        *,
        schema_hint: str = "",
        session_id: str = "",
        purpose: str = "json",
        temperature: float = 0.1,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Ask for a JSON object and parse it robustly.

        Returns ``(parsed_or_None, raw_text)``.  The helper strips markdown code
        fences and repairs common model mistakes (single quotes, trailing commas).

        ``thinking`` disables the reasoning channel.  A structured answer over a
        large prompt can spend the whole token budget reasoning and then emit
        nothing, which parses as "the model did not return JSON".
        """
        instruction = (
            "Respond with a single JSON object and nothing else. "
            "Do not wrap it in markdown fences. "
            "Use double quotes for keys and strings."
        )
        if schema_hint:
            instruction += f"\nThe object must conform to this shape:\n{schema_hint}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{instruction}\n\n{user}"},
        ]
        resp = await self.chat(messages, temperature=temperature, max_tokens=max_tokens,
                               session_id=session_id, purpose=purpose, thinking=thinking)
        return extract_json(resp.content), resp.content

    async def _audit(self, session_id: str, purpose: str, messages: list[dict[str, Any]],
                     payload: dict[str, Any], resp: LLMResponse) -> None:
        """Persist the prompt audit record (best effort, never breaks the call)."""
        try:
            from app.storage.db import session_scope
            from app.storage.models import AuditRecord

            system_prompt = next((m.get("content", "") for m in messages
                                  if m.get("role") == "system"), "")
            user_prompt = next((m.get("content", "") for m in reversed(messages)
                                if m.get("role") == "user"), "")
            async with session_scope() as db:
                db.add(AuditRecord(
                    session_id=session_id or "",
                    purpose=purpose or "chat",
                    system_prompt=system_prompt if isinstance(system_prompt, str) else "",
                    user_prompt=user_prompt if isinstance(user_prompt, str) else "",
                    tools=payload.get("tools", []) or [],
                    response=resp.content[:20000],
                    reasoning=resp.reasoning[:8000],
                    tool_calls=[{"name": tc.name, "arguments": tc.arguments} for tc in resp.tool_calls],
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    latency_ms=resp.latency_ms,
                    ok=resp.finish_reason != "error",
                    error=str(resp.raw.get("error", "")) if resp.finish_reason == "error" else "",
                ))
        except Exception as exc:  # pragma: no cover - auditing must not break flow
            logger.debug("audit persist failed: %s", exc)


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from a model response."""
    if not text:
        return None
    candidate = text.strip()

    # Strip markdown fences.
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        if candidate.endswith("```"):
            candidate = candidate[: -3]
        candidate = candidate.strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()

    # Try the raw payload first.
    for attempt in (candidate, _first_brace_block(candidate)):
        if not attempt:
            continue
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Repair common deviations.
    repaired = (candidate.replace("'", '"')
                        .replace("True", "true").replace("False", "false")
                        .replace("None", "null"))
    repaired = _strip_trailing_commas(repaired)
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Last resort: pull the outermost {...} block.
    block = _first_brace_block(candidate)
    if block:
        try:
            parsed = json.loads(_strip_trailing_commas(block.replace("'", '"')))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def _first_brace_block(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    out, in_str, escape = [], False, False
    for i, ch in enumerate(text):
        if in_str:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            continue
        if ch == ",":
            rest = text[i + 1 :]
            stripped = rest.lstrip()
            if stripped.startswith("}"):
                continue
        out.append(ch)
    return "".join(out)


_client: LLMClient | None = None
_client_signature: tuple[str, str, str] | None = None


def get_llm_client() -> LLMClient:
    """Return the process-wide LLM client, rebuilt when the active model changes.

    The active endpoint can be switched from the interface, so the client is
    keyed by the endpoint it was built for: a stale client would keep answering
    from the model the operator just replaced.
    """
    global _client, _client_signature
    from app.llm import providers

    profile = providers.active()
    signature = (profile.base_url, profile.model, profile.api_key) if profile else None
    if _client is None or signature != _client_signature:
        if profile is not None:
            _client = LLMClient(base_url=profile.base_url, model=profile.model,
                                api_key=profile.api_key)
        else:
            _client = LLMClient()
        _client_signature = signature
    return _client
