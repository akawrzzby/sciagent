"""In-process publish/subscribe bus backing the Server-Sent-Events streams.

The backend emits fine grained events (agent thinking, tool calls, token deltas,
parse progress) on named channels.  Every connected browser session subscribes to
``session:<id>`` and receives the events in order.

Design notes
------------
* A single :class:`EventBus` instance lives in the FastAPI lifespan.
* Each subscriber owns an ``asyncio.Queue``; publishing never blocks the producer
  (queues are bounded and drop the oldest event when a slow client stalls).
* Event payloads are plain JSON-serialisable dicts.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from app.core.logging import get_logger

logger = get_logger(__name__)

# Bound on the per-subscriber queue.  Slow clients lose the oldest frames
# instead of stalling the producer.
_QUEUE_MAXSIZE = 1000


@dataclass(slots=True)
class Event:
    """A single stream event."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_sse(self) -> str:
        """Serialise to the Server-Sent-Events wire format."""
        import json

        body = json.dumps({"type": self.type, "id": self.id, "ts": self.ts, **self.data},
                          ensure_ascii=False)
        return f"id: {self.id}\nevent: {self.type}\ndata: {body}\n\n"


class EventBus:
    """A tiny multi-channel publish/subscribe hub."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[Event]]] = {}
        self._history: dict[str, list[Event]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ publish
    async def publish(self, channel: str, event_type: str, **data: Any) -> Event:
        """Publish an event to every subscriber of ``channel``."""
        ev = Event(type=event_type, data=data)
        async with self._lock:
            queues = list(self._subscribers.get(channel, ()))
            history = self._history.setdefault(channel, [])
            history.append(ev)
            # Keep a bounded replay buffer so a client that connects late can
            # still catch up on the first few frames of a long-running job.
            if len(history) > 200:
                del history[:-200]

        for q in queues:
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                # Drop the oldest frame and retry once: better to lose history
                # than to stall the agent pipeline.
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(ev)
        return ev

    # ------------------------------------------------------------------ subscribe
    async def subscribe(self, channel: str, replay: int = 0) -> AsyncIterator[Event]:
        """Yield events for ``channel`` until the consumer disconnects."""
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        async with self._lock:
            self._subscribers.setdefault(channel, set()).add(q)
            backlog = list(self._history.get(channel, ())[-replay:]) if replay else []

        try:
            for ev in backlog:
                yield ev
            while True:
                ev = await q.get()
                yield ev
        finally:
            async with self._lock:
                subs = self._subscribers.get(channel)
                if subs:
                    subs.discard(q)
                    if not subs:
                        self._subscribers.pop(channel, None)

    def subscriber_count(self, channel: str) -> int:
        return len(self._subscribers.get(channel, ()))


# ---------------------------------------------------------------------- helpers
def session_channel(session_id: str) -> str:
    """Channel name for a chat/agent session."""
    return f"session:{session_id}"


def paper_channel(paper_id: str) -> str:
    """Channel name for a document processing job."""
    return f"paper:{paper_id}"


def job_channel(job_id: str) -> str:
    """Channel name for a generic background job."""
    return f"job:{job_id}"


_bus: EventBus | None = None


def get_bus() -> EventBus:
    """Return the process-wide event bus singleton."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
