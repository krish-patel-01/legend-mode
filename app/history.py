"""In-memory log of recent routing decisions, for the console UI.

Bounded ring buffer, single process, no persistence — this is an observability aid
for a local dev tool, not a durable audit log. Every caller of /v1/chat/completions
(curl, the UI, any OpenAI client) shows up here, which is the point: the UI is a
window onto real traffic, not a sandbox that only sees its own requests.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from typing import Any

from pydantic import BaseModel

from app.router.types import RouteDecision

_MAX_PROMPT_CHARS = 300
_counter = itertools.count(1)


class HistoryEntry(BaseModel):
    id: int
    ts: float
    prompt: str
    model: str
    tag: str
    route: str
    stage: str
    reason: str
    confidence: float
    elapsed_ms: float
    scores: dict[str, float] = {}
    completion_tokens: int | None = None
    error: str | None = None


class HistoryStore:
    def __init__(self, maxlen: int = 200) -> None:
        self._entries: deque[HistoryEntry] = deque(maxlen=maxlen)

    def add(
        self,
        *,
        prompt: str,
        tag: str,
        decision: RouteDecision,
        completion_tokens: int | None = None,
        error: str | None = None,
    ) -> HistoryEntry:
        entry = HistoryEntry(
            id=next(_counter),
            ts=time.time(),
            prompt=prompt[:_MAX_PROMPT_CHARS],
            model=decision.model,
            tag=tag,
            route=decision.route,
            stage=decision.stage,
            reason=decision.reason,
            confidence=decision.confidence,
            elapsed_ms=decision.elapsed_ms,
            scores=decision.scores,
            completion_tokens=completion_tokens,
            error=error,
        )
        self._entries.append(entry)
        return entry

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        items = list(self._entries)[-limit:]
        items.reverse()
        return [e.model_dump() for e in items]
