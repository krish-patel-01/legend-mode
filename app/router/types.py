"""Shared routing types. Kept separate so the stages and the engine can both import
them without a cycle."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Stage = Literal[
    "override", "cache", "rules", "sticky", "embed", "classifier", "fallback"
]


class RouteDecision(BaseModel):
    """Why a request ended up on a particular model.

    Surfaced verbatim on /route/debug and in the `x_legend_route` response field, so
    every field here is meant to be read by a human tuning thresholds.
    """

    route: str
    model: str = ""  # model alias; filled in by the engine from the route table
    stage: Stage
    reason: str
    confidence: float = 1.0
    scores: dict[str, float] = Field(default_factory=dict)
    elapsed_ms: float = 0.0
    grounded: str | None = None
    """Which guard in app/guardrails.py supplied a computed fact, if any."""

    followup: str | None = None
    """Set by the sticky stage: "dispute", "weak_dispute" or "continuation"."""

    def as_meta(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "model": self.model,
            "stage": self.stage,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "grounded": self.grounded,
        }


class RouteRequest(BaseModel):
    """The subset of an incoming chat request that routing actually looks at."""

    text: str = ""
    has_tools: bool = False
    has_images: bool = False
    forced_model: str | None = None
    message_count: int = 0

    anchor_text: str = ""
    """The last *substantive* user turn — the question the thread is actually about.

    Not simply the previous message. In a run of follow-ups ("its incorrect", "nope",
    "nope") every recent turn is a follow-up, and re-routing one of those in isolation
    says nothing about which tier is handling the thread: routing "nope" on its own
    lands on the trivial tier, which is exactly the demotion the sticky stage exists to
    prevent. Skipping back to the last message that isn't a follow-up finds the real
    question instead.

    Routing stays a pure function of the messages, with no per-conversation state, so
    two clients sharing a thread cannot corrupt each other's routing.
    """
