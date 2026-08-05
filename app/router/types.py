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

    origin: Stage | None = None
    """The stage that originally made this decision, set only on a cache hit.

    `stage` says how the decision was *retrieved*; this says how it was *made*. The
    effort controller needs the second: a cached guess from the stage-3 classifier is
    still a guess, and without this it would read as a confident `cache` hit.
    """
    confidence: float = 1.0
    scores: dict[str, float] = Field(default_factory=dict)
    elapsed_ms: float = 0.0
    grounded: str | None = None
    """Which guard in app/guardrails.py supplied a computed fact, if any."""

    followup: str | None = None
    """Set by the sticky stage: "dispute", "weak_dispute", "continuation" or "correction"."""

    effort: dict[str, Any] | None = None
    """The plan from app/effort.py: level, token budget, and why that level was picked."""

    adjudicated: dict[str, Any] | None = None
    """What app/adjudicate.py did, if anything — verdict, whether it repaired, or why it
    could not run. The "why it could not run" case is reported rather than hidden: with
    two models there is no independent critic for a reasoning-tier answer, and a system
    that silently skipped the check would look identical to one that passed it."""

    remembered: str | None = None
    """A fact stored from this turn, if the user stated one. Surfaced so capture is
    visible rather than silent — a memory the user cannot see forming is one they cannot
    correct."""

    retrieved: list[str] | None = None
    """Citations for corpus chunks injected into the prompt, if any cleared the score
    threshold. Empty is not the same as None: None means retrieval never ran."""

    tools: dict[str, Any] | None = None
    """What app/tools/ did: which families the gate allowed, what was called, how long
    each took. Present only when the gate opened, so an ordinary reply's metadata is not
    padded with a permanently-empty field — and absent is itself the useful signal, since
    the gate declining is the common and correct case."""

    def as_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "route": self.route,
            "model": self.model,
            "stage": self.stage,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "grounded": self.grounded,
            "followup": self.followup,
            "effort": self.effort,
        }
        # Only present when something happened, so a plain chat reply's metadata stays
        # readable in the console instead of carrying two permanently-null fields.
        if self.adjudicated is not None:
            meta["adjudicated"] = self.adjudicated
        if self.retrieved is not None:
            meta["retrieved"] = self.retrieved
        if self.remembered is not None:
            meta["remembered"] = self.remembered
        if self.tools is not None:
            meta["tools"] = self.tools
        return meta


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
