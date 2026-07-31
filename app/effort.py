"""Decide how much work a request is worth *before* doing it (ROADMAP step 3).

The controller exists because verification costs ~28 s on this machine. That single
measurement rules out every design that checks every answer, so the question stops being
"should we verify?" and becomes "which few requests earn it?".

Everything here is computed from signals the cascade already produced — which rule fired,
which stage answered, the confidence it carried, whether a guardrail grounded the
question, whether this is a follow-up. No extra model call, so the estimate itself is
free and can be wrong without costing anything.

Three levels, each fixing a token budget and a permission to adjudicate:

    fast      small budget, no adjudication   greetings, identity, grounded facts
    standard  the tier's own budget           ordinary chat and reasoning
    careful   tuned budget, adjudication on   disputes, corrections, unrouteable prompts

**The budget is the part that fixes a live bug.** Before this, every request got its
tier's fixed budget: 1536 tokens on the reasoning tier. A bare "nope" therefore arrived
at a thinking model with 1536 tokens and nothing concrete to think about, and roughly 1
turn in 6 burned the lot and returned empty content — papered over by a fallback message
in app/api.py. A contentless denial does not need 1536 tokens; it needs one sentence
asking what the user thinks is wrong.

**`careful` is a permission, not a promise.** What adjudication is actually possible
depends on which model answered, and app/adjudicate.py decides that. See its docstring
for why a 1.2B answer has no available critic on this hardware.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.router.types import RouteDecision

Level = Literal["fast", "standard", "careful"]
LEVELS: tuple[Level, ...] = ("fast", "standard", "careful")

# Budgets for the follow-up kinds, in tokens. These are absolute rather than a fraction
# of the tier default because the reply length wanted is a property of the *turn*, not of
# the model: a dispute reply is two or three sentences whichever tier writes it.
#
#   weak_dispute  a bare "no" carries no claim. The wanted reply is one sentence asking
#                 which part is wrong (see persona.DISPUTE_NOTE), so the budget is sized
#                 for that. This is the 1-in-6 empty-reply bug named above.
#   dispute       a stated "that's wrong" gives something concrete to re-check, so there
#                 is real work to do — but still a short answer at the end of it.
#   correction    the user supplied a missed fact and the whole problem gets re-worked
#                 from the start, which needs the most room of the three.
#
# None of these is the tier default (1536). That number was measured as *too much* for
# this shape of turn, not too little: raising it to 4096 on two hard puzzles bought a
# slower wrong answer, and the box word problem finishes inside 935.
_FOLLOWUP_BUDGET = {
    "weak_dispute": 384,
    "continuation": 768,
    "dispute": 768,
    "correction": 1024,
}

# `fast` means the model is phrasing something already decided — a greeting, or a value
# app/guardrails.py computed exactly. Long replies there are pure latency.
_FAST_BUDGET = 256

# Stages that mean "nothing recognised this prompt". `classifier` counts: reaching stage 3
# at all means rules and embeddings both declined, and the 350M's label is a guess from a
# model with 50%-accuracy discrimination behind it (see the critic table in models.yaml).
#
# Uncertainty is read off the *stage*, not off `confidence`. The confidences are not on a
# common scale — the embedding stage reports a raw cosine, the classifier a flat 0.5, the
# fallback 0.3 — so one threshold across all three would mostly measure which stage
# answered, and badly. Each stage already has its own acceptance thresholds
# (`Thresholds` in app/config.py); second-guessing them here would create a second place
# to tune that disagrees with the first.
_UNSURE_STAGES = frozenset({"fallback", "classifier"})

# Stages whose `trivial` verdict can be trusted without further checking. The rules stage
# recognises greetings and identity questions by pattern, which is exact. The classifier
# saying "trivial" is a guess, and a real question misfiled as trivial is answered by the
# 350M in one line — observed: "which model verifies answers in this system?" came back as
# "The question itself is the answer."
_DETERMINISTIC_STAGES = frozenset({"rules", "override", "sticky"})

# Questions that plausibly have an answer written down somewhere, as opposed to ones that
# want reasoning, arithmetic or prose. Deliberately loose: this only decides whether to
# spend ~15 ms embedding the prompt, and the real gate is the retrieval score threshold
# in app/retrieval/. The paper behind this plan found indiscriminate retrieval *hurt*
# GPQA by 5.0 points, so being loose here is only safe because something stricter follows.
_LOOKUP = re.compile(
    r"^\s*(?:who|what|when|where|which|whose|why)\b.{0,200}$"
    r"|\b(?:capital|population|founded|invented|discovered|located|headquarter"
    r"|born|died|signed|released|version|default)\b"
    r"|\bin what year\b|\bhow (?:do|does) (?:i|you|it)\b"
    r"|\btell me about\b|\bwhat (?:is|are|was|were) the\b",
    re.IGNORECASE,
)

# Shapes that should never trigger retrieval however lookup-ish they read. Arithmetic and
# code are answered by computation, not by documents, and injecting a passage into either
# is the failure mode the paper measured.
_NOT_LOOKUP = re.compile(
    r"```|\bwrite (?:me )?(?:a|an|some)\b|\bdraft\b|\bbrainstorm\b|\bgive me \d"
    r"|\bidea(?:s)? for\b|\bpoem|\bstory\b|\bjoke\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Plan:
    """What this request is allowed to spend, and on what."""

    level: Level
    max_tokens: int
    reason: str

    verify: bool = False
    """Adjudication is permitted. Whether it is *possible* is app/adjudicate.py's call."""

    guard_capitulation: bool = False
    """Check the reply against the previous one for a flip under contentless pressure.

    Free — a numeric comparison, no generation — so it is on wherever it is meaningful
    rather than being rationed like verification.
    """

    retrieve: bool = False
    """Consult the local corpus before answering."""

    def as_meta(self) -> dict[str, object]:
        return {"level": self.level, "max_tokens": self.max_tokens, "why": self.reason}


def estimate(
    decision: RouteDecision,
    *,
    text: str,
    tier_max_tokens: int,
    grounded: bool = False,
    override: str | None = None,
    retrieval_text: str | None = None,
) -> Plan:
    """Pick an effort level for a request that has been routed but not yet answered.

    `override` is the caller's `effort` field: "fast", "standard", "careful", or "auto"
    / None to let this decide. An explicit level still gets a sensible budget, so a
    caller can ask for care without also having to know the token numbers.

    `retrieval_text` is what to look up when that differs from what to answer — on a
    follow-up the turn itself is "explain that", and the question the thread is actually
    about is the anchor.
    """
    if override and override.lower() not in {"auto", ""}:
        level = override.lower()
        if level not in LEVELS:
            raise ValueError(f"unknown effort {override!r}; expected one of {LEVELS} or 'auto'")
        return _explicit(  # type: ignore[arg-type]
            level, tier_max_tokens, decision, retrieval_text or text, grounded
        )

    # A guardrail already computed the answer exactly. The model's only remaining job is
    # to say it in a sentence, and any budget beyond that is spent inventing a second
    # opinion — which is measurably what it does: the 350M overrode the supplied value on
    # 5 of 6 discount questions, which is why app/api.py corrects contradictions at all.
    # Nothing is looked up either: a document can only muddy an exact value.
    if grounded:
        return Plan(
            level="fast",
            max_tokens=_FAST_BUDGET,
            reason="a guardrail computed this exactly; the model is only phrasing it",
        )

    retrieve = wants_retrieval(retrieval_text if retrieval_text is not None else text)

    if decision.followup:
        return _followup_plan(decision.followup, tier_max_tokens, retrieve)

    # `origin` is set only on a cache hit and carries the stage that actually decided.
    stage = decision.origin or decision.stage

    if decision.route == "trivial" and stage in _DETERMINISTIC_STAGES:
        return Plan(
            level="fast",
            max_tokens=min(_FAST_BUDGET, tier_max_tokens),
            retrieve=retrieve,
            reason="greeting, acknowledgement or identity question",
        )

    if stage in _UNSURE_STAGES:
        # The cascade did not recognise this. That is the cheapest honest uncertainty
        # signal available, and it is rare — rules and embeddings between them resolve
        # most traffic — so it is affordable to adjudicate. It is also the one case where
        # cross-model checking is *possible*: an unrecognised prompt lands on a small
        # tier, and the 1.2B can judge a 350M answer.
        return Plan(
            level="careful",
            max_tokens=tier_max_tokens,
            verify=True,
            retrieve=retrieve,
            reason=f"no stage recognised this prompt (decided by {stage})",
        )

    return Plan(
        level="standard",
        max_tokens=tier_max_tokens,
        retrieve=retrieve,
        reason=f"routed confidently to {decision.route!r} by {stage}",
    )


def _followup_plan(kind: str, tier_max_tokens: int, retrieve: bool) -> Plan:
    budget = min(_FOLLOWUP_BUDGET.get(kind, tier_max_tokens), tier_max_tokens)

    if kind == "weak_dispute":
        return Plan(
            level="standard",
            max_tokens=budget,
            guard_capitulation=True,
            reason="a bare denial names no claim; short reply asking what is wrong",
        )
    if kind == "continuation":
        return Plan(
            level="standard",
            max_tokens=budget,
            retrieve=retrieve,
            reason="continuing an answer that already exists",
        )
    if kind in {"dispute", "correction"}:
        return Plan(
            level="careful",
            max_tokens=budget,
            verify=True,
            guard_capitulation=True,
            retrieve=retrieve,
            reason=f"user {kind}; re-checking the previous answer",
        )
    return Plan(level="standard", max_tokens=budget, reason=f"follow-up ({kind})")


def _explicit(
    level: Level, tier_max_tokens: int, decision: RouteDecision, text: str, grounded: bool
) -> Plan:
    if level == "fast":
        return Plan(
            level="fast",
            max_tokens=min(_FAST_BUDGET, tier_max_tokens),
            reason="caller asked for fast",
        )
    if level == "careful":
        budget = _FOLLOWUP_BUDGET.get(decision.followup or "", tier_max_tokens)
        return Plan(
            level="careful",
            max_tokens=min(budget, tier_max_tokens),
            verify=True,
            guard_capitulation=bool(decision.followup),
            retrieve=not grounded and wants_retrieval(text),
            reason="caller asked for careful",
        )
    return Plan(
        level="standard",
        max_tokens=tier_max_tokens,
        retrieve=not grounded and wants_retrieval(text),
        reason="caller asked for standard",
    )


def wants_retrieval(text: str) -> bool:
    """Whether this prompt is worth looking up in the corpus at all.

    A first, cheap gate — see the note on `_LOOKUP`. Passing it does not mean anything
    gets injected; the retrieved chunks still have to clear a similarity threshold.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 400:
        return False
    if _NOT_LOOKUP.search(stripped):
        return False
    return bool(_LOOKUP.search(stripped))
