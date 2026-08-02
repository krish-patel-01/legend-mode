"""Tests for the effort controller.

The estimator is pure — a RouteDecision in, a Plan out, no I/O — so everything it does
is testable without a model. That is deliberate: the expensive part of step 3 is the
verification it authorises, and the cheap part deciding *whether* to authorise it should
be provably right before any of it costs 28 seconds.
"""

from __future__ import annotations

import pytest

from app import effort
from app.router.types import RouteDecision

TIER = 1536  # the reasoning tier's default_max_tokens


def _decision(**kwargs) -> RouteDecision:
    base = {"route": "chat", "stage": "embed", "reason": "test", "confidence": 0.9}
    return RouteDecision(**{**base, **kwargs})


def plan_for(**kwargs) -> effort.Plan:
    text = kwargs.pop("text", "some ordinary question about things")
    grounded = kwargs.pop("grounded", False)
    override = kwargs.pop("override", None)
    return effort.estimate(
        _decision(**kwargs),
        text=text,
        tier_max_tokens=TIER,
        grounded=grounded,
        override=override,
    )


# --- levels ------------------------------------------------------------------


def test_grounded_questions_are_fast():
    """A guardrail already computed the answer; the model is only phrasing it."""
    plan = plan_for(grounded=True, route="chat")
    assert plan.level == "fast"
    assert plan.max_tokens == effort._FAST_BUDGET
    assert not plan.verify


def test_greetings_are_fast():
    plan = plan_for(route="trivial", stage="rules", text="hi there")
    assert plan.level == "fast"
    assert not plan.verify


def test_trivial_from_the_classifier_is_not_trusted():
    """The rules stage recognises a greeting by pattern, which is exact. The classifier
    saying "trivial" is a guess, and a real question misfiled that way gets answered by
    the 350M in one line — observed: "which model verifies answers in this system?" came
    back as "The question itself is the answer.\""""
    plan = plan_for(route="trivial", stage="classifier", text="which model verifies answers")
    assert plan.level == "careful"
    assert plan.retrieve


def test_a_cached_guess_is_still_a_guess():
    """A cache hit reports stage="cache"; `origin` carries the stage that decided."""
    plan = plan_for(stage="cache", origin="classifier")
    assert plan.level == "careful"
    assert "classifier" in plan.reason


def test_a_cached_confident_match_stays_standard():
    assert plan_for(stage="cache", origin="embed").level == "standard"


def test_ordinary_reasoning_is_standard_at_the_tier_budget():
    plan = plan_for(route="think", stage="rules", confidence=1.0)
    assert plan.level == "standard"
    assert plan.max_tokens == TIER
    assert not plan.verify


def test_an_unrouteable_prompt_earns_careful():
    """`fallback` means no stage recognised the prompt. That is the cheapest honest
    uncertainty signal available, and it is rare enough to be worth adjudicating."""
    plan = plan_for(stage="fallback", confidence=0.3)
    assert plan.level == "careful"
    assert plan.verify


def test_the_classifier_stage_also_counts_as_unsure():
    """Reaching stage 3 means rules and embeddings both declined, and the 350M's label
    is a guess from a model that scores at chance as a discriminator."""
    assert plan_for(stage="classifier", confidence=0.9).level == "careful"


def test_a_confident_embedding_match_is_not_adjudicated():
    plan = plan_for(stage="embed", confidence=0.88)
    assert plan.level == "standard"
    assert not plan.verify


# --- follow-up budgets: the bug this controller exists to fix -----------------


def test_a_bare_denial_gets_a_small_budget():
    """The named bug. A contentless "nope" used to arrive at a thinking model with 1536
    tokens and nothing concrete to think about, and roughly 1 turn in 6 burned the lot
    and returned empty content."""
    plan = plan_for(route="think", stage="sticky", followup="weak_dispute", confidence=0.7)
    assert plan.max_tokens == 384
    assert plan.max_tokens < TIER
    assert plan.guard_capitulation


def test_a_stated_dispute_gets_more_room_than_a_bare_one():
    weak = plan_for(route="think", stage="sticky", followup="weak_dispute")
    stated = plan_for(route="think", stage="sticky", followup="dispute")
    assert weak.max_tokens < stated.max_tokens < TIER


def test_a_correction_gets_the_most_room_of_the_three():
    """The whole problem is re-worked from the start with the new fact included."""
    kinds = ["weak_dispute", "dispute", "correction"]
    budgets = [plan_for(route="think", stage="sticky", followup=k).max_tokens for k in kinds]
    assert budgets == sorted(budgets)
    assert budgets[-1] < TIER


def test_disputes_and_corrections_authorise_adjudication():
    for kind in ("dispute", "correction"):
        plan = plan_for(route="think", stage="sticky", followup=kind)
        assert plan.level == "careful", kind
        assert plan.verify, kind
        assert plan.guard_capitulation, kind


def test_a_continuation_does_not_pay_for_adjudication():
    """Being asked to explain an answer is not evidence the answer was wrong."""
    plan = plan_for(route="think", stage="sticky", followup="continuation")
    assert not plan.verify
    assert not plan.guard_capitulation


def test_a_reasoning_tier_keeps_its_full_budget():
    """The correction that matters. A thinking model emits its <think> block first, so a
    budget below what the reasoning needs produces no answer at all rather than a short
    one — six consecutive "produced no content in 384 tokens" warnings in one eval run,
    where the complaint being fixed was 1 empty reply in 6. Measured floor: at 256 tokens
    the 1.2B emitted nothing, at 1024 it reached an answer 6 times in 8."""
    for kind in ("weak_dispute", "dispute", "correction"):
        plan = effort.estimate(
            _decision(route="think", stage="sticky", followup=kind),
            text="nope", tier_max_tokens=TIER, thinking=True,
        )
        assert plan.max_tokens == TIER, kind


def test_a_grounded_question_on_a_reasoning_tier_is_not_starved():
    """Observed: "how much is 15 percent of 240" routed to think, got the 256-token fast
    budget, and returned the exhaustion message instead of the grounded 36."""
    plan = effort.estimate(
        _decision(route="think", stage="rules"),
        text="how much is 15 percent of 240", tier_max_tokens=TIER,
        grounded=True, thinking=True,
    )
    assert plan.level == "fast"
    assert plan.max_tokens == TIER


def test_budgets_still_shrink_on_a_tier_that_does_not_reason():
    """The lever still works where it is safe — the failure mode is specific to models
    that emit a reasoning block before the answer."""
    plan = effort.estimate(
        _decision(route="chat", stage="sticky", followup="weak_dispute"),
        text="nope", tier_max_tokens=512, thinking=False,
    )
    assert plan.max_tokens == 384


def test_no_followup_budget_can_exceed_its_tier():
    """A tier with a small ceiling must not be handed a follow-up budget above it."""
    for kind in ("weak_dispute", "dispute", "correction"):
        plan = effort.estimate(
            _decision(route="trivial", followup=kind), text="nope",
            tier_max_tokens=256, grounded=False,
        )
        assert plan.max_tokens <= 256, kind


# --- caller override ---------------------------------------------------------


@pytest.mark.parametrize("level", ["fast", "standard", "careful"])
def test_an_explicit_effort_is_honoured(level):
    plan = plan_for(override=level, route="think", stage="rules")
    assert plan.level == level
    assert "caller asked" in plan.reason


def test_auto_and_blank_defer_to_the_estimate():
    for override in ("auto", "", None, "AUTO"):
        assert plan_for(override=override, route="trivial", stage="rules").level == "fast"


def test_an_unknown_effort_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown effort"):
        plan_for(override="maximum")


def test_explicit_careful_still_respects_the_followup_budget():
    """Asking for care should not undo the fix for contentless disputes."""
    plan = plan_for(override="careful", route="think", followup="weak_dispute")
    assert plan.verify
    assert plan.max_tokens == 384


# --- the retrieval gate ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "who wrote the routing rules",
        "what is the capital of Australia",
        "in what year was the Treaty of Westphalia signed",
        "how do I change the retrieval threshold",
        "which model verifies answers",
    ],
)
def test_lookup_questions_open_the_gate(text):
    assert effort.wants_retrieval(text)


@pytest.mark.parametrize(
    "text",
    [
        "hi there",
        "17 * 23",
        "write me a poem about routers",
        "give me 3 ideas for dinner",
        "```\ndef f(): pass\n```\nfind the bug",
        "thanks!",
    ],
)
def test_non_lookups_stay_shut(text):
    """Arithmetic, code and creative work are answered by computation or by the model.
    Injecting a passage into any of them is the regression the paper measured."""
    assert not effort.wants_retrieval(text)


def test_a_very_long_prompt_does_not_trigger_retrieval():
    """Past a few hundred characters the prompt carries its own context, and embedding
    it whole retrieves against an average of several topics rather than one."""
    assert not effort.wants_retrieval("what is " + "x" * 500)


def test_grounded_requests_never_retrieve():
    """The answer is already computed exactly; a document can only muddy it."""
    assert not plan_for(grounded=True, text="what is the capital of Australia").retrieve
