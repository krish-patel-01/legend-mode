"""Tests for adjudication.

The parts that matter here are the ones that decide whether to *replace* a user-visible
answer, so they are tested for restraint as much as for reach: `operative_number` must
return None whenever the reply is ambiguous, and the flip guard must sit out rather than
guess. A guard that fires on a misread is worse than no guard.
"""

from __future__ import annotations

import pytest

from app import adjudicate
from app.effort import Plan

# --- extracting the answer's number ------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("4", 4.0),
        ("The answer is 4.", 4.0),
        ("You have four boxes in total.", 4.0),
        ("So the total is 26 legs.", 26.0),
        ("2 + 2 = 4", 4.0),
        ("That comes to 391.", 391.0),
        ("It's 1,648.", 1648.0),
        ("The result is exactly 30.", 30.0),
    ],
)
def test_operative_number_finds_the_assertion(reply, expected):
    assert adjudicate.operative_number(reply) == expected


def test_the_last_conclusion_wins_over_working():
    """A reply that reasons and then concludes asserts the number at the end."""
    reply = "Two boxes, and each is one box, so the answer is 4."
    assert adjudicate.operative_number(reply) == 4.0


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "I'm not sure about that.",
        "You start with 2 boxes and add 1 to each, then consider 3 more cases.",
    ],
)
def test_operative_number_declines_when_ambiguous(reply):
    """Several candidate numbers and no phrase marking one as the answer -> abstain."""
    assert adjudicate.operative_number(reply) is None


def test_a_single_number_needs_no_cue():
    assert adjudicate.operative_number("Paris has 2 airports of note") == 2.0


# --- the capitulation guard --------------------------------------------------


def test_a_changed_number_is_detected():
    assert adjudicate.capitulated("The answer is 4.", "The answer is 3.") == (4.0, 3.0)


def test_an_unchanged_answer_is_not_a_flip():
    assert adjudicate.capitulated("The answer is 4.", "It is still 4, because …") is None


def test_prose_only_replies_are_not_judged():
    """No numbers to compare, so nothing to compare exactly — and prose is never scored
    against prose anywhere in this project."""
    assert adjudicate.capitulated("Four boxes.", "Which part do you disagree with?") is None


def test_unstable_reply_names_both_candidates_and_picks_neither():
    text = adjudicate.unstable_reply(4.0, 3.0)
    assert "4" in text and "3" in text
    assert "won't pick one" in text


def test_unstable_reply_does_not_print_a_trailing_zero():
    assert "26.0" not in adjudicate.unstable_reply(26.0, 10.0)


# --- parsing the critic's verdict --------------------------------------------


def test_the_think_block_is_skipped():
    """An earlier version of this measurement parsed the <think> block instead of the
    verdict: 44% accuracy and 8 false alarms from a model that actually scores 100%."""
    reply = "<think>Hmm, is this CORRECT? Let me check. 2+2=4, the answer said 5.</think>\nINCORRECT"
    assert adjudicate.parse_verdict(reply) == "incorrect"


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("CORRECT", "correct"),
        ("incorrect", "incorrect"),
        ("The answer is not correct.", "incorrect"),
        ("UNSURE", "unsure"),
        ("I cannot tell from this.", "unsure"),
        ("", "unsure"),
    ],
)
def test_verdict_parsing(reply, expected):
    assert adjudicate.parse_verdict(reply) == expected


def test_incorrect_is_never_read_as_correct():
    """"INCORRECT" contains "CORRECT"; a naive substring check inverts every rejection."""
    assert adjudicate.parse_verdict("INCORRECT") != "correct"


# --- orchestration -----------------------------------------------------------


class _Spec:
    def __init__(self, alias: str, default_max_tokens: int = 512) -> None:
        self.alias = alias
        self.default_max_tokens = default_max_tokens


class _Settings:
    verify_enabled = True
    self_consistency = False


class _Recorder:
    """Stands in for generation. Records every call so the one-repair cap is testable."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str | None, str | None]] = []

    async def __call__(self, note, target) -> str:
        self.calls.append((note, getattr(target, "alias", None)))
        return self.replies.pop(0) if self.replies else ""


async def _run(plan, *, answer, previous="", answered_by="think", critic="think",
               regenerate=None, settings=None):
    return await adjudicate.run(
        question="how many boxes?",
        answer=answer,
        previous=previous,
        plan=plan,
        answered_by=_Spec(answered_by),
        critic_spec=_Spec(critic) if critic else None,
        client=None,
        settings=settings or _Settings(),
        regenerate=regenerate or _Recorder(),
    )


_FLIP_PLAN = Plan(level="careful", max_tokens=384, reason="t", guard_capitulation=True)
_VERIFY_PLAN = Plan(level="careful", max_tokens=512, reason="t", verify=True)
_NOTHING = Plan(level="standard", max_tokens=512, reason="t")


async def test_a_flip_is_re_worked_once():
    regen = _Recorder("Re-checked: it is 4.")
    out = await _run(_FLIP_PLAN, answer="It's 3.", previous="It's 4.", regenerate=regen)
    assert out.repaired and out.verdict == "flip"
    assert out.content == "Re-checked: it is 4."
    assert len(regen.calls) == 1  # at most one repair, always


async def test_a_third_answer_abstains_instead_of_picking():
    """Three different answers in three attempts. Nothing here can break that tie, and
    picking one anyway is the confident wrongness this project exists to avoid."""
    regen = _Recorder("Actually it is 7.")
    out = await _run(_FLIP_PLAN, answer="It's 3.", previous="It's 4.", regenerate=regen)
    assert out.verdict == "unstable"
    assert "4" in (out.content or "") and "7" in (out.content or "")


async def test_no_flip_means_no_generation_at_all():
    regen = _Recorder()
    out = await _run(_FLIP_PLAN, answer="Still 4.", previous="It's 4.", regenerate=regen)
    assert out.content is None and regen.calls == []


async def test_a_reasoning_tier_answer_reports_that_it_has_no_critic():
    """The roadmap's two critic rules — always the 1.2B, never the model that answered —
    intersect at exactly one case on this hardware, and it is not this one. The gap is
    reported rather than papered over with self-verification."""
    out = await _run(_VERIFY_PLAN, answer="4 boxes.", answered_by="think", critic="think")
    assert out.skipped is not None and "no independent critic" in out.skipped
    assert out.content is None


async def test_verification_is_skipped_when_disabled():
    settings = _Settings()
    settings.verify_enabled = False
    out = await _run(_VERIFY_PLAN, answer="4.", answered_by="general", settings=settings)
    assert out.verdict is None and out.skipped is None


async def test_nothing_runs_when_the_plan_authorises_nothing():
    out = await _run(_NOTHING, answer="4.", previous="3.")
    assert out.as_meta() is None


async def test_self_consistency_abstains_on_disagreement():
    settings = _Settings()
    settings.self_consistency = True
    regen = _Recorder("The answer is 3.")
    out = await _run(
        _VERIFY_PLAN, answer="The answer is 4.", answered_by="think", critic="think",
        regenerate=regen, settings=settings,
    )
    assert out.verdict == "unstable"
    assert "4" in (out.content or "") and "3" in (out.content or "")


async def test_self_consistency_is_off_by_default():
    """It doubles latency on the slowest tier, so it stays a measured knob rather than
    a default."""
    regen = _Recorder("The answer is 3.")
    out = await _run(_VERIFY_PLAN, answer="The answer is 4.", regenerate=regen)
    assert regen.calls == []
    assert out.verdict is None


# --- previous turn extraction ------------------------------------------------


def test_previous_assistant_reply_takes_the_most_recent():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "nope"},
    ]
    assert adjudicate.previous_assistant_reply(messages) == "a2"


def test_previous_assistant_reply_is_empty_on_the_first_turn():
    assert adjudicate.previous_assistant_reply([{"role": "user", "content": "hi"}]) == ""
