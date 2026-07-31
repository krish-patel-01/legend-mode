"""Checking an answer after it has been produced (ROADMAP step 3).

**Read this before adding a verification pass anywhere else.** The roadmap sets two
rules for the critic: it is always the 1.2B and never the 350M, and it is never the model
that produced the answer. With a two-model palette those two rules intersect at exactly
one case — *the 1.2B judging a 350M answer* — and that is the only cross-model check this
file will ever be able to perform. An answer from the reasoning tier has no independent
critic available on this hardware, and the file says so in the response metadata rather
than quietly falling back to self-verification, which is the thing the measurements
warned against:

    350M                 50% accurate as a critic (chance), rubber-stamped 8/8 wrong
    Qwen3.5-0.8B (fast)  56%, rubber-stamped 7/8
    LFM2.5-1.2B          100% on the 14/16 it finished, 0 false alarms, 28 s

So there are three mechanisms here, in descending order of how often they can run:

1. **The capitulation guard** — free, deterministic, no generation. Catches the model
   changing a numeric answer under pressure that carried no information.
2. **Cross-model verification** — the 1.2B judging a chat-tier answer. ~28 s, so it is
   rationed by app/effort.py to prompts the cascade could not classify at all.
3. **Self-consistency** — two samples from the same model, compared *numerically*. Not
   self-verification: no model judges anything, two numbers are compared exactly. Off by
   default because it doubles latency on the slowest tier; it is a knob for the tuning
   phase, with the eval harness to say whether it pays.

At most one repair, then stop. Unbounded critique oscillates — A says 4, B says 2, A says
2 — and a small critic is exactly noisy enough to make that likely. If the repair does not
settle it, the answer is uncertain and saying so is better than picking one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.backends.ollama import OllamaError

log = logging.getLogger(__name__)

_WORD_NUMBERS = {
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0, "eleven": 11.0,
    "twelve": 12.0, "twenty": 20.0, "thirty": 30.0, "forty": 40.0, "fifty": 50.0,
    "hundred": 100.0, "thousand": 1000.0,
}
_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
_WORD_ALT = "|".join(_WORD_NUMBERS)

# A phrase that marks the number as *the answer* rather than a number used along the way.
# Restricted to a short gap so "the answer is bigger than the 5 boxes we started with"
# doesn't hand back 5.
# `=` sits outside the \b…\b group on purpose: it is not a word character, so \b=\b
# never matches and "2 + 2 = 4" fell through to the ambiguous path and returned nothing.
_ANSWER_CUE = re.compile(
    rf"(?:\b(?:answer|total|result|altogether|in total|equals?|comes? to|is|are)\b|=)"
    rf"[^\d\w]{{0,4}}(?:exactly\s+|about\s+|roughly\s+)?({_NUMBER}|{_WORD_ALT})\b",
    re.IGNORECASE,
)
_ANY_NUMBER = re.compile(rf"\b({_NUMBER})\b|\b({_WORD_ALT})\b", re.IGNORECASE)

# The critic's own reply. The 1.2B emits a <think> block first, and an earlier version of
# this measurement parsed *that* instead of the verdict — 44% accuracy and 8 false alarms
# from a model that actually scores 100%. Always read the text after the block.
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_VERDICT = re.compile(r"\b(NOT\s+CORRECT|INCORRECT|CORRECT|UNSURE)\b", re.IGNORECASE)

# Ends on what to judge, not on a verdict word. A prompt that ends on a quotable answer
# gets echoed back verbatim by models this size — twice observed, see app/persona.py.
_CRITIC_SYSTEM = (
    "You are checking another assistant's answer to a question. Work the question out "
    "yourself first, then compare. Your reply is exactly one word: CORRECT, INCORRECT or "
    "UNSURE. Judge only whether the answer is factually right, never how it is worded."
)

# Measured, not chosen. The critic reasons before it judges, and below ~2048 tokens it
# spends the whole budget in the <think> block and never reaches the verdict. Over 8
# question/answer pairs (4 right, 4 wrong):
#
#   think off, 64 tokens    38% accurate, 4/4 wrong answers waved through,   7.4 s
#   think off, 192 tokens   50% accurate, 4/4 wrong answers waved through,  21.4 s
#   think on,  512 tokens   25% accurate, verdict emitted on only 2 of 8,   45.1 s
#   think on, 1024 tokens   75% accurate, verdict emitted on 6 of 8,        24.9 s
#   think on, 2048 tokens   88% accurate, verdict emitted on 8 of 8,        26.7 s
#
# Two things follow. Turning reasoning off does not buy a cheap critic — it buys the
# 350M's behaviour, waving through every wrong answer. And a budget that looks generous
# can be below the floor: 512 tokens produced "unsure" on 6 of 8 pairs and charged 45 s
# for it, which reads exactly like a working verifier that never fires.
_CRITIC_MAX_TOKENS = 2048


@dataclass
class Adjudication:
    """What checking produced. `content` is None when the original reply stands."""

    content: str | None = None
    verdict: str | None = None
    """"correct" | "incorrect" | "unsure" | "flip" | "unstable", or None if nothing ran."""

    repaired: bool = False
    skipped: str | None = None
    """Why no check ran, when one was permitted. Surfaced so the gap is visible."""

    notes: list[str] = field(default_factory=list)

    def as_meta(self) -> dict[str, Any] | None:
        if self.verdict is None and self.skipped is None:
            return None
        meta: dict[str, Any] = {}
        if self.verdict:
            meta["verdict"] = self.verdict
        if self.repaired:
            meta["repaired"] = True
        if self.skipped:
            meta["skipped"] = self.skipped
        return meta


# --- extracting the operative number -----------------------------------------


def operative_number(text: str) -> float | None:
    """The number a reply is actually asserting, or None when that is ambiguous.

    Conservative on purpose. This drives the capitulation guard, and a guard that fires on
    a misread is worse than no guard — so anything with several candidate numbers and no
    phrase marking one of them as the answer returns None and the guard simply sits out.
    """
    stripped = text.strip()
    if not stripped:
        return None

    # Prefer the last cue: a reply that reasons and then concludes states the answer at
    # the end, and the closing statement is the assertion the user reads.
    cues = _ANSWER_CUE.findall(stripped)
    if cues:
        return _to_float(cues[-1])

    found = {_to_float(a or b) for a, b in _ANY_NUMBER.findall(stripped)}
    found.discard(None)
    if len(found) == 1:
        return next(iter(found))  # type: ignore[arg-type]
    return None


def _to_float(token: str) -> float | None:
    token = token.strip().lower()
    if token in _WORD_NUMBERS:
        return _WORD_NUMBERS[token]
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def previous_assistant_reply(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return ""


# --- the capitulation guard ---------------------------------------------------


def capitulated(previous: str, current: str) -> tuple[float, float] | None:
    """Did the model change its numeric answer? Returns (was, now), or None.

    Only meaningful when the pressure that caused the change carried no information — a
    bare "nope" — which is why the caller gates on the follow-up kind rather than this
    function doing it. Under a real correction a changed answer is the wanted behaviour.

    Measured on the box word problem: told "its incorrect" eight times, the reasoning tier
    held its answer 5 times, caved once and went vague twice. This catches the cave.
    """
    was, now = operative_number(previous), operative_number(current)
    if was is None or now is None or was == now:
        return None
    return was, now


_HOLD_NOTE = (
    "You just changed your answer, and the user gave you no new information to justify "
    "the change — they only said you were wrong. Work the problem once more from the "
    "beginning. State the answer you reach and the one step that settles it, whichever "
    "of your two answers that turns out to support."
)


def unstable_reply(first: float, second: float) -> str:
    """What to say when two attempts disagree and nothing can break the tie.

    Abstention, phrased so the user can do something with it. Confident wrongness is the
    worst output available here — it is what started this project — so the honest move is
    to name both candidates rather than pick one on no evidence.
    """
    return (
        f"I've now got two different answers to that — {_pretty(first)} and "
        f"{_pretty(second)} — and I can't tell which is right, so I won't pick one and "
        f"pretend otherwise. If you tell me which part you think is off, I can work "
        f"that piece through on its own."
    )


def _pretty(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)


# --- cross-model verification -------------------------------------------------


async def verify(client, critic_spec, question: str, answer: str) -> str:
    """Ask the critic whether `answer` is right. Returns correct / incorrect / unsure.

    Never call this with `critic_spec` set to the model that produced `answer`. The caller
    enforces that; this function has no way to check it.
    """
    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nAnswer given: {answer}"},
    ]
    try:
        result = await client.chat(
            critic_spec,
            messages,
            options={"num_predict": _CRITIC_MAX_TOKENS, "temperature": 0.0},
        )
    except OllamaError as exc:
        log.warning("critic %s failed: %s", critic_spec.alias, exc)
        return "unsure"

    return parse_verdict((result.get("message") or {}).get("content") or "")


def parse_verdict(reply: str) -> str:
    tail = _THINK_CLOSE.split(reply)[-1]
    match = _VERDICT.search(tail) or _VERDICT.search(reply)
    if not match:
        return "unsure"
    word = match.group(1).upper().replace(" ", "").replace("\t", "")
    if word in {"NOTCORRECT", "INCORRECT"}:
        return "incorrect"
    if word == "CORRECT":
        return "correct"
    return "unsure"


# --- orchestration ------------------------------------------------------------


async def run(
    *,
    question: str,
    answer: str,
    previous: str,
    plan,
    answered_by,
    critic_spec,
    client,
    settings,
    regenerate,
) -> Adjudication:
    """Apply whichever checks this request earned and this hardware allows.

    `regenerate(note, spec)` re-runs generation — the caller owns it because locking,
    sampling options and the persona prompt all live in the API layer. It is called at
    most once per request: that is the "at most one repair" rule, and it is a hard cap
    rather than a convention because an unbounded critic loop oscillates.
    """
    # 1. Free, deterministic, and therefore first. A numeric answer that changed under
    #    pressure carrying no information is not evidence of a better answer.
    if plan.guard_capitulation and previous:
        flip = capitulated(previous, answer)
        if flip is not None:
            return await _settle_flip(flip, regenerate)

    # 2. The paid check. Only ever the 1.2B judging a smaller model's answer — see this
    #    module's docstring for why the reverse direction does not exist here.
    if plan.verify and settings.verify_enabled:
        if critic_spec is None or critic_spec.alias == answered_by.alias:
            skipped = (
                f"no independent critic for {answered_by.alias!r}: the only model that "
                f"can judge an answer on this machine is the one that wrote it"
            )
            consistency = await _self_consistency(answer, plan, settings, regenerate)
            if consistency is not None:
                consistency.skipped = skipped
                return consistency
            return Adjudication(skipped=skipped)

        verdict = await verify(client, critic_spec, question, answer)
        if verdict == "incorrect":
            # One repair: hand the question to the stronger model rather than asking the
            # weaker one to try again. Deliberately no "your last answer was wrong" note —
            # quoting a wrong answer back at a small model anchors it to that answer.
            repaired = await regenerate(None, critic_spec)
            if repaired.strip():
                return Adjudication(content=repaired, verdict="incorrect", repaired=True)
        return Adjudication(verdict=verdict)

    return Adjudication()


async def _settle_flip(flip: tuple[float, float], regenerate) -> Adjudication:
    was, now = flip
    repaired = (await regenerate(_HOLD_NOTE, None)).strip()
    if not repaired:
        return Adjudication(verdict="flip", repaired=True)

    again = operative_number(repaired)
    if again is None or again in (was, now):
        # The re-work landed on one of the two candidates. Whichever it picked, it picked
        # it after being told the pressure carried no information, so it stands.
        return Adjudication(content=repaired, verdict="flip", repaired=True)

    # Three different answers in three attempts. Nothing here can break that tie, and
    # picking one anyway is exactly the confident wrongness this project exists to avoid.
    return Adjudication(
        content=unstable_reply(was, again), verdict="unstable", repaired=True
    )


async def _self_consistency(
    answer: str, plan, settings, regenerate
) -> Adjudication | None:
    """Answer twice, compare the two numbers. Off by default; doubles the slowest tier.

    Not a critic: no model judges anything, two extracted numbers are compared exactly.
    That keeps it inside the rule the rest of this project follows — prose is never
    scored against prose — while still detecting an answer the model cannot reproduce.
    """
    if not settings.self_consistency or plan.level != "careful":
        return None
    first = operative_number(answer)
    if first is None:
        return None  # nothing mechanically comparable; a prose diff would be a guess

    second_text = await regenerate(None, None)
    second = operative_number(second_text)
    if second is None or second == first:
        return Adjudication(verdict="consistent")
    return Adjudication(content=unstable_reply(first, second), verdict="unstable")
