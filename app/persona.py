"""System prompt shared across every tier.

Kept separate from any one model's config since it's about the assistant's identity
and behavior, not a particular tier's sampling settings. No name is hardcoded — one
hasn't been picked yet — so the prompt says so rather than inventing one.

Three things here were settled by measurement against the actual models, not by
guessing, and the wording should not be "tidied" without re-running that check
(scripts in the scratchpad; 3 samples per probe per tier):

1. **Never end the prompt with a quotable answer.** An earlier brief version closed
   with `If the user asks your name, say you don't have one yet.` The 350M then
   replied "you don't have one yet." to *hi*, *thanks!*, and *what is the capital of
   France* — it treats the final sentence as a completion prefix. Moving the identity
   clause into the middle and ending on a behavioral directive took the 350M from
   failing most neutral prompts to 0 problems in 21 samples.

2. **Length matters on the 350M.** The full prompt is ~350 characters; at 350M that
   crowds out the actual question. `brief` is half that and is what `general` uses.

3. **Don't enumerate brands in the full prompt.** Naming a model primes it. Adding
   "never identify yourself as Qwen" made the Qwen3.5 tier's identity leak *worse*,
   not better (it began volunteering "I am Qwen" on unrelated turns). The full prompt
   therefore disclaims brands as a category instead of listing them.

Known limitation, only partly mitigated: models assert their own training identity no
matter what the prompt says. Qwen3.5-0.8B answers "who are you?" with "I am Qwen3.5,
developed by Tongyi Lab" 6/6 across three wordings, so `app/router/rules.py` pins
self-identity questions to the `trivial` route (the LFM 350M) instead of fighting it
in the prompt — a direct `model: "small"` call can still surface it. The 350M is much
better but not clean either: asked point-blank "who are you?" it named Liquid AI in
roughly 1 of 4 samples, sometimes alongside a confabulated second maker. Three brief
wordings were measured; the one kept here scored best (5/28 vs 7/28 for a
category-style "no company or product brand" disclaimer), so this is the floor
reachable by prompting, not a solved problem.

Answering "what model are you" with a real model name is *not* counted as a leak —
the prompt permits discussing internals when the user asks about them directly.

Which style a tier gets is set per model (`persona:` in models.yaml, default `full`).
"""

from __future__ import annotations

_BRIEF_UNNAMED = (
    "You have no name yet, and you are not ChatGPT or any other commercial assistant."
)
_FULL_UNNAMED = (
    "You have no name yet, and no company or product brand to claim as your identity. "
    "That applies to your own name only: if the user tells you theirs, remember it "
    "and use it."
)
_FULL_NAMED = "If the user tells you their name, remember it and use it."

# {identity} sits mid-prompt on purpose — see note 1 above.
_BRIEF = "You are a helpful local AI assistant. {identity} Answer the user's question directly and concisely."
_FULL = (
    "You are a helpful, direct AI assistant running locally. {identity} Don't bring up "
    "how you work internally unless the user asks. Answer directly and concisely "
    "unless real depth is asked for."
)


# Appended when the sticky stage sees the user disputing the previous answer.
#
# Two failures to fix at once. The first is sycophancy: told "its incorrect", the 350M
# replied "You're right! The number of boxes is indeed 2, not 3" — agreeing without
# rechecking anything. The second is budget exhaustion: routing disputes to the 1.2B
# means a bare "nope" arrives with no concrete claim attached, and the model reasons in
# circles until the token cap and returns nothing at all.
#
# So the note does three jobs — refuse the reflex to capitulate, bound the reply, and
# give the model something cheap to do when the dispute carries no information at all.
# That last clause matters: a bare "nope" offers nothing to re-check, and without an
# alternative the model reasons in circles until the budget runs out.
#
# The length instruction sits mid-note, not at the end. Ending on "Answer in no more
# than three sentences." made the 1.2B reply "The total is four. Three sentences: Four
# boxes." — the same trailing-instruction echo this module's docstring warns about. The
# note now ends on an action, and if the model does echo that one it asks the user which
# part they disagree with, which is the wanted behaviour anyway.
DISPUTE_NOTE = (
    "The user is disputing your previous answer. Keep this reply to two or three "
    "sentences. Do not simply agree that you were wrong: re-check the specific claim, "
    "and if your answer was right, say so plainly and give the one reason why. If they "
    "have not said what is wrong, ask them which part they disagree with."
)

# The mirror image, and deliberately not the same text. A dispute says "that's wrong" and
# carries no information, so the right instruction is to hold firm unless the recheck
# shows otherwise. A correction says "but the monkeys are on the bed" — it hands over a
# fact the answer missed, and there the stubbornness DISPUTE_NOTE encourages is exactly
# wrong. Observed: the previous answer was repeated verbatim after the user supplied the
# missing constraint, because nothing marked the turn as a correction at all.
CORRECTION_NOTE = (
    "The user is pointing out something your previous answer missed or got wrong. Treat "
    "what they just said as true. Re-work the answer from the start with it included, "
    "in two or three sentences, and say plainly if it changes your conclusion."
)


def build_system_prompt(assistant_name: str | None, style: str = "full") -> str:
    brief = style == "brief"
    if assistant_name:
        identity = f"Your name is {assistant_name}."
        if not brief:
            identity = f"{identity} {_FULL_NAMED}"
    else:
        identity = _BRIEF_UNNAMED if brief else _FULL_UNNAMED
    return (_BRIEF if brief else _FULL).format(identity=identity)


def ensure_system_prompt(
    messages: list[dict], assistant_name: str | None, style: str = "full"
) -> list[dict]:
    """Prepend the persona system message unless the caller already supplied one."""
    if messages and messages[0].get("role") == "system":
        return messages
    return [
        {"role": "system", "content": build_system_prompt(assistant_name, style)},
        *messages,
    ]
