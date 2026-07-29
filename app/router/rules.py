"""Stage 1: deterministic signals.

These cost nothing and are more reliable than any model for the cases they cover, so
they run first and short-circuit the rest of the cascade. Anything ambiguous returns
None and falls through to embeddings.
"""

from __future__ import annotations

import re

from app.router.types import RouteDecision, RouteRequest

# There was a _TOOL_PATTERNS rule here that sent "search the web" / "current weather"
# style prompts to a dedicated tools route. Both that route and the tier behind it are
# gone (see routes.yaml), so these prompts now fall through to chat/think like any
# other question. Nothing can act on them either way; the difference is only which
# model explains that it can't.

# Phrasings that ask for explicit multi-step reasoning.
_THINK_PATTERNS = re.compile(
    r"\b(step[- ]by[- ]step|think (carefully|through|hard)|reason (about|through)"
    r"|prove|derive|justify|trade[- ]?offs?|root cause|why does|why is"
    r"|debug|find the bug|time complexity|edge cases?|solve for)\b",
    re.IGNORECASE,
)

# Greetings and acknowledgements, matched only when they are the entire message (a
# trailing pleasantry like "there" / "everyone" / a name is allowed).
_TRIVIAL = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|thanks?|thank you|thx|ok(ay)?|got it|cool|nice"
    r"|yes|no|yep|nope|sure|bye|goodbye|good (morning|evening|night)|never ?mind)"
    r"(\s+\w+){0,2}[\s.!?]*$",
    re.IGNORECASE,
)

# Questions about the assistant's own identity. These are pinned to the LFM tier for
# a measured reason: Qwen3.5-0.8B answers "who are you?" with "I am Qwen3.5, developed
# by Tongyi Lab" no matter what the system prompt says (6/6 across three wordings —
# see app/persona.py), while the 350M answers correctly every time. Prompting can't
# fix that, so routing does. Note the negative lookahead: "what are you doing" is a
# normal question, not an identity one, and "your name" must not catch "my name".
_SELF_IDENTITY = re.compile(
    r"\b(what('?s| is) your name|who are you|what are you(?!\s+(doing|working|up|going"
    r"|talking|trying|looking|planning))|who (made|built|created|trained) you"
    r"|what (model|llm|ai) are you|are you (chat ?gpt|claude|gemini|gpt|qwen|llama))\b",
    re.IGNORECASE,
)

# Quantity questions, which split two ways: "how many days in a leap year" is a
# lookup, "how many boxes do I have if I have two boxes with one box inside each" is
# arithmetic wearing a sentence. The question phrase alone isn't enough to tell them
# apart, so it only counts as reasoning when the prompt also sets up a scenario —
# a hypothetical/distributive word, or two or more quantities to combine.
_QUANTITY_Q = re.compile(
    r"\b(how many|how much|how long|how old|how far"
    r"|what('s| is) the (total|sum|average|remainder|difference))\b",
    re.IGNORECASE,
)
_SCENARIO = re.compile(
    r"\b(if|given|suppose|assuming|each|per|every|both|apiece|inside|left over|"
    r"altogether|in total)\b",
    re.IGNORECASE,
)
_QUANTITY = re.compile(
    r"(\b\d+(\.\d+)?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven"
    r"|twelve|dozen|half|third|quarter|twice|double|triple)\b)",
    re.IGNORECASE,
)

_CODE_FENCE = re.compile(r"```")

# Above this many characters, a prompt is doing something substantial enough that the
# 0.8B chat model is the wrong default even without an explicit reasoning cue.
_LONG_PROMPT_CHARS = 1200


def apply(req: RouteRequest) -> RouteDecision | None:
    if req.forced_model:
        return RouteDecision(
            route="forced",
            model=req.forced_model,
            stage="override",
            reason=f"caller pinned model to {req.forced_model!r}",
        )

    # `has_images` used to force a vision tier and `has_tools` a tools tier. Neither
    # exists now. Image requests are rejected up front in app/api.py rather than
    # routed, and a tools array is forwarded without any tier claiming to honour it,
    # so both fields are left for the caller's own inspection and routed on text alone.

    text = req.text.strip()
    if not text:
        return RouteDecision(route="trivial", stage="rules", reason="empty prompt")

    # Unlike the greeting rule below, this applies at any point in a conversation:
    # "who are you?" is an identity question on turn 1 and on turn 30 alike.
    if _SELF_IDENTITY.search(text):
        return RouteDecision(
            route="trivial",
            stage="rules",
            reason="asks about the assistant's own identity",
        )

    # Only treat a bare greeting as trivial on the opening turn. Mid-conversation an
    # "ok" usually means "continue", which the tiny model would handle badly.
    if req.message_count <= 1 and _TRIVIAL.match(text):
        return RouteDecision(
            route="trivial", stage="rules", reason="greeting or acknowledgement"
        )

    if _THINK_PATTERNS.search(text):
        return RouteDecision(
            route="think", stage="rules", reason="explicit reasoning cue in prompt"
        )

    if _QUANTITY_Q.search(text) and (
        _SCENARIO.search(text) or len(_QUANTITY.findall(text)) >= 2
    ):
        return RouteDecision(
            route="think",
            stage="rules",
            reason="counting or arithmetic word problem",
            confidence=0.8,
        )

    if _CODE_FENCE.search(text):
        return RouteDecision(
            route="think", stage="rules", reason="prompt contains a code block"
        )

    if len(text) >= _LONG_PROMPT_CHARS:
        return RouteDecision(
            route="think",
            stage="rules",
            reason=f"long prompt ({len(text)} chars)",
            confidence=0.7,
        )

    return None
