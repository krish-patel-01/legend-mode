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

# --- follow-ups -------------------------------------------------------------
#
# Routing reads only the latest message, which breaks on short replies: the box word
# problem routes to `think`, which answers "4" correctly, and then "its incorrect"
# routes as its own three-word message to the 350M, which capitulates and invents "3".
# The tier that produced an answer should be the tier asked to defend it.
#
# Disputes are split by strength on purpose. "that's wrong" is unambiguous whatever came
# before it, so it escalates to the reasoning tier outright. A bare "no" is not — it is
# just as likely to be answering a question the assistant asked — so it only counts as a
# dispute when the previous turn was already on a tier worth staying on.
_DISPUTE_STRONG = re.compile(
    r"^\s*(that'?s |it'?s |thats |this is )?(not (right|correct|true)|wrong|incorrect"
    r"|mistaken|false)\b"
    r"|^\s*(are you sure|you'?re wrong|check (it |that |again)|check again|recheck"
    r"|re-check|try again|that'?s not (it|right)|doesn'?t (look|seem) right)\b",
    re.IGNORECASE,
)
_DISPUTE_WEAK = re.compile(r"^\s*(no+|nope|nah|uh-?uh)[\s.!?]*$", re.IGNORECASE)

# Asks for more of the same reasoning, so it belongs on the same tier. Anchored at both
# ends: a continuation is the whole message, not a prefix of one. Without the trailing
# anchor "explain" would swallow "explain what a REST API is", which is a fresh question
# that deserves its own routing rather than inheriting the previous turn's tier.
_CONTINUATION = re.compile(
    r"^\s*(why|how (come|so)"
    r"|explain( (that|this|it|more|again|further))?"
    r"|elaborate( on (that|this|it))?"
    r"|go on|continue|keep going|say more|expand( on (that|this|it))?"
    r"|(show|walk) (me )?(your work|the steps?|through (it|that))"
    r"|show your work|prove it|break it down|(in )?more detail"
    r"|what about (that|this|it)|are you certain)"
    r"[\s.,!?]*$",
    re.IGNORECASE,
)


def followup_kind(text: str) -> str | None:
    """Classify a short reply as a follow-up to the previous turn, or None.

    Returns "dispute" (escalate regardless of history), "weak_dispute" or
    "continuation" (both only meaningful if the previous turn was on a sticky tier).
    """
    stripped = text.strip()
    # A long message carries its own signal and should route on its own merits.
    if not stripped or len(stripped) > 120:
        return None
    if _DISPUTE_STRONG.search(stripped):
        return "dispute"
    if _DISPUTE_WEAK.match(stripped):
        return "weak_dispute"
    if _CONTINUATION.match(stripped):
        return "continuation"
    return None


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
