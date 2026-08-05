"""Which tool families, if any, a request is allowed to see.

**This file exists because attaching tools makes every model here worse.** Measured on
2026-08-05 with four tool definitions attached (get_weather, read_file, run_command,
web_search) against six prompts that need no tool at all, greedy, each model asked the
same question twice — once with the tools and once without:

    model                positives   spurious calls   degraded answers   median
    LFM2.5-230M             1/6           2/6              2/6            0.8 s
    LFM2.5-350M             6/6           1/6              2/6            1.0 s
    LFM2.5-1.2B-Instruct    3/6           2/6              3/6            4.0 s
    LFM2.5-1.2B-Thinking    4/6           1/6              5/6           10.6 s

"Degraded" is not a near miss. With tools attached the 1.2B instruct answered *"I'm sorry,
but I can't provide that information"* to **what is the capital of France**, and *"I don't
have access to specific calendar data"* to **how many days are in a leap year** — both of
which it answers correctly when no tools are present. It wrote `web_search` calls for
*write me a haiku about winter*. The reasoning tier returned empty content on 4 of 6.
Offering tools gives these models a refusal posture: they stop believing they know things.

So the gate is not an optimisation to save a dispatch call. It is what stops tools
damaging the 90% of traffic that never needed one, and it is why the answer-writing model
never sees a tool schema at all (see `dispatch.py`).

Two consequences shaped the patterns below:

- **Decline by default.** An unmatched request gets no tools, which is the behaviour that
  was correct before this package existed. Same rule as `app/guardrails.py`: a gate that
  fires on a misread request is worse than no gate.
- **Match the trigger, not the topic.** "What time is it" needs a clock; "how do
  timezones work" does not. The patterns anchor on the user *asking for a current value or
  an action*, not on subject matter, which is the distinction the negatives above kept
  failing to make.

The same shape as `app/effort.py`'s `wants_retrieval`, for the same reason: indiscriminate
retrieval cost the source paper 5.0 GPQA points, and indiscriminate tool exposure costs
more than that here.
"""

from __future__ import annotations

import re

BASICS = "basics"
FILES = "files"
WEB = "web"
NOTES = "notes"

# --- families -----------------------------------------------------------------

# Asking for the current value, not for the concept. "what time is it", "today's date",
# "what time is it in Tokyo" — but never "how does daylight saving work".
_CLOCK = re.compile(
    r"\b(?:what(?:'s| is)\s+(?:the\s+)?(?:time|date|day)\b"
    r"|what\s+time\s+is\s+it\b"
    r"|current\s+(?:time|date)\b"
    r"|today'?s\s+date\b"
    r"|what\s+day\s+is\s+(?:it|today)\b)",
    re.IGNORECASE,
)

# Arithmetic on numbers the user supplied. Requires digits *and* an operator or an
# explicit ask, so "I have three ideas" never reaches it.
_MATH = re.compile(
    r"(?:\d\s*[+\-*/%^]\s*\d"
    r"|\b(?:calculate|compute|work out|what(?:'s| is))\b[^.?!]{0,40}?\d"
    r"|\d+\s*(?:%|percent)\s+(?:of|off)\s+\d)",
    re.IGNORECASE,
)

_MACHINE = re.compile(
    r"\b(?:disk\s+space|free\s+space|storage\s+left"
    r"|how\s+much\s+(?:disk|ram|memory|space|storage)"
    r"|how\s+many\s+(?:cpu|cores|logical\s+cores)"
    r"|cpu\s+(?:count|cores)|system\s+(?:status|info)"
    r"|what\s+os\b|this\s+(?:machine|computer))",
    re.IGNORECASE,
)

# A path or URL written out literally. Windows drive letters take either slash — this
# repo's own paths appear as both `k:\Projects\Legend_Mode` and `k:/Projects/Legend_Mode`
# — so accepting only the backslash form would miss half of them.
_LITERAL_TARGET = r"https?://|(?:^|\s)(?:[~.]?/|[A-Za-z]:[\\/])\S+"

_FILES = re.compile(
    r"\b(?:read|open|show me|cat|list|write|create|save|delete)\b[^.?!]{0,30}?"
    r"\b(?:file|files|folder|directory|dir|\.txt|\.md|\.py|\.json|\.yaml|\.csv)\b"
    r"|\bwhat(?:'s| is)\s+in\s+(?:the\s+)?(?:file|folder|directory)\b"
    r"|(?:^|\s)(?:[~.]?/|[A-Za-z]:[\\/])\S+",
    re.IGNORECASE,
)

_WEB = re.compile(
    r"\b(?:search (?:the )?web|google it|look (?:it |this )?up online|search online"
    r"|latest news|current (?:news|price|weather|score)|what'?s happening"
    r"|weather (?:in|at|for)\b|who won\b|fetch (?:the )?(?:url|page|site)"
    r"|https?://)",
    re.IGNORECASE,
)

_NOTES = re.compile(
    # Writing something down.
    r"\b(?:remember (?:that|this|my)|don'?t forget|make a note|take a note"
    r"|save (?:this|that) (?:to|as) (?:a )?note|write (?:this|that) down"
    r"|jot (?:this|that) down|add (?:this|that) to my notes"
    # Getting it back. The first version had only "what did I say about", and both live
    # read requests missed: "what did I *write* about coffee" and "read me the note about
    # coffee" reached no tool at all. Recall is phrased at least as many ways as capture.
    r"|what did I (?:say|write|note|tell you) about"
    r"|did I (?:write|note|say) anything about"
    r"|(?:read|open|show|find|check|look up|search) (?:me )?(?:the |my |your )?notes?\b"
    r"|(?:in|from) my notes\b|what'?s in my notes"
    r"|remind me (?:what|about))\b",
    re.IGNORECASE,
)

_FAMILY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_CLOCK, BASICS),
    (_MATH, BASICS),
    (_MACHINE, BASICS),
    (_FILES, FILES),
    (_WEB, WEB),
    (_NOTES, NOTES),
]

# Requests that name a tool-ish word while wanting prose about it. Checked first and
# checked hard: these are the ones the raw model got wrong.
_DISCUSSION = re.compile(
    r"\b(?:how (?:do|does|did|would|can) (?:i|you|we|it|they)\b"
    r"|explain|what is a|what'?s a|what does .{0,20} mean|difference between"
    r"|why (?:do|does|is|are)|write (?:me )?a (?:haiku|poem|story|song|joke)"
    r"|tell me a joke)",
    re.IGNORECASE,
)


def wanted(
    text: str,
    *,
    enabled: set[str] | None = None,
    grounded: bool = False,
) -> set[str]:
    """Tool families this request may see. Empty means attach nothing.

    `enabled` is the caller's hard allowlist — a session that disables `files` never gets
    them however the request is phrased. `grounded` suppresses everything: a guardrail has
    already computed the exact answer, so a tool could only disagree with it.
    """
    if grounded or not text.strip():
        return set()

    families = {family for pattern, family in _FAMILY_PATTERNS if pattern.search(text)}

    # A request that is asking to be *told about* something wins over a keyword match:
    # "explain how to read a file in Python" is not a request to read a file. The one
    # exception is an explicit path or URL, which is too concrete to be conceptual.
    if families and _DISCUSSION.search(text) and not _has_literal_target(text):
        return set()

    if enabled is not None:
        families &= enabled
    return families


def _has_literal_target(text: str) -> bool:
    """A concrete path or URL — evidence the request means a real object, not an idea."""
    return bool(re.search(_LITERAL_TARGET, text))
