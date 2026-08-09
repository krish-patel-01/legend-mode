"""How a tool result is shown to the model that has to write the answer.

**Measured 2026-08-09, and the answer is no: production's frame stays.** Five cases, four
samples each, at the temperature 0.6 that models.yaml actually gives this tier:

    frame       correct   refused   what changed
    tool         15/20      2/20    what ships
    assistant    11/20      0/20    refusals gone, answers gone vague
    prefill       9/20      0/20    same, worse

The hypothesis was half right, and the wrong half is the expensive one. Re-framing **does**
remove the refusal — 2 down to 0 — and it takes the answer with it. What replaces "I don't
have real-time capabilities" is not the price; it is *"the current gold price varies by
ounce weight, but as of my latest data..."*, or a range of "$4,200–$4,300" for a figure
printed in the evidence as 4,280.81. Both clear a refusal filter and tell the user nothing.

The refusal was never the disease. It is the model declining to commit, and a frame that
forbids declining moves the non-commitment somewhere a scorer likes better rather than
removing it — which is why the honest reading of the 0/20 refusal columns is that they are
worth less than the 4 and 6 points of accuracy paid for them.

The filter is not even airtight: under `assistant` the disk case answered *"I currently
don't have direct access to your local system's disk usage"* twice, with the figure in
context. Scored as a non-refusal because the phrasing was novel. Any conclusion drawn from
a refusal *rate* alone would have been wrong in both directions at once.

So this module is bench scaffolding, not production code, and lives in scripts/ for that
reason. It stays because the negative result is worth as much as the code would have been:
the next person to notice a refusal will have this idea too.

---

The result is the same bytes either way; what changes is who the conversation says
fetched them. That turns out to matter, because the failure this file exists to fix is not
a comprehension failure — the model can read the number, it declines to say it:

    user   what time is it in Tokyo
    tool   Sunday, 09 August 2026, 05:31:14 in Tokyo (Asia/Tokyo, UTC+0900)
    model  I don't have real-time capabilities, so I can't provide the current local time

**The refusal is intermittent, not topical.** The case that started this was a gold price;
re-measured against fresh search results the same tier answered "$4,280.81 per ounce"
without hesitating and refused `get_time` instead. Production samples this tier at
temperature 0.6, so this is one branch of the distribution rather than a fixed opinion
about prices — which is why a single transcript reads like a rule and why every arm here
is measured over repeats rather than greedily.

The lever came from scripts/tool_bench.py measuring FunctionGemma, which never refused in
5/5 cases. Its format has no third speaker: a call and its result live inside the
*model's own turn*, so the evidence arrives as something the assistant already did rather
than as something it was handed. That is a prompt shape, not a model capability, and
nothing stopped us applying it to the tier we already run.

Frames, all carrying identical result text:

    tool       the OpenAI shape: an assistant turn with `tool_calls`, then `role: "tool"`
               results. What shipped, and the baseline here.
    assistant  the call and its result written into one assistant turn as plain prose,
               closed, followed by the user's question again.
    prefill    the same assistant turn left open, so the model continues its own sentence
               instead of starting a reply.

`FRAMES` is the table the bench iterates; `build` is what app/api.py calls.
"""

from __future__ import annotations

import json
from typing import Any

TOOL = "tool"
ASSISTANT = "assistant"
PREFILL = "prefill"

FRAMES = (TOOL, ASSISTANT, PREFILL)

DEFAULT = TOOL
"""Overwritten below once the measurement lands. Kept as a name so callers never spell a
frame literal and the switch is one line."""


def _call_line(name: str, arguments: dict[str, Any] | None) -> str:
    args = json.dumps(arguments or {}, ensure_ascii=False)
    return f"{name}({args})"


def _narrate(calls: list[tuple[str, dict[str, Any] | None, str]]) -> str:
    """One assistant-voice paragraph covering every call and what it returned."""
    parts = []
    for name, arguments, result in calls:
        parts.append(f"I called {_call_line(name, arguments)} and it returned:\n\n{result}")
    return "\n\n".join(parts)


def build(
    frame: str,
    *,
    question: str,
    calls: list[tuple[str, dict[str, Any] | None, str]],
    note: str,
) -> list[dict[str, Any]]:
    """Messages to append after the system prompt, for the model writing the answer.

    `calls` is `(name, arguments, result_text)` in the order they ran. `note` is the
    system-level reminder that a tool has already answered this — passed in rather than
    imported so app/persona.py stays the single place that wording lives.
    """
    if frame == TOOL:
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": name, "arguments": arguments or {}}}
                for name, arguments, _ in calls
            ],
        })
        for _, _, result in calls:
            messages.append({"role": "tool", "content": result})
        return messages

    narration = _narrate(calls)

    if frame == ASSISTANT:
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": narration},
            {"role": "user", "content": note},
        ]

    if frame == PREFILL:
        # No trailing user turn: the assistant turn is left open and the model continues
        # it. Ollama passes a final assistant message through as a prefix rather than
        # closing it, which is what makes the answer a continuation of the model's own
        # sentence instead of a fresh decision about whether to answer at all.
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"{narration}\n\nSo, to answer directly:"},
        ]

    raise ValueError(f"unknown frame {frame!r}; expected one of {FRAMES}")
