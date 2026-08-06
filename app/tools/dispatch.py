"""The tool loop: a small model picks the tool, a bigger one writes the answer.

**The two roles are split across two models on purpose, and the split is measured.** From
the table in `gate.py`: the 350M picks the right tool 6/6 in about a second, beating both
1.2B tiers, while any model holding tool schemas answers ordinary questions badly — the
1.2B instruct told a user it *"can't provide that information"* when asked the capital of
France, purely because four unrelated tools were attached.

Those two facts point the same way. The model that decides needs no world knowledge, which
is exactly the 350M's profile and exactly why it is already the stage-3 classifier. The
model that writes needs its world knowledge intact, which means it must never see a tool
schema. So:

    dispatcher   350M, tool schemas attached, no persona, no history beyond the request
    writer       1.2B, tool *results* attached, no schemas, full persona

Verified directly, since the whole design rests on it: a model consumes a `role: "tool"`
turn perfectly well with no `tools` array on the request. Asked the time in Tokyo with the
result supplied that way, `instruct-q3` answered "Wednesday, 5 August 2026 at 21:14 (Japan
Standard Time)" — and answered "the capital of France is Paris" on the very next turn,
because nothing ever told it that it was a function-calling agent.

The dispatcher gets **no system prompt**. `scripts/cot_bench.py` measured system prompts
displacing the question entirely on this model — a formatting instruction cut it from 39
tokens of correct work to 13 tokens of echoed template. The tool schemas are already a
large addition to its context; the persona would be competing with them for a model this
size, and the dispatcher is not talking to the user anyway.

Results are passed on as `role: "tool"` rather than folded into the system prompt. Both
work, but the system-note version made the 350M echo the note back verbatim — the
prompt-echo failure `app/persona.py` documents — while the tool turn produced a real
sentence from both tiers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.backends.ollama import OllamaError
from app.tools.registry import ToolRegistry, ToolResult

log = logging.getLogger(__name__)

MAX_ITERATIONS = 3
"""Upper bound on dispatch rounds. Reached only while calls are *failing* — see below.

Bounded for the same reason `app/adjudicate.py` allows at most one repair: a small model
in a loop does not reliably notice that it is in a loop.
"""

STOP_WHEN_SATISFIED = True
"""Stop as soon as a round of calls all succeed, instead of asking the model again.

**Measured, and the reason is invention rather than repetition.** Asked "what's the time",
the dispatcher called `get_time({"city": ""})`, got the correct local time back, and then
called `get_time({"city": "London"})` — a city the user never mentioned. Repeat-detection
does not catch that, because it is a genuinely different call; it is the model finding
something to do because it was asked again.

So another round is only granted when the previous one *failed*, which is the case where
looping earns its cost: a wrong path, a bad argument, an unknown tool name are all things
a model can fix when told. A round that succeeded needs no follow-up, and asking for one
on this hardware reliably produces a worse request than the first.

Genuine chaining — search, then fetch what the search returned — is a different problem
and belongs to the family that needs it, not to every request by default.
"""

_DISPATCH_MAX_TOKENS = 256
"""A tool call is a name and a short JSON object. Anything longer is the model writing
prose instead of calling something, and capping it keeps the misfire cheap."""


@dataclass
class ToolRun:
    """What the loop produced. `messages` is what the writer should be handed."""

    messages: list[dict[str, Any]]
    results: list[ToolResult] = field(default_factory=list)
    iterations: int = 0
    families: set[str] = field(default_factory=set)
    error: str | None = None

    @property
    def ran(self) -> bool:
        return bool(self.results)

    def as_meta(self) -> dict[str, Any] | None:
        """Routing metadata, so the console can show what was called and what it cost."""
        if not self.results and self.error is None:
            return None
        meta: dict[str, Any] = {
            "families": sorted(self.families),
            "iterations": self.iterations,
            "calls": [
                {"name": r.name, "ok": r.ok, "ms": round(r.elapsed_ms)} for r in self.results
            ],
        }
        if self.error:
            meta["error"] = self.error
        return meta


def _signature(name: str, arguments: dict[str, Any] | None) -> str:
    """Identity of a call, for spotting the model repeating itself.

    Empty values are dropped before comparing. Asked "what's the time", the dispatcher
    called `get_time` twice — once as `{}` and once as `{"city": ""}` — which is the same
    call written two ways, and comparing the raw dicts let the second one through. An
    omitted optional argument and an empty one mean the same thing to every tool here.
    """
    meaningful = {
        k: v for k, v in (arguments or {}).items() if v not in (None, "", [], {})
    }
    return f"{name}({json.dumps(meaningful, sort_keys=True)})"


def _parse_calls(message: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            # Ollama normally hands back a dict, but the OpenAI wire format is a JSON
            # string and some templates pass it through unchanged.
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append((name, args if isinstance(args, dict) else {}))
    return calls


_REFERENCE = re.compile(
    r"\b(?:it|that|this|them|those|these|the same|again)\b", re.IGNORECASE
)


def needs_context(text: str) -> bool:
    """Does this request point at something said earlier?

    A turn like "then search it in the web" carries no subject. Given only that, the
    dispatcher called `web_search(query="latest news")` and the reply came back as a list
    of CNN and NBC links — for a question that had been about the weather in Ahmedabad one
    turn before.

    Checked rather than always prepending, because context is not free on a model this
    size: the prior turn is one more thing competing with the request for its attention.
    Only turns that cannot be resolved alone pay for it.
    """
    return bool(_REFERENCE.search(text))


async def run(
    *,
    text: str,
    registry: ToolRegistry,
    families: set[str],
    dispatcher_spec: Any,
    client: Any,
    context: str = "",
    allow_writes: bool = True,
    max_iterations: int = MAX_ITERATIONS,
    stop_when_satisfied: bool = STOP_WHEN_SATISFIED,
) -> ToolRun:
    """Decide and execute. Returns a ToolRun whose `messages` carry the results.

    `text` is the user's request. `context` is the previous question it refers back to,
    supplied by the caller only when `needs_context` says the request cannot stand alone —
    the dispatcher is choosing a function for *this* turn, and extra history mostly gives
    a model this size more chances to pick the wrong one.
    """
    schemas = registry.schemas(families, allow_writes=allow_writes)
    run = ToolRun(messages=[], families=set(families))
    if not schemas:
        return run

    # Phrased as one turn rather than a real exchange: the assistant's own previous reply
    # is what the dispatcher would otherwise start summarising instead of acting on.
    request = f"Earlier question: {context}\n\nFollow-up: {text}" if context.strip() else text
    convo: list[dict[str, Any]] = [{"role": "user", "content": request}]
    seen: set[str] = set()

    for iteration in range(1, max_iterations + 1):
        try:
            reply = await client.chat(
                dispatcher_spec,
                convo,
                tools=schemas,
                options={"num_predict": _DISPATCH_MAX_TOKENS, "temperature": 0.0},
            )
        except OllamaError as exc:
            log.warning("tool dispatch failed: %s", exc)
            run.error = str(exc)
            return run

        message = reply.get("message") or {}
        calls = _parse_calls(message)
        if not calls:
            # No call on the first pass means the gate was over-eager and the dispatcher
            # disagreed — the cheap second opinion the design wants. On a later pass it
            # means the model is done.
            run.iterations = iteration - 1
            return run

        # Repeats are checked *before* anything is committed to the conversation. A model
        # asking for a call it has already made is not using the result it has, so
        # re-running is pure cost — but bailing out midway through a batch would leave an
        # assistant turn advertising tool_calls that have no matching results, which is a
        # malformed conversation to hand any template.
        if any(_signature(name, args) in seen for name, args in calls):
            run.iterations = iteration
            return run
        seen.update(_signature(name, args) for name, args in calls)

        # Keep the assistant turn: dropping it leaves tool results answering nothing.
        convo.append({k: v for k, v in message.items() if k in {"role", "content", "tool_calls"}})

        round_ok = True
        for name, arguments in calls:
            result = await registry.invoke(name, arguments, request=text)
            run.results.append(result)
            convo.append(result.as_message())
            round_ok &= result.ok

        run.iterations = iteration
        run.messages = convo[1:]  # everything after the user turn the caller already has

        if round_ok and stop_when_satisfied:
            return run

    return run
