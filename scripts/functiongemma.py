"""Speaking FunctionGemma's native prompt format, because Ollama does not.

`ollama show --template functiongemma:270m` returns `{{ .Prompt }}` — the library build
ships **no chat template**, only a raw passthrough, while still advertising a `tools`
capability. So a normal `/api/chat` call with a `tools=` array renders none of it. What the
model actually received in that state was the message bodies concatenated with the literal
words `user` and `model` between them, and it answered:

    "I apologize, but I cannot provide real-time information about current time ... My
     available tools are focused on computer tasks"

which reads exactly like a capability verdict and is nothing of the sort. It scored 0/10 on
tool selection that way. **Any measurement taken through `/api/chat` against this build is
measuring the missing template, not the model.**

FunctionGemma does not use JSON on the wire. Tools are declared, called and answered in a
bespoke bracket notation with `<escape>` as the string delimiter:

    <bos><start_of_turn>developer
    You are a model that can do function calling with the following functions
    <start_function_declaration>declaration:get_weather{description:<escape>...<escape>,
      parameters:{type:<escape>OBJECT<escape>,...}}<end_function_declaration><end_of_turn>
    <start_of_turn>user
    what's the weather in Paris<end_of_turn>
    <start_of_turn>model
    <start_function_call>call:get_weather{city:<escape>paris<escape>}<end_function_call>

and a result comes back the same way, after which the model writes prose:

    <start_function_response>response:get_weather{temperature:26}<end_function_response>

The developer line is required and its wording is fixed by the fine-tune, so it is a
constant here rather than something to tune.

This module converts our existing JSON tool schemas into that notation and parses calls
back out, so the rest of the codebase keeps one schema format. It is deliberately not in
`app/` yet: it exists to let `scripts/tool_bench.py` ask a fair question, and it earns a
place in the router only if the answer says so.
"""

from __future__ import annotations

import json
import re
from typing import Any

DEVELOPER_LINE = (
    "You are a model that can do function calling with the following functions"
)

BOS = "<bos>"
TURN = "<start_of_turn>"
END_TURN = "<end_of_turn>"

# **No stop sequences, because none are usable.** Ollama strips special tokens out of the
# text it returns, so `<end_of_turn>` and `<end_function_call>` never appear in the
# response however faithfully the model emits them — a completion comes back as
# `call:get_time{city: Tokyo} model\n call:get_time{city: Tokyo} model\n ...`, where each
# bare ` model` is a stripped turn boundary. Passing them as stops silently does nothing.
# So generation is bounded by `num_predict` and trimmed afterwards by `first_reply`.
STOP: list[str] = []

# What a stripped turn boundary looks like once the tags are gone, plus the start of a
# fresh call. Either means the model has moved past the answer we asked for.
_OVERRUN = re.compile(r"(?:\bcall:\w|\bresponse:\w|\s+model\b|\s+user\b)")


# --- JSON schema -> declaration notation ---------------------------------------


def _encode(value: Any) -> str:
    """One value in the bracket notation. Strings are wrapped in `<escape>` pairs."""
    if isinstance(value, str):
        return f"<escape>{value}<escape>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_encode(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_encode(v)}" for k, v in value.items()) + "}"
    return "<escape><escape>"


def _upper_types(node: Any) -> Any:
    """`"type": "string"` -> `"type": "STRING"`, which is what the examples show.

    Gemma's function-declaration schema is the Google AI one, where types are the
    uppercase protobuf spellings, rather than the lowercase JSON Schema ones our
    registry emits for Ollama.
    """
    if isinstance(node, dict):
        return {
            k: (v.upper() if k == "type" and isinstance(v, str) else _upper_types(v))
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_upper_types(v) for v in node]
    return node


def declaration(schema: dict[str, Any]) -> str:
    """One `<start_function_declaration>` from one of our Ollama tool schemas."""
    fn = schema.get("function") or schema
    body = {
        "description": fn.get("description", ""),
        "parameters": _upper_types(fn.get("parameters") or {"type": "object"}),
    }
    return (
        f"<start_function_declaration>declaration:{fn['name']}{_encode(body)}"
        f"<end_function_declaration>"
    )


# --- prompt assembly -----------------------------------------------------------


def _developer_turn(schemas: list[dict[str, Any]]) -> str:
    decls = "".join(declaration(s) for s in schemas)
    return f"{BOS}{TURN}developer\n{DEVELOPER_LINE}\n{decls}{END_TURN}\n"


def dispatch_prompt(request: str, schemas: list[dict[str, Any]]) -> str:
    """Ask for a tool call and stop."""
    return (
        _developer_turn(schemas)
        + f"{TURN}user\n{request}{END_TURN}\n"
        + f"{TURN}model\n"
    )


def answer_prompt(
    request: str,
    schemas: list[dict[str, Any]],
    name: str,
    arguments: dict[str, Any],
    result: str,
) -> str:
    """Replay a completed call and ask for the sentence that reports it.

    **The response goes inside the model's own turn, straight after the call — not in a
    following user turn.** That is the whole of "unified action and chat", and it is
    measured, not inferred. Same question, same evidence, three placements:

        response in a following user turn   ` call:get_time{city: Tokyo} model` x8
        response as plain prose in the user turn
                                            "I cannot assist with finding the time"
        response inside the model turn      "The current date and time in Tokyo is
                                             Sunday, ... 05:31:14"

    Only the third produces a sentence. The first leaves the model believing its call was
    never answered, so it repeats it until the budget runs out; the second is outside the
    function-calling frame entirely, and **this model refuses ordinary prose questions** —
    handed the exact same clock reading as narrative text it answered "my current
    capabilities are limited to interacting with clock information using the provided
    tools". It is not a chat model with tools bolted on; the frame is the only mode it has.
    """
    call = f"<start_function_call>call:{name}{_encode(arguments)}<end_function_call>"
    response = (
        f"<start_function_response>response:{name}{_encode({'result': result})}"
        f"<end_function_response>"
    )
    return (
        _developer_turn(schemas)
        + f"{TURN}user\n{request}{END_TURN}\n"
        + f"{TURN}model\n{call}{response}"
    )


def first_reply(text: str) -> str:
    """The answer sentence, cut where the model runs on into another turn or call.

    Needed because the stop sequences cannot work (see STOP). Without it a good answer and
    the eight repeated calls that follow it score as one blob.
    """
    match = _OVERRUN.search(text)
    return (text[: match.start()] if match else text).strip()


# --- parsing the call back out -------------------------------------------------

# **The wrapper tags are optional here, and in practice they are always absent.** Ollama
# strips special tokens from the response, so a perfectly-formed call comes back as the
# bare text `call:get_time{city: Tokyo}`. Requiring `<start_function_call>` matched nothing
# at all and scored the model 0/10 for a defect in the reader.
_CALL = re.compile(
    r"(?:<start_function_call>)?\s*\bcall:\s*(?P<name>[A-Za-z_]\w*)\s*"
    r"(?P<args>\{[^{}]*\})?",
)
_PAIR = re.compile(r'(\w+)\s*:\s*(?:"([^"]*)"|\[([^\]]*)\]|([^,}]+))')


def parse_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Every call in a completion, as `(name, arguments)`.

    Lenient on the argument syntax on purpose. Published examples show both
    `city:<escape>paris<escape>` and `city: "paris"` for the same thing, and a scorer that
    accepted only one spelling would report a formatting disagreement as a wrong answer.
    """
    calls = []
    for match in _CALL.finditer(text):
        raw = (match.group("args") or "{}").replace("<escape>", '"')
        calls.append((match.group("name"), _parse_args(raw)))
    return calls


def _parse_args(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    out: dict[str, Any] = {}
    for key, quoted, listed, bare in _PAIR.findall(raw):
        if quoted:
            out[key] = quoted
        elif listed:
            out[key] = [v.strip().strip('"') for v in listed.split(",") if v.strip()]
        else:
            out[key] = _coerce(bare.strip())
    return out


def _coerce(token: str) -> Any:
    if token in ("true", "false"):
        return token == "true"
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token.strip('"')
