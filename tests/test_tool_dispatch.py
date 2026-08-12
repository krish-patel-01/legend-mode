"""The dispatch loop, against a stubbed backend.

No model is loaded here. What is being tested is the control flow around the model —
iteration bounds, repeat detection, malformed arguments, backend failure — because those
are the paths a live model reaches only occasionally and the ones that hurt when wrong.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.backends.ollama import OllamaError
from app.tools import dispatch
from app.tools.registry import Tool, ToolRegistry


def _call(name: str, arguments: Any) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": arguments}}


class StubClient:
    """Returns queued replies; records what it was asked."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self._replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def chat(self, spec, messages, *, tools=None, options=None, think=None):
        self.requests.append({"messages": messages, "tools": tools, "options": options})
        if not self._replies:
            return {"message": {"role": "assistant", "content": "done"}}
        return {"message": {"role": "assistant", **self._replies.pop(0)}}


def _registry(calls: list[str] | None = None) -> ToolRegistry:
    def echo(value: str = "") -> str:
        if calls is not None:
            calls.append(value)
        return f"echoed {value}"

    return ToolRegistry([
        Tool(
            "echo",
            "d",
            {"type": "object", "properties": {"value": {"type": "string"}}},
            echo,
            "basics",
        )
    ])


async def _run(client: StubClient, registry: ToolRegistry, **kw) -> dispatch.ToolRun:
    return await dispatch.run(
        text="do the thing",
        registry=registry,
        families={"basics"},
        dispatcher_spec=object(),
        client=client,
        **kw,
    )


async def test_no_tool_call_means_the_gate_was_over_eager() -> None:
    """The dispatcher is a second opinion: it may decline what the gate let through."""
    run = await _run(StubClient([{"content": "just chatting"}]), _registry())
    assert not run.ran
    assert run.messages == []
    assert run.as_meta() is None


async def test_a_tool_call_is_executed_and_appended() -> None:
    client = StubClient([{"content": "", "tool_calls": [_call("echo", {"value": "hi"})]}])
    run = await _run(client, _registry())
    assert run.ran
    assert run.results[0].content == "echoed hi"
    assert [m["role"] for m in run.messages] == ["assistant", "tool"]
    assert run.messages[-1]["content"] == "echoed hi"


async def test_no_family_means_no_backend_call_at_all() -> None:
    client = StubClient([])
    run = await dispatch.run(
        text="x", registry=_registry(), families=set(),
        dispatcher_spec=object(), client=client,
    )
    assert not run.ran
    assert client.requests == []


async def test_the_dispatcher_gets_schemas_but_no_system_prompt() -> None:
    """Measured: a system prompt displaces the question on this model."""
    client = StubClient([{"content": "no"}])
    await _run(client, _registry())
    sent = client.requests[0]
    assert sent["tools"], "the dispatcher must see the schemas"
    assert [m["role"] for m in sent["messages"]] == ["user"]


async def test_repeated_identical_calls_stop_the_loop() -> None:
    """Only reachable on the failure path now, which is the only path that iterates.

    A model that answers a failed call by making the identical failed call again is stuck,
    and the retry it earned is worth nothing.
    """
    same = {"content": "", "tool_calls": [_call("nope", {"value": "a"})]}
    run = await _run(StubClient([same, same, same]), _registry())
    assert len(run.results) == 1, "the second identical call must not execute"
    assert run.iterations == 2


async def test_an_omitted_argument_matches_an_empty_one() -> None:
    """Measured: `get_time` was called as {} then {"city": ""} — the same call twice."""
    calls: list[str] = []
    run = await _run(
        StubClient([
            {"content": "", "tool_calls": [_call("echo", {})]},
            {"content": "", "tool_calls": [_call("echo", {"value": ""})]},
        ]),
        _registry(calls),
    )
    assert len(calls) == 1
    assert len(run.results) == 1


async def test_a_successful_round_ends_the_loop() -> None:
    """Measured: after a correct answer the model invented get_time(city="London")."""
    client = StubClient([
        {"content": "", "tool_calls": [_call("echo", {"value": "a"})]},
        {"content": "", "tool_calls": [_call("echo", {"value": "london"})]},
    ])
    calls: list[str] = []
    run = await _run(client, _registry(calls))
    assert calls == ["a"], "a satisfied request must not be dispatched again"
    assert run.iterations == 1


async def test_a_failed_call_earns_another_round() -> None:
    """Recovering from a bad tool name or argument is what looping is actually for."""
    client = StubClient([
        {"content": "", "tool_calls": [_call("nope", {})]},
        {"content": "", "tool_calls": [_call("echo", {"value": "b"})]},
    ])
    calls: list[str] = []
    run = await _run(client, _registry(calls))
    assert calls == ["b"]
    assert len(run.results) == 2
    assert not run.results[0].ok and run.results[1].ok


async def test_iterations_are_capped_while_calls_keep_failing() -> None:
    replies = [{"content": "", "tool_calls": [_call("nope", {"n": str(i)})]} for i in range(10)]
    run = await _run(StubClient(replies), _registry(), max_iterations=2)
    assert run.iterations == 2
    assert len(run.results) == 2


async def test_chaining_is_available_when_a_family_needs_it() -> None:
    client = StubClient([
        {"content": "", "tool_calls": [_call("echo", {"value": "a"})]},
        {"content": "", "tool_calls": [_call("echo", {"value": "b"})]},
        {"content": "finished"},
    ])
    calls: list[str] = []
    run = await _run(client, _registry(calls), stop_when_satisfied=False)
    assert calls == ["a", "b"]
    assert len(run.results) == 2


async def test_arguments_as_a_json_string_are_parsed() -> None:
    """The OpenAI wire format sends arguments as a string; some templates pass it on."""
    client = StubClient([{"content": "", "tool_calls": [_call("echo", '{"value": "x"}')]}])
    run = await _run(client, _registry())
    assert run.results[0].content == "echoed x"


async def test_unparseable_arguments_do_not_crash() -> None:
    client = StubClient([{"content": "", "tool_calls": [_call("echo", "not json")]}])
    run = await _run(client, _registry())
    assert run.ran  # ran with empty arguments rather than exploding


async def test_a_backend_failure_is_reported_not_raised() -> None:
    class Failing(StubClient):
        async def chat(self, *a, **k):
            raise OllamaError("daemon is down")

    run = await _run(Failing([]), _registry())
    assert not run.ran
    assert run.error is not None and "daemon is down" in run.error
    assert (run.as_meta() or {}).get("error")


async def test_an_unknown_tool_is_reported_back_to_the_model() -> None:
    client = StubClient([{"content": "", "tool_calls": [_call("nope", {})]}])
    run = await _run(client, _registry())
    assert not run.results[0].ok
    assert "echo" in run.results[0].content


async def test_write_tools_can_be_withheld() -> None:
    registry = ToolRegistry([
        Tool("w", "d", {"type": "object", "properties": {}}, lambda: "", "basics", writes=True)
    ])
    client = StubClient([])
    run = await dispatch.run(
        text="x", registry=registry, families={"basics"}, dispatcher_spec=object(),
        client=client, allow_writes=False,
    )
    assert not run.ran
    assert client.requests == [], "no schemas left, so no dispatch call is worth making"


async def test_meta_reports_what_ran() -> None:
    client = StubClient([{"content": "", "tool_calls": [_call("echo", {"value": "hi"})]}])
    meta = (await _run(client, _registry())).as_meta()
    assert meta is not None
    assert meta["families"] == ["basics"]
    assert meta["calls"][0]["name"] == "echo"
    assert meta["calls"][0]["ok"] is True


@pytest.mark.parametrize("bad", [{"function": {}}, {"function": {"name": ""}}, {}])
async def test_malformed_tool_calls_are_skipped(bad: dict[str, Any]) -> None:
    run = await _run(StubClient([{"content": "", "tool_calls": [bad]}]), _registry())
    assert not run.ran
