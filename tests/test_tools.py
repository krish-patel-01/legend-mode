"""Tests for the tool gate, registry and the deterministic family.

The gate cases are not invented. `NO_TOOLS` is the exact prompt set that damaged the
models in the measurement recorded in `app/tools/gate.py` — the ones that produced "I'm
sorry, but I can't provide that information" for *what is the capital of France* once
tool definitions were attached. Every one of them must gate to nothing, because the gate
is the only thing standing between those prompts and that behaviour.
"""

from __future__ import annotations

import pytest

from app.tools import basics, gate
from app.tools.registry import MAX_RESULT_CHARS, Tool, ToolRegistry

# --- the gate -----------------------------------------------------------------

NO_TOOLS = [
    # The measured regression set.
    "What is the capital of France?",
    "Explain what a REST API is in one line.",
    "Write me a haiku about winter.",
    "What does idempotent mean?",
    "How many days are in a leap year?",
    "Tell me a joke.",
    # Ordinary chat that mentions a tool-ish word without asking for one.
    "how do timezones work",
    "explain how to read a file in python",
    "what is a directory",
    "how does web search work",
    "why do people take notes",
    "hi there",
    "thanks!",
    "who are you",
]


@pytest.mark.parametrize("prompt", NO_TOOLS)
def test_gate_declines_prompts_that_need_no_tool(prompt: str) -> None:
    assert gate.wanted(prompt) == set()


@pytest.mark.parametrize(
    ("prompt", "family"),
    [
        ("what time is it", gate.BASICS),
        ("what's the time in Tokyo", gate.BASICS),
        ("what is today's date", gate.BASICS),
        ("what day is it today", gate.BASICS),
        ("calculate 240 * 0.75", gate.BASICS),
        ("what is 17 * 23", gate.BASICS),
        ("how much disk space do I have", gate.BASICS),
        ("read the file at /etc/hosts", gate.FILES),
        ("show me what's in the folder", gate.FILES),
        ("list the files in this directory", gate.FILES),
        ("search the web for LFM2.5 release notes", gate.WEB),
        ("what's the latest news about Artemis", gate.WEB),
        ("what's the weather in Mumbai", gate.WEB),
        ("remember that I take my coffee black", gate.NOTES),
        ("make a note that the meeting moved to Thursday", gate.NOTES),
    ],
)
def test_gate_recognises_real_requests(prompt: str, family: str) -> None:
    assert family in gate.wanted(prompt)


def test_a_literal_path_beats_the_discussion_guard() -> None:
    """"How do I..." normally means prose, but not when a real path is named."""
    assert gate.wanted("how do I read k:/Projects/notes.md") == {gate.FILES}
    assert gate.wanted("how do I read a file in python") == set()


def test_a_url_is_concrete_enough_to_fetch() -> None:
    assert gate.wanted("what does https://example.com/page say") == {gate.WEB}


def test_grounded_requests_get_no_tools() -> None:
    """A guardrail already computed the exact answer; a tool could only disagree."""
    assert gate.wanted("what is 17 * 23", grounded=True) == set()


def test_enabled_is_a_hard_allowlist() -> None:
    assert gate.wanted("read the file at /etc/hosts", enabled={gate.WEB}) == set()
    assert gate.wanted("read the file at /etc/hosts", enabled={gate.FILES}) == {gate.FILES}


def test_empty_input_gates_to_nothing() -> None:
    assert gate.wanted("") == set()
    assert gate.wanted("   ") == set()


# --- the registry -------------------------------------------------------------


def _registry() -> ToolRegistry:
    return ToolRegistry(basics.tools())


async def test_unknown_tool_names_the_real_ones() -> None:
    result = await _registry().invoke("get_wether", {"city": "Tokyo"})
    assert not result.ok
    assert "calculate" in result.content  # tells the model what does exist


async def test_bad_arguments_are_reported_not_raised() -> None:
    result = await _registry().invoke("calculate", {"wrong_kwarg": "2+2"})
    assert not result.ok
    assert "calculate" in result.content


async def test_a_raising_tool_becomes_a_result() -> None:
    def boom() -> str:
        raise RuntimeError("disk on fire")

    registry = ToolRegistry([
        Tool("boom", "d", {"type": "object", "properties": {}}, boom, "basics")
    ])
    result = await registry.invoke("boom", {})
    assert not result.ok
    assert "disk on fire" in result.content


async def test_long_results_are_truncated_and_say_so() -> None:
    registry = ToolRegistry([
        Tool(
            "big",
            "d",
            {"type": "object", "properties": {}},
            lambda: "x" * (MAX_RESULT_CHARS * 2),
            "basics",
        )
    ])
    result = await registry.invoke("big", {})
    assert result.truncated
    assert len(result.content) == MAX_RESULT_CHARS
    assert "truncated" in result.as_message()["content"]


async def test_async_tools_are_awaited() -> None:
    async def slow() -> str:
        return "done"

    registry = ToolRegistry([
        Tool("slow", "d", {"type": "object", "properties": {}}, slow, "basics")
    ])
    assert (await registry.invoke("slow", {})).content == "done"


def test_families_filter_and_write_flag() -> None:
    registry = ToolRegistry([
        Tool("r", "d", {"type": "object", "properties": {}}, lambda: "", "files"),
        Tool("w", "d", {"type": "object", "properties": {}}, lambda: "", "files", writes=True),
    ])
    assert len(registry.schemas({"files"})) == 2
    assert len(registry.schemas({"files"}, allow_writes=False)) == 1
    assert registry.schemas({"web"}) == []


def test_duplicate_names_are_rejected() -> None:
    tool = Tool("dup", "d", {"type": "object", "properties": {}}, lambda: "", "basics")
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([tool, tool])


def test_schema_shape_matches_what_ollama_expects() -> None:
    schema = basics.tools()[0].schema()
    assert schema["type"] == "function"
    assert set(schema["function"]) == {"name", "description", "parameters"}


# --- the deterministic tools --------------------------------------------------


def test_calculate_uses_the_same_evaluator_as_the_guardrails() -> None:
    assert "391" in basics.calculate("17 * 23")
    assert "170" in basics.calculate("(240 * 0.75) - 10")


def test_calculate_declines_rather_than_guessing() -> None:
    out = basics.calculate("the number of boxes")
    assert "not an arithmetic expression" in out


def test_calculate_refuses_to_execute_code() -> None:
    assert "not an arithmetic expression" in basics.calculate("__import__('os').getcwd()")


def test_an_unknown_place_still_reports_the_local_time() -> None:
    """A model guess must not cost the user an answer."""
    out = basics.now("Middle Earth")
    assert "Middle Earth" in out
    assert "local time" in out


def test_time_defaults_to_local_not_utc() -> None:
    """The bug this parameter was renamed to prevent: `what day is it` answering in UTC."""
    assert "local time" in basics.now()
    assert "local time" in basics.now("   ")


def test_cities_resolve_without_the_user_knowing_iana_names() -> None:
    for city in ("Tokyo", "new york", "Mumbai"):
        out = basics.now(city)
        assert city.lower() in out.lower() or "timezone data" in out


def test_a_real_iana_name_is_still_accepted() -> None:
    out = basics.now("Asia/Kolkata")
    assert "Asia/Kolkata" in out or "timezone data" in out


def test_system_status_reports_disk_and_cores() -> None:
    out = basics.system_status()
    assert "GB free" in out
    assert "cores" in out
