from __future__ import annotations

from app.persona import build_system_prompt, ensure_system_prompt


def test_the_brief_tier_is_never_told_the_name():
    """Measured over 16 probes per wording: naming the 350M makes it reply with the name
    and nothing else — 16/16 broken with a style clause, 8/16 with the bare name, 16/16
    with the name as a conditional rule, against 0/16 unnamed. "what is the capital of
    France" came back as "Lucy". It costs nothing to withhold, because identity questions
    route to `chat` now, so this tier only sees greetings."""
    brief = build_system_prompt("Lucy", "brief")
    assert "Lucy" not in brief
    assert brief == build_system_prompt(None, "brief")


def test_the_full_tier_gets_the_name_and_the_personality():
    full = build_system_prompt("Lucy", "full")
    assert "You are Lucy," in full
    assert "direct and a little dry" in full
    # The clause that took the name from 3/12 used to 11/12. Stating the name is not
    # enough; stating when to act on it is.
    assert "asks who you are, you say your name" in full


def test_no_name_set_says_so_honestly():
    assert "no name yet" in build_system_prompt(None)
    assert "no name yet" in build_system_prompt(None, "brief")


def test_name_set_is_used_directly():
    """`brief` is excluded now, and that is measured rather than a preference: naming the
    350M makes it answer with the name and nothing else, 16/16 on one wording. It keeps
    the unnamed prompt — including the "no name yet" clause, which is the honest thing for
    a tier that has not been told one."""
    prompt = build_system_prompt("Vex", "full")
    assert "You are Vex," in prompt
    assert "no name yet" not in prompt


def test_prompt_never_ends_on_the_identity_clause():
    """Regression guard for the worst bug this prompt has had.

    When the brief prompt ended with "...say you don't have one yet.", the 350M
    replied "you don't have one yet." to `hi`, `thanks!`, and `what is the capital of
    France` — it copies the trailing sentence. The identity clause must stay
    mid-prompt, with a behavioral directive after it.
    """
    for style in ("full", "brief"):
        for name in (None, "Vex"):
            last = build_system_prompt(name, style).rstrip().rstrip(".").split(". ")[-1]
            assert "name" not in last.lower()
            assert "answer" in last.lower()


def test_full_prompt_does_not_enumerate_model_brands():
    """Naming a brand primes it: adding "never identify yourself as Qwen" made the
    Qwen tier's identity leak worse, so the full prompt disclaims brands as a
    category instead of listing them."""
    prompt = build_system_prompt(None).lower()
    for brand in ("qwen", "chatgpt", "claude", "gemini", "llama"):
        assert brand not in prompt


def test_brief_style_is_substantially_shorter():
    # The whole point of the brief style: a 350M drowns in the full prompt.
    assert len(build_system_prompt(None, "brief")) < len(build_system_prompt(None)) * 0.6


def test_ensure_system_prompt_prepends_when_absent():
    messages = [{"role": "user", "content": "hi"}]
    result = ensure_system_prompt(messages, None)
    assert result[0]["role"] == "system"
    assert result[1:] == messages


def test_ensure_system_prompt_passes_style_through():
    messages = [{"role": "user", "content": "hi"}]
    brief = ensure_system_prompt(messages, None, "brief")[0]["content"]
    assert brief == build_system_prompt(None, "brief")


def test_ensure_system_prompt_respects_caller_supplied_one():
    messages = [{"role": "system", "content": "custom persona"}, {"role": "user", "content": "hi"}]
    result = ensure_system_prompt(messages, None)
    assert result == messages


def test_ensure_system_prompt_does_not_mutate_input():
    messages = [{"role": "user", "content": "hi"}]
    ensure_system_prompt(messages, None)
    assert messages == [{"role": "user", "content": "hi"}]


def test_the_capabilities_clause_is_off_unless_asked_for():
    assert "find out rather than recall" not in build_system_prompt("Lucy", "full")


def test_the_full_tier_can_be_told_what_it_can_look_up():
    full = build_system_prompt("Lucy", "full", capabilities=True)
    assert "find out rather than recall" in full
    # What it can find out, never how to ask for it. Four tool definitions attached to
    # this tier produced 2/6 spurious calls and 3/6 degraded answers — see the table in
    # app/tools/gate.py — and call syntax in prose is the same information.
    for mechanism in ("function", "tool_call", "JSON", "arguments", "web_search"):
        assert mechanism not in full


def test_the_brief_tier_never_gets_the_capabilities_clause():
    """Same reason it never gets the name: at 350M the prompt already competes with the
    question, and app/api.py escalates off this tier the moment a tool has run."""
    brief = build_system_prompt(None, "brief", capabilities=True)
    assert brief == build_system_prompt(None, "brief")


def test_the_prompt_still_ends_on_a_directive():
    """Note 1 of the module docstring: the 350M treats a trailing sentence as a completion
    prefix, and the capabilities clause is a sentence about the assistant — exactly the
    kind that came back as an answer to "hi". It has to sit mid-prompt in both branches."""
    for name in ("Lucy", None):
        prompt = build_system_prompt(name, "full", capabilities=True)
        assert prompt.rstrip().endswith("unless real depth is asked for.")
