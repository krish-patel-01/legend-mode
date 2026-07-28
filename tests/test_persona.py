from __future__ import annotations

from app.persona import build_system_prompt, ensure_system_prompt


def test_no_name_set_says_so_honestly():
    assert "no name yet" in build_system_prompt(None)
    assert "no name yet" in build_system_prompt(None, "brief")


def test_name_set_is_used_directly():
    for style in ("full", "brief"):
        prompt = build_system_prompt("Vex", style)
        assert "Your name is Vex." in prompt
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
