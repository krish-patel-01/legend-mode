"""Guardrail tests.

Two halves, and the second matters more than the first. Firing correctly is easy;
*not* firing on something the guard has misread is what keeps a wrong "verified fact"
out of the prompt, which would be worse than having no guard at all.
"""

from __future__ import annotations

import pytest

from app.guardrails import Grounding, contradicts, fmt, ground, safe_eval


# --- safe evaluation ----------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected",
    [("17*23", 391), ("2+2", 4), ("10/4", 2.5), ("(3+4)*2", 14), ("2**8", 256), ("7%3", 1)],
)
def test_safe_eval_arithmetic(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",  # code execution
        "open('x')",
        "1/0",
        "2**999999999",  # would hang on a naive evaluator
        "x + 1",
        "[1,2,3]",
        "'a'*10",
    ],
)
def test_safe_eval_refuses_anything_that_is_not_arithmetic(expr):
    assert safe_eval(expr) is None


def test_fmt_drops_trailing_zeros():
    assert fmt(391.0) == "391"
    assert fmt(2.5) == "2.5"


# --- guards that should fire --------------------------------------------------------


def test_arithmetic_grounds_the_case_the_models_got_wrong():
    g = ground("What is 17 * 23?")
    assert g is not None and g.kind == "arithmetic" and g.value == "391"


@pytest.mark.parametrize(
    "text", ["what is 17 x 23", "calculate 17 × 23", "17*23", "how much is 17 * 23 ?"]
)
def test_arithmetic_phrasings(text):
    g = ground(text)
    assert g is not None and g.value == "391"


def test_percent_of():
    g = ground("how much is 15 percent of 240")
    assert g is not None and g.kind == "percent_of" and g.value == "36"


def test_percent_off_handles_the_shirt_problem():
    g = ground("if a shirt is 40 dollars with 25% off, what do I pay in the end")
    assert g is not None and g.kind == "percent_off" and g.value == "30"


def test_leap_year_is_366_not_365():
    # The 350M answered 365 in one sample and 366 in another.
    g = ground("how many days in a leap year")
    assert g is not None and g.kind == "calendar" and g.value == "366"
    assert "366" in g.claim


def test_ist_is_not_eastern_standard_time():
    # Observed answers: "Eastern Standard Time" and "Central European Time".
    g = ground("what time zone is IST")
    assert g is not None and g.kind == "timezone"
    assert "+05:30" in g.claim
    # The abbreviation genuinely is ambiguous; the claim should say so rather than
    # asserting one meaning as if it were the only one.
    assert "Israel" in g.claim


def test_unit_conversion():
    g = ground("convert 10 km to miles")
    assert g is not None and g.kind == "unit_convert" and g.value.startswith("6.21")


def test_temperature_conversion():
    g = ground("convert 100 c to f")
    assert g is not None and g.value == "212"


# --- guards that must stay quiet ----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "what is 17 * 23 in roman numerals",   # guard can't answer the actual question
        "how many boxes do I have if I have two boxes with one box inside each?",
        "explain what a REST API is",
        "who are you?",
        "what is the meaning of 42",           # a number, but no operation
        "tell me about the year 2024",
        "hi",
        "",
    ],
)
def test_guards_do_not_fire_on_questions_they_cannot_answer(text):
    assert ground(text) is None


def test_word_problems_are_left_to_the_reasoning_tier():
    # This is the case that started all of it. There is no correct deterministic
    # parse, so the guard must decline and let routing send it to `think`.
    assert ground("How many boxes do I have if I have two boxes with one box inside each?") is None


def test_very_long_input_is_skipped():
    assert ground("what is 2+2 " + "padding " * 200) is None


# --- contradiction detection --------------------------------------------------------


def test_contradicts_spots_a_wrong_restatement():
    g = Grounding(kind="arithmetic", claim="17 * 23 equals exactly 391.", value="391")
    assert contradicts("17 times 23 is 371.", g) is True
    assert contradicts("That works out to 391.", g) is False


def test_contradicts_tolerates_thousands_separators():
    g = Grounding(kind="arithmetic", claim="x", value="1234")
    assert contradicts("the answer is 1,234", g) is False


def test_contradicts_is_silent_when_there_is_no_number_to_compare():
    g = Grounding(kind="arithmetic", claim="x", value="391")
    assert contradicts("I'm not sure about that.", g) is False


def test_contradicts_skips_prose_groundings():
    # Timezone claims are prose; matching prose against prose is the unreliable step
    # this module exists to avoid, so it must not report a contradiction either way.
    g = Grounding(kind="timezone", claim="IST is UTC+05:30, India Standard Time.",
                  value="UTC+05:30")
    assert contradicts("IST is Eastern Standard Time.", g) is False
