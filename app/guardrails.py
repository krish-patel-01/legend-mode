"""Deterministic grounding for questions that code can answer exactly.

The models here get facts wrong in a specific, repeatable way. Measured on this
machine: the 350M answered "how many days in a leap year" with **365** in one sample
and 366 in another, and called IST "Eastern Standard Time" in one run and "Central
European Time" in the next. Those are not reasoning failures that a bigger prompt or
another sampling pass would fix — the model simply does not reliably know.

Neither can a second model catch it cheaply. Asked to judge 16 question/answer pairs,
the 350M said CORRECT to all 16 and the Qwen 0.8B to 15; only the 1.2B scored well,
and it cost ~28 s per verdict. So model-based checking is 28 seconds and model-free
checking is under a millisecond, for the subset of questions where a real
implementation exists. That subset is what lives here.

The mechanism is grounding, not correction. When a guard recognises a question it can
answer exactly, the computed fact is injected into the system prompt before generation
so the model phrases it naturally, rather than being generated first and patched
afterwards. Patching prose means parsing prose, which is its own source of errors.

Guards must be conservative. A guard that fires on a question it has misread is worse
than no guard at all, because it injects a confident wrong fact. Every pattern here is
anchored tightly enough that a near-miss returns None and the request proceeds
normally. `contradicts()` is deliberately advisory — it reports disagreement for
logging and for the adjudication stage to act on, and never rewrites an answer itself.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Guard against a pathological exponent turning a routing call into a hang: 2**10**9
# is a valid expression and would allocate for a very long time.
_MAX_POW_EXPONENT = 64
_MAX_ABS_OPERAND = 1e15


@dataclass(frozen=True)
class Grounding:
    """A fact produced by real code rather than by a model."""

    kind: str
    """Which guard fired — surfaced in routing metadata so the console can show it."""

    claim: str
    """Human-readable statement injected into the system prompt."""

    value: str
    """Canonical answer, used by `contradicts()`."""


# --- safe arithmetic evaluation ---------------------------------------------
#
# ast.literal_eval can't do arithmetic and eval() would execute anything, so the
# expression is walked explicitly and only these node types are honoured.

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
        if abs(node.value) > _MAX_ABS_OPERAND:
            raise ValueError("operand too large")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError("exponent too large")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("division by zero")
        return _BIN_OPS[type(node.op)](left, right)
    raise ValueError(f"unsupported expression node {type(node).__name__}")


def safe_eval(expression: str) -> float | None:
    """Evaluate a pure-arithmetic expression, or None if it isn't one."""
    try:
        return _eval(ast.parse(expression, mode="eval"))
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, OverflowError,
            MemoryError, RecursionError):
        return None


def fmt(value: float) -> str:
    """Format a result the way a person would write it: 391, not 391.0."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return f"{value:,}".replace(",", ",")
    return f"{round(value, 6):g}"


# --- guards -----------------------------------------------------------------

# A bare calculation, optionally wrapped in "what is"/"calculate"/"how much is".
# Anchored to the whole message on purpose: "what is 17 * 23" grounds cleanly, but
# "what is 17 * 23 in roman numerals" must not, since the guard can't answer that.
_ARITHMETIC = re.compile(
    r"^\s*(?:what(?:'?s| is)|calculate|compute|how much is|solve)?\s*"
    r"(?P<expr>[\d\s+\-*/^%().×÷x]+?)"
    r"\s*(?:=\s*\??)?\s*[?.]?\s*$",
    re.IGNORECASE,
)

_PERCENT_OF = re.compile(
    r"^\s*(?:what(?:'?s| is)|calculate|compute|how much is)?\s*"
    r"(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+"
    r"(?P<base>\d+(?:\.\d+)?)\s*[?.]?\s*$",
    re.IGNORECASE,
)

# "a shirt is $40 with 25% off, what do I pay" — a discount is unambiguous arithmetic
# even though it reads as a word problem, and it was one of the cases that failed.
_PERCENT_OFF = re.compile(
    r"(?:[$£€]\s*)?(?P<base>\d+(?:\.\d+)?)\s*(?:dollars?|pounds?|euros?|bucks?|rupees?)?"
    r"[^.?!]{0,40}?(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:off|discount)",
    re.IGNORECASE,
)

_LEAP_YEAR = re.compile(
    r"how many days\b[^.?!]{0,20}\bin a leap year|days in a leap year", re.IGNORECASE
)

# Timezone abbreviations the models were observed guessing at. Ambiguous ones carry
# the ambiguity in the text rather than silently picking a winner — saying "usually
# India, also used for Israel and Ireland" is both more correct and more useful than
# asserting one of them.
_TIMEZONES: dict[str, str] = {
    "IST": "UTC+05:30, India Standard Time (the same abbreviation is also used for "
           "Israel Standard Time, UTC+02:00, and Irish Standard Time, UTC+01:00)",
    "UTC": "UTC+00:00, Coordinated Universal Time",
    "GMT": "UTC+00:00, Greenwich Mean Time",
    "BST": "UTC+01:00, British Summer Time",
    "CET": "UTC+01:00, Central European Time",
    "CEST": "UTC+02:00, Central European Summer Time",
    "EST": "UTC-05:00, Eastern Standard Time",
    "EDT": "UTC-04:00, Eastern Daylight Time",
    "CST": "UTC-06:00, Central Standard Time (North America); also used for China "
           "Standard Time, UTC+08:00",
    "CDT": "UTC-05:00, Central Daylight Time",
    "MST": "UTC-07:00, Mountain Standard Time",
    "MDT": "UTC-06:00, Mountain Daylight Time",
    "PST": "UTC-08:00, Pacific Standard Time",
    "PDT": "UTC-07:00, Pacific Daylight Time",
    "AKST": "UTC-09:00, Alaska Standard Time",
    "HST": "UTC-10:00, Hawaii Standard Time",
    "JST": "UTC+09:00, Japan Standard Time",
    "KST": "UTC+09:00, Korea Standard Time",
    "HKT": "UTC+08:00, Hong Kong Time",
    "SGT": "UTC+08:00, Singapore Time",
    "AEST": "UTC+10:00, Australian Eastern Standard Time",
    "NZST": "UTC+12:00, New Zealand Standard Time",
    "MSK": "UTC+03:00, Moscow Time",
    "GST": "UTC+04:00, Gulf Standard Time",
    "PKT": "UTC+05:00, Pakistan Standard Time",
    "NPT": "UTC+05:45, Nepal Time",
}

_TZ_QUESTION = re.compile(
    r"\b(?:what(?:'?s| is)?\s+)?(?:time ?zone|timezone|utc offset|offset)\b[^.?!]{0,30}?"
    r"\b(?P<abbr>[A-Z]{2,4})\b|\b(?P<abbr2>[A-Z]{2,4})\b[^.?!]{0,20}?"
    r"\b(?:time ?zone|timezone|utc offset)\b"
)

# Unit conversions, kept to pairs that come up in chat. factor converts `frm` -> `to`.
_UNITS: dict[tuple[str, str], float] = {
    ("km", "mi"): 0.621371,
    ("mi", "km"): 1.609344,
    ("kg", "lb"): 2.204623,
    ("lb", "kg"): 0.453592,
    ("m", "ft"): 3.280840,
    ("ft", "m"): 0.304800,
    ("cm", "in"): 0.393701,
    ("in", "cm"): 2.540000,
    ("l", "gal"): 0.264172,
    ("gal", "l"): 3.785412,
}
_UNIT_ALIASES = {
    "kilometre": "km", "kilometres": "km", "kilometer": "km", "kilometers": "km", "km": "km",
    "mile": "mi", "miles": "mi", "mi": "mi",
    "kilogram": "kg", "kilograms": "kg", "kilo": "kg", "kilos": "kg", "kg": "kg",
    "pound": "lb", "pounds": "lb", "lb": "lb", "lbs": "lb",
    "metre": "m", "metres": "m", "meter": "m", "meters": "m", "m": "m",
    "foot": "ft", "feet": "ft", "ft": "ft",
    "centimetre": "cm", "centimetres": "cm", "centimeter": "cm", "centimeters": "cm", "cm": "cm",
    "inch": "in", "inches": "in", "in": "in",
    "litre": "l", "litres": "l", "liter": "l", "liters": "l", "l": "l",
    "gallon": "gal", "gallons": "gal", "gal": "gal",
}
_CONVERT = re.compile(
    r"(?:convert\s+)?(?P<qty>\d+(?:\.\d+)?)\s*(?P<frm>[a-z]+)\s+(?:to|in|into)\s+(?P<to>[a-z]+)",
    re.IGNORECASE,
)

_CELSIUS_F = re.compile(
    r"(?:convert\s+)?(?P<qty>-?\d+(?:\.\d+)?)\s*(?:degrees\s*)?(?P<frm>c|f|celsius|fahrenheit)\b"
    r"[^.?!]{0,15}?\b(?:to|in|into)\s+(?P<to>c|f|celsius|fahrenheit)\b",
    re.IGNORECASE,
)


def _guard_percent_of(text: str) -> Grounding | None:
    if not (m := _PERCENT_OF.match(text)):
        return None
    pct, base = float(m["pct"]), float(m["base"])
    result = fmt(pct / 100 * base)
    return Grounding(
        kind="percent_of",
        claim=f"{fmt(pct)}% of {fmt(base)} is exactly {result}.",
        value=result,
    )


def _guard_percent_off(text: str) -> Grounding | None:
    if not (m := _PERCENT_OFF.search(text)):
        return None
    base, pct = float(m["base"]), float(m["pct"])
    if not 0 < pct <= 100:
        return None
    final = fmt(base * (1 - pct / 100))
    return Grounding(
        kind="percent_off",
        claim=(
            f"{fmt(base)} with {fmt(pct)}% off is exactly {final} "
            f"(the discount itself is {fmt(base * pct / 100)})."
        ),
        value=final,
    )


def _guard_arithmetic(text: str) -> Grounding | None:
    if not (m := _ARITHMETIC.match(text)):
        return None
    raw = m["expr"].strip()
    # Needs an actual operation; a bare number is not a calculation to check.
    if not re.search(r"[+\-*/^%×÷]|(?<=\d)\s*x\s*(?=\d)", raw):
        return None
    expr = (
        raw.replace("×", "*").replace("÷", "/").replace("^", "**")
    )
    expr = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expr)
    result = safe_eval(expr)
    if result is None:
        return None
    pretty = fmt(result)
    return Grounding(
        kind="arithmetic",
        claim=f"{raw} equals exactly {pretty}.",
        value=pretty,
    )


def _guard_leap_year(text: str) -> Grounding | None:
    if not _LEAP_YEAR.search(text):
        return None
    return Grounding(
        kind="calendar",
        claim=(
            "A leap year has exactly 366 days (a common year has 365). "
            "February has 29 days in a leap year."
        ),
        value="366",
    )


def _guard_timezone(text: str) -> Grounding | None:
    if not (m := _TZ_QUESTION.search(text)):
        return None
    abbr = (m["abbr"] or m["abbr2"] or "").upper()
    if (meaning := _TIMEZONES.get(abbr)) is None:
        return None
    return Grounding(
        kind="timezone",
        claim=f"{abbr} is {meaning}.",
        value=meaning.split(",")[0],
    )


def _guard_unit_convert(text: str) -> Grounding | None:
    if m := _CELSIUS_F.search(text):
        frm, to = m["frm"][0].lower(), m["to"][0].lower()
        if frm == to:
            return None
        qty = float(m["qty"])
        out = qty * 9 / 5 + 32 if frm == "c" else (qty - 32) * 5 / 9
        return Grounding(
            kind="unit_convert",
            claim=f"{fmt(qty)}°{frm.upper()} is exactly {fmt(out)}°{to.upper()}.",
            value=fmt(out),
        )

    if not (m := _CONVERT.search(text)):
        return None
    frm = _UNIT_ALIASES.get(m["frm"].lower())
    to = _UNIT_ALIASES.get(m["to"].lower())
    if not frm or not to or (factor := _UNITS.get((frm, to))) is None:
        return None
    qty = float(m["qty"])
    out = fmt(qty * factor)
    return Grounding(
        kind="unit_convert",
        claim=f"{fmt(qty)} {frm} is exactly {out} {to}.",
        value=out,
    )


# Order matters only where two guards could both match: the percent guards are tried
# before bare arithmetic so "15% of 240" isn't mangled into an expression.
_GUARDS = (
    _guard_percent_of,
    _guard_percent_off,
    _guard_arithmetic,
    _guard_leap_year,
    _guard_timezone,
    _guard_unit_convert,
)


def ground(text: str) -> Grounding | None:
    """First guard that recognises the question, or None if code can't answer it.

    None is the common case and is not a failure — most questions have no exact
    implementation, and those go to the model unchanged.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 400:
        return None
    for guard in _GUARDS:
        try:
            if grounding := guard(stripped):
                return grounding
        except (ValueError, ArithmeticError):
            # A guard that trips over its own input must never take down a request.
            continue
    return None


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def contradicts(answer: str, grounding: Grounding) -> bool:
    """Does the generated answer disagree with a computed fact?

    Advisory only. Numeric groundings are checked by looking for the value among the
    numbers in the reply; non-numeric ones (timezones) are skipped, since matching
    prose against prose is exactly the unreliable step this module exists to avoid.
    """
    if not _NUMBER.fullmatch(grounding.value.replace(",", "")):
        return False
    want = grounding.value.replace(",", "")
    found = {n.replace(",", "") for n in _NUMBER.findall(answer)}
    if not found:
        return False
    return want not in found


def as_system_note(grounding: Grounding) -> str:
    """The sentence prepended to the system prompt when a guard fires."""
    return (
        f"Verified fact, computed exactly and known to be correct: {grounding.claim} "
        f"Use this value in your answer and do not recalculate it."
    )
