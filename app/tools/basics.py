"""The deterministic family: clock, arithmetic, machine state.

Built first on purpose. These have no security surface, no network, no new dependency and
no ambiguity about whether the answer is right, which makes them the cheapest possible way
to find out whether the dispatch loop and the gate actually work. Proving a mechanism
*fires* before trusting it with anything that matters is the pattern the rest of this
project already follows — see the note in `app/adjudicate.py` about a critic that scored
"unsure" on 6 of 8 pairs and charged 45 seconds for it without erroring once.

`calculate` deliberately calls `app.guardrails.safe_eval` rather than growing its own
evaluator. Two implementations of arithmetic can disagree, and if they ever did, the
grounding note injected before generation would contradict the tool result injected after
it — with the model holding both and no way to tell which to believe.
"""

from __future__ import annotations

import os
import platform
import shutil
from datetime import UTC, datetime

from app.guardrails import fmt, safe_eval
from app.tools.registry import Tool

# Cities people actually ask about, mapped to IANA zones. The dispatcher is asked for a
# *city* rather than a zone name for a measured reason — see the note on the schema below
# — so this is what turns its answer into something `zoneinfo` accepts. A miss here is not
# fatal: an unrecognised value is tried against zoneinfo directly, which covers the case
# where the model does supply "Asia/Tokyo".
_CITY_ZONES = {
    "utc": "UTC", "gmt": "UTC",
    "london": "Europe/London", "paris": "Europe/Paris", "berlin": "Europe/Berlin",
    "madrid": "Europe/Madrid", "rome": "Europe/Rome", "moscow": "Europe/Moscow",
    "new york": "America/New_York", "nyc": "America/New_York",
    "los angeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles", "chicago": "America/Chicago",
    "toronto": "America/Toronto", "sao paulo": "America/Sao_Paulo",
    "tokyo": "Asia/Tokyo", "osaka": "Asia/Tokyo", "seoul": "Asia/Seoul",
    "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong", "singapore": "Asia/Singapore",
    "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata", "bengaluru": "Asia/Kolkata", "kolkata": "Asia/Kolkata",
    "chennai": "Asia/Kolkata", "hyderabad": "Asia/Kolkata", "pune": "Asia/Kolkata",
    "ahmedabad": "Asia/Kolkata", "india": "Asia/Kolkata",
    "dubai": "Asia/Dubai", "karachi": "Asia/Karachi", "dhaka": "Asia/Dhaka",
    "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
    "auckland": "Pacific/Auckland", "johannesburg": "Africa/Johannesburg",
    "cairo": "Africa/Cairo", "lagos": "Africa/Lagos", "nairobi": "Africa/Nairobi",
}


def _resolve(city: str) -> str | None:
    key = city.strip().lower().replace("_", " ")
    if key in _CITY_ZONES:
        return _CITY_ZONES[key]
    # The model sometimes supplies a real IANA name anyway; accept it if zoneinfo does.
    try:
        from zoneinfo import ZoneInfo

        ZoneInfo(city)
    except Exception:
        return None
    return city


def now(city: str = "") -> str:
    """Current date and time — local by default, or in a named city."""
    utc = datetime.now(UTC)
    local = utc.astimezone()
    local_text = f"{local:%A, %d %B %Y, %H:%M:%S} local time (UTC{local:%z})"
    if not city.strip():
        return local_text

    zone = _resolve(city)
    if zone is None:
        # An unknown place is a model guess, not a system fault, and so is a missing
        # tzdata on Windows. Both end the same way: give the time that *is* known rather
        # than failing the turn, and name what was not understood.
        return f"I don't know the timezone for {city!r}. Locally it is {local_text}."

    try:
        from zoneinfo import ZoneInfo

        at = utc.astimezone(ZoneInfo(zone))
    except Exception:
        return f"I don't have timezone data installed. Locally it is {local_text}."
    return f"{at:%A, %d %B %Y, %H:%M:%S} in {city} ({zone}, UTC{at:%z})"


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly."""
    value = safe_eval(expression)
    if value is None:
        return (
            f"{expression!r} is not an arithmetic expression I can evaluate. "
            f"Only numbers and + - * / // % ** are supported."
        )
    return f"{expression} = {fmt(value)}"


def system_status() -> str:
    """Disk, CPU and platform for the machine this assistant runs on."""
    usage = shutil.disk_usage(os.getcwd())
    gb = 1024**3
    return (
        f"{platform.system()} {platform.release()}, "
        f"{os.cpu_count()} logical cores. "
        f"Disk on the working drive: {usage.free / gb:.1f} GB free "
        f"of {usage.total / gb:.1f} GB."
    )


def tools() -> list[Tool]:
    # Descriptions are written *at the model*: each says when to reach for the tool,
    # because that is the only decision the dispatcher makes. "Returns the current time"
    # describes the function; "Use when asked what time or date it is" describes the
    # trigger, and the trigger is what has to match a user's phrasing.
    return [
        Tool(
            name="get_time",
            description=(
                "Use when asked the current date, day or time. Do not guess the time; "
                "call this. Not for disk space, memory or anything else about the computer."
            ),
            parameters={
                "type": "object",
                # **The parameter asks for a city, not a timezone, and that is the fix.**
                # As `timezone_name` it drew UTC: asked "what day is it today" the
                # dispatcher passed "UTC" and the answer came back a day early — Tuesday
                # in UTC while it was already Wednesday where the user is. Wording the
                # description harder did not shift it ("what's the time" went local,
                # "what day is it today" still went UTC).
                #
                # A city has no such attractor. There is no city called UTC, so the only
                # way to fill this field is to name a place the user actually mentioned,
                # and an empty field is the natural answer when they mentioned none. It
                # also matches how the question gets asked — "what time is it in Tokyo",
                # never "in Asia/Tokyo". `_resolve` still accepts a real IANA name for the
                # times the model supplies one anyway.
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "The city the user named, for example Tokyo or New York. "
                            "Leave empty when they did not name one — that gives their "
                            "own local time, which is what a plain question about the "
                            "time or the date means."
                        ),
                    }
                },
                "required": [],
            },
            run=now,
            family="basics",
        ),
        Tool(
            name="calculate",
            description=(
                "Use for exact arithmetic on numbers the user gave you. Supports "
                "+ - * / // % and **. Example expression: (240 * 0.75) - 10"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The expression alone, no words. Example: 3 * 12 - 17",
                    }
                },
                "required": ["expression"],
            },
            run=calculate,
            family="basics",
        ),
        Tool(
            name="system_status",
            # Asked "how much disk space do I have", the dispatcher called get_time —
            # twice — and the answer became "I don't have access to your local disk
            # space". A tool taking no arguments appears to be a weaker candidate to a
            # model this size than one it can fill a field in, so the description carries
            # the weight instead, quoting the phrasings a user actually types.
            description=(
                "Use for any question about this computer's hardware or storage: "
                "'how much disk space do I have', 'how much free space is left', "
                "'how much RAM', 'how many CPU cores', 'what OS is this'. "
                "This is the only tool that knows anything about the machine."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            run=system_status,
            family="basics",
        ),
    ]


__all__: list[str] = ["calculate", "now", "system_status", "tools"]
