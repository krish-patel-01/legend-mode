"""Tests for the non-streaming response envelope.

`X-Legend-Route` was sent on the streaming paths but not this one, so a client that did
not ask for SSE got the routing metadata in the body only — while the README promised the
header on every reply. The console streams, so nothing in the browser ever revealed it.
These tests pin the header to the one function every non-streaming return goes through.
"""

from __future__ import annotations

import json
from typing import Any

from app.api import _route_header, _to_openai_response
from app.router.types import RouteDecision


def _decision(**overrides: Any) -> RouteDecision:
    # Built then copied rather than splatted into the constructor: `stage` is a Literal,
    # and widening it through a **dict costs the type checker that guarantee.
    base = RouteDecision(
        route="chat", model="instruct-q3", stage="rules", reason="conversational"
    )
    return base.model_copy(update=overrides) if overrides else base


def _body(response) -> dict:
    return json.loads(bytes(response.body))


def test_the_non_streaming_reply_carries_the_route_header() -> None:
    response = _to_openai_response(
        {"message": {"role": "assistant", "content": "hi"}}, "legend/general", "id-1",
        _decision(route="trivial", model="general"),
    )
    assert response.headers["X-Legend-Route"] == "general;stage=rules;route=trivial"


def test_the_header_and_the_body_agree_about_the_route() -> None:
    """Two copies of the same fact, so a client reading either gets the same answer."""
    decision = _decision()
    response = _to_openai_response({"message": {}}, "legend/instruct-q3", "id-2", decision)

    meta = _body(response)["x_legend_route"]
    header = response.headers["X-Legend-Route"]
    assert header == f"{meta['model']};stage={meta['stage']};route={meta['route']}"
    assert header == _route_header(decision)


def test_the_body_still_looks_like_an_openai_completion() -> None:
    """Wrapping the dict in a JSONResponse must not change what clients parse."""
    body = _body(
        _to_openai_response(
            {
                "message": {"role": "assistant", "content": "hello"},
                "prompt_eval_count": 12,
                "eval_count": 5,
            },
            "legend/general",
            "id-3",
            _decision(),
        )
    )

    assert body["id"] == "id-3"
    assert body["object"] == "chat.completion"
    assert body["model"] == "legend/general"
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }


def test_a_tool_call_still_reports_finish_reason_tool_calls() -> None:
    body = _body(
        _to_openai_response(
            {"message": {"role": "assistant", "tool_calls": [{"function": {"name": "x"}}]}},
            "legend/general",
            "id-4",
            _decision(),
        )
    )
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_a_decision_with_every_optional_field_still_serialises() -> None:
    """JSONResponse serialises directly, without FastAPI's encoder in front of it, so
    anything as_meta() can put in the body has to be JSON-native on its own."""
    body = _body(
        _to_openai_response(
            {"message": {}},
            "legend/think",
            "id-5",
            _decision(
                effort={"level": "careful", "max_tokens": 512, "why": "reasoning"},
                adjudicated={"verdict": "kept"},
                retrieved=["Welcome.md#2"],
                remembered="the user's name",
                tools={"called": ["web_search"]},
                scores={"chat": 0.81},
            ),
        )
    )

    meta = body["x_legend_route"]
    assert meta["effort"]["level"] == "careful"
    assert meta["adjudicated"] == {"verdict": "kept"}
    assert meta["retrieved"] == ["Welcome.md#2"]
    assert meta["remembered"] == "the user's name"
    assert meta["tools"] == {"called": ["web_search"]}
