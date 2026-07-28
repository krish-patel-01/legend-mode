from __future__ import annotations

from app.history import HistoryStore
from app.router.types import RouteDecision


def _decision(**kw) -> RouteDecision:
    return RouteDecision(
        route=kw.get("route", "chat"),
        model=kw.get("model", "small"),
        stage=kw.get("stage", "rules"),
        reason=kw.get("reason", "test"),
        elapsed_ms=1.23,
    )


def test_recent_returns_newest_first():
    store = HistoryStore(maxlen=10)
    store.add(prompt="first", tag="legend/small", decision=_decision())
    store.add(prompt="second", tag="legend/small", decision=_decision())
    entries = store.recent()
    assert [e["prompt"] for e in entries] == ["second", "first"]


def test_ring_buffer_bounded():
    store = HistoryStore(maxlen=3)
    for i in range(5):
        store.add(prompt=str(i), tag="legend/small", decision=_decision())
    entries = store.recent(limit=10)
    assert [e["prompt"] for e in entries] == ["4", "3", "2"]


def test_error_entries_recorded():
    store = HistoryStore()
    store.add(prompt="oops", tag="legend/large", decision=_decision(), error="boom")
    entry = store.recent()[0]
    assert entry["error"] == "boom"


def test_prompt_truncated():
    store = HistoryStore()
    store.add(prompt="x" * 5000, tag="legend/small", decision=_decision())
    assert len(store.recent()[0]["prompt"]) == 300
