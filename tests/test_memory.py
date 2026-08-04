"""Tests for fact capture and consolidation.

Capture is tested for restraint first. A store that fills with conversational noise is
worse than an empty one — noise is exactly what makes retrieval degrade answers, which is
the 5.0-point GPQA drop the gating elsewhere exists to avoid.
"""

from __future__ import annotations

import pytest

from app import memory
from app.retrieval.store import VectorStore


# --- what gets captured ------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_key",
    [
        ("my name is Krish", "my name"),
        ("My name's Krish", "my name"),
        ("i'm called Krish", "my name"),
        ("I work on Legend Mode", "where i work"),
        ("i am building a local model router", "what i am working on"),
        ("I live in Pune", "where i live"),
        ("I prefer concise answers", None),
        ("remember that I use uv, not pip", None),
        ("Remember: the server runs on port 8000", None),
        ("don't forget I have a deadline on Friday", None),
    ],
)
def test_durable_statements_are_captured(text, expected_key):
    fact = memory.extract(text)
    assert fact is not None, text
    assert fact.key == expected_key


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hi there",
        "what is the capital of France",
        "how many boxes do I have",
        "can you explain recursion",
        # Questions that *contain* a capture pattern. These are the dangerous ones: they
        # look exactly like statements to a regex.
        "what is my name?",
        "do you remember my name",
        "what am I working on?",
        "where do I live?",
        # Long enough to be conversation rather than a fact.
        "my name is Krish and " + "a lot of context " * 20,
    ],
)
def test_questions_and_chatter_are_not_captured(text):
    assert memory.extract(text) is None


def test_the_captured_text_is_the_users_own_words():
    """Verbatim, never paraphrased — nothing generated may enter the store."""
    assert memory.extract("my name is Krish").text == "my name is Krish"
    assert memory.extract("remember that I use uv, not pip").text == "I use uv, not pip"


# --- consolidation -----------------------------------------------------------


class _StubClient:
    """Returns a deterministic vector; no model, no Ollama."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, spec, texts):
        self.calls += 1
        return [[float(len(t) % 7), 1.0, 0.5] for t in texts]


class _Spec:
    tag = "legend/embed"
    alias = "embed"


@pytest.fixture
def mem(tmp_path):
    store = VectorStore(tmp_path / "corpus.db")
    yield memory.MemoryStore(_StubClient(), _Spec(), store), store
    store.close()


async def test_a_restated_fact_replaces_the_old_one(mem):
    """Two answers to "what is my name" is worse than none. mem0 makes this call with an
    LLM function call; a key comparison gets the same result for nothing."""
    store_api, store = mem
    await store_api.remember(memory.Fact("my name is Krish", "my name"))
    saved = await store_api.remember(memory.Fact("my name is Kris", "my name"))

    entries = store_api.entries()
    assert len(entries) == 1
    assert entries[0]["text"] == "my name is Kris"
    assert saved is not None and saved["replaced"] == "my name is Krish"


async def test_keyless_facts_accumulate(mem):
    """Preferences are not mutually exclusive, so they sit beside each other."""
    store_api, _ = mem
    await store_api.remember(memory.Fact("I prefer concise answers", None))
    await store_api.remember(memory.Fact("I use uv, not pip", None))
    assert len(store_api.entries()) == 2


async def test_storing_a_known_fact_twice_is_a_no_op(mem):
    store_api, _ = mem
    await store_api.remember(memory.Fact("my name is Krish", "my name"))
    assert await store_api.remember(memory.Fact("my name is Krish", "my name")) is None
    assert len(store_api.entries()) == 1


async def test_a_memory_can_be_forgotten(mem):
    store_api, _ = mem
    saved = await store_api.remember(memory.Fact("my name is Krish", "my name"))
    assert saved is not None
    assert store_api.forget(int(saved["id"])) is True
    assert store_api.entries() == []
    assert store_api.forget(9999) is False


async def test_memories_share_the_document_store(mem):
    """Same table, different source — so both retrieval gates and the score threshold
    apply to memories exactly as they do to ingested files."""
    store_api, store = mem
    store.set_embedder("legend/embed", 3)
    store.replace_source("notes.md", [("H", "some document text here")], [[1.0, 0.0, 0.0]])
    await store_api.remember(memory.Fact("my name is Krish", "my name"))

    assert sorted(store.sources) == ["memory", "notes.md"]
    assert len(store) == 2


async def test_capture_costs_exactly_one_embed_and_no_model_call(mem):
    store_api, _ = mem
    await store_api.remember(memory.Fact("my name is Krish", "my name"))
    assert store_api._client.calls == 1
