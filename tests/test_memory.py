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


@pytest.mark.parametrize(
    "said,stored",
    [
        ("my name is Krish", "The user's name is Krish"),
        ("i'm called Krish", "The user's name is Krish"),
        ("I work on Legend Mode", "The user said they work on Legend Mode"),
        ("i work at Acme", "The user said they work at Acme"),
        ("I am building a router", "The user said they are building a router"),
        ("I live in Pune", "The user said they live in Pune"),
        ("I prefer concise answers", "The user said they prefer concise answers"),
        ("I always use uv", "The user said they always use uv"),
        ("remember that I use uv, not pip", "The user asked you to remember: I use uv, not pip"),
    ],
)
def test_facts_are_stored_in_the_third_person(said, stored):
    """Verbatim storage reads fine to a human and has no owner to a model: asked "what is
    my name and what do I do?" the assistant answered "My name is Krish. I work on Legend
    Mode." Quoting and attributing it in the prompt helped and did not settle it. Rendering
    the fact from the assistant's point of view before storing removes the pronouns
    entirely, so the embedding, the prompt and the panel all read the same sentence.

    Note the templates route through "The user said they …" — after "they" the base verb
    form is correct for every verb, so no conjugation is needed."""
    fact = memory.extract(said)
    assert fact is not None and fact.text == stored


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
    await store_api.remember(memory.Fact("The user's name is Krish", key="my name"))
    saved = await store_api.remember(memory.Fact("The user's name is Kris", key="my name"))

    entries = store_api.entries()
    assert len(entries) == 1
    assert entries[0]["text"] == "The user's name is Kris"
    assert saved is not None and saved["replaced"] == "The user's name is Krish"


async def test_keyless_facts_accumulate(mem):
    """Preferences are not mutually exclusive, so they sit beside each other."""
    store_api, _ = mem
    await store_api.remember(memory.Fact("The user said they prefer concise answers"))
    await store_api.remember(memory.Fact("The user said they use uv, not pip"))
    assert len(store_api.entries()) == 2


async def test_storing_a_known_fact_twice_is_a_no_op(mem):
    store_api, _ = mem
    await store_api.remember(memory.Fact("The user's name is Krish", key="my name"))
    assert await store_api.remember(memory.Fact("The user's name is Krish", key="my name")) is None
    assert len(store_api.entries()) == 1


async def test_a_memory_can_be_forgotten(mem):
    store_api, _ = mem
    saved = await store_api.remember(memory.Fact("The user's name is Krish", key="my name"))
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
    await store_api.remember(memory.Fact("The user's name is Krish", key="my name"))

    assert sorted(store.sources) == ["memory", "notes.md"]
    assert len(store) == 2


async def test_capture_costs_exactly_one_embed_and_no_model_call(mem):
    store_api, _ = mem
    await store_api.remember(memory.Fact("The user's name is Krish", key="my name"))
    assert store_api._client.calls == 1


# --- answering directly ------------------------------------------------------


_STORED = [
    {"id": 1, "key": "my name", "text": "The user's name is Krish"},
    {"id": 2, "key": "where i work", "text": "The user said they work on Legend Mode"},
]


@pytest.mark.parametrize(
    "question",
    ["what is my name?", "What's my name", "who am I?", "do you know my name",
     "do you remember what my name is?", "say my name", "tell me my name",
     "hey what is my name?", "okay so who am i"],
)
def test_the_name_is_answered_from_the_store(question):
    """Asked to phrase this, the 1.2B answered "My name is Krish." on roughly half of
    samples — reporting the user's identity as its own. Four prompt-side fixes helped and
    none settled it, so the answer is computed instead. Same rule as app/guardrails.py."""
    assert memory.direct_answer(question, _STORED) == "Your name is Krish."


@pytest.mark.parametrize(
    "question",
    [
        "where do I work?",              # answerable, but needs a verb this cannot form
        "what is my name and what I do?",  # compound: the second half needs the model
        "make my name bold",             # an instruction, not a question
        "what is the capital of France",
        "my name is Krish",              # a statement being captured, not a question
        "what is your name?",            # about the assistant, not the user
    ],
)
def test_everything_else_still_goes_to_the_model(question):
    """A wrong deterministic answer is worse than a wrong generated one — it arrives with
    the authority of a computed fact — so this declines whenever it is not certain."""
    assert memory.direct_answer(question, _STORED) is None


def test_no_stored_name_means_no_direct_answer():
    assert memory.direct_answer("what is my name?", []) is None
    assert memory.direct_answer("what is my name?", [_STORED[1]]) is None
