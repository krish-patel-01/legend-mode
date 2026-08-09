"""Tests for chunking, the vector store, and the retrieval gate.

All of it runs without Ollama: chunking is pure text, and the store takes vectors from
whoever supplies them, so a handful of hand-written three-dimensional vectors exercises
the search path exactly as a 384-dimensional bge embedding would.
"""

from __future__ import annotations

import pytest

from app.retrieval import chunk_markdown
from app.retrieval.chunk import MIN_CHARS
from app.retrieval.service import (
    Retrieval,
    RetrievalResult,
    as_citation_line,
    as_system_note,
)
from app.retrieval.store import Hit, IndexMismatch, VectorStore


# --- chunking ----------------------------------------------------------------

_DOC = """# Legend Mode

A router that picks a model per request. This opening paragraph is long enough to
survive the minimum-length filter that drops stray fragments from the index.

## Guardrails

Deterministic answers are computed in Python and injected before generation, which
is why the model phrases a correct fact instead of producing prose to be patched.

### Correction

A contradicted numeric grounding is replaced, because comparing two numbers is exact
and comparing two pieces of prose is the unreliable step this design avoids.
"""


def test_headings_become_the_chunk_path():
    chunks = chunk_markdown(_DOC)
    headings = [c.heading for c in chunks]
    assert "Legend Mode" in headings
    assert "Legend Mode > Guardrails" in headings
    assert "Legend Mode > Guardrails > Correction" in headings


def test_the_heading_path_is_part_of_what_gets_embedded():
    """A paragraph on its own embeds near nothing useful; prefixed with its heading it
    embeds near the question someone would actually ask."""
    chunk = next(c for c in chunk_markdown(_DOC) if "Correction" in c.heading)
    assert chunk.embed_text.startswith("Legend Mode > Guardrails > Correction")
    assert chunk.text in chunk.embed_text


def test_fragments_are_dropped():
    assert chunk_markdown("# Title\n\nshort.\n") == []


def test_every_chunk_clears_the_minimum():
    assert all(len(c.text) >= MIN_CHARS for c in chunk_markdown(_DOC))


def test_headings_inside_a_code_fence_are_not_headings():
    doc = (
        "# Real\n\nSome body text that is comfortably longer than the minimum length "
        "filter so it is definitely kept.\n\n"
        "```\n# not a heading\nprint('hi')\n```\n"
    )
    assert {c.heading for c in chunk_markdown(doc)} == {"Real"}


def test_long_sections_are_split_without_losing_text():
    body = "\n\n".join(f"Paragraph number {i} with enough words to matter here." * 3
                       for i in range(12))
    chunks = chunk_markdown(f"# T\n\n{body}", max_chars=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    assert "Paragraph number 11" in " ".join(c.text for c in chunks)


def test_plain_text_with_no_headings_still_chunks():
    text = "A note with no markdown structure at all, but long enough to be indexed."
    assert [c.heading for c in chunk_markdown(text)] == [""]


# --- the store ---------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = VectorStore(tmp_path / "corpus.db")
    yield s
    s.close()


def _seed(store: VectorStore) -> None:
    store.set_embedder("legend/embed", 3)
    store.replace_source(
        "notes.md",
        [("Cats", "Cats are small carnivores."), ("Dogs", "Dogs are pack animals.")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )


def test_search_ranks_by_cosine(store):
    _seed(store)
    hits = store.search([0.9, 0.1, 0.0], k=2)
    assert [h.heading for h in hits] == ["Cats", "Dogs"]
    assert hits[0].score > hits[1].score


def test_an_empty_store_returns_nothing(store):
    assert store.search([1.0, 0.0, 0.0]) == []
    assert len(store) == 0


def test_re_ingesting_a_source_replaces_it(store):
    _seed(store)
    store.replace_source("notes.md", [("Cats", "Only one chunk now, replacing both.")],
                         [[1.0, 0.0, 0.0]])
    assert len(store) == 1


def test_sources_are_listed_and_droppable(store):
    _seed(store)
    assert store.sources == ["notes.md"]
    store.drop_source("notes.md")
    assert store.sources == []


def test_the_index_survives_a_reopen(store, tmp_path):
    _seed(store)
    store.close()
    reopened = VectorStore(tmp_path / "corpus.db")
    try:
        assert len(reopened) == 2
        assert reopened.search([1.0, 0.0, 0.0], k=1)[0].heading == "Cats"
    finally:
        reopened.close()


def test_a_different_embedder_is_refused_not_tolerated(store):
    """An index searched with the wrong model does not fail — it returns plausible,
    wrong neighbours. Refusing is the only way that stays visible."""
    _seed(store)
    with pytest.raises(IndexMismatch):
        store.check_embedder("legend/other-embedder", 3)
    with pytest.raises(IndexMismatch):
        store.check_embedder("legend/embed", 384)


def test_a_dimension_mismatch_returns_nothing_rather_than_crashing(store):
    _seed(store)
    assert store.search([1.0, 0.0], k=1) == []


def test_citations_name_the_source_and_heading():
    assert Hit("README.md", "Evals", "…", 0.9).citation == "README.md#Evals"
    assert Hit("README.md", "", "…", 0.9).citation == "README.md"


# --- the query-time gate -----------------------------------------------------


class _StubClient:
    def __init__(self, vector=(1.0, 0.0, 0.0)):
        self.vector = list(vector)
        self.calls = 0

    async def embed(self, spec, texts):
        self.calls += 1
        return [self.vector for _ in texts]


class _Spec:
    tag = "legend/embed"
    alias = "embed"


async def test_a_relevant_hit_is_returned(store):
    _seed(store)
    r = Retrieval(_StubClient(), _Spec(), store, top_k=2, min_score=0.5)
    found = await r.lookup("tell me about cats")
    assert found is not None
    assert found.citations == ["notes.md#Cats"]


async def test_a_memory_is_not_crowded_out_by_documents(store):
    """`top_k` selects before the per-source filter runs, so a memory can rank below
    documents that then fail the stricter document threshold — and never get checked
    against its own. Over-fetching first is what makes two thresholds work at all."""
    store.set_embedder("legend/embed", 3)
    store.replace_source(
        "notes.md",
        [(f"H{i}", f"document chunk number {i} with filler text") for i in range(3)],
        [[0.80, 0.60, 0.0], [0.81, 0.59, 0.0], [0.82, 0.58, 0.0]],
    )
    store.replace_source("memory", [("my job", "I work on Legend Mode")], [[0.70, 0.71, 0.0]])

    r = Retrieval(_StubClient((1.0, 0.0, 0.0)), _Spec(), store,
                  top_k=3, min_score=0.95, memory_min_score=0.55)
    found = await r.lookup("where do I work")
    assert found is not None
    assert found.citations == ["memory#my job"]


def test_trailing_punctuation_is_stripped_before_embedding():
    """A question mark cost 0.057 cosine — 0.599 for "where do I work" against 0.542 for
    the same question with a "?" — which was the difference between recalling the right
    memory and recalling nothing."""
    from app.retrieval.service import _normalize_query
    assert _normalize_query("where do I work?") == "where do I work"
    assert _normalize_query("  what is my name?  ") == "what is my name"
    assert _normalize_query("17 * 23") == "17 * 23"


async def test_a_weak_hit_is_discarded(store):
    """The threshold is the real gate. Injecting a merely-plausible passage is how
    retrieval makes answers worse — measured at -5.0 GPQA points in the paper."""
    _seed(store)
    # A query vector that sits between the two indexed chunks: closest neighbour at
    # ~0.70 cosine, which is exactly the "plausible but not about this" band.
    r = Retrieval(_StubClient((0.7, 0.7, 0.1)), _Spec(), store, top_k=2, min_score=0.99)
    assert await r.lookup("something unrelated") is None


async def test_an_empty_corpus_never_embeds(store):
    """No index means no reason to spend even 15 ms on the query."""
    client = _StubClient()
    r = Retrieval(client, _Spec(), store, top_k=2, min_score=0.5)
    assert await r.lookup("anything") is None
    assert client.calls == 0


async def test_a_mismatched_index_disables_retrieval_rather_than_grounding_in_noise(store):
    _seed(store)
    r = Retrieval(_StubClient((1.0, 0.0, 0.0)), _Spec(), store, top_k=2, min_score=0.5)
    store.set_embedder("legend/some-other-model", 3)
    assert await r.lookup("cats") is None
    assert not r.available


# --- the injected note -------------------------------------------------------


def test_the_note_names_the_source_without_asking_for_a_citation():
    """The first version labelled each passage `[source#heading]` and asked the model to
    copy that. The 1.2B replied with only the bracketed citation and no answer, having
    matched the most recent pattern in the prompt. Citations are computed now."""
    note = as_system_note(RetrievalResult([Hit("README.md", "Evals", "Run eval.py.", 0.8)]))
    assert "README.md" in note and "Evals" in note
    assert "[README.md#Evals]" not in note
    assert "Run eval.py." in note


def test_the_note_does_not_end_on_a_quotable_instruction():
    """Models this size echo a prompt's final sentence verbatim — twice observed, see
    app/persona.py. Ending on the retrieved material has no such failure mode."""
    note = as_system_note(RetrievalResult([Hit("a.md", "", "The corpus text.", 0.8)]))
    assert note.rstrip().endswith("The corpus text.")


def test_the_citation_line_is_computed_not_generated():
    line = as_citation_line(RetrievalResult([
        Hit("README.md", "Evals", "x", 0.9), Hit("ROADMAP.md", "", "y", 0.8),
    ]))
    assert line == "Sources: README.md#Evals, ROADMAP.md"


def test_duplicate_sources_are_cited_once():
    result = RetrievalResult([
        Hit("a.md", "One", "x", 0.9),
        Hit("a.md", "One", "y", 0.8),
        Hit("b.md", "", "z", 0.7),
    ])
    assert result.citations == ["a.md#One", "b.md"]
