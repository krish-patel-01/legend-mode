"""Runtime side of retrieval: embed the question, search, decide whether to use it.

The deciding is the point. Retrieving is easy and injecting whatever comes back is how
retrieval makes a system *worse* — the paper behind this plan measured a 5.0-point GPQA
drop from indiscriminate retrieval, because a passage that is merely present will override
knowledge the model already had right. So a hit has to clear a similarity threshold before
anything is injected, and when nothing clears it the request is answered exactly as it
would have been with no corpus at all.

The threshold is a calibration, not a constant. bge-small puts unrelated English text
around 0.6 cosine, which is high enough that a naive 0.5 cut-off would inject on every
question. `scripts/ingest.py --probe "..."` prints the actual scores for a question so the
number can be set from data rather than from this comment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.backends.ollama import OllamaError
from app.retrieval.store import Hit, IndexMismatch, VectorStore

log = logging.getLogger(__name__)

# Kept in step with app.memory.SOURCE. Duplicated rather than imported because
# app/memory.py builds on this package, and importing back would close the cycle.
MEMORY_SOURCE = "memory"


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[Hit]

    @property
    def citations(self) -> list[str]:
        seen: list[str] = []
        for hit in self.hits:
            if hit.citation not in seen:
                seen.append(hit.citation)
        return seen

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


# Two things about the shape of this, both learned by watching it fail.
#
# **The model is not asked to write citations.** The first version said "name the source
# in square brackets exactly as written below" and labelled each passage `[source#heading]`.
# The 1.2B replied with *only* `[ROADMAP.md#Roadmap > Step 3 — Effort controller and
# adjudication]` and no answer at all — it copied the most recent pattern in the prompt
# instead of using the text under it. Citations are known exactly at this point, so they
# are appended to the reply by app/api.py rather than requested from a model that has to
# spend budget formatting them and can get them wrong. Computing what can be computed is
# the same rule app/guardrails.py follows.
#
# **It ends on the material, not on an instruction.** A prompt closing on a quotable
# directive gets it echoed back verbatim by models this size — app/persona.py records two
# separate instances. Passages have no such failure mode.
_PREAMBLE = (
    "Reference material retrieved for this question. Answer the question using it. If it "
    "does not cover part of what was asked, answer that part from your own knowledge and "
    "say plainly which part came from where."
)

# Memories arrive already in the third person — app/memory.py renders them that way at
# capture. Two attempts to fix the ownership confusion here instead both failed: a prompt
# line saying "my and I mean the user", then quoting and attributing each fact inline. The
# flip back to "My name is Krish" survived both. Pronouns in the stored text beat
# instructions about the stored text, so the text is what changed.
#
# This block therefore only introduces them, and must not re-attribute — doing so produced
# `- The user told you: "The user's name is Krish"`.
_MEMORY_LEAD = "What you know about the user from earlier conversations:"


def as_system_note(result: RetrievalResult) -> str:
    memories = [h for h in result.hits if h.source == MEMORY_SOURCE]
    documents = [h for h in result.hits if h.source != MEMORY_SOURCE]

    parts = [_PREAMBLE]
    if memories:
        bullets = "\n".join(f"- {h.text}" for h in memories)
        parts.append(f"{_MEMORY_LEAD}\n{bullets}")
    for hit in documents:
        heading = f" ({hit.heading})" if hit.heading else ""
        parts.append(f"--- from {hit.source}{heading} ---\n{hit.text}")
    return "\n\n".join(parts)


def _normalize_query(text: str) -> str:
    """Trim trailing punctuation before embedding.

    Measured, not tidiness. Against a stored "I work on Legend Mode":

        "where do I work"    0.599
        "where do I work?"   0.542

    A question mark carries no meaning for retrieval and cost 0.057 cosine — enough to
    drop the right answer below the threshold on one phrasing and not the other. The
    alternative was lowering the cut-off to 0.50, which would have started matching
    "I prefer short answers" (0.504) to a question about employment. Fix the query, not
    the gate.
    """
    return text.strip().rstrip("?!.,;: \t\n")


def as_citation_line(result: RetrievalResult) -> str:
    """The sources line appended to the reply. Deterministic, so it is always right."""
    return "Sources: " + ", ".join(result.citations)


class Retrieval:
    """Query-time wrapper over a VectorStore and the pinned embedder."""

    def __init__(self, client, embed_spec, store: VectorStore | None, *, top_k: int,
                 min_score: float, memory_min_score: float | None = None) -> None:
        self._client = client
        self._embed_spec = embed_spec
        self._store = store
        self._top_k = top_k
        self._min_score = min_score
        # Memories are one short sentence against an 800-character document chunk, and
        # short-to-short cosine runs lower for the same relevance — see the note on
        # Settings.retrieval_memory_min_score.
        self._memory_min_score = (
            min_score if memory_min_score is None else memory_min_score
        )
        self._checked = False

    def _threshold_for(self, hit: Hit) -> float:
        return self._memory_min_score if hit.source == MEMORY_SOURCE else self._min_score

    @property
    def available(self) -> bool:
        return self._store is not None and len(self._store) > 0

    @property
    def store(self) -> VectorStore | None:
        """The backing store. Exposed so app/memory.py can write to the same table
        rather than standing up a second one."""
        return self._store

    @property
    def size(self) -> int:
        return len(self._store) if self._store else 0

    @property
    def sources(self) -> list[str]:
        return self._store.sources if self._store else []

    async def lookup(self, text: str) -> RetrievalResult | None:
        """Chunks worth showing the model, or None when nothing is relevant enough."""
        if not self.available or not text.strip():
            return None
        assert self._store is not None

        try:
            vectors = await self._client.embed(self._embed_spec, [_normalize_query(text)])
        except OllamaError as exc:
            log.warning("retrieval embed failed: %s", exc)
            return None
        if not vectors:
            return None

        if not self._checked:
            # Deferred to the first query rather than done at startup: the dimension is
            # only known once the embedder has actually run, and a corpus-less server
            # should not fail to boot over an index it will never read.
            try:
                self._store.check_embedder(self._embed_spec.tag, len(vectors[0]))
            except IndexMismatch as exc:
                log.error("%s", exc)
                self._store = None
                return None
            self._checked = True

        # Over-fetch, then filter, then trim. Selecting `top_k` first and filtering after
        # loses hits whenever the two sources have different thresholds: "where do I work"
        # scored 0.599 against its own stored memory and was ranked below three document
        # chunks at 0.61-0.63, all of which then failed the stricter document cut-off. The
        # memory never reached its own threshold check, so the feature silently did
        # nothing on exactly the query it was built for.
        candidates = self._store.search(vectors[0], self._top_k * 4)
        hits = [h for h in candidates if h.score >= self._threshold_for(h)][: self._top_k]
        if not hits:
            return None
        return RetrievalResult(hits=hits)
