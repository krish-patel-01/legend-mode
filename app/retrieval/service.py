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
    "Reference material from this machine's local notes, retrieved for this question. "
    "Answer the question using it. If it does not cover part of what was asked, answer "
    "that part from your own knowledge and say plainly which part came from where."
)


def as_system_note(result: RetrievalResult) -> str:
    blocks = [
        f"--- from {hit.source}{f' ({hit.heading})' if hit.heading else ''} ---\n{hit.text}"
        for hit in result.hits
    ]
    return _PREAMBLE + "\n\n" + "\n\n".join(blocks)


def as_citation_line(result: RetrievalResult) -> str:
    """The sources line appended to the reply. Deterministic, so it is always right."""
    return "Sources: " + ", ".join(result.citations)


class Retrieval:
    """Query-time wrapper over a VectorStore and the pinned embedder."""

    def __init__(self, client, embed_spec, store: VectorStore | None, *, top_k: int,
                 min_score: float) -> None:
        self._client = client
        self._embed_spec = embed_spec
        self._store = store
        self._top_k = top_k
        self._min_score = min_score
        self._checked = False

    @property
    def available(self) -> bool:
        return self._store is not None and len(self._store) > 0

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
            vectors = await self._client.embed(self._embed_spec, [text])
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

        hits = [h for h in self._store.search(vectors[0], self._top_k)
                if h.score >= self._min_score]
        if not hits:
            return None
        return RetrievalResult(hits=hits)
