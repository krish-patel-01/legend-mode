"""Facts the user has told the assistant, kept across sessions.

ROADMAP step 4 called this retrieval's natural home: *"Recalling a name from many turns
back is a context-capacity limit at this size, and retrieval solves it where prompting
cannot."* The console sends full history every turn, so `recalls-user-name` passes inside
a conversation — but a new tab has never met you, and a thread longer than `num_ctx`
(4096 on the 350M) silently drops its own beginning.

**It is the same store, not a second system.** Memories are rows in `data/corpus.db` with
`source = "memory"`, so the pinned embedder, both retrieval gates, the score threshold and
the trace panel all apply unchanged. The only new parts are capture and consolidation.

Two rules, both hard:

**Only the user's own words are ever stored.** Never a model's output. In one session a
model here invented "developed by NLP research team at Stanford University" and read
"LFM2.5-350M" out of a document as its own identity. If generated text could become
memory, one confabulation becomes permanent truth.

**No model call anywhere in this file.** Capture is regex, consolidation is a key
comparison. mem0's design runs an LLM extraction *and* an LLM ADD/UPDATE/DELETE decision
per stored turn, which is sound on GPT-4o-mini and unaffordable here — the only local
model that could judge reliably costs ~25 s, and the 350M scores at chance. MemPalace
reports 96.6% R@5 on LongMemEval with no LLM at any stage, which is the evidence that the
cheap version is worth building. The consolidation *idea* is borrowed from mem0; the
implementation is deterministic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

SOURCE = "memory"

# Long messages are conversation, not facts. A fact the user wants kept is short.
_MAX_CAPTURE_CHARS = 240


@dataclass(frozen=True)
class Fact:
    text: str
    key: str | None
    """What this fact is *about*, when that can be named.

    Drives consolidation: a new "my name" replaces the old "my name" instead of sitting
    beside it, which is how a memory store avoids confidently recalling that the user is
    called two different things. `None` means the fact stands alone and accumulates —
    preferences and freeform notes are not mutually exclusive.
    """


# Ordered: the explicit form wins, since "remember that my name is X" is unambiguous.
#
# Everything here is deliberately narrow. A generous capture rule fills the store with
# conversational noise, and noise is precisely what makes retrieval harm answers — the
# 5.0-point GPQA drop this project's gating exists to avoid.
_PATTERNS: list[tuple[re.Pattern[str], str | None]] = [
    (re.compile(r"^\s*(?:please\s+)?(?:remember|note|keep in mind)"
                r"(?:\s+that)?\s*[:,-]?\s*(?P<fact>\S.{2,200}?)\s*$", re.IGNORECASE), None),
    (re.compile(r"^\s*don'?t forget\s*[:,-]?\s*(?P<fact>\S.{2,200}?)\s*$",
                re.IGNORECASE), None),
    (re.compile(r"\b(?P<fact>my name(?:'s| is)\s+(?P<v>[\w'’-]{1,40}))", re.IGNORECASE),
     "my name"),
    (re.compile(r"\b(?P<fact>i(?:'m| am) (?:called|named)\s+(?P<v>[\w'’-]{1,40}))",
                re.IGNORECASE), "my name"),
    (re.compile(r"\b(?P<fact>i work (?:on|at|for)\s+(?P<v>[\w'’ .,&-]{2,60}))",
                re.IGNORECASE), "where i work"),
    (re.compile(r"\b(?P<fact>i(?:'m| am) (?:building|working on)\s+"
                r"(?P<v>[\w'’ .,&-]{2,60}))", re.IGNORECASE), "what i am working on"),
    (re.compile(r"\b(?P<fact>i live in\s+(?P<v>[\w'’ .,-]{2,50}))", re.IGNORECASE),
     "where i live"),
    (re.compile(r"\b(?P<fact>i (?:prefer|always use|never use|can'?t stand|hate|love)\s+"
                r"(?P<v>[\w'’ .,&/-]{2,60}))", re.IGNORECASE), None),
]

# A question is never a fact, however much it looks like one. "do you remember my name?"
# and "what is my name" both match capture patterns and must not be stored.
_QUESTION = re.compile(r"\?\s*$|^\s*(?:what|who|when|where|which|why|how|do|does|did|is|"
                       r"are|can|could|would|will)\b", re.IGNORECASE)


def extract(text: str) -> Fact | None:
    """A durable fact the user stated, or None. Regex only — see the module docstring."""
    stripped = " ".join(text.split())
    if not stripped or len(stripped) > _MAX_CAPTURE_CHARS:
        return None

    for pattern, key in _PATTERNS:
        match = pattern.search(stripped)
        if not match:
            continue
        fact = (match.groupdict().get("fact") or "").strip(" .,;:")
        if not fact:
            continue
        # The explicit forms ("remember that ...") are exempt: they open with an
        # imperative, and their payload may legitimately be phrased as anything.
        if key is not None and _QUESTION.search(stripped):
            return None
        return Fact(text=fact, key=key)
    return None


class MemoryStore:
    """Capture and recall over the shared VectorStore. Recall needs no code here — the
    rows are in the same table, so `Retrieval.lookup` already finds them."""

    def __init__(self, client, embed_spec, store) -> None:
        self._client = client
        self._embed_spec = embed_spec
        self._store = store

    @property
    def available(self) -> bool:
        return self._store is not None

    def entries(self) -> list[dict[str, object]]:
        """Not named `list` — that shadows the builtin inside the class's own annotations."""
        # `is None`, not truthiness: VectorStore defines __len__, so an empty store is
        # falsy and every write silently no-opped until the first row existed.
        if self._store is None:
            return []
        return [{"id": i, "key": h or None, "text": t}
                for i, h, t in self._store.rows_for(SOURCE)]

    def forget(self, memory_id: int) -> bool:
        return self._store is not None and self._store.delete(memory_id)

    async def remember(self, fact: Fact) -> dict[str, object] | None:
        """Store a fact, replacing any earlier one about the same thing.

        The replace is mem0's ADD/UPDATE decision without the model call: same key means
        the user has restated something, and two answers to "what is my name" is worse
        than none. Facts with no key accumulate, because preferences are not exclusive.
        """
        if self._store is None:
            return None

        vectors = await self._client.embed(self._embed_spec, [fact.text])
        if not vectors:
            return None

        replaced = None
        if fact.key:
            for row_id, heading, text in self._store.rows_for(SOURCE):
                if heading == fact.key:
                    if " ".join(text.split()).lower() == fact.text.lower():
                        return None  # already known; storing it again buys nothing
                    self._store.delete(row_id)
                    replaced = text
                    break

        memory_id = self._store.add(SOURCE, fact.key or "", fact.text, vectors[0])
        log.info("remembered %r%s", fact.text,
                 f" (replacing {replaced!r})" if replaced else "")
        return {"id": memory_id, "text": fact.text, "key": fact.key, "replaced": replaced}
