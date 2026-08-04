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
    """Third person, for storing and for showing the model. See the note on _PATTERNS."""

    said: str = ""
    """What the user actually typed, used only as the embedding input.

    Rewriting to third person fixed the pronouns and broke recall: stored as "The user
    said they work on Legend Mode", the question "where do I work" no longer matched,
    because the shared phrasing it was matching on had been rewritten away.

    So the two jobs are separated. The embedding is built from the user's own wording,
    which is the wording their later questions will resemble; the text handed to the model
    is the unambiguous third-person one. Nothing needs to be embedded twice — a memory has
    exactly one vector, it is simply computed from the sentence that retrieves best.
    """

    key: str | None = None
    """What this fact is *about*, when that can be named.

    Drives consolidation: a new "my name" replaces the old "my name" instead of sitting
    beside it, which is how a memory store avoids confidently recalling that the user is
    called two different things. `None` means the fact stands alone and accumulates —
    preferences and freeform notes are not mutually exclusive.
    """


# Ordered: the explicit form wins, since "remember that my name is X" is unambiguous.
# Everything here is deliberately narrow — a generous capture rule fills the store with
# conversational noise, and noise is what makes retrieval harm answers.
#
# **Facts are stored in the third person, and that is the fix for a real bug.**
#
# Storing the user's words verbatim — "my name is Krish" — reads correctly to a human and
# has no owner to a model. Asked "what is my name and what do I do?" the assistant replied
# **"My name is Krish. I work on Legend Mode."** Quoting and attributing it in the prompt
# ("The user told you: ...") helped and did not settle it; the flip still came back on some
# samples. Instruction cannot reliably beat the pronouns sitting in the text.
#
# So each pattern carries a template that renders the fact from the assistant's point of
# view before it is ever stored. Everything downstream — the embedding, the injected
# prompt, the memory panel — then reads the same unambiguous sentence.
#
# The templates route through "The user said they …" rather than conjugating. "The user
# works on X" needs a verb agreement engine the moment the verb is "prefer" or "always
# use"; after "they", the base form is already correct for every verb, so a one-line
# template covers all of them.
_PATTERNS: list[tuple[re.Pattern[str], str | None, str]] = [
    (re.compile(r"^\s*(?:please\s+)?(?:remember|note|keep in mind)"
                r"(?:\s+that)?\s*[:,-]?\s*(?P<fact>\S.{2,200}?)\s*$", re.IGNORECASE),
     None, "The user asked you to remember: {fact}"),
    (re.compile(r"^\s*don'?t forget\s*[:,-]?\s*(?P<fact>\S.{2,200}?)\s*$",
                re.IGNORECASE), None, "The user asked you not to forget: {fact}"),
    (re.compile(r"\bmy name(?:'s| is)\s+(?P<v>[\w'’-]{1,40})", re.IGNORECASE),
     "my name", "The user's name is {v}"),
    (re.compile(r"\bi(?:'m| am) (?:called|named)\s+(?P<v>[\w'’-]{1,40})",
                re.IGNORECASE), "my name", "The user's name is {v}"),
    (re.compile(r"\bi work (?P<prep>on|at|for)\s+(?P<v>[\w'’ .,&-]{2,60})",
                re.IGNORECASE), "where i work",
     "The user said they work {prep} {v}"),
    (re.compile(r"\bi(?:'m| am) (?P<verb>building|working on)\s+(?P<v>[\w'’ .,&-]{2,60})",
                re.IGNORECASE), "what i am working on",
     "The user said they are {verb} {v}"),
    (re.compile(r"\bi live in\s+(?P<v>[\w'’ .,-]{2,50})", re.IGNORECASE),
     "where i live", "The user said they live in {v}"),
    (re.compile(r"\bi (?P<verb>prefer|always use|never use|can'?t stand|hate|love)\s+"
                r"(?P<v>[\w'’ .,&/-]{2,60})", re.IGNORECASE), None,
     "The user said they {verb} {v}"),
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

    for pattern, key, template in _PATTERNS:
        match = pattern.search(stripped)
        if not match:
            continue
        groups = {k: (v or "").strip(" .,;:") for k, v in match.groupdict().items()}
        if not any(groups.values()):
            continue
        # The explicit forms ("remember that ...") are exempt: they open with an
        # imperative, and their payload may legitimately be phrased as anything.
        if key is not None and _QUESTION.search(stripped):
            return None
        return Fact(text=template.format(**groups), said=stripped, key=key)
    return None


# --- answering directly ------------------------------------------------------
#
# Some questions about a stored fact are answerable exactly, and asking a 1.2B to phrase
# one is how "what is my name?" came back as **"My name is Krish."** on roughly half of
# samples — the assistant reporting the user's identity as its own. Four fixes were tried
# first: a prompt line about pronouns, quoting each fact, rewriting facts to third person
# at capture, and changing which tier reads them. The first three helped and none settled
# it, because the model is being asked to do a transformation it is not reliable at.
#
# So it is not asked. This is the same rule as app/guardrails.py — compute what can be
# computed — applied to memory rather than arithmetic.
#
# Deliberately narrow. Only the name, only when the question is unmistakably about it,
# and only when a stored fact matches the template exactly. Everything else still goes to
# the model, because a wrong deterministic answer is worse than a wrong generated one: it
# arrives with the authority of a computed fact.
_NAME_QUESTION = re.compile(
    # Any number of leading fillers — "okay so who am i" is two of them.
    r"^\s*(?:(?:hey|hi|hello|so|ok(?:ay)?|and|umm?|well)[\s,]+)*"
    r"(?:what(?:'s| is| was)?\s+my\s+name"
    r"|who\s+am\s+i"
    # "...what my name is" puts the verb after the noun, so it has to be optional here.
    r"|do\s+you\s+(?:know|remember)\s+(?:what\s+)?my\s+name(?:\s+is)?"
    r"|say\s+my\s+name"
    r"|tell\s+me\s+my\s+name)"
    r"\s*[?.!]*\s*$",
    re.IGNORECASE,
)
_STORED_NAME = re.compile(r"^The user's name is (?P<v>.+)$")


def direct_answer(text: str, entries: list[dict[str, object]]) -> str | None:
    """An exact answer from stored memory, or None to let the model handle it."""
    if not _NAME_QUESTION.match(text.strip()):
        return None
    for entry in entries:
        if entry.get("key") != "my name":
            continue
        match = _STORED_NAME.match(str(entry.get("text", "")))
        if match:
            return f"Your name is {match.group('v').strip(' .')}."
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

        # Embed what the user said; store what the model can read. See Fact.said.
        vectors = await self._client.embed(self._embed_spec, [fact.said or fact.text])
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
