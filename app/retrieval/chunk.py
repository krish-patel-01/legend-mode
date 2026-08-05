"""Splitting documents into retrievable pieces.

Two rules, both about what the embedder can actually work with:

**Split on headings first, paragraphs second.** bge-small has a 512-token window, so a
chunk has to be small; splitting on a fixed character count alone cuts mid-argument and
produces pieces that answer nothing. Markdown headings are a free, author-supplied
statement of where the topic changes.

**Carry the heading path into the chunk text.** A paragraph reading "It is always the
1.2B and never the 350M" embeds near nothing useful on its own. Prefixed with
"Step 3 - Effort controller and adjudication" it embeds near the question someone would
actually ask. The prefix is part of the embedded text and part of what the model reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*```")

# Big enough to hold a whole argument, small enough to stay inside bge-small's window
# once the heading prefix is added. Roughly 200 tokens of English.
MAX_CHARS = 800

# A chunk shorter than this is a fragment — a stray heading, a one-word line — and
# retrieving it can only displace something useful.
MIN_CHARS = 60

# **Notes are exempt, because for a note brevity is the content.** Ingesting the vault
# with the document threshold dropped every note written so far: "I take my coffee black"
# is 22 characters and "the standup moved to Thursday at 10" is 35, so all four were
# reported as "no chunks (too short?)" and the corpus gained nothing. The reasoning behind
# MIN_CHARS still holds for prose — a 20-character fragment of an essay is noise — but a
# deliberately written one-line note is the whole document.
NOTE_MIN_CHARS = 12

# `---` delimited YAML at the very top. Obsidian, Jekyll and this project's own
# app/tools/notes.py all write it, and it is metadata rather than text: left in, a note's
# embedding carries "created: 2026-08-05" and "source: lucy", which no question resembles.
_FRONTMATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


@dataclass(frozen=True)
class Chunk:
    heading: str
    text: str

    @property
    def embed_text(self) -> str:
        return f"{self.heading}\n\n{self.text}" if self.heading else self.text


def chunk_markdown(
    content: str, *, max_chars: int = MAX_CHARS, min_chars: int = MIN_CHARS
) -> list[Chunk]:
    """Split markdown (or plain text, which simply has no headings) into chunks.

    `min_chars` is lowered to NOTE_MIN_CHARS for vault notes — see the note on that
    constant for why a threshold tuned on documentation throws away every note.
    """
    content = _FRONTMATTER.sub("", content, count=1)
    chunks: list[Chunk] = []
    for heading, body in _sections(content):
        for piece in _pack(body, max_chars):
            if len(piece) >= min_chars:
                chunks.append(Chunk(heading=heading, text=piece))
    return chunks


def _sections(content: str) -> list[tuple[str, str]]:
    """(heading path, body) pairs. Headings inside fenced code are not headings."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((" > ".join(path), body))
        buffer.clear()

    for line in content.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path = path[: level - 1] + [match.group(2).strip()]
        else:
            buffer.append(line)
    flush()
    return sections


def _pack(body: str, max_chars: int) -> list[str]:
    """Group paragraphs into pieces no larger than `max_chars`.

    A paragraph longer than the limit on its own is split on sentence boundaries rather
    than mid-word, and only falls back to a hard cut if a single sentence is oversized.
    """
    pieces: list[str] = []
    current: list[str] = []
    size = 0

    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if current:
                pieces.append("\n\n".join(current))
                current, size = [], 0
            pieces.extend(_split_long(para, max_chars))
            continue
        if size + len(para) > max_chars and current:
            pieces.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2

    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _split_long(para: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", para)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:  # one oversized sentence, or a code block
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces
