"""Notes, written into an Obsidian vault as ordinary markdown files.

A vault *is* a folder of markdown files — Obsidian adds no database and no format of its
own — so this writes plain files and Obsidian picks them up live, with backlinks, graph
and search all working. That is why this is not an MCP integration: a protocol and a
second process to do `write_text` would buy nothing.

**Every path is confined to the vault.** `_resolve` is the whole safety story here: a
title arrives from a model, becomes a filename, and must not be able to escape. Titles
like `../../.ssh/authorized_keys` are the obvious case, but on Windows the quiet ones
matter more — a drive-relative `C:notes`, a reserved device name like `CON` or `NUL`, a
trailing dot or space that the filesystem silently strips. The resolved path is compared
against the resolved root, which is the only check that survives all of them.

Notes are appended to rather than overwritten. An assistant that can silently replace a
file the user has been adding to for a month is a worse tool than one that cannot write
at all, and appending under a timestamped heading is what "remember this too" means
anyway.

This is deliberately separate from `app/memory.py`. That captures facts in passing, by
regex, from ordinary conversation. This writes when the user *asks* for something to be
written, and produces a document they can open, edit and link. The vault can also be fed
to `scripts/ingest.py`, which puts these notes into the same retrieval corpus as anything
else — so what Lucy writes becomes what Lucy can later recall.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from app.tools.registry import Tool

log = logging.getLogger(__name__)

SUBFOLDER = "Notes"
"""Everything written here lands in one folder, so the vault root stays the user's."""

MAX_NOTE_CHARS = 20_000
MAX_SEARCH_HITS = 8
_SNIPPET = 240

# Windows reserves these regardless of extension: CON.md is still CON.
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


def slugify(title: str) -> str | None:
    """A title turned into a safe bare filename, or None if nothing usable is left.

    Returns a *name*, never a path — the caller joins it to the notes folder. Anything
    that looks like a directory separator is removed rather than escaped, because a title
    has no legitimate reason to contain one.
    """
    text = unicodedata.normalize("NFC", title).strip()
    text = _ILLEGAL.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    # A trailing dot or space is silently dropped by Windows, so "notes." and "notes"
    # would be the same file while comparing as different strings.
    text = text.strip(". ")
    if not text or set(text) <= {"."}:
        return None
    if text.lower() in _RESERVED:
        text = f"{text} note"
    return text[:120]


def _resolve(root: Path, title: str) -> Path | None:
    """The file a title refers to, or None if it would land outside the vault."""
    name = slugify(title)
    if name is None:
        return None
    folder = (root / SUBFOLDER).resolve()
    candidate = (folder / f"{name}.md").resolve()
    # The comparison is done on resolved paths on purpose: symlinks, `..`, short 8.3
    # names and case differences are all normalised by then, and nothing else reliably
    # catches all four.
    if candidate != folder / f"{name}.md" or not candidate.is_relative_to(folder):
        return None
    return candidate


class NotesConfig:
    def __init__(self, vault: Path | str | None) -> None:
        self.vault = Path(vault).expanduser() if vault else None


def _unavailable(config: NotesConfig) -> str | None:
    if config.vault is None:
        return "No notes vault is configured. Set LEGEND_VAULT_PATH to an Obsidian vault."
    if not config.vault.is_dir():
        return f"The notes vault at {config.vault} does not exist."
    return None


def write_note(title: str, content: str, config: NotesConfig) -> str:
    """Create a note, or append to it if it already exists."""
    problem = _unavailable(config)
    if problem:
        return problem
    assert config.vault is not None

    if not content.strip():
        return "Nothing to write — the note content was empty."
    path = _resolve(config.vault, title)
    if path is None:
        return f"{title!r} is not a usable note title."

    content = content.strip()[:MAX_NOTE_CHARS]
    stamp = datetime.now().astimezone()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        # Appending, not replacing. See the module docstring.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n## {stamp:%Y-%m-%d %H:%M}\n\n{content}\n")
        return f"Added to the existing note {path.stem!r} in the vault."

    front = (
        "---\n"
        f"created: {stamp:%Y-%m-%d %H:%M}\n"
        "source: lucy\n"
        "---\n\n"
    )
    path.write_text(f"{front}# {path.stem}\n\n{content}\n", encoding="utf-8")
    return f"Wrote a new note {path.stem!r} to the vault."


def read_note(title: str, config: NotesConfig) -> str:
    problem = _unavailable(config)
    if problem:
        return problem
    assert config.vault is not None

    path = _resolve(config.vault, title)
    if path is None:
        return f"{title!r} is not a usable note title."
    if not path.exists():
        # The dispatcher passes the words the user used — "coffee" — and the note is
        # called "Coffee Black". Returning "there is no note called coffee" was
        # technically true and useless: the writer apologised for a note that exists.
        # One unambiguous near match is what the user meant, so read it and say so.
        near = _search_titles(config.vault, title)
        if len(near) == 1:
            return _rendered(near[0], matched=title)
        if near:
            names = ", ".join(repr(p.stem) for p in near[:5])
            return (
                f"There is no note called exactly {path.stem!r}. These match: {names}. "
                f"Ask for one of those by name."
            )
        return f"There is no note called {path.stem!r} in the vault."
    return _rendered(path)


def _strip_frontmatter(body: str) -> str:
    return re.sub(r"^---.*?---", "", body, count=1, flags=re.DOTALL).strip()


def _rendered(path: Path, *, matched: str | None = None) -> str:
    """A note as the model should see it: no frontmatter, no duplicated heading.

    The frontmatter is bookkeeping and the `# Title` line repeats the name that is
    already in the sentence around it. Both were being handed over verbatim, and the
    writer described the result as "a bit fragmented" — which it was.
    """
    body = _strip_frontmatter(path.read_text(encoding="utf-8")[:MAX_NOTE_CHARS])
    body = re.sub(rf"^#\s*{re.escape(path.stem)}\s*\n+", "", body).strip()
    lead = (
        f"Your note {path.stem!r} (the closest match to {matched!r}) says:"
        if matched
        else f"Your note {path.stem!r} says:"
    )
    return f"{lead}\n\n{body}"


def _markdown_files(vault: Path) -> list[Path]:
    # `.obsidian` holds the vault's own config as JSON; `.trash` holds deleted notes.
    return [
        p
        for p in vault.rglob("*.md")
        if not any(part.startswith(".") for part in p.relative_to(vault).parts)
    ]


def _search_titles(vault: Path, query: str) -> list[Path]:
    needle = (slugify(query) or query).lower()
    return [p for p in _markdown_files(vault) if needle in p.stem.lower()]


def search_notes(query: str, config: NotesConfig) -> str:
    """Find notes by title or content. Reads the vault directly, so it is never stale."""
    problem = _unavailable(config)
    if problem:
        return problem
    assert config.vault is not None

    query = query.strip()
    if not query:
        return "No search query was given."

    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
    scored: list[tuple[int, Path, str]] = []
    for path in _markdown_files(config.vault):
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        haystack = f"{path.stem}\n{body}".lower()
        # Title matches count double: a note called "Coffee" is a better answer to
        # "coffee" than one that mentions coffee once in passing.
        score = sum(haystack.count(t) for t in terms) + sum(
            2 * path.stem.lower().count(t) for t in terms
        )
        if score:
            scored.append((score, path, body))

    if not scored:
        return f"No notes matching {query!r}."

    scored.sort(key=lambda row: (-row[0], row[1].stem))
    lines = []
    for _, path, body in scored[:MAX_SEARCH_HITS]:
        text = _strip_frontmatter(body)
        # Drop the note's own `# Title` line: it repeats the name printed right beside it,
        # and leaving it in produced snippets like "Coffee Black: # Coffee Black I take my
        # coffee black" that the writer read as fragmented text rather than as an answer.
        text = re.sub(rf"^#\s*{re.escape(path.stem)}\s*\n+", "", text).strip()
        lines.append(f'- "{path.stem}" — {" ".join(text.split())[:_SNIPPET]}')

    count = "one note" if len(scored) == 1 else f"{len(scored)} notes"
    return (
        f"Found {count} in the user's own notes matching {query!r}. "
        f"This is what they wrote:\n\n" + "\n".join(lines)
    )


def tools(config: NotesConfig) -> list[Tool]:
    return [
        Tool(
            name="write_note",
            description=(
                "Use when the user asks you to write something down, save a note, or keep "
                "a record of something: 'make a note that...', 'write this down', "
                "'save this'. Adds to the note if one with that title already exists."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "A short title, a few words. No slashes.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The note text itself, in markdown.",
                    },
                },
                "required": ["title", "content"],
            },
            run=lambda title, content: write_note(title, content, config),
            family="notes",
            writes=True,
        ),
        Tool(
            name="read_note",
            description="Use to read back a note the user asks about by name.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The note's title."}
                },
                "required": ["title"],
            },
            run=lambda title: read_note(title, config),
            family="notes",
        ),
        Tool(
            name="search_notes",
            description=(
                "Use when the user asks what they wrote about something, or what they "
                "told you earlier, and you do not know the exact note title. Searches "
                "titles and contents."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Words to look for."}
                },
                "required": ["query"],
            },
            run=lambda query: search_notes(query, config),
            family="notes",
        ),
    ]
