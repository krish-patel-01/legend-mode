"""Notes into an Obsidian vault.

Most of this is about `_resolve`. A note title comes from a model, becomes a filename,
and must not be able to name a file outside the vault — so the escape attempts are
tested far more thoroughly than the happy path, including the Windows-specific ones that
fail quietly rather than raising.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.tools import notes


@pytest.fixture
def vault(tmp_path: Path) -> notes.NotesConfig:
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "app.json").write_text("{}", encoding="utf-8")
    return notes.NotesConfig(tmp_path)


# --- confinement --------------------------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "../escape",
        "../../etc/passwd",
        "..\\..\\windows\\system32",
        "/etc/passwd",
        "C:/Windows/system32/config",
        "C:notes",  # drive-relative: resolves against the CWD of that drive
        "sub/dir/note",
        "sub\\dir\\note",
        "....//....//x",
    ],
)
def test_titles_cannot_escape_the_vault(vault: notes.NotesConfig, title: str) -> None:
    assert vault.vault is not None
    resolved = notes._resolve(vault.vault, title)
    if resolved is not None:
        assert resolved.is_relative_to((vault.vault / notes.SUBFOLDER).resolve())


@pytest.mark.parametrize("title", ["", "   ", "...", "/", "\\", "?", "<>"])
def test_unusable_titles_are_refused(vault: notes.NotesConfig, title: str) -> None:
    assert vault.vault is not None
    assert notes._resolve(vault.vault, title) is None


def test_writing_with_a_traversing_title_stays_inside(vault: notes.NotesConfig) -> None:
    assert vault.vault is not None
    notes.write_note("../../owned", "payload", vault)
    escaped = vault.vault.parent / "owned.md"
    assert not escaped.exists()
    assert not (vault.vault.parent / "Notes").exists()


@pytest.mark.parametrize("name", ["CON", "nul", "com1", "LPT9"])
def test_windows_reserved_names_are_renamed(name: str) -> None:
    """CON.md is still CON to Windows, whatever the extension says."""
    slug = notes.slugify(name)
    assert slug is not None and slug.lower() not in notes._RESERVED


def test_trailing_dots_and_spaces_are_stripped() -> None:
    """Windows drops them silently, so "notes." and "notes" would be one file."""
    assert notes.slugify("notes.  ") == "notes"
    assert notes.slugify(" notes ") == "notes"


# --- writing ------------------------------------------------------------------


def test_a_new_note_gets_frontmatter_and_a_heading(vault: notes.NotesConfig) -> None:
    assert vault.vault is not None
    out = notes.write_note("Coffee", "I take it black.", vault)
    assert "Wrote a new note" in out
    body = (vault.vault / notes.SUBFOLDER / "Coffee.md").read_text(encoding="utf-8")
    assert body.startswith("---\n")
    assert "source: lucy" in body
    assert "# Coffee" in body
    assert "I take it black." in body


def test_writing_twice_appends_and_never_overwrites(vault: notes.NotesConfig) -> None:
    assert vault.vault is not None
    notes.write_note("Ideas", "first thought", vault)
    out = notes.write_note("Ideas", "second thought", vault)
    assert "Added to the existing note" in out
    body = (vault.vault / notes.SUBFOLDER / "Ideas.md").read_text(encoding="utf-8")
    assert "first thought" in body and "second thought" in body
    assert body.count("---\n") == 2  # one frontmatter block, not two


def test_empty_content_is_refused(vault: notes.NotesConfig) -> None:
    assert notes.write_note("Title", "   ", vault) == (
        "Nothing to write — the note content was empty."
    )


def test_wikilinks_survive_verbatim(vault: notes.NotesConfig) -> None:
    """Obsidian's whole graph depends on these not being mangled."""
    assert vault.vault is not None
    notes.write_note("Linked", "see [[Legend Mode]] and [[Coffee]]", vault)
    body = (vault.vault / notes.SUBFOLDER / "Linked.md").read_text(encoding="utf-8")
    assert "[[Legend Mode]]" in body and "[[Coffee]]" in body


def test_oversized_content_is_capped(vault: notes.NotesConfig) -> None:
    assert vault.vault is not None
    notes.write_note("Big", "x" * (notes.MAX_NOTE_CHARS * 2), vault)
    body = (vault.vault / notes.SUBFOLDER / "Big.md").read_text(encoding="utf-8")
    assert body.count("x") == notes.MAX_NOTE_CHARS


# --- reading and searching ----------------------------------------------------


def test_one_near_match_is_read_rather_than_reported_as_missing(
    vault: notes.NotesConfig,
) -> None:
    """The dispatcher passes "coffee"; the note is "Coffee preferences"."""
    notes.write_note("Coffee preferences", "black, no sugar", vault)
    out = notes.read_note("Coffee", vault)
    assert "black, no sugar" in out
    assert "Coffee preferences" in out


def test_several_near_matches_ask_which_one(vault: notes.NotesConfig) -> None:
    notes.write_note("Coffee beans", "ethiopian", vault)
    notes.write_note("Coffee gear", "aeropress", vault)
    out = notes.read_note("Coffee", vault)
    assert "Coffee beans" in out and "Coffee gear" in out


def test_a_read_note_omits_frontmatter_and_the_duplicate_heading(
    vault: notes.NotesConfig,
) -> None:
    """Both were handed to the writer verbatim; it called the result "fragmented"."""
    notes.write_note("Coffee", "black, no sugar", vault)
    out = notes.read_note("Coffee", vault)
    assert "source: lucy" not in out
    assert "# Coffee" not in out
    assert "black, no sugar" in out


def test_reading_a_missing_note_with_no_near_match(vault: notes.NotesConfig) -> None:
    assert "no note called" in notes.read_note("Nothing At All", vault)


def test_search_ranks_title_matches_above_passing_mentions(
    vault: notes.NotesConfig,
) -> None:
    notes.write_note("Coffee", "how I take it", vault)
    notes.write_note("Shopping", "buy coffee filters at some point", vault)
    out = notes.search_notes("coffee", vault)
    assert out.index('"Coffee"') < out.index('"Shopping"')


def test_search_skips_the_obsidian_config_folder(vault: notes.NotesConfig) -> None:
    assert vault.vault is not None
    (vault.vault / ".obsidian" / "hidden.md").write_text("coffee", encoding="utf-8")
    notes.write_note("Real", "coffee", vault)
    out = notes.search_notes("coffee", vault)
    assert "hidden" not in out and "Real" in out


def test_search_with_no_hits_says_so(vault: notes.NotesConfig) -> None:
    notes.write_note("A", "something", vault)
    assert "No notes matching" in notes.search_notes("zzzzzz", vault)


def test_frontmatter_is_stripped_from_search_snippets(vault: notes.NotesConfig) -> None:
    notes.write_note("Note", "the actual body text", vault)
    out = notes.search_notes("actual", vault)
    assert "source: lucy" not in out
    assert "# Note" not in out
    assert "the actual body text" in out


# --- unconfigured -------------------------------------------------------------


def test_no_vault_configured_says_so_rather_than_failing() -> None:
    config = notes.NotesConfig(None)
    assert "No notes vault is configured" in notes.write_note("t", "c", config)
    assert "No notes vault is configured" in notes.read_note("t", config)
    assert "No notes vault is configured" in notes.search_notes("q", config)


def test_a_missing_vault_directory_is_reported(tmp_path: Path) -> None:
    config = notes.NotesConfig(tmp_path / "does-not-exist")
    assert "does not exist" in notes.write_note("t", "c", config)


def test_note_tools_declare_the_notes_family_and_one_writer() -> None:
    built = notes.tools(notes.NotesConfig(None))
    assert {t.family for t in built} == {"notes"}
    assert [t.name for t in built if t.writes] == ["write_note"]


def test_only_two_note_tools_are_offered() -> None:
    """Measured: a third schema made the dispatcher truncate write_note's content."""
    assert {t.name for t in notes.tools(notes.NotesConfig(None))} == {
        "write_note",
        "search_notes",
    }


def test_a_single_search_hit_returns_the_whole_note(vault: notes.NotesConfig) -> None:
    """This is what lets read_note not exist."""
    notes.write_note("Coffee", "black, no sugar, and never after four", vault)
    out = notes.search_notes("coffee", vault)
    assert "black, no sugar, and never after four" in out
