"""Build the retrieval corpus (ROADMAP step 4).

    uv run python scripts/ingest.py                     # re-ingest the default docs
    uv run python scripts/ingest.py notes/ handbook.md  # ingest specific paths
    uv run python scripts/ingest.py --vault             # the notes vault
    uv run python scripts/ingest.py --list              # what is indexed
    uv run python scripts/ingest.py --probe "who verifies answers here"
    uv run python scripts/ingest.py --rebuild           # drop everything first

`--vault` is what closes the loop between the two ways this assistant remembers things.
`app/tools/notes.py` writes notes when asked to; indexing them puts what it wrote into
the same corpus as everything else, so recall no longer depends on the user phrasing a
request in a way `app/tools/gate.py` recognises as a note operation. Measured before it
existed: asked "when is the Q3 review" with a note saying the 14th, the gate matched
nothing, no tool ran, and the model answered "next week" from nowhere. Explicit note
operations stay with the tools; implicit recall belongs to retrieval, which is semantic
and needs no pattern to match.

`--probe` is the important one. The similarity threshold that decides whether retrieved
text is injected at all cannot be guessed: bge-small puts unrelated English around 0.6
cosine, so a plausible-sounding 0.5 cut-off would inject on every question and reproduce
the 5.0-point GPQA regression the roadmap cites. Probe a few questions you care about,
look at where the relevant chunk actually scores against where the irrelevant ones do,
and set LEGEND_RETRIEVAL_MIN_SCORE between them.

Talks to Ollama directly rather than through the running server: ingesting is a
maintenance job, and requiring the API to be up to index a file would be a needless
coupling. The embedder is the same pinned bge model either way.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backends.ollama import OllamaClient  # noqa: E402
from app.config import get_registry, get_settings  # noqa: E402
from app.retrieval.chunk import MIN_CHARS, NOTE_MIN_CHARS, chunk_markdown  # noqa: E402
from app.retrieval.service import _normalize_query  # noqa: E402
from app.retrieval.store import VectorStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# The corpus that ships with the project: its own documentation. Nothing here is
# knowledge the models could plausibly hold, which makes it an honest test of whether
# retrieval works — unlike seeding the corpus with the answers to the factual eval cases,
# which would measure nothing except that the harness can read a file.
DEFAULT_SOURCES = ["README.md", "ROADMAP.md"]

SUFFIXES = {".md", ".txt", ".markdown", ".rst"}

# Ollama's embedding endpoint takes a list, but a long list on CPU is one long blocking
# call. Batching keeps progress visible and memory flat.
BATCH = 32


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = (ROOT / raw) if not Path(raw).is_absolute() else Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in SUFFIXES))
        elif path.is_file():
            files.append(path)
        else:
            print(f"  skip (not found): {raw}", file=sys.stderr)
    return files


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


async def embed_all(client: OllamaClient, spec, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        vectors.extend(await client.embed(spec, batch))
        print(f"    embedded {min(start + BATCH, len(texts))}/{len(texts)}", end="\r")
    print(" " * 40, end="\r")
    return vectors


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    registry = get_registry()
    embed_spec = registry.embedder
    store = VectorStore(settings.retrieval_db)
    client = OllamaClient(settings)

    try:
        if args.list:
            print(f"{settings.retrieval_db}  —  {len(store)} chunk(s)")
            for source in store.sources:
                print(f"  {source}")
            if not store.sources:
                print("  (empty; run without --list to ingest the default docs)")
            return 0

        if args.probe:
            if not len(store):
                print("corpus is empty; nothing to probe", file=sys.stderr)
                return 1
            # **Normalised exactly as the service normalises it.** This did not used to
            # be, and the discrepancy pointed the wrong way at the worst possible moment:
            # probing "what is the current gold price?" reported 0.626 against a README
            # chunk, comfortably below the 0.66 cut-off, while the live request scored the
            # same pair at roughly 0.68 and injected it. The whole difference was the
            # trailing question mark, which app/retrieval/service.py strips and this did
            # not — worth 0.057 cosine, per the measurement in `_normalize_query`.
            #
            # A calibration instrument that disagrees with the thing it calibrates is
            # worse than none, because every threshold set with it is set from the wrong
            # number.
            query = _normalize_query(args.probe)
            vectors = await client.embed(embed_spec, [query])
            shown = f"{args.probe!r}" + (f" -> {query!r}" if query != args.probe else "")
            print(f"\nprobe: {shown}   (threshold is "
                  f"{settings.retrieval_min_score}, {len(store)} chunks indexed)\n")
            for hit in store.search(vectors[0], args.top_k):
                mark = "USED " if hit.score >= settings.retrieval_min_score else "below"
                snippet = " ".join(hit.text.split())[:96]
                print(f"  [{mark}] {hit.score:.3f}  {hit.citation}")
                print(f"           {snippet}")
            return 0

        if args.rebuild:
            store.clear()
            print("cleared existing index")

        paths = list(args.paths)
        if args.vault:
            if settings.vault_path is None:
                print("no vault configured; set LEGEND_VAULT_PATH", file=sys.stderr)
                return 1
            paths.append(str(settings.vault_path))

        files = collect(paths or DEFAULT_SOURCES)
        if not files:
            print("nothing to ingest", file=sys.stderr)
            return 1

        vault = settings.vault_path.resolve() if settings.vault_path else None

        total = 0
        for path in files:
            source = relative(path)
            # A note gets the lower floor; anything else keeps the document threshold.
            in_vault = vault is not None and path.resolve().is_relative_to(vault)
            chunks = chunk_markdown(
                path.read_text(encoding="utf-8", errors="replace"),
                min_chars=NOTE_MIN_CHARS if in_vault else MIN_CHARS,
            )
            if not chunks:
                print(f"  {source}: no chunks (too short?)")
                continue
            vectors = await embed_all(client, embed_spec, [c.embed_text for c in chunks])
            store.set_embedder(embed_spec.tag, len(vectors[0]))
            written = store.replace_source(
                source, [(c.heading, c.text) for c in chunks], vectors
            )
            total += written
            print(f"  {source}: {written} chunk(s)")

        print(f"\n{total} chunk(s) ingested; {len(store)} in the index "
              f"-> {settings.retrieval_db}")
        return 0
    finally:
        await client.aclose()
        store.close()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help=f"files or dirs (default: {DEFAULT_SOURCES})")
    ap.add_argument(
        "--vault",
        action="store_true",
        help="ingest the notes vault named by LEGEND_VAULT_PATH",
    )
    ap.add_argument("--rebuild", action="store_true", help="drop the whole index first")
    ap.add_argument("--list", action="store_true", help="show what is indexed")
    ap.add_argument("--probe", help="score a question against the corpus and exit")
    ap.add_argument("--top-k", type=int, default=6, help="hits to show when probing")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
