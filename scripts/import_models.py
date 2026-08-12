"""Import every model in models.yaml into Ollama as `legend/<alias>`.

Idempotent: models already tagged are skipped unless --force is passed.

    uv run python scripts/import_models.py
    uv run python scripts/import_models.py --only router,large --force
    uv run python scripts/import_models.py --no-download   # fail instead of fetching
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ModelSpec, get_registry
from app.resolve import resolve, resolve_mmproj

log = logging.getLogger("import")


def ollama(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ollama", *args], capture_output=True, text=True, check=check, encoding="utf-8"
    )


def existing_tags() -> set[str]:
    proc = ollama("list", check=False)
    if proc.returncode != 0:
        raise SystemExit(
            "could not talk to Ollama. Is it installed and running?\n" + proc.stderr
        )
    tags: set[str] = set()
    for line in proc.stdout.splitlines()[1:]:
        if name := line.split()[0:1]:
            tags.add(name[0])
            tags.add(name[0].removesuffix(":latest"))
    return tags


def modelfile_for(spec: ModelSpec, weights: Path, mmproj: Path | None) -> str:
    lines = [f"FROM {weights}"]
    if mmproj:
        lines.append(f"FROM {mmproj}")
    lines.append(f"PARAMETER num_ctx {spec.num_ctx}")
    return "\n".join(lines) + "\n"


def import_one(spec: ModelSpec, *, force: bool, allow_download: bool, tags: set[str]) -> str:
    """Returns a one-word status for the summary line."""
    if spec.tag in tags and not force:
        return "skipped"

    try:
        weights = resolve(spec, allow_download=allow_download)
    except Exception as exc:
        log.error("%s: download failed: %s", spec.alias, exc)
        return "failed"

    if weights is None:
        level = log.info if spec.optional else log.error
        level("%s: weights not found (%s / %s)", spec.alias, spec.repo, spec.file)
        return "missing"

    mmproj = resolve_mmproj(spec, allow_download=allow_download)
    log.info("%s -> %s%s", spec.tag, weights.name, " (+mmproj)" if mmproj else "")

    # ollama create wants a file path; NamedTemporaryFile on Windows can't be reopened
    # while held, so write, close, then hand over the path.
    with tempfile.TemporaryDirectory() as tmp:
        mf = Path(tmp) / "Modelfile"
        mf.write_text(modelfile_for(spec, weights, mmproj), encoding="utf-8")
        proc = ollama("create", spec.tag, "-f", str(mf), check=False)

    if proc.returncode != 0:
        log.error("%s: ollama create failed\n%s", spec.alias, proc.stderr.strip())
        return "failed"
    return "imported"


def smoke_test(spec: ModelSpec) -> bool:
    """One tiny generation to prove the tag actually loads and has a chat template."""
    if spec.embedding:
        return True  # embedding models have no chat template to exercise
    proc = ollama("run", spec.tag, "hi", check=False)
    if proc.returncode != 0:
        log.error("%s: smoke test failed\n%s", spec.alias, proc.stderr.strip())
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="comma-separated aliases to import")
    ap.add_argument("--force", action="store_true", help="re-import even if the tag exists")
    ap.add_argument(
        "--no-download", action="store_true", help="never fetch from Hugging Face"
    )
    ap.add_argument("--smoke", action="store_true", help="generate one token per model")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    registry = get_registry()
    specs = registry.models
    if args.only:
        wanted = {a.strip() for a in args.only.split(",")}
        specs = [s for s in specs if s.alias in wanted]
        if missing := wanted - {s.alias for s in specs}:
            raise SystemExit(f"unknown aliases: {', '.join(sorted(missing))}")

    tags = existing_tags()
    results = {
        spec.alias: import_one(
            spec, force=args.force, allow_download=not args.no_download, tags=tags
        )
        for spec in specs
    }

    if args.smoke:
        for spec in specs:
            if results[spec.alias] in {"imported", "skipped"} and not smoke_test(spec):
                results[spec.alias] = "failed"

    print("\n--- summary ---")
    for alias, status in results.items():
        print(f"  {alias:<12} {status}")

    # A missing *optional* model (the 230M fallback) is expected and not an error.
    fatal = [
        alias
        for alias, status in results.items()
        if status == "failed"
        or (status == "missing" and not registry.by_alias(alias).optional)
    ]
    if fatal:
        print(f"\nfailed: {', '.join(fatal)}", file=sys.stderr)
        return 1
    print("\nall good. start the server with: uv run uvicorn app.main:app --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
