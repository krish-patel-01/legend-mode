"""Locate a model's .gguf on disk, downloading it only when necessary.

Shared by the import script and by startup diagnostics, so both agree on where a
model lives and whether it is present at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import ModelSpec, hf_cache_dirs

log = logging.getLogger(__name__)

# HF stores incomplete downloads alongside real ones. Never resolve to those.
_JUNK_PARTS = {"blobs", "incomplete"}
_MIN_BYTES = 1024 * 1024  # anything smaller is a stub/pointer, not weights


def _plausible(p: Path) -> bool:
    if any(part in _JUNK_PARTS for part in p.parts):
        return False
    try:
        return p.stat().st_size >= _MIN_BYTES
    except OSError:
        return False


def find_local(patterns: list[str]) -> Path | None:
    """First real .gguf matching any pattern, searched across known cache roots."""
    for root in hf_cache_dirs():
        if not root.is_dir():
            continue
        for pattern in patterns:
            matches = sorted(p for p in root.glob(pattern) if _plausible(p))
            if matches:
                return matches[0]
    return None


def download(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    log.info("downloading %s/%s ...", repo, filename)
    return Path(hf_hub_download(repo_id=repo, filename=filename))


def resolve(spec: ModelSpec, allow_download: bool = True) -> Path | None:
    """Path to the model weights, or None if absent and we may not fetch them."""
    patterns = spec.glob or [f"**/{spec.file}"]
    if found := find_local(patterns):
        return found
    if allow_download and spec.auto_download:
        return download(spec.repo, spec.file)
    return None


def resolve_mmproj(spec: ModelSpec, allow_download: bool = True) -> Path | None:
    """Vision projector for multimodal models. Absence is never fatal."""
    if not spec.mmproj_file:
        return None
    patterns = spec.mmproj_glob or [f"**/{spec.mmproj_file}"]
    if found := find_local(patterns):
        return found
    if allow_download and spec.auto_download:
        try:
            return download(spec.repo, spec.mmproj_file)
        except Exception as exc:
            log.warning("could not fetch mmproj for %s: %s", spec.alias, exc)
    return None
