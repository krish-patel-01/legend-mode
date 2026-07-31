"""Vector storage: sqlite for the text, numpy for the arithmetic.

No vector database. The corpus this serves is hundreds to low thousands of chunks, which
is a numpy matrix of a few megabytes — brute-force cosine over it costs well under a
millisecond, and every dedicated store would add a dependency, a daemon, or both to beat a
number that is already invisible next to a 1.2B's 28 s.

sqlite carries the text and the vectors survive restarts; the matrix is rebuilt in memory
on open. Vectors are stored as raw float32 so a row is 1.5 KB for bge-small's 384 dims
rather than the ~9 KB a JSON array would take.

The embedding model and its dimension are recorded in the file. Searching an index built
by a different embedder returns confident nonsense rather than an error, so that mismatch
is checked rather than trusted.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    source  TEXT    NOT NULL,
    ordinal INTEGER NOT NULL,
    heading TEXT    NOT NULL DEFAULT '',
    text    TEXT    NOT NULL,
    vector  BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
"""


class IndexMismatch(RuntimeError):
    """The stored index was built by a different embedding model."""


@dataclass(frozen=True)
class Hit:
    source: str
    heading: str
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.heading}" if self.heading else self.source


class VectorStore:
    """Brute-force cosine search over a small local corpus."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self._matrix: np.ndarray | None = None
        self._rows: list[tuple[str, str, str]] = []
        self._load()

    def close(self) -> None:
        self._db.close()

    # --- reading ------------------------------------------------------------

    def _load(self) -> None:
        """Pull every vector into one normalized matrix, so search is a single dot."""
        cur = self._db.execute(
            "SELECT source, heading, text, vector FROM chunks ORDER BY source, ordinal"
        )
        rows = cur.fetchall()
        if not rows:
            self._matrix, self._rows = None, []
            return

        self._rows = [(r[0], r[1], r[2]) for r in rows]
        matrix = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        self._matrix = _normalize(matrix)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def sources(self) -> list[str]:
        cur = self._db.execute("SELECT DISTINCT source FROM chunks ORDER BY source")
        return [r[0] for r in cur.fetchall()]

    def get_meta(self, key: str) -> str | None:
        cur = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def check_embedder(self, model: str, dim: int) -> None:
        """Raise if this index was built by a different embedder.

        An index searched with the wrong model does not fail — it returns plausible,
        wrong neighbours. Better to refuse than to silently ground answers in noise.
        """
        stored_model = self.get_meta("embed_model")
        stored_dim = self.get_meta("embed_dim")
        if stored_model is None:
            return  # empty index; whatever writes first sets the record
        if stored_model != model or (stored_dim and int(stored_dim) != dim):
            raise IndexMismatch(
                f"{self.path.name} was built with {stored_model!r} "
                f"({stored_dim} dims); this server embeds with {model!r} ({dim} dims). "
                f"Re-run scripts/ingest.py --rebuild."
            )

    def search(self, vector: list[float] | np.ndarray, k: int = 3) -> list[Hit]:
        if self._matrix is None or not self._rows:
            return []
        query = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        if query.shape[1] != self._matrix.shape[1]:
            return []
        scores = (self._matrix @ query.T).ravel()
        top = np.argsort(-scores)[: min(k, len(scores))]
        return [
            Hit(
                source=self._rows[i][0],
                heading=self._rows[i][1],
                text=self._rows[i][2],
                score=float(scores[i]),
            )
            for i in top
        ]

    # --- writing ------------------------------------------------------------

    def set_embedder(self, model: str, dim: int) -> None:
        self._db.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            [("embed_model", model), ("embed_dim", str(dim))],
        )
        self._db.commit()

    def replace_source(
        self, source: str, chunks: list[tuple[str, str]], vectors: list[list[float]]
    ) -> int:
        """Replace every chunk from one source. Re-ingesting a file is idempotent.

        `chunks` is (heading, text) pairs, positionally matched to `vectors`.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self._db.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self._db.executemany(
            "INSERT INTO chunks(source, ordinal, heading, text, vector) VALUES(?, ?, ?, ?, ?)",
            [
                (source, i, heading, text, np.asarray(v, dtype=np.float32).tobytes())
                for i, ((heading, text), v) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
        self._db.commit()
        self._load()
        return len(chunks)

    def drop_source(self, source: str) -> int:
        cur = self._db.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self._db.commit()
        self._load()
        return cur.rowcount

    def clear(self) -> None:
        self._db.execute("DELETE FROM chunks")
        self._db.commit()
        self._load()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
