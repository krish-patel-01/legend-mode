"""Stage 2: nearest-centroid classification over bge-small embeddings.

Each route's example prompts are embedded once at startup and averaged into a unit
centroid. Classifying a prompt is then one embed call plus a 5x384 matrix-vector
product, so this stage costs roughly the latency of the embedding itself.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from app.backends.ollama import OllamaClient
from app.config import ModelRegistry, RouteTable
from app.router.types import RouteDecision

log = logging.getLogger(__name__)


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


class EmbeddingRouter:
    def __init__(
        self, client: OllamaClient, registry: ModelRegistry, routes: RouteTable
    ) -> None:
        self._client = client
        self._registry = registry
        self._routes = routes
        self._names: list[str] = []
        self._centroids: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self._centroids is not None

    async def build(self) -> None:
        """Embed every labeled example and average per route. Called once at startup."""
        names: list[str] = []
        vectors: list[np.ndarray] = []
        spec = self._registry.embedder

        for route in self._routes.routes:
            if not route.examples:
                log.warning("route %r has no examples; excluded from embedding stage", route.name)
                continue
            raw = await self._client.embed(spec, route.examples)
            centroid = _unit(np.asarray(raw, dtype=np.float32)).mean(axis=0)
            names.append(route.name)
            vectors.append(centroid)

        if not vectors:
            log.warning("no route examples at all; embedding stage disabled")
            return

        self._names = names
        self._centroids = _unit(np.vstack(vectors))
        log.info("embedding centroids built for %d routes", len(names))

    async def classify(self, text: str) -> RouteDecision | None:
        """Nearest centroid, or None if the match is too weak or too close to call."""
        if self._centroids is None:
            return None

        started = time.perf_counter()
        raw = await self._client.embed(self._registry.embedder, [text])
        query = _unit(np.asarray(raw[0], dtype=np.float32))

        sims = self._centroids @ query
        order = np.argsort(sims)[::-1]
        best, runner_up = int(order[0]), int(order[1]) if len(order) > 1 else None

        top = float(sims[best])
        margin = top - float(sims[runner_up]) if runner_up is not None else top
        scores = {name: round(float(s), 4) for name, s in zip(self._names, sims)}
        elapsed = (time.perf_counter() - started) * 1000

        th = self._routes.thresholds
        if top < th.min_score or margin < th.min_margin:
            log.debug(
                "embedding unsure: top=%s score=%.3f margin=%.3f",
                self._names[best], top, margin,
            )
            return None

        return RouteDecision(
            route=self._names[best],
            stage="embed",
            reason=f"nearest centroid (score {top:.2f}, margin {margin:.2f})",
            confidence=top,
            scores=scores,
            elapsed_ms=elapsed,
        )
