"""Stage 2: nearest-example classification over bge-small embeddings.

Every labeled example in `routes.yaml` is embedded once at startup and kept. A prompt
scores against a route by its *closest single example*, not by distance to the route's
average. Classifying is then one embed call plus a 55x384 matrix-vector product, so this
stage still costs about the latency of the embedding itself.

**It used to average, and that made the stage almost inert.** Over 82 prompts — the eval
set plus twenty ordinary requests — the embedding stage decided exactly once; everything
else fell through to the 350M classifier or to the fallback default, and 80% of everyday
prompts reached `fallback`. Nothing was broken: the embed calls succeeded and the top
route was usually right. The *margin* was the problem.

Averaging fifteen diverse examples pulls every centroid toward the same "average English
sentence" direction, so the centroids end up similar to each other and a query sits
roughly equidistant from all of them. Measured on ten prompts with known answers:

    prompt                                   centroid            nearest example
    explain what a REST API is in one line   chat 0.773 m0.048   chat 0.908 m0.246
    what is the capital of France            triv 0.653 m0.056   triv 1.000 m0.473
    tell me a joke                           chat 0.761 m0.018   chat 0.817 m0.144
    rewrite this more politely               chat 0.779 m0.040   chat 0.906 m0.233
    why does this recursion overflow         think 0.756 m0.066  think 0.974 m0.240

    accepted and correct:  centroid 3/10     nearest example 7/10

Top scores were fine under both — 0.64 to 0.87, comfortably over `min_score` 0.42. It was
`min_margin` 0.06 that rejected nearly everything, because the centroids were crowded
together. Nearest-example margins run 5-10x wider, which is what a threshold can actually
separate on.

Where it still declines it declines *safely*: "how do I undo the last git commit" and
"what does idempotent mean" both fall through rather than misroute, which is the stage
doing its job.

This is also what the original design specified — compare against a labeled example bank
and accept on the top-1 margin. The centroid was an implementation shortcut that quietly
changed the method.
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
        self._examples: np.ndarray | None = None
        # Row i of `_examples` belongs to route `_names[_owner[i]]`. One flat matrix plus
        # an owner index beats a list of per-route matrices: scoring is a single dot
        # product against everything, then a grouped max.
        self._owner: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self._examples is not None

    async def build(self) -> None:
        """Embed every labeled example and keep them all. Called once at startup."""
        names: list[str] = []
        rows: list[np.ndarray] = []
        owners: list[int] = []
        spec = self._registry.embedder

        for route in self._routes.routes:
            if not route.examples:
                log.warning("route %r has no examples; excluded from embedding stage", route.name)
                continue
            raw = await self._client.embed(spec, route.examples)
            vectors = _unit(np.asarray(raw, dtype=np.float32))
            index = len(names)
            names.append(route.name)
            rows.append(vectors)
            owners.extend([index] * len(vectors))

        if not rows:
            log.warning("no route examples at all; embedding stage disabled")
            return

        self._names = names
        self._examples = np.vstack(rows)
        self._owner = np.asarray(owners, dtype=np.int32)
        log.info(
            "embedding bank built: %d examples across %d routes",
            len(self._owner), len(names),
        )

    async def classify(self, text: str) -> RouteDecision | None:
        """Nearest example, or None if the match is too weak or too close to call."""
        if self._examples is None or self._owner is None:
            return None

        started = time.perf_counter()
        raw = await self._client.embed(self._registry.embedder, [text])
        query = _unit(np.asarray(raw[0], dtype=np.float32))

        # Best single example per route. `np.maximum.at` groups without a Python loop.
        per_example = self._examples @ query
        sims = np.full(len(self._names), -1.0, dtype=np.float32)
        np.maximum.at(sims, self._owner, per_example)

        order = np.argsort(sims)[::-1]
        best = int(order[0])
        runner_up = int(order[1]) if len(order) > 1 else None

        top = float(sims[best])
        margin = top - float(sims[runner_up]) if runner_up is not None else top
        scores = {name: round(float(s), 4) for name, s in zip(self._names, sims, strict=True)}
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
            reason=f"nearest example (score {top:.2f}, margin {margin:.2f})",
            confidence=top,
            scores=scores,
            elapsed_ms=elapsed,
        )
