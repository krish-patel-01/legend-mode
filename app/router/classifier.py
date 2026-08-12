"""Stage 3: ask the pinned tiny model to pick a label.

Only reached when rules found no signal and the embedding margin was too thin, so this
runs on a small minority of requests. The prompt is deliberately terse and the output
is capped at a few tokens, since all we need back is one word.
"""

from __future__ import annotations

import logging
import time

from app.backends.ollama import OllamaClient
from app.config import ModelSpec, RouteTable
from app.router.types import RouteDecision

log = logging.getLogger(__name__)

_SYSTEM = """You are a request router. Read the user's message and reply with exactly \
one word naming the best category. Reply with the category name only: no punctuation, \
no explanation.

Categories:
{catalog}"""


class LlmClassifier:
    def __init__(self, client: OllamaClient, spec: ModelSpec, routes: RouteTable) -> None:
        self._client = client
        self._spec = spec
        self._routes = routes
        self._catalog = "\n".join(
            f"- {r.name}: {r.description}" for r in routes.routes
        )
        self._valid = {r.name for r in routes.routes}

    async def classify(self, text: str) -> RouteDecision | None:
        started = time.perf_counter()
        try:
            resp = await self._client.chat(
                self._spec,
                [
                    {"role": "system", "content": _SYSTEM.format(catalog=self._catalog)},
                    {"role": "user", "content": text[:2000]},
                ],
                options={"temperature": 0.0, "num_predict": 8},
            )
        except Exception as exc:
            log.warning("classifier failed, falling back: %s", exc)
            return None

        raw = (resp.get("message", {}).get("content") or "").strip().lower()
        elapsed = (time.perf_counter() - started) * 1000

        # A 350M model will sometimes pad the answer ("category: think."). Take the
        # first known label that appears anywhere in the reply.
        label = next((name for name in self._valid if name in raw), None)
        if label is None:
            log.debug("classifier returned unusable output: %r", raw[:120])
            return None

        return RouteDecision(
            route=label,
            stage="classifier",
            reason=f"{self._spec.alias} classified as {label!r}",
            confidence=0.6,
            elapsed_ms=elapsed,
        )
