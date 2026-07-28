"""Async client for the Ollama daemon.

Ollama owns model residency: it lazily loads on first token and evicts by LRU once
`OLLAMA_MAX_LOADED_MODELS` is exceeded. All this layer does is pass the right
`keep_alive` per tier so the pinned models are never the ones evicted.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import ModelSpec, Settings

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_host, timeout=settings.request_timeout
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- health -------------------------------------------------------------

    async def version(self) -> str:
        resp = await self._client.get("/api/version")
        resp.raise_for_status()
        return resp.json().get("version", "unknown")

    async def tags(self) -> set[str]:
        """Model tags Ollama currently knows about."""
        resp = await self._client.get("/api/tags")
        resp.raise_for_status()
        names = {m["name"] for m in resp.json().get("models", [])}
        return names | {n.removesuffix(":latest") for n in names}

    async def loaded(self) -> list[dict[str, Any]]:
        """Currently resident models, as reported by /api/ps."""
        resp = await self._client.get("/api/ps")
        resp.raise_for_status()
        return resp.json().get("models", [])

    # --- inference ----------------------------------------------------------

    async def chat(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        payload = self._chat_payload(spec, messages, tools, options, think, stream=False)
        resp = await self._client.post("/api/chat", json=payload)
        if resp.status_code >= 400:
            raise OllamaError(f"{spec.tag}: {resp.status_code} {resp.text}")
        return resp.json()

    async def chat_stream(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        payload = self._chat_payload(spec, messages, tools, options, think, stream=True)
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode(errors="replace")
                raise OllamaError(f"{spec.tag}: {resp.status_code} {body}")
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    log.warning("unparseable chunk from %s: %r", spec.tag, line[:200])

    def _chat_payload(
        self,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        options: dict[str, Any] | None,
        think: bool | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": spec.tag,
            "messages": messages,
            "stream": stream,
            "keep_alive": spec.keep_alive,
            # Precedence: model's tuned sampling defaults, then num_ctx, then
            # whatever the caller passed — a caller-supplied temperature/top_p
            # always wins over the tier's own default.
            "options": {**spec.sampling_defaults(), "num_ctx": spec.num_ctx, **(options or {})},
        }
        if tools:
            payload["tools"] = tools
        # An explicit per-call override wins; otherwise fall back to the model's
        # configured default (see ModelSpec.default_think). Leave the field out
        # entirely when neither is set, since some GGUFs reject `think` outright.
        resolved_think = think if think is not None else spec.default_think
        if resolved_think is not None:
            payload["think"] = resolved_think
        return payload

    async def embed(self, spec: ModelSpec, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post(
            "/api/embed",
            json={"model": spec.tag, "input": texts, "keep_alive": spec.keep_alive},
        )
        if resp.status_code >= 400:
            raise OllamaError(f"{spec.tag}: {resp.status_code} {resp.text}")
        return resp.json()["embeddings"]

    # --- residency ----------------------------------------------------------

    async def preload(self, spec: ModelSpec) -> None:
        """Load weights without generating. An empty message list does exactly this."""
        try:
            await self._client.post(
                "/api/chat",
                json={"model": spec.tag, "messages": [], "keep_alive": spec.keep_alive},
            )
        except httpx.HTTPError as exc:
            # Warming is best-effort; the real request will load it anyway.
            log.debug("preload of %s failed: %s", spec.tag, exc)

    async def unload(self, spec: ModelSpec) -> None:
        """Evict immediately. keep_alive of 0 tells Ollama to drop it now."""
        try:
            await self._client.post(
                "/api/chat", json={"model": spec.tag, "messages": [], "keep_alive": 0}
            )
        except httpx.HTTPError as exc:
            log.debug("unload of %s failed: %s", spec.tag, exc)
