"""Cascade orchestration.

    rules  ->  embeddings  ->  tiny LLM  ->  default

Each stage may return a decision or defer. The first decision wins, and the stage that
produced it is recorded so /route/debug can explain any choice after the fact.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

from app.backends.ollama import OllamaClient
from app.config import ModelRegistry, ModelSpec, RouteTable, Settings
from app.router import rules
from app.router.classifier import LlmClassifier
from app.router.embed import EmbeddingRouter
from app.router.types import RouteDecision, RouteRequest

log = logging.getLogger(__name__)


def extract_text(messages: list[dict[str, Any]]) -> str:
    """Last user turn, flattened. OpenAI allows content to be a list of parts."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "\n".join(filter(None, parts))
    return ""


def anchor_text(messages: list[dict[str, Any]]) -> str:
    """The most recent user turn that isn't itself a follow-up.

    Walks back past a run of "its incorrect" / "nope" / "why" so the sticky stage can
    ask which tier the thread's actual question belongs on. Returns "" when every prior
    user turn is a follow-up, or when there is no prior turn at all.
    """
    users = [m for m in messages if m.get("role") == "user"]
    for msg in reversed(users[:-1]):  # exclude the turn being routed
        text = extract_text([msg])
        if text.strip() and rules.followup_kind(text) is None:
            return text
    return ""


def has_images(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            if any(
                isinstance(p, dict) and p.get("type") in {"image_url", "input_image"}
                for p in content
            ):
                return True
        if msg.get("images"):  # Ollama-native shape, accepted for convenience
            return True
    return False


class RouterEngine:
    def __init__(
        self,
        client: OllamaClient,
        registry: ModelRegistry,
        routes: RouteTable,
        settings: Settings,
    ) -> None:
        self._client = client
        self._registry = registry
        self._routes = routes
        self._settings = settings

        self._router_spec = self._resolve_router_spec()
        self.embedder = EmbeddingRouter(client, registry, routes)
        self.classifier = LlmClassifier(client, self._router_spec, routes)

        self._cache: OrderedDict[str, RouteDecision] = OrderedDict()

        # One large-tier generation at a time. Two concurrent requests would otherwise
        # thrash the model in and out of a single residency slot, which on CPU costs
        # far more than simply queueing.
        self._large_lock = asyncio.Lock()

    # --- setup --------------------------------------------------------------

    def _resolve_router_spec(self) -> ModelSpec:
        """Prefer the configured router model; fall back if it was never imported."""
        spec = self._registry.by_alias(self._settings.router_alias)
        return spec

    async def verify_models(self) -> ModelSpec:
        """Swap in the fallback router if the primary tag is missing from Ollama."""
        try:
            available = await self._client.tags()
        except Exception as exc:  # noqa: BLE001 - startup diagnostics only
            log.warning("could not list Ollama tags: %s", exc)
            return self._router_spec

        primary = self._registry.by_alias(self._settings.router_alias)
        if primary.tag in available:
            return self._router_spec

        fallback = self._registry.get(primary.fallback or "")
        if fallback and fallback.tag in available:
            log.warning(
                "%s not imported; routing with fallback %s. "
                "Run scripts/import_models.py once the 350M is downloaded.",
                primary.tag,
                fallback.tag,
            )
            self._router_spec = fallback
            self.classifier = LlmClassifier(self._client, fallback, self._routes)
        else:
            log.error(
                "neither %s nor its fallback is imported. Run scripts/import_models.py.",
                primary.tag,
            )
        return self._router_spec

    @property
    def router_spec(self) -> ModelSpec:
        return self._router_spec

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    async def warmup(self) -> None:
        await self.verify_models()
        try:
            await self.embedder.build()
        except Exception as exc:  # noqa: BLE001 - degrade to rules + classifier
            log.warning("embedding stage unavailable: %s", exc)
        for spec in self._registry.pinned:
            await self._client.preload(spec)

    # --- resolution ---------------------------------------------------------

    def spec_for(self, decision: RouteDecision) -> ModelSpec:
        """Map a decision onto a concrete model, tolerating an unknown route."""
        if decision.stage == "override":
            if spec := self._registry.get(decision.model):
                return spec
            raise KeyError(f"unknown model alias {decision.model!r}")

        try:
            alias = self._routes.by_name(decision.route).model
        except KeyError:
            log.warning("route %r not in routes.yaml; using default", decision.route)
            alias = self._routes.by_name(self._settings.default_route).model

        # The router role resolves to whichever model is actually backing it — not
        # necessarily an alias literally named "router" (see Settings.router_alias).
        if alias == self._settings.router_alias:
            return self._router_spec
        return self._registry.by_alias(alias)

    def spec_for_route(self, name: str) -> ModelSpec | None:
        """The model behind a named route, or None if the route isn't defined.

        Lets callers name a *role* rather than an alias. app/api.py uses it to find the
        reading tier, which should track whatever `chat` runs on rather than being pinned
        to a quantisation-specific alias that drifts the next time a model is swapped.
        """
        try:
            alias = self._routes.by_name(name).model
        except KeyError:
            return None
        if alias == self._settings.router_alias:
            return self._router_spec
        return self._registry.get(alias)

    def lock_for(self, spec: ModelSpec) -> asyncio.Lock | None:
        return self._large_lock if spec.tier == "large" else None

    # --- the cascade --------------------------------------------------------

    async def route(self, req: RouteRequest) -> RouteDecision:
        started = time.perf_counter()

        if decision := rules.apply(req):
            decision.elapsed_ms = (time.perf_counter() - started) * 1000
            return self._finalize(decision)

        if decision := await self._sticky(req):
            decision.elapsed_ms = (time.perf_counter() - started) * 1000
            return self._finalize(decision)

        key = self._cache_key(req)
        if cached := self._cache_get(key):
            hit = cached.model_copy(
                update={
                    "stage": "cache",
                    "origin": cached.stage,
                    "reason": f"cached ({cached.stage}: {cached.reason})",
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
            )
            return hit

        decision = await self._slow_path(req)
        decision.elapsed_ms = (time.perf_counter() - started) * 1000
        decision = self._finalize(decision)
        self._cache_put(key, decision)
        return decision

    async def _sticky(self, req: RouteRequest) -> RouteDecision | None:
        """Keep a short follow-up on the tier that answered the turn it refers to.

        Runs after rules so it can never override a structural signal — "who are you?"
        mid-thread is still an identity question, not a continuation.

        Which tier handled the previous turn is recovered by routing that turn's text
        again rather than by storing conversation state. Routing is a pure function of
        the message, so the answer is identical, and the decision cache means the
        previous turn was almost always resolved a moment ago and costs nothing to look
        up. The recursive call passes no `prev_user_text`, so it cannot recurse further.
        """
        kind = rules.followup_kind(req.text, req.message_count)
        if kind is None:
            return None

        # An explicit "that's wrong", or a correction supplying a missed fact, needs no
        # history: whatever answered last was not good enough, so send it somewhere
        # better. They differ in the instruction attached at generation time, not here.
        if kind in ("dispute", "correction"):
            return RouteDecision(
                route=self._settings.escalate_route,
                stage="sticky",
                reason=(
                    "user disputed the previous answer; escalating"
                    if kind == "dispute"
                    else "user corrected or added a missed detail; escalating"
                ),
                confidence=0.75,
                followup=kind,
            )

        if not req.anchor_text:
            return None

        anchor = await self.route(
            RouteRequest(
                text=req.anchor_text,
                message_count=max(req.message_count - 2, 1),
            )
        )
        if anchor.route not in self._settings.sticky_routes:
            return None

        return RouteDecision(
            route=anchor.route,
            stage="sticky",
            reason=f"{kind} in a {anchor.route!r} thread; staying on that tier",
            confidence=0.7,
            followup=kind,
        )

    async def _slow_path(self, req: RouteRequest) -> RouteDecision:
        if decision := await self.embedder.classify(req.text):
            return decision

        if not self._settings.disable_classifier:
            if decision := await self.classifier.classify(req.text):
                return self._guard_cheap_guess(decision)

        return RouteDecision(
            route=self._settings.default_route,
            stage="fallback",
            reason="no stage was confident; using default route",
            confidence=0.3,
        )

    def _guard_cheap_guess(self, decision: RouteDecision) -> RouteDecision:
        """Don't let stage 3 route a request it didn't recognise to the cheapest tier.

        Reaching the classifier means rules found nothing and the embedding margin was
        too thin. Everything `trivial` is actually for — greetings, acknowledgements,
        identity questions — is caught by regex in app/router/rules.py before any model
        runs, so a classifier verdict of `trivial` is not a cheap request being spotted;
        it is the 350M guessing about a request that already looked unfamiliar.

        The case that prompted this: "please check in the web then answer that question"
        was labelled `trivial` here, and the 350M read "check in" as hotel check-in and
        invented a stay in Mexico City. `confidence` cannot express this — the classifier
        reports a flat 0.6 for every verdict it returns — so the escalation keys on the
        stage and the route, which are the two facts that actually carry the doubt.
        """
        if not self._settings.escalate_classifier_trivial:
            return decision
        if decision.route != "trivial" or decision.route == self._settings.default_route:
            return decision
        log.info("classifier guessed 'trivial' for an unrecognised request; escalating")
        return decision.model_copy(
            update={
                "route": self._settings.default_route,
                "reason": f"{decision.reason}; escalated (stage 3 is a guess, not a match)",
            }
        )

    def _finalize(self, decision: RouteDecision) -> RouteDecision:
        if decision.stage != "override":
            try:
                decision.model = self.spec_for(decision).alias
            except KeyError:
                decision.model = self._settings.default_route
        return decision

    # --- decision cache -----------------------------------------------------

    def _cache_key(self, req: RouteRequest) -> str:
        blob = f"{req.text}|{req.has_tools}|{req.has_images}|{req.message_count > 1}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> RouteDecision | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _cache_put(self, key: str, decision: RouteDecision) -> None:
        self._cache[key] = decision
        self._cache.move_to_end(key)
        while len(self._cache) > self._settings.decision_cache_size:
            self._cache.popitem(last=False)


# --- extension point --------------------------------------------------------
#
# Server-side tool execution goes here. Today the API passes `tools` straight to the
# large tier and returns `tool_calls` to the caller unmodified. A future agent loop
# would sit between `route()` and the API layer: dispatch tool_calls against a local
# registry, append the results as `role: "tool"` messages, and re-enter the model —
# re-routing on each turn so a cheap follow-up doesn't stay pinned to the large model.
