"""Settings plus loaders for models.yaml / routes.yaml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEGEND_", env_file=".env", extra="ignore")

    ollama_host: str = "http://127.0.0.1:11434"
    models_file: Path = ROOT / "models.yaml"
    routes_file: Path = ROOT / "routes.yaml"

    # Ollama keeps at most this many models resident. The pinned tier occupies one
    # slot permanently, so 2 means "pinned + one swapped model".
    max_loaded_models: int = 2

    # Cap on cached routing decisions, keyed by prompt hash.
    decision_cache_size: int = 512

    # Skip stage 3 entirely; fall back to `default_route` when embeddings are unsure.
    disable_classifier: bool = False
    default_route: str = "chat"

    # Alias in models.yaml that answers "trivial" replies and backs the stage-3
    # classifier — the router "role", not necessarily a model literally named
    # "router". Currently the 350M `general` tier fills this role.
    router_alias: str = "general"

    # Routes worth staying on when the user sends a short follow-up. Only tiers that
    # cost something to reach belong here: sticking to a cheap tier is what the cascade
    # would do anyway, so listing one would just spend a routing pass to change nothing.
    sticky_routes: list[str] = ["think"]

    # Where a disputed answer goes. Being told "that's wrong" is the strongest signal
    # available that the previous tier was not good enough, so it escalates rather than
    # re-asking whichever model just got it wrong.
    escalate_route: str = "think"

    # Start loading the predicted model while classification is still running.
    preload_predicted: bool = True

    request_timeout: float = 300.0

    # --- effort and adjudication (app/effort.py, app/adjudicate.py) ---------
    #
    # "auto" lets app/effort.py estimate per request; "fast"/"standard"/"careful" pin
    # every request to one level, which is mostly useful for A/B runs of the eval suite.
    default_effort: str = "auto"

    # Master switch for the cross-model critic, and off by default for a measured reason:
    # **on this hardware, escalating dominates verifying.**
    #
    # A verification pass is the 1.2B reading a question and an answer and reasoning to a
    # verdict — 26.7 s median at the budget where it actually works (see the table in
    # app/adjudicate.py). Answering the question on the 1.2B instead costs about the same
    # ~25 s, and produces a better answer rather than a grade on a worse one. Verifying
    # therefore buys nothing the cheaper move does not, and in the failure case it costs
    # double, because a verdict of "incorrect" still has to be followed by a regeneration.
    #
    # The machinery stays because the reasoning is hardware-specific, not permanent: a
    # second model that could judge in 2 s would flip this immediately. Turn it on with
    # LEGEND_VERIFY_ENABLED=true, or per request with {"effort": "careful"}; the eval
    # harness reports what fraction of requests paid for it and whether anything changed.
    #
    # Off does not mean unchecked. The free deterministic checks — guardrails, and the
    # capitulation guard in app/adjudicate.py — run regardless. Only the paid one is off.
    verify_enabled: bool = False

    # Answer twice and compare the two answers numerically, abstaining when they
    # disagree. Not self-verification — no model judges anything, two numbers are
    # compared exactly — but it doubles latency on the slowest tier, so it is off until
    # the eval harness says the accuracy is worth it.
    self_consistency: bool = False

    # Which model does the judging. Must never be the model that produced the answer,
    # and must never be the 350M, which scores at chance as a critic (see models.yaml).
    critic_alias: str = "think"

    # --- retrieval (app/retrieval/) ----------------------------------------
    retrieval_enabled: bool = True

    # Which tier answers once a passage has been retrieved. Reading a document and
    # answering strictly from it is a different skill from recalling a fact, and it is
    # the one the 1.2B has. Measured: handed a chunk reading "the verifier is always the
    # 1.2B and never the 350M", the 350M answered "The model that verifies answers is
    # LFM2.5-350M" — it inverted the source. Retrieval only pays if the reader can read.
    reader_alias: str = "think"
    retrieval_db: Path = ROOT / "data" / "corpus.db"
    retrieval_top_k: int = 3

    # Append a "Sources: …" line to a retrieval-grounded reply. Computed by the server,
    # never written by the model — see the note in app/retrieval/service.py.
    retrieval_cite: bool = True

    # Cosine cut-off below which a hit is discarded. bge-small puts unrelated English
    # around 0.6, so this is well above the naive midpoint on purpose — injecting a
    # merely-plausible passage is how retrieval makes answers worse rather than better.
    # Calibrate with: uv run python scripts/ingest.py --probe "your question"
    retrieval_min_score: float = 0.66

    # Left unset until a name is picked. The system prompt (app/persona.py) reads
    # this and tells the model to say it doesn't have a name yet rather than either
    # inventing one or claiming to be ChatGPT/Claude/whatever it was trained near.
    assistant_name: str | None = None

    @property
    def ollama_env(self) -> dict[str, str]:
        return {"OLLAMA_MAX_LOADED_MODELS": str(self.max_loaded_models)}


class ModelSpec(BaseModel):
    alias: str
    tier: str
    repo: str
    file: str
    glob: list[str] = Field(default_factory=list)
    mmproj_glob: list[str] = Field(default_factory=list)
    mmproj_file: str | None = None
    keep_alive: str | int = "5m"
    num_ctx: int = 8192
    auto_download: bool = True
    optional: bool = False
    fallback: str | None = None
    embedding: bool = False
    thinking: bool = False
    tools: bool = False
    vision: bool = False
    # Explicit default for Ollama's `think` request field: True forces reasoning on,
    # False forces it off, None leaves the model/GGUF default untouched. Some models
    # (observed: unsloth Qwen3.5-2B) default to thinking ON and can burn the entire
    # token budget on reasoning before ever emitting a visible answer.
    default_think: bool | None = None
    # Default completion cap (num_predict) unless the caller passes max_tokens.
    # Reasoning models need real headroom: LFM2.5-1.2B-Thinking spent its whole 512
    # cap on a <think> block for a proof question and never reached an answer.
    default_max_tokens: int = 512

    # Which system-prompt length this tier gets (see app/persona.py). "brief" is for
    # models small enough that a long instruction block crowds out the actual request.
    persona: Literal["full", "brief"] = "full"

    # Per-tier sampling defaults. None means "let Ollama use its own default";
    # a caller-supplied temperature/top_p in the request always overrides these.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None

    def sampling_defaults(self) -> dict[str, float | int]:
        """Non-None sampling params, ready to merge into an Ollama options dict."""
        fields = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
        }
        return {k: v for k, v in fields.items() if v is not None}

    @property
    def tag(self) -> str:
        """The Ollama model tag this spec is imported as."""
        return f"legend/{self.alias}"


class RouteSpec(BaseModel):
    name: str
    model: str
    description: str = ""
    examples: list[str] = Field(default_factory=list)


class Thresholds(BaseModel):
    min_score: float = 0.42
    min_margin: float = 0.06


class ModelRegistry(BaseModel):
    models: list[ModelSpec]

    def by_alias(self, alias: str) -> ModelSpec:
        for m in self.models:
            if m.alias == alias:
                return m
        raise KeyError(f"no model with alias {alias!r} in models.yaml")

    def get(self, alias: str) -> ModelSpec | None:
        try:
            return self.by_alias(alias)
        except KeyError:
            return None

    @property
    def pinned(self) -> list[ModelSpec]:
        return [m for m in self.models if str(m.keep_alive) == "-1"]

    @property
    def embedder(self) -> ModelSpec:
        for m in self.models:
            if m.embedding:
                return m
        raise KeyError("models.yaml defines no embedding model")


class RouteTable(BaseModel):
    routes: list[RouteSpec]
    thresholds: Thresholds = Field(default_factory=Thresholds)

    def by_name(self, name: str) -> RouteSpec:
        for r in self.routes:
            if r.name == name:
                return r
        raise KeyError(f"no route named {name!r} in routes.yaml")


def load_models(path: Path) -> ModelRegistry:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults") or {}
    specs = [ModelSpec(**{**defaults, **entry}) for entry in raw.get("models") or []]
    if not specs:
        raise ValueError(f"{path} defines no models")
    return ModelRegistry(models=specs)


def load_routes(path: Path) -> RouteTable:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    routes = [
        RouteSpec(name=name, **(body or {}))
        for name, body in (raw.get("routes") or {}).items()
    ]
    if not routes:
        raise ValueError(f"{path} defines no routes")
    return RouteTable(routes=routes, thresholds=Thresholds(**(raw.get("thresholds") or {})))


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_registry() -> ModelRegistry:
    return load_models(get_settings().models_file)


@lru_cache
def get_route_table() -> RouteTable:
    return load_routes(get_settings().routes_file)


def hf_cache_dirs() -> list[Path]:
    """Places a downloaded GGUF might legitimately live on this machine."""
    candidates = [
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
        ROOT / "models",
    ]
    seen: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    return seen
