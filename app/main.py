"""FastAPI app entrypoint.

    uv run uvicorn app.main:app --port 8000

Lifespan wires the Ollama client, model registry, and router engine together, warms
the pinned tier, and builds the embedding centroids before serving traffic.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router as api_router
from app.backends.ollama import OllamaClient
from app.config import get_registry, get_route_table, get_settings
from app.history import HistoryStore
from app.memory import MemoryStore
from app.retrieval import Retrieval, VectorStore
from app.router.engine import RouterEngine
from app.tools.registry import build_registry

STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("legend_mode")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    registry = get_registry()
    routes = get_route_table()

    client = OllamaClient(settings)
    try:
        version = await client.version()
        log.info("connected to Ollama %s at %s", version, settings.ollama_host)
    except Exception as exc:  # noqa: BLE001 - fail loudly but still let uvicorn boot
        log.error(
            "cannot reach Ollama at %s (%s). Is it running? "
            "Requests will error until it is.",
            settings.ollama_host, exc,
        )

    engine = RouterEngine(client, registry, routes, settings)
    await engine.warmup()

    app.state.client = client
    app.state.registry = registry
    app.state.routes = routes
    app.state.engine = engine
    app.state.history = HistoryStore()
    app.state.settings = settings
    retrieval = _open_retrieval(client, registry, settings)
    app.state.retrieval = retrieval
    # Same store, different source — see app/memory.py. Sharing it means memories go
    # through the gates and the threshold that documents already do.
    app.state.memory = (
        MemoryStore(client, registry.embedder, retrieval.store) if retrieval else None
    )
    # Built once: the families are fixed at startup and each Tool holds only a callable
    # and a schema, so there is nothing per-request to rebuild.
    app.state.tools = build_registry(settings) if settings.tools_enabled else None
    log.info("legend-mode ready (router=%s)", engine.router_spec.tag)

    yield

    await client.aclose()


def _open_retrieval(client, registry, settings) -> Retrieval | None:
    """Attach the corpus if there is one. An absent or unreadable index is not fatal.

    A missing corpus is the normal state on a fresh checkout — nothing has been ingested
    yet — and it degrades to exactly the behaviour that existed before retrieval, so it
    warrants a log line rather than a failed boot.
    """
    if not settings.retrieval_enabled:
        return None
    try:
        store = VectorStore(settings.retrieval_db)
    except Exception as exc:  # noqa: BLE001 - corpus is optional
        log.warning("retrieval disabled: cannot open %s (%s)", settings.retrieval_db, exc)
        return None

    if len(store) == 0:
        log.info(
            "retrieval index is empty (%s); run scripts/ingest.py to populate it",
            settings.retrieval_db,
        )
    else:
        log.info("retrieval: %d chunk(s) from %s", len(store), ", ".join(store.sources))

    return Retrieval(
        client,
        registry.embedder,
        store,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
        memory_min_score=settings.retrieval_memory_min_score,
    )


app = FastAPI(title="Legend Mode", lifespan=lifespan)
app.include_router(api_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


# Registered last: catches everything the API router above didn't, so it never
# shadows /v1/*, /route/*, or /healthz.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
