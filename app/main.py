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
from app.router.engine import RouterEngine

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
    log.info("legend-mode ready (router=%s)", engine.router_spec.tag)

    yield

    await client.aclose()


app = FastAPI(title="Legend Mode", lifespan=lifespan)
app.include_router(api_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


# Registered last: catches everything the API router above didn't, so it never
# shadows /v1/*, /route/*, or /healthz.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
