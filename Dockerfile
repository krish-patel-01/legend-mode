# Legend Mode -- the router service only.
#
# This image deliberately does NOT contain Ollama or any model weights. The router is a
# few megabytes of Python whose job is to decide which model answers; the models are
# gigabytes of state that belong on a volume and outlive any image. Baking them in would
# produce a multi-gigabyte image that is stale as soon as models.yaml changes.
#
# See docker-compose.yml for the wiring, and note that models still have to be imported
# once into the Ollama volume:
#
#   docker compose run --rm router python scripts/import_models.py

# ---- builder ---------------------------------------------------------------
FROM python:3.12-slim AS builder

# uv resolves and installs from the committed lockfile. Pinned by digest-free tag on
# purpose: the lockfile, not uv's version, is what determines the dependency set.
COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /usr/local/bin/uv

WORKDIR /app

# Byte-compile on install and never build from source. Wheels exist for everything in
# the lockfile; if that stops being true the build should fail loudly here rather than
# silently pulling a compiler into the runtime image.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_INSTALLER_METADATA=1

# Only the dependency manifests, so this layer is cached until they actually change --
# editing app/ should not re-resolve the dependency tree.
COPY pyproject.toml uv.lock ./

# --frozen fails if uv.lock disagrees with pyproject.toml, which is what makes the
# build reproducible. --no-dev leaves pytest and ruff out of the runtime image.
#
# Note that pyproject sets `[tool.uv] package = false`, so this installs dependencies
# into /app/.venv without installing the project itself; the source is copied in below.
RUN uv sync --frozen --no-dev --no-install-project

# ---- runtime ---------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Runs as a normal user. The app writes only to data/, which is a volume owned by this
# user; everything else in the image can stay read-only.
RUN useradd --create-home --uid 10001 legend

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder --chown=legend:legend /app/.venv /app/.venv

# Source last: it is what changes most often, so it invalidates the fewest layers here.
COPY --chown=legend:legend app/ ./app/
COPY --chown=legend:legend scripts/ ./scripts/
COPY --chown=legend:legend models.yaml routes.yaml ./

# The retrieval index lives here. Created empty so the directory exists and is writable
# even when no volume is mounted -- an absent corpus is inert, not fatal (app/main.py).
RUN mkdir -p /app/data && chown legend:legend /app/data

USER legend

EXPOSE 8000

# Inside the container the service must bind 0.0.0.0 to be reachable at all; compose
# is what decides whether the published port is loopback-only. urllib rather than curl
# so the runtime image needs no extra packages.
#
# start-period is generous because the router warms the pinned tier and builds the
# routing centroids before it serves traffic, and a cold model load is seconds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
