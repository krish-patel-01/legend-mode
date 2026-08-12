# Legend Mode

**A local model router. A 350M model reads every request and decides which model should
answer it — then hot-swaps that model in only if the request actually needs it.**

[![CI](https://github.com/krish-patel-01/legend-mode/actions/workflows/ci.yml/badge.svg)](https://github.com/krish-patel-01/legend-mode/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Runs entirely on CPU. No GPU, no API keys, no data leaving the machine. It speaks the
OpenAI chat-completions API, so anything that talks to OpenAI talks to this.

## Why

Running one big model for every request wastes RAM and time when most messages are "hi"
or "thanks". Legend Mode keeps two small models pinned in memory and swaps a heavier one
in only for the requests that earn it — then spends the time it saved on the things that
actually improve answers: computing exact facts instead of generating them, retrieving
sources, and checking the result.

The goal is accuracy on a laptop that stands up against a single 10–15B model. The path
is architectural, not more parameters — nothing bigger fits.

## The models

| Tier | Model | Residency | Role |
|---|---|---|---|
| `general` | LFM2.5-350M | pinned | trivial replies, routing, tool selection |
| `embed` | bge-small-en-v1.5 | pinned | routing embeddings only, never answers |
| `think` | LFM2.5-1.2B-Thinking | swapped | reasoning, and the only tier that can verify |

Two answering models, deliberately. That number came out of a measurement rather than a
preference — see [the finding that shaped everything](#the-finding-that-shaped-everything).

## Quickstart

### With Docker

```bash
git clone https://github.com/krish-patel-01/legend-mode.git
cd legend-mode

docker compose up -d --build                              # router + Ollama
docker compose run --rm router python scripts/import_models.py   # ~1 GB, once
```

Then open <http://127.0.0.1:8000>.

For the `web` tool family, generate the SearXNG config once and start it:

```bash
./deploy/searxng/up.sh                       # writes settings.yml with a fresh key
docker compose --profile tools up -d
```

### On the host

Needs [Ollama](https://ollama.com) installed and running.

```bash
uv sync
uv run python scripts/import_models.py       # resolves local GGUFs, downloads the rest
uv run uvicorn app.main:app --port 8000
```

Optionally give Ollama a slot per tier with `OLLAMA_MAX_LOADED_MODELS=4 ollama serve`.
Measured, it makes no difference to speed; it just avoids reloads.

## Using it

Point any OpenAI-compatible client at `http://127.0.0.1:8000/v1`:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hi"}]}'
```

- `model: "auto"` (or omitted) lets the cascade choose. Any alias — `general`, `think` —
  pins that tier instead.
- **Every response says why it was answered that way.** `x_legend_route` in the body and
  `X-Legend-Route` in the headers carry which model answered, which routing stage decided,
  its reasoning, which guardrail fired, whether retrieval or a tool ran, and the effort
  level chosen. When you report a bug, include this.
- `effort: "fast" | "standard" | "careful"` overrides the automatic estimate for one
  request.
- Requests containing an image get a 422 rather than a confident answer about an image no
  tier can see.

Other endpoints: `GET /healthz`, `GET /route/debug` (route a prompt without generating),
`GET /route/history`, `GET /retrieval/status`, `GET /tools/status`.

### Web console

<http://127.0.0.1:8000> serves a chat UI for watching the router work: which model
answered, which stage decided, why, and how long it took, inline on every reply — plus a
table of every request any client made to the server, not just the browser tab. "Preview
route only" checks the routing without paying for a generation.

## What's inside

A request falls through the cheapest thing that can handle it:

1. **Routing** — three stages, cheapest first: regex rules, then embedding similarity
   against per-route centroids, then the 350M as a classifier if the margin is thin. Plus
   a sticky stage so that answering "that's wrong" doesn't demote the thread to a smaller
   model than the one that got it right.
2. **Guardrails** — questions with an exact answer (arithmetic, percentages, leap years,
   timezones, unit conversions) are *computed*, and the result is injected before
   generation so the model phrases it rather than derives it. When a model contradicts a
   computed number anyway, the number wins.
3. **Effort** — how much to spend, decided before answering, from signals routing already
   produced. Free, so it can be wrong without costing anything.
4. **Retrieval** — gated, not always-on, because a merely-plausible passage makes answers
   worse rather than better.
5. **Tools** — `basics`, `web`, `notes`, behind a gate, with the tool *picked* by a
   different model than the one that answers.
6. **Adjudication** — a free capitulation guard always; a paid cross-model critic only if
   you turn it on.

Full detail, with the measurement behind each decision, in
**[docs/architecture.md](docs/architecture.md)**.

## The finding that shaped everything

Each model was asked to judge 16 question/answer pairs — 8 right, 8 wrong:

| Critic | Accuracy | Rubber-stamped a wrong answer | Finished | Cost |
|---|---|---|---|---|
| LFM2.5-350M | 50% (chance) | **8/8** | 16/16 | 0.6 s |
| Qwen3.5-0.8B, thinking off | 56% | 7/8 | 16/16 | 1.1 s |
| Qwen3.5-0.8B, thinking on | 100% | 0/8 | 6/16 | 28.3 s |
| LFM2.5-1.2B-Thinking | **100%** | **0/8** | 14/16 | 28.1 s |

The 350M said `CORRECT` to all sixteen, under two different prompts, including one that
ordered it to work the answer out first. That is not poor accuracy — it is *no
discriminative power at all*, and it means checking cannot be delegated downward. Only one
model here can tell a right answer from a wrong one, verification costs ~28 s, and
self-verification is worthless because a model's critic is only as good as its generator.

Nearly every design decision in this repository falls out of those three facts.

**[docs/measurements.md](docs/measurements.md)** has the rest — including why a latency
number measured here is worth less than you think, and the limitations that routing cannot
fix.

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | How a request is routed, grounded, answered and checked |
| [docs/measurements.md](docs/measurements.md) | The numbers behind the design, and the known limitations |
| [docs/configuration.md](docs/configuration.md) | Every `LEGEND_` setting, and what it costs |
| [docs/evaluation.md](docs/evaluation.md) | Running the tests and the eval suite |
| [ROADMAP.md](ROADMAP.md) | What is planned, and what was ruled out with reasons |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, conventions, and how to report a bug usefully |

## Development

```bash
uv sync
uv run pytest          # 478 tests, ~6s, no Ollama needed
uv run ruff check .
```

The router cascade tests run against a stub backend — no model loads, no network. See
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request; the one convention that
matters here is that **claims carry their measurements**.

## Status

Working and in daily use, but this is a research project on a laptop, not a hosted
service. Routing, guardrails, effort, retrieval, tools and adjudication are built; what
remains is mostly tuning defaults that are each a switch with a measurement behind them.
[ROADMAP.md](ROADMAP.md) has the detail.

**It has no authentication and is designed for loopback.** Every published port binds to
`127.0.0.1` on purpose — see [SECURITY.md](SECURITY.md) before exposing it anywhere.

## Licence

[MIT](LICENSE).
