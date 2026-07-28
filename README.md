# Legend Mode

A local model router: a tiny always-on model classifies each request, and the right
specialist is hot-swapped in via Ollama only when the request actually needs it. Runs
entirely on CPU — no GPU required.

## Why

Running one big model for every request wastes RAM and time when most messages are
"hi" or "thanks". This router keeps a few small models pinned in memory and swaps a
heavier one in only for the requests that need it, evicting it when idle.

## Model tiers

Currently running an accuracy/efficiency test with three answering models (down from
five): the Qwen3.5-2B tool/vision tier is parked, and the dedicated 230M router was
dropped too, after it turned out to be a measurably worse stage-3 classifier than the
350M model — it misrouted a reasoning question straight into a bad trivial-tier
answer. `general` (350M) now fills the router role.

| Tier | Model | Residency | Role |
|---|---|---|---|
| general | LFM2.5-350M | pinned | trivial replies + stage-3 classifier + everyday chat |
| embed | bge-small-en-v1.5 | pinned | routing embeddings only, never answers |
| small | Qwen3.5-0.8B | swapped | lighter reasoning; also the only tools/vision-capable tier right now |
| think | LFM2.5-1.2B-Thinking | swapped | heavier, explicit step-by-step reasoning |

Which alias fills the router role is configurable — `Settings.router_alias` in
`app/config.py` (default `"general"`) — rather than hardcoded, so this can change
again without touching engine code, just `models.yaml`/`routes.yaml` and one setting.

See `models.yaml` for exact GGUF sources and `routes.yaml` for the routing table and
labeled examples used by the embedding stage. The parked `large` (Qwen3.5-2B, and a
commented Qwen3.5-4B alternative) tier is still defined there — uncomment it and
re-point the `tools`/`vision` routes at it to bring back dedicated tool-calling. The
230M weights stay cached locally (HF cache untouched) but its Ollama tag was removed;
re-add a `router`-style entry to `models.yaml` and re-run the import script to bring
it back.

## Setup

1. Make sure [Ollama](https://ollama.com) is installed and running.
2. Install dependencies:
   ```
   uv sync
   ```
3. Import the models into Ollama (resolves local GGUFs, downloads anything missing):
   ```
   uv run python scripts/import_models.py
   ```
   The `general` (LFM2.5-350M) tier is not an Unsloth build (Unsloth has no 350M in
   the LFM2.5 line) — it comes from `LiquidAI/LFM2.5-350M-GGUF`, and on this machine
   it was originally pulled straight through `ollama pull` and tagged in with
   `ollama cp` rather than downloaded as a raw GGUF. The registry entry still lists a
   normal repo/file so a from-scratch machine can resolve and import it the usual way.
4. Start the server:
   ```
   uv run uvicorn app.main:app --port 8000
   ```
5. Open `http://localhost:8000` for the chat console, or use the API directly (below).

## Web console

`http://localhost:8000` serves a small chat UI (`app/static/index.html`) for watching
the router work in real time: send a message and see which model answered, which
routing stage decided, why, and how long it took — inline on each reply, plus a
"Recent requests" table that shows every call any client made to this server (curl,
scripts, the UI itself), not just this browser tab. "Preview route only" checks
`/route/debug` without spending a generation. Full conversation history is sent with
every turn, so follow-ups have real context even though routing itself only looks at
the latest message (see below).

## Usage

Point any OpenAI-compatible client at `http://localhost:8000/v1`:

```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hi"}]}'
```

- `model: "auto"` (or omitted) lets the cascade choose.
- `model: "small"` (or `general` / `think`) pins a specific tier.
- The response includes `x_legend_route` (which model, which stage, why) and the same
  info is echoed in the `X-Legend-Route` response header.
- Pass an OpenAI `tools` array as usual — it's forwarded to the `small` tier (the only
  one currently marked `tools: true`) and `tool_calls` come back untouched. The router
  does not execute tools itself.
- Every request is capped at a per-model token budget by default (`max_tokens`
  overrides it — see `default_max_tokens` in `models.yaml`). The `think` tier gets a
  larger budget (1536) since it visibly reasons before answering; everything else
  gets 512. Every Qwen3.5 GGUF tried so far defaults to thinking mode ON and will
  burn its whole budget on a `<think>` block and return empty content if that isn't
  turned off — `models.yaml` sets `default_think: false` on every Qwen3.5 tier.

Inspect a routing decision without generating anything:

```
curl "http://localhost:8000/route/debug?prompt=prove+sqrt+2+is+irrational"
```

## How routing works

Three stages, cheapest first, first confident answer wins:

1. **Rules** (`app/router/rules.py`) — regex/heuristic signals: a `tools` array or
   image content forces the matching tier; explicit reasoning language, code fences,
   counting/arithmetic word problems, or long prompts route to `think`; opening-turn
   greetings and self-identity questions ("who are you?") route to `trivial`. The
   identity rule exists because Qwen3.5 claims to be Qwen no matter what the system
   prompt says — see Persona below.
2. **Embeddings** (`app/router/embed.py`) — bge-small embeds the prompt and compares
   it to per-route centroids built from `routes.yaml`'s labeled examples at startup.
3. **Classifier** (`app/router/classifier.py`) — only if the embedding margin is too
   thin, whichever model fills the router role (`Settings.router_alias`) is asked to
   name a category directly.

Decisions are cached by prompt hash so repeats skip straight past all three stages —
note this caches the *routing decision*, not the generated answer; the model still
generates fresh output every time.

Routing only looks at the **last** message, not the full thread — cheap and fast on
purpose. Generation always gets the full conversation regardless, so whichever model
gets picked still has complete context; the tradeoff is a short ambiguous follow-up
(e.g. "so what's the answer?") can get classified into a lighter tier than the thread
really warrants, even though that tier still answers coherently since it sees
everything.

## Persona

Every request gets a shared system prompt (`app/persona.py`) prepended automatically
unless the caller already supplies their own `system` message. No name is hardcoded
yet — `LEGEND_ASSISTANT_NAME` is unset.

There are two lengths, picked per tier via `persona:` in `models.yaml`. The 350M gets
`brief` (~170 chars); everything else gets `full` (~350). All of the wording below was
settled by measuring against the real models, and re-wording it without re-measuring
is a good way to reintroduce a bug that unit tests cannot see:

- **The prompt must never end on a quotable answer.** A version ending
  `If the user asks your name, say you don't have one yet.` made the 350M reply
  "you don't have one yet." to *hi*, *thanks!*, and *what is the capital of France* —
  it treats the last sentence as a completion prefix. Moving the identity clause into
  the middle and closing on a behavioral directive took it from failing most neutral
  prompts to 0 problems in 21 samples.
- **Length is not free at 350M.** The full prompt measurably crowds out the question.
- **Naming model brands primes them.** Adding "never identify yourself as Qwen" made
  the Qwen tier's identity leak *worse*. The full prompt disclaims brands as a
  category instead of listing them.

Known limitation, only partly solved: models assert their training identity regardless
of the prompt. Qwen3.5-0.8B says "I am Qwen3.5, developed by Tongyi Lab" 6/6 across
three different wordings, so self-identity questions are pinned by a routing rule to
the 350M rather than fought in the prompt. The 350M is better but not clean — asked
"who are you?" it named Liquid AI in about 1 of 4 samples, occasionally with an
invented second maker. Three brief wordings were compared; the one shipped scored best
(5/28 vs 7/28). Treat this as the floor reachable by prompting.

Separately, the small tiers still conflate "my name is Krish" with a question about
their own name unless the prompt separates the two explicitly, which it now does —
verified working, but recalling a fact from several turns back remains a genuine
capability limit at this size, not a wording problem.

Set a name once it's picked and it propagates everywhere with no prompt-editing:

```
LEGEND_ASSISTANT_NAME=Whatever uv run uvicorn app.main:app --port 8000
```

## Configuration

Environment variables (prefix `LEGEND_`), see `app/config.py`:

- `LEGEND_OLLAMA_HOST` — default `http://127.0.0.1:11434`
- `LEGEND_MAX_LOADED_MODELS` — default `2` (pinned tier + one swapped model)
- `LEGEND_DEFAULT_ROUTE` — used when every stage is unsure, default `chat`
- `LEGEND_DISABLE_CLASSIFIER` — skip stage 3, go straight to the default route
- `LEGEND_ROUTER_ALIAS` — which `models.yaml` alias fills the router role (trivial
  replies + stage-3 classifier), default `general`
- `LEGEND_ASSISTANT_NAME` — unset until a name is picked; see Persona above

## Tests

```
uv run pytest
```

Router cascade tests run against a stub backend — no model loads, no Ollama needed.

## Not built yet

- **Tool execution.** The API forwards `tools` and returns `tool_calls` as-is; there's
  no server-side agent loop that dispatches them. `app/router/engine.py` documents the
  extension point.
- Dedicated tool/vision tier. Parked with the Qwen3.5-2B tier above; `small`
  (Qwen3.5-0.8B) covers both in the meantime.
- Routing that reads more than the last message (e.g. a "sticky tier" for an ongoing
  reasoning thread, or feeding recent turns into the classifier).
