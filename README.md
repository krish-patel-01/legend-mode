# Legend Mode

A local model router: a tiny always-on model classifies each request, and the right
specialist is hot-swapped in via Ollama only when the request actually needs it. Runs
entirely on CPU — no GPU required.

## Why

Running one big model for every request wastes RAM and time when most messages are
"hi" or "thanks". This router keeps a few small models pinned in memory and swaps a
heavier one in only for the requests that need it, evicting it when idle.

## Model tiers

Two answering models, deliberately. Everything else is parked.

| Tier | Model | Residency | Role |
|---|---|---|---|
| general | LFM2.5-350M | pinned | trivial replies + stage-3 classifier + everyday chat |
| embed | bge-small-en-v1.5 | pinned | routing embeddings only, never answers |
| think | LFM2.5-1.2B-Thinking | swapped | reasoning, and the only tier that can verify an answer |

### Why only these two

Each model was asked to judge 16 question/answer pairs — 8 right, 8 wrong — and say
whether the answer was correct. Measured on this machine:

| Critic | Accuracy | Rubber-stamped a wrong answer | Finished | Cost |
|---|---|---|---|---|
| LFM2.5-350M | 50% (chance) | 8/8 | 16/16 | 0.6 s |
| Qwen3.5-0.8B, thinking off | 56% | 7/8 | 16/16 | 1.1 s |
| Qwen3.5-0.8B, thinking on | 100% | 0/8 | **6/16** | 28.3 s |
| LFM2.5-1.2B-Thinking | 100% | 0/8 | 14/16 | 28.1 s |

The 350M said `CORRECT` to all sixteen, under two different prompts, including one that
ordered it to work the answer out first — it has no discriminative power at all, not
merely poor accuracy. That is the same behaviour as agreeing "you're right, it's 2, not
3" when told an answer is wrong.

So checking cannot be delegated downward: the 1.2B is the only model here that can tell
a right answer from a wrong one. That leaves the second model with nothing but cheap,
latency-critical work — trivial replies and stage-3 classification — and for that the
350M beats the 0.8B outright: 71.8 tok/s vs 39, 229 MB vs 737 MB. (An earlier 230M was
tried in that role and was measurably worse than the 350M, misrouting a reasoning
question straight into a bad trivial-tier answer.)

Which alias fills the router role is configurable — `Settings.router_alias` in
`app/config.py` (default `"general"`) — rather than hardcoded, so this can change
again without touching engine code, just `models.yaml`/`routes.yaml` and one setting.

**Parked, not deleted.** Qwen3.5-0.8B (`small`) is commented out in `models.yaml` but
still tagged in Ollama and still on disk, as are Qwen3.5-2B/4B and the 230M. Uncomment
an entry and re-run `scripts/import_models.py` to bring one back — anything restored
also needs its route re-added to `routes.yaml`, which no longer defines `tools` or
`vision`. `legend/large` was removed from Ollama to reclaim 1.9 GB; its GGUF is still in
the Hugging Face cache, so the import script can recreate it.

### No image support

No tier is vision-capable now that `small` is parked, so `/v1/chat/completions` rejects
any request containing an image with a 422 rather than handing it to a text-only model
that would answer confidently about an image it never saw.

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
- `model: "general"` (or `think`) pins a specific tier.
- The response includes `x_legend_route` (which model, which stage, why, and which
  guardrail fired) and the same info is echoed in the `X-Legend-Route` response header.
- A `tools` array is still accepted and forwarded, but nothing is marked `tools: true`
  any more, so no tier will emit `tool_calls` and you get an ordinary text reply. The
  router has never executed tools itself.
- Requests containing an image get a 422 — see "No image support" above.
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

1. **Rules** (`app/router/rules.py`) — regex/heuristic signals: explicit reasoning
   language, code fences, counting/arithmetic word problems, or long prompts route to
   `think`; opening-turn greetings and self-identity questions ("who are you?") route
   to `trivial`. The identity rule exists because Qwen3.5 claims to be Qwen no matter
   what the system prompt says — see Persona below. It is kept even though that tier is
   parked, since restoring `small` would otherwise silently reintroduce the leak.
2. **Embeddings** (`app/router/embed.py`) — bge-small embeds the prompt and compares
   it to per-route centroids built from `routes.yaml`'s labeled examples at startup.
3. **Classifier** (`app/router/classifier.py`) — only if the embedding margin is too
   thin, whichever model fills the router role (`Settings.router_alias`) is asked to
   name a category directly.

Decisions are cached by prompt hash so repeats skip straight past all three stages —
note this caches the *routing decision*, not the generated answer; the model still
generates fresh output every time.

Between rules and embeddings sits a **sticky** stage (`RouterEngine._sticky`) for short
follow-ups. Routing otherwise reads only the latest message, which broke threads: the
box word problem routed to `think`, which answered "4" correctly, and then "its
incorrect" routed on its own as a three-word message to the 350M, which agreed and
invented "3".

- An explicit dispute ("that's wrong", "are you sure") **escalates** to `think`
  regardless of history — whatever answered last was evidently not good enough.
- A bare "no"/"nope", or a continuation ("why", "explain that", "go on"), sticks to the
  tier already handling the thread, but only if that tier is in `sticky_routes`.
- Pleasantries and fresh questions are untouched. Continuation patterns are anchored at
  both ends so "explain that" sticks while "explain what a REST API is" routes normally.

Which tier is handling the thread is recovered by re-routing the last user turn that is
**not itself a follow-up** (`anchor_text`), rather than by storing conversation state.
Routing stays a pure function of the messages, so the API remains stateless. The
walk-back matters: in a run of "its incorrect" / "nope" / "nope", re-routing merely the
previous turn finds another "nope", which lands on the trivial tier in isolation — the
exact demotion the stage exists to prevent.

Generation always gets the full conversation regardless, so whichever model is picked
has complete context.

## Guardrails

Some questions have an exact answer that code can compute, and on those the models were
simply unreliable — the 350M answered "how many days in a leap year" with **365** in one
sample and 366 in another, and called IST "Eastern Standard Time" in one run and
"Central European Time" in the next. No amount of prompting fixes a model that doesn't
know, and having a second model check costs ~28 s (see the critic table above), against
under a millisecond for a real implementation.

`app/guardrails.py` recognises those questions and computes the answer directly:
arithmetic (via a restricted `ast` walk, never `eval`), percentages and discounts, leap
years, timezone abbreviations, and common unit conversions.

The mechanism is **grounding, not correction**. When a guard fires, the computed fact is
appended to the system prompt *before* generation, so the model phrases it naturally:

```
Verified fact, computed exactly and known to be correct: A leap year has exactly
366 days (a common year has 365). Use this value in your answer and do not recalculate it.
```

Patching the answer afterwards would mean parsing prose, which is its own source of
errors. The note ends on a directive rather than on the answer itself, for the reason
documented under Persona below.

Guards are deliberately conservative — a guard that fires on a question it has misread
injects a confident wrong fact, which is worse than no guard. `what is 17 * 23 in roman
numerals` and `how many boxes do I have if I have two boxes with one box inside each?`
both decline to ground and go to the model unchanged. `contradicts()` is advisory: it
logs when a model restates a supplied fact wrongly, and never rewrites the reply.

Which guard fired shows up as `grounded` in `x_legend_route`. Measured end to end, two
samples each: leap year, IST, `17 * 23`, `15% of 240`, a 25%-off discount, and a km→miles
conversion all now answer correctly, where several were wrong before.

Note what this does *not* buy: grounding fixes the number, not the explanation. One
sample answered "$30" correctly while its supporting prose said "the discount reduces
the price by 10%".

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

Separately, the 350M still conflates "my name is Krish" with a question about its
own name unless the prompt separates the two explicitly, which it now does —
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

## Known limitations

**Capitulation under repeated pushback is reduced, not solved.** Sticky routing keeps the
whole thread on `think` — verified over the original transcript, where all four turns now
stay on the reasoning tier and none contradicts the correct answer. `DISPUTE_NOTE` in
`app/persona.py` tells the model not to simply agree it was wrong.

Measured over 8 dispute turns (the box problem, answered correctly, then "its
incorrect"): **5 held the answer, 1 caved, 2 were vague.** One vague reply was the
empty-budget fallback; the other defended the answer without restating the number, so the
scorer — which required the digit — under-counts holds. Against the original transcript,
where it caved on the first push every time, that is a large improvement and still not a
guarantee. The remaining cave was "the count depends on interpretation; it's 3". Holding a
correct answer against a user who insists otherwise is a model-capability limit, and no
routing change reaches it.

Two smaller things the dispute path exposed, both handled:

- Terse disputes gave the 1.2B nothing concrete to re-check, and it reasoned in circles
  until the token budget ran out and returned **empty content**. `app/api.py` now
  substitutes an honest message, and `DISPUTE_NOTE` tells the model to ask which part is
  disputed when the user hasn't said.
- An earlier `DISPUTE_NOTE` ended with "Answer in no more than three sentences." and the
  1.2B replied "The total is four. Three sentences: Four boxes." — the same
  trailing-instruction echo documented under Persona. The length cue now sits mid-note.

## Not built yet

See [ROADMAP.md](ROADMAP.md) for the full plan, the measured constraints behind it, and
why RL post-training isn't viable on this hardware. In short:

- **An eval set** — next, and a prerequisite for the rest. Every number in this README came
  from a throwaway script; improvements can't be told from noise without fixed cases.
- **Adjudication with an effort controller.** The 1.2B verifies at ~100% on the measured
  set but costs ~28 s, so it needs a difficulty estimate made *before* answering. Per-effort
  token budgets would also fix the ~1-in-6 dispute turns that currently exhaust the budget.
- **Retrieval**, for the facts neither model knows — gated, not always-on.
- Tool execution, vision, and RL on weights: all deferred, with reasons in the roadmap.
- **Tool execution.** The API forwards `tools` and returns `tool_calls` as-is; there's
  no server-side agent loop that dispatches them, and no tier currently advertises tool
  support at all. `app/router/engine.py` documents the extension point.
- Vision. Parked with the Qwen3.5-0.8B tier; image requests are rejected with a 422.
