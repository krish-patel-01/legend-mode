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

1. Make sure [Ollama](https://ollama.com) is installed and running. Optionally give it a
   slot per tier — `OLLAMA_MAX_LOADED_MODELS=4 ollama serve` — which keeps the embedder
   and all three answering models resident. Measured, it makes no difference to speed
   (see "A latency number here is worth less than you think" below); it just avoids
   reloads if a session uses every tier.
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
- `effort: "fast" | "standard" | "careful"` overrides the automatic estimate for one
  request; `"auto"` or omitted lets the controller decide. See "Effort" below.
- Token budgets come from the effort plan, not straight from the model. Each tier's
  `default_max_tokens` in `models.yaml` is the ceiling the plan works within (1536 on
  `think`, 512 elsewhere), and a caller's explicit `max_tokens` overrides both. Every
  Qwen3.5 GGUF tried so far defaults to thinking mode ON and will burn its whole budget
  on a `<think>` block and return empty content if that isn't turned off —
  `models.yaml` sets `default_think: false` on every Qwen3.5 tier.

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
- A **correction** ("but the monkeys are on the bed", "you forgot the bed") also
  escalates, and is labelled separately because it needs the opposite instruction. A
  dispute carries no information, so `DISPUTE_NOTE` tells the model not to simply cave; a
  correction hands over a fact the answer missed, so `CORRECTION_NOTE` tells it to take
  that as true and re-work from the start. Folding the two together would make the model
  stubborn at exactly the wrong moment. Only counts from the third message onward —
  opening with "but" is an ordinary question.
- A bare "no"/"nope", or a continuation ("why", "explain that", "go on", "what?"), sticks
  to the tier already handling the thread, but only if that tier is in `sticky_routes`.
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
both decline to ground and go to the model unchanged.

**Injection alone is not enough, and this took an eval suite to notice.** A two-sample
check said the guarded cases all answered correctly. At six samples the 350M was
overriding the supplied value on 5 of 6 discount questions ("You pay $20" against a
grounded 30) and 5 of 6 temperature conversions ("100 ÷ 32 = 3.125" against 212). So
`contradicts()` no longer just logs: when a **numeric** grounding is contradicted, the
reply is replaced with the guard's own sentence. That is sound precisely because the
comparison is numeric and exact — a provably wrong number swapped for a provably right
one, never prose judging prose.

The correction rate is a direct measure of how often the answering tier ignores a value it
was handed, and it is reported by the eval runner every run. It moved with the model:

| chat tier | grounded samples needing correction |
|---|---|
| LFM2.5-350M | **23%** |
| LFM2.5-1.2B-Instruct (Q4) | 8% |
| LFM2.5-1.2B-Instruct (UD-Q3) | **0%** |

`computable` passes 100% throughout. The substitution stays regardless — a rate that is
zero on one model and a quarter on another is exactly the kind of thing that should not be
load-bearing — but it is now catching almost nothing, which is the point.

Which guard fired shows up as `grounded` in `x_legend_route`, suffixed `(corrected)` when
the substitution ran.

Note what this does *not* buy: grounding fixes the number, not the reasoning around it.
One sample answered "$30" correctly while its supporting prose said "the discount reduces
the price by 10%".

## Effort

`app/effort.py` decides how much a request is worth spending **before** answering it,
from signals the cascade already produced — which rule fired, which stage decided,
whether a guardrail grounded the question, whether this is a follow-up. No extra model
call, so the estimate is free and can be wrong without costing anything.

| Level | Budget | What it means |
|---|---|---|
| `fast` | 256 | greetings, identity questions, anything a guardrail already computed |
| `standard` | the tier's own ceiling | ordinary chat and reasoning |
| `careful` | tuned per follow-up kind | disputes, corrections, prompts nothing recognised |

**A token cap is not a brevity control on a reasoning model**, and finding that out cost a
regression worth recording.

The budgets were reasoned from the reply: a dispute answer is two or three sentences, so
it does not need 1536 tokens. That is true of the reply and irrelevant to the mechanism.
A thinking model emits its `<think>` block *first* and the answer only after it, so a
budget below what the reasoning needs does not produce a short answer — it produces **no
answer at all**, and the request falls through to the exhaustion message.

So the first version made the bug it was written to fix strictly worse. The original
complaint was that roughly 1 dispute turn in 6 returned empty content; at 384 tokens
essentially every dispute turn did, six consecutive `think produced no content in 384
tokens` warnings in a single eval run. A grounded question that happened to route to
`think` was starved the same way, and answered "I couldn't reach an answer I'd trust"
while holding a computed 36.

Measured floor, from the critic probe: at 256 tokens the 1.2B emits nothing at all, at
512 it reaches an answer 2 times in 8, at 1024 6 times, at 2048 always. So on a thinking
tier the plan does not shrink the budget — the tier default is the floor. Follow-up
budgets of 384 / 768 / 1024 still apply on tiers that answer without a reasoning block,
where they are safe. Brevity on the reasoning tier has to come from the prompt, which is
what `DISPUTE_NOTE` does, or not at all.

**The eval suite did not catch any of it, which is the more useful finding.** The
exhaustion reply is a well-formed sentence, so it satisfies `regex: \w`, and it carries
no forbidden number, so `not_number` passes too. Every dispute case scored 100% while
every dispute turn was failing; the evidence was only in the server log. The runner now
counts that reply across the whole run and says so loudly.

Uncertainty is read off the routing **stage**, not off `confidence`. The confidences are
not on a common scale — the embedding stage reports a raw cosine, the classifier a flat
constant, the fallback 0.3 — so one threshold across all three would mostly measure which
stage answered. `fallback` and `classifier` mean nothing recognised the prompt; that is
the honest uncertainty signal, and `RouteDecision.origin` carries it through the routing
cache so a cached guess doesn't read as a confident hit.

A `trivial` verdict is only trusted from the **rules** stage, which matches greetings by
pattern. The classifier saying "trivial" is a guess, and it mis-filed a real question
during testing: *"which model verifies answers in this system?"* came back from the 350M
as *"The question itself is the answer."*

The chosen level, budget and reason appear as `effort` in `x_legend_route` and on
`/route/debug`.

## Adjudication

`app/adjudicate.py` checks an answer after it exists. Three mechanisms, in descending
order of how often they can run.

**The capitulation guard** is free — a numeric comparison, no generation. On a bare
denial, if the reply's operative number differs from the previous reply's, the model
changed its answer under pressure that carried no information. That earns exactly one
re-work with a note saying so; if the re-work lands on a third number, the reply says so
and names both candidates instead of picking one. Number extraction is deliberately
conservative and returns nothing when the reply is ambiguous — a guard that fires on a
misread is worse than no guard.

**Cross-model verification** is the 1.2B judging a smaller model's answer. Note the
constraint carefully: the verifier must always be the 1.2B *and* never the model that
wrote the answer, and with two models those two rules intersect at exactly one case. An
answer from the reasoning tier has **no independent critic on this hardware**, and the
response metadata says so (`adjudicated.skipped`) rather than quietly falling back to
self-verification.

**It is off by default, and that is a measurement, not a default nobody revisited.** The
critic was measured over 8 question/answer pairs, 4 right and 4 wrong:

| Critic budget | Accuracy | Verdict emitted | Wrong answers waved through | Median |
|---|---|---|---|---|
| 512 tokens | 25% | **2/8** | 0/4 | 45.1 s |
| 1024 tokens | 75% | 6/8 | 0/4 | 24.9 s |
| 2048 tokens | **88%** | 8/8 | 1/4 | 26.7 s |

A budget that looks generous can sit below the floor: at 512 tokens the critic spends
everything inside `<think>`, returns "unsure" on 6 of 8, and charges 45 seconds for it —
which from the outside is indistinguishable from a working verifier that never fires.

**Two rows have been struck from this table, and why is worth keeping.** The same probe
also ran the critic with reasoning "off" at 64 and 192 tokens, where it waved through 4
of 4 wrong answers, and that was written up here as *"turning the reasoning block off
buys the 350M's behaviour"*. It is not supported. Ollama advertises a `thinking`
capability for this model, but **`think: false` does not actually suppress the block** —
verified directly, `<think>` is emitted either way. Those rows measured a *truncated*
reasoning block rather than a disabled one, and the verdict parser's fallback then
scraped a verdict word out of the reasoning itself.

That is the exact failure `app/adjudicate.py` is written to avoid, reappearing in the
measurement harness instead of the parser. Reading a `<think>` block as a verdict has now
produced a wrong conclusion twice here.

At 2048 tokens it works, at 26.7 s median. But **answering the question on the 1.2B costs
about the same ~25 s and produces a better answer rather than a grade on a worse one**, and
in the failure case verification costs double because "incorrect" still has to be followed
by a regeneration. On this hardware escalating dominates verifying, so `verify_enabled`
defaults to false. The machinery stays because that reasoning is hardware-specific: a
second model that could judge in 2 s would flip it immediately. Turn it on with
`LEGEND_VERIFY_ENABLED=true` or per request with `{"effort": "careful"}`; the eval runner
reports how many samples paid for it and how many it changed.

**Self-consistency** — answering twice and comparing the two numbers exactly — is
implemented and off by default (`LEGEND_SELF_CONSISTENCY`). It is not self-verification:
no model judges anything, two extracted numbers are compared. It doubles latency on the
slowest tier, so it is a knob for tuning rather than a default.

At most one repair, ever. Unbounded critique oscillates (A says 4, B says 2, A says 2…)
and small critics are exactly noisy enough to make that likely.

## Retrieval

The last error class the other layers cannot touch: facts neither model holds, where more
thinking cannot help because the answer was never in the weights.

```
uv run python scripts/ingest.py                     # ingest README.md + ROADMAP.md
uv run python scripts/ingest.py notes/ handbook.md  # ingest your own documents
uv run python scripts/ingest.py --list              # what is indexed
uv run python scripts/ingest.py --probe "…"         # score a question, to set the threshold
```

Storage is sqlite plus numpy (`data/corpus.db`, gitignored and rebuildable) — no vector
database, because brute-force cosine over a few thousand chunks costs well under a
millisecond, which is invisible next to a 1.2B's 25 s. Embeddings come from the bge model
already pinned for routing. The index records which embedder built it and refuses a
mismatch, since searching with the wrong model doesn't fail, it returns confident nonsense.

**Gated, not always-on.** In the paper this design follows, indiscriminate retrieval *hurt*
GPQA by 5.0 points, because a passage that is merely present overrides knowledge the model
already had right. Two gates stand in front: a cheap syntactic one in `app/effort.py` that
asks whether the prompt is a lookup at all (arithmetic, code, greetings and creative work
never qualify), and a similarity threshold that discards anything not actually about the
question. Calibrated on this corpus with `--probe`:

| question | top score | injected? |
|---|---|---|
| "which model verifies answers in this system" | 0.759 | yes |
| "how do I run the evals" | 0.739 | yes |
| "what is the capital of Australia" | 0.512 | no |

bge-small puts unrelated English around 0.5–0.6, so the default cut-off is 0.66 — well
above the naive midpoint. Re-probe after changing the corpus rather than trusting that
number.

**A retrieval hit escalates off the 350M.** The roadmap's premise is that a grounded small
model beats an ungrounded large one, and that assumes the small model can read. Handed the
chunk saying *"the verifier is always the 1.2B and never the 350M"*, the 350M answered
*"The model that verifies answers is LFM2.5-350M"* — it inverted the source.

It escalates to whatever `chat` runs on, not to the reasoning tier. That was the original
rule and it stopped being right the moment `chat` itself became a 1.2B:

| tier | same memory-backed question | |
|---|---|---|
| chat (1.2B instruct) | "Krish. Legend Mode." | correct, ~2–4 s |
| reasoning | "Your name is Krish. You work on…" | correct, 11–25 s |
| 350M | "My name is Krish. I work on…" | wrong |

Both read it correctly, so the reasoning tier was charging 25 s for nothing — and on a
formatting request (*"make my name bold"*) it spent 51.7 s and returned no content at all.
Reading a passage is not a reasoning task.

**Citations are computed, never requested.** The first version labelled each passage
`[source#heading]` and asked the model to copy that; the 1.2B replied with only the
bracketed citation and no answer, having matched the most recent pattern in the prompt.
The sources are known exactly at that point, so `app/api.py` appends a `Sources:` line
itself — the same rule `app/guardrails.py` follows.

On a follow-up, retrieval runs against the thread's **anchor** turn, not against "explain
that". Corpus text is injected per request rather than kept in the conversation, so
without that a grounded thread would lose its source material on turn two.

Retrieval activity shows up as `retrieved` in `x_legend_route`; `GET /retrieval/status`
reports what is indexed.

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
- `LEGEND_DEFAULT_EFFORT` — `auto` (default), or pin every request to one level
- `LEGEND_VERIFY_ENABLED` — cross-model critic, default **false**; see Adjudication
- `LEGEND_SELF_CONSISTENCY` — answer twice and compare, default false
- `LEGEND_CRITIC_ALIAS` — which tier judges, default `think`
- `LEGEND_RETRIEVAL_ENABLED` — default true (a missing corpus is simply inert)
- `LEGEND_RETRIEVAL_DB` — index location, default `data/corpus.db`
- `LEGEND_RETRIEVAL_MIN_SCORE` — cosine cut-off, default `0.66`; calibrate with `--probe`
- `LEGEND_RETRIEVAL_TOP_K` — chunks injected, default 3
- `LEGEND_RETRIEVAL_CITE` — append the computed `Sources:` line, default true
- `LEGEND_READER_ROUTE` — which *route's* model reads retrieved text, default `chat`, so
  it tracks whatever that tier runs on rather than pinning an alias

## Tests

```
uv run pytest
```

Router cascade tests run against a stub backend — no model loads, no Ollama needed.

## Evals

Unit tests prove the routing logic; they say nothing about whether the answers are
right. That needs real generation, so it lives separately in `evals/cases.yaml` and
`scripts/eval.py`, run against a server that's already up:

```
uv run python scripts/eval.py                     # everything
uv run python scripts/eval.py --category dispute  # one category
uv run python scripts/eval.py --routes-only       # routing only, ~1 ms a case
uv run python scripts/eval.py --save-baseline     # accept the current numbers
```

Cases are grouped as `computable`, `reasoning`, `dispute`, `factual`, `persona`, `cheap`,
`effort` and `retrieval`. `cheap` asserts that lookups are *not* promoted to the 1.2B,
which protects latency — the thing that otherwise rots silently. Every case came from
something observed in real use.

Alongside the pass rates, a run reports the numbers the tuning phase needs: the spread of
effort levels, how many samples paid for adjudication and how many it changed, how many
had no available critic, and how often retrieval injected corpus text. Budget discipline
is a number, not a feeling — an accuracy win that triples median latency is not a win.

Two properties worth preserving if you extend it:

- **Checks are declarative** — substrings, numbers, regexes, route names. There is no
  model-judges-the-answer step, because a model judging output is the thing this project
  measured as unreliable (the 350M rubber-stamped 8 of 8 wrong answers).
- **Cases are sampled, not run once.** A case scores as the fraction of samples that
  passed, so flakiness reads as 50% rather than as a coin-flip pass. This is not
  theoretical: a 2-sample check reported the discount and temperature guards as working,
  and at 6 samples they were passing 1 in 6.

`evals/baseline.json` holds the last accepted run; a normal run diffs against it and
exits non-zero on any regression.

## A latency number here is worth less than you think

**This laptop loses roughly 30% of its throughput under sustained load**, and that fact
invalidated three separate conclusions before it was found. Any timing comparison made by
running A and then B is really measuring which one ran second.

The controlled version — alternating chat and reasoning requests with Ollama's residency
cap at 3 and at 4, back to back, in both orders:

| order | first | second |
|---|---|---|
| 3 then 4 | cap 3 — **49.9 s** | cap 4 — 60.9 s |
| 4 then 3 | cap 4 — **52.5 s** | cap 3 — 71.6 s |

Whichever ran second lost, both times, by a similar margin. Free memory was 2.5 GB in
every run and total reload time never exceeded 1.7 s, so neither pressure nor eviction
explains it. Throughput also falls monotonically *within* a single run — 31.6 → 21.1 tok/s
across six requests.

What this means in practice:

- **The eval suite's latency line is not a benchmark.** A full run takes tens of minutes,
  so the later cases are measured on a hotter machine than the earlier ones. Successive
  full runs reported medians of 1.8 s, 11.9 s and 18.7 s with no change that could account
  for it. Pass rates are unaffected — those are content checks.
- **Compare configurations by interleaving or reversing order**, never by running one
  after the other and reading off the totals.
- **Isolated, cold measurements are the fair ones.** A warm chat request on the 1.2B
  instruct tier is 1.5–3 s; the 8.9 s the eval reported for the same case is throttling.

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

**Some questions are past both tiers, and routing cannot fix that.** Two puzzles from a
live session — a hidden-number word sequence (stONE, ofTEN, caNINE, frEIGHT) and a QWERTY
letter-shift substitution — used to reach the 350M through `fallback` and got instant
nonsense. They now route to `think` correctly, and `think` cannot solve them either:

| budget | word sequence | keyboard shift | box word problem |
|---|---|---|---|
| 1536 tokens | no answer, ~134 s | no answer, ~141 s | **"4." correct**, 658 tok |
| 4096 tokens | **"A"** — wrong, 342 s | **"S"** — wrong, 215 s | "4." correct, 927 tok |

More budget bought a *slower wrong answer*, so `default_max_tokens` stays at 1536. A
brevity instruction was measured too: ignored outright on the hard puzzles, though it cut
a solvable problem's tokens by 26% — a candidate for the effort controller, not a
prompt change to make on one sample. Both puzzles are in the eval suite as
`known_failing`, skipped by default and runnable with `--include-known-failing`.

When the reasoning tier exhausts its budget it now says it couldn't reach an answer it
trusts, rather than "I ran out of thinking room", which wrongly implied a retry would help.

A third case joins them: *"Five monkeys are jumping around on a four poster bed while
three chickens stand and watch. How many legs are on the floor?"* The answer is 10 — the
monkeys are on the bed, three standing chickens give 6, and a four-poster bed has 4 legs
of its own. The 1.2B answers 26 (5×4 + 3×2) and keeps answering 26 after being told
directly that the monkeys are on the bed.

That case also exposes a real tension with no clean resolution here. `DISPUTE_NOTE` makes
the model hold its ground, which is right when it is correct and being pushed around (the
box problem) and wrong when it is not (this riddle). Nothing in the system can tell those
apart, because nothing can verify a lateral riddle — which is the gap the effort
controller in `ROADMAP.md` is meant to address, and a reason not to keep patching prompts
at it.

Two smaller things the dispute path exposed, both handled:

- Terse disputes gave the 1.2B nothing concrete to re-check, and it reasoned in circles
  until the token budget ran out and returned **empty content**. `app/api.py` now
  substitutes an honest message, and `DISPUTE_NOTE` tells the model to ask which part is
  disputed when the user hasn't said.
- An earlier `DISPUTE_NOTE` ended with "Answer in no more than three sentences." and the
  1.2B replied "The total is four. Three sentences: Four boxes." — the same
  trailing-instruction echo documented under Persona. The length cue now sits mid-note.

## Not built yet

See [ROADMAP.md](ROADMAP.md) for the full plan and the measured constraints behind it.
Steps 2 (evals), 3 (effort and adjudication) and 4 (retrieval) are built; what remains:

- **Tuning.** Everything above ships at a default chosen from measurement, but several are
  single knobs with a switch attached — `verify_enabled`, `self_consistency`,
  `retrieval_min_score`, the follow-up budgets. The eval harness now reports the cost side
  of each, so these can be settled with data rather than argued about.
- **A general-knowledge corpus.** Retrieval works, but the corpus that ships is the
  project's own documentation. The `factual` cases still fail because nothing indexed
  knows when the Treaty of Westphalia was signed. That is a corpus to supply, not code to
  write.
- **A contextual bandit over the router** — five routes, reward from whether a guardrail
  passed and how long the answer took, trained on the decisions `/route/history` already
  logs. No GPU, ~50 lines of numpy. It needed the eval set first, which now exists.
- **Tool execution.** The API forwards `tools` and returns `tool_calls` as-is; there's
  no server-side agent loop that dispatches them, and no tier currently advertises tool
  support at all. `app/router/engine.py` documents the extension point.
- Vision. Parked with the Qwen3.5-0.8B tier; image requests are rejected with a 422.
- RL on model weights: deferred with reasons in the roadmap — no NVIDIA GPU, and TRL's
  GRPO path requires vLLM and CUDA.
