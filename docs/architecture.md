# Architecture

How a request travels through Legend Mode, and why each layer is shaped the way it is.
The measurements behind these choices are in [measurements.md](measurements.md); the
settings that control them are in [configuration.md](configuration.md).

The short version: a request is routed to the cheapest tier that can handle it, grounded
with anything that can be computed or looked up exactly, answered, and then checked by
whatever check is affordable. Each layer exists because the one before it was measured
to be insufficient.

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

bge-small puts unrelated English around 0.5–0.6, so the default cut-off is well above the
naive midpoint. Re-probe after changing the corpus rather than trusting that number.

**It is 0.70, raised from 0.66 after a live false positive.** Asked *"what is the current
gold price?"*, retrieval injected this project's own Guardrails section at 0.666 and the
answer came back citing it. Note where the false positives sat when the corpus was scored
afterwards — just *above* the old cut-off, not far below it:

| question | top score | wanted? |
|---|---|---|
| "which model verifies answers in this system" | 0.767 | yes |
| "how does the router decide which model" | 0.782 | yes |
| "how do I add a new route" → vault boilerplate | 0.688 | no |
| "what is the current gold price" → README | 0.666 | no |

A threshold calibrated on a handful of questions drifts as the corpus grows, and this one
had only ever been checked against questions the corpus could actually answer. Memories
carry their own, lower cut-off (0.55): they are single sentences, and short-to-short
similarity runs lower for the same relevance.

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

**A question whose answer is already stored exactly is answered from the store**, not by a
model — the same rule the guardrails follow for arithmetic. Asked to phrase it, the chat
tier answered *"what is my name?"* with **"My name is Krish."** on roughly half of samples,
reporting the user's identity as its own. Four prompt-side fixes were tried — a line about
pronouns, quoting each fact, rewriting facts to third person at capture, changing which
tier reads them — and the first three helped without settling it. Computing the answer
settles it, and takes it from ~3 s to **0.05 s**.

Deliberately narrow: only the name, only on unmistakable phrasings, only when a stored
fact matches the template exactly. *"where do I work?"* still goes to the model, because
forming that answer needs a verb this cannot conjugate. A wrong deterministic answer is
worse than a wrong generated one — it arrives with the authority of a computed fact.

On a follow-up, retrieval runs against the thread's **anchor** turn, not against "explain
that". Corpus text is injected per request rather than kept in the conversation, so
without that a grounded thread would lose its source material on turn two.

Retrieval activity shows up as `retrieved` in `x_legend_route`; `GET /retrieval/status`
reports what is indexed.

## Tools

Tools are grouped into **families** — `basics`, `web`, `notes` — because that is the
granularity everything else works at. `GET /tools/status` lists what exists and probes
whether each backend actually answers, since a tool whose service is down is otherwise
indistinguishable from one that simply never fires.

**The gate exists because attaching tools makes every model here worse.** Measured with
four tool definitions attached against six prompts that need no tool at all, each model
asked the same question twice — once with the tools and once without:

| model | picked correctly | spurious calls | degraded answers | median |
|---|---|---|---|---|
| LFM2.5-230M | 1/6 | 2/6 | 2/6 | 0.8 s |
| LFM2.5-350M | **6/6** | 1/6 | 2/6 | 1.0 s |
| LFM2.5-1.2B-Instruct | 3/6 | 2/6 | 3/6 | 4.0 s |
| LFM2.5-1.2B-Thinking | 4/6 | 1/6 | 5/6 | 10.6 s |

"Degraded" is not a near miss. With tools attached the 1.2B instruct answered *"I'm sorry,
but I can't provide that information"* to **what is the capital of France**, and wrote
`web_search` calls for **write me a haiku about winter**. Offering tools gives these models
a refusal posture: they stop believing they know things.

Three consequences shape the design:

- **The model that picks the tool is not the model that answers.** Those are different
  jobs and measurably different models — the 350M picks correctly 6/6 at 1.0 s, the 1.2B
  manages 3/6 at 4.0 s. Picking a function needs no world knowledge, which is the same
  reason the smallest model backs the stage-3 classifier. The answering model never sees a
  tool schema at all.
- **Decline by default.** An unmatched request gets no tools — the behaviour that was
  correct before the package existed. A gate that fires on a misread request is worse than
  no gate.
- **Match the trigger, not the topic.** "What time is it" needs a clock; "how do timezones
  work" does not. The patterns anchor on the user asking for a current value or an action,
  not on subject matter.

A tool result is retrieved text by another name, so it goes through the same treatment:
results are appended as `role: "tool"` turns, and a result arriving on the 350M escalates
to the reader tier for the same reason a retrieval hit does — the small model inverts
sources it is asked to read.

Results are capped (`MAX_RESULT_CHARS`). A fetched page can exceed the context window, and
the failure mode is not an error — it is the model silently losing the question at the top
of the prompt. Truncation is marked so the model can say the result was cut.

## Persona

Every request gets a shared system prompt (`app/persona.py`) prepended automatically
unless the caller already supplies their own `system` message. The name defaults to
`Lucy` (`LEGEND_ASSISTANT_NAME`); setting it to empty restores the anonymous persona,
which tells the model to say it has no name rather than invent one.

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

