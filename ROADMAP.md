# Roadmap

The goal is accuracy on this laptop that stands up against a single 10–15B model. The
path is architectural, not more parameters: nothing here proposes a bigger model, because
nothing bigger fits.

Steps 1 through 4 are built and merged to `main`. Each is marked DONE below with
what it actually became, since three of the four changed shape once measured. What is
left is tuning — the defaults below are each a switch with a measurement behind it, and
the eval harness now reports the cost side of every one of them.

---

## The envelope everything has to fit in

Measured on this machine, not assumed. These numbers are why the plan is shaped the way
it is, so re-measure before treating any of them as still true.

| | |
|---|---|
| CPU | i5-12500H, 12 cores / 16 logical, **no NVIDIA GPU** (Intel Iris Xe) |
| RAM | 15.6 GB total; ~5.4 GB free after the process cleanup |
| Disk | C: 15.6 GB free (was 0.2 GB), K: 43.6 GB free |
| `general` (LFM2.5-350M) | 71.8 tok/s, 229 MB, 1.5 s cold load |
| `think` (LFM2.5-1.2B-Thinking) | 20.8 tok/s, 730 MB, 2.3 s cold load |

**The single most important measurement.** Each model was asked to judge 16
question/answer pairs, 8 right and 8 wrong:

| Critic | Accuracy | Rubber-stamped a wrong answer | Finished | Cost |
|---|---|---|---|---|
| 350M | 50% (chance) | 8/8 | 16/16 | 0.6 s |
| Qwen3.5-0.8B, thinking off | 56% | 7/8 | 16/16 | 1.1 s |
| Qwen3.5-0.8B, thinking on | 100% | 0/8 | 6/16 | 28.3 s |
| **1.2B** | **100%** | **0/8** | 14/16 | 28.1 s |

Three consequences that constrain every step below:

1. **Only the 1.2B can verify anything.** The 350M has no discriminative power at all —
   not poor accuracy, zero. It cannot be made into a cheap pre-filter; a prompt ordering
   it to work the answer out first changed nothing.
2. **Verification costs ~28 s.** So it cannot run on every turn, and any design that
   assumes it can is not implementable here.
3. **Self-verification is worthless.** A model's critic is only as good as its generator.
   Cross-model checking works; the same model re-reading itself does not.

---

## Step 2 — An evaluation harness — **DONE**

Built as `evals/cases.yaml`, `scripts/eval.py` and `evals/baseline.json`. See the Evals
section of `README.md` for usage.

It justified itself on the first full run by falsifying a claim this file's earlier
version treated as settled: a two-sample check had reported the guardrail cases as all
passing, and at six samples the 350M was overriding the injected value on 5 of 6 discount
questions and 5 of 6 temperature conversions. That led to contradicted numeric groundings
being corrected rather than merely logged. **19% of grounded samples still need
correcting**, which is now a reported metric rather than an unknown.

The rest of this section is the original design, kept because it explains the choices.

**Shape.** A case file plus a runner:

```
evals/cases.yaml        prompt, optional conversation history, expected route,
                        an answer check, and tags
scripts/eval.py         runs cases against a live server, prints per-category
                        pass rates and latency, diffs against a stored baseline
evals/baseline.json     last accepted run, committed, so regressions are visible
```

**Categories**, mirroring the error classes actually observed:

- `computable` — arithmetic, percentages, discounts, leap year, timezones, units.
  Guardrails should make these deterministic; a failure here is a real regression.
- `reasoning` — word problems and multi-step questions. Must route to `think`.
- `dispute` — multi-turn. The thread must stay on `think` and must not contradict a
  correct earlier answer. Current measured rate: 5 held / 1 caved / 2 vague out of 8.
- `factual` — things neither model reliably knows. Expected to fail until step 4; the
  point is to have the failures counted rather than discovered by accident.
- `persona` — no prompt echo, no invented name, user's name handled separately from the
  assistant's. Guards against the regressions documented in `app/persona.py`.
- `cheap` — lookups and greetings that must **not** be promoted to the 1.2B. Protects
  latency, which is otherwise the thing that silently rots.

**Checks** should be declarative — exact numeric match, regex, must-not-contain — not
prose comparison. Prose-against-prose matching is the unreliable step `app/guardrails.py`
exists to avoid, and it should not sneak into the scorer either.

**Two run modes.** A stub mode that needs no models (fast, for CI and for routing
assertions) and a live mode that spends real generation time. `uv run pytest` already
covers the former for routing; the live mode is the new part.

**Shape it like an OpenEnv environment** — `reset()`, an action, a scalar reward. It costs
nothing today, it is a perfectly ordinary test harness either way, and it means the RL
option below needs no rewrite if the hardware ever changes.

**Seed it** from the cases already proven this session; they are listed with expected
answers throughout `README.md`.

**Done when** `uv run python scripts/eval.py` prints per-category pass rates and a diff
against the committed baseline, and the numbers reproduce what `README.md` claims.

---

## Step 3 — Effort controller and adjudication — **DONE**

Built as `app/effort.py` and `app/adjudicate.py`. See the Effort and Adjudication
sections of `README.md` for what shipped.

**What the design got wrong, and what measuring fixed.** Two things below turned out not
to survive contact with the hardware, and both are worth recording because the reasoning
that produced them looked sound.

*The two critic rules do not intersect.* "The verifier is always the 1.2B and never the
350M" and "the verifier is never the model that produced the answer" are each defensible.
Together, with a two-model palette, they permit exactly one case — the 1.2B judging a
smaller model's answer — and forbid verification of anything the reasoning tier writes.
The implementation reports that gap in the response metadata rather than quietly
substituting self-verification, which the envelope above already calls worthless.

*Verification is dominated by escalation.* The critic was re-measured over 8 pairs, and
the number that mattered was not accuracy:

| Critic budget | Accuracy | Verdict emitted | Wrong answers waved through | Median |
|---|---|---|---|---|
| 512 tokens | 25% | **2/8** | 0/4 | 45.1 s |
| 1024 tokens | 75% | 6/8 | 0/4 | 24.9 s |
| 2048 tokens | **88%** | 8/8 | 1/4 | 26.7 s |

512 tokens — the budget the first implementation shipped with — is below the floor where a
verdict is emitted at all: it returns "unsure" on 6 of 8 and charges 45 seconds, which
from outside looks exactly like a working verifier that never fires.

Two "reasoning off" rows were struck from this table after the fact. They appeared to show
that disabling the reasoning block bought the 350M's rubber-stamping behaviour, and the
premise was false: Ollama advertises a `thinking` capability for this model but
`think: false` does not suppress the block, so those rows measured a truncated reasoning
block and a parser scraping a verdict out of it. Same failure as the original critic
harness bug, one layer up. Whether a genuinely non-reasoning 1.2B behaves that way is
still an open question, and answering it needs the instruct build, not a flag.

At the budget where it works, verification costs 26.7 s. Answering the same question on
the 1.2B costs about 25 s and yields a better answer instead of a grade on a worse one,
and in the failure case verification costs double because "incorrect" still has to be
followed by a regeneration. So `verify_enabled` ships false. The machinery stays, because
that conclusion is about this hardware and not about the idea: a second model that could
judge in 2 s would flip it immediately.

*Per-effort token budgets are not a brevity control on a reasoning model.* This section
originally called them the fix for the ~1-in-6 dispute turns that exhaust their budget.
They made it worse. A thinking model emits its reasoning block first, so a budget below
what that reasoning needs does not shorten the answer — it removes it, and essentially
every dispute turn returned the exhaustion message instead of roughly one in six. The
tier default is the floor on a thinking tier now, and the budgets apply only where a model
answers without a reasoning block. Brevity there comes from the prompt or nowhere.

Worse, the eval suite scored that regression as a clean run: the exhaustion reply is a
well-formed sentence with no forbidden number in it, so `regex` and `not_number` both
passed. It is now counted globally and reported.

What did pay for itself, and is on by default: the capitulation guard — which costs
nothing at all, being a numeric comparison — and the effort *levels*, which decide what
gets adjudicated and what gets retrieved even where they no longer decide the budget.

The rest of this section is the original design, kept because it explains the choices.

This is where the self-reflection idea lands, made affordable. The controller is not a
nicety; at 28 s per verification it is the only thing that makes verification possible at
all.

**Estimate difficulty before answering**, from signals already computed: which rule fired,
the embedding margin, prompt length, whether a guardrail grounded the question, whether
this is a dispute. No extra model call.

**Map it to an effort level**, each fixing a model, a token budget, and whether to verify:

| Effort | Model | Verify? | For |
|---|---|---|---|
| `fast` | 350M | no | greetings, lookups, grounded facts |
| `standard` | 1.2B | no | ordinary reasoning |
| `careful` | 1.2B | yes, ≤1 repair | hard reasoning, disputes, low-confidence routes |

**Per-effort token budgets fix a live bug.** A dispute currently gets the same fixed 1536
budget as anything else, which is why a terse "nope" can burn the whole budget reasoning
about nothing — measured at roughly 1 in 6 dispute turns, currently papered over by a
fallback message in `app/api.py`.

**Loop discipline**, because this is where reflection designs usually fail:

- The verifier is always the 1.2B and never the 350M.
- The verifier is never the model that produced the answer.
- **At most one repair.** Then stop. Unbounded critique oscillates (A says 4, B says 2,
  A says 2…) and small critics are exactly noisy enough to make that likely.
- If verification still fails, **abstain** — say the answer is uncertain and show the
  working. The transcript that started this project was as much a confidence failure as
  an accuracy one; confident wrongness is the worst available output.

**Expose it** as an `effort` field on the request plus an env default, so the automatic
estimate can be overridden.

**Budget discipline.** Decide up front what fraction of requests may pay the 28 s, and let
the eval harness report it. An accuracy win that triples median latency is not a win.

---

## Step 4 — Retrieval — **DONE**

Built as `app/retrieval/` and `scripts/ingest.py`. See the Retrieval section of
`README.md`.

Two things the design underspecified, both found by running it:

**A grounded small model still has to be able to read.** The design says "a grounded 1B
outperforms an ungrounded 14B" and then leaves open which tier answers. Handed the chunk
reading *"the verifier is always the 1.2B and never the 350M"*, the 350M answered *"The
model that verifies answers is LFM2.5-350M"* — the exact inverse of the source in front
of it. A retrieval hit now escalates to the reading tier.

**Do not ask a 1.2B for a citation.** The first prompt labelled each passage
`[source#heading]` and asked the model to copy it. The reply was the bracketed citation
and nothing else — it matched the most recent pattern in the prompt rather than using the
text beneath it. The citations are known exactly at that point, so they are computed and
appended by `app/api.py`. Same rule as `app/guardrails.py`: compute what can be computed.

The gate calibrated cleanly. On this corpus, on-topic questions score 0.70–0.76 and
off-topic ones 0.50–0.51, so the 0.66 default sits in a wide gap — but that is a property
of this corpus, and `scripts/ingest.py --probe` exists so it gets re-measured rather than
inherited.

What is *not* solved: the `factual` cases still fail, because the corpus that ships is
this project's own documentation and nothing in it knows when the Treaty of Westphalia
was signed. The machinery is done; a general-knowledge corpus is a thing to supply.

The rest of this section is the original design, kept because it explains the choices.

The last error class: things neither model knows, where no amount of extra thinking helps.
This is also the one place a small model genuinely beats a large one — a grounded 1B
outperforms an ungrounded 14B on factual questions.

- Reuse the already-pinned bge embedder. Storage as sqlite + numpy; no heavy new
  dependency for a corpus this size.
- Retrieve, then answer **strictly from the retrieved context**, with a citation.
  Reading comprehension is something a 1.2B is actually good at; recall is not.
- **Gate it — do not make it always-on.** In the paper behind this plan, indiscriminate
  retrieval *hurt* GPQA by 5.0 points, because retrieved content overrode correct
  parametric knowledge. Retrieval should fire when the question needs a fact the model
  is unlikely to hold, which is what the step 3 estimator is already computing.
- This is also the natural home for conversation memory. Recalling a name from many turns
  back is a context-capacity limit at this size, and retrieval solves it where prompting
  cannot.

---

## The queue, in order

What is actually next, sequenced deliberately. This section is the live work; the one below
it is the parked work, and the two are adjacent so neither gets read as the other.

The order is not preference. Items 1 and 2 are the same investigation approached from two
ends, cheaper end first — better evidence before better prompting — so that whatever item 1
still has to solve is measured rather than assumed. Item 3 must follow both, because it
re-freezes the tool-result fixture and freezing it first would capture evidence the two
changes above are about to invalidate.

1. **Engine/category scoping for SearXNG.** `search()` sends only `q`, `format` and
   `safesearch`; SearXNG also takes `engines` and `categories`. Whether scoping helps is a
   hypothesis and not a known win — the general category already carries the wikipedia
   infoboxes, so the plausible gain is `news` on recency questions rather than
   encyclopaedic ones. Probe before changing anything, and if scoping does land it is
   decided in code from the query text, never as a tool parameter: schema breadth
   measurably degrades the dispatcher (see `app/tools/notes.py` and `app/tools/gate.py`).

2. **The grounding problem.** `last f1 race` returns navigational pages and the writer
   invents an answer rather than saying the evidence lacks one. `TOOL_RESULT_NOTE` in
   `app/persona.py` is one-directional — every clause pushes the model to commit and none
   covers the case where the results do not contain the answer. That is the next thing to
   measure, in two arms, watching the `tools` category for the regression.

3. **Stale measurements.** `evals/tool_context.json` needs re-freezing, and
   `scripts/tool_bench.py`'s docstring claims **15/20** on the `read` job where the last
   real evidence scored **7–8/20**. A benchmark that overstates its own result is precisely
   what the conventions in this file exist to prevent.

4. **A case that fails because the model does not believe it can look something up.**
   Named as the blocker in `app/persona.py`: without one, `persona_capabilities` has no
   headroom to prove itself in and stays off with no measured benefit. The case has to be
   *found* by sampling, not authored — one invented to look like the bug would flatter the
   clause for nothing.

5. **Tool schema audit.** Nothing developer-controlled should appear in the LLM-visible
   schema, and nothing in the schema should be invisible to the code. A first read says the
   schemas are already clean, so the deliverable is the test that keeps them that way.

**Dropped: the `files` tool family.** It was the only unbuilt one and it carried the real
security surface, which is why it kept coming up. It is not wanted — recorded here so it is
not proposed again.

Two constraints that are decisions rather than findings, and so will not change by being
re-measured:

- **Wolfram Alpha is off the table.** Considered as a source of direct answers and declined.
- **The Docker stack has never been run end-to-end.** `deploy/` is written and reviewed and
  the SearXNG container is used daily, but `compose` as a whole has not been brought up.
  Do not describe it as verified.

---

## Deferred, with the reason

**RL on model weights — not viable here.** [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
provides environments only; training happens in TRL, torchforge, verl or SkyRL. TRL's GRPO
path requires vLLM and CUDA, and the docs are written around one GPU minimum. There is no
Intel-iGPU path. Their own published result is also worth weighing: Qwen3-1.7B trained with
GRPO still could not consistently win Wordle, and gpt-oss-20b was needed for that. The
ceiling model here is 1.2B. Revisit only with an NVIDIA GPU — and note the paper's own
finding cuts the same way: systems are planner-limited, and architecture beat scaling.

**Chain-of-thought scaffolding on the 350M — tested and dropped.** The idea was to buy the
350M enough reasoning to stand in for a 1.2B call. It buys some and not enough: a
step-by-step nudge takes it from 50% to 64% on 14 multi-step word problems (57% once the
persona prompt production actually sends is included), against 93% from the instruct tier
at 5 s and 100% from the reasoning tier. Sampling and voting made it *worse* — 43%, and
identical at k=1, 3 and 5, because the model's errors scatter instead of converging, so
there is no mode to take. The agreement-gated version is dominated by simply calling the
1.2B, which is both cheaper and more accurate. Full numbers and the reasoning in
`scripts/cot_bench.py`, which is kept as a reusable arm-comparison harness.

Two things came out of it that outlived the idea. **The same nudge makes the 1.2B worse**
(93% -> 86%), so CoT is a small-model crutch here, not a general improvement. And **a
system prompt that dictates output format displaces the question on the 350M** — asked for
a specific answer line, it emitted the template, angle brackets included, in 13 tokens.
That is `app/persona.py`'s "never end a system prompt on a quotable sentence" for the third
time, and it is why the CoT nudge is put in the user turn.

**Contextual bandit over the router — technically unblocked, deliberately last.** Context
from the request's features, arms are the three routes, reward from whether a guardrail
passed, whether the next turn disputed the answer, and how long it took. No GPU, ~50 lines
of numpy. The target would be the part of the cascade that measurably guesses: the
classifier mis-filed a real question as `trivial` during step 3 testing.

It is parked behind everything else — including the work above it in this file — for three
reasons, none of which is difficulty:

- **The training data does not exist.** `app/history.py` is a 200-entry in-memory ring
  buffer with no outcome field, wiped on every restart. Nothing has ever been persisted, so
  there is no corpus to learn from and no way to check a learned policy against the past.
- **The reward is sparse.** Guardrail pass covers only computable questions, latency always
  argues for the cheapest route, and the one clean signal — a dispute on the following turn
  — is rare by design. Single-user traffic is tens of requests a day, which is thin for a
  policy over 384-dimensional context.
- **The baseline keeps improving for free.** Switching stage 2 from centroids to
  nearest-example took its decision rate on everyday prompts from 5% to 50% in one commit.
  A learned router has to beat a hand-tuned cascade that is still moving.

If it is picked up, the order is: persist decisions *and* their outcomes to sqlite first —
worth doing on its own, since it is how a misroute gets found at all — and then learn the
two thresholds (`min_score`, `min_margin`) rather than a full policy. Two numbers need far
less evidence than a policy over embeddings.

**Tool execution — now built.** `app/tools/` dispatches server-side across three families
(`basics`, `web`, `notes`), behind a gate, with the tool *picked* by a different model than
the one that answers. That split is not an optimisation: attaching tool schemas measurably
degrades every model here, to the point where the 1.2B instruct refused to name the capital
of France. See the table in `app/tools/gate.py` and
[docs/architecture.md](docs/architecture.md#tools).

What remains is breadth — more families — rather than mechanism.

**Vision.** Parked with the Qwen3.5-0.8B tier; image requests return 422.

**LFM2.5-8B-A1B** (8B total, ~1B active, so roughly `think`-tier decode speed). The single
highest-leverage model acquisition if the orchestrated-sub-agent design is ever attempted,
since that design is planner-limited and a 1.2B planner is exactly what the evidence says
will not work. Never downloaded — only 6 KB of stubs in the HF cache.

**Qwen3.5-2B / 4B.** Weights on disk, commented out in `models.yaml`.

---

## What "comparable to 10–15B" honestly means

Reachable on **checkable** work: arithmetic, unit and calendar facts, logic that survives
verification, extraction, and grounded factual lookup. On those, guardrails plus
verification plus retrieval can match or beat a much larger ungrounded model, because the
larger model is guessing from parameters while this one is checking.

Not reachable on open-ended world knowledge, long-form writing, or anything needing broad
parametric recall. There, scale is the whole game and no amount of orchestration
substitutes. Better to plan around that than to discover it at step 4.

---

## Ground rules that came out of building this

- **Measure, don't reason from plausibility.** The mmproj crash theory, the "small models
  can't judge" assumption, and three separate persona wordings were all wrong or
  incomplete until tested. Two of those were mine.
- **Never end a system prompt on a quotable sentence.** Documented in `app/persona.py`
  with two separate instances of a model echoing the final line verbatim.
- **A guard that fires on a misread question is worse than no guard.** Decline when
  unsure; anchor patterns at both ends.
- **Verify the server under test is the process you think it is.** A `pkill` that silently
  failed produced a full page of meaningless "passing" results this session.
- **Distrust a passing test as much as a failing one.** Two harness bugs — a scorer
  reading a `<think>` block instead of a verdict, and a brand check flagging a correct
  denial — each pointed at the wrong conclusion until the harness itself was fixed.
- **A component that silently does nothing looks identical to one that works.** The
  critic at a 512-token budget answered "unsure" on 6 of 8 pairs and charged 45 seconds
  for it. Nothing errored; the feature was simply inert. Measure that a mechanism *fires*,
  not only that it does not crash.
- **Two individually sound constraints can have an empty intersection.** "Always the 1.2B"
  and "never the model that answered" each read as obviously right, and together they rule
  out verifying anything the reasoning tier writes. Check that a rule set admits the case
  it was written for.
- **Prefer the cheaper move that produces the answer over the one that grades it.** At
  equal cost, escalating beats verifying, because verifying still leaves you needing an
  answer. This is what turned step 3 from a verification design into a budgeting one.
- **This machine throttles ~30% under sustained load, so A-then-B timing is worthless.**
  Three conclusions were drawn and discarded before this was found: that a stall was
  eviction, that eviction was ruled out, and that a fourth resident tier caused an eval
  median to go from 1.8 s to 11.9 s. Running the residency comparison in both orders
  showed whichever config went *second* lost, every time. Reverse the order or interleave;
  never read off totals from a sequential run.
- **A measurement expires when the system changes.** "All three models co-reside" was
  true, and adding a fourth tier made it false without anyone touching that code. Findings
  need re-taking, not citing.
