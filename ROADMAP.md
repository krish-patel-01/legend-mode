# Roadmap

The goal is accuracy on this laptop that stands up against a single 10–15B model. The
path is architectural, not more parameters: nothing here proposes a bigger model, because
nothing bigger fits.

Steps 1 through 4 are built and on `two-model-guardrails`. Each is marked DONE below with
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

## Deferred, with the reason

**RL on model weights — not viable here.** [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
provides environments only; training happens in TRL, torchforge, verl or SkyRL. TRL's GRPO
path requires vLLM and CUDA, and the docs are written around one GPU minimum. There is no
Intel-iGPU path. Their own published result is also worth weighing: Qwen3-1.7B trained with
GRPO still could not consistently win Wordle, and gpt-oss-20b was needed for that. The
ceiling model here is 1.2B. Revisit only with an NVIDIA GPU — and note the paper's own
finding cuts the same way: systems are planner-limited, and architecture beat scaling.

**Contextual bandit over the router — now unblocked.** Five routes, reward from whether a
guardrail passed and how long the answer took, trained on decisions `/route/history`
already logs. No GPU, ~50 lines of numpy. It was waiting on step 2, which is built, so the
"is this better or is this noise" question now has an answer. The highest-value target is
the stage that measurably guesses: the classifier mis-filed a real question as `trivial`
during step 3 testing, and that is exactly a decision a bandit could learn from feedback
the system already collects.

**Tool execution.** `tools` is forwarded and `tool_calls` returned untouched; no tier
advertises tool support. `app/router/engine.py` documents the extension point. Deliberately
after routing is trustworthy.

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
