# Measurements

Every number here was measured on the development machine, not assumed. **Re-measure
before treating any of them as still true** — most are properties of constrained hardware
rather than of the models, and better hardware would flip several of the conclusions.

The envelope everything has to fit in:

| | |
|---|---|
| CPU | i5-12500H, 12 cores / 16 logical, **no NVIDIA GPU** (Intel Iris Xe) |
| RAM | 15.6 GB total; ~5.4 GB free |
| `general` (LFM2.5-350M) | 71.8 tok/s, 229 MB, 1.5 s cold load |
| `think` (LFM2.5-1.2B-Thinking) | 20.8 tok/s, 730 MB, 2.3 s cold load |

## Only one model here can tell a right answer from a wrong one

This is the single most important measurement in the project, and three consequences
follow from it that constrain every other design decision.

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

