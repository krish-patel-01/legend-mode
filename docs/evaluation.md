# Testing and evaluation

Two suites that answer different questions and are not interchangeable. The tests prove
the routing *logic*; the evals measure whether the *answers* are any good.

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


## Benchmarking scripts

Standalone probes under `scripts/`, kept because each produced a conclusion recorded in
[measurements.md](measurements.md):

- `scripts/tool_bench.py` — measures a model against both halves of the tool path:
  whether it picks the right tool, and whether holding tool schemas degrades its ordinary
  answers. It is the source of the table in [architecture.md](architecture.md#tools).
- `scripts/cot_bench.py` — chain-of-thought and prompt-displacement probing.
- `scripts/frames.py`, `scripts/functiongemma.py` — tool-calling frame experiments.

All of them need a live server and real generation, so treat their timings with the same
suspicion as the eval suite's — see the throttling note in
[measurements.md](measurements.md).
