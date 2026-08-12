# Contributing

## Getting set up

```bash
uv sync                                    # dependencies, from the lockfile
uv run pytest                              # 487 tests, ~6s, no Ollama needed
```

The test suite runs the router cascade against a stub backend. No models load, nothing
touches the network, and you do not need Ollama installed to work on routing logic. If a
change you are making requires a live model to test, that is a signal it belongs in the
eval suite rather than in `tests/`.

To run the actual service you need Ollama and the imported models — see the README.

## Before opening a pull request

```bash
uv run ruff check .
uv run pytest
```

Both run in CI on Python 3.11, 3.12 and 3.13, alongside a Docker build.

**Do not run `ruff format`.** The lint rules in `pyproject.toml` are pinned deliberately
and the codebase is hand-formatted; `ruff format` would rewrite 37+ files. Adopting a
formatter is a reasonable thing to decide, but it should be its own pull request with
nothing else in it.

If you change dependencies, commit the updated `uv.lock` — CI installs with `--frozen`
and will fail if the lockfile has drifted from `pyproject.toml`.

## The one convention that matters here

**Claims in this repository are expected to carry their measurements.**

This is not a style preference; it is the thing that makes the project work. Look at the
comments in `app/config.py` — nearly every default is annotated with the numbers that
produced it, including the ones that record a plausible idea being measured and rejected.
Several of those settings were originally set to the "obviously better" value and were
wrong.

So:

- If you change a default, say what you measured and on what hardware. A default without
  a number behind it is a guess, and guesses here have a track record of being wrong in
  ways that took days to find.
- If you measured something and it did *not* work, write that down too. The
  `verify_enabled` and `persona_capabilities` flags both exist to document a negative
  result and to let the next person re-run the comparison.
- **Compare configurations by interleaving or reversing order, never by running A then
  B.** Constrained hardware loses throughput under sustained load — roughly 30% on the
  machine this was developed on — and that fact invalidated three separate conclusions
  before it was found. Whichever configuration runs second loses. See
  `docs/measurements.md`.
- Sample more than twice. A two-sample check once reported the numeric guardrails as
  fully working; at six samples they were passing 1 in 6.

## Tests versus evals

They answer different questions and are not interchangeable.

- **`tests/`** proves the routing *logic* — that a given input reaches a given route,
  that a guardrail fires, that a malformed request is rejected. Deterministic, stubbed,
  fast. Every change should keep these green.
- **`evals/`** measures whether the *answers* are any good. It needs a running server and
  real generation, so it lives outside the test suite:

  ```bash
  uv run python scripts/eval.py                     # everything
  uv run python scripts/eval.py --routes-only       # routing only, ~1ms a case
  uv run python scripts/eval.py --save-baseline     # accept the current numbers
  ```

Two properties of the eval suite are worth preserving if you extend it:

- **Checks are declarative** — substrings, numbers, regexes, route names. There is no
  model-judges-the-answer step, because a model judging output is precisely what this
  project measured as unreliable: the 350M rubber-stamped 8 of 8 wrong answers.
- **Cases are sampled, not run once.** A case scores as the fraction of samples that
  passed, so flakiness reads as 50% rather than as a coin flip that happened to land.

Note that the eval suite's latency line **is not a benchmark** — a full run takes tens of
minutes, so later cases are measured on a hotter machine than earlier ones.

## Reporting a bug

Include the routing decision, not just the bad answer. Every response carries an
`x_legend_route` field and an `X-Legend-Route` header saying which model answered, which
stage decided, why, and which guardrail fired. "It gave a bad answer" and "it gave a bad
answer *and the router sent it to the 350M*" are different bugs with different fixes.

## Scope

`ROADMAP.md` lists what is planned and, more usefully, what was considered and ruled out
with the reasoning attached. Worth reading before proposing something large — several
attractive ideas are already in there marked as measured-and-rejected on this hardware.

## Licence

By contributing you agree that your contributions are licensed under the
[MIT Licence](LICENSE).
