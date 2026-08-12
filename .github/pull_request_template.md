<!--
The convention that matters here: claims carry their measurements. See CONTRIBUTING.md.
-->

## What this changes

## Why

<!--
If this changes a default, a threshold, a prompt, or a budget, the "why" needs numbers.
Nearly every default in app/config.py was set from a measurement, several after the
obvious value was tried and found worse — so a change without one is hard to evaluate
and easy to regress later.

Two things that have repeatedly produced wrong conclusions in this project:

  - Comparing A then B. Constrained hardware loses throughput under sustained load
    (~30% on the development machine), so whichever ran second loses. Interleave or
    reverse the order.
  - Sampling twice. A two-sample check once reported the numeric guardrails as fully
    working; at six samples they were passing 1 in 6.
-->

## Checks

- [ ] `uv run pytest`
- [ ] `uv run ruff check .`
- [ ] `uv.lock` committed, if dependencies changed
- [ ] Prompt wording in `app/persona.py` re-measured, if touched — the wording there was
      settled against the real models, and unit tests cannot see a regression in it
