"""Run evals/cases.yaml against a live server and report per-category pass rates.

    uv run python scripts/eval.py                    # everything
    uv run python scripts/eval.py --category dispute # one category
    uv run python scripts/eval.py --routes-only      # routing checks, no generation
    uv run python scripts/eval.py --save-baseline    # accept the current numbers

Why this exists: every accuracy claim in README.md was produced by a throwaway script in
a temp directory, and there are now enough of them that re-checking by hand isn't
sustainable. Anything built on top of routing — the effort controller especially — needs
a way to tell an improvement from sampling noise.

Two design choices worth keeping:

**Checks are declarative.** Substrings, numbers, regexes, route names. There is no
model-judges-the-answer step. A model judging output is exactly what app/guardrails.py
exists to avoid — measured here, the 350M rubber-stamped 8 of 8 wrong answers — and
letting it into the scorer would make every number suspect.

**Cases are sampled, not run once.** These models are stochastic and single samples have
repeatedly pointed the wrong way: the 350M gave 365 days for a leap year in one sample
and 366 in the next. A case scores as the fraction of samples that passed, so a flaky
case reads as 0.5 rather than as a coin-flip pass or fail.

The scoring shape (reset -> act -> scalar reward in [0,1]) is deliberately the shape
OpenEnv expects, so this can be reused as an RL environment without a rewrite if the
hardware ever allows it. Today it is an ordinary test harness.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "cases.yaml"
BASELINE = ROOT / "evals" / "baseline.json"

CATEGORIES = (
    "computable", "reasoning", "dispute", "factual", "persona", "cheap",
    "effort", "retrieval", "tools",
)

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_NUMBER_TOKEN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# The reply app/api.py substitutes when a model spends its whole budget reasoning and
# emits nothing. Counted globally, because it is the one failure the per-case checks are
# structurally blind to: it is a well-formed English sentence, so it satisfies
# `regex: \w`, and it contains no forbidden number, so `not_number` passes too. A change
# that made every dispute turn return it scored as a clean run — the warnings were only
# in the server log. If this number is not near zero, something is starving a tier.
EXHAUSTION_MARKER = "couldn't reach an answer"


def numbers_in(text: str) -> set[float]:
    """Every number in a reply, digits and English words alike.

    Word forms matter: the reasoning tier answers the box problem as "four" about as
    often as "4", and a digits-only check would score a correct answer as a miss.
    """
    found: set[float] = set()
    for token in _NUMBER_TOKEN.findall(text):
        try:
            found.add(float(token.replace(",", "")))
        except ValueError:
            continue
    for word, value in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            found.add(float(value))
    return found


def base_kind(grounded: str | None) -> str | None:
    """Strip the " (corrected)" suffix so the guard kind can be compared on its own."""
    if grounded is None:
        return None
    return grounded.split(" (")[0]


@dataclass
class Sample:
    """One run of one case."""

    passed: bool
    problems: list[str] = field(default_factory=list)
    reply: str = ""
    route: str = ""
    model: str = ""
    grounded: str | None = None
    elapsed_ms: float = 0.0
    effort: str = ""
    max_tokens: int = 0
    adjudicated: dict[str, Any] | None = None
    retrieved: list[str] | None = None


def check(reply: str, meta: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    """Every failed expectation, as readable strings. Empty list means the sample passed."""
    problems: list[str] = []
    low = reply.lower()

    if (want := expect.get("route")) is not None:
        allowed = want if isinstance(want, list) else [want]
        if meta.get("route") not in allowed:
            problems.append(f"route={meta.get('route')} not in {allowed}")

    if (forbidden := expect.get("not_route")) is not None:
        if meta.get("route") == forbidden:
            problems.append(f"route wrongly {forbidden}")

    if "grounded" in expect:
        want_g = expect["grounded"]
        got = meta.get("grounded")
        # YAML parses a bare `none` as the *string* "none", not as null, so both spellings
        # have to mean "assert that no guard fired". Getting this wrong silently inverted
        # the check and failed a case that was behaving correctly.
        if want_g is None or want_g == "none":
            if got is not None:
                problems.append(f"expected no grounding, got {got!r}")
        # "<kind> (corrected)" means the model contradicted the computed value and the
        # substitution put it right — the user got a correct answer, so the case passes.
        # It is still worth counting, since the rate is a direct measure of how often the
        # tier ignores an instruction, so the runner reports corrections separately.
        # "<kind> (contradicted)" would mean the wrong answer went out unfixed and fails.
        elif base_kind(got) != want_g:
            problems.append(f"grounded={got!r} expected {want_g!r}")

    for needle in expect.get("contains") or []:
        if needle.lower() not in low:
            problems.append(f"missing {needle!r}")

    if any_of := expect.get("any_of"):
        if not any(n.lower() in low for n in any_of):
            problems.append(f"none of {any_of} present")

    for needle in expect.get("not_contains") or []:
        if needle.lower() in low:
            problems.append(f"contains forbidden {needle!r}")

    if (want_n := expect.get("number")) is not None:
        if float(want_n) not in numbers_in(reply):
            problems.append(f"number {want_n} absent")

    for bad in expect.get("not_number") or []:
        if float(bad) in numbers_in(reply):
            problems.append(f"forbidden number {bad} present")

    if (pattern := expect.get("regex")) is not None:
        if not re.search(pattern, reply):
            problems.append(f"regex {pattern!r} did not match")

    if (want_effort := expect.get("effort")) is not None:
        allowed = want_effort if isinstance(want_effort, list) else [want_effort]
        got_effort = (meta.get("effort") or {}).get("level")
        if got_effort not in allowed:
            problems.append(f"effort={got_effort} not in {allowed}")

    if (budget := expect.get("max_tokens_below")) is not None:
        got_budget = (meta.get("effort") or {}).get("max_tokens", 0)
        if got_budget >= budget:
            problems.append(f"token budget {got_budget} not below {budget}")

    # The mirror check, and the one that matters on a reasoning tier: a budget below the
    # floor removes the answer rather than shortening it (see app/effort.py).
    if (floor := expect.get("max_tokens_at_least")) is not None:
        got_budget = (meta.get("effort") or {}).get("max_tokens", 0)
        if got_budget < floor:
            problems.append(f"token budget {got_budget} below the floor {floor}")

    # The exhaustion reply is well-formed prose carrying no number, so `regex` and
    # `not_number` both wave it through. Any case that expects a real answer should say so
    # explicitly, since the global counter only says *that* it happened, not where.
    if expect.get("real_answer") and EXHAUSTION_MARKER in low:
        problems.append("returned the budget-exhaustion reply instead of an answer")

    if (want_r := expect.get("retrieved")) is not None:
        # `retrieved` is None when retrieval never ran and [] when it ran and found
        # nothing above threshold. Both mean "no corpus text reached the model", and the
        # distinction matters for cost, not for correctness — so the check flattens them.
        got_r = bool(meta.get("retrieved"))
        if got_r != bool(want_r):
            problems.append(f"retrieved={meta.get('retrieved')!r}, expected {want_r}")

    # `tools: false` asserts nothing ran; a name or list asserts which. The false case is
    # the one that matters most and the one worth writing out for ordinary prompts: the
    # measured cost of attaching tool schemas to a request that does not need them is the
    # model refusing questions it can answer — "I'm sorry, but I can't provide that
    # information" to *what is the capital of France*. See app/tools/gate.py.
    if (want_t := expect.get("tools")) is not None:
        called = [c["name"] for c in ((meta.get("tools") or {}).get("calls") or [])]
        if want_t is False:
            if called:
                problems.append(f"tools ran ({', '.join(called)}), expected none")
        else:
            wanted = [want_t] if isinstance(want_t, str) else list(want_t)
            if not any(name in called for name in wanted):
                problems.append(f"tools={called or 'none'}, expected one of {wanted}")

    return problems


class Runner:
    def __init__(self, base_url: str, routes_only: bool, timeout: float) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._routes_only = routes_only
        self.retries = 0

    def close(self) -> None:
        self._client.close()

    def _chat(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        # One retry on a 5xx, because a full run takes tens of minutes and a single
        # transient upstream hiccup used to abort the whole thing with nothing recorded.
        # Observed twice: generation on the reasoning tier stalled to ~0.08 tok/s, the
        # server's own 300 s timeout to Ollama fired, and the resulting 502 killed a run
        # that was 40% done. The cause of the stall is not established — it is not model
        # eviction, which was tested and ruled out — so this is deliberately a resilience
        # measure and not a fix. A retried case is reported, so it can never quietly
        # inflate a score.
        for attempt in (1, 2):
            resp = self._client.post(
                "/v1/chat/completions", json={"model": "auto", "messages": messages}
            )
            if resp.status_code < 500:
                break
            if attempt == 1:
                self.retries += 1
                print(f"    (upstream {resp.status_code}, retrying once)", flush=True)
                time.sleep(5)
        resp.raise_for_status()
        body = resp.json()
        reply = (body["choices"][0]["message"].get("content") or "").strip()
        return reply, body["x_legend_route"]

    def _route_only(self, prompt: str) -> tuple[str, dict[str, Any]]:
        resp = self._client.get("/route/debug", params={"prompt": prompt})
        resp.raise_for_status()
        return "", resp.json()

    def run_once(self, case: dict[str, Any]) -> Sample:
        started = time.perf_counter()
        turns: list[str] = case.get("turns") or [case["prompt"]]

        expect = case.get("expect") or {}
        if self._routes_only:
            # /route/debug generates nothing, so every content check would compare
            # against an empty string and "fail" for the wrong reason. Only the routing
            # expectations survive. (Multi-turn cases are skipped by the caller: debug
            # is single-turn and would lose the sticky stage, which is their whole
            # point.)
            reply, meta = self._route_only(turns[-1])
            expect = {k: v for k, v in expect.items() if k in ("route", "not_route")}
        else:
            messages: list[dict[str, str]] = []
            reply, meta = "", {}
            for user in turns:
                messages.append({"role": "user", "content": user})
                reply, meta = self._chat(messages)
                messages.append({"role": "assistant", "content": reply})

        problems = check(reply, meta, expect)
        plan = meta.get("effort") or {}
        return Sample(
            passed=not problems,
            problems=problems,
            reply=" ".join(reply.split()),
            route=meta.get("route", ""),
            model=meta.get("model", ""),
            grounded=meta.get("grounded"),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            effort=plan.get("level", ""),
            max_tokens=plan.get("max_tokens", 0),
            adjudicated=meta.get("adjudicated"),
            retrieved=meta.get("retrieved"),
        )


def load_cases(path: Path, categories: list[str] | None, only: str | None) -> list[dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = raw.get("cases") or []
    if not cases:
        raise SystemExit(f"{path} defines no cases")
    if categories:
        cases = [c for c in cases if c.get("category") in categories]
    if only:
        cases = [c for c in cases if c["id"] == only]
        if not cases:
            raise SystemExit(f"no case with id {only!r}")
    return cases


def main() -> int:
    # Model replies routinely contain characters the Windows console codepage can't
    # encode (≈, ×, en-dashes). Without this the harness dies mid-run on a print, which
    # looks like a failing eval rather than a broken reporter.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--category", action="append", choices=CATEGORIES,
                    help="restrict to a category; repeatable")
    ap.add_argument("--case", help="run a single case by id")
    ap.add_argument("--routes-only", action="store_true",
                    help="check routing via /route/debug without generating")
    ap.add_argument("--samples", type=int, help="override every case's sample count")
    ap.add_argument("--include-known-failing", action="store_true",
                    help="also run cases marked known_failing (slow, and expected to fail)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--save-baseline", action="store_true",
                    help="write the current scores to evals/baseline.json")
    ap.add_argument("--verbose", "-v", action="store_true", help="print every sample")
    args = ap.parse_args()

    cases = load_cases(CASES, args.category, args.case)
    runner = Runner(args.base_url, args.routes_only, args.timeout)

    try:
        runner._client.get("/v1/models").raise_for_status()
    except Exception as exc:
        print(f"cannot reach {args.base_url}: {exc}\n"
              f"start it with: uv run uvicorn app.main:app --port 8000", file=sys.stderr)
        return 2

    scores: dict[str, float] = {}
    by_category: dict[str, list[float]] = {}
    latencies: list[float] = []
    failed_detail: list[tuple[str, Sample]] = []
    skipped: list[str] = []
    corrected = grounded_total = 0
    # Step 3's budget discipline is a number, not a feeling: an accuracy win that triples
    # median latency is not a win, so what fraction of requests paid for adjudication is
    # reported alongside the scores rather than left to be inferred.
    effort_counts: dict[str, int] = {}
    adj_ran = adj_repaired = adj_no_critic = retrieval_fired = exhausted = 0

    for case in cases:
        cid, category = case["id"], case["category"]
        # Cases pinned at a measured capability ceiling. They are worth keeping — if a
        # later change ever solves one, that should be visible — but each costs minutes
        # to fail, so a routine run skips them.
        if case.get("known_failing") and not args.include_known_failing:
            skipped.append(cid)
            continue
        if args.routes_only:
            expect = case.get("expect") or {}
            if case.get("turns") or not ({"route", "not_route"} & set(expect)):
                skipped.append(cid)
                continue

        n = args.samples or case.get("samples", 1)
        samples = [runner.run_once(case) for _ in range(n)]
        score = sum(s.passed for s in samples) / len(samples)
        scores[cid] = score
        by_category.setdefault(category, []).append(score)
        latencies.extend(s.elapsed_ms for s in samples)
        corrected += sum(1 for s in samples if (s.grounded or "").endswith("(corrected)"))
        grounded_total += sum(1 for s in samples if s.grounded)
        for s in samples:
            if s.effort:
                effort_counts[s.effort] = effort_counts.get(s.effort, 0) + 1
            if EXHAUSTION_MARKER in s.reply:
                exhausted += 1
            if s.retrieved:
                retrieval_fired += 1
            if s.adjudicated:
                adj_ran += 1
                adj_repaired += bool(s.adjudicated.get("repaired"))
                adj_no_critic += bool(s.adjudicated.get("skipped"))

        mark = "ok  " if score == 1.0 else ("FAIL" if score == 0.0 else "flaky")
        first = samples[0]
        print(f"  [{mark}] {score:>4.0%} {cid:<34} {first.route:<8} {first.effort:<9}"
              f"{str(first.grounded or '-'):<14} {first.elapsed_ms:>6.0f}ms")
        if args.verbose or score < 1.0:
            for s in samples:
                if not s.passed:
                    print(f"          - {'; '.join(s.problems)}")
                    print(f"            reply: {s.reply[:110]!r}")
                    break
        if score < 1.0:
            failed_detail.append((cid, next(s for s in samples if not s.passed)))

    runner.close()

    print("\n" + "=" * 78)
    print(f"{'category':<14} {'score':>7}  cases")
    for category in CATEGORIES:
        if values := by_category.get(category):
            print(f"{category:<14} {sum(values) / len(values):>7.0%}  {len(values)}")
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"{'OVERALL':<14} {overall:>7.0%}  {len(scores)}")
    if latencies:
        latencies.sort()
        print(f"\nlatency  median {latencies[len(latencies) // 2]:.0f}ms  "
              f"p90 {latencies[int(len(latencies) * 0.9)]:.0f}ms  "
              f"max {latencies[-1]:.0f}ms")
    if grounded_total:
        # Corrections are passes, but the rate is how often the answering tier ignored a
        # value it was handed — a direct quality signal, and the thing to watch if the
        # persona wording or the note format ever changes.
        print(f"grounded {grounded_total} sample(s), of which {corrected} needed "
              f"correction ({corrected / grounded_total:.0%})")

    total_samples = sum(effort_counts.values())
    if total_samples:
        spread = "  ".join(
            f"{level} {effort_counts.get(level, 0)} ({effort_counts.get(level, 0) / total_samples:.0%})"
            for level in ("fast", "standard", "careful")
        )
        print(f"effort   {spread}")
    if adj_ran:
        # `no_critic` is the honest count of times adjudication was authorised and could
        # not run: with two models, a reasoning-tier answer has no independent judge.
        print(f"adjudicated {adj_ran} sample(s): {adj_repaired} repaired, "
              f"{adj_no_critic} had no available critic")
    if retrieval_fired:
        print(f"retrieval injected corpus text into {retrieval_fired} sample(s)")
    if exhausted:
        print(f"** {exhausted} sample(s) hit the budget-exhaustion reply — see "
              f"EXHAUSTION_MARKER; a tier is being starved **")
    if runner.retries:
        # Reported rather than swallowed: a run that needed retries is a run where the
        # upstream was unhealthy, and its latency numbers should not be trusted.
        print(f"** {runner.retries} request(s) needed a retry after a 5xx — treat this "
              f"run's latencies as unreliable **")
    if skipped:
        print(f"\nskipped ({len(skipped)}: known-failing, or in --routes-only "
              f"multi-turn / no routing expectation): {', '.join(skipped)}")

    # The diff is printed even when saving. A full run takes tens of minutes, so making
    # "see what changed" and "accept the change" mutually exclusive would mean running
    # the suite twice to do both.
    exit_code = 0
    if BASELINE.exists():
        old = json.loads(BASELINE.read_text(encoding="utf-8")).get("scores", {})
        regressions = [
            (cid, old[cid], scores[cid])
            for cid in scores
            if cid in old and scores[cid] < old[cid] - 1e-9
        ]
        improvements = [
            (cid, old[cid], scores[cid])
            for cid in scores
            if cid in old and scores[cid] > old[cid] + 1e-9
        ]
        added = [cid for cid in scores if cid not in old]
        if regressions:
            print(f"\n{len(regressions)} REGRESSION(S) vs baseline:")
            for cid, was, now in regressions:
                print(f"  {cid:<34} {was:.0%} -> {now:.0%}")
            exit_code = 1
        if improvements:
            print(f"\n{len(improvements)} improvement(s) vs baseline:")
            for cid, was, now in improvements:
                print(f"  {cid:<34} {was:.0%} -> {now:.0%}")
        if added:
            print(f"\nnot in baseline yet: {', '.join(added)}")
        if not (regressions or improvements or added):
            print("\nno change vs baseline")
    else:
        print("\nno baseline yet - run with --save-baseline to record one")

    if args.save_baseline:
        # Merge rather than overwrite: a partial run (--category, --case) must not drop
        # the scores for cases it never executed.
        existing = {}
        if BASELINE.exists():
            existing = json.loads(BASELINE.read_text(encoding="utf-8")).get("scores", {})
        merged = {**existing, **scores}
        BASELINE.write_text(
            json.dumps(
                {"scores": dict(sorted(merged.items())),
                 "overall": round(sum(merged.values()) / len(merged), 4)},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"\nbaseline updated ({len(scores)} case(s)) -> {BASELINE.relative_to(ROOT)}")
        exit_code = 0  # accepting the numbers is not a failure

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
