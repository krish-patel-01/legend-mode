"""Can a chain-of-thought loop make the 350M reason? Measured: a little, not enough.

**Result, 14 multi-step word problems, arms interleaved, greedy arms reproduced exactly
across two runs.** `decode` is Ollama's own eval_duration; see the note on Call.

    arm                        correct   decode   tok/q   tok/s
    350M direct                 7/14  50%    0.7s     70    93
    350M + CoT                  9/14  64%    0.9s     93    93
    350M + CoT, vote@5          6/14  43%    4.5s    459    93
    1.2B instruct (`chat`)     13/14  93%    5.0s    195    33
    1.2B instruct + CoT        12/14  86%    9.8s    354    33
    1.2B thinking (`think`)    14/14 100%   20.6s    877    33

Three findings, in order of how much they close the question:

1. **CoT works on the 350M and does not close the gap.** 50% -> 64% for 0.2 s is a real
   effect — it fixed jacket, pages, eggs, marbles and trip — but it also broke train
   (3 instead of 70) and paint (0.4 instead of 3), which the unprompted arm got right.
   64% is thirty points below a tier that already answers in 5 s.

2. **The same nudge makes the 1.2B worse**: 93% -> 86%, at double the latency. So this is
   not "CoT helps"; it is "CoT helps a model that cannot reason and gets in the way of one
   that can". A result read off the 350M alone would have been read wrong.

3. **Sampling and voting is a dead end, and the shape of the failure is why.** vote@1,
   vote@3 and vote@5 all score 43% — extra samples add exactly nothing. The 350M's five
   samples per question hold 2.7 distinct values on average, and split cleanly in two:

       crates   19x5          chairs   108x5        <- when it knows, 5/5 identical
       eggs     500, 1, 4, 94, 84                   <- when it does not, 5 distinct
       series   30, 24, 60, 45, 18

   Self-consistency assumes reasoning paths converge on the truth more often than on any
   one error. These errors *scatter* rather than converge, so there is usually no mode to
   take. The right answer is somewhere in the five samples 10/14 times and voting recovers
   6/14; picking it out needs something that can recognise a right answer, the 350M scores
   at chance at exactly that (models.yaml), and using the 1.2B as the judge means paying
   for the 1.2B — at which point asking it the question directly is both cheaper and
   better. Which is ROADMAP.md's "prefer the move that produces the answer over the one
   that grades it", reached again from a different direction.

The one design that might have rescued it — sample k times, keep the answer if the samples
agree, escalate if they do not — is dominated outright:

    k=2 gate    83%   4.8 s/question
    k=3 gate    89%   6.4 s/question
    1.2B instruct alone   93%   6.4 s/question     <- cheaper *and* better than k=3

Agreement does predict correctness (unanimous at k=5: 5/5 right; split: 1/9 right), so the
signal is real. It is just not economically extractable, because detecting it costs about
what the better answer costs.

**Nothing here was wired into the router.** The scaffolding does what it claims and still
lands ~30 points under a tier that already exists.

Footnote on the persona prompt, since this file measures the 350M with none and production
sends it one. Same 14 questions, same nudge:

    no system prompt              9/14  64%   93 tok/q
    `brief`, what production uses 8/14  57%   92 tok/q
    `full`                        6/14  43%   82 tok/q

So the production number is 57%, not 64%, and `persona: brief` in models.yaml is
load-bearing on reasoning too — not only on the greetings it was originally measured
against. `brief` dilutes; the format-dictating prompts below *displace*, which is a
different and much larger failure.

---

## Original question

The question is whether prompting scaffolding can buy the 350M enough of the 1.2B's
reasoning to be worth its latency. It is asked here as a *model* comparison — this script
calls Ollama directly and never touches the router — because guardrails, retrieval and
effort budgets would otherwise be answering half the questions for it.

**One CoT shape is ruled out before testing: draft, self-critique, revise.** models.yaml
records the 350M scoring 50% as a judge and rubber-stamping 8 of 8 wrong answers. A loop
whose repair step depends on the model recognising its own error inherits that number, so
it cannot work here regardless of how the prompt is written.

What is left is the family that needs no judgement:

  direct   one call, no scaffolding. The current behaviour, as the baseline.
  cot      one greedy call that is told to work step by step before answering.
  vote     k sampled CoT calls; the *numbers* they end on are tallied and the mode wins.

`vote` is legal for the same reason app/adjudicate.py's self-consistency pass is: no model
grades anything, k extracted numbers are compared exactly. Ties abstain rather than pick.

Against those, the two models that already exist — the 1.2B instruct tier that serves
`chat`, and the 1.2B reasoning tier that serves `think`.

**Arms are interleaved per question and their order rotates.** This laptop loses ~30% of
its throughput under sustained load, so an A-then-B run charges the second arm for the
first arm's heat; three conclusions were drawn and discarded that way before it was found.
Rotating spreads that penalty evenly instead of concentrating it on whoever ran last.

    uv run python scripts/cot_bench.py                  # everything
    uv run python scripts/cot_bench.py --arms direct,cot,vote
    uv run python scripts/cot_bench.py --limit 4 --k 3  # a quick shape check
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adjudicate import operative_number
from app.backends.ollama import OllamaClient, OllamaError
from app.config import get_registry, get_settings

# --- the questions -------------------------------------------------------------
#
# Every one needs at least two steps and lands on a single unambiguous number, because
# the scorer compares numbers and nothing else. No prose is judged anywhere in this file.
# Deliberately not drawn from evals/cases.yaml: those exist to test routing, and several
# are answered by a guardrail before any model sees them.

QUESTIONS: list[tuple[str, str, float]] = [
    ("crates", "A shop has 3 crates of apples with 12 apples in each crate. It sells 17 apples. How many apples are left?", 19),
    ("pens", "Maya buys 4 packs of pens with 6 pens in each pack, then gives away 9 pens. How many pens does she have left?", 15),
    ("train", "A train travels 60 km in its first hour and 80 km in its second hour. What is its average speed in km/h over the two hours?", 70),
    ("jacket", "A jacket costs 240 rupees. It is discounted by 25%, and then a further 10 rupees is taken off at the till. What is the final price in rupees?", 170),
    ("pages", "Sam reads 22 pages on Monday, twice that many on Tuesday, and 8 fewer than Tuesday on Wednesday. How many pages did he read in total?", 102),
    ("chairs", "A hall has 15 rows of 8 chairs. 12 chairs are broken and taken away. How many chairs are left?", 108),
    ("ages", "Ravi is 3 times as old as his sister. In 6 years he will be twice as old as she is. How old is Ravi now?", 18),
    ("tank", "A tank holds 500 litres and is currently 3/5 full. How many more litres are needed to fill it?", 200),
    ("eggs", "Eggs cost 6 rupees each. If you buy as many as you can with 100 rupees, how much change is left?", 4),
    ("stairs", "A building has 9 floors. There are 14 steps between each floor and the next. How many steps is it from the ground floor to the top floor?", 112),
    ("paint", "One tin of paint covers 12 square metres. A wall is 5 metres by 6 metres. How many whole tins are needed to paint it?", 3),
    ("marbles", "Ali has 5 more marbles than Ben. Together they have 31 marbles. How many marbles does Ali have?", 18),
    ("trip", "A car drives 150 km at 50 km/h, then 120 km at 60 km/h. How many hours does the whole trip take?", 5),
    ("series", "What is the next number in this sequence: 2, 6, 12, 20, 30, ?", 42),
]

# --- prompts -------------------------------------------------------------------
#
# **The CoT nudge goes in the user turn, and there is no system prompt at all.** That is
# measured, and it was measured because the first version of this file did the obvious
# thing and got a meaningless result. Asked the crates question with `num_predict` 512:
#
#   system "…Finish with a line of exactly this form: ANSWER: <number>"   ANSWER: <12>
#   system "…ends with a line reading ANSWER: followed by the number"     12 - 17 = -5
#   system "…then state the final number."                                12 - 17 = -5
#   no system prompt at all                                               36 - 17 = 19
#
# Thirteen tokens against thirty-nine. Every instruction added displaced the question
# rather than shaping the answer, and the first row is app/persona.py's "never end a
# system prompt on a quotable sentence" landing for the third time in this project — the
# model emitted the template, angle brackets included, instead of using it.
#
# So the 350M gets nothing in the system slot. The same six questions across prompt
# shapes, greedy:
#
#   bare                                          4/6   36 tok
#   user turn + "Think step by step."             5/6   80 tok
#   user turn + "Solve this step by step…"        5/6   83 tok
#   system "Think step by step before answering." 5/6   87 tok
#
# The nudge is worth about one question in six and doubles what the model writes. Which
# short phrasing carries it is within noise, so the user-turn version wins on the
# principle the table above established: the system slot is the fragile one.
_COT_NUDGE = "Think step by step."

# `\boxed{}` is not requested anywhere — LFM2.5 emits it unprompted on arithmetic, and
# asking for a different sentinel measurably made things worse. Read what the model
# actually does rather than insisting it do something else.
_BOXED = re.compile(r"\\boxed\{\s*(-?\d[\d,]*(?:\.\d+)?)")
_SENTINEL = re.compile(r"ANSWER\s*[:\-]\s*\**\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_TRAILING = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


def extract(text: str) -> float | None:
    """The number a reply lands on, or None if it never states one.

    Looser than app/adjudicate.py's `operative_number`, on purpose. That one guards a
    live reply and abstains whenever several numbers are candidates, because a guard
    firing on a misread is worse than no guard. This one scores a benchmark answer to a
    question that has exactly one right number, where declining to read a reply that
    plainly ends in "= 19" would just under-report every arm equally.
    """
    tail = _THINK_CLOSE.split(text)[-1]
    body = tail if tail.strip() else text

    for pattern in (_BOXED, _SENTINEL):
        hits = pattern.findall(body) or pattern.findall(text)
        if hits:
            return _to_float(hits[-1])

    strict = operative_number(body)
    if strict is not None:
        return strict

    # Last resort: the final number written. These prompts end on their conclusion, so
    # the closing number is the assertion — but it is the weakest read here, hence last.
    hits = _TRAILING.findall(body)
    return _to_float(hits[-1]) if hits else None


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def vote(values: list[float | None]) -> float | None:
    """Modal value, or None on a tie or when nothing was extractable.

    Abstaining on a tie is the same choice adjudicate.unstable_reply makes: with two
    equally supported candidates and no way to break the tie, picking one is guessing.
    """
    tally = collections.Counter(v for v in values if v is not None)
    if not tally:
        return None
    ranked = tally.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


# --- one measured call ---------------------------------------------------------


@dataclass
class Call:
    text: str = ""
    seconds: float = 0.0
    """Wall clock, which on a swapped tier includes loading the weights."""

    generate_seconds: float = 0.0
    """Ollama's own `eval_duration` — decode only, no load, no prompt ingest.

    Both are reported because they answer different questions and this benchmark needs
    both. Wall clock is what a user waits. Decode time is what the *model* costs, and it
    is the only number comparable across arms here: the 350M is pinned in RAM while the
    two 1.2B tiers evict each other between questions, so wall clock charges them for a
    load the 350M never pays. Reading arm-to-arm latency off wall clock alone would
    repeat the mistake in ROADMAP.md's ground rules for the fourth time.
    """

    load_seconds: float = 0.0
    tokens: int = 0
    error: str | None = None


async def ask(
    client: OllamaClient, spec, question: str, *, max_tokens: int, temperature: float
) -> Call:
    # No system message on any arm. See the prompt notes above: on the 350M every system
    # prompt tried displaced the question, and leaving it off the 1.2B arms too is what
    # keeps this a comparison of models rather than of prompts.
    messages = [{"role": "user", "content": question}]
    started = time.perf_counter()
    try:
        result = await client.chat(
            spec,
            messages,
            options={"num_predict": max_tokens, "temperature": temperature},
        )
    except OllamaError as exc:
        return Call(seconds=time.perf_counter() - started, error=str(exc)[:200])
    return Call(
        text=(result.get("message") or {}).get("content") or "",
        seconds=time.perf_counter() - started,
        # Ollama reports these in nanoseconds.
        generate_seconds=float(result.get("eval_duration") or 0) / 1e9,
        load_seconds=float(result.get("load_duration") or 0) / 1e9,
        tokens=int(result.get("eval_count") or 0),
    )


# --- the arms ------------------------------------------------------------------


@dataclass
class Arm:
    name: str
    alias: str
    max_tokens: int
    nudge: bool = False
    """Append the step-by-step line to the user turn."""

    temperature: float = 0.0
    samples: int = 1
    note: str = ""

    def prompt(self, question: str) -> str:
        return f"{question}\n\n{_COT_NUDGE}" if self.nudge else question


def build_arms(k: int) -> list[Arm]:
    return [
        Arm("direct", "general", 256,
            note="350M as it answers today"),
        Arm("cot", "general", 512, nudge=True,
            note="350M, one greedy step-by-step pass"),
        # Sampled, not greedy: k identical greedy replies would make the tally
        # meaningless. 0.7 is the temperature self-consistency is normally run at.
        Arm("vote", "general", 512, nudge=True, temperature=0.7, samples=k,
            note=f"350M, {k} sampled CoT passes, modal number wins"),
        Arm("instruct", "instruct-q3", 512,
            note="1.2B instruct — the tier `chat` runs on"),
        # Present so a win for `cot` can be read correctly. If the nudge lifts both
        # models by the same amount it is a prompting result, not a case for scaffolding
        # the small one specifically; only a lift that *closes the gap* argues for that.
        Arm("instruct-cot", "instruct-q3", 512, nudge=True,
            note="1.2B instruct, same step-by-step nudge"),
        Arm("think", "think", 2048,
            note="1.2B reasoning — the tier `think` runs on"),
    ]


@dataclass
class Result:
    arm: str
    question: str
    expected: float
    got: float | None = None
    seconds: float = 0.0
    generate_seconds: float = 0.0
    load_seconds: float = 0.0
    tokens: int = 0
    calls: int = 1
    error: str | None = None
    # Only `vote` fills this: what each sample landed on, so a wrong vote can be told
    # apart from k wrong samples.
    samples: list[float | None] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.got is not None and abs(self.got - self.expected) < 1e-6


async def run_arm(client, registry, arm: Arm, qid: str, prompt: str, answer: float) -> Result:
    spec = registry.by_alias(arm.alias)
    res = Result(arm=arm.name, question=qid, expected=answer, calls=arm.samples)
    values: list[float | None] = []

    for _ in range(arm.samples):
        call = await ask(
            client, spec, arm.prompt(prompt),
            max_tokens=arm.max_tokens, temperature=arm.temperature,
        )
        res.seconds += call.seconds
        res.generate_seconds += call.generate_seconds
        res.load_seconds += call.load_seconds
        res.tokens += call.tokens
        if call.error:
            res.error = call.error
            return res
        values.append(extract(call.text))

    res.samples = values
    res.got = vote(values) if arm.samples > 1 else values[0]
    return res


# --- reporting -----------------------------------------------------------------


def report(results: list[Result], arms: list[Arm], k: int) -> None:
    by_arm: dict[str, list[Result]] = collections.defaultdict(list)
    for r in results:
        by_arm[r.arm].append(r)

    total = len({r.question for r in results})
    print(f"\n{'=' * 92}\n{total} questions, one row per arm\n{'=' * 92}")
    print(f"{'arm':<13}{'correct':>9}{'blank':>7}{'decode s':>10}{'wall s':>9}"
          f"{'load s':>8}{'tok/q':>7}{'tok/s':>7}  note")
    print("-" * 92)

    for arm in arms:
        rows = by_arm.get(arm.name)
        if not rows:
            continue
        ok = sum(r.correct for r in rows)
        blank = sum(r.got is None for r in rows)
        decode = statistics.median(r.generate_seconds for r in rows)
        wall = statistics.median(r.seconds for r in rows)
        load = sum(r.load_seconds for r in rows) / len(rows)
        toks = statistics.mean(r.tokens for r in rows)
        gen = sum(r.generate_seconds for r in rows)
        rate = sum(r.tokens for r in rows) / gen if gen else 0.0
        print(f"{arm.name:<13}{ok:>4}/{len(rows):<3}{ok / len(rows):>5.0%}"
              f"{blank:>6}{decode:>10.1f}{wall:>9.1f}{load:>8.1f}{toks:>7.0f}"
              f"{rate:>7.1f}  {arm.note}")

    print("\ndecode = Ollama's eval_duration, the only column comparable across arms;")
    print("wall   = what a caller waits, including any weight load the tier had to pay.")

    # Voting is only worth its k calls if the tally beats a single sample from the same
    # distribution. Both numbers come out of the samples already collected.
    voted = by_arm.get("vote")
    if voted and k > 1:
        print(f"\nvote breakdown (same {k} samples, re-tallied):")
        for width in sorted({1, min(3, k), k}):
            hits = 0
            for r in voted:
                if not r.samples:
                    continue
                tallied = vote(r.samples[:width])
                hits += tallied is not None and abs(tallied - r.expected) < 1e-6
            label = "single sample" if width == 1 else f"majority of {width}"
            print(f"  {label:<18}{hits:>3}/{len(voted)}  {hits / len(voted):>5.0%}")

    print("\nper question (. correct, X wrong, - no answer):")
    names = [a.name for a in arms if a.name in by_arm]
    print(f"{'':<10}" + "".join(f"{n:<10}" for n in names))
    for qid, _, _ in QUESTIONS:
        row = {r.arm: r for r in results if r.question == qid}
        if not row:
            continue
        cells = ""
        for name in names:
            r = row.get(name)
            if r is None:
                cells += f"{'':<10}"
                continue
            mark = "." if r.correct else ("-" if r.got is None else "X")
            got = "" if r.got is None else _pretty(r.got)
            cells += f"{mark} {got:<8}"
        print(f"{qid:<10}{cells}")

    errors = [r for r in results if r.error]
    if errors:
        print(f"\n{len(errors)} call(s) failed:")
        for r in errors[:5]:
            print(f"  {r.arm:<10}{r.question:<10}{r.error}")


def _pretty(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:g}"


# --- driver --------------------------------------------------------------------


async def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="", help="comma-separated subset of arm names")
    parser.add_argument("--limit", type=int, default=0, help="first N questions only")
    parser.add_argument("--k", type=int, default=5, help="samples for the voting arm")
    parser.add_argument("--json", type=Path, help="write raw results here")
    args = parser.parse_args()

    arms = build_arms(args.k)
    if args.arms:
        wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = set(wanted) - {a.name for a in arms}
        if unknown:
            parser.error(f"unknown arm(s): {', '.join(sorted(unknown))}")
        arms = [a for a in arms if a.name in wanted]

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    settings, registry = get_settings(), get_registry()
    client = OllamaClient(settings)
    results: list[Result] = []

    print(f"{len(questions)} questions x {len(arms)} arms "
          f"({sum(a.samples for a in arms)} calls per question)")

    try:
        for index, (qid, prompt, answer) in enumerate(questions):
            # Rotate the running order so no arm is always the one paying for a hot CPU.
            order = arms[index % len(arms):] + arms[: index % len(arms)]
            marks = []
            for arm in order:
                res = await run_arm(client, registry, arm, qid, prompt, answer)
                results.append(res)
                marks.append(
                    f"{arm.name} {'ok' if res.correct else ('--' if res.got is None else 'X')}"
                    f" {res.seconds:.0f}s"
                )
            print(f"  [{index + 1}/{len(questions)}] {qid:<9} " + "  ".join(marks))
    finally:
        await client.aclose()

    report(results, arms, args.k)

    if args.json:
        args.json.write_text(
            json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8"
        )
        print(f"\nraw results -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
