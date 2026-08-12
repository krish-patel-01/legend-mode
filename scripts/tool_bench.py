"""Is FunctionGemma-270M a better dispatcher — or a better tool-result reader — than what we run?

**Result, 2026-08-09: no to the first, interesting-but-no to the second. Nothing adopted.**

    job   arm                 correct   decode   tok/q   refused
    pick  350M                  6/10      0.3s     21              greedy, 1 sample
    pick  functiongemma         0/10      0.4s     25
    read  1.2B/tool            15/20      1.1s     42     2/20     t=0.6, 4 samples
    read  1.2B/assistant       11/20      1.2s     45     0/20
    read  1.2B/prefill          9/20      0.9s     32     0/20
    read  functiongemma        12/20      1.2s     83     0/20

`read` is sampled at 0.6 rather than run greedily because the defect under test is a
refusal that appears on some samples and not others; at temperature 0 it either always
fires or never does, which measures the wrong thing precisely. The three `1.2B/*` rows are
the tool-result *framings* in scripts/frames.py — see that file for why the shipped one
stays despite having the only refusals.

Read the `read` row with the replies, not off the score — the scorer is deliberately crude
(a shared figure, no refusal phrase) and reciting the evidence passes it. What the models
actually wrote, given identical frozen search results:

    gold   1.2B  "approximately **$4,280.81 per ounce** (as of August 6, 2026)"      38 tok
           FG    four bullets: the right price, then "$4,350", then "1 ounce of
                 Gold is $137.63" — which is the *gram* price off the same page    174 tok
    time   1.2B  "I don't have real-time capabilities, so I can't provide..."        53 tok
           FG    "...Sunday, September 09, 2026, at 05:31:14 UTC" — August read
                 as September, JST relabelled UTC                                   34 tok
    movie  1.2B  "approximately **$653 million** in North American box office"       47 tok
           FG    empty. One token, on the longest context of the five.                1 tok

So the two failure modes are opposites. **FunctionGemma never refuses** — 0/5 against the
1.2B's 1/5, and that is the honest win here — but it digests rather than answers, and its
figures are unreliable in a way that matters more than a refusal does: a model that says it
cannot help is a nuisance, one that confidently reports the gram price as the ounce price
is a liability. The 1.2B is accurate and concise when it commits, and sometimes will not.

Two findings worth more than the verdict:

1. **The gold refusal did not reproduce.** The bug that started this — the 1.2B refusing a
   price with `4,247.20` in its context — answered "$4,280.81 per ounce" here without
   complaint. The refusal moved to `get_time` instead, with the exact clock reading
   attached. So it is not one poisoned topic; it is an intermittent posture that lands on
   different cases run to run, which is why a single failing transcript reads as a rule.

2. **FunctionGemma's frame removes refusals on the 1.2B too — and does not help.** Its
   format puts the call and the result inside the *model's own turn*, so the evidence
   arrives as something the assistant already did rather than as something it was told
   (see scripts/functiongemma.py). Ported to the 1.2B it took refusals from 2/20 to 0/20
   and accuracy from 15/20 to 11/20: "I don't have real-time capabilities" was replaced by
   a $4,200–$4,300 range for a price the evidence printed as 4,280.81. Refuses nothing,
   says nothing. The refusal is a symptom of not committing, not the cause — see
   scripts/frames.py, which keeps the arms and the reasoning.

`pick` is a clean loss and needs less interpretation: 0/10, mostly by emitting no call at
all against seven tools whose descriptions were written for LFM2.5. Google reports 58%
zero-shot rising to 85% fine-tuned, and positions the model as a base to fine-tune rather
than a drop-in, so this is roughly the advertised behaviour rather than a surprise.

---

Two questions, deliberately separated, because they are two different jobs and the models
here are good at different ones (see app/tools/dispatch.py for why the roles are split at
all):

  pick    given the real tool schemas and a real request, name the right tool with usable
          arguments. Today: the 350M (`general`), which scored 6/6 on selection.
  read    given a tool *result* and the question it answers, state the fact. Today: the
          1.2B (`instruct-q3`), which **refuses** — the open bug. Asked the gold price with
          `ounce 4,247.20` sitting in its context it answered "I don't have access to
          real-time data" under three separate framings, including no persona at all.

`read` is the one that decides whether FunctionGemma is worth adopting. `pick` already
works; a win there is a sidegrade. A win on `read` fixes the thing that is actually broken.

**Scoring is mechanical on both, and neither scorer judges prose.** `pick` compares the
tool name and looks for a required substring in the arguments. `read` counts two things
that need no taste: whether the reply repeats any number that was in the tool result
(grounding), and whether it contains a refusal phrase (the bug). A reply can fail both, and
the interesting failure — confident refusal — is exactly "no shared number, refusal
present".

**The `read` fixture is captured live, once, then frozen.** Scoring a model against a
search that changes between arms would measure the search. `--capture` runs the real
registry against the real backends and writes evals/tool_context.json; every later run
replays that file, so the arms see byte-identical evidence and the run is reproducible
after the price moves.

Arms rotate per case for the reason cot_bench.py documents: this laptop loses ~30% of its
throughput under sustained load, so a fixed order charges the last arm for the first one's
heat.

    uv run python scripts/tool_bench.py --capture     # refresh the frozen evidence
    uv run python scripts/tool_bench.py               # both jobs
    uv run python scripts/tool_bench.py --job read    # just the one that matters
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import frames
import functiongemma as fg
import httpx

from app.backends.ollama import OllamaClient, OllamaError
from app.config import ModelSpec, get_registry, get_settings
from app.persona import TOOL_RESULT_NOTE
from app.tools.dispatch import _parse_calls
from app.tools.registry import build_registry

FIXTURE = Path(__file__).resolve().parent.parent / "evals" / "tool_context.json"

# FunctionGemma is pulled from the Ollama library rather than imported from a GGUF by
# models.yaml, so it has no entry there yet — that is the decision this script exists to
# inform. `ollama cp functiongemma:270m legend/functiongemma` puts it where `tag` expects,
# which is the same route the 350M took in (see the note on `general` in models.yaml).
FUNCTIONGEMMA = ModelSpec(
    alias="functiongemma",
    tier="dispatch",
    repo="google/functiongemma-270m-it",
    file="functiongemma-270m-it-Q4_K_M.gguf",
    keep_alive="5m",
    num_ctx=8192,
    tools=True,
    default_max_tokens=512,
    temperature=0.0,
)


# --- job one: picking a tool ---------------------------------------------------


@dataclass
class PickCase:
    request: str
    tool: str
    must_contain: str | None = None
    """Substring the arguments must carry, lowercased. None means any arguments pass.

    Deliberately weak. Argument *fidelity* on a 270M-class model is known-bad and is
    already repaired deterministically for the case that mattered (notes.content_from_request),
    so demanding an exact object here would score a problem we have solved by other means.
    What this checks is that the model carried the subject across at all.
    """


PICK_CASES: list[PickCase] = [
    # Verbatim from the transcripts where the gate failed. They reach a tool now, so what
    # is measured here is only the dispatcher's choice given that they do.
    PickCase("what is the current gold price?", "web_search", "gold"),
    PickCase("what is the weather today in Ahmedabad?", "web_search", "ahmedabad"),
    PickCase("search how much the new spiderman movie had earned till now?", "web_search", "spider"),
    PickCase("look up the price of bitcoin", "web_search", "bitcoin"),
    # The deterministic family, where a wrong pick is cheap but a wrong argument is not.
    PickCase("what time is it in Tokyo", "get_time", "tokyo"),
    PickCase("what is 17 * 23", "calculate", "17"),
    PickCase("how much disk space do I have", "system_status"),
    # Notes. `write_note` is where the 350M truncated the body; `search_notes` is the
    # recall side, which reached no tool at all before the gate was widened.
    PickCase("make a note that the retro is on Friday", "write_note", "friday"),
    PickCase("remember that I take my coffee black", "write_note", "coffee"),
    PickCase("what did I write about coffee", "search_notes", "coffee"),
]


# --- job two: reading a tool result --------------------------------------------


@dataclass
class ReadCase:
    """A question, and the tool result that already contains its answer."""

    name: str
    question: str
    tool: str
    arguments: dict
    result: str


# Which requests get captured. The first is the open bug, verbatim.
CAPTURE: list[tuple[str, str, str, dict]] = [
    ("gold", "what is the current gold price?", "web_search", {"query": "current gold price per ounce"}),
    ("weather", "what is the weather today in Ahmedabad?", "web_search", {"query": "weather today Ahmedabad"}),
    ("movie", "how much has the new spiderman movie earned so far?", "web_search", {"query": "new Spider-Man movie box office total"}),
    ("time", "what time is it in Tokyo", "get_time", {"city": "Tokyo"}),
    ("disk", "how much disk space do I have", "system_status", {}),
]

_REFUSAL = re.compile(
    r"(?:do(?:n'?t| not) have (?:access|real[- ]time|current|up[- ]to[- ]date)"
    r"|(?:can'?t|cannot|unable to) (?:provide|access|give|retrieve|browse|check)"
    r"|no access to|not able to access|I'?m an AI|as an AI"
    r"|real[- ]time (?:data|information|access)"
    r"|check a (?:reliable|live|financial) (?:source|website)"
    r"|recommend (?:checking|visiting))",
    re.IGNORECASE,
)

# Three or more digits, or anything with a decimal point — the shape of a price, a
# temperature, a gross. One and two digit numbers are excluded because they are list
# markers as often as they are facts.
#
# **It was four digits until it scored a right answer wrong.** The evidence read
# `Total $653M/Wk 2` and both models answered "$653 million"; three digits did not clear
# the old bar, so a correct, grounded, well-phrased reply counted as a miss for every arm
# on that case. Same failure as the `not_number` artifact in evals/cases.yaml: a scorer
# tuned against the answers it expected to see rather than the ones it got.
_FIGURE = re.compile(r"\d[\d,]{2,}(?:\.\d+)?|\d+\.\d+")


def figures(text: str) -> set[str]:
    return {m.replace(",", "") for m in _FIGURE.findall(text)}


# --- one measured call ---------------------------------------------------------


@dataclass
class Call:
    text: str = ""
    calls: list[tuple[str, dict]] = field(default_factory=list)
    seconds: float = 0.0
    generate_seconds: float = 0.0
    tokens: int = 0
    error: str | None = None


class RawClient:
    """`/api/generate` with `raw: true` — no template, no message assembly.

    FunctionGemma needs this because the Ollama build has no chat template to apply (see
    scripts/functiongemma.py). Raw mode is the only way to put the model's own turn markers
    in front of it, and it is not something the router's client should learn to do for one
    model, so it lives here until the bake-off says it should.
    """

    def __init__(self, settings) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_host, timeout=settings.request_timeout
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(self, tag: str, prompt: str, *, max_tokens: int) -> dict:
        resp = await self._client.post("/api/generate", json={
            "model": tag,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.0,
                "stop": fg.STOP,
            },
        })
        if resp.status_code >= 400:
            raise OllamaError(f"{tag}: {resp.status_code} {resp.text[:200]}")
        return resp.json()


async def ask_raw(raw: RawClient, tag: str, prompt: str, *, max_tokens: int) -> Call:
    started = time.perf_counter()
    try:
        reply = await raw.generate(tag, prompt, max_tokens=max_tokens)
    except (OllamaError, httpx.HTTPError) as exc:
        return Call(seconds=time.perf_counter() - started, error=str(exc)[:200])
    text = reply.get("response") or ""
    return Call(
        text=text,
        calls=fg.parse_calls(text),
        seconds=time.perf_counter() - started,
        generate_seconds=float(reply.get("eval_duration") or 0) / 1e9,
        tokens=int(reply.get("eval_count") or 0),
    )


async def ask(client, spec, messages, *, tools=None, max_tokens=256, temperature=0.0) -> Call:
    started = time.perf_counter()
    try:
        reply = await client.chat(
            spec, messages, tools=tools,
            options={"num_predict": max_tokens, "temperature": temperature},
        )
    except OllamaError as exc:
        return Call(seconds=time.perf_counter() - started, error=str(exc)[:200])
    message = reply.get("message") or {}
    return Call(
        text=message.get("content") or "",
        calls=_parse_calls(message),
        seconds=time.perf_counter() - started,
        generate_seconds=float(reply.get("eval_duration") or 0) / 1e9,
        tokens=int(reply.get("eval_count") or 0),
    )


# --- results -------------------------------------------------------------------


@dataclass
class Row:
    arm: str
    case: str
    ok: bool = False
    detail: str = ""
    seconds: float = 0.0
    generate_seconds: float = 0.0
    tokens: int = 0
    grounded: bool = False
    refused: bool = False
    error: str | None = None


@dataclass
class Arm:
    name: str
    spec: ModelSpec
    native: bool = False
    """Talk FunctionGemma's own format over raw `/api/generate` instead of `/api/chat`."""

    frame: str = frames.TOOL
    """How the tool result is presented — see app/tools/frame.py. Ignored when `native`."""

    temperature: float = 0.0
    """**The `read` arms run at 0.6, which is what models.yaml gives this tier in
    production.** Greedy would be the cleaner comparison for anything else, but the defect
    under test is a refusal that appears on some samples and not others, and at temperature
    0 it either always fires or never does — measuring the wrong thing precisely."""


async def run_pick(client, raw, arm: Arm, case: PickCase, schemas) -> Row:
    if arm.native:
        call = await ask_raw(raw, arm.spec.tag, fg.dispatch_prompt(case.request, schemas),
                             max_tokens=256)
    else:
        call = await ask(client, arm.spec, [{"role": "user", "content": case.request}],
                         tools=schemas)
    row = Row(arm=arm.name, case=case.request[:34], seconds=call.seconds,
              generate_seconds=call.generate_seconds, tokens=call.tokens, error=call.error)
    if call.error:
        row.detail = "error"
        return row
    if not call.calls:
        row.detail = "no call"
        return row

    name, args = call.calls[0]
    if name != case.tool:
        row.detail = f"{name} (wanted {case.tool})"
        return row
    if case.must_contain and case.must_contain not in json.dumps(args).lower():
        row.detail = f"{name}, args missing {case.must_contain!r}"
        return row
    row.ok = True
    row.detail = name
    return row


async def run_read(client, raw, arm: Arm, case: ReadCase, schemas) -> Row:
    if arm.native:
        mine = [s for s in schemas if s["function"]["name"] == case.tool]
        call = await ask_raw(
            raw, arm.spec.tag,
            fg.answer_prompt(case.question, mine, case.tool, case.arguments, case.result),
            max_tokens=384,
        )
    else:
        # No persona — this compares frames and models, and persona length is a separate
        # measured variable (see cot_bench.py's footnote).
        messages = [{"role": "system", "content": TOOL_RESULT_NOTE}] + frames.build(
            arm.frame,
            question=case.question,
            calls=[(case.tool, case.arguments, case.result)],
            note=TOOL_RESULT_NOTE,
        )
        call = await ask(client, arm.spec, messages, max_tokens=384,
                         temperature=arm.temperature)
    row = Row(arm=arm.name, case=case.name, seconds=call.seconds,
              generate_seconds=call.generate_seconds, tokens=call.tokens, error=call.error)
    if call.error:
        row.detail = "error"
        return row

    text = fg.first_reply(call.text) if arm.native else call.text
    row.refused = bool(_REFUSAL.search(text))
    row.grounded = bool(figures(text) & figures(case.result))
    # The bar: repeat a figure from the evidence and do not refuse. Both halves are
    # required — a reply that quotes a number *and* then disclaims it is the failure the
    # user actually saw.
    row.ok = row.grounded and not row.refused
    row.detail = " ".join(text.split())[:70] or "(empty)"
    return row


# --- capture -------------------------------------------------------------------


async def capture(settings) -> int:
    registry = build_registry(settings)
    out = []
    for name, question, tool, arguments in CAPTURE:
        print(f"  {name:<9} {tool}({json.dumps(arguments)}) ...", end="", flush=True)
        result = await registry.invoke(tool, arguments, request=question)
        print(f" {'ok' if result.ok else 'FAILED'}, {len(result.content)} chars")
        out.append({
            "name": name, "question": question, "tool": tool,
            "arguments": arguments, "result": result.content, "ok": result.ok,
        })
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nfrozen -> {FIXTURE}")
    failed: list[str] = [str(r["name"]) for r in out if not r["ok"]]
    if failed:
        print(f"warning: {', '.join(failed)} returned an error and will measure nothing")
    return 0


def load_read_cases() -> list[ReadCase]:
    if not FIXTURE.exists():
        raise SystemExit(f"no frozen evidence at {FIXTURE} — run with --capture first")
    return [
        ReadCase(r["name"], r["question"], r["tool"], r["arguments"], r["result"])
        for r in json.loads(FIXTURE.read_text(encoding="utf-8")) if r.get("ok")
    ]


# --- reporting -----------------------------------------------------------------


def report(title: str, rows: list[Row], arms: list[str], *, read: bool) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    header = f"{'arm':<16}{'correct':>10}{'decode s':>10}{'tok':>7}"
    if read:
        header += f"{'grounded':>10}{'refused':>9}"
    print(header)
    print("-" * 88)
    for arm in arms:
        mine = [r for r in rows if r.arm == arm]
        if not mine:
            continue
        ok = sum(r.ok for r in mine)
        gen = sum(r.generate_seconds for r in mine)
        line = (f"{arm:<16}{ok:>4}/{len(mine):<3}{ok / len(mine):>5.0%}"
                f"{gen / len(mine):>10.1f}{sum(r.tokens for r in mine) // len(mine):>7}")
        if read:
            line += f"{sum(r.grounded for r in mine):>10}{sum(r.refused for r in mine):>9}"
        print(line)

    print("\nper case:")
    for case in dict.fromkeys(r.case for r in rows):
        print(f"  {case}")
        for arm in arms:
            mine = [x for x in rows if x.arm == arm and x.case == case]
            if not mine:
                continue
            ok = sum(x.ok for x in mine)
            tally = f"{ok}/{len(mine)}" if len(mine) > 1 else ("." if ok else "X")
            # Show a failing sample when there is one: the successes all look alike and
            # the failure is the thing worth reading.
            shown = next((x for x in mine if not x.ok), mine[0])
            print(f"    {tally:>5} {arm:<16}{shown.detail}")


# --- driver --------------------------------------------------------------------


async def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="refresh the frozen evidence and exit")
    parser.add_argument("--job", choices=["pick", "read", "both"], default="both")
    parser.add_argument("--repeat", type=int, default=4,
                        help="samples per read case; the defect is stochastic, so >1")
    parser.add_argument("--json", type=Path, help="write raw rows here")
    args = parser.parse_args()

    settings, registry_specs = get_settings(), get_registry()
    if args.capture:
        return await capture(settings)

    client, raw = OllamaClient(settings), RawClient(settings)
    schemas = build_registry(settings).schemas({"basics", "web", "notes"}, allow_writes=True)
    rows: list[Row] = []
    try:
        if args.job in ("pick", "both"):
            arms = [Arm("350M", registry_specs.by_alias("general")),
                    Arm("functiongemma", FUNCTIONGEMMA, native=True)]
            print(f"pick: {len(PICK_CASES)} cases x {len(arms)} arms, {len(schemas)} schemas")
            picks: list[Row] = []
            for index, case in enumerate(PICK_CASES):
                order = arms[index % len(arms):] + arms[: index % len(arms)]
                marks = []
                for arm in order:
                    row = await run_pick(client, raw, arm, case, schemas)
                    picks.append(row)
                    marks.append(f"{arm.name} {'ok' if row.ok else 'X'}")
                print(f"  [{index + 1}/{len(PICK_CASES)}] {case.request[:38]:<40}{'  '.join(marks)}")
            report("pick — name the right tool", picks, [a.name for a in arms], read=False)
            rows += picks

        if args.job in ("read", "both"):
            cases = load_read_cases()
            writer = registry_specs.by_alias("instruct-q3")
            arms = [Arm(f"1.2B/{f}", writer, frame=f, temperature=0.6) for f in frames.FRAMES]
            arms.append(Arm("functiongemma", FUNCTIONGEMMA, native=True))
            print(f"\nread: {len(cases)} cases x {len(arms)} arms x {args.repeat} samples")
            reads: list[Row] = []
            for index, case in enumerate(cases):
                for sample in range(args.repeat):
                    # Rotate on every sample, not every case: with repeats the run is long
                    # enough that a fixed order would let the CPU heat correlate with arm.
                    turn = (index * args.repeat + sample) % len(arms)
                    marks = []
                    for arm in arms[turn:] + arms[:turn]:
                        row = await run_read(client, raw, arm, case, schemas)
                        reads.append(row)
                        marks.append(f"{arm.name} {'ok' if row.ok else 'X'}")
                    print(f"  [{index + 1}/{len(cases)}.{sample + 1}] {case.name:<9} "
                          f"{'  '.join(marks)}")
            report("read — state the fact the tool returned", reads,
                   [a.name for a in arms], read=True)
            rows += reads
    finally:
        await client.aclose()
        await raw.aclose()

    if args.json:
        args.json.write_text(json.dumps([r.__dict__ for r in rows], indent=2), encoding="utf-8")
        print(f"\nraw rows -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
