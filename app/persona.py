"""System prompt shared across every tier.

Kept separate from any one model's config since it's about the assistant's identity
and behavior, not a particular tier's sampling settings. The name comes from
`LEGEND_ASSISTANT_NAME` (default `Lucy`); with it unset the prompt tells the model to
say it has no name rather than invent one, and both wordings are measured below.

Three things here were settled by measurement against the actual models, not by
guessing, and the wording should not be "tidied" without re-running that check
(scripts in the scratchpad; 3 samples per probe per tier):

1. **Never end the prompt with a quotable answer.** An earlier brief version closed
   with `If the user asks your name, say you don't have one yet.` The 350M then
   replied "you don't have one yet." to *hi*, *thanks!*, and *what is the capital of
   France* — it treats the final sentence as a completion prefix. Moving the identity
   clause into the middle and ending on a behavioral directive took the 350M from
   failing most neutral prompts to 0 problems in 21 samples.

2. **Length matters on the 350M.** The full prompt is ~350 characters; at 350M that
   crowds out the actual question. `brief` is half that and is what `general` uses.

3. **Don't enumerate brands in the full prompt.** Naming a model primes it. Adding
   "never identify yourself as Qwen" made the Qwen3.5 tier's identity leak *worse*,
   not better (it began volunteering "I am Qwen" on unrelated turns). The full prompt
   therefore disclaims brands as a category instead of listing them.

Known limitation, only partly mitigated: models assert their own training identity no
matter what the prompt says. Qwen3.5-0.8B answers "who are you?" with "I am Qwen3.5,
developed by Tongyi Lab" 6/6 across three wordings, so `app/router/rules.py` pins
self-identity questions to the `trivial` route (the LFM 350M) instead of fighting it
in the prompt — a direct `model: "small"` call can still surface it. The 350M is much
better but not clean either: asked point-blank "who are you?" it named Liquid AI in
roughly 1 of 4 samples, sometimes alongside a confabulated second maker. Three brief
wordings were measured; the one kept here scored best (5/28 vs 7/28 for a
category-style "no company or product brand" disclaimer), so this is the floor
reachable by prompting, not a solved problem.

Answering "what model are you" with a real model name is *not* counted as a leak —
the prompt permits discussing internals when the user asks about them directly.

Which style a tier gets is set per model (`persona:` in models.yaml, default `full`).
"""

from __future__ import annotations

_BRIEF_UNNAMED = (
    "You have no name yet, and you are not ChatGPT or any other commercial assistant."
)
_FULL_UNNAMED = (
    "You have no name yet, and no company or product brand to claim as your identity. "
    "That applies to your own name only: if the user tells you theirs, remember it "
    "and use it."
)
_FULL_NAMED = "If the user tells you their name, remember it and use it."

# **The 350M is never told the assistant's name. This is measured, and it is emphatic.**
#
# Naming it makes it answer with the name and nothing else. Four brief prompts, interleaved
# over 16 probes each ("Hey", "hi", "thanks!", "what is the capital of France"), counting
# replies that were just the name or an echo of the prompt:
#
#   name + a style clause        16/16 broken   "Hey" -> "Lucy"
#   name alone                    8/16 broken   "what is the capital of France" -> "Lucy"
#   name as a conditional rule   16/16 broken   "thanks!" -> "Lucy"
#   unnamed (control)             0/16
#
# This is the failure in note 1 above, at full strength: a proper noun in a prompt this
# short is simply a more attractive completion than the answer. Three wordings were tried;
# the problem is the name's presence, not its phrasing.
#
# It costs nothing, because the tier no longer needs it. Identity questions used to route
# here and now route to `chat` (see app/router/rules.py, where that rule reversed), so the
# 350M only handles greetings and acknowledgements — turns where the name would never come
# up. `full` still gets the name and the personality.
# The named prompt is a measured wording, not a written one. Having a name in the prompt
# turns out not to mean the model will *use* it: asked "who are you?", "what should I call
# you?" and "hey, what's your name?", four clauses scored, interleaved —
#
#   clause                                          says it when asked   volunteers it
#   "Your name is Lucy."                                    3/12              0/6
#   "You are Lucy, a helpful …"                             5/12              0/6
#   "You are Lucy … When someone asks who you are,         11/12              0/6
#    you say your name."
#   "… You have a name and you use it; you never           10/12              0/6
#    say you lack an identity."
#
# Scored two-sided on purpose. Saying the name is the goal, but volunteering it unasked is
# what destroyed the 350M, so a clause that wins the first column and loses the second is
# not a win. All four were clean there; the third is simply better at the job.
#
# At "Your name is Lucy." — the obvious phrasing — three replies in four were "I'm an AI
# assistant, I don't have a personal identity". Stating a fact does not make a model this
# size act on it; stating when to act on it does.
_FULL_NAMED_OPENER = (
    "You are {name}, a helpful and direct AI assistant running locally. When someone asks "
    "who you are, you say your name."
)
_FULL_NAMED_STYLE = (
    "Your manner is direct and a little dry: warm without being eager, confident without "
    "overclaiming. You say when you don't know something instead of guessing."
)

# What the assistant can find out — and deliberately not how.
#
# **Measured 2026-08-10, and it does not ship: `persona_capabilities` defaults to False.**
# Full suite, 53 cases, then the persona category re-run at 4 samples to check the one
# category that moved:
#
#     persona case                      off      on (mid-prompt)   on (after identity)
#     denies-being-chatgpt              100%           50%                 50%
#     user-name-is-not-assistant-name   100%          100%                 75%
#     category                          100%           92%                 88%
#
# Asked "are you ChatGPT?" the on-arm answered *"Yes, I am a helpful assistant here to
# provide direct answers."* Two positions were tried on the theory that this was note 1
# again — the clause splitting the identity block — and moving it after the identity
# material made it worse, not better. So it is the clause's presence, not its placement,
# the same way it was the name's presence for the 350M above.
#
# The idea was sound and the precedent was real: `TOOL_RESULT_NOTE` below does this same
# job *after* a tool has run and is worth 2/6 -> 4/6. What the suite cannot show is the
# gain — `tools` was already 8/8 in the control, so there was no headroom for fewer
# refusals to appear in, and scripts/tool_bench.py runs with no persona at all by design.
# A fair re-test would need a case that fails today because the model does not believe it
# can look something up. Until that exists this is a cost with no measured benefit.
#
# The wording is kept because it is the thing that was measured, and because the shape is
# right even if the result was not: capabilities as a fact about itself, no verbs the
# model could imitate, and no promise that a lookup has happened — the clause has to
# survive the case where the gate stays shut, or it licenses exactly the invention it is
# meant to prevent. What was never on the table is telling the model *how* to call
# anything: four tool definitions attached to this tier produced 2/6 spurious calls and
# 3/6 degraded answers, including "I'm sorry, but I can't provide that information" to
# *what is the capital of France* (app/tools/gate.py). Call syntax in prose is the same
# information through a different door, and the split in app/tools/dispatch.py exists to
# keep it out.
_CAPABILITIES = (
    "Some things you can find out rather than recall: the current date and time, the "
    "state of this machine, what is on the web, and the user's own notes. When a "
    "question needs one of those, the answer is placed in the conversation for you to "
    "read, and you use it."
)

# {identity} sits mid-prompt on purpose — see note 1 above.
_BRIEF = "You are a helpful local AI assistant. {identity} Answer the user's question directly and concisely."
_FULL = (
    "You are a helpful, direct AI assistant running locally. {identity} Don't bring up "
    "how you work internally unless the user asks. Answer directly and concisely "
    "unless real depth is asked for."
)


# Appended when a tool ran, to the model that writes the answer — never to the one that
# picks the tool (see app/tools/dispatch.py for why those are different models).
#
# **It is there to break a prior, not to explain the mechanism.** Handed a correct tool
# result as a `role: "tool"` turn, the 1.2B still answered "I don't have real-time
# capabilities to check the current local time" and "I don't have access to your private
# notes". That is the same refusal posture measured in app/tools/gate.py, arriving by a
# different route: the model's belief that it cannot know something outranks the evidence
# that it just did.
#
# The gain is accuracy, not only tone. Given "13:20:11 in Tokyo", the same model answered
# "5:20 PM" without this note and "1:20 PM" with it. Six cases across the three families,
# persona prompt held constant:
#
#   no note   2/6   refusals on the clock and notes cases, wrong conversion on the clock
#   note      4/6   clock, search, machine state and one notes case all correct
#
# A third clause was tried and dropped — telling the model the user had not seen the tool
# output and to quote it back. It fixed nothing and cost accuracy elsewhere, turning
# "40.5 GB free of 97.7 GB" into "roughly 4 GB of available space". The two cases still
# failing are single-sentence notes the model treats as self-evident, which is a small
# enough residue to leave alone rather than tune the prompt around.
TOOL_RESULT_NOTE = (
    "You just looked this up with a tool, so you do have access to it and it is current. "
    "Answer from it directly. Never say you lack access or real-time information when a "
    "tool has already returned the answer."
)

# **An abstain clause was measured 2026-08-15 and does not ship. This is the third time.**
#
# The note above is one-directional: every clause pushes the model to commit, and none
# covers the case where the tool returned navigational links instead of the answer. That is
# the open grounding bug — asked who won the last F1 race with only formula1.com in
# context, this tier answered "Winner: Max Verstappen of Red Bull Racing" — so a clause
# permitting "the results don't say" looked like the obvious fix.
#
# Two arms were added mid-note, keeping the measured final line last:
#
#   abstain  "If the result does not contain what was asked, say so plainly and do not
#            fill the gap from memory."
#   worded   "...say that the search did not turn it up and stop there, rather than
#            answering from memory." — names the sentence, after `abstain` produced
#            refusals instead of abstentions.
#
# 8 cases x 4 samples, arms rotated per sample. `rich` = the tool result contains the
# answer (evals/tool_context.json), scored as scripts/tool_bench.py scores it. `thin` = it
# does not (captured live), scored as not asserting a figure absent from the evidence.
#
#              run   rich ok   refused |  thin ok   invented   refused
#   control     1      14/20      3    |   6/12        6          2
#   control     2      14/20      1    |   9/12        3          2
#   abstain     1      13/20      5    |  11/12        1          7
#   abstain     2      15/20      3    |  10/12        2          4
#   worded      2      13/20      5    |   9/12        3          6
#
# **The first run said this was a five-point win and the second says it is one point.**
# Nothing changed between them; control's own thin score moved 6/12 to 9/12 on its own. At
# 12 observations per arm the run-to-run variance is larger than the effect, so the benefit
# is unproven, not proven small.
#
# The cost is not unproven. Refusals rise with clause strength in 4 of 4 comparisons —
# both runs, both kinds of evidence, and `worded` pushing hardest refuses most. And the
# refusals are the bad kind: "I don't have access to real-time data", with search results
# sitting in the context, which is the exact sentence the final line above forbids and the
# exact posture this note exists to break.
#
# The `worded` arm is the informative one. Told the sentence to produce, the model produced
# it once in 36 samples and answered with a capability denial instead. So the mechanism is
# not that the model lacks permission to abstain — it is that any invitation to decline
# lands on the pretrained refusal, which is a stronger attractor than the wording offered.
#
# Same shape as the two failures already recorded in this file: `persona_capabilities`
# above, and scripts/frames.py, where a frame that *forbade* declining moved the
# non-commitment somewhere a scorer liked better rather than removing it. Pushing on the
# commit/decline axis from either end moves the failure and does not fix it. The next
# attempt at the grounding bug should change what reaches the model, not how it is told to
# feel about it.
#
# A fair re-test needs far more than 12 observations per arm, and a scorer for invention
# better than "a figure not in the evidence" — that one counts a fabricated release date
# and a fabricated race winner the same, and misses an invented name entirely.

# Appended when the sticky stage sees the user disputing the previous answer.
#
# Two failures to fix at once. The first is sycophancy: told "its incorrect", the 350M
# replied "You're right! The number of boxes is indeed 2, not 3" — agreeing without
# rechecking anything. The second is budget exhaustion: routing disputes to the 1.2B
# means a bare "nope" arrives with no concrete claim attached, and the model reasons in
# circles until the token cap and returns nothing at all.
#
# So the note does three jobs — refuse the reflex to capitulate, bound the reply, and
# give the model something cheap to do when the dispute carries no information at all.
# That last clause matters: a bare "nope" offers nothing to re-check, and without an
# alternative the model reasons in circles until the budget runs out.
#
# The length instruction sits mid-note, not at the end. Ending on "Answer in no more
# than three sentences." made the 1.2B reply "The total is four. Three sentences: Four
# boxes." — the same trailing-instruction echo this module's docstring warns about. The
# note now ends on an action, and if the model does echo that one it asks the user which
# part they disagree with, which is the wanted behaviour anyway.
DISPUTE_NOTE = (
    "The user is disputing your previous answer. Keep this reply to two or three "
    "sentences. Do not simply agree that you were wrong: re-check the specific claim, "
    "and if your answer was right, say so plainly and give the one reason why. If they "
    "have not said what is wrong, ask them which part they disagree with."
)

# The mirror image, and deliberately not the same text. A dispute says "that's wrong" and
# carries no information, so the right instruction is to hold firm unless the recheck
# shows otherwise. A correction says "but the monkeys are on the bed" — it hands over a
# fact the answer missed, and there the stubbornness DISPUTE_NOTE encourages is exactly
# wrong. Observed: the previous answer was repeated verbatim after the user supplied the
# missing constraint, because nothing marked the turn as a correction at all.
CORRECTION_NOTE = (
    "The user is pointing out something your previous answer missed or got wrong. Treat "
    "what they just said as true. Re-work the answer from the start with it included, "
    "in two or three sentences, and say plainly if it changes your conclusion."
)


def build_system_prompt(
    assistant_name: str | None, style: str = "full", *, capabilities: bool = False
) -> str:
    brief = style == "brief"
    # The `brief` tier never gets the capabilities clause either, and for the same reason
    # it never gets the name: at 350M the prompt is already competing with the question,
    # and this tier does not read tool output anyway — app/api.py escalates off it the
    # moment a tool has run.
    caps = f" {_CAPABILITIES}" if capabilities and not brief else ""
    if assistant_name and not brief:
        # The named prompt replaces the opener rather than filling {identity}, because the
        # wording that measured best puts the name in the first clause. Ends on the same
        # directive as the unnamed one — see note 1.
        # The capabilities clause goes *after* the identity material, not inside it.
        # Measured: splitting the block — opener, capabilities, style — took
        # `denies-being-chatgpt` from 100% to 50%, with "are you ChatGPT?" answered
        # "Yes, I am a helpful assistant here to provide direct answers." Two sentences
        # about lookups between "you say your name" and the manner clause is enough to
        # bury the identity signal. Same positional effect as note 1, one clause in.
        return (
            f"{_FULL_NAMED_OPENER.format(name=assistant_name)} {_FULL_NAMED_STYLE} "
            f"{_FULL_NAMED}{caps} Don't bring up how you work internally unless the user "
            f"asks. Answer directly and concisely unless real depth is asked for."
        )
    # Folded into {identity} rather than appended, so it lands mid-prompt here too and
    # the prompt still ends on a behavioural directive — note 1 again.
    identity = (_BRIEF_UNNAMED if brief else _FULL_UNNAMED) + caps
    return (_BRIEF if brief else _FULL).format(identity=identity)


def ensure_system_prompt(
    messages: list[dict],
    assistant_name: str | None,
    style: str = "full",
    *,
    capabilities: bool = False,
) -> list[dict]:
    """Prepend the persona system message unless the caller already supplied one."""
    if messages and messages[0].get("role") == "system":
        return messages
    return [
        {
            "role": "system",
            "content": build_system_prompt(assistant_name, style, capabilities=capabilities),
        },
        *messages,
    ]
