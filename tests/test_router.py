"""Cascade tests against a stubbed backend — no Ollama process required.

The stub's embed() returns a deterministic bag-of-words vector so centroids behave
sensibly without needing a real embedding model, and its chat() returns a canned
label so the classifier stage is exercised too.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import ModelRegistry, ModelSpec, RouteTable, Settings, load_models, load_routes
from app.router import rules
from app.router.classifier import LlmClassifier
from app.router.embed import EmbeddingRouter
from app.router.engine import RouterEngine, anchor_text, extract_text, has_images
from app.router.types import RouteRequest
from app.config import ROOT


VOCAB_DIM = 64


def _bow_vector(text: str) -> list[float]:
    """Deterministic bag-of-hashed-words embedding, good enough for cosine similarity
    tests without loading a real model."""
    vec = np.zeros(VOCAB_DIM, dtype=np.float32)
    for word in text.lower().split():
        vec[hash(word) % VOCAB_DIM] += 1.0
    if not vec.any():
        vec[0] = 1.0
    return vec.tolist()


class StubClient:
    """Same surface as OllamaClient, backed by no process at all."""

    def __init__(self, classifier_label: str = "chat"):
        self.classifier_label = classifier_label
        self.chat_calls: list[str] = []

    async def embed(self, spec, texts):
        return [_bow_vector(t) for t in texts]

    async def chat(self, spec, messages, tools=None, options=None, think=None):
        self.chat_calls.append(spec.alias)
        return {"message": {"role": "assistant", "content": self.classifier_label}}

    async def preload(self, spec):
        pass

    async def tags(self):
        return {"legend/general", "legend/small", "legend/think", "legend/embed"}


@pytest.fixture
def registry() -> ModelRegistry:
    return load_models(ROOT / "models.yaml")


@pytest.fixture
def routes() -> RouteTable:
    return load_routes(ROOT / "routes.yaml")


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
async def engine(registry, routes, settings) -> RouterEngine:
    client = StubClient()
    eng = RouterEngine(client, registry, routes, settings)
    await eng.embedder.build()
    return eng


# --- stage 1: rules -----------------------------------------------------------------


async def test_forced_model_overrides_everything(engine):
    decision = await engine.route(RouteRequest(text="hi", forced_model="think"))
    assert decision.stage == "override"
    assert decision.model == "think"


async def test_every_route_resolves_to_a_registered_model(engine, registry):
    # routes.yaml and models.yaml drifted apart when the tools/vision tiers were
    # parked; this catches a route pointing at an alias that no longer exists.
    for route in engine._routes.routes:
        assert registry.get(route.model) is not None, (
            f"route {route.name!r} points at unknown model {route.model!r}"
        )


async def test_images_and_tools_no_longer_force_a_tier(engine):
    # The vision and tools routes are gone. These flags must not produce a decision
    # naming a route that routes.yaml no longer defines.
    known = {r.name for r in engine._routes.routes}
    for req in (
        RouteRequest(text="what is this", has_images=True, message_count=1),
        RouteRequest(text="do something", has_tools=True, message_count=1),
    ):
        decision = await engine.route(req)
        assert decision.route in known


async def test_opening_greeting_is_trivial(engine):
    decision = await engine.route(RouteRequest(text="hi there", message_count=1))
    assert decision.route == "trivial"
    assert decision.stage == "rules"


async def test_midconversation_ok_is_not_trivial(engine):
    # "ok" mid-thread usually means "continue", not "hello" — should not hit the
    # greeting rule and should fall through to a later stage.
    decision = await engine.route(RouteRequest(text="ok", message_count=4))
    assert decision.stage != "rules" or decision.route != "trivial"


async def test_explicit_reasoning_cue_routes_to_think(engine):
    decision = await engine.route(
        RouteRequest(text="think step by step about this proof", message_count=2)
    )
    assert decision.route == "think"
    assert decision.stage == "rules"


async def test_code_fence_routes_to_think(engine):
    decision = await engine.route(RouteRequest(text="fix this:\n```\nx=1\n```", message_count=2))
    assert decision.route == "think"


@pytest.mark.parametrize(
    "text",
    ["who are you?", "what's your name?", "what model are you", "who made you"],
)
async def test_identity_questions_go_to_the_tier_that_gets_them_right(engine, text):
    # This assertion used to be `trivial`, because Qwen3.5-0.8B insisted it was Qwen
    # regardless of prompt while the 350M answered cleanly. That tier is parked and the
    # 350M has since become the leaker: measured interleaved over 16 identity questions
    # each, 3/16 maker claims for the 350M (one of them inventing OpenAI) against 0/16
    # for the instruct build. The rule survives; its direction reversed.
    decision = await engine.route(RouteRequest(text=text, message_count=6))
    assert decision.route == "chat"
    assert decision.stage == "rules"


@pytest.mark.parametrize("text", ["what is my name?", "my name is Krish", "what are you doing"])
async def test_identity_rule_does_not_overmatch(engine, text):
    decision = await engine.route(RouteRequest(text=text, message_count=6))
    assert decision.reason != "asks about the assistant's own identity"


async def test_counting_word_problem_routes_to_think(engine):
    # The case that motivated the rule: the 350M chat tier answered "2", was told it
    # was wrong, and simply agreed with the correction.
    decision = await engine.route(
        RouteRequest(
            text="How many boxes do I have if I have two boxes with one box inside each?",
            message_count=2,
        )
    )
    assert decision.route == "think"
    assert decision.stage == "rules"


async def test_multi_quantity_question_routes_to_think(engine):
    decision = await engine.route(
        RouteRequest(text="how much is 15 percent of 240", message_count=2)
    )
    assert decision.route == "think"


@pytest.mark.parametrize(
    "text",
    ["how many days in a leap year", "how much is a stamp", "how long is the film"],
)
def test_plain_quantity_lookup_is_not_a_word_problem(text):
    # No scenario and only one quantity -> a lookup, not arithmetic. Must not burn the
    # 1.2B reasoning tier on it.
    #
    # Asserted against rules.apply() rather than the full engine on purpose. Routed end
    # to end this depends on the stub embedder's centroids, and those move whenever
    # routes.yaml gains examples — adding seven puzzle prompts to the `think` bank was
    # enough to flip this to `think` under the stub while the real bge embedder still
    # routed it to `chat`. The rule is what this test is actually about, and it is
    # deterministic. Real end-to-end routing for these lives in evals/cases.yaml.
    decision = rules.apply(RouteRequest(text=text, message_count=2))
    assert decision is None or decision.route != "think"


@pytest.mark.parametrize(
    "text",
    [
        "Which word comes next: Stone, Often, Canine, _: A Helpful B Freight C Glow D Grape",
        "Given a QWERTY keyboard layout, if HEART goes to JRSTY, what does AFTER go to?",
        "what comes next in this series",
        "find the odd one out",
        "which of these is right? A cat B dog C fish",
    ],
)
async def test_puzzles_route_to_think(engine, text):
    # Both of the first two reached the 350M via `fallback` in live testing and got
    # nonsense back ("AFTER goes to G"). Neither contains a reasoning keyword, a
    # quantity, or a code fence, so nothing in stage 1 saw them.
    decision = await engine.route(RouteRequest(text=text, message_count=1))
    assert decision.route == "think"


@pytest.mark.parametrize(
    "text",
    [
        "what's wrong with this code",         # "wrong" about the user's problem
        "explain what a REST API is",
        "a quick question about b trees and c",  # lowercase letters are ordinary prose
        "what is the capital of France",
    ],
)
async def test_puzzle_and_dispute_rules_do_not_overmatch(engine, text):
    decision = await engine.route(RouteRequest(text=text, message_count=1))
    assert decision.stage != "sticky"
    assert not (decision.stage == "rules" and decision.route == "think")


@pytest.mark.parametrize(
    "text",
    [
        "But the monkeys are on the bed",
        "actually they're on the bed",
        "wait, the bed has legs too",
        "you forgot the bed",
        "you didn't count the bed",
        "what about the bed",
    ],
)
async def test_correction_escalates_and_is_labelled(engine, text):
    # Observed live: "But the monkeys are on the bed" matched no follow-up pattern, so
    # no note was attached and the model repeated its original answer unchanged.
    decision = await engine.route(RouteRequest(text=text, message_count=4))
    assert decision.route == "think"
    assert decision.stage == "sticky"
    # The label drives which instruction gets attached; a correction must not be told to
    # hold its ground the way a dispute is.
    assert decision.followup == "correction"


async def test_but_is_not_a_correction_on_the_first_turn(engine):
    # Opening a conversation with "but..." is an ordinary question, not a correction.
    decision = await engine.route(RouteRequest(text="but what is a REST API", message_count=1))
    assert decision.followup != "correction"


async def test_bare_confusion_is_a_continuation(engine):
    # "What?" after a one-letter answer means "try that again", not a new question.
    decision = await engine.route(
        RouteRequest(text="What?", message_count=4, anchor_text=_BOX)
    )
    assert decision.route == "think"
    assert decision.stage == "sticky"


async def test_live_data_phrase_still_routes_somewhere_valid(engine):
    # Used to assert route == "tools". With no tools tier there is nothing correct to
    # assert beyond "it resolves"; the model will explain it has no live data access.
    known = {r.name for r in engine._routes.routes}
    decision = await engine.route(
        RouteRequest(text="what's the weather right now", message_count=2)
    )
    assert decision.route in known


# --- sticky follow-ups ----------------------------------------------------------------

_BOX = "How many boxes do I have if I have two boxes with one box inside each?"


@pytest.mark.parametrize(
    "text",
    [
        "its incorrect",
        "that's wrong",
        "thats not right",
        "are you sure",
        "wrong",
        # Found in live testing: anchoring the pattern to the start of the message meant
        # this matched nothing at all and the 350M answered "The answer is correct."
        "No the answer is wrong",
        "no, that's incorrect",
        "the answer is wrong",
        "you're wrong",
        "sorry, incorrect",
        "that result is false",
    ],
)
async def test_dispute_escalates_even_without_history(engine, text):
    # The transcript that started this: `think` answered the box problem correctly,
    # then "its incorrect" routed on its own as a three-word message to the 350M, which
    # agreed and invented a new wrong number. A dispute must never land on a tier that
    # cannot reason about the thing being disputed.
    decision = await engine.route(RouteRequest(text=text, message_count=3))
    assert decision.route == "think"
    assert decision.stage == "sticky"


async def test_weak_dispute_sticks_only_when_the_previous_turn_was_sticky(engine):
    stuck = await engine.route(
        RouteRequest(text="nope", message_count=4, anchor_text=_BOX)
    )
    assert stuck.route == "think"
    assert stuck.stage == "sticky"

    # Same word, ordinary chat thread: "no" is as likely to be answering a question the
    # assistant asked, so it must not drag a cheap thread onto the reasoning tier.
    loose = await engine.route(
        RouteRequest(
            text="nope", message_count=4, anchor_text="give me five names for a coffee shop"
        )
    )
    assert loose.stage != "sticky"


@pytest.mark.parametrize("text", ["why", "explain that", "show your work", "go on"])
async def test_continuation_stays_on_the_reasoning_tier(engine, text):
    decision = await engine.route(
        RouteRequest(text=text, message_count=4, anchor_text=_BOX)
    )
    assert decision.route == "think"
    assert decision.stage == "sticky"


@pytest.mark.parametrize("text", ["prove it", "walk me through it"])
async def test_continuations_already_covered_by_the_reasoning_rule(engine, text):
    # These carry an explicit reasoning cue of their own, so stage 1 claims them before
    # sticky ever runs. Same destination, cheaper path — assert the tier, not the stage.
    decision = await engine.route(
        RouteRequest(text=text, message_count=4, anchor_text=_BOX)
    )
    assert decision.route == "think"


@pytest.mark.parametrize(
    "text",
    [
        "thanks!",
        "ok got it",
        "explain what a REST API is",   # a fresh question, not a continuation
        "why does this recursive function overflow the stack",
        "what about the tax on top of that, does it apply before or after the discount",
    ],
)
async def test_sticky_does_not_capture_ordinary_messages(engine, text):
    decision = await engine.route(
        RouteRequest(text=text, message_count=4, anchor_text=_BOX)
    )
    assert decision.stage != "sticky"


async def test_rules_still_beat_sticky(engine):
    # An identity question mid-thread is an identity question, not a continuation.
    decision = await engine.route(
        RouteRequest(text="who are you?", message_count=6, anchor_text=_BOX)
    )
    # The tier moved from `trivial` to `chat` when identity routing reversed; what this
    # case guards is that the rules stage decided at all, rather than sticky inheriting
    # the reasoning tier from the thread.
    assert decision.stage == "rules"
    assert decision.route == "chat"
    assert decision.stage == "rules"


async def test_sticky_lookup_cannot_recurse(engine):
    # The previous turn is itself a dispute; resolving it must terminate.
    decision = await engine.route(
        RouteRequest(text="nope", message_count=6, anchor_text="that's wrong")
    )
    assert decision.route == "think"


# --- stage 2: embeddings --------------------------------------------------------------


async def test_ambiguous_prompt_falls_through_to_embed_or_classifier(engine):
    # No rule matches; must be resolved by stage 2 or 3, not silently dropped.
    decision = await engine.route(
        RouteRequest(text="write a short email asking for a deadline extension", message_count=2)
    )
    assert decision.stage in {"embed", "classifier", "fallback"}
    assert decision.route in {r.name for r in engine._routes.routes}


# --- stage 3: classifier --------------------------------------------------------------


async def test_classifier_used_when_embeddings_disabled(registry, routes, settings):
    client = StubClient(classifier_label="think")
    eng = RouterEngine(client, registry, routes, settings)
    # Do not build embedder centroids -> embed stage always defers.
    decision = await eng.route(RouteRequest(text="zzz completely novel gibberish qq", message_count=2))
    assert decision.stage == "classifier"
    assert decision.route == "think"
    assert client.chat_calls  # classifier actually ran


async def test_classifier_unusable_output_falls_back(registry, routes, settings):
    client = StubClient(classifier_label="not a real category at all")
    eng = RouterEngine(client, registry, routes, settings)
    decision = await eng.route(RouteRequest(text="zzz completely novel gibberish qq", message_count=2))
    assert decision.stage == "fallback"
    assert decision.route == settings.default_route


# --- caching ---------------------------------------------------------------------------


async def test_repeated_prompt_hits_cache(engine):
    req = RouteRequest(text="write a haiku about autumn leaves please", message_count=2)
    first = await engine.route(req)
    second = await engine.route(req)
    assert first.stage != "cache"
    assert second.stage == "cache"
    assert second.route == first.route


# --- helpers -----------------------------------------------------------------------


def test_anchor_text_skips_a_run_of_follow_ups():
    # The failure this fixes: with only the immediately-previous turn to go on, a
    # second "nope" sees a first "nope", which routes to the trivial tier in isolation,
    # and the thread silently falls off the reasoning tier.
    messages = [
        {"role": "user", "content": _BOX},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "its incorrect"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "nope"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "nope"},
    ]
    assert anchor_text(messages) == _BOX


def test_anchor_text_is_empty_on_the_first_turn():
    assert anchor_text([{"role": "user", "content": "hi"}]) == ""


def test_anchor_text_ignores_assistant_turns():
    messages = [
        {"role": "user", "content": "what is a monad"},
        {"role": "assistant", "content": "why is this hard"},  # looks like a follow-up
        {"role": "user", "content": "explain that"},
    ]
    assert anchor_text(messages) == "what is a monad"


def test_extract_text_prefers_last_user_message():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    assert extract_text(messages) == "second"


def test_extract_text_flattens_content_parts():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {}}]}]
    assert extract_text(messages) == "hello"


def test_has_images_detects_content_parts():
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    assert has_images(messages) is True


def test_has_images_false_for_plain_text():
    messages = [{"role": "user", "content": "just text"}]
    assert has_images(messages) is False


async def test_a_classifier_guess_of_trivial_escalates(registry, routes, settings):
    """Reaching stage 3 means rules found no signal and the embedding margin was thin.
    Everything `trivial` is for is caught by regex first, so a classifier verdict of
    `trivial` is the 350M guessing about a request that already looked unfamiliar.

    The live case: "please check in the web then answer that question" was labelled
    trivial here, and the 350M read "check in" as hotel check-in and invented a stay in
    Mexico City."""
    client = StubClient(classifier_label="trivial")
    eng = RouterEngine(client, registry, routes, settings)
    decision = await eng.route(RouteRequest(text="zzz completely novel gibberish qq", message_count=2))
    assert decision.stage == "classifier"
    assert decision.route == settings.default_route
    assert "guess" in decision.reason


async def test_the_classifier_still_decides_every_other_route(registry, routes, settings):
    """The escalation is aimed at one route, not at stage 3 generally — a classifier that
    says `think` has picked the expensive tier and there is nothing to second-guess."""
    client = StubClient(classifier_label="think")
    eng = RouterEngine(client, registry, routes, settings)
    decision = await eng.route(RouteRequest(text="zzz completely novel gibberish qq", message_count=2))
    assert decision.route == "think"
    assert "guess" not in decision.reason


async def test_the_cheap_guess_guard_can_be_turned_off(registry, routes, settings):
    client = StubClient(classifier_label="trivial")
    eng = RouterEngine(
        client, registry, routes, settings.model_copy(update={"escalate_classifier_trivial": False})
    )
    decision = await eng.route(RouteRequest(text="zzz completely novel gibberish qq", message_count=2))
    assert decision.route == "trivial"
