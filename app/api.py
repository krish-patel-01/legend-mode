"""OpenAI-compatible surface.

/v1/chat/completions accepts the standard request shape (messages, tools, stream,
model) and routes it through the cascade before forwarding to Ollama. `model` is a
routing policy, not a literal model name: "auto" (default) lets the cascade decide;
any known alias (router/small/think/large) pins that tier for the request.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app import adjudicate, effort, guardrails
from app.backends.ollama import OllamaError
from app.persona import CORRECTION_NOTE, DISPUTE_NOTE, ensure_system_prompt
from app.retrieval import service as retrieval_service
from app.router.engine import anchor_text, extract_text, has_images
from app.router.types import RouteRequest

log = logging.getLogger(__name__)
router = APIRouter()

_POLICY_ALIASES = {"auto", "route", ""}


def _engine(request: Request):
    return request.app.state.engine


def _client(request: Request):
    return request.app.state.client


def _history(request: Request):
    return request.app.state.history


def _retrieval(request: Request):
    return getattr(request.app.state, "retrieval", None)


@router.get("/v1/models")
async def list_models(request: Request) -> dict[str, Any]:
    registry = request.app.state.registry
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": spec.alias, "object": "model", "created": now, "owned_by": "legend-mode"}
            for spec in registry.models
        ],
    }


@router.get("/route/debug")
async def route_debug(request: Request, prompt: str) -> dict[str, Any]:
    """Show what the cascade would pick, without generating anything.

    Includes the effort plan, since which budget a prompt earns is now as much a tuning
    question as which model it lands on, and both are free to compute.
    """
    engine = _engine(request)
    settings = request.app.state.settings
    req = RouteRequest(text=prompt, message_count=1)
    decision = await engine.route(req)

    grounding = guardrails.ground(prompt)
    decision.grounded = grounding.kind if grounding else None
    decision.effort = effort.estimate(
        decision,
        text=prompt,
        tier_max_tokens=engine.spec_for(decision).default_max_tokens,
        grounded=grounding is not None,
        override=settings.default_effort,
    ).as_meta()
    return decision.as_meta() | {"scores": decision.scores}


@router.get("/retrieval/status")
async def retrieval_status(request: Request) -> dict[str, Any]:
    """What the corpus holds, so a silent empty index is visible rather than assumed."""
    store = _retrieval(request)
    settings = request.app.state.settings
    if store is None:
        return {"enabled": False, "chunks": 0, "sources": []}
    return {
        "enabled": settings.retrieval_enabled,
        "chunks": store.size,
        "sources": store.sources,
        "min_score": settings.retrieval_min_score,
        "top_k": settings.retrieval_top_k,
    }


@router.get("/route/history")
async def route_history(request: Request, limit: int = 50) -> dict[str, Any]:
    """Recent routing decisions across all callers, newest first. Powers the console UI."""
    return {"entries": _history(request).recent(limit=min(limit, 200))}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    messages: list[dict[str, Any]] = body.get("messages") or []
    if not messages:
        raise HTTPException(400, "messages must be a non-empty array")

    engine = _engine(request)
    client = _client(request)

    # No tier is vision-capable since the Qwen3.5-0.8B tier was parked. Saying so is
    # better than routing the request to a text-only model, which would answer
    # confidently about an image it never saw.
    if has_images(messages):
        raise HTTPException(
            422,
            "this server has no vision-capable model loaded; send text only "
            "(see the parked `small` tier in models.yaml to restore image support)",
        )

    requested = str(body.get("model") or "").strip()
    forced = None if requested.lower() in _POLICY_ALIASES else requested
    if forced and engine._registry.get(forced) is None:
        raise HTTPException(404, f"unknown model {requested!r}; see /v1/models")

    req = RouteRequest(
        text=extract_text(messages),
        has_tools=bool(body.get("tools")),
        has_images=False,  # image requests are rejected above, never routed
        forced_model=forced,
        message_count=len(messages),
        anchor_text=anchor_text(messages),
    )
    decision = await engine.route(req)
    spec = engine.spec_for(decision)

    tools = body.get("tools")
    if tools and not spec.tools:
        log.info(
            "route %r picked %s, which has no tool support; forwarding tools anyway "
            "so the caller sees an empty tool_calls rather than a silent drop",
            decision.route, spec.alias,
        )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    settings = request.app.state.settings
    history = _history(request)

    # Deterministic grounding (app/guardrails.py). Injected before generation rather
    # than checked afterwards: the model then phrases a correct fact naturally instead
    # of producing prose that has to be parsed and patched.
    grounding = guardrails.ground(req.text)
    if grounding is not None:
        decision.grounded = grounding.kind

    # How much this request is worth spending (app/effort.py). Computed from signals the
    # cascade already produced, so the estimate costs nothing and mainly decides the
    # token budget — which is the part that fixes a measured bug, not a nicety.
    # What to look up is not always what to answer. On a follow-up the turn itself is
    # "explain that", which retrieves nothing useful; the question the thread is about is
    # the anchor the sticky stage already found. Without this a retrieved thread loses its
    # source material on turn two, since the corpus text is injected per request rather
    # than kept in the conversation.
    retrieval_query = req.anchor_text if (decision.followup and req.anchor_text) else req.text

    try:
        plan = effort.estimate(
            decision,
            text=req.text,
            tier_max_tokens=spec.default_max_tokens,
            grounded=grounding is not None,
            override=str(body.get("effort") or settings.default_effort),
            retrieval_text=retrieval_query,
            thinking=spec.thinking,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # Retrieval (app/retrieval/). Two gates already stand in front of this: `plan.retrieve`
    # asks whether the prompt is a lookup at all, and `lookup` discards anything below the
    # similarity threshold. Both exist because injecting a merely-plausible passage
    # overrides knowledge the model had right — measured at -5.0 GPQA points in the paper
    # this design follows.
    #
    # Runs before the prompt is assembled because a hit can change which tier answers, and
    # the persona style is per-tier.
    found = None
    if plan.retrieve and settings.retrieval_enabled and (store := _retrieval(request)):
        found = await store.lookup(retrieval_query)
        decision.retrieved = found.citations if found else []

    if found is not None:
        # Reading a passage and answering strictly from it is a different skill from
        # recalling a fact, and the roadmap's premise — a grounded small model beating an
        # ungrounded large one — assumes the small model can read. Measured here: handed
        # the chunk that says "the verifier is always the 1.2B and never the 350M", the
        # 350M answered "The model that verifies answers is LFM2.5-350M". It inverted the
        # source. So a retrieval hit escalates to the tier that can actually read it.
        reader = engine.registry.get(settings.reader_alias)
        if reader is not None and reader.alias != spec.alias:
            log.info(
                "retrieval hit (%s); escalating %s -> %s to read it",
                ", ".join(decision.retrieved or []), spec.alias, reader.alias,
            )
            spec = reader
            decision.model = reader.alias
            decision.reason = f"{decision.reason}; escalated to {reader.alias} to read retrieved text"
            plan = effort.estimate(
                decision,
                text=req.text,
                tier_max_tokens=spec.default_max_tokens,
                grounded=grounding is not None,
                override=str(body.get("effort") or settings.default_effort),
                retrieval_text=retrieval_query,
                thinking=spec.thinking,
            )
    decision.effort = plan.as_meta()

    # Routing (`req`) was computed from the caller's original messages above; the
    # persona system prompt is added only for generation, so it never influences
    # which tier gets picked.
    chat_messages = ensure_system_prompt(messages, settings.assistant_name, spec.persona)

    if grounding is not None:
        chat_messages = _append_system(chat_messages, guardrails.as_system_note(grounding))

    if found is not None:
        chat_messages = _append_system(chat_messages, retrieval_service.as_system_note(found))

    if decision.followup in {"dispute", "weak_dispute"}:
        chat_messages = _append_system(chat_messages, DISPUTE_NOTE)
    elif decision.followup == "correction":
        chat_messages = _append_system(chat_messages, CORRECTION_NOTE)

    options = _openai_options(body, plan.max_tokens)

    async def _run(
        target=None, note: str | None = None, *, budget: int | None = None
    ) -> dict[str, Any]:
        """One generation. Also the repair path, hence the overridable model and note."""
        model = target or spec
        msgs = _append_system(chat_messages, note) if note else chat_messages
        opts = options if budget is None else _openai_options(body, budget)
        model_lock = engine.lock_for(model)
        if model_lock:
            async with model_lock:
                return await client.chat(model, msgs, tools=tools, options=opts)
        return await client.chat(model, msgs, tools=tools, options=opts)

    previous_reply = adjudicate.previous_assistant_reply(messages)
    critic_spec = engine.registry.get(settings.critic_alias)
    will_adjudicate = bool(
        (plan.guard_capitulation and previous_reply)
        or (plan.verify and settings.verify_enabled)
    )

    # A buffered "stream" when adjudication is planned. Adjudication can replace the
    # answer outright, and streaming tokens that may then be retracted is worse for the
    # reader than a pause: they would watch a wrong answer appear and then change. Total
    # latency is identical either way.
    if body.get("stream") and not will_adjudicate:
        return StreamingResponse(
            _stream(
                client, spec, chat_messages, tools, options, decision, completion_id,
                engine.lock_for(spec), history=history, prompt=req.text,
            ),
            media_type="text/event-stream",
            headers={"X-Legend-Route": _route_header(decision)},
        )

    try:
        result = await _run()
    except OllamaError as exc:
        history.add(prompt=req.text, tag=spec.tag, decision=decision, error=str(exc))
        raise HTTPException(502, str(exc)) from exc

    # A thinking model that spends its whole budget reasoning returns content="" with the
    # answer never emitted. Empty output reads as a broken server, so something honest
    # goes back instead.
    #
    # The wording matters, and an earlier version got it wrong. "I ran out of thinking
    # room" implies a larger budget would help. Measured, it does not: at 4096 tokens the
    # keyboard-substitution puzzle answered "S" after 215 s and the word-sequence puzzle
    # answered "A" after 342 s, both wrong, where 1536 produced nothing in ~135 s. More
    # budget bought a slower wrong answer. Meanwhile the box word problem finishes inside
    # 935 tokens. So exhaustion here means the question is past this tier's ability, not
    # that it was interrupted — and the reply should not invite a retry that will fail
    # the same way.
    message = result.get("message") or {}
    if not (message.get("content") or "").strip() and not message.get("tool_calls"):
        log.warning(
            "%s produced no content in %s tokens; likely past this tier's ability",
            spec.alias, result.get("eval_count"),
        )
        message["content"] = (
            "I worked through that but couldn't reach an answer I'd trust, so I'd rather "
            "say so than guess. A smaller piece of it is more likely to get somewhere."
        )
        result["message"] = message

    # Injecting a computed fact into the system prompt is not enough on its own. Measured
    # over 6 samples per case, the 350M overrode the supplied value on 5 of 6 discount
    # questions ("You pay $20", "You pay $15" against a grounded 30) and 5 of 6
    # temperature conversions ("100 / 32 = 3.125"). An earlier 2-sample check missed this
    # entirely, which is precisely why evals/ exists.
    #
    # So a contradicted numeric grounding is corrected rather than merely logged.
    # `contradicts()` only fires on numeric values it can compare exactly, so this
    # replaces a provably wrong number with a provably right one — it is not prose
    # judging prose. The claim text is already written as a complete sentence.
    if grounding is not None:
        answered = (result.get("message") or {}).get("content") or ""
        if guardrails.contradicts(answered, grounding):
            log.warning(
                "%s contradicted a %s grounding (wanted %r, said %r); substituting",
                spec.alias, grounding.kind, grounding.value, answered[:80],
            )
            result["message"] = {**(result.get("message") or {}), "content": grounding.claim}
            decision.grounded = f"{grounding.kind} (corrected)"

    # Adjudication (app/adjudicate.py). Runs last so it sees the answer the user would
    # otherwise have received, including any grounding substitution above — a computed
    # value is authoritative and nothing downstream should second-guess it. In practice
    # a grounded request is `fast` and never reaches here at all.
    if will_adjudicate:
        async def _regenerate(note: str | None, target) -> str:
            """The single repair. Returns "" on failure so the original answer stands."""
            budget = target.default_max_tokens if target is not None else None
            try:
                out = await _run(target, note, budget=budget)
            except OllamaError as exc:
                log.warning("repair generation failed: %s", exc)
                return ""
            return ((out.get("message") or {}).get("content") or "").strip()

        outcome = await adjudicate.run(
            question=req.text,
            answer=(result.get("message") or {}).get("content") or "",
            previous=previous_reply,
            plan=plan,
            answered_by=spec,
            critic_spec=critic_spec,
            client=client,
            settings=settings,
            regenerate=_regenerate,
        )
        decision.adjudicated = outcome.as_meta()
        if outcome.content:
            result["message"] = {
                **(result.get("message") or {}), "content": outcome.content
            }

    # Citations are appended here rather than requested from the model. They are known
    # exactly, so computing them is both cheaper and more reliable than asking a 1.2B to
    # format them — asked to, it replied with the citation and nothing else. Skipped when
    # the reply is the abstention or exhaustion message, which cites nothing.
    if found is not None and settings.retrieval_cite:
        answered = (result.get("message") or {}).get("content") or ""
        if answered.strip() and not answered.startswith("I worked through that"):
            result["message"] = {
                **result["message"],
                "content": f"{answered.rstrip()}\n\n{retrieval_service.as_citation_line(found)}",
            }

    history.add(
        prompt=req.text, tag=spec.tag, decision=decision,
        completion_tokens=result.get("eval_count"),
    )

    # An adjudicated request that asked for streaming was generated whole (see above), so
    # the "stream" it gets is one content chunk. Clients see a well-formed SSE response
    # either way; only the token-by-token effect is lost, and only when the answer was
    # liable to be replaced.
    if body.get("stream"):
        return StreamingResponse(
            _stream_prepared(result, spec.tag, completion_id, decision),
            media_type="text/event-stream",
            headers={"X-Legend-Route": _route_header(decision)},
        )
    return _to_openai_response(result, spec.tag, completion_id, decision)


# Some GGUFs (observed with the unsloth Qwen3.5 line) ramble unboundedly on CPU when
# no length cap is set — one "hi" produced 1056 tokens of looping, mixed-language
# output. Without a default, num_predict falls back to Ollama's own default (often
# unbounded up to num_ctx), which turns a one-line reply into a minute-plus generation.
#
# `max_tokens` now comes from the effort plan rather than straight from the model spec.
# The tier default (ModelSpec.default_max_tokens) is the ceiling the plan works within,
# not the number every request gets: a bare "nope" and a proof both used to receive 1536
# tokens on the reasoning tier, and the former burned the lot. Callers can still ask for
# more explicitly via max_tokens, which overrides the plan.
def _openai_options(body: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    options: dict[str, Any] = {"num_predict": max_tokens}
    if (t := body.get("temperature")) is not None:
        options["temperature"] = t
    if (p := body.get("top_p")) is not None:
        options["top_p"] = p
    if (mt := body.get("max_tokens") or body.get("max_completion_tokens")) is not None:
        options["num_predict"] = mt
    if (stop := body.get("stop")) is not None:
        options["stop"] = stop if isinstance(stop, list) else [stop]
    return options


def _append_system(messages: list[dict[str, Any]], note: str) -> list[dict[str, Any]]:
    """Append a note to the system message.

    Appended rather than prepended, and to the system turn rather than the user's, so
    the persona wording measured in app/persona.py keeps its position — that module's
    notes record that moving text around in these prompts changes small-model behaviour
    in ways unit tests can't see.
    """
    head, *rest = messages
    if head.get("role") != "system":
        return [{"role": "system", "content": note}, *messages]
    return [{**head, "content": f"{head['content']}\n\n{note}"}, *rest]


def _route_header(decision) -> str:
    return f"{decision.model};stage={decision.stage};route={decision.route}"


def _to_openai_response(
    result: dict[str, Any], model_tag: str, completion_id: str, decision
) -> dict[str, Any]:
    message = result.get("message", {"role": "assistant", "content": ""})
    finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_tag,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": {
            "prompt_tokens": result.get("prompt_eval_count", 0),
            "completion_tokens": result.get("eval_count", 0),
            "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
        },
        "x_legend_route": decision.as_meta(),
    }


async def _stream_prepared(
    result: dict[str, Any], model_tag: str, completion_id: str, decision
):
    """SSE-wrap an answer that was already generated in full.

    Used when adjudication was planned: the answer could have been replaced, so it is
    produced whole and then emitted as one chunk rather than streamed and retracted.
    """
    message = result.get("message") or {}
    delta: dict[str, Any] = {"role": "assistant"}
    if content := message.get("content"):
        delta["content"] = content
    if tool_calls := message.get("tool_calls"):
        delta["tool_calls"] = tool_calls

    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_tag,
    }
    yield "data: " + json.dumps(
        {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    ) + "\n\n"
    yield "data: " + json.dumps(
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                }
            ],
            "x_legend_route": decision.as_meta(),
        }
    ) + "\n\n"
    yield "data: [DONE]\n\n"


async def _stream(
    client, spec, messages, tools, options, decision, completion_id: str, lock,
    *, history, prompt: str,
):
    async def _chunks():
        first = True
        tokens = 0
        async for chunk in client.chat_stream(spec, messages, tools=tools, options=options):
            message = chunk.get("message", {})
            delta: dict[str, Any] = {}
            if first:
                delta["role"] = "assistant"
                first = False
            if content := message.get("content"):
                delta["content"] = content
            if tool_calls := message.get("tool_calls"):
                delta["tool_calls"] = tool_calls

            done = chunk.get("done", False)
            if done:
                tokens = chunk.get("eval_count", tokens)
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": spec.tag,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": (
                            ("tool_calls" if message.get("tool_calls") else "stop")
                            if done
                            else None
                        ),
                    }
                ],
            }
            if done:
                payload["x_legend_route"] = decision.as_meta()
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
        history.add(prompt=prompt, tag=spec.tag, decision=decision, completion_tokens=tokens)

    try:
        if lock:
            async with lock:
                async for item in _chunks():
                    yield item
        else:
            async for item in _chunks():
                yield item
    except OllamaError as exc:
        history.add(prompt=prompt, tag=spec.tag, decision=decision, error=str(exc))
        err = {"error": {"message": str(exc), "type": "upstream_error"}}
        yield f"data: {json.dumps(err)}\n\n"
