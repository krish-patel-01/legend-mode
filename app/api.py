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

from app import guardrails
from app.backends.ollama import OllamaError
from app.persona import DISPUTE_NOTE, ensure_system_prompt
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
    """Show what the cascade would pick, without generating anything."""
    engine = _engine(request)
    req = RouteRequest(text=prompt, message_count=1)
    decision = await engine.route(req)
    return decision.as_meta() | {"scores": decision.scores}


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

    options = _openai_options(body, spec)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    lock = engine.lock_for(spec)

    history = _history(request)

    # Routing (`req`) was computed from the caller's original messages above; the
    # persona system prompt is added only for generation, so it never influences
    # which tier gets picked.
    settings = request.app.state.settings
    chat_messages = ensure_system_prompt(messages, settings.assistant_name, spec.persona)

    # Deterministic grounding (app/guardrails.py). Injected before generation rather
    # than checked afterwards: the model then phrases a correct fact naturally instead
    # of producing prose that has to be parsed and patched.
    grounding = guardrails.ground(req.text)
    if grounding is not None:
        decision.grounded = grounding.kind
        chat_messages = _append_system(chat_messages, guardrails.as_system_note(grounding))

    if decision.followup in {"dispute", "weak_dispute"}:
        chat_messages = _append_system(chat_messages, DISPUTE_NOTE)

    if body.get("stream"):
        return StreamingResponse(
            _stream(
                client, spec, chat_messages, tools, options, decision, completion_id, lock,
                history=history, prompt=req.text,
            ),
            media_type="text/event-stream",
            headers={"X-Legend-Route": _route_header(decision)},
        )

    async def _run() -> dict[str, Any]:
        if lock:
            async with lock:
                return await client.chat(spec, chat_messages, tools=tools, options=options)
        return await client.chat(spec, chat_messages, tools=tools, options=options)

    try:
        result = await _run()
    except OllamaError as exc:
        history.add(prompt=req.text, tag=spec.tag, decision=decision, error=str(exc))
        raise HTTPException(502, str(exc)) from exc

    # A thinking model that spends its whole budget reasoning returns content="" with
    # the answer never emitted. Empty output reads as a broken server, and sticky
    # routing made it reachable by sending terse follow-ups to the reasoning tier —
    # "nope" against a hard problem is exactly the shape that runs the budget out.
    # Saying so is worse than a real answer but better than silence.
    message = result.get("message") or {}
    if not (message.get("content") or "").strip() and not message.get("tool_calls"):
        log.warning(
            "%s produced no content (%s tokens); returning a fallback",
            spec.alias, result.get("eval_count"),
        )
        message["content"] = (
            "I ran out of thinking room before I finished that one. "
            "Ask me for a specific part of it and I'll work through that."
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

    history.add(
        prompt=req.text, tag=spec.tag, decision=decision,
        completion_tokens=result.get("eval_count"),
    )
    return _to_openai_response(result, spec.tag, completion_id, decision)


# Some GGUFs (observed with the unsloth Qwen3.5 line) ramble unboundedly on CPU when
# no length cap is set — one "hi" produced 1056 tokens of looping, mixed-language
# output. Without a default, num_predict falls back to Ollama's own default (often
# unbounded up to num_ctx), which turns a one-line reply into a minute-plus generation.
# The cap is per-model (see ModelSpec.default_max_tokens) since reasoning tiers need
# more headroom than a quick chat reply. Callers can still ask for more via max_tokens.
def _openai_options(body: dict[str, Any], spec) -> dict[str, Any]:
    options: dict[str, Any] = {"num_predict": spec.default_max_tokens}
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
