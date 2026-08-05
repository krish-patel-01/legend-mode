"""What a tool is, and how the set of them is assembled.

Tools are grouped into *families* (`basics`, `files`, `web`, `notes`) because that is the
granularity everything else works at: the gate decides which families a request could
plausibly need, and the caller can disable a family outright. Nothing here decides whether
a tool runs — see `gate.py` for that, and `dispatch.py` for the loop.

`writes` is carried per tool rather than per family. A family is a topic; whether
something mutates state is a property of the individual call, and that distinction is what
lets a read-only mode exist without maintaining a second registry.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 4000
"""Cap on what a tool may hand back to a model.

A directory listing or a fetched page can be far larger than the context window, and the
failure mode is not an error — it is the model silently losing the question at the top of
the prompt, which is the same displacement effect measured in `scripts/cot_bench.py`.
Truncation is marked so the model can say the result was cut rather than assume it saw
everything.
"""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    """Written for the model, not for a human reader. It is the only thing the model has
    to decide with, so it says when to use the tool, not what the tool is."""

    parameters: dict[str, Any]
    """JSON Schema for the arguments, in the shape Ollama expects."""

    run: Callable[..., Any]
    """Sync or async; `ToolRegistry.invoke` handles both."""

    family: str
    writes: bool = False

    def schema(self) -> dict[str, Any]:
        """This tool in the OpenAI/Ollama `tools` array format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """What came back. `ok=False` still goes to the model — it needs to know it failed."""

    name: str
    content: str
    ok: bool = True
    truncated: bool = False
    elapsed_ms: float = 0.0

    def as_message(self) -> dict[str, Any]:
        """The `role: "tool"` turn appended to the conversation."""
        body = self.content
        if self.truncated:
            body += f"\n\n[truncated at {MAX_RESULT_CHARS} characters]"
        return {"role": "tool", "name": self.name, "content": body}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def families(self) -> set[str]:
        return {t.family for t in self._tools.values()}

    def for_families(self, families: set[str], *, allow_writes: bool = True) -> list[Tool]:
        return [
            t
            for t in self._tools.values()
            if t.family in families and (allow_writes or not t.writes)
        ]

    def schemas(self, families: set[str], *, allow_writes: bool = True) -> list[dict[str, Any]]:
        return [t.schema() for t in self.for_families(families, allow_writes=allow_writes)]

    async def invoke(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        """Run one tool. Never raises: a failure is a result the model has to see.

        A tool that raised and took the request down with it would make the whole loop
        brittle in exactly the place least worth being brittle — the model can recover
        from "that path does not exist" and cannot recover from a 500.
        """
        import time

        tool = self._tools.get(name)
        if tool is None:
            # Models invent tool names. Say which ones exist rather than only refusing.
            known = ", ".join(sorted(self._tools)) or "none"
            return ToolResult(name, f"No tool named {name!r}. Available: {known}.", ok=False)

        started = time.perf_counter()
        try:
            result = tool.run(**(arguments or {}))
            if inspect.isawaitable(result):
                result = await result
            text = result if isinstance(result, str) else repr(result)
            ok = True
        except TypeError as exc:
            # Almost always the model supplying wrong or missing arguments, which is a
            # recoverable mistake and worth naming precisely.
            text, ok = f"Bad arguments for {name}: {exc}", False
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - see docstring
            log.warning("tool %s failed: %s", name, exc)
            text, ok = f"{type(exc).__name__}: {exc}", False

        elapsed = (time.perf_counter() - started) * 1000
        truncated = len(text) > MAX_RESULT_CHARS
        return ToolResult(
            name, text[:MAX_RESULT_CHARS], ok=ok, truncated=truncated, elapsed_ms=elapsed
        )


def build_registry(settings: Any = None, memory: Any = None) -> ToolRegistry:
    """Assemble every available family. Imported here to keep optional deps optional."""
    from app.tools import basics

    registry = ToolRegistry(basics.tools())

    # The web family needs a running SearXNG and `trafilatura` installed. Neither is
    # guaranteed, and a missing one should cost the web tools rather than the process:
    # `basics` has no such dependency and must keep working regardless.
    try:
        from app.tools import web

        config = web.WebConfig(
            searxng_url=getattr(settings, "searxng_url", "http://127.0.0.1:8080"),
            timeout=getattr(settings, "web_timeout", 15.0),
            max_results=getattr(settings, "web_max_results", 5),
        )
        for tool in web.tools(config):
            registry.add(tool)
    except ImportError as exc:  # pragma: no cover - depends on the install
        log.warning("web tools unavailable: %s", exc)

    return registry
