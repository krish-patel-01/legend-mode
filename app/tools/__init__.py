"""Tool calling.

The measurements that shaped this package are in `gate.py`. The short version: every
model here gets *worse* when tool definitions are attached, including on questions that
need no tool at all, so the design question is not "how do we call tools" — Ollama and
the GGUF chat templates handle that — but "how do we avoid attaching them".
"""

from app.tools.registry import Tool, ToolRegistry, build_registry

__all__ = ["Tool", "ToolRegistry", "build_registry"]
