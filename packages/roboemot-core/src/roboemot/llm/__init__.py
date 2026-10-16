"""LLM routing over OpenAI-compatible endpoints (SPEC 8)."""

from .prompt import system_prompt
from .router import Done, NoRouteAvailable, Offline, Router, TextDelta, ToolCall, build_messages

__all__ = [
    "Done",
    "NoRouteAvailable",
    "Offline",
    "Router",
    "TextDelta",
    "ToolCall",
    "build_messages",
    "system_prompt",
]
