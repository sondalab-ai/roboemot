"""The wire codec a realtime client speaks through (SPEC 8, Realtime speech route).

A codec turns the route's client events (the OpenAI realtime subset of ``protocol``) into the
messages its server takes, and one server frame into the subset's ``ServerEvent`` types:

- ``encode(event) -> (messages, local)``: the messages to send (dicts, JSON-encoded by the
  client), and the events the codec answers itself (the client hands them out as it does
  the server's);
- ``decode(frame) -> events``: never raises; a bad frame is ``Malformed``.

:class:`OpenAIWire` is the identity over ``protocol.parse`` (Pollen and every
``openai_realtime`` route); ``gemini_wire.GeminiWire`` maps Gemini Live.
"""

from __future__ import annotations

from typing import Any, Protocol

from roboemot.realtime import protocol


class Wire(Protocol):
    """Thread contract: a codec has no lock. ``encode`` and ``decode`` run on the client's one
    loop thread (``encode`` inside the client's send, never in the caller's ``send``, which
    the route calls holding its own lock), and the local events ``encode`` returns are
    delivered from that loop after the send, never synchronously to the caller."""

    def encode(
        self, event: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[protocol.ServerEvent]]: ...

    def decode(self, frame: str | bytes) -> list[protocol.ServerEvent]: ...


class OpenAIWire:
    """The OpenAI realtime subset as is: each event goes unchanged, each frame is
    ``protocol.parse`` with the client's ``max_bytes`` cap."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes

    def encode(
        self, event: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[protocol.ServerEvent]]:
        """The event itself, and nothing answered locally."""
        return [event], []

    def decode(self, frame: str | bytes) -> list[protocol.ServerEvent]:
        """The one event ``protocol.parse`` reads from ``frame``."""
        return [protocol.parse(frame, self.max_bytes)]
