"""Realtime speech route (SPEC 8, Realtime speech route): optional, off by default.

``protocol`` builds and parses the subset of the OpenAI realtime protocol the route speaks,
``session`` holds the session-provider seam (``roboemot.realtime`` entry-point group) and the
URL and key rules, ``client`` is the WebSocket client. ``client`` needs the ``realtime`` extra
(``websockets``) and is imported on its own: nothing here imports it.
"""

from .session import (
    ENTRY_POINT_GROUP,
    Allocation,
    AllocationRequest,
    DirectSession,
    SessionProvider,
    load_session_provider,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "Allocation",
    "AllocationRequest",
    "DirectSession",
    "SessionProvider",
    "load_session_provider",
]
