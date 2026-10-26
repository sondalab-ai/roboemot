"""Shared helpers of the realtime tests: a route, fast client settings, a wait loop."""

from __future__ import annotations

import time
from collections.abc import Callable

from roboemot.config.schema import LLMRoute, RealtimeRouteConfig
from roboemot.realtime import protocol

KEY = "sk-SENTINELKEY0123456789"
UPDATE = protocol.session_update(
    instructions="be kind",
    input_rate_hz=16000,
    output_rate_hz=None,
    voice="v",
    vad_threshold=0.5,
    silence_ms=500,
    create_response=False,
    interrupt_response=False,
)


def route(base_url: str, *, locality: str = "local", keyed: bool = False, **kw: object) -> LLMRoute:
    return LLMRoute(
        name="rt",
        base_url=base_url,
        model="s2s",
        kind="realtime",
        locality=locality,
        api_key="${secret:rt_key}" if keyed else "none",
        **kw,
    )


def settings(**kw: float) -> RealtimeRouteConfig:
    """Below the schema's floors, so the tests run fast."""
    values = {"connect_timeout_s": 2.0, "send_timeout_s": 0.5, "max_message_bytes": 65_536, **kw}
    return RealtimeRouteConfig.model_construct(**values)


def bearer() -> dict[str, str]:
    return {"Authorization": f"Bearer {KEY}"}


def wait_until(cond: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()
