"""SPEC 11A, 12, 12A: the assembled runtime carries the Live tab's levels and transcript ring. The
ring follows the flag on reload, is cleared when the robot falls asleep, and the real debug report
holds no transcript."""

from __future__ import annotations

import json
import threading
import time

from roboemot.config import ConfigStack
from roboemot.manifest import load_manifest
from roboemot.runtime import Runtime
from test_runtime import CONFIGS, DaytimeHost, RecordingPlatform, wait_for

SECRET = "tokenSECRETwiring"


def test_levels_and_transcript_are_wired_into_the_running_runtime():
    stack = ConfigStack.from_files(preset="default", preset_dirs=[CONFIGS / "presets"])
    stack.set("webui.live.enabled", True, source="test", admin=True)
    stack.set("webui.live.transcript", True, source="test", admin=True)
    manifest = load_manifest(
        CONFIGS / "manifests" / "desk-head-v1.yaml",
        allow_unmeasured_dynamics=True,
        safety=stack.effective().safety,
    )
    platform = RecordingPlatform()
    session = platform.bind(object())
    host = DaytimeHost()
    stop = threading.Event()
    thread = threading.Thread(
        target=lambda: Runtime(stack, manifest, platform, session, host=host).run(stop), daemon=True
    )
    thread.start()
    try:
        assert wait_for(lambda: host.diagnostics is not None and host.bus is not None)
        ring, levels = host.transcript, host.levels
        assert ring is not None and levels is not None and levels.reference is None  # no voice
        assert ring.add("user", SECRET, "en") == 1

        report = json.dumps(host.diagnostics.report(), default=str)  # the real SPEC 12A report
        assert SECRET not in report

        host.bus.publish("robot.asleep", None, time.monotonic())
        assert ring.size == 0  # sleeping forgets

        ring.add("user", SECRET, "en")
        stack.set("webui.live.transcript", False, source="test", admin=True)
        assert wait_for(lambda: ring.size == 0)  # the flag off clears it, via the reload
        assert ring.add("user", SECRET, "en") is None
    finally:
        stop.set()
        thread.join(timeout=30)
    assert not thread.is_alive()
