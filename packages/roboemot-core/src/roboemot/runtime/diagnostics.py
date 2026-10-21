"""The runtime's debug report sources (SPEC 12A): one section per part, counts and states only."""

from __future__ import annotations

import os
import time

from roboemot.config import ConfigStack
from roboemot.diagnostics import Diagnostics, FnSource, Target, platform_sources
from roboemot.diagnostics.system import local_sources
from roboemot.manifest import Manifest
from roboemot.vision.service import world_snapshot

from .assembly import Pipeline
from .host import RuntimeHost
from .reports import affect_report, register_report


def build_diagnostics(
    host: RuntimeHost, stack: ConfigStack, manifest: Manifest, p: Pipeline
) -> Diagnostics:
    """The registry of debug sources over the built parts, plus the robot's own sources
    registered under its ``robot.id``."""
    listener, bargein, affect = p.listener, p.bargein, p.runner.engine
    return Diagnostics().add(
        *local_sources(process_pid=os.getpid()),
        *platform_sources(manifest.robot.id, Target()),
        FnSource("roboemot.motion", p.loop.snapshot),
        FnSource("roboemot.network", p.network.snapshot),
        FnSource("roboemot.vision", p.perception.vision_snapshot),
        FnSource("roboemot.world", lambda: world_snapshot(p.world.entries, time.monotonic())),
        FnSource("roboemot.registry", p.perception.registry_snapshot),
        FnSource("roboemot.minor", p.presence.snapshot),
        FnSource("roboemot.register", lambda: register_report(p.tracker)),
        FnSource("roboemot.circadian", p.circ.snapshot),
        FnSource("roboemot.reflexes", p.reflexes.snapshot),
        FnSource(
            "roboemot.voice",
            lambda: (
                {
                    **listener.snapshot(),
                    **({"conversation": c.snapshot()} if (c := host.conversation) else {}),
                    **({"bargein": bargein.snapshot()} if bargein is not None else {}),
                }
                if listener is not None
                else {"state": "off"}
            ),
        ),
        FnSource(
            "roboemot.app",
            lambda: {
                "manifest": manifest.robot.id,
                "language": stack.effective().conversation.language,
                **affect_report(affect, time.monotonic()),
            },
        ),
    )
