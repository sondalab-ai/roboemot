"""Reachy Mini app wrapper (SPEC 4A, Packaging as a Reachy Mini app).

The daemon starts this as a subprocess and passes a connected ``ReachyMini``; this process is
the SDK bridge (the only owner of the SDK client and its media). It never wakes or sleeps the
robot: the platform does. The app is a thin adapter: it loads the config and the manifest, looks
the robot platform up by the manifest ``robot.id`` (entry-point group ``roboemot.robots``), binds
the connected robot to a session and hands both to core's ``Runtime``, which builds and runs the
whole pipeline until ``stop_event`` is set. The app is also the runtime's host (``RuntimeHost``):
the seams tests override and the parts the runtime publishes live on it.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from roboemot.config import ConfigStack
from roboemot.config.schema import RoboEmotConfig
from roboemot.manifest import load_manifest
from roboemot.robot_platform import load_robot_platform
from roboemot.runtime import Runtime, RuntimeHost

try:  # the SDK is optional so unit tests run without GStreamer
    from reachy_mini import ReachyMiniApp
except ImportError:  # pragma: no cover
    ReachyMiniApp = object  # type: ignore[misc,assignment]

DATA = Path(__file__).resolve().parent / "data"


class RoboEmotApp(RuntimeHost, ReachyMiniApp):  # type: ignore[misc,valid-type]
    # The enrollment and registry web UI (SPEC 4A, 6): the daemon reads this class attribute to
    # show the page; the port is webui.port (change both together). The app serves the page
    # itself (roboemot.webui), so the SDK must not start its own server on that port.
    custom_app_url: str | None = f"http://0.0.0.0:{RoboEmotConfig().webui.port}"
    dont_start_webserver: bool = True

    def run(self, reachy_mini: Any, stop_event: threading.Event) -> None:
        stack = ConfigStack.from_files(
            app=_optional(DATA / "app.yaml"), preset="default", preset_dirs=[DATA / "presets"]
        )
        cfg = stack.effective()
        # Unmeasured dynamics only when the owner enabled it in the app config (SPEC 12).
        manifest = load_manifest(
            DATA / "reachy-mini-wireless.yaml",
            allow_unmeasured_dynamics=cfg.safety.allow_unmeasured_dynamics,
            safety=cfg.safety,
        )
        platform = load_robot_platform(manifest.robot.id)
        platform.require_runtime()
        session = platform.bind(reachy_mini)
        Runtime(stack, manifest, platform, session, host=self).run(stop_event)


def _optional(p: Path) -> Path | None:
    return p if p.exists() else None
