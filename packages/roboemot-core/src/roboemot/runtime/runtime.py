"""The robot-agnostic runtime (SPEC 4A, 11, 12A).

``Runtime`` builds the whole pipeline (motion, affect, attention, vision, circadian, voice,
conversation, web UI, diagnostics) from the config stack, the manifest, the robot platform and the
session bound to the connected robot, runs it until ``stop_event`` is set and stops it within the
shutdown budget. It never imports a robot package: everything robot-specific enters through the
platform and the session (``roboemot.robot_platform``).
"""

from __future__ import annotations

import threading

from roboemot import shutdown
from roboemot.config import ConfigStack
from roboemot.manifest import Manifest
from roboemot.robot_platform import RobotPlatform, RobotSession
from roboemot.webui.robot_section import load_settings_plugin, supervisor_stop_timeout_s

from .assembly import assemble
from .diagnostics import build_diagnostics
from .host import RuntimeHost
from .loop import LoopClock, run_loop
from .reload import make_reload
from .startup import start_all


class Runtime:
    """One run of the robot-agnostic pipeline over one bound robot session. ``host`` is the
    owner's override seam and the place the running parts are published (default: a bare
    :class:`RuntimeHost`)."""

    def __init__(
        self,
        stack: ConfigStack,
        manifest: Manifest,
        platform: RobotPlatform,
        session: RobotSession,
        host: RuntimeHost | None = None,
    ) -> None:
        self.stack, self.manifest = stack, manifest
        self.platform, self.session = platform, session
        self.host = host if host is not None else RuntimeHost()

    def run(self, stop_event: threading.Event) -> None:
        """Build the parts, start them, tick until ``stop_event`` is set, then stop them in
        reverse order within the shutdown budget."""
        stack, manifest, host = self.stack, self.manifest, self.host
        p = assemble(host, stack, manifest, self.platform, self.session)
        host.diagnostics = build_diagnostics(host, stack, manifest, p)
        stack.subscribe(make_reload(host, p))
        stack.subscribe(p.worker.on_change)
        clock = LoopClock()  # idle breathing phase clock (SPEC 7A, Drowsy)
        runtime_cfg = stack.effective().runtime
        stop_budget_s = shutdown.capped_budget_s(
            runtime_cfg.shutdown_budget_s,
            supervisor_stop_timeout_s(load_settings_plugin(manifest.robot.id)),
            runtime_cfg.shutdown_margin_s,
        )
        with shutdown.BudgetedExitStack(stop_budget_s) as cleanup:
            start_all(host, stack, manifest, p, cleanup)
            run_loop(host, stack, p, clock, p.pitch_dof, stop_event)
