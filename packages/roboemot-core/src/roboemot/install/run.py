"""Start the installed app on a robot and say where its web UI is (SPEC 4A, Installer).

Runs on a PC on the robot's LAN. Robot-agnostic like the installer: the platform does
everything over the robot's own API, so no SSH connection is opened.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import Options, Platform, installed_platforms, report_app_url, select_robot
from .ui import Ui, WizardError

RUN_STEPS = 3


def run_start(opts: Options, ui: Ui, root: Path) -> None:
    """Find the robot, check the app is installed, start it and print its web UI address.
    With ``restart`` an app of ours already running is restarted through the platform's
    ``restart`` hook (anything else takes the start flow). With ``dry_run`` it only reports
    what it would do."""
    ui.steps = RUN_STEPS
    ui.title("run", "dry run: the robot is not modified" if opts.dry_run else "")
    ui.step("Robot")
    host, platform = select_robot(ui, opts, root)
    ui.ok(platform.display_name)
    platform.connect(ui, host)

    ui.step("App")
    if not platform.app_installed():
        raise WizardError(
            f"{platform.app_name} is not installed: run the installer first "
            "(python scripts/install_on_robot.py)"
        )
    ui.ok(f"{platform.app_name} is installed")

    ui.step("Start")
    if opts.dry_run:
        running = platform.running_app()
        app = platform.app_name
        if opts.restart and running == app:
            ui.info(f"{app} is running; would restart {app} (asks first)")
            blocker = _no_restart(platform) if platform.restart is None else None
        else:
            action = f"would stop {running}, " if running else "no app running; would "
            ui.info(f"{action}wake the robot and start {app}")
            blocker = None
        blocker = blocker or platform.start_blocker()
        if blocker:
            ui.warn(f"would refuse: {blocker}")
        url = platform.app_url(host)
        if url:
            ui.info(f"web UI would be at {url}")
        ui.title("dry run passed")
        return
    if opts.restart and platform.running_app() == platform.app_name:
        if platform.restart is None:
            raise WizardError(_no_restart(platform))
        platform.restart(ui)
    else:
        platform.start(ui)
    report_app_url(ui, platform, host)
    ui.title("running")


def _no_restart(platform: Platform) -> str:
    """Why ``--restart`` is refused on a platform whose ``restart`` is None. ``restart`` is read
    only for ``--restart``, so a platform written before it still runs and starts its app."""
    return f"{platform.display_name} cannot restart {platform.app_name} from here"


def parse_args(argv: list[str] | None = None) -> Options:
    p = argparse.ArgumentParser(description="Start RoboEmot on a robot (SPEC 4A).")
    p.add_argument("--host", help="robot hostname or IP (default: pick from robots found by mDNS)")
    p.add_argument(
        "--robot",
        help=f"manifest robot.id (installed: {', '.join(installed_platforms()) or 'none'})",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="report what would happen; nothing changes"
    )
    p.add_argument(
        "--restart",
        action="store_true",
        help="restart the app if it is already running (asks first: the head moves)",
    )
    p.add_argument("-y", "--yes", action="store_true", help="answer yes to every confirmation")
    a = p.parse_args(argv)
    return Options(host=a.host, robot=a.robot, dry_run=a.dry_run, yes=a.yes, restart=a.restart)


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    opts = parse_args(argv)
    ui = Ui(interactive=sys.stdin.isatty() and not os.environ.get("CI"), assume_yes=opts.yes)
    try:
        run_start(opts, ui, root or Path.cwd())
    except WizardError as e:
        ui.fail(str(e))
        return 1
    except KeyboardInterrupt:
        ui.fail("interrupted")
        return 130
    return 0


__all__ = ["main", "run_start"]
