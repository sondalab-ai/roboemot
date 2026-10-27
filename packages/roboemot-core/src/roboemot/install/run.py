"""Start the installed app on a robot and say where its web UI is (SPEC 4A, Installer).

Runs on a PC on the robot's LAN. Robot-agnostic like the installer: the platform does
everything over the robot's own API, so no SSH connection is opened.
"""

from __future__ import annotations

import os
import shlex
import sys
import traceback
from pathlib import Path

from . import Options, Platform, installed_platforms, report_app_url, select_robot
from .ui import HeaderParser, TermError, Ui, WizardError, app_version, print_header

try:
    from sondalab_palette import term
except ImportError:  # pragma: no cover - the robot imports this package and has no palette
    term = None

RUN_STEPS = 3
RUN_COMMAND = "python scripts/run_on_robot.py"


def run_start(opts: Options, ui: Ui, root: Path) -> None:
    """Find the robot, check the app is installed, start it and print its web UI address.
    With ``restart`` an app of ours already running is restarted through the platform's
    ``restart`` hook (anything else takes the start flow). With ``dry_run`` it only reports
    what it would do."""
    ui.steps = RUN_STEPS
    ui.title("run", "dry run, the robot is not modified" if opts.dry_run else "")
    ui.step("Robot")
    host, platform = select_robot(ui, opts, root)
    ui.ok("Selected the robot", f"{platform.display_name} at {host}")
    platform.connect(ui, host)

    ui.step("App")
    if not platform.app_installed():
        raise WizardError(
            f"{platform.app_name} is not installed",
            "run the installer first",
            "python scripts/install_on_robot.py",
        )
    ui.ok("Checked the app", f"{platform.app_name} is installed")

    ui.step("Start")
    if opts.dry_run:
        running = platform.running_app()
        app = platform.app_name
        if opts.restart and running == app:
            ui.would(f"Restart {app}", "it is running; asks first")
            blocker = _no_restart(platform) if platform.restart is None else None
        else:
            if running:
                ui.would(f"Stop {running}", "only one app runs at a time")
            ui.would(f"Start {app}", "after waking the robot")
            blocker = None
        blocker = blocker or platform.start_blocker()
        if blocker:
            ui.warn("Would refuse", blocker)
        url = platform.app_url(host)
        if url:
            ui.kv("Web UI", url)
        ui.ok("Dry run passed", "the robot was not modified")
        ui.summary()
        return
    if opts.restart and platform.running_app() == platform.app_name:
        if platform.restart is None:
            raise WizardError("Cannot restart", _no_restart(platform))
        platform.restart(ui)
    else:
        platform.start(ui)
    report_app_url(ui, platform, host)
    ui.ok("Running", f"{platform.app_name} on {host}")
    ui.summary()


def _no_restart(platform: Platform) -> str:
    """Why ``--restart`` is refused on a platform whose ``restart`` is None. ``restart`` is read
    only for ``--restart``, so a platform written before it still runs and starts its app."""
    return f"{platform.display_name} cannot restart {platform.app_name} from here"


def parse_args(argv: list[str] | None = None, plain: bool = False) -> Options:
    p = HeaderParser(description="Start RoboEmot on a robot (SPEC 4A).")
    p.plain = plain
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
    p.add_argument(
        "--debug", action="store_true", help="show raw tool output and tracebacks on errors"
    )
    p.add_argument("--version", action="store_true", help="print the version and exit")
    a = p.parse_args(argv)
    return Options(
        host=a.host,
        robot=a.robot,
        dry_run=a.dry_run,
        yes=a.yes,
        restart=a.restart,
        debug=a.debug,
        version=a.version,
    )


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    """Start the app; 0 on success, 1 on an error block, 130 on Ctrl-C. Same terminal output as
    the installer (``roboemot.install.main``): header, Term error block, ``--debug``."""
    args = sys.argv[1:] if argv is None else list(argv)
    vt = term is not None and all([term.enable_vt(sys.stdout), term.enable_vt(sys.stderr)])
    opts = parse_args(args, plain=not vt)
    if term is None:
        print(
            'The runner needs sondalab-palette: pip install -e "packages/roboemot-core[install]"',
            file=sys.stderr,
        )
        return 1
    if opts.version:
        mode = term.color_mode(isatty=sys.stdout.isatty()) if vt else "none"
        print(term.lockup("roboemot", app_version(), mode))
        return 0
    print_header(plain=not vt)
    ui = Ui(
        interactive=sys.stdin.isatty() and not os.environ.get("CI"),
        assume_yes=opts.yes,
        command=shlex.join([*RUN_COMMAND.split(), *args]),
        vt=vt,
        debug=opts.debug,
    )
    try:
        run_start(opts, ui, root or Path.cwd())
    except TermError as e:
        ui.error(e)
        return 1
    except KeyboardInterrupt:
        ui.error(WizardError("Interrupted", "stopped with Ctrl-C"))
        return 130
    except Exception as e:
        if opts.debug:
            traceback.print_exc()
        fix = "" if opts.debug else f"{ui.command} --debug"
        ui.error(WizardError(f"Unexpected error: {type(e).__name__}", str(e), fix))
        return 1
    return 0


__all__ = ["main", "run_start"]
