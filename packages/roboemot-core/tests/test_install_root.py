"""SPEC 4A, Installer: root over SSH for the overclock (``Remote.run_root``) never handles the
sudo password, and the reboot wait is bounded (robot-agnostic, subprocess faked)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from roboemot.install import overclock as oc_mod
from roboemot.install import remote as remote_mod
from roboemot.install.remote import Remote
from roboemot.install.ui import Ui


class Calls:
    def __init__(self, rc: int = 0, stderr: str = "") -> None:
        self.rc, self.stderr, self.seen = rc, stderr, []

    def __call__(self, args, **kw):
        self.seen.append((args, kw))
        return subprocess.CompletedProcess(args, self.rc, "", self.stderr)


@pytest.fixture()
def calls(monkeypatch):
    fake = Calls()
    monkeypatch.setattr(remote_mod.subprocess, "run", fake)
    return fake


def remote() -> Remote:
    r = Remote("pollen", "robot", Path("/tmp/x"))
    r.connected = True
    return r


def test_on_a_terminal_sudo_gets_the_terminal_and_nothing_is_captured_or_sent(calls):
    assert remote().run_root("rm -f /boot/x", tty=True) == 0
    ((args, kw),) = calls.seen
    assert "-t" in args and kw == {}  # stdin, stdout and stderr stay the owner's terminal
    command = args[-1]
    assert command.startswith("sudo -p ") and "sh -c 'rm -f /boot/x'" in command
    assert "-S" not in command.split()  # sudo never reads a password from stdin


def test_without_a_terminal_sudo_never_asks(calls):
    remote().run_root("true", tty=False)
    ((args, kw),) = calls.seen
    assert "-t" not in args and args[-1].startswith("sudo -n ")
    assert kw["stdin"] is subprocess.DEVNULL and "input" not in kw


@pytest.mark.parametrize(
    ("rc", "stderr", "up"),
    [
        (0, "", True),
        (255, "pollen@robot: Permission denied (publickey,password).", True),
        (255, "Connection refused", False),
    ],
)
def test_reachable_counts_a_password_prompt_as_up(calls, rc, stderr, up):
    calls.rc, calls.stderr = rc, stderr
    assert remote().reachable() is up
    ((args, _),) = calls.seen
    assert "ControlPath=none" in args and "BatchMode=yes" in args


class DownRobot:
    """A robot that goes down on the reboot and never answers again."""

    def __init__(self) -> None:
        self.connected = True
        self.polls = 0

    def run(self, command, check=True, stdin=None):
        assert command == oc_mod.BOOT_ID
        return "boot-1"

    def run_root(self, script, tty):
        return 255

    def disconnect(self):
        self.connected = False

    def reachable(self):
        self.polls += 1
        return False


def test_the_reboot_wait_ends_at_the_timeout(monkeypatch):
    clock = iter(range(0, 10**6, 10))
    monkeypatch.setattr(oc_mod.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setattr(oc_mod.time, "sleep", lambda s: None)
    robot = DownRobot()
    assert oc_mod.reboot_and_wait(Ui(False, False), robot, False, "reboot") is False
    assert 0 < robot.polls <= oc_mod.BOOT_TIMEOUT_S / 10 + 1 and not robot.connected
