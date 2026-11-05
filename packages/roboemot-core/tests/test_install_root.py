"""SPEC 4A, Installer: root over SSH for the overclock (``Remote.run_root``) never handles the
sudo password, and the reboot wait is bounded (robot-agnostic, subprocess faked)."""

from __future__ import annotations

import shutil
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
    assert command.startswith("/usr/bin/sudo -p ") and "/bin/sh -c 'rm -f /boot/x'" in command
    assert "-S" not in command.split()  # sudo never reads a password from stdin


def test_without_a_terminal_sudo_never_asks(calls):
    remote().run_root("true", tty=False)
    ((args, kw),) = calls.seen
    assert "-t" not in args and args[-1].startswith("/usr/bin/sudo -n /bin/sh -c ")
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


# ---------------------------------------------------------------- the root script, through /bin/sh


def bins(folder: Path, **override: str) -> dict[str, str]:
    """This machine's binaries for the script's absolute ones (``sha256sum`` through
    ``shasum`` where only that exists), with ``override`` in place of some."""
    out = {name: shutil.which(name) or "" for name in oc_mod.ROOT_BINS}
    if not out["sha256sum"]:
        wrapper = folder / "sha256sum"
        wrapper.write_text('#!/bin/sh\nexec shasum -a 256 "$@"\n')
        wrapper.chmod(0o755)
        out["sha256sum"] = str(wrapper)
    for name, body in override.items():
        tool = folder / f"fake-{name}"
        tool.write_text(f"#!/bin/sh\n{body}\n")
        tool.chmod(0o755)
        out[name] = str(tool)
    return out


def sh(script: str) -> int:
    return subprocess.run(["/bin/sh", "-c", script], capture_output=True).returncode


@pytest.fixture()
def boot(tmp_path):
    folder = tmp_path / "boot"
    folder.mkdir()
    (folder / "config.txt").write_text("a=1\n")
    return folder


def test_the_script_writes_and_deletes_and_leaves_no_temporary_file(boot, tmp_path):
    cfg, trial = str(boot / "config.txt"), str(boot / "tryboot.txt")
    (boot / "tryboot.txt").write_text("a=1\nb=2\n")
    changes = [(cfg, "a=1\n", "a=1\nb=2\n"), (trial, "a=1\nb=2\n", None)]
    assert sh(oc_mod.root_script(changes, bins(tmp_path))) == 0
    assert (boot / "config.txt").read_text() == "a=1\nb=2\n"
    assert sorted(f.name for f in boot.iterdir()) == ["config.txt"]


def test_new_text_travels_in_the_script_not_in_a_file(boot, tmp_path):
    script = oc_mod.root_script([(str(boot / "x"), None, "kernel=ok\n")], bins(tmp_path))
    assert "/tmp" not in script and "kernel=ok" not in script  # base64, decoded by root


def test_a_target_changed_since_it_was_read_exits_unchanged(boot, tmp_path):
    cfg = str(boot / "config.txt")
    (boot / "config.txt").write_text("a=1\nowner=edit\n")
    rc = sh(oc_mod.root_script([(cfg, "a=1\n", "a=1\nb=2\n")], bins(tmp_path)))
    assert rc == oc_mod.CHANGED_RC
    assert (boot / "config.txt").read_text() == "a=1\nowner=edit\n"
    assert sorted(f.name for f in boot.iterdir()) == ["config.txt"]


def test_a_target_that_appeared_exits_unchanged(boot, tmp_path):
    (boot / "tryboot.txt").write_text("foreign\n")
    rc = sh(oc_mod.root_script([(str(boot / "tryboot.txt"), None, "x\n")], bins(tmp_path)))
    assert rc == oc_mod.CHANGED_RC and (boot / "tryboot.txt").read_text() == "foreign\n"


def test_swapped_bytes_fail_the_hash_and_nothing_is_written(boot, tmp_path):
    """What a swapped staged file did before: other bytes than the ones checked on the PC.
    Here the decoder itself is replaced and answers attacker bytes; the hash check stops it."""
    evil = bins(tmp_path, base64="printf 'kernel=evil.img\\n'")
    rc = sh(oc_mod.root_script([(str(boot / "config.txt"), "a=1\n", "a=1\nb=2\n")], evil))
    assert rc == oc_mod.CHANGED_RC
    assert (boot / "config.txt").read_text() == "a=1\n"
    assert sorted(f.name for f in boot.iterdir()) == ["config.txt"]


def test_a_temporary_file_that_cannot_be_written_exits_unchanged(boot, tmp_path):
    boot.chmod(0o555)  # the failing cp of before: the target folder is read-only
    try:
        rc = sh(oc_mod.root_script([(str(boot / "config.txt"), "a=1\n", "b=2\n")], bins(tmp_path)))
    finally:
        boot.chmod(0o755)
    assert rc == oc_mod.CHANGED_RC and (boot / "config.txt").read_text() == "a=1\n"


def test_a_failed_rename_exits_partial(boot, tmp_path):
    failing = bins(tmp_path, mv="exit 1")
    rc = sh(oc_mod.root_script([(str(boot / "config.txt"), "a=1\n", "b=2\n")], failing))
    assert rc == oc_mod.PARTIAL_RC


def test_the_script_syncs_before_and_after_the_renames(tmp_path):
    lines = oc_mod.root_script([("/b/c", "x\n", "y\n")]).splitlines()
    sync = oc_mod.ROOT_BINS["sync"]
    mv = next(i for i, ln in enumerate(lines) if ln.startswith(oc_mod.ROOT_BINS["mv"]))
    assert lines[mv - 1].startswith(sync) and lines[mv + 1].startswith(sync)
    assert all(ln.split()[0].startswith(("/", "set", "if", "h=", "[", "printf")) for ln in lines)


class Robot:
    def __init__(self, rc: int) -> None:
        self.rc = rc

    def run_root(self, script, tty):
        return self.rc


@pytest.mark.parametrize(
    ("rc", "kind", "why"),
    [
        (oc_mod.CHANGED_RC, oc_mod.RootUnchanged, "unchanged"),
        (oc_mod.SUDO_DENIED_RC, oc_mod.RootUnchanged, "the robot is unchanged"),
        (oc_mod.PARTIAL_RC, oc_mod.RootPartial, "may be partly changed"),
        (255, oc_mod.RootPartial, "may be partly changed"),
    ],
)
def test_exit_codes_say_whether_anything_may_have_changed(rc, kind, why):
    with pytest.raises(kind) as e:
        oc_mod.root_write(Ui(False, False), Robot(rc), False, [("/b/c", "x\n", "y\n")])
    assert why in e.value.why


def test_a_check_only_change_pins_a_file_it_does_not_write(boot, tmp_path):
    cfg, trial = str(boot / "config.txt"), str(boot / "tryboot.txt")
    pinned = [(trial, None, "a=1\nb=2\n"), (cfg, "a=1\n", "a=1\n")]
    script = oc_mod.root_script(pinned, bins(tmp_path))
    assert f"{cfg}{oc_mod.NEW_SUFFIX}" not in script
    (boot / "config.txt").write_text("a=1\nowner=edit\n")  # changed since it was read
    assert sh(script) == oc_mod.CHANGED_RC and not (boot / "tryboot.txt").exists()
    (boot / "config.txt").write_text("a=1\n")
    assert sh(script) == 0 and (boot / "tryboot.txt").read_text() == "a=1\nb=2\n"


def test_a_missing_binary_stops_the_script_before_anything(boot, tmp_path):
    missing = {**bins(tmp_path), "base64": str(tmp_path / "no-such-base64")}
    rc = sh(oc_mod.root_script([(str(boot / "config.txt"), "a=1\n", "b=2\n")], missing))
    assert rc == oc_mod.MISSING_RC and (boot / "config.txt").read_text() == "a=1\n"


@pytest.mark.parametrize(
    ("rc", "what"),
    [
        (oc_mod.MISSING_RC, "A binary is missing on the robot"),
        (oc_mod.TOO_LONG_RC, "The root command could not run"),
    ],
)
def test_missing_binaries_and_unrunnable_commands_change_nothing(rc, what):
    with pytest.raises(oc_mod.RootUnchanged, match=what):
        oc_mod.root_write(Ui(False, False), Robot(rc), False, [("/b/c", "x\n", "y\n")])


def test_a_boot_config_too_large_for_the_command_is_refused_before_sudo():
    huge = "x=1\n" * (oc_mod.MAX_SCRIPT_BYTES // 4)
    with pytest.raises(oc_mod.RootUnchanged, match="too large"):
        oc_mod.root_write(Ui(False, False), Robot(0), False, [("/b/c", huge, huge + "y\n")])
