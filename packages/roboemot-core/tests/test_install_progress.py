"""SPEC 4A, Installer: live progress for long remote steps (streaming, filter, status line)."""

from __future__ import annotations

import io
import os
import re
import stat
import subprocess
import time
from pathlib import Path

import pytest
from roboemot.install.pyenv import RemotePython, pip_progress_line
from roboemot.install.remote import Remote
from roboemot.install.ui import Ui, WizardError, format_elapsed


class Lines:
    def __init__(self, lines: list[str]) -> None:
        self.it, self.closed = iter(lines), False

    def __iter__(self):
        return self.it

    def close(self) -> None:
        self.closed = True


class FakeProc:
    def __init__(self, lines: list[str], rc: int) -> None:
        self.stdout = Lines(lines)
        self.returncode = rc
        self.killed = False

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        return self.returncode


@pytest.fixture
def remote() -> Remote:
    return Remote("pollen", "robot", Path("/tmp/x"))


def patch_popen(monkeypatch, proc: FakeProc) -> list[list[str]]:
    calls: list[list[str]] = []

    def popen(cmd, **kw):
        calls.append(cmd)
        assert kw["stderr"] == subprocess.STDOUT and kw["stdout"] == subprocess.PIPE
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen)
    return calls


def test_stream_hands_lines_over_in_order_and_returns_everything(monkeypatch, remote):
    calls = patch_popen(monkeypatch, FakeProc(["Collecting a\n", "Installing b\n"], 0))
    seen: list[str] = []
    out = remote.stream("pip install", seen.append)
    assert seen == ["Collecting a", "Installing b"]
    assert out == "Collecting a\nInstalling b"
    assert calls[0][0] == "ssh" and calls[0][-2:] == ["pollen@robot", "pip install"]
    assert any("ControlPath=/tmp/x/%C" in a for a in calls[0])


def test_stream_raises_with_the_output_on_a_non_zero_exit(monkeypatch, remote):
    patch_popen(monkeypatch, FakeProc(["boom\n"], 3))
    with pytest.raises(WizardError, match="boom"):
        remote.stream("false", lambda line: None)
    patch_popen(monkeypatch, FakeProc(["boom\n"], 3))
    assert remote.stream("false", lambda line: None, check=False) == "boom"


@pytest.mark.parametrize(
    ("line", "shown"),
    [
        ("", None),
        ("   ", None),
        ("Requirement already satisfied: numpy in /x (2.0)", None),
        ("[notice] A new release of pip is available", None),
        ("__rc=0", None),
        ("Collecting reachy-mini==1.12.0", "Collecting reachy-mini==1.12.0"),
        ("  Downloading numpy-2.0.whl (14 MB)", "Downloading numpy-2.0.whl (14 MB)"),
        ("Successfully installed a-1 b-2", "Successfully installed a-1 b-2"),
    ],
)
def test_pip_progress_line(line, shown):
    assert pip_progress_line(line) == shown


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0s"), (42.9, "42s"), (102, "1m42s"), (3725, "1h02m05s")],
)
def test_format_elapsed(seconds, text):
    assert format_elapsed(seconds) == text


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_progress_without_a_terminal_prints_plain_lines(monkeypatch, capsys):
    ui = Ui(interactive=False, assume_yes=True)
    with ui.progress("installing", pip_progress_line, done="installed") as p:
        p("Collecting a")
        p("Requirement already satisfied: b")
        p("Installing collected packages: a")
    out = capsys.readouterr().out
    assert "\r" not in out and "\x1b" not in out
    assert "installing (can take minutes on the robot)" in out
    assert "Collecting a" in out and "Installing collected packages: a" in out
    assert "Requirement already" not in out
    assert re.search(r"✓ installed +0s\n", out)  # a done step line, the total time as detail


def test_progress_on_a_terminal_rewrites_one_line_then_clears_it(monkeypatch):
    # the running line is transient, so it goes to stderr (Term spec); the result to stdout
    out, err = Tty(), Tty()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("SONDALAB_REDUCED_MOTION", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr("shutil.get_terminal_size", lambda *a, **k: os.terminal_size((60, 24)))
    clock = Clock()
    ui = Ui(interactive=False, assume_yes=True)
    ui.clock, ui.tick_s = clock, 0.001
    with ui.progress("installing", pip_progress_line) as p:
        clock.now = 102
        p("Collecting " + "x" * 200)
        deadline = time.monotonic() + 5
        while err.getvalue().count("1m42s") < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
    text = err.getvalue()
    assert "\r" in text and "\x1b[K" in text
    assert "1m42s  Collecting x" in text and ("." if ui.ascii() else "…") + "\x1b[K" in text
    frames = [r.removesuffix("\x1b[K") for r in text.split("\r") if r.endswith("\x1b[K")]
    assert frames and max(map(len, frames)) <= 60
    assert text.endswith("\r\x1b[K")  # cleared before the result line
    assert "✓" not in text and "\r" not in out.getvalue()
    assert re.search(r"installing +1m42s\n", out.getvalue())


def test_progress_skips_the_check_mark_on_a_failure(capsys):
    ui = Ui(interactive=False, assume_yes=True)
    with ui.progress("installing") as p:
        p.failed = True
    assert "✓" not in capsys.readouterr().out
    with pytest.raises(RuntimeError), ui.progress("installing"):
        raise RuntimeError
    assert "✓" not in capsys.readouterr().out


class StreamRemote:
    def __init__(self, out: str) -> None:
        self.out, self.stdin, self.command = out, None, ""

    def stream(self, command, on_line, check=True):
        self.command = command
        for line in self.out.splitlines():
            on_line(line)
        return self.out

    def run(self, *a, **k):
        raise AssertionError("long pip calls must stream")


def test_a_failed_pip_keeps_its_full_output_in_the_error(capsys):
    ui = Ui(interactive=False, assume_yes=True)
    remote = StreamRemote("Collecting a\nERROR: no network\n__rc=1")
    py = RemotePython(remote, "/v/bin/python", ui)
    # the cause is the error block's why; the full output stays on the error (--debug prints it)
    with pytest.raises(WizardError, match="pip install failed on the robot") as e:
        py._pip("install a", announce="installing")
    assert e.value.why == "ERROR: no network"
    assert e.value.output.startswith("Collecting a\nERROR: no network")
    assert "✓" not in capsys.readouterr().out


def test_long_pip_drops_quiet_and_keeps_the_cache_flags():
    ui = Ui(interactive=False, assume_yes=True)
    remote = StreamRemote("Successfully installed a-1\n__rc=0")
    py = RemotePython(remote, "/v/bin/python", ui)
    assert py.install(["a"], "a==1\n").ok
    assert "--quiet" not in remote.command
    assert "--progress-bar off" in remote.command and "--no-cache-dir" in remote.command
    assert "a==1" in remote.command


def test_stream_kills_the_process_and_closes_the_pipe_when_on_line_raises(monkeypatch, remote):
    proc = FakeProc(["a\n", "b\n"], 0)
    patch_popen(monkeypatch, proc)

    def boom(line: str) -> None:
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        remote.stream("pip install", boom)
    assert proc.killed and proc.stdout.closed


def test_stream_kills_the_process_on_ctrl_c(monkeypatch, remote):
    proc = FakeProc(["a\n"], 0)
    patch_popen(monkeypatch, proc)

    def interrupt(line: str) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        remote.stream("pip install", interrupt)
    assert proc.killed


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch):
    """A real ``ssh`` on PATH that runs the remote command through the local shell."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ssh"
    script.write_text('#!/bin/sh\nfor last; do :; done\nexec sh -c "$last"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return Remote("u", "h", tmp_path)


def test_stream_with_a_real_process_delivers_lines_as_they_come(fake_ssh):
    seen: list[tuple[str, float]] = []
    out = fake_ssh.stream(
        "echo one; sleep 0.3; echo two >&2; echo three",
        lambda line: seen.append((line, time.monotonic())),
    )
    assert [line for line, _ in seen] == ["one", "two", "three"]
    assert out == "one\ntwo\nthree"
    assert seen[1][1] - seen[0][1] > 0.15


def test_stream_with_a_real_process_checks_the_exit_code(fake_ssh):
    with pytest.raises(WizardError, match="nope"):
        fake_ssh.stream("echo nope; exit 3", lambda line: None)
    assert fake_ssh.stream("echo nope; exit 3", lambda line: None, check=False) == "nope"


def test_stream_with_a_real_process_survives_undecodable_bytes(fake_ssh):
    seen: list[str] = []
    fake_ssh.stream("printf 'a\\377b\\n'", seen.append)
    assert len(seen) == 1 and seen[0].startswith("a") and seen[0].endswith("b")


def test_progress_strips_control_characters_from_shown_lines(capsys):
    ui = Ui(interactive=False, assume_yes=True)
    with ui.progress("installing") as p:
        p("Collecting \x1b[2Ja\rb\x07")
    out = capsys.readouterr().out
    assert "Collecting [2Jab" in out and "\x1b[2J" not in out


def test_progress_notes_the_wait_only_when_slow(capsys):
    ui = Ui(interactive=False, assume_yes=True)
    with ui.progress("quick thing", slow=False, report=False):
        pass
    out = capsys.readouterr().out
    assert "quick thing" in out and "can take minutes" not in out and "✓" not in out
