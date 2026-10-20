"""SPEC 4A, Installer: live progress for long remote steps (streaming, filter, status line)."""

from __future__ import annotations

import io
import subprocess
import time
from pathlib import Path

import pytest
from roboemot.install.pyenv import RemotePython, pip_progress_line
from roboemot.install.remote import Remote
from roboemot.install.ui import Ui, WizardError, format_elapsed


class Sink:
    def __init__(self) -> None:
        self.sent = ""

    def write(self, text: str) -> None:
        self.sent += text

    def close(self) -> None:
        pass


class FakeProc:
    def __init__(self, lines: list[str], rc: int) -> None:
        self.stdout = iter(lines)
        self.stdin = Sink()
        self.returncode = rc

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


def test_stream_sends_stdin_and_never_echoes_it(monkeypatch, remote):
    proc = FakeProc(["ok\n"], 0)
    patch_popen(monkeypatch, proc)
    seen: list[str] = []
    remote.stream("cat >/dev/null", seen.append, stdin="s3cret")
    assert proc.stdin.sent == "s3cret" and seen == ["ok"]


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
    assert "✓ installed (0s)" in out


def test_progress_on_a_terminal_rewrites_one_line_then_clears_it(monkeypatch):
    out = Tty()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("shutil.get_terminal_size", lambda: type("S", (), {"columns": 60})())
    clock = Clock()
    ui = Ui(interactive=False, assume_yes=True)
    ui.clock, ui.tick_s = clock, 0.001
    with ui.progress("installing", pip_progress_line) as p:
        clock.now = 102
        p("Collecting " + "x" * 200)
        deadline = time.monotonic() + 5
        while out.getvalue().count("1m42s") < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
    text = out.getvalue()
    assert "\r" in text and "\x1b[K" in text
    assert "1m42s  Collecting x" in text and "…" in text
    frames = [r.removesuffix("\x1b[K") for r in text.split("\r") if r.endswith("\x1b[K")]
    assert frames and max(map(len, frames)) <= 60
    assert "installing (1m42s)\n" in text.split("✓")[-1]
    assert text.rindex("\r\x1b[K") < text.index("✓")


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

    def stream(self, command, on_line, check=True, stdin=None):
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
    with pytest.raises(WizardError, match="pip install failed on the robot:\nCollecting a\nERROR"):
        py._pip("install a", announce="installing")
    assert "✓" not in capsys.readouterr().out


def test_long_pip_drops_quiet_and_keeps_the_cache_flags():
    ui = Ui(interactive=False, assume_yes=True)
    remote = StreamRemote("Successfully installed a-1\n__rc=0")
    py = RemotePython(remote, "/v/bin/python", ui)
    assert py.install(["a"], "a==1\n").ok
    assert "--quiet" not in remote.command
    assert "--progress-bar off" in remote.command and "--no-cache-dir" in remote.command
    assert "a==1" in remote.command
