"""The installer's terminal output against the Sondalab Term spec (sondalab-ui
docs/tui-patterns.md): step-line grid, pending for dry-run work, ASCII and accessible forms,
error blocks with raw output only under ``--debug``, the spinner's running line, the summary."""

from __future__ import annotations

import copy
import io
import pickle
import subprocess
import threading

import pytest
import roboemot.install as inst
from roboemot.install import ui as ui_mod
from roboemot.install.remote import Remote
from roboemot.install.ui import Ui, WizardError


@pytest.fixture(autouse=True)
def plain_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "100")
    for name in ("TERM", "SONDALAB_ASCII", "SONDALAB_ACCESSIBLE", "ACCESSIBLE", "FORCE_COLOR"):
        monkeypatch.delenv(name, raising=False)


def quiet() -> Ui:
    return Ui(interactive=False, assume_yes=False)


def test_step_lines_sit_on_the_grid(capsys: pytest.CaptureFixture[str]) -> None:
    ui = quiet()
    ui.ok("Checked free space", "800 MB on /tmp")
    ui.would("Create the data folder", "/home/pollen/.local/share/roboemot")
    ui.warn("Not a Wireless daemon", "the apps venv layout may differ")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "  ✓ Checked free space          800 MB on /tmp"
    assert out[1].startswith("  ○ Create the data folder      /home")
    assert out[2].startswith("  ! Not a Wireless daemon       the apps")
    assert ui.counts == {"ok": 1, "warn": 1, "failed": 0}  # pending is not counted


def test_a_long_detail_wraps_under_the_detail_column(capsys: pytest.CaptureFixture[str]) -> None:
    quiet().ok("Removed packages", ", ".join(f"package-{n} 1.0.{n}" for n in range(12)))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) > 1 and all(len(ln) <= 100 for ln in lines)
    assert all(ln.startswith(" " * 32) and ln[32] != " " for ln in lines[1:])


def test_ascii_mode_uses_the_ascii_glyphs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SONDALAB_ASCII", "1")
    ui = quiet()
    ui.ok("Done thing")
    ui.would("Pending thing")
    ui.summary()
    out = capsys.readouterr().out
    assert "  + Done thing" in out and "  . Pending thing" in out
    assert "Summary  1 ok | 0 fixed | 0 warn | 0 failed" in out


def test_without_vt_the_output_is_plain(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FORCE_COLOR", "1")
    Ui(interactive=False, assume_yes=False, vt=False).ok("Done thing")
    assert capsys.readouterr().out == "  + Done thing\n"


def test_accessible_mode_writes_each_state_as_a_word(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SONDALAB_ACCESSIBLE", "1")
    ui = quiet()
    ui.ok("Checked free space", "800 MB")
    ui.would("Create the data folder")
    out = capsys.readouterr().out
    assert "  ✓ done Checked free space" in out and "  ○ pending Create the data folder" in out


def test_summary_counts_the_run(capsys: pytest.CaptureFixture[str]) -> None:
    ui = quiet()
    ui.ok("A")
    ui.ok("B")
    ui.warn("C")
    ui.summary()
    assert capsys.readouterr().out.endswith("\n\nSummary  2 ok · 0 fixed · 1 warn · 0 failed\n")


def test_an_error_is_three_lines_on_stderr_without_raw_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = WizardError(
        "Could not install the app", "pip install failed: boom", output="line 1\nline 2\nboom"
    )
    Ui(interactive=False, assume_yes=False, command="python x.py --host h").error(error)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "  ✗ Could not install the app",
        "    pip install failed: boom",
        "    Try: python x.py --host h --debug",
    ]


def test_debug_prints_the_raw_output_above_the_block(capsys: pytest.CaptureFixture[str]) -> None:
    error = WizardError("Could not build the wheels", "pip wheel exited 1: boom", output="a\nboom")
    Ui(interactive=False, assume_yes=False, debug=True).error(error)
    err = capsys.readouterr().err.splitlines()
    assert err[:2] == ["a", "boom"] and err[2] == "  ✗ Could not build the wheels"
    assert not any("Try:" in ln for ln in err)


def test_main_reports_an_unexpected_error_and_names_debug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(opts, ui, root):
        raise KeyError("surprise")

    monkeypatch.setattr(inst, "run_install", boom)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert inst.main(["--host", "h"]) == 1
    err = capsys.readouterr().err
    assert "✗ Unexpected error: KeyError" in err and "Traceback" not in err
    assert "Try: python scripts/install_on_robot.py --host h --debug" in err
    assert inst.main(["--host", "h", "--debug"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" in err and "--debug --debug" not in err


def test_a_quick_step_prints_no_running_line(capsys: pytest.CaptureFixture[str]) -> None:
    with quiet().running("Building the wheels…"):
        pass
    assert capsys.readouterr().err == ""


def test_a_slow_step_without_motion_prints_its_running_line_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(ui_mod.SPINNER, "delayMs", 0)
    with quiet().running("Building the wheels…"):
        threading.Event().wait(0.2)  # stderr is not a terminal: motion is off
    assert capsys.readouterr().err == "  ● Building the wheels…\n"


def test_the_cause_is_pips_error_line_not_its_notice() -> None:
    output = "Collecting x\nERROR: No matching distribution for x\n[notice] upgrade pip"
    assert ui_mod.last_line(output) == "ERROR: No matching distribution for x"
    assert ui_mod.last_line("a\nb\n") == "b" and ui_mod.last_line("") == ""


def test_without_vt_the_header_is_the_plain_lockup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdout", type("Tty", (io.StringIO,), {"isatty": lambda s: True})())
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert ui_mod.header(plain=True) == f"sonda/lab roboemot {ui_mod.app_version()}\n"


def test_prompts_and_running_lines_go_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    ui = Ui(interactive=True, assume_yes=False, read=lambda: "y\n")
    assert ui.confirm("Start roboemot now?", False) is True
    ui.active("Connecting over SSH…", "pollen@robot")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "? Start roboemot now? [y/N]" in captured.err
    assert "● Connecting over SSH…" in captured.err


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def tty_stderr(monkeypatch: pytest.MonkeyPatch) -> _Tty:
    """stderr as a terminal with motion allowed, frames every 10ms. Called in the test body:
    pytest re-installs its own capture after fixture setup."""
    err = _Tty()
    monkeypatch.setattr("sys.stderr", err)
    for name in ("CI", "SONDALAB_REDUCED_MOTION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(ui_mod.SPINNER, "intervalMs", 10)
    return err


def test_the_spinner_draws_frames_on_stderr_then_clears(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    err = tty_stderr(monkeypatch)
    monkeypatch.setitem(ui_mod.SPINNER, "delayMs", 0)
    with quiet().running("Building the wheels…"):
        threading.Event().wait(0.1)
    drawn = err.getvalue()
    frames = drawn.count("\x1b[?2026h\r\x1b[K")
    assert frames >= 3 and drawn.count("Building the wheels…") == frames
    assert drawn.endswith("\r\x1b[J")  # nothing animated stays on screen
    assert capsys.readouterr().out == ""


def test_the_spinner_draws_nothing_when_the_step_ends_before_the_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err = tty_stderr(monkeypatch)
    with quiet().running("Building the wheels…"):
        pass
    assert err.getvalue() == ""


def test_a_spinner_error_leaves_the_terminal_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    err = tty_stderr(monkeypatch)
    monkeypatch.setitem(ui_mod.SPINNER, "delayMs", 0)

    def broken(*a, **kw):
        raise RuntimeError("draw")

    monkeypatch.setattr(ui_mod._term, "spinner_frame", broken)
    with quiet().running("Building the wheels…"):
        threading.Event().wait(0.05)
    assert err.getvalue() == "\x1b[?2026l\r\x1b[J"


def test_a_wizard_error_keeps_its_output_through_copy_and_pickle() -> None:
    error = WizardError("Could not build", "pip exited 1", "try", output="raw")
    for clone in (copy.copy(error), pickle.loads(pickle.dumps(error))):
        assert (clone.what, clone.why, clone.fix, clone.output) == (
            "Could not build",
            "pip exited 1",
            "try",
            "raw",
        )


class _Ssh:
    """subprocess.run for Remote: records argv, answers with ``rc`` and ``stderr``."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.rc, self.stderr = 0, ""

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        return subprocess.CompletedProcess(argv, self.rc, "", self.stderr)


@pytest.fixture
def ssh_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> tuple[_Ssh, Remote]:
    fake = _Ssh()
    monkeypatch.setattr("roboemot.install.remote.subprocess.run", fake)
    return fake, Remote("pollen", "robot", tmp_path)


def test_after_the_login_ssh_never_prompts(ssh_run: tuple[_Ssh, Remote]) -> None:
    fake, remote = ssh_run
    remote.run("true")
    assert "BatchMode=yes" not in fake.calls[-1]
    remote.connect()
    assert "BatchMode=yes" not in fake.calls[-1]  # the login itself may ask for the password
    remote.run("uptime")
    remote.copy([], "/tmp/x")
    assert all("BatchMode=yes" in argv for argv in fake.calls[-2:])
    fake.rc, fake.stderr = 255, "Permission denied (publickey,password)."
    with pytest.raises(WizardError, match="Lost the SSH connection to pollen@robot"):
        remote.run("uptime")


@pytest.mark.parametrize("debug", [False, True])
def test_a_failing_command_never_echoes_its_stdin(
    ssh_run: tuple[_Ssh, Remote], capsys: pytest.CaptureFixture[str], debug: bool
) -> None:
    fake, remote = ssh_run
    fake.rc, fake.stderr = 1, "set-pin: store is readable by group or others"
    with pytest.raises(WizardError) as e:
        remote.run("python -m roboemot.webui set-pin --data-dir /d", stdin="246810\n")
    assert e.value.why == "a command given a value on stdin exited 1"
    assert "246810" not in str(e.value) and "246810" not in e.value.output
    Ui(interactive=False, assume_yes=False, debug=debug).error(e.value)
    err = capsys.readouterr().err
    assert "246810" not in err and ("readable by group" in err) is debug
