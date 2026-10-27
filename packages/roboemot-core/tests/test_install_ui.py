"""The installer's terminal output against the Sondalab Term spec (sondalab-ui
docs/tui-patterns.md): step-line grid, pending for dry-run work, ASCII and accessible forms,
error blocks with raw output only under ``--debug``, the spinner's running line, the summary."""

from __future__ import annotations

import io
import threading

import pytest
import roboemot.install as inst
from roboemot.install import ui as ui_mod
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


def test_prompts_and_running_lines_go_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    ui = Ui(interactive=True, assume_yes=False, read=lambda: "y\n")
    assert ui.confirm("Start roboemot now?", False) is True
    ui.active("Connecting over SSH…", "pollen@robot")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "? Start roboemot now? [y/N]" in captured.err
    assert "● Connecting over SSH…" in captured.err
