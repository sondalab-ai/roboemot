"""The installer opens with the Sondalab header (sondalab-ui docs/tui-patterns.md, Header):
full on a terminal and before ``--help``, nothing on a pipe or in accessible mode (the palette
decides since 0.9.0, the installer has no guard of its own), and ``--version`` prints the
one-line lockup."""

from __future__ import annotations

import io

import pytest
import roboemot.install as inst
import sondalab_palette
from roboemot.install import ui


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def _tty_stdout(monkeypatch: pytest.MonkeyPatch) -> _Tty:
    """stdout as a 120-column terminal with colour off, so the header compares byte for byte.
    Called in the test body: pytest re-installs its own capture after fixture setup."""
    out = _Tty()
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setenv("NO_COLOR", "1")
    for name in ("SONDALAB_ACCESSIBLE", "ACCESSIBLE", "SONDALAB_ASCII", "TERM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    return out


def _skip_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inst, "run_install", lambda opts, ui_, root: None)


def test_the_installer_opens_with_the_full_header_on_a_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tty_stdout = _tty_stdout(monkeypatch)
    _skip_the_run(monkeypatch)
    assert inst.main(["--dry-run"]) == 0
    lines = tty_stdout.getvalue().splitlines()
    assert lines[0] == sondalab_palette.HEADER["mark"][0] + "  sonda/lab"
    assert " roboemot " in f" {lines[1]} "
    assert lines[2].endswith("  " + ui.TAGLINE)


def test_help_opens_with_the_full_header(monkeypatch: pytest.MonkeyPatch) -> None:
    tty_stdout = _tty_stdout(monkeypatch)
    with pytest.raises(SystemExit):
        inst.main(["--help"])
    lines = tty_stdout.getvalue().splitlines()
    assert lines[0] == sondalab_palette.HEADER["mark"][0] + "  sonda/lab"
    assert lines[3].startswith("usage:")  # argparse's help follows, unchanged


def test_no_header_when_stdout_is_not_a_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _skip_the_run(monkeypatch)
    assert inst.main(["--dry-run"]) == 0
    assert capsys.readouterr().out == ""  # piped output is the result alone


def test_the_header_is_the_palettes_own(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(ui._term, "header", lambda *a, **kw: seen.append((a, kw)) or "H\n")
    assert ui.header() == "H\n"
    assert seen == [(("roboemot", ui.app_version(), ui.TAGLINE), {})]


@pytest.mark.parametrize("tty", [True, False])
def test_version_prints_the_lockup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tty: bool
) -> None:
    out = _tty_stdout(monkeypatch) if tty else None
    _skip_the_run(monkeypatch)
    assert inst.main(["--version"]) == 0
    text = out.getvalue() if out else capsys.readouterr().out
    assert text == f"sonda/lab roboemot {ui.app_version()}\n"


def test_no_header_in_accessible_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    tty_stdout = _tty_stdout(monkeypatch)
    monkeypatch.setenv("SONDALAB_ACCESSIBLE", "1")
    _skip_the_run(monkeypatch)
    assert inst.main(["--dry-run"]) == 0
    assert "sonda" not in tty_stdout.getvalue()
