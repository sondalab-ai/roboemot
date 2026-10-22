"""``run --restart`` on a platform without a restart hook (SPEC 4A, Installer)."""

from __future__ import annotations

from pathlib import Path

import pytest
from roboemot.install import Options, Ui, WizardError
from roboemot.install import run as run_mod


class NoRestartPlatform:
    """A platform whose app is installed and running, and which offers no ``restart``."""

    display_name = "Bare robot"
    app_name = "bare"
    restart = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    def connect(self, ui, host):
        pass

    def app_installed(self):
        return True

    def running_app(self):
        return self.app_name

    def start_blocker(self):
        return None

    def app_url(self, host):
        return None

    def start(self, ui):
        self.calls.append("start")


@pytest.fixture()
def platform(monkeypatch):
    fake = NoRestartPlatform()
    monkeypatch.setattr(run_mod, "select_robot", lambda ui, opts, root: ("bare.local", fake))
    return fake


def test_restart_on_a_platform_without_the_hook_is_a_clear_error(platform):
    with pytest.raises(WizardError, match="Bare robot cannot restart bare"):
        run_mod.run_start(
            Options(restart=True, yes=True), Ui(interactive=False, assume_yes=True), Path()
        )
    assert platform.calls == []


def test_dry_run_restart_without_the_hook_says_it_would_refuse(platform, capsys):
    run_mod.run_start(
        Options(restart=True, dry_run=True), Ui(interactive=False, assume_yes=False), Path()
    )
    out = capsys.readouterr().out
    assert "would refuse: Bare robot cannot restart bare" in out
    assert platform.calls == []


class OlderPlatform(NoRestartPlatform):
    """An out-of-tree platform written before ``Platform.restart``: no attribute at all."""

    restart = property(lambda self: (_ for _ in ()).throw(AttributeError("restart")))


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_plain_run_never_reads_restart(monkeypatch, capsys, dry_run):
    fake = OlderPlatform()
    monkeypatch.setattr(run_mod, "select_robot", lambda ui, opts, root: ("bare.local", fake))
    run_mod.run_start(
        Options(dry_run=dry_run, yes=True), Ui(interactive=False, assume_yes=True), Path()
    )
    assert fake.calls == ([] if dry_run else ["start"])
