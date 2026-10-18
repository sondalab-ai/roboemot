"""SPEC 4A / 12A: scripts/robot_ssh.py logic (roboemot.install.ssh) with fake discovery and exec."""

from __future__ import annotations

import pytest
import roboemot.install as inst
from roboemot.install import ssh
from roboemot.install.ui import Ui


class FakePlatform:
    manifest_id = "fake-v1"
    display_name = "Fake Robot"
    default_host = "fake.local"
    default_user = "pi"
    python = "/venvs/apps/bin/python"


@pytest.fixture
def env(monkeypatch):
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(inst, "load_platform", lambda rid: FakePlatform())
    monkeypatch.setattr(inst, "installed_platforms", lambda: ["fake-v1"])
    monkeypatch.setattr(inst, "find_robots", lambda root: [])
    ui = Ui(interactive=False, assume_yes=False)

    def run(argv):
        return ssh.main(argv, execvp=lambda f, a: calls.append((f, a)), ui=ui)

    return run, calls


def test_interactive_shell_uses_the_platform_user(env):
    run, calls = env
    assert run(["--host", "robot.lan"]) == 0
    ((file, argv),) = calls
    assert file == "ssh" and argv[0] == "ssh" and argv[-1] == "pi@robot.lan"


def test_user_flag_and_command_after_double_dash(env):
    run, calls = env
    run(["--host", "h", "--user", "pollen", "--", "journalctl", "-n", "5"])
    assert calls[0][1][-4:] == ["pollen@h", "journalctl", "-n", "5"]


def test_discovery_picks_the_found_robot(env, monkeypatch):
    run, calls = env
    monkeypatch.setattr(
        inst, "find_robots", lambda root: [("Neve 10.0.0.7", "10.0.0.7", "fake-v1")]
    )
    run([])
    assert calls[0][1][-1] == "pi@10.0.0.7"


def test_copy_id_runs_ssh_copy_id(env):
    run, calls = env
    run(["--host", "h", "--copy-id"])
    assert calls == [("ssh-copy-id", ["ssh-copy-id", "pi@h"])]


def test_debug_runs_the_installed_cli_or_explains(env):
    run, calls = env
    run(["--host", "h", "--debug"])
    remote = calls[0][1][-1]
    assert "/venvs/apps/bin/roboemot debug --robot fake-v1 --json" in remote
    assert "test -x /venvs/apps/bin/roboemot" in remote
    assert "scripts/install_on_robot.py" in remote and "exit 127" in remote


def test_dry_run_prints_and_does_not_exec(env, capsys):
    run, calls = env
    assert run(["--host", "h", "--dry-run", "--", "uptime"]) == 0
    assert calls == []
    assert capsys.readouterr().out.strip().endswith("pi@h uptime")


def test_conflicting_flags_are_rejected(env):
    run, _ = env
    with pytest.raises(SystemExit):
        run(["--debug", "--copy-id"])
    with pytest.raises(SystemExit):
        run(["--debug", "--", "ls"])


def test_a_missing_ssh_binary_is_reported_not_raised(env, capsys):
    _, _ = env

    def boom(f, a):
        raise FileNotFoundError(f)

    ui = Ui(interactive=False, assume_yes=False)
    assert ssh.main(["--host", "h"], execvp=boom, ui=ui) == 1
    assert "cannot run ssh" in capsys.readouterr().out


def test_stdout_stays_clean_for_redirecting_the_report(env, monkeypatch, capsys):
    run, calls = env
    monkeypatch.setattr(
        inst, "find_robots", lambda root: [("Neve 10.0.0.7", "10.0.0.7", "fake-v1")]
    )
    run(["--debug"])
    captured = capsys.readouterr()
    assert captured.out == "" and "looking for robots" in captured.err
    assert calls[0][1][-2] == "pi@10.0.0.7"
