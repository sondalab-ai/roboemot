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


def typed(*lines: str) -> Ui:
    """An interactive Ui whose prompts read ``lines`` in order."""
    it = iter(lines)
    return Ui(interactive=True, assume_yes=False, read=lambda: next(it))


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
    assert calls[0][1][-3:] == ["--", "pollen@h", "journalctl -n 5"]


def test_discovery_picks_the_found_robot(env, monkeypatch):
    _, calls = env
    monkeypatch.setattr(
        inst, "find_robots", lambda root: [("Neve 10.0.0.7", "10.0.0.7", "fake-v1")]
    )
    ssh.main([], execvp=lambda f, a: calls.append((f, a)), ui=typed("1\n"))
    assert calls[0][1][-1] == "pi@10.0.0.7"


def test_without_a_terminal_the_host_flag_is_required(env, monkeypatch, capsys):
    run, calls = env
    monkeypatch.setattr(
        inst, "find_robots", lambda root: [("Neve 10.0.0.7", "10.0.0.7", "fake-v1")]
    )
    assert run([]) == 1 and calls == []
    assert "Missing --host" in capsys.readouterr().err


def test_copy_id_runs_ssh_copy_id(env):
    run, calls = env
    run(["--host", "h", "--copy-id"])
    assert calls == [("ssh-copy-id", ["ssh-copy-id", "--", "pi@h"])]


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
    assert "Could not run ssh" in capsys.readouterr().err


def test_stdout_stays_clean_for_redirecting_the_report(env, monkeypatch, capsys):
    _, calls = env
    monkeypatch.setattr(
        inst, "find_robots", lambda root: [("Neve 10.0.0.7", "10.0.0.7", "fake-v1")]
    )
    ssh.main(["--debug"], execvp=lambda f, a: calls.append((f, a)), ui=typed("1\n"))
    captured = capsys.readouterr()
    assert captured.out == "" and "Which robot?" in captured.err
    assert calls[0][1][-3:-1] == ["--", "pi@10.0.0.7"]


@pytest.mark.parametrize(
    "flags",
    [
        ["--host", "h", "--user=-oProxyCommand=x"],
        ["--host", "h", "--user", "a b"],
        ["--host=-oProxyCommand=x"],
        ["--host", "user@h"],
        ["--host", "fe80::1%eth0"],
        ["--host", "bad_host"],
        ["--host=-h", "--copy-id"],
    ],
)
def test_option_looking_user_and_host_are_refused_before_any_exec(env, flags, capsys):
    run, calls = env
    assert run(flags) == 1
    assert calls == []


def test_an_option_looking_discovered_or_typed_host_is_refused(env, monkeypatch):
    run, calls = env
    monkeypatch.setattr(inst, "find_robots", lambda root: [("x", "-oProxyCommand=y", "fake-v1")])
    ui = typed("1\n")
    assert ssh.main([], execvp=lambda f, a: calls.append((f, a)), ui=ui) == 1 and calls == []


def test_valid_hosts_and_users_pass():
    for host in ("reachy-mini.local", "192.168.1.20", "fe80::1", "robot.", "a" * 63):
        assert ssh.valid_host(host) == host
    for user in ("pollen", "_svc", "a.b-c_1"):
        assert ssh.valid_user(user) == user


def test_double_dash_precedes_the_target_and_the_command_keeps_its_quoting(env):
    run, calls = env
    run(["--host", "h", "--", "echo", "a b", "$HOME;ls"])
    argv = calls[0][1]
    assert argv[argv.index("--") + 1] == "pi@h"
    assert argv[-1] == "echo 'a b' '$HOME;ls'" and argv[-2] == "pi@h"
