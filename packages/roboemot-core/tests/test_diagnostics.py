"""Debug report (SPEC 12A, Diagnostics): isolation, JSON safety, platform selection, sources."""

from __future__ import annotations

import enum
import json
import math
import subprocess
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot import cli, diagnostics
from roboemot.diagnostics import (
    FnSource,
    Target,
    collect,
    platform_sources,
    render_text,
    system,
)


class Mode(enum.Enum):
    ON = "on"


def test_failing_and_hanging_sections_do_not_break_the_report():
    release = threading.Event()

    def boom():
        raise OSError("sensor gone")

    report = collect(
        [
            FnSource("good", lambda: {"x": 1}),
            FnSource("bad", boom, "local"),
            FnSource("slow", lambda: release.wait(10), "robot"),
        ],
        timeout_s=0.2,
    )
    release.set()
    s = report["sections"]
    assert s["good"] == {"scope": "runtime", "ok": True, "data": {"x": 1}}
    assert s["bad"] == {"scope": "local", "ok": False, "error": "OSError: sensor gone"}
    assert s["slow"]["ok"] is False and "timed out" in s["slow"]["error"]
    assert list(s) == ["good", "bad", "slow"]  # registration order, not completion order


def test_report_is_plain_json_with_secrets_redacted():
    data = {
        "f": np.float32(1.5),
        "i": np.int64(3),
        "nan": math.nan,
        "mode": Mode.ON,
        "t": (1, 2),
        "nt": SimpleNamespace(),
        "api_key": "sk-123",
        "nested": {"HF_TOKEN": "hf_x", "keep": "ok"},
    }
    report = collect([FnSource("s", lambda: data)])
    out = report["sections"]["s"]["data"]
    assert json.loads(json.dumps(report)) == report
    assert out["f"] == 1.5 and out["i"] == 3 and out["nan"] is None and out["mode"] == "on"
    assert out["t"] == [1, 2] and isinstance(out["nt"], str)
    assert out["api_key"] == "***" and out["nested"] == {"HF_TOKEN": "***", "keep": "ok"}
    assert report["schema"] == diagnostics.SCHEMA_VERSION


def test_unknown_platform_gives_an_error_section():
    (src,) = platform_sources("no-such-robot", Target())
    sec = diagnostics.collect_section(src)
    assert sec["scope"] == "robot" and not sec["ok"] and "no-such-robot" in sec["error"]


def test_platform_selected_by_manifest_id(monkeypatch):
    seen = []

    def factory(target):
        seen.append(target)
        return [FnSource("robot.x", lambda: 1, "robot")]

    eps = [
        SimpleNamespace(name="other", load=lambda: pytest.fail("wrong platform loaded")),
        SimpleNamespace(name="bot-v1", load=lambda: factory),
    ]
    monkeypatch.setattr(diagnostics, "entry_points", lambda group: eps)
    (src,) = platform_sources("bot-v1", Target("10.0.0.2", 9000))
    assert src.name == "robot.x" and seen == [Target("10.0.0.2", 9000)]


def test_target_remote_only_for_other_hosts():
    assert not Target().remote and not Target("localhost").remote
    assert not Target("127.0.0.1", 8000).remote
    assert Target("192.168.1.20").remote


def test_render_text_shows_errors_and_nested_values():
    report = collect(
        [FnSource("a", lambda: {"cpu": {"percent": 12.5}}), FnSource("b", int.bit_length)]
    )
    text = render_text(report)
    assert "[a] (runtime)" in text and "percent: 12.5" in text
    assert "[b] (runtime)" in text and "error: TypeError" in text


# ---------------------------------------------------------------- generic sources


def write_supply(root, name, **files):
    d = root / name
    d.mkdir(parents=True)
    for k, v in files.items():
        (d / k).write_text(f"{v}\n")


def test_power_supplies_in_si_units(tmp_path):
    write_supply(
        tmp_path,
        "BAT0",
        type="Battery",
        status="Discharging",
        present=1,
        capacity=76,
        health="Good",
        voltage_now=7400000,
        current_now=-350000,
        temp=312,
    )
    write_supply(tmp_path, "AC", type="Mains", online=0)
    s = system.power_supplies(tmp_path)
    assert s["BAT0"]["capacity_pct"] == 76 and s["BAT0"]["status"] == "Discharging"
    assert s["BAT0"]["voltage_v"] == pytest.approx(7.4)
    assert s["BAT0"]["current_a"] == pytest.approx(-0.35)
    assert s["BAT0"]["temperature_c"] == pytest.approx(31.2)
    assert s["BAT0"]["present"] is True and s["AC"] == {"type": "Mains", "online": False}


def test_battery_says_so_when_the_platform_reports_none(tmp_path, monkeypatch):
    monkeypatch.setattr(system.psutil, "sensors_battery", lambda: None, raising=False)
    monkeypatch.setattr(system, "throttling", lambda: None)
    out = system.battery(tmp_path / "missing")
    assert out == {"available": False, "note": "no battery reported by this platform"}


def test_battery_from_sysfs_counts_as_available(tmp_path, monkeypatch):
    monkeypatch.setattr(system.psutil, "sensors_battery", lambda: None, raising=False)
    monkeypatch.setattr(system, "throttling", lambda: None)
    write_supply(tmp_path, "battery", type="Battery", capacity=40)
    out = system.battery(tmp_path)
    assert out["available"] and out["supplies"]["battery"]["capacity_pct"] == 40


def test_throttling_decodes_raspberry_pi_flags(monkeypatch):
    monkeypatch.setattr(system.shutil, "which", lambda cmd: "/usr/bin/vcgencmd")

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="throttled=0x50005\n")

    flags = system.throttling(run)
    assert flags["raw"] == "0x50005"
    assert flags["under_voltage_now"] and flags["throttled_now"]
    assert flags["under_voltage_occurred"] and flags["throttled_occurred"]
    assert not flags["arm_frequency_capped_now"]


def test_throttling_absent_off_a_pi(monkeypatch):
    monkeypatch.setattr(system.shutil, "which", lambda cmd: None)
    assert system.throttling() is None


def test_temperatures_without_sensor_support(monkeypatch):
    monkeypatch.delattr(system.psutil, "sensors_temperatures", raising=False)
    monkeypatch.delattr(system.psutil, "sensors_fans", raising=False)
    monkeypatch.setattr(system, "throttling", lambda: None)
    assert system.temperatures() == {
        "sensors": {},
        "note": "no temperature sensor reported by this platform",
    }


def test_local_sources_run_on_this_machine():
    report = collect(system.local_sources(process_pid=None, process_match="no-such-process-x"))
    s = report["sections"]
    assert s["system"]["ok"] and s["system"]["data"]["cpu"]["count"] >= 1
    assert s["battery"]["ok"] and s["temperatures"]["ok"]
    assert s["process"]["data"] == {
        "running": False,
        "note": "no process matching 'no-such-process-x'",
    }


def test_process_info_for_own_pid():
    import os

    out = system.process_info(pid=os.getpid())
    (p,) = out["processes"]
    assert p["pid"] == os.getpid() and p["rss_bytes"] > 0 and p["threads"] >= 1


# ---------------------------------------------------------------- CLI


def test_cli_leaves_local_sections_out_for_a_remote_robot(monkeypatch, capsys):
    monkeypatch.setattr(
        diagnostics,
        "platform_sources",
        lambda rid, t: [FnSource("robot.x", lambda: t.host, "robot")],
    )
    rc = cli.main(["debug", "--robot", "bot-v1", "--host", "192.168.1.20", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert list(report["sections"]) == ["robot.x"]
    assert report["sections"]["robot.x"]["data"] == "192.168.1.20"
    assert "left out for the remote robot" in report["notes"][0]


def test_cli_exit_code_reports_failed_sections(monkeypatch, capsys):
    monkeypatch.setattr(
        diagnostics,
        "platform_sources",
        lambda rid, t: [FnSource("robot.x", int.bit_length, "robot")],
    )
    assert cli.main(["debug", "--robot", "bot-v1", "--host", "10.9.9.9"]) == 1
    assert "error: TypeError" in capsys.readouterr().out


def test_numpy_arrays_become_lists():
    report = collect([FnSource("s", lambda: {"v": np.arange(3), "m": np.eye(4)})])
    data = report["sections"]["s"]["data"]
    assert data["v"] == [0, 1, 2] and data["m"][0] == [1.0, 0.0, 0.0, 0.0]
    assert json.loads(json.dumps(report)) == report


def test_render_text_escapes_terminal_control_sequences():
    evil = "Neve\x1b]0;pwned\x07\x1b[2J"
    text = render_text(collect([FnSource("robot", lambda: {"name": evil, evil: 1})]))
    assert "\x1b" not in text and "\x07" not in text
    assert "Neve\\x1b]0;pwned\\x07" in text


def test_broken_platform_plugin_becomes_an_error_section(monkeypatch):
    def broken_load():
        raise ImportError("missing dependency")

    eps = [SimpleNamespace(name="bot-v1", load=broken_load)]
    monkeypatch.setattr(diagnostics, "entry_points", lambda group: eps)
    (src,) = platform_sources("bot-v1", Target())
    sec = diagnostics.collect_section(src)
    assert not sec["ok"] and sec["error"] == "ImportError: missing dependency"
