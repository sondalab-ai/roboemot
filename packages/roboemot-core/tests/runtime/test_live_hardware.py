"""SPEC 11A, 12: the hardware readings of the Live tab's System card."""

from __future__ import annotations

import json
import subprocess
import threading
from types import SimpleNamespace

import psutil
import pytest
from roboemot.diagnostics import system
from roboemot.runtime import live_hardware
from roboemot.runtime.live_hardware import HardwareReader, busy_percent, throttle_state


def times(**fields):
    """A stand-in for ``psutil.cpu_times()``, which is a named tuple."""
    return SimpleNamespace(_asdict=lambda: fields)


WIRELESS = """Inter-| sta-|   Quality        |   Discarded packets
 face | tus | link level noise |  nwid  crypt   frag  retry   misc
wlan0: 0000   58.  -52.  -256        0      0      0      0      0
wlan1: 0000    0.    0.  -256        0      0      0      0      0
"""


@pytest.fixture
def machine(monkeypatch):
    """Fixed psutil answers for every reading, so the tests never depend on this machine."""
    process = SimpleNamespace(
        cpu_percent=lambda interval: 80.0,
        memory_info=lambda: SimpleNamespace(rss=150 * 1024**2),
        num_threads=lambda: 21,
    )
    clock = {"times": times(user=10.0, idle=90.0)}
    monkeypatch.setattr(psutil, "Process", lambda: process)
    monkeypatch.setattr(psutil, "cpu_times", lambda: clock["times"])
    monkeypatch.setattr(psutil, "cpu_count", lambda: 4)
    monkeypatch.setattr(
        psutil, "cpu_freq", lambda: SimpleNamespace(current=1500.4, min=600.0, max=1800.0)
    )
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=41.26))
    monkeypatch.setattr(psutil, "swap_memory", lambda: SimpleNamespace(total=1024, percent=3.0))
    monkeypatch.setattr(
        psutil, "disk_usage", lambda path: SimpleNamespace(free=8 * 1024**3, percent=62.5)
    )
    return clock


def reader(tmp_path, flags=lambda: None, wireless=WIRELESS):
    path = tmp_path / "wireless"
    path.write_text(wireless)
    return HardwareReader(proc_wireless=path, flags=flags)


def test_the_cpu_load_comes_from_the_change_between_two_reads(machine, tmp_path):
    read = reader(tmp_path)
    assert read()["cpu"]["percent"] is None  # nothing to compare with yet
    machine["times"] = times(user=40.0, idle=160.0)  # 30 busy of 100 elapsed
    assert read()["cpu"]["percent"] == 30.0


def test_the_cpu_load_is_right_when_each_request_has_its_own_thread(machine, tmp_path):
    """Regression guard: ``psutil.cpu_percent(None)`` would keep one sample per thread."""
    read, out = reader(tmp_path), []
    machine_times = [times(user=10.0, idle=90.0), times(user=40.0, idle=160.0)]
    for sample in machine_times:
        machine["times"] = sample
        worker = threading.Thread(target=lambda: out.append(read()["cpu"]["percent"]))
        worker.start()
        worker.join()
    assert out == [None, 30.0]


def test_busy_percent_ignores_guest_time_and_counts_iowait_as_idle():
    before = times(user=0.0, idle=0.0, iowait=0.0, guest=0.0, guest_nice=0.0)
    after = times(user=20.0, idle=70.0, iowait=10.0, guest=20.0, guest_nice=0.0)
    assert busy_percent(before, after) == pytest.approx(20.0)  # guest is already in user
    assert busy_percent(after, after) is None


def test_the_readings_are_rounded_and_json_able(machine, tmp_path):
    read = reader(tmp_path)
    read()
    block = read()
    assert block["cpu"]["freq_mhz"] == 1500 and block["cpu"]["freq_max_mhz"] == 1800
    assert block["memory"] == {"percent": 41.3, "swap_percent": 3.0}
    assert block["disk"] == {"free_gb": 8.0, "percent": 62.5}
    json.dumps(block, allow_nan=False)


def test_the_process_block_is_a_share_of_the_whole_machine(machine, tmp_path):
    read = reader(tmp_path)
    assert read()["process"]["cpu_percent"] is None  # psutil's first answer has no meaning
    assert read()["process"] == {"cpu_percent": 20.0, "rss_mb": 150, "threads": 21}


def test_wifi_reports_the_associated_interface_and_never_a_network_name(machine, tmp_path):
    assert reader(tmp_path)()["wifi"] == {"signal_dbm": -52.0, "link_quality": 58.0}


def test_wifi_is_none_without_a_link_or_the_file(machine, tmp_path):
    unlinked = WIRELESS.replace("58.  -52.", "0.    0.")
    assert reader(tmp_path, wireless=unlinked)()["wifi"] is None
    assert HardwareReader(proc_wireless=tmp_path / "missing")()["wifi"] is None


def test_throttling_names_what_holds_now_and_since_boot():
    flags = {
        "raw": "0x50001",
        "under_voltage_now": True,
        "throttled_now": False,
        "under_voltage_occurred": True,
        "throttled_occurred": True,
    }
    assert throttle_state(flags) == {
        "now": ["under_voltage"],
        "since_boot": ["throttled", "under_voltage"],
    }
    assert throttle_state(None) is None


def test_one_failed_reading_leaves_the_others(machine, monkeypatch, tmp_path):
    def broken(*args):
        raise OSError("unreadable")

    monkeypatch.setattr(psutil, "disk_usage", broken)
    monkeypatch.setattr(psutil, "cpu_freq", broken)
    block = reader(tmp_path, flags=broken)()
    assert block["disk"] is None and block["throttling"] is None
    assert block["cpu"]["freq_mhz"] is None
    assert block["memory"]["percent"] == 41.3


def test_swap_is_none_on_a_system_without_swap(machine, monkeypatch, tmp_path):
    monkeypatch.setattr(psutil, "swap_memory", lambda: SimpleNamespace(total=0, percent=0.0))
    assert reader(tmp_path)()["memory"]["swap_percent"] is None


@pytest.mark.parametrize("found", ["/usr/bin/vcgencmd", None])
def test_vcgencmd_is_looked_up_once_for_every_refresh(machine, tmp_path, monkeypatch, found):
    lookups, runs = [], []

    def which(cmd):
        lookups.append(cmd)
        return found

    def run(cmd, **kw):
        runs.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="throttled=0x50005\n")

    monkeypatch.setattr(live_hardware.shutil, "which", which)
    monkeypatch.setattr(
        live_hardware, "read_throttling", lambda cmd: system.read_throttling(cmd, run)
    )
    (tmp_path / "wireless").write_text(WIRELESS)
    read = HardwareReader(proc_wireless=tmp_path / "wireless")
    blocks = [read()["throttling"] for _ in range(3)]
    assert lookups == ["vcgencmd"], "one PATH scan, also when it found nothing"
    if found:
        assert runs == [[found, "get_throttled"]] * 3
        assert blocks[-1]["now"] == ["throttled", "under_voltage"]
    else:
        assert runs == [] and blocks == [None] * 3
