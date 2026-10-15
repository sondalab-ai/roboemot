"""Generic sources for the machine that builds the report (scope ``local``).

Nothing here knows a robot model: system and OS, CPU, memory, disks, network, temperatures,
battery and power supply, and the RoboEmot process. Values the platform does not report are
left out or ``None``, never guessed: ``psutil.sensors_temperatures`` does not exist on macOS,
and ``psutil.sensors_battery`` is ``None`` on most Linux boards, whose supplies (if any) appear
under ``/sys/class/power_supply``. On a Raspberry Pi, ``vcgencmd get_throttled`` tells whether
the supply voltage dropped too low, the usual sign of a weak battery or charger.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil

from . import FnSource, Source

CPU_SAMPLE_S = 0.2
COMMAND_TIMEOUT_S = 2.0
POWER_SUPPLY_ROOT = Path("/sys/class/power_supply")
DEVICE_MODEL = Path("/proc/device-tree/model")
# sysfs power_supply units: micro-volts, micro-amps, micro-watts, micro-watt/amp-hours, 0.1 °C.
MICRO = 1e-6
DECI = 0.1
SYSFS_SCALE = {
    "voltage_now": MICRO,
    "voltage_min_design": MICRO,
    "voltage_max_design": MICRO,
    "current_now": MICRO,
    "power_now": MICRO,
    "energy_now": MICRO,
    "energy_full": MICRO,
    "energy_full_design": MICRO,
    "charge_now": MICRO,
    "charge_full": MICRO,
    "charge_full_design": MICRO,
    "temp": DECI,
}
SYSFS_RENAME = {
    "voltage_now": "voltage_v",
    "voltage_min_design": "voltage_min_design_v",
    "voltage_max_design": "voltage_max_design_v",
    "current_now": "current_a",
    "power_now": "power_w",
    "energy_now": "energy_wh",
    "energy_full": "energy_full_wh",
    "energy_full_design": "energy_full_design_wh",
    "charge_now": "charge_ah",
    "charge_full": "charge_full_ah",
    "charge_full_design": "charge_full_design_ah",
    "temp": "temperature_c",
    "capacity": "capacity_pct",
}
SYSFS_FIELDS = (
    "type",
    "status",
    "present",
    "online",
    "capacity",
    "capacity_level",
    "health",
    "technology",
    "cycle_count",
    "model_name",
    "manufacturer",
    *SYSFS_SCALE,
)
# vcgencmd get_throttled bits (Raspberry Pi firmware).
THROTTLE_BITS = {
    0: "under_voltage_now",
    1: "arm_frequency_capped_now",
    2: "throttled_now",
    3: "soft_temperature_limit_now",
    16: "under_voltage_occurred",
    17: "arm_frequency_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temperature_limit_occurred",
}
PSEUDO_FS = {"tmpfs", "devtmpfs", "squashfs", "overlay", "autofs", "devfs"}


def system_info() -> dict[str, Any]:
    """OS, board, uptime, CPU, memory, disks and network interfaces."""
    boot = psutil.boot_time()
    freq = psutil.cpu_freq()
    return {
        "os": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "board": _read(DEVICE_MODEL),
        "python": sys.version.split()[0],
        "roboemot": _version("roboemot-core"),
        "uptime_s": time.time() - boot,
        "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "cpu": {
            "count": psutil.cpu_count(),
            "percent": psutil.cpu_percent(interval=CPU_SAMPLE_S),
            "freq_mhz": freq.current if freq else None,
            "freq_max_mhz": freq.max if freq else None,
        },
        "memory": psutil.virtual_memory()._asdict(),
        "swap": psutil.swap_memory()._asdict(),
        "disks": _disks(),
        "network": _network(),
    }


def temperatures() -> dict[str, Any]:
    """Every sensor the OS exposes, in °C, plus Raspberry Pi throttling flags if present."""
    read = getattr(psutil, "sensors_temperatures", None)
    sensors = {
        chip: [e._asdict() for e in entries] for chip, entries in (read() if read else {}).items()
    }
    fans_read = getattr(psutil, "sensors_fans", None)
    fans = {c: [e._asdict() for e in f] for c, f in (fans_read() if fans_read else {}).items()}
    out: dict[str, Any] = {"sensors": sensors}
    if fans:
        out["fans"] = fans
    if not sensors:
        out["note"] = "no temperature sensor reported by this platform"
    throttled = throttling()
    if throttled is not None:
        out["throttling"] = throttled
    return out


def battery(root: Path = POWER_SUPPLY_ROOT) -> dict[str, Any]:
    """Charge, status, voltage, health of every power supply the OS reports."""
    out: dict[str, Any] = {}
    psb = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if psb is not None:
        secs = psb.secsleft
        out["summary"] = {
            "percent": psb.percent,
            "power_plugged": psb.power_plugged,
            "time_left_s": secs if isinstance(secs, int | float) and secs >= 0 else None,
        }
    supplies = power_supplies(root)
    if supplies:
        out["supplies"] = supplies
    throttled = throttling()
    if throttled is not None:
        out["supply_voltage"] = {
            k: v for k, v in throttled.items() if k.startswith("under_voltage") or k == "raw"
        }
    out["available"] = bool(psb or any(s.get("type") == "Battery" for s in supplies.values()))
    if not out["available"]:
        out["note"] = "no battery reported by this platform"
    return out


def power_supplies(root: Path = POWER_SUPPLY_ROOT) -> dict[str, dict[str, Any]]:
    """``/sys/class/power_supply/*`` in SI units; empty where the OS has none."""
    if not root.is_dir():
        return {}
    supplies = {}
    for d in sorted(root.iterdir()):
        fields = {}
        for f in SYSFS_FIELDS:
            raw = _read(d / f)
            if raw is None:
                continue
            fields[SYSFS_RENAME.get(f, f)] = _sysfs_value(f, raw)
        if fields:
            supplies[d.name] = fields
    return supplies


def throttling(run: Any = subprocess.run) -> dict[str, Any] | None:
    """Raspberry Pi firmware flags (``vcgencmd get_throttled``); ``None`` off a Pi."""
    if shutil.which("vcgencmd") is None:
        return None
    try:
        res = run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
            check=True,
        )
        raw = int(res.stdout.strip().split("=")[-1], 16)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return {"raw": hex(raw)} | {name: bool(raw >> bit & 1) for bit, name in THROTTLE_BITS.items()}


def process_info(pid: int | None = None, match: str | None = None) -> dict[str, Any]:
    """Resources of ``pid``, or of every other process whose command line contains ``match``."""
    if pid is not None:
        procs = [psutil.Process(pid)]
    else:
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        procs = [
            p
            for p in psutil.process_iter(["pid", "cmdline"])
            if p.pid not in own and match and match in " ".join(p.info["cmdline"] or [])
        ]
    if not procs:
        return {"running": False, "note": f"no process matching {match!r}"}
    return {"running": True, "processes": [_proc(p) for p in procs]}


def local_sources(process_pid: int | None = None, process_match: str | None = None) -> list[Source]:
    """The generic sources; the process one only when told which process to look at."""
    sources: list[Source] = [
        FnSource("system", system_info, "local"),
        FnSource("temperatures", temperatures, "local"),
        FnSource("battery", battery, "local"),
    ]
    if process_pid is not None or process_match:
        sources.append(
            FnSource("process", lambda: process_info(process_pid, process_match), "local")
        )
    return sources


def _proc(p: psutil.Process) -> dict[str, Any]:
    with p.oneshot():
        mem = p.memory_info()
        return {
            "pid": p.pid,
            "cmdline": " ".join(p.cmdline()),
            "uptime_s": time.time() - p.create_time(),
            "cpu_percent": p.cpu_percent(interval=CPU_SAMPLE_S),
            "rss_bytes": mem.rss,
            "vms_bytes": mem.vms,
            "memory_percent": p.memory_percent(),
            "threads": p.num_threads(),
            "open_files": _safe(lambda: len(p.open_files())),
            "fds": _safe(p.num_fds) if hasattr(p, "num_fds") else None,
        }


def _disks() -> list[dict[str, Any]]:
    out = []
    for part in psutil.disk_partitions(all=False):
        if part.fstype in PSEUDO_FS:
            continue
        usage = _safe(lambda m=part.mountpoint: psutil.disk_usage(m)._asdict())
        if usage:
            out.append({"mount": part.mountpoint, "fstype": part.fstype, **usage})
    return out


def _network() -> dict[str, Any]:
    stats = psutil.net_if_stats()
    addrs = psutil.net_if_addrs()
    io = psutil.net_io_counters(pernic=True)
    out = {}
    for nic, st in stats.items():
        if not st.isup:
            continue
        out[nic] = {
            "speed_mbps": st.speed or None,
            "mtu": st.mtu,
            "addresses": [
                a.address for a in addrs.get(nic, []) if a.family.name in ("AF_INET", "AF_INET6")
            ],
            "io": io[nic]._asdict() if nic in io else None,
        }
    return out


def _sysfs_value(field: str, raw: str) -> Any:
    try:
        number = int(raw)
    except ValueError:
        return raw
    if field in ("present", "online"):
        return bool(number)
    return number * SYSFS_SCALE[field] if field in SYSFS_SCALE else number


def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip("\x00\n ") or None
    except OSError:
        return None


def _version(dist: str) -> str | None:
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _safe(fn: Any) -> Any:
    try:
        return fn()
    except (psutil.Error, OSError):
        return None


__all__ = [
    "battery",
    "local_sources",
    "power_supplies",
    "process_info",
    "system_info",
    "temperatures",
    "throttling",
]
