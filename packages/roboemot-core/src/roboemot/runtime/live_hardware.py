"""Hardware readings for the Live tab's System card (SPEC 11A, 12).

``HardwareReader`` builds the card's numbers on a cache miss of ``LiveState``. No read waits for a
sample period. A reading the OS does not report is None, and one failed reading never hides the
others. Numbers and state codes only: the Wi-Fi block reads ``/proc/net/wireless``, which holds no
SSID and no access point address.

CPU shares are differences between two reads. ``psutil.cpu_percent(None)`` keeps its last sample
per thread and the web server starts one thread per request, so this class keeps the samples
itself. The first read has nothing to compare with and reports None.
"""

from __future__ import annotations

import functools
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from roboemot.diagnostics.system import (
    PROC_WIRELESS,
    VCGENCMD,
    parse_proc_wireless,
    read_throttling,
)

PERCENT = 100
BYTES_PER_MB = 1024**2
BYTES_PER_GB = 1024**3
ROOT_DISK = "/"
THROTTLE_NOW = "_now"
THROTTLE_PAST = "_occurred"
GUEST_FIELDS = ("guest", "guest_nice")  # Linux counts these inside user and nice as well
IDLE_FIELDS = ("idle", "iowait")


def _attempt(read: Callable[[], Any]) -> Any:
    """The answer of ``read``, or None where this OS cannot give it."""
    try:
        return read()
    except Exception:  # missing on this OS, unreadable, or the process state changed
        return None


def _rounded(value: float | None) -> float | None:
    """``value`` to one decimal, for the numbers sent to the page."""
    return None if value is None else round(value, 1)


def busy_percent(before: Any, after: Any) -> float | None:
    """The busy share of all CPUs between two ``psutil.cpu_times()``; None without elapsed time."""
    old, new = before._asdict(), after._asdict()
    total = sum(new.values()) - sum(old.values())
    total -= sum(new.get(f, 0) - old.get(f, 0) for f in GUEST_FIELDS)
    if total <= 0:
        return None
    idle = sum(new.get(f, 0) - old.get(f, 0) for f in IDLE_FIELDS)
    return min(max(PERCENT * (total - idle) / total, 0), PERCENT)


def throttle_state(flags: dict[str, Any] | None) -> dict[str, list[str]] | None:
    """Firmware flags as the names that hold ``now`` and ``since_boot``; None off a Pi."""
    if flags is None:
        return None

    def names(suffix: str) -> list[str]:
        return sorted(k.removesuffix(suffix) for k, v in flags.items() if k.endswith(suffix) and v)

    return {"now": names(THROTTLE_NOW), "since_boot": names(THROTTLE_PAST)}


class HardwareReader:
    """Reads CPU, memory, throttling, disk, Wi-Fi and this process. Callers serialize the calls:
    ``LiveState`` holds its lock during a read. ``proc_wireless`` and ``flags`` are seams for
    tests. By default ``vcgencmd`` is looked up on ``PATH`` once, here, and the result (also
    "not found") is reused by every refresh: the tools on the robot do not move at runtime."""

    def __init__(
        self,
        proc_wireless: Path = PROC_WIRELESS,
        flags: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        self._proc_wireless = proc_wireless
        self._flags = flags or functools.partial(read_throttling, shutil.which(VCGENCMD))
        self._cpu_times: Any = None
        self._process = _attempt(psutil.Process)
        self._process_sampled = False

    def __call__(self) -> dict[str, Any]:
        """The ``cpu``, ``memory``, ``throttling``, ``disk``, ``wifi`` and ``process`` blocks."""
        return {
            "cpu": self._cpu(),
            "memory": self._memory(),
            "throttling": throttle_state(_attempt(self._flags)),
            "disk": _attempt(self._disk),
            "wifi": _attempt(self._wifi),
            "process": _attempt(self._process_block),
        }

    def _cpu(self) -> dict[str, Any]:
        """Load since the previous read and the clock; each is None where it is unknown."""
        freq = _attempt(psutil.cpu_freq)
        return {
            "percent": _rounded(_attempt(self._busy)),
            "freq_mhz": None if not freq or not freq.current else round(freq.current),
            "freq_max_mhz": None if not freq or not freq.max else round(freq.max),
        }

    def _busy(self) -> float | None:
        """Take a new ``cpu_times`` sample and compare it with the previous one."""
        now = psutil.cpu_times()
        before, self._cpu_times = self._cpu_times, now
        return None if before is None else busy_percent(before, now)

    def _memory(self) -> dict[str, Any]:
        """The used share of RAM and of swap; swap is None on a system without any."""
        ram, swap = _attempt(psutil.virtual_memory), _attempt(psutil.swap_memory)
        return {
            "percent": None if ram is None else _rounded(ram.percent),
            "swap_percent": None if swap is None or not swap.total else _rounded(swap.percent),
        }

    def _disk(self) -> dict[str, Any]:
        """Free space and the used share of the root disk."""
        usage = psutil.disk_usage(ROOT_DISK)
        return {"free_gb": _rounded(usage.free / BYTES_PER_GB), "percent": _rounded(usage.percent)}

    def _wifi(self) -> dict[str, Any] | None:
        """Signal and link quality of the best associated interface; None without a link."""
        links = parse_proc_wireless(self._proc_wireless.read_text()).values()
        best = max(links, key=lambda link: link["link_quality"], default=None)
        if best is None or best["link_quality"] <= 0:
            return None
        return {"signal_dbm": best["signal_dbm"], "link_quality": best["link_quality"]}

    def _process_block(self) -> dict[str, Any]:
        """This process: its share of the whole machine, resident memory and thread count."""
        process = self._process
        cpu = process.cpu_percent(None) / (psutil.cpu_count() or 1)
        sampled, self._process_sampled = self._process_sampled, True
        return {
            "cpu_percent": _rounded(cpu) if sampled else None,
            "rss_mb": round(process.memory_info().rss / BYTES_PER_MB),
            "threads": process.num_threads(),
        }
