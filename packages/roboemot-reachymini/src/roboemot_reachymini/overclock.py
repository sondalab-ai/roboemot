"""Reachy Mini Wireless CPU overclock: the CM4 at 1.8 GHz (SPEC 4A, Installer).

The robot's Raspberry Pi CM4 runs at its default 1.5 GHz and sits near 99 % CPU with vision on.
The installer's ``--overclock`` (``roboemot.install.overclock``) writes ``BLOCK`` at the end of
``/boot/firmware/config.txt``, only when every support check passes:

- the device tree model says ``Compute Module 4``;
- ``vcgencmd get_throttled`` is ``0x0``: no under-voltage, frequency capping, throttling or soft
  temperature limit, now or at any time since boot;
- the SoC is below ``TEMP_LIMIT_C``;
- the firmware (``vcgencmd version``) is at least ``MIN_BOOT_DATE``, and the boot config sets no
  clock or voltage of its own (``CONFLICT_KEYS``) and holds no RoboEmot marker already.

The bootloader (``vcgencmd bootloader_version``) decides the path: from ``MIN_BOOT_DATE`` on it
has ``tryboot``, so the block goes first into ``tryboot.txt`` (a full copy of ``config.txt`` plus
the block: the firmware loads it *instead of* ``config.txt``) and the robot boots it once with
``reboot '0 tryboot'``; the flag is cleared before the firmware starts, so a hang and a power
cycle boot the unchanged ``config.txt``. After that boot ``verify`` checks the clock, the flags
and the temperature before the block is promoted. An older bootloader gets ``config.txt``
written directly.

Values and sources (Raspberry Pi documentation, read 2026-11-05):

- ``arm_freq`` default is 1500 on the CM4; ``[cm4]`` is the CM4's model filter and ``[all]``
  resets every filter (config.txt, Overclocking and Conditional filters:
  https://www.raspberrypi.com/documentation/computers/config_txt.html).
- No over-voltage line: the same page says the latest firmware scales the voltage up by itself
  when the system is overclocked (Adaptive Voltage Scaling), that setting ``over_voltage`` by
  hand turns that off, and that ``over_voltage_delta`` (microvolts on top of the DVFS value) is
  the step for an unstable overclock. It documents no fixed value for 1.8 GHz, so none is
  written; ``force_turbo`` stays unset, which keeps dynamic frequency scaling.
- The Arm cores are throttled from 80 °C (same page, Monitoring core temperature). The checks
  want at least 10 °C below that with today's load at 1.5 GHz: ``TEMP_LIMIT_C`` 70.
- ``tryboot``, ``tryboot.txt``, ``reboot '0 tryboot'`` and ``/proc/device-tree/chosen/bootloader/
  tryboot`` (1 on a tryboot boot): https://www.raspberrypi.com/documentation/computers/
  raspberry-pi.html and .../configuration.html. The first stable Pi 4 family bootloader with
  tryboot is pieeprom-2020-12-11 (rpi-eeprom ``firmware-2711/release-notes.md``, promoted
  2020-12-14), hence ``MIN_BOOT_DATE``.
- Recovery when a CM4 does not boot at all: ``rpiboot`` (``usbboot``) from a computer.
"""

from __future__ import annotations

import shlex
from datetime import UTC, date, datetime

from roboemot.diagnostics.system import THROTTLE_BITS
from roboemot.install.overclock import Check, OverclockProfile, read
from roboemot.install.remote import Remote

BOOT = "/boot/firmware"
CONFIG = f"{BOOT}/config.txt"
TRIAL = f"{BOOT}/tryboot.txt"
MODEL_PATH = "/proc/device-tree/model"
MODEL = "Compute Module 4"
TRYBOOT_FLAG = "/proc/device-tree/chosen/bootloader/tryboot"  # 4-byte big-endian, 1: tryboot
CPU_MAX_FREQ = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"  # kHz
THERMAL = "/sys/class/thermal/thermal_zone0/temp"  # millidegrees Celsius, the SoC
ARM_FREQ_MHZ = 1800
DEFAULT_ARM_FREQ_MHZ = 1500  # the CM4's default arm_freq
TEMP_LIMIT_C = 70.0  # 10 °C under the 80 °C where the firmware throttles the Arm cores
MIN_BOOT_DATE = date(2020, 12, 11)  # first stable bootloader with tryboot (pieeprom-2020-12-11)
KHZ_PER_MHZ = 1000
MHZ_PER_GHZ = 1000
MILLI = 1000
BEGIN = "# >>> RoboEmot overclock: written and removed by the RoboEmot installer (SPEC 4A)"
END = "# <<< RoboEmot overclock"
OVERCLOCK_LINES = (f"arm_freq={ARM_FREQ_MHZ}",)  # no voltage line: see the module docstring
BLOCK = "\n".join([BEGIN, "[all]", "[cm4]", *OVERCLOCK_LINES, "[all]", END]) + "\n"
CONFLICT_KEYS = ("arm_freq", "force_turbo", "over_voltage")  # over_voltage: every over_voltage*
TRIAL_REBOOT = "reboot '0 tryboot'"
REBOOT = "reboot"
LABEL = f"{ARM_FREQ_MHZ / MHZ_PER_GHZ:.1f} GHz, from {DEFAULT_ARM_FREQ_MHZ / MHZ_PER_GHZ:.1f}"
WARNING = (
    "Overclock: if the board ever fails to boot, recovering the CM4's eMMC needs rpiboot "
    "(Raspberry Pi usbboot) from a computer over USB."
)
REBOOT_WARNING = (
    "The reboot stops RoboEmot and the robot's daemon, and the motors lose power, so the head "
    "may drop."
)


def _run(remote: Remote, command: str) -> str:
    """A read-only command whose failure is an empty answer, never an exception."""
    return remote.run(f"{command} 2>/dev/null || true")


def _cat(path: str) -> str:
    return f"cat {shlex.quote(path)}"


def model_check(remote: Remote) -> Check:
    model = _run(remote, f"tr -d '\\000' < {shlex.quote(MODEL_PATH)}").strip()
    return Check("Board", MODEL in model, model or f"{MODEL_PATH} unreadable")


def throttle_check(remote: Remote) -> Check:
    """``vcgencmd get_throttled`` is 0x0: no flag now and none since boot."""
    answer = _run(remote, "vcgencmd get_throttled").strip()
    try:
        raw = int(answer.split("=")[-1], 16)
    except ValueError:
        return Check("Throttling", False, f"vcgencmd get_throttled answered {answer!r}")
    flags = [name for bit, name in sorted(THROTTLE_BITS.items()) if raw >> bit & 1]
    return Check("Throttling", raw == 0, "none since boot" if raw == 0 else ", ".join(flags))


def temperature_check(remote: Remote) -> Check:
    answer = _run(remote, _cat(THERMAL)).strip()
    try:
        celsius = int(answer) / MILLI
    except ValueError:
        return Check("Temperature", False, f"{THERMAL} answered {answer!r}")
    return Check(
        "Temperature", celsius < TEMP_LIMIT_C, f"{celsius:.1f} °C, limit {TEMP_LIMIT_C:.0f} °C"
    )


def firmware_date(answer: str) -> date | None:
    """The build date on the first line of ``vcgencmd version`` (``Mar 17 2023 10:52:00``)."""
    first = " ".join(answer.strip().splitlines()[0].split()) if answer.strip() else ""
    try:
        return datetime.strptime(first, "%b %d %Y %H:%M:%S").date()
    except ValueError:
        return None


def bootloader_date(answer: str) -> date | None:
    """The build date of ``vcgencmd bootloader_version``: its ``timestamp`` line (UTC), else a
    first line like ``2023/01/11 17:40:52``."""
    lines = [ln.strip() for ln in answer.strip().splitlines()]
    for line in lines:
        key, _, value = line.partition(" ")
        if key == "timestamp" and value.strip().isdigit():
            return datetime.fromtimestamp(int(value), UTC).date()
    try:
        return datetime.strptime(lines[0], "%Y/%m/%d %H:%M:%S").date() if lines else None
    except ValueError:
        return None


def firmware_check(remote: Remote) -> Check:
    built = firmware_date(_run(remote, "vcgencmd version"))
    if built is None:
        return Check("Firmware", False, "vcgencmd version gave no build date")
    return Check("Firmware", built >= MIN_BOOT_DATE, f"built {built}, needs {MIN_BOOT_DATE} on")


def tryboot_check(remote: Remote) -> tuple[bool, Check]:
    """Whether the bootloader has tryboot; either way a passed check that names the path."""
    built = bootloader_date(_run(remote, "vcgencmd bootloader_version"))
    if built is not None and built >= MIN_BOOT_DATE:
        return True, Check("Bootloader", True, f"built {built}: trial boot through tryboot")
    why = f"built {built}" if built else "no build date"
    return False, Check("Bootloader", True, f"{why}, no tryboot: {CONFIG} written directly")


def conflicts(text: str) -> list[str]:
    """Lines of the boot config that already set a clock or a voltage (``CONFLICT_KEYS``)."""
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        key = line.split("#", 1)[0].split("=", 1)[0].strip()
        if "=" in line.split("#", 1)[0] and key.startswith(CONFLICT_KEYS):
            found.append(f"{key} (line {n})")
    return found


def config_check(text: str | None, tryboot: bool, trial: str | None) -> Check:
    """The boot config can take the block: it exists, sets no clock or voltage of its own, has
    no RoboEmot marker yet, and no ``tryboot.txt`` that is not ours is in the way."""
    if text is None:
        return Check("Boot config", False, f"{CONFIG} not found")
    lines = text.splitlines()
    if BEGIN in lines or END in lines:
        return Check("Boot config", False, f"{CONFIG} already has a RoboEmot block")
    own = conflicts(text)
    if own:
        return Check("Boot config", False, f"{CONFIG} already sets {', '.join(own)}")
    if tryboot and trial is not None:
        return Check("Boot config", False, f"{TRIAL} exists and is not the installer's")
    return Check("Boot config", True, f"{CONFIG}, nothing that conflicts")


class ReachyOverclock:
    """``Platform.overclock`` of the Reachy Mini Wireless (``roboemot.install.overclock``)."""

    warning = WARNING
    config_path = CONFIG
    trial_path = TRIAL
    block = BLOCK
    reboot = REBOOT
    trial_reboot = TRIAL_REBOOT
    reboot_warning = REBOOT_WARNING

    def __init__(self, reboot_hint: str) -> None:
        self.reboot_hint = reboot_hint

    def support(self, remote: Remote) -> tuple[OverclockProfile | None, list[Check]]:
        """Every check, read-only; the profile only when all of them pass."""
        tryboot, boot = tryboot_check(remote)
        text = read(remote, CONFIG)
        checks = [
            model_check(remote),
            throttle_check(remote),
            temperature_check(remote),
            firmware_check(remote),
            boot,
            config_check(text, tryboot, read(remote, TRIAL) if tryboot else None),
        ]
        if not all(c.ok for c in checks) or text is None:
            return None, checks
        return OverclockProfile(LABEL, CONFIG, TRIAL if tryboot else None, BLOCK, text), checks

    def trial_booted(self, remote: Remote) -> bool:
        """The bootloader's tryboot flag in the device tree is 1 (this boot used tryboot)."""
        words = _run(remote, f"od -An -tu1 {shlex.quote(TRYBOOT_FLAG)}").split()
        return bool(words) and all(w.isdigit() for w in words) and _be(words) == 1

    def verify(self, remote: Remote) -> list[Check]:
        """After the trial boot: the firmware runs ``arm_freq`` 1800, the kernel's top CPU
        frequency is 1800 MHz, no throttling flag is set and the SoC is below the limit."""
        config = _run(remote, "vcgencmd get_config arm_freq").strip()
        top = _run(remote, _cat(CPU_MAX_FREQ)).strip()
        want_khz = str(ARM_FREQ_MHZ * KHZ_PER_MHZ)
        return [
            Check("arm_freq", config == f"arm_freq={ARM_FREQ_MHZ}", config or "no answer"),
            Check("CPU top frequency", top == want_khz, f"{top or '?'} kHz, want {want_khz}"),
            throttle_check(remote),
            temperature_check(remote),
        ]


def _be(words: list[str]) -> int:
    """Bytes (decimal words of ``od``) as one big-endian integer."""
    return int.from_bytes(bytes(int(w) for w in words), "big")


def block_state(config: str | None, trial: str | None) -> dict[str, bool]:
    """Whether the RoboEmot block is in the boot config, and a trial config with it waits."""
    return {
        "block_present": config is not None and BEGIN in config.splitlines(),
        "trial_pending": trial is not None and BEGIN in trial.splitlines(),
    }


__all__ = ["BLOCK", "CONFIG", "TRIAL", "ReachyOverclock", "block_state"]
