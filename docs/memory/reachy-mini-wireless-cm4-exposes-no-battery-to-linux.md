---
name: reachy-mini-wireless-cm4-exposes-no-battery-to-linux
description: Reachy Mini Wireless (CM4, daemon 1.11): no battery data anywhere in software; /sys/class/power_supply empty, daemon has no battery API; vcgencmd works for pollen. SSH is pollen/root, key not installed.
metadata:
  type: project
---
Checked on Neve 2026-10-15: `/sys/class/power_supply` is empty and no fuel-gauge driver is bound;
the daemon REST API has no battery endpoint and only logs motor input-voltage errors.
`vcgencmd get_throttled` / `measure_temp` work as `pollen` (throttled=0x0, 54.5 C at idle).
I2C devices seen: 0-000c, 0-001a, 10-000c, 10-001a, 10-002f.

**Why:** the debug report (SPEC 12A, Diagnostics) can only show supply health via under-voltage flags.
**How to apply:** don't promise battery charge on Reachy; if needed, ask Pollen how the battery board
is read. SSH login is password `root`; no key installed yet, so `ssh -o BatchMode=yes` fails.
Related: [[reachy-mini-sdk-1-11-1-12-wsclient-never-reconnects-after-th]]
