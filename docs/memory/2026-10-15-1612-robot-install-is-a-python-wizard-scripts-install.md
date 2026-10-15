---
name: 2026-10-15-1612-robot-install-is-a-python-wizard-scripts-install
description: Robot installer is robot-agnostic (roboemot.install) with per-robot platforms via the roboemot.installers entry point; uninstall undoes only its own recorded changes because Reachy apps share /venvs/apps_venv.
metadata:
  type: project
---
`scripts/install_on_robot.py` is a thin wrapper over `roboemot.install` (core). Reachy Mini
Wireless specifics live in `roboemot_reachymini.install.ReachyMiniWireless`, registered in the
`roboemot.installers` entry-point group under the manifest id (same pattern as diagnostics).

**Why:** the user asked for "agnostic first, then specialized", and for an uninstall that
leaves the robot clean. Reachy apps share `/venvs/apps_venv` (verified on daemon 1.11), so a
full snapshot restore would clobber other apps; the change record
(`/venvs/.roboemot-install.json`) holds only what this installer added or replaced.

**How to apply:** keep `--no-cache-dir` on every pip call and keep dry runs write-free (tests
assert both); new robot models add a platform class, not branches in core.
Related: [[reachy-mini-wireless-cm4-exposes-no-battery-to-linux]]
