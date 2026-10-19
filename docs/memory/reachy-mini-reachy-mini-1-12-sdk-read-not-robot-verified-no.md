---
name: reachy-mini-reachy-mini-1-12-sdk-read-not-robot-verified-no
description: Reachy Mini (reachy-mini 1.12 SDK read, not robot-verified): no daemon reboot route; app restart via POST /api/apps/restart-current-app (SIGINT, 20s then kill, daemon goto zero after); pollen has passwordless sudo via /etc/sudoers.d/010-pollen-reachy (daemon runs 'sudo shutdown -h now', exact-argv 'sudo systemctl restart reachy-mini-daemon'); reboot coverage needs 'sudo -n -l' on robot.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-18
---
