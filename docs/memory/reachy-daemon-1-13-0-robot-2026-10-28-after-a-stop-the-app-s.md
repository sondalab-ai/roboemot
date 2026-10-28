---
name: reachy-daemon-1-13-0-robot-2026-10-28-after-a-stop-the-app-s
description: Reachy daemon 1.13.0 (robot 2026-10-28): after a stop, the app state can stay 'stopping' though the process exited; start-app 400 'An app is already running', stop/restart-current-app 400 'No app is currently running', POST /api/daemon/restart does not clear it; only a robot reboot does (sudo needs an interactive robot_ssh.py shell). run_on_robot.py now detects it.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-28
---
