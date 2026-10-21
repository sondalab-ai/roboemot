---
name: reachy-mini-daemon-1-12-robot-verified-2026-10-20-venvs-apps
description: Reachy Mini daemon 1.12 (robot verified 2026-10-20): /venvs/apps_venv is created lazily on the first app install (fresh Debian 13 image has only /venvs/mini_daemon); AppManager.is_app_running counts ERROR as running, so an errored app blocks every start until stop-current-app; the mobile app holds the robot app slot (GET /api/daemon/robot-app-lock-status = remote_session) and the daemon refuses local app starts incl. the startup app; web dashboard is retired.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-21
---
