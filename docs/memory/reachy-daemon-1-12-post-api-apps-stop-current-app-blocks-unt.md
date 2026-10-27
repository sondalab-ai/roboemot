---
name: reachy-daemon-1-12-post-api-apps-stop-current-app-blocks-unt
description: Reachy daemon 1.12 POST /api/apps/stop-current-app blocks until the app exits (<=20 s SIGINT->kill) + 1 s goto-zero; any client calling it needs a >=21 s timeout (installer now 30 s, slice R PR 70).
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-19
---
