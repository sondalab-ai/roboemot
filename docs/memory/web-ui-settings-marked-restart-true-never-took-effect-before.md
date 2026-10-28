---
name: web-ui-settings-marked-restart-true-never-took-effect-before
description: Web UI settings marked restart=True never took effect before b5294a4 (PR 131, 2026-10-28): the overrides file was replayed in start_webui, after assemble(). Now Runtime.run calls restore_settings() first. If a page setting seems ignored after a restart, check that order first.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-28
---
