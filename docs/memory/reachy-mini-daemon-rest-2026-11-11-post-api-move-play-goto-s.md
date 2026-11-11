---
name: reachy-mini-daemon-rest-2026-11-11-post-api-move-play-goto-s
description: Reachy Mini daemon REST (2026-11-11): POST /api/move/play/goto_sleep disables the motors, but POST /api/move/play/wake_up alone does not enable them again; send POST /api/motors/set_mode/enabled first, then wake_up. Robot baseline awake on the review branch: RoboEmot 173% of one core, 2358 wake-ups/s; asleep 35.9%, 1426 wake-ups/s (mostly audio/GStreamer threads).
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-11-11
---
