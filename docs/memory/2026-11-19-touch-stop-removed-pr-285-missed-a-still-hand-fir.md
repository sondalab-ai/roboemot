---
name: 2026-11-19-touch-stop-removed-pr-285-missed-a-still-hand-fir
description: 2026-11-19: touch stop removed (PR 285: missed a still hand, fired on the robot's own head-yaw overshoot). Radio now stops by voice without the name (PR 283, voice.keywords media/media_stop, per-language only, owner: no language mixing) and by a held fist (PRs 286-288, vision.hands, on by default, loads only with tools.radio.enabled+fist_stop; int8 hand-landmark model unusable, fp32 default; CPU cost with the check on is unmeasured on the robot).
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-11-19
---
