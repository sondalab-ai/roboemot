---
name: robot-freeze-2026-10-28-reachy-head-pose-yaw-atan2-wraps-at
description: Robot freeze 2026-10-28: Reachy head pose yaw (atan2) wraps at +-pi while commands sit at the -180 deg limit; external-control check read the 2*pi jump as a hand and yielded forever (still never reached). Fixed in PR 133: driver reads head yaw within pi of body yaw. Symptom in debug report: motion.external true, external_peak_err_rad ~6.28 on head.yaw.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-28
---
