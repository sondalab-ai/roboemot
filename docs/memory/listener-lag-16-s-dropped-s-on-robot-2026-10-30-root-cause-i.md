---
name: listener-lag-16-s-dropped-s-on-robot-2026-10-30-root-cause-i
description: Listener lag 16 s / dropped_s on robot (2026-10-30): root cause is the Reachy audio hub drain (roboemot_reachymini/audio.py poll_once/_run): after a >150 ms poll gap it reads up to DRAIN_MAX=1000 chunks (16 ms each = 16 s) before pushing, last=now is set before the poll so drains chain; hub dropped_samples equals listener dropped_s. Listener stages use only 12-14% of real time. Fix proposed (CV-0), awaiting owner OK.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-30
---
