---
name: reachy-mini-head-pitch-positive-is-nose-down
description: Reachy Mini head pitch read by the driver is right-handed about y (REP-103), so positive pitch tilts the nose down; world-frame elevation from a pitched head is -pitch.
metadata:
  type: project
---

`roboemot_reachymini/driver.py` turns the SDK head pose into angles with scipy Euler `xyz`
(roll, pitch, yaw) in a frame with x forward, y left, z up. Pitch is a right-handed rotation
about y, so positive pitch = nose down. M1 world model (SPEC 5A, Frame) first shipped with
the opposite sign; review of PR #4 caught it (2026-10-16).

**Why:** a wrong sign flips the elevation of every observation made with the head tilted, so
the world map and attention aim at the wrong height.
**How to apply:** any code mapping joint state to sensor bearings uses REP-103 conventions;
pin the sign with a test against the driver's convention. Still worth one check on the real
robot (SPEC 15, verify list).
