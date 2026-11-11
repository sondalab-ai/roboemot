---
name: robot-profiling-2026-11-11-review-branch-cm4-1-8-ghz-awake-s
description: Robot profiling 2026-11-11 (review branch, CM4 1.8 GHz, awake, steady state): RoboEmot about 147% of one core; vision thread 33% + ONNX pool 32% + embed 13%; listener 11.5% (Silero VAD about 8% at 2.4 ms/call, music about 3%); motion-loop 10.5%; audio hub 8%; camera pump 7.4%. A 32% listener reading 90 s after app start was a start-up value, not steady state. Method without sudo: 'pip install --no-cache-dir --target /tmp/x py-spy' with the apps venv pip, then '/tmp/x/bin/py-spy record --pid P --threads --format raw' (blocking mode gives on-CPU samples; --nonblocking gives wall samples that overstate Session.run), remove /tmp/x after. Per-thread CPU and wake-ups: /proc/P/task/*/stat and status, grouped by comm (thread names exist since PR 225).
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-11-11
---
