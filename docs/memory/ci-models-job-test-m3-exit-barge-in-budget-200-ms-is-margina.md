---
name: ci-models-job-test-m3-exit-barge-in-budget-200-ms-is-margina
description: CI 'models' job: test_m3_exit barge-in budget 200 ms is marginal on GH runners (observed 88-268 ms over 38 runs, fails ~10% incl. main); Opus A/B on CI (run 39173499497, 2026-10-19) proved it is not a config regression: 183cd00 and 6fc2faf both fail 3/6 in the same run, median ~200 ms; causes are listener lag behind the mic (chunks 140-163 ms late under CPU contention from Piper/onnx/Vosk) and gen-2 GC pauses of 200-220 ms in the window. Rerun before suspecting a change; real fix = profile the ~150 ms detect_to_stop gap or best-of-N in the test (2026-10-19).
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-19
---
