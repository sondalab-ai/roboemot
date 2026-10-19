---
name: ci-models-job-test-m3-exit-barge-in-budget-200-ms-is-margina
description: CI 'models' job test_m3_exit barge-in 200 ms budget failed ~50% (incl. main) because BargeIn ran Vosk keyword spotting inline on the listener thread (KaldiRecognizer build 44-76 ms + whole-ring accept 65-224 ms), plus gen-2 GC pauses. Fixed in PR 63 (2026-10-19): KeywordWorker thread + runtime.gc_freeze; detect_to_stop went 107-183 ms -> max 81 ms, 20/20 CI runs green. If it regresses, check keyword_seconds/detect_to_stop in the assertion message first.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-19
---
