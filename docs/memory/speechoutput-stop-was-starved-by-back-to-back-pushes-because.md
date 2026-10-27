---
name: speechoutput-stop-was-starved-by-back-to-back-pushes-because
description: SpeechOutput.stop() was starved by back-to-back pushes because CPython locks are not fair; fixed with a _stopping pending flag (epic be7c8b2, PR 123). CI flakes on PR 123 were not thread leaks.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-27
---
