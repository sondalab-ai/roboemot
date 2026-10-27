---
name: robot-asr-failures-2026-10-26-ciao-neve-toner-root-cause-was
description: Robot ASR failures (2026-10-26, 'ciao neve'->'toner'): root cause was ~20% of mic audio dropped before ASR (voice queue 10 s, listener behind by up to 17 s on CM4); a new Vosk KaldiRecognizer per utterance was slow to build and decoded ~3x slower while warming. Slice ZU reuses one recognizer per language and moves the grammar pass off the listener thread. The Reachy daemon holds the ALSA mic exclusively, so arecord on the robot fails with 'Device or resource busy'.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-10-26
---
