---
name: robot-wake-misses-2026-11-04-cloud-voice-profile-not-asr-all
description: Robot wake misses (2026-11-04, cloud-voice profile): not ASR. All utterances passed voice-at-face but failed voice.facing (yaw 40-73 deg vs max 30; owner frontal read 17-23 on 11-02). Facing runs before the wake word; Vosk then decodes only the first 3 s (wake_decode_max_s) and bare 'Neve' wakes only in finals <= alias_wake_max_words (4). Candidates: real turn, wrong face (yaw history per world entry, unknown faces fold into owner), bias at 1280x720 (eye span 5-10 detector px), timing. Soothe hums open the speaking window and can cut a leading 'Neve' (dropped_speaking).
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-11-04
---
