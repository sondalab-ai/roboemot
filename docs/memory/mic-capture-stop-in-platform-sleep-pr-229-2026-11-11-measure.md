---
name: mic-capture-stop-in-platform-sleep-pr-229-2026-11-11-measure
description: Mic capture stop in platform sleep (PR 229, 2026-11-11) measured on the robot: asleep load 36.2% -> 15.3% of one core, wake-ups 1399 -> 453 per second; 10 of 10 sleep/wake cycles restarted the capture and the listener. Remaining asleep load: SDK Python threads (recv_events, _recv_loop) about 7.5%, the camera feed thread queue0:src about 5%, motion-loop 2%. reachy-mini 1.12/1.13 use one GStreamer pipeline for capture and playback: stop_recording sets it to NULL, so the speaker is silent too.
metadata:
  node_type: memory
  loadout_kind: memory
  scope: repo
  created: 2026-11-11
---
