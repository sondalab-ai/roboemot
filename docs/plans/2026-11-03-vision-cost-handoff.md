# Handoff 2026-11-03: robot checks and vision cost slices

State on `main` (`ed2a95e`): V-1, V-2, V-6, CV-20, CV-21, the nap wake stimuli (#171) and the quieter soothe sounds (#172) are merged. No open PRs, no worktrees.

## 1. On the robot (owner)

1. Reinstall with `python scripts/install_on_robot.py --cloud-voice --host <robot>`. The profile sets `voice.realtime_route.tone_tool: true`.
2. After some use, run `roboemot debug --host <robot> --json` and save the output. Read:
   - `stage_timing`: CPU cost of each vision stage (YuNet, SFace, DINOv2, NanoDet), including NanoDet during a shown-object look.
   - `object_gate`: how often object detection runs or skips, and why.
   - `roboemot.attention.shown_object`: episodes, how each ended, and how long each lasted.
   - `tone_retry` against `tones`: how often the model calls the tone tool without speaking in the same response.
3. Hold up a cup and move it. Check that the robot looks at it and follows it while it moves.
4. Listen to the soothe hums at -24 dBFS. If they are too quiet or too loud, change the per-variant `soothe.variants[].shape.gain_db`.

## 2. Slices that wait for the step 2 numbers

Start none of these before the `stage_timing` numbers from the robot exist. Each one is a single slice PR into `main`.

| Slice | Change | Start only if |
| --- | --- | --- |
| V-3 | Lower `detection_hz` (face detection) while nobody is in view | YuNet is a large share of `cpu_share` |
| V-4 | Run DINOv2 less often per tracked object | DINOv2 is a large share of `cpu_share` |
| V-5 | INT8 SFace model, opt-in | SFace is a large share of `cpu_share` |

## 3. Proposed, not approved

- Put the shown object into the conversation context ("the person is showing you a cup").
