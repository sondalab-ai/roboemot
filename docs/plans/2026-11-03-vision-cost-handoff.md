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
| V-5 | Dropped 2026-11-04 (owner decision) | INT8 SFace measured slower on the Mac: OpenCV 18.8 ms vs FP32 5.3 ms; ONNX Runtime 12.1 ms. The CM4 (Cortex-A72) has no int8 dot-product instructions. |

## 3. Approved 2026-11-04

- **Sticky identity for still objects (done 2026-11-04, https://github.com/sondalab-ai/roboemot/pull/185).** A recognised object that sits still loses its identity: gate-spaced object runs (about 10 s apart) exceed `identity.vote_max_age_s` (10 s), so a known mug can turn unknown and get a long novelty look again (`attention/interest.py`, `familiar_novelty`). Keep a recognised identity until a strong contrary vote arrives, instead of letting it lapse. Principle: best UX drives every vision catalogue decision (owner, 2026-11-04).

- **Wake path test on the robot (owner, later; now section C of `docs/plans/2026-11-04-qa-handoff.md`).** Diagnosis 2026-11-04: the facing check, not ASR, blocks answers (yaw 40-73 deg vs 30). Run 5 times each, then send `roboemot debug`: (1) alone at 50 cm looking at the robot, no name; (2) same, looking away; (3) looking away, "Neve, come stai?"; (4) looking away, "Neve" plus 5+ words; (5) looking away, "ciao Neve" plus a long sentence; (6) as 1 with a second person 1 m aside; (7) turn from the other person and speak at once vs face the robot 3 s first. Step 1 > 30 deg: bias (raise `voice.facing.max_yaw_deg` or recalibrate `vision.facing_nose_depth_ratio`); 1 passes and 6 fails: wrong face (keep yaw per face track); only 7 fails: timing (judge yaw after speech start).
- **Flaky tests (done 2026-11-04, https://github.com/sondalab-ai/roboemot/pull/186)** blocking merges: `test_app_soothe.py::test_a_sound_plays_next_to_a_visible_person_and_a_queued_sentence_cuts_it`, `test_realtime_turns.py::test_a_late_calls_items_are_deleted_with_its_turn` (Python 3.11 CI). Make them timing-independent.

## 4. Proposed, not approved

- Put the shown object into the conversation context ("the person is showing you a cup").
