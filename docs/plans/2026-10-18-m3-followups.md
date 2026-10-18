# M3 follow-ups — epic plan

Integration branch `epic/m3-followups`, one PR per slice, one merge to `main`.
Scope: the items that need neither the owner nor the real robot.

## Owner decisions (2026-10-18)
- Barge-in latency (SPEC 13) is measured from the first sound, for every language.
- Piper as a GPL subprocess (`piper_worker.py`, own venv) is acceptable.
- Children's built-in lines and blocklist keep their defaults until they are editable from the web UI.
- Wake phrases stay as they are (synthetic-voice recall says little about natural voices).

## Slices
1. Decisions into SPEC/TODO; `scripts/robot_ssh.py` (mDNS discovery via `roboemot.install`, ssh, `--copy-id`, `--debug`);
   Wi-Fi in the debug report (core `network.wifi` local source; Reachy `robot.wifi` from `GET /wifi/status`).
2. CI job for `-m models` (cached voice models, Piper on the Linux runner); fix the flaky tests
   (`test_app_conversation.py::test_the_app_offers_the_default_skills_and_wires_barge_in`,
   `test_cli_llm_check::test_exit_zero_when_every_check_passes`,
   `test_llm_router::test_streams_text_and_charges_reported_usage`).
3. English barge-in onset: stop within 200 ms of the first sound (the /s/ of "stop") without new false triggers.
4. Scan soon after a person leaves (SPEC 5A); `NetworkMonitor` follows routes added by hot reload.
5. Pseudonyms to cloud routes (SPEC 12), including `look_at` names.
6. Named gestures `nod` / `shake` for the `play` skill.

## Out
Secret store (key management is an owner design call), web UI settings editor (wake phrases, name,
children's lines), gaze-shift scaling (gaze shifts do not run yet), anything needing the robot (SPEC 15).
