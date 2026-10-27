# QA handoff: robot plugin hierarchy + realtime speech route (2026-10-27)

State: the integration branch `epic/robot-plugin-hierarchy` is merged to `main`. Plans: `docs/plans/2026-10-19-robot-plugin-hierarchy.md`, `docs/plans/2026-10-26-realtime-speech-route.md`. The per-plan SDD workspaces are deleted; this file carries what is still open.

## Update the robot

```
cd <repo checkout on main>
git pull --ff-only
. .venv/bin/activate
pip install --no-deps -e packages/roboemot-core -e packages/roboemot-reachymini
python scripts/install_on_robot.py --dev
```

Owner rules for agents: an agent does not move the motors, does not start or restart RoboEmot, and asks before any SSH write to the robot. The robot SSH password is never stored.

## 1. Smoke test of the robot slices not yet tested on the robot

Last robot test: 2026-10-26 on 86c6b7f (ASR garbled; root cause: dropped mic audio, fixed in ZU). Not yet tested on the robot: ZU (listener real time), ZV (barge-in ducking, paced speaker), and the earlier batch ZM, ZN, ZO, ZP, ZQ, ZR, ZS, ZT1, ZT2 where the 2026-10-26 run did not cover them.

Checks, from the debug report (`roboemot debug --host <robot> --json`) and on the robot:
- `voice.realtime.share` below 0.5 while listening; `voice.dropped` stays 0.
- `ps -L -o tid,ni,comm -p <roboemot pid>`: background threads have the `runtime.background_threads` nice values; real-time threads keep 0.
- Barge-in over music: the robot's voice ducks while you talk over it, and recovers.
- `speech.pacing.underruns` stays 0 during long answers.
- "ciao Neve" / "hey Neve" wakes the robot; music with lyrics from a speaker away from any face does not get an answer (ZR).
- Motion while speaking (ZT, preset lively): the head and antennas move with the voice, no step at the end of a sentence.

## 2. Realtime speech route on the robot (SR-5, hardware in the loop)

Off by default. Nothing below runs unless the admin enables it.

Known limitation: the Reachy provider sends the HF token only to the allocator's own origin (`KEY_HOSTS = ()` in `packages/roboemot-reachymini/src/roboemot_reachymini/realtime.py`). Until the real connect host is declared there, a keyed `pollen` route fails with `url_key_host` and the local pipeline answers. Use a read-only HF token.


Updated in fix round 1. Before you start: the robot runs this branch's app, with the `pollen` route (see `configs/companion.yaml`) and `voice.realtime_route` `enabled: true`, `mode: platform`, `input: audio`, and `privacy.realtime_audio` on. The HF token is stored with `roboemot secrets set hf_token`.

1. **Installer dry run** (SR-4 concern 2): `python scripts/install_on_robot.py --dry-run`. Expect `would install the realtime extra (websockets <version>, already present)` and no new package.
2. **Realtime check, to learn the connect host.** Prefer the web UI: Routes tab, `pollen`, **Check** (admin session). It uses the app's own config and provider.
   - From a shell, use the venv's full path: a non-interactive ssh has no venv on `PATH`. Run `python scripts/robot_ssh.py -- /venvs/apps_venv/bin/roboemot realtime check pollen --app ~/.local/share/roboemot/app.yaml --robot reachy-mini-wireless --json`.
   - `app.yaml` is in the data folder the installer used. The CLI reads the secret store from the config's `data_dir`, which is the same default folder unless you installed with `--data-dir`; then pass the `app.yaml` from that folder.
   - The CLI does not see the web UI's runtime overrides, and it charges only its own process's budget, not the app's.
   - While `KEY_HOSTS` is empty, expect `failure: url_key_host`.
   - **Watch `roboemot.realtime_route` `state`** in the debug report during the check. The check opens a second session with the same token next to the app's live one; record whether the live session stays `connected` (SPEC 15).
   - Field meanings:
     - `mode`: the provider the check used (`platform` or `direct`; a mode change takes effect only at a restart).
     - `allocator_host`: the host of the route's `base_url`, in platform mode.
     - `connect_host`: the host the allocator handed out. **Declare this exact host in `KEY_HOSTS`** in `roboemot_reachymini/realtime.py`.
     - `header_to_connect_host`: `sent`, `refused` (`url_key_host`), `not_sent` (the connect stopped before the handshake), or `none` (the route has no key).
     - `platform_id_sent`: whether `hardware_id` went with the allocation (only with `privacy.realtime_platform_id`); never the value.
     - `update_seen`: the server answered `session.updated`.
     - `first_audio_ms`: from `response.create` of one text turn to the first audio delta.
     - `item_delete`: `accepted`, `refused`, `unconfirmed` (no answer within `stall_timeout_s` or by the deadline) or `not_run`.
     - `charged`: tokens charged.
     - `failure`: a reason code (`url_key_host`, `provider_http_<code>`, `connect_timeout`, `server_error`, `stall`, `send_failed`, `pseudonyms`, ...).
3. **Declare the host and recheck.** Put `connect_host` in `KEY_HOSTS`, redeploy, and check again. Expect `header_to_connect_host: sent`, `update_seen: true`, a `first_audio_ms` figure and `item_delete: accepted`. Try `privacy.realtime_platform_id` only if the allocator refuses anonymous sessions.
4. **HIL measurements**, from a Mac on the LAN, while the app runs. Restart the app first: the latency figures cover its last 50 turns. Then run:
   - `read -s ROBOEMOT_HIL_PIN && export ROBOEMOT_HIL_PIN` (type the web UI PIN: it stays out of files and history).
   - `ROBOEMOT_HIL_WEBUI=http://<robot>:8042 ROBOEMOT_HIL_ROUTE=pollen pytest -m hil -s -p no:randomly packages/roboemot-reachymini/tests/test_realtime_hil.py`.
   - Tests run in file order: latency, underruns, barge-in (five interrupts, each stop timed on its own), nod (answer y or n), "Neve" looking away (local Vosk wake word), "Neve" facing the robot (server transcript), item ids, TV, and last the listener rounds.
   - The item-ids test sets `conversation.history_turns` to 1 for its round and restores it.
   - The listener test asks you to turn the route off and restart the app between its rounds, and to turn it back on and restart at the end. The file logs in again by itself after a restart or when the session ages out, waiting up to 180 s for the app.
   - Each `HIL <name>: <json>` line is a figure for SPEC 13 or 15:
     - `transcript_ms` and `first_audio_ms` (count, p50, p95, max) and `server_stages_ms`;
     - `underruns` and `late_sentences`;
     - `barge_in_onset_to_stop_ms` (one value per stop) and `route_cuts`;
     - `nod_sync`;
     - `neve_local_wake_word`, and `neve_server` (heard, transcribed);
     - `realtime_check`, `live_state_around_check`, and `item_ids` (`tool_calls`, `items_deleted`, `items_delete_confirmed`, `items_truncated`, `items_truncate_confirmed`, `errors_unrelated`);
     - `tv_audio_sent` and `tv_upload_s`;
     - `listener_rounds`: per mode, the share over the round from the listener's totals, `budget_share`, `vosk_skipped`, and the mean CPU sampled during the round with the sample count.
   - A failed assert means a budget or an expectation was missed: first audio p95 over 2000 ms, any underrun, a stop over 200 ms, nod answered n, "Neve" heard fewer than 4 of 5 times on either path, a delete or truncate not confirmed by the server, audio sent with the TV, or a share over budget. If the server sends no `conversation.item.truncated`, the truncate assert fails by design: record truncation as unverified in SPEC 15.
5. **Record the figures** in the SPEC 13 rows ("pending HIL") and close the SPEC 15 items, including the second-session question. Do the owner A/B listen (Piper against the server's voice) as its SPEC 15 item says.


## 3. Backlog (not blocking QA)

- Realtime: with `runtime.pause_when_asleep` on, a text turn already queued can reopen the realtime session while the robot sleeps (pre-existing, text only).
- Realtime, server behaviour to confirm in step 2: item ids on input events, deletes and truncates confirmed, `server_silence_ms` honoured, smart-turn hold, `audio_start_ms` origin and 300 ms prefix padding, CM4 local decode time of a maximum-length utterance against `local_decode_timeout_s`, whether a second session with the same token disturbs the live one.
- Face tracking: rebirth distance p50 0.66 against the 0.6 gate; tune after a robot run with the ZK diagnostics.
- Barge-in echo buffer may drop the playing audio for sentences over 2 s (`voice/bargein.py`, noted during ZT design). Verify against the current code.
- Flaky test: `packages/roboemot-reachymini/tests/test_app_soothe.py::test_a_sound_plays_next_to_a_visible_person_and_a_queued_sentence_cuts_it` (3 of 4 runs on 127abdf).
- Voice delivery: try `voice.mood_source: llm_tagged` after the robot test (owner found the delivery flat).
- Installer: check that the main-flow pip steps no longer print a timed ✓ before the final "installed" line (slice J deferral; main's installer output changed since).
- README status rows: update for the robot plugin hierarchy and the realtime route.
