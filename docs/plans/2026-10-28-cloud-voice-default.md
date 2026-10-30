# Cloud voice default (Reachy install): epic plan

**Goal:** On the robot, all voice interaction goes through the Pollen realtime route by default, with no local speech
synthesis, so the conversation feels organic and the CM4 is not overloaded.

**Architecture:** The realtime route (SR-1..SR-5, `docs/plans/2026-10-26-realtime-speech-route.md`) already does
transcription, answer and voice on the server with `input: audio`. This epic fixes the robot-side listener lag,
sends a configurable transcription language, adds a speech backend `none` (non-verbal cues replace every line only
Piper speaks, plus a listening cue), a skip tool the server LLM calls for an utterance not meant for the robot, and an
installer profile `--cloud-voice` that writes these settings into the robot's `app.yaml`. Library defaults stay off
(SPEC 12).

**Tech stack:** Python 3.11+, the realtime client (`core/realtime/`), the conversation worker, the robot-agnostic
installer (`roboemot.install`) and the Reachy installer plugin.

Paths: `core/` = `packages/roboemot-core/src/roboemot/`, `rm/` = `packages/roboemot-reachymini/src/roboemot_reachymini/`.

---

## Context

Robot QA, 2026-10-28, on `41eee34` (debug reports at 09:43, 09:56 and 09:59, local pipeline, route off):
- Piper is slower than real time on the CM4: `tts_rtf_sentence` p50 1.10, p90 1.17. Answers queue: the last one had
  `conversation.latency_ms.first_audio` 12733 ms and `speech.first_audio_ms_last` 18054 ms, against an LLM
  `first_sentence` of 3611 ms. `late_ms_total` 5638 ms over 12 sentences.
- Vosk small misrecognises free speech. The wake word was accepted once out of about four tries
  (`rejected_no_window` 4, `grammar.guarded.wake` 1).
- CPU 80–92 %, load average 4.3 on 4 cores.
- One false barge-in cut (`bargein.triggers` 1, no speech from the owner). Not explained; typing did not reproduce
  it in two runs, one of them with `onset_enabled` on. Out of scope here (see Open items).

## Owner decisions (2026-10-28, binding)

1. **No local speech synthesis** and nothing that overloads the CPU and weakens the experience. Voice interaction goes
   to the Pollen route by default on the robot.
2. **Where the default lives: the robot install.** The installer writes it into `app.yaml`. Library defaults stay off
   (SPEC 12: audio leaving the robot is an admin opt-in).
3. **Cloud down: non-verbal cue.** A gesture (antennas, head) plus a short recorded sound. No Piper fallback.
4. **Which LLM answers: Pollen's** (confirms owner decision 1 of 2026-10-26). The persona reaches it only as
   instructions; the chat routes answer only while the route is down, and then without voice (decision 3).
5. **Wake policy `never`**, with an instruction for the server LLM to skip an utterance that is out of context
   (design A below: a skip tool). **Amended 2026-10-30 (owner):** a video near the robot passed the voice-at-face
   check (debug report 08:33 UTC: `not_required` 44 of 45, `passed` 29, `degraded_passed` 15 via `muted`) and the robot
   commented on it. Without the wake word, an utterance is addressed only while the speaker's face looks at the robot
   (CV-6). Above a CPU threshold the facing check is not required and the gate falls back to `never` plus a strict skip
   instruction (CV-3).
6. **Opt-in only with `--cloud-voice`** (or a yes on the terminal prompt). `-y` alone does not turn on
   `privacy.realtime_audio`.
7. **A cloud-voice install removes the recorded `piper-venv`.** Switching back to local voice needs a reinstall.
8. **Vision: no change.** `vision.rates.conversation_scale` is already 0.5 while the robot thinks or speaks, and 0.5 is
   the schema floor (SPEC 13 identity budget). CPU is measured again after Piper leaves the robot.

### Privacy consequence of decision 5 (owner informed 2026-10-28)
With `never`, the audio of every utterance spoken while facing the robot (voice-at-face check) leaves the robot,
phone calls included. The skip decides only whether the robot answers; the audio is already at Pollen. The
`--cloud-voice` prompt says this in plain words.

## Verified facts (code read 2026-10-28)

- In audio mode the listener skips Vosk only when the gate would address the utterance without the wake word
  (`core/voice/listener.py:580-587`, `_addressable`). With policy `never` every face-at-voice utterance skips Vosk,
  so local Vosk runs only for utterances that are not addressable.
- `voice.backend` is `Literal["piper", "openai_speech", "kokoro"]`, `voice.fallback_backend` is
  `Literal["piper", "kokoro"]` (`core/config/schema.py:2939-2944`). `build_backend` always builds a `PiperBackend`
  (`core/voice/factory.py:26-36`).
- Fixed lines (confirmation question, fallback, offline cue, health) go to the server voice while the route is up
  and to Piper (`_say_local`) while it is down (`core/conversation/worker.py:988-1050`, `1425-1440`).
- `conversation.capped` counts answers cut at the sentence cap, not tool rounds (`core/conversation/worker.py:1396-1406`).
- The Reachy provider sends the HF token only to the allocator and to hosts under `.endpoints.huggingface.cloud`
  (`KEY_HOSTS`, `rm/realtime.py`; the allocator rotates connect hosts).
- Pollen's server (verified with Mac-side probes, 2026-10-29): input `rate` must be null (native 16 kHz, #136); the
  reply is 16 kHz with no rate reported (#137); it transcribes a clean Italian phrase in about 1.3 s, with or without
  a `transcription` field; it counts `audio_start_ms`/`audio_end_ms` only up to each speech end and drops a burst's
  trailing silence (#138 matches bursts by the server's own positions).
- `conversation.language` (default `it`, `core/config/schema.py` ConversationConfig) is the robot's language.

## Status (handoff, 2026-10-30)

**Epic status (2026-10-30, end of the build):** every slice is merged into `epic/cloud-voice-default`; the epic is
not on `main` yet. It waits for the owner's robot run (see "QA handoff" below).

| Slice | PR |
| --- | --- |
| CV-1 transcription language | https://github.com/sondalab-ai/roboemot/pull/139 |
| CV-0 audio hub catch-up | https://github.com/sondalab-ai/roboemot/pull/140 |
| CV-6 facing gate | https://github.com/sondalab-ai/roboemot/pull/141 |
| CV-3 skip tool | https://github.com/sondalab-ai/roboemot/pull/142 |
| CV-2 no local voice, cues, listening cue | https://github.com/sondalab-ai/roboemot/pull/143 |
| CV-4 installer `--cloud-voice` | https://github.com/sondalab-ai/roboemot/pull/144 |
| CV-5 docs | this PR |

Owner decisions taken during the build (2026-10-30): `muted` utterances pass when the attended face looks at the
robot; with no local voice and the route down, the robot plays the fallback cue and sends no chat request; the
profile writes `transcript_timeout_s` 10 s until the robot run measures the transcript p95.

Before the epic (step 0), merged on `main`:

| PR | What |
| --- | --- |
| #131 | Web UI settings replayed before the pipeline is built (restart-only edits took effect) |
| #132, #135 | `KEY_HOSTS`: Pollen's connect hosts, then the `.endpoints.huggingface.cloud` suffix (owner decision) |
| #133 | Reachy head yaw read within pi of the body (freeze at the back); face habituation (`attention.habituation_*`) |
| #134 | Server error code and the event it names in the realtime check |
| #135 | `run_on_robot.py` detects the daemon's stuck `stopping` state (only a reboot clears it) |
| #136 | Input `rate: null` at 16 kHz (Pollen refuses 16000) |
| #137 | Reply played at 16 kHz when the server reports no rate (the voice was 1.5x too fast) |
| #138 | Bursts matched by the server's own positions; a burst not fully reported resets the session |

Robot run 2026-10-30 on #138 (debug report 08:20 UTC, 8 turns, policy `unless_enrolled_in_view`):
- `utterances_sent` 8, `transcripts` 8, `transcript_timeouts` 0, `local_fallbacks` 0, `input_resets` 0,
  `tool_calls` 9, server voice (no Piper synthesis). Owner: "much better than before".
- Still slow: `latency_ms.transcript` p50 8299 ms, p95 17647 ms; `first_audio` p50 9938 ms, p95 20737 ms. The
  server is not the cause (~1.3 s by probe): `voice.latency.final_vs_last_speech` matches (8.3 s, max 17.7 s), the
  listener lags up to 16 s (`voice.realtime.lag_max_s`) and dropped 35.6 s of 275 s of audio (`dropped_s`).
- Two transcripts of noise came back in Chinese ("嗯。", "德克·米斯"): no transcription language is sent.
- Owner: no feedback while the robot listens or waits for the cloud, so the wait feels longer.

**Gate decision (owner, 2026-10-29/30):** the HIL suite was impractical (the robot did not answer without the wake
word); the owner chose to go ahead with the epic. The first-audio p95 gate (2000 ms) is not met; CV-0 targets the
main cause. The server does call tools (9 calls in 8 turns).

## Robot setup for the next session (facts, 2026-10-30)

- Robot `Neve`, `192.168.88.198`, SSH user `pollen` (password only; agents ask before any SSH write, never store the
  password; owner rules: no motor moves, no app start/restart by an agent).
- `reachy-mini.local` (mDNS) is unreliable: the robot is on 5 GHz at about -66 to -82 dBm and misses broadcasts. Use
  `--host 192.168.88.198`; if the Mac says "No route to host", run
  `sudo arp -s 192.168.88.198 88:a2:9e:8d:58:b5` (`docs/memory/robot-unreachable-from-the-mac-2026-10-29-*.md`).
- Settings on the robot: route `pollen` (platform, audio input), `privacy.realtime_audio` on,
  `voice.wake_word_policy` `unless_enrolled_in_view`, `transcript_timeout_s` 10 (to confirm), voice maybe `Serena`.
- Setup guide: `docs/realtime-voice-setup.md`. Probes used for Pollen's server live outside the repo (scratchpad);
  rewrite one when needed (allocate with `X-Reachy-Mini-Authorization: Bearer <HF token>`, connect with
  `Authorization: Bearer`, send `session_update(RoboEmotConfig(), ...)` from `core/realtime/route.py`).
- After an install, start with `run_on_robot.py --host 192.168.88.198` (not `--restart` right after an install);
  wait ~2 minutes after a start before talking (the listener lags while models load).

## Slices (integration branch `epic/cloud-voice-default`, one PR each)

Estimate (owner OK 2026-10-30): 6 slices, about 12-13 agents; with CV-6, 7 slices and about 14-15 agents. Opus authors; Sonnet reviews; Opus reviews for the
privacy slices (CV-3, CV-4) and for CV-0 if it touches the listener's audio path.

CV-0. **Listener lag on the CM4 (root cause found 2026-10-30, owner OK for the fix).** Cause: the Reachy audio hub
   drain (`rm/audio.py` `poll_once`/`_run`). After a poll gap over `max_gap_s` (150 ms) it reads up to `DRAIN_MAX`
   (1000 chunks of 16 ms = 16 s) of live audio before it pushes the batch, and `last` is taken before the poll, so
   drains chain. Hub `dropped_samples` equals the listener's `dropped_s` (35.6 s, 151.9 s); listener stages use only
   12-14 % of real time. Fix: push each chunk at once, bound the drain to the stall, time the gap after the poll.
   Original scope: The listener falls up to 16 s behind and drops audio, which is
   most of the transcript latency. Measure before changing: per-stage `ms_per_audio_s` (vad 68-92, bargein up to 49 in
   the last window), `lag_max_s`, `dropped_s`, CPU per thread, Vision at full rate while listening
   (`rate_scales.conversation` 1 while listening), the Reachy audio hub. Memory note: a similar lag was fixed in slice
   ZU (one recognizer per language). Deliverable: the cause, a fix, and the figures before/after on the robot
   (`final_vs_last_speech`, `lag_max_s`, `dropped_s`, `latency_ms.transcript`).

CV-1. **Transcription language, configurable.** New `voice.realtime_route.transcription_language` (empty: follow
   `conversation.language`), sent as `audio.input.transcription.language` in `session.update`; optional
   `transcription_model` (empty: the server's default; Pollen's app sends `gpt-4o-transcribe`). Probe Pollen first that
   the field is accepted with our session shape (it was, in the 2026-10-29 probe). Tests: the field follows the
   conversation language, an override wins, empty model leaves the key out. SPEC 8.

CV-2. **No local voice; non-verbal cues; listening feedback.** (was CV-1)
   - `voice.backend` gains `none`; `voice.fallback_backend` gains `none`. With `none`, `build_backend` builds no
     `PiperBackend` and needs no Piper virtualenv or voice files.
   - Every line only Piper speaks today becomes a non-verbal cue while the route is down: offline cue, health lines,
     the fallback line, and `fallback_mid_turn` (before first audio: no local answer, the cue; after first audio: cut
     as today).
   - **Listening / thinking cue (owner, 2026-10-30):** when an utterance is addressed and its audio goes to the
     cloud, the robot shows it heard (e.g. antennas up, small head tilt) until the first audio of the answer. A robot
     capability through a core protocol (SPEC 4A); config for its timing in the schema.
   - A transcript timeout while the route is connected does not answer through Piper (the local decode may still
     feed the chat route as text-only, or the robot shows the "did not get it" cue; decide in the slice).
   - A confirmation question that cannot be spoken **fails closed**: nothing is stored, counted as `confirm_unspoken`.
   - The cue is a robot capability (SPEC 4A hierarchy): core emits it through a protocol; the Reachy platform plays a
     gesture and a vendored short sound. Core never names the robot.
   - Tests: no Piper built with `none`; each Piper call site falls to the cue; the confirmation fails closed;
     `tests/test_robot_hierarchy.py` and `tests/test_no_magic_numbers.py` stay green.

CV-3. **Skip tool (design A).** (was CV-2)
   - The realtime session gets a tool `ignore_utterance` and an instruction: call it, and say nothing, when the
     utterance is not meant for the robot (a phone call, people talking to each other, out of context).
   - On the tool call: no audio plays, the user item and the response are deleted from the server history and the
     local history mirror, counted `realtime_route.skipped`. Audio already played (the model spoke before the call) is
     cut like a barge-in and counted `skipped_late`.
   - Config: `voice.realtime_route.skip_tool` (default false in the library), the instruction text in the persona
     prompt files, not in code.
   - Tests against the fake server: tool call before audio, tool call after audio, no tool call.

CV-4. **Installer `--cloud-voice` profile.** (was CV-3)
   - Robot-agnostic flag in `roboemot.install`; the Reachy installer plugin supplies the route (name, kind, base URL,
     key name, model) and `mode: platform`.
   - Writes to `app.yaml` (merge of its own keys only): the `pollen` route, `voice.realtime_route` (`enabled`, `route`,
     `mode`, `input: audio`, `skip_tool: true`, `transcript_timeout_s`), `privacy.realtime_audio: true`,
     `voice.backend: none`, `voice.fallback_backend: none`, `voice.wake_word_policy: never`.
   - Asks for the opt-in on a terminal, with the privacy text of decision 5. `-y` does not imply it; non-interactive
     runs need `--cloud-voice` or `--no-cloud-voice`.
   - Asks for `hf_token` (as `--secret`) when the store has none.
   - Skips the Piper step and the Piper voice models; still fetches Silero and Vosk. Removes the recorded
     `piper-venv`. Uninstall removes only the keys it wrote. The dry run never writes.
   - Tests: dry run plan, merge of own keys, uninstall, `-y` without the flag leaves the opt-in off.

CV-6. **Facing gate (owner, 2026-10-30).** An utterance addressed without the wake word (policy `never`, enrolled
   in view, follow-up window) needs the matched face (or, in degraded paths, the attended face) to look at the robot,
   from a head-yaw proxy on the 5 YuNet landmarks (no new model). CPU above a threshold for a hold time, or no facing
   data: the check is not required (counted). Config `voice.facing.*`, library default off; CV-4 turns it on.
   Opus review (privacy: it decides audio egress).

CV-5. **Docs.** SPEC 8 (backend `none`, skip tool, transcription language, listening cue), SPEC 12 (installer opt-in
   text, decision 5 egress), SPEC 13 rows from the robot runs, README status, `CLAUDE.md` installer line,
   `docs/realtime-voice-setup.md` (simplified by `--cloud-voice`), QA handoff.

**Design question for the owner, after CV-0 (not a slice yet):** (b) let the server start its answer without waiting
for the robot's transcript round-trip, as Pollen's app does (`create_response`); it affects the wake-word check and
the skip tool (CV-3), which need the text.

## QA handoff (owner, on the robot)

1. Install the epic branch: `python scripts/install_on_robot.py --dry-run --cloud-voice --host 192.168.88.198`, then
   the same without `--dry-run`. The installer asks for `hf_token` if the store has none and stops before any change
   if saved web UI settings contradict the privacy choice.
2. Start with `python scripts/run_on_robot.py --host 192.168.88.198` and wait about 2 minutes.
3. Run the SPEC 15 item "Cloud-voice install on the Wireless" and fill the SPEC 13 "after" figures.
4. Watch the facing yaw: `direction.facing.yaw_deg` in the debug report. Real landmark noise was -15 to +9 degrees on
   a frontal test portrait; if the owner is rejected while looking at the robot, raise `voice.facing.max_yaw_deg`.
5. If the figures hold, merge `epic/cloud-voice-default` into `main` (one merge, full CI).

Not verified: the SDK 1.13 read behaviour on the robot (CV-0 handles both an immediate `None` and a waiting read),
and the facing yaw on real faces at the robot's distance.

Build cost: 14 agents (Opus authors, Sonnet/Opus reviews), within the 14-15 estimate.

## Open items (not in this epic)

- Owner's new TODO notes in `TODO.md` (2026-10-30): motion-flow face prediction, emotion debug page, document
  critical configurations, simplify configuration, no mixed language on the dashboard, daytime sleep after long
  inactivity, stillness when alone, mirror-net test.
- QA on the robot of #133: no freeze with a person behind the robot; a photo stops holding the gaze after ~20 s.
- `voice.barge_in.onset_enabled` back to true on the robot (default; turned off during QA).
- False barge-in at 09:43 (`triggers` 1, cause unknown). Next step: per-path trigger diagnostics in
  `core/voice/bargein.py`.
- Barge-in onset-to-stop 313 ms against the 200 ms budget (SPEC 13); 2026-10-30 report shows 9.6-9.9 s
  `onset_to_stop` under listener lag (CV-0 first).
- Answers cut at the sentence cap (`capped` 3 of 5 turns) in local mode; recheck under the realtime route.
- Driver angle contract (follow-up PR, owner 2026-10-28): SPEC 4A and the `PoseGroupDriver` protocol state that a
  driver reports each angle on the commands' continuous scale, never wrapped; a contract test over every registered
  driver. Origin: the Reachy head yaw read at +pi against a -pi command froze the robot (external control).
- Daemon 1.13.0 stuck `stopping` state: only a reboot clears it (`docs/memory/reachy-daemon-1-13-0-*.md`).
- Wi-Fi: the robot sits at -66 to -82 dBm on a DFS 5 GHz channel (5500 MHz); move it closer or to 2.4 GHz.

## Verification

- Per slice: `ruff check . && ruff format --check .`, `pytest`, `pytest -m sim` where the slice touches runtime wiring.
- CV-0 and CV-2 need the owner on the robot for before/after figures.
- End of epic, on the robot: installer dry run, install with `--cloud-voice`, then a 10-minute conversation; debug
  report shows `realtime_route.state: connected`, `speech.backend: none`, no Piper process, `local_fallbacks` 0,
  `latency_ms.transcript` p95 well below the 2026-10-30 figure (17.6 s), CPU below the 2026-10-28 figure,
  `skipped` counted for a phone call in front of the robot.
