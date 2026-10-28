# Cloud voice default (Reachy install): epic plan

**Goal:** On the robot, all voice interaction goes through the Pollen realtime route by default, with no local speech
synthesis, so the conversation feels organic and the CM4 is not overloaded.

**Architecture:** The realtime route (SR-1..SR-5, `docs/plans/2026-10-26-realtime-speech-route.md`) already does
transcription, answer and voice on the server with `input: audio`. This epic adds a speech backend `none` (non-verbal
cues replace every line only Piper speaks), a skip tool the server LLM calls for an utterance not meant for the robot,
and an installer profile `--cloud-voice` that writes these settings into the robot's `app.yaml`. Library defaults stay
off (SPEC 12).

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
   (design A below: a skip tool).
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
- The Reachy provider sends the HF token only to `KEY_HOSTS`, which is empty (`rm/realtime.py`), so a keyed `pollen`
  route fails with `url_key_host` today.

## Step 0: the route works on the robot (gate for every slice)

Owner on the robot, no SSH needed:
1. Routes tab: API key `hf_token` (fine-grained, read-only), route `pollen` (kind `realtime`, base URL
   `https://pollen-robotics-reachy-mini-realtime-url.hf.space/session`, key `hf_token`, model `s2s`).
2. Settings: `voice.realtime_route.enabled` true, `.route` `pollen`, `.mode` `platform`, `.input` `audio`;
   `privacy.realtime_audio` on (Show with PIN). Restart the app.
3. Routes tab, `pollen`, **Check**. Expected `failure: url_key_host`; record `connect_host`.

Agent:
4. PR `KEY_HOSTS = ("<connect_host>",)` in `rm/realtime.py`, with its test. Owner redeploys and checks again:
   `header_to_connect_host: sent`, `update_seen: true`, `item_delete: accepted`.
5. Owner runs the latency, underrun and item-id HIL tests (`packages/roboemot-reachymini/tests/test_realtime_hil.py`).
   Record whether the server calls tools (`item_ids.tool_calls`).

**Gate:** `first_audio_ms` p95 at most 2000 ms and the server calls tools. If either fails, stop and decide again
with the owner before slice 1.

## Slices (integration branch `epic/cloud-voice-default`, one PR each)

CV-1. **Speech backend `none`, non-verbal cues.**
   - `voice.backend` gains `none`; `voice.fallback_backend` gains `none`. With `none`, `build_backend` builds no
     `PiperBackend` and needs no Piper virtualenv or voice files.
   - Every line only Piper speaks today becomes a non-verbal cue while the route is down: offline cue, health lines,
     the fallback line, and `fallback_mid_turn` (before first audio: no local answer, the cue; after first audio: cut
     as today).
   - A confirmation question that cannot be spoken **fails closed**: nothing is stored, counted as `confirm_unspoken`.
   - The cue is a robot capability (SPEC 4A hierarchy): core emits it through a protocol; the Reachy platform plays a
     gesture and a vendored short sound. Core never names the robot.
   - Tests: no Piper built with `none`; each Piper call site falls to the cue; the confirmation fails closed;
     `tests/test_robot_hierarchy.py` and `tests/test_no_magic_numbers.py` stay green.

CV-2. **Skip tool (design A).**
   - The realtime session gets a tool `ignore_utterance` and an instruction: call it, and say nothing, when the
     utterance is not meant for the robot (a phone call, people talking to each other, out of context).
   - On the tool call: no audio plays, the user item and the response are deleted from the server history and the
     local history mirror, counted `realtime_route.skipped`. Audio already played (the model spoke before the call) is
     cut like a barge-in and counted `skipped_late`.
   - Config: `voice.realtime_route.skip_tool` (default false in the library), the instruction text in the persona
     prompt files, not in code.
   - Tests against the fake server: tool call before audio, tool call after audio, no tool call.

CV-3. **Installer `--cloud-voice` profile.**
   - Robot-agnostic flag in `roboemot.install`; the Reachy installer plugin supplies the route (name, kind, base URL,
     key name, model) and `mode: platform`.
   - Writes to `app.yaml` (merge of its own keys only): the `pollen` route, `voice.realtime_route` (`enabled`, `route`,
     `mode`, `input: audio`, `skip_tool: true`), `privacy.realtime_audio: true`, `voice.backend: none`,
     `voice.fallback_backend: none`, `voice.wake_word_policy: never`.
   - Asks for the opt-in on a terminal, with the privacy text of decision 5. `-y` does not imply it; non-interactive
     runs need `--cloud-voice` or `--no-cloud-voice`.
   - Asks for `hf_token` (as `--secret`) when the store has none.
   - Skips the Piper step and the Piper voice models; still fetches Silero and Vosk. Removes the recorded
     `piper-venv`. Uninstall removes only the keys it wrote. The dry run never writes.
   - Tests: dry run plan, merge of own keys, uninstall, `-y` without the flag leaves the opt-in off.

CV-4. **Docs.** SPEC 8 (backend `none`, skip tool), SPEC 12 (installer opt-in text, decision 5 egress), SPEC 13 rows
   filled from step 0, README status, `CLAUDE.md` installer line, QA handoff.

## Estimate

1 tiny PR (step 0) plus 4 slices. About 8 agents: Opus authors for CV-1..CV-3, Sonnet for CV-4 and for the reviews.
CV-2 and CV-3 touch privacy: their reviews run on Opus.

## Open items (not in this epic)

- False barge-in at 09:43 (`triggers` 1, cause unknown). Next step: per-path trigger diagnostics in
  `core/voice/bargein.py`.
- Barge-in onset-to-stop 313 ms against the 200 ms budget (SPEC 13).
- Answers cut at the sentence cap (`capped` 3 of 5 turns) in local mode; recheck under the realtime route.
- Robot `wake_word_policy` was set to `never`, then `always`, by hand during QA; CV-3 sets it.
- Driver angle contract (follow-up PR, owner 2026-10-28): SPEC 4A and the `PoseGroupDriver` protocol state that a
  driver reports each angle on the commands' continuous scale, never wrapped; a contract test over every registered
  driver. Origin: the Reachy head yaw read at +pi against a -pi command froze the robot (external control).

## Verification

- Step 0 figures recorded in SPEC 13.
- Per slice: `ruff check . && ruff format --check .`, `pytest`, `pytest -m sim` where the slice touches runtime wiring.
- End of epic, on the robot: installer dry run, install with `--cloud-voice`, then a 10-minute conversation; debug
  report shows `realtime_route.state: connected`, `speech.backend: none`, no Piper process, CPU below the 09:59
  figure, `skipped` counted for a phone call in front of the robot.
