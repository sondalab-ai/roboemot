# Voice tools: design

Date: 2026-11-16. Status: draft, waiting for owner approval.
SPEC sections touched: 7A (circadian, day nap), 8 (LLM router, Realtime speech route, Conversation state), 10 (LLM tools), 11A (configuration), 12 (privacy, secrets, web UI), 4A (installer: new dependencies).

## Intent

The owner wants the robot to act on free speech: "go to sleep", "set a timer for ten minutes", "remind me to call Anna at six", "what's the weather tomorrow", "play some jazz". Fixed phrase lists are not acceptable: natural language understanding is a must.

Why (owner, 2026-11-16): two goals of equal weight.

1. **Useful home helper.** Timers, reminders and weather must be reliable and survive a restart.
2. **Richer companion.** The tools feed the robot's character: it naps when asked, it nods when it announces a reminder, it can dance to the music it plays.

## Goals

- Five tools, each reachable by free speech in Italian and English through the LLM tool calls that already exist (SPEC 10, LLM tools):
  - **sleep**: the robot takes a day nap on request.
  - **timer**: set, list and cancel countdown timers.
  - **reminder**: set, list and cancel reminders at a clock time, one-shot or with simple repetition (daily, weekdays, a set of weekdays).
  - **weather**: current weather and a short forecast for the configured location.
  - **radio**: play, stop and change the volume of internet radio stations from a configured list; the model picks a station by name or genre.
- A plugin point so other packages can add tools the same way.
- The robot can speak first: when a timer or reminder is due, it starts a turn of its own.
- A **Tools tab** in the web UI dashboard (owner request 2026-11-16).

## Non-goals

- Offline understanding. Tools work only through an LLM route (chat or realtime). With no route reachable, the robot says it cannot do that now (owner decision 2026-11-16: "LLM only").
- YouTube and any streaming service with an account. YouTube needs `yt-dlp`, which breaks YouTube's terms of service and breaks often. It is a separate opt-in slice, decided after radio works on the robot (owner decision 2026-11-16).
- Full calendar recurrence rules (every second Tuesday, monthly).
- Person-bound reminders ("remind Anna when you see her"). A reminder is announced at its due time to whoever is there (owner decision 2026-11-16).
- MCP or any remote tool protocol.

## Anti-goals (failure even if it works)

- Hard-coded utterances or keyword matching for tool intent.
- A tool that leaves the robot with no voice way back: "go to sleep" must not turn off the listener.
- The robot dancing to, transcribing or answering the lyrics of its own radio.
- A due reminder that is lost silently (restart, route down, robot napping).
- Any new outbound network call that is on before an admin turns it on (SPEC 12).

## Constraints

- SPEC 11A: every behavioural constant is a config field with default, unit and range. `tests/test_no_magic_numbers.py` covers the new modules.
- SPEC 12: no frames, audio or embeddings in logs; keys only from the secret store; outbound calls are admin opt-ins.
- SPEC 4A / robot hierarchy: the tools are robot-agnostic and live in core. Core never names a robot. `tests/test_robot_hierarchy.py` baseline may not grow.
- SPEC 8: chat routes use only the common OpenAI subset (`tools`, `tool_choice="auto"`).
- SPEC 10: tool arguments use the existing strict validation (`str`, `int`, `float`, `bool`, `Literal`, `Annotated` with `Range`/`MaxLen`/`Desc`).
- One process owns the SDK media (SPEC 4A). The radio player plays through the same speaker path as TTS and cues.
- Raspberry Pi CM4 CPU budget: stream decoding must fit next to the listener thread (SPEC 8, Listener real-time budget).
- Replies in Italian and English. Tool descriptions are in English (the model reads them); fallback sentences exist per language.

## Approaches considered

1. **Tool pack in core (chosen).** Core adds the three missing parts (a plugin point for tools, a persistent scheduler that can make the robot speak first, a media player) and the Tools tab. The five tools ship in core as plugins of the new entry-point group, the same way a third-party package would add one. One install, one CI.
2. **Separate `roboemot-tools` package.** Same core parts, tools in their own package. Cleaner opt-out, but more packaging and installer work and a second version to pin. Not chosen.
3. **MCP client.** Dropped: timer, sleep and music still need all three core parts, and MCP adds latency and an attack surface.

Owner picked approach 1 on 2026-11-16.

## What exists today (facts the design builds on)

- `@skill` (`skills/registry.py`) marks a function as a tool; `SkillRegistry.tools()` builds the OpenAI-format schemas; `execute()` validates and runs a call with `conversation.tool_timeout_s`. `conversation.llm_skills` lists the skills offered to the model. Five default skills live in `skills/defaults.py`.
- Chat routes, `openai_realtime` and `gemini_live` all send `tools` and run tool calls. **Unverified:** whether Pollen's HF speech-to-speech server (the cloud-voice profile) honours function tools. If it ignores them, the tools are silent on that profile.
- No skill entry-point group exists in `plugins.GROUPS`.
- No path exists for the robot to speak without a user turn.
- The circadian day nap (SPEC 7A) is woken by the robot's name and by motion in front of it. Platform sleep (`robot.asleep`) pauses the listener.
- `voice/music.py` and `voice/music_classifier.py` detect music from the microphone and drive the dance (SPEC music-dance design, 2026-11-04). `voice/duck.py` ducks the robot's own voice on barge-in. `SpeechOutput.play_aside` plays non-speech samples.

## Design

### 1. Tool plugin point

- New entry-point group `roboemot.skills`, registered in `plugins.GROUPS` with a contract: a factory `(ToolHost) -> list[skill functions]`. `ToolHost` is a narrow core protocol that gives a tool what it may use: the scheduler, the media player, the nap request, the config reader, the clock, the secret store reader.
- Core's own tools are registered through the same group from `roboemot-core`'s `pyproject.toml`.
- A tool from a plugin is offered to the model only when its name is in `conversation.llm_skills`. The five new tools are added to the default list. A tool whose own `tools.<name>.enabled` is false is not offered.
- Each tool group may have several skills (for example `timer_set`, `timer_list`, `timer_cancel`). Builder's call on the exact split; fewer skills with an `action` `Literal` is acceptable if the schemas stay clear.

### 2. Sleep tool

- `go_to_sleep()` starts a **requested nap** now, with the same posture, affect, voice and vision rates as a nap that came from idle time (SPEC 7A).
- Wake rules (owner decision 2026-11-16): for `circadian.requested_nap_min_s` after it starts, only the robot's name or a due timer/reminder wakes it; a visible face, motion, light, loud sound and unaddressed speech do not. After that time the usual day-nap wake rules apply. This is a small new state in the circadian nap logic, not a thin call. "Wake up" needs no tool: the name already wakes the robot.
- During quiet hours the tool still works (the robot is already resting; the result says so).
- Fire-and-forget after the reply: the model says a short good-night, then the nap starts. The nap must start after the reply has finished playing, not during it.

### 3. Scheduler and robot-initiated turns

- **Store.** Timers and reminders are kept in one file in the app data folder (`~/.local/share/roboemot/`), written atomically on every change, read on start. A timer stores its due time as wall-clock time so that a restart keeps it.
- **Items.** Timer: label (optional), due time. Reminder: text, due time, repetition (`once`, `daily`, `weekdays`, or a set of weekdays), time zone `circadian.timezone`.
- **Limits.** Maximum active items, maximum text length, maximum timer duration: config fields.
- **Missed items.** An item that became due while RoboEmot was not running is announced at start if it is late by less than `tools.scheduler.missed_grace_s`; older ones are dropped and listed as missed in the Tools tab. The grace applies only across a process restart: an item that waited during platform sleep always fires when the robot is awake again, however late (builder check, my call).
- **Busy robot.** A due item waits until the conversation is idle: nobody mid-utterance, the robot not speaking, no tool round open. After `tools.scheduler.announce_max_wait_s` it starts its turn anyway, at the next point between two robot sentences. The radio ducks during the announcement (my call).
- **Firing.** A due item:
  1. wakes the robot from a nap (not from platform sleep: with the motors off, the item waits and fires when the robot is awake again);
  2. ignores quiet hours, because the owner set it;
  3. plays the `nod` gesture ("yes" gesture) when the announcement starts (owner decision 2026-11-16);
  4. starts a **robot-initiated turn**.
- **Robot-initiated turn.** A new conversation input kind, "event", next to user utterances. Core sends the model a short system-authored message ("A timer the person set is due: <label>" or "A reminder is due: <text>") and the model phrases it in the persona's words and language. The person can reply, so the turn opens a normal conversation window. It works on chat routes and on realtime routes (`conversation.item.create` with text, then `response.create`; Gemini Live: a client text turn).
- **Fallback.** If no route answers within `tools.scheduler.announce_timeout_s`, the robot says a fixed sentence per language through the local voice when one is available, plus the cue sound; otherwise the cue and the gesture only. An item is marked done only after an announcement was played.
- **Repeat.** If nobody reacts (no addressed speech before `tools.scheduler.repeat_after_s`), a timer announcement repeats up to `tools.scheduler.repeat_max` times. Reminders do not repeat (builder's call to revisit after robot tests).
- Event text sent to the model is the person's own reminder text: it counts as conversation text for SPEC 12 (it may reach an off-board route, like any transcript).

### 4. Weather tool

- Provider: Open-Meteo (no API key). Only the configured location (latitude, longitude) and the requested days leave the robot.
- Location: the existing private `circadian.location` (lat, lon) plus a display name `tools.weather.place_name`, set in the Tools tab (with a place search through Open-Meteo's geocoding API, called only from the web UI by the admin). No second coordinate pair (my call, 2026-11-16).
- Result to the model: current conditions and a daily forecast for up to `tools.weather.forecast_days`, in metric units, as structured data; the model phrases it.
- Short cache (`tools.weather.cache_s`) so a follow-up question costs no extra call.
- Off by default; the admin turns it on (`tools.weather.enabled`).

### 5. Media player and radio tool

- **Player.** A core media channel that decodes a network stream to PCM and plays it through the robot's speaker path at the hub's rate. One stream at a time. Volume in dB, separate from the voice volume.
- **Sharing the speaker (owner decision 2026-11-16: pause now, mixer later).** Today one speaker queue carries all robot audio, so true ducking needs a new real-time mixer. In this epic the radio **pauses**: it stops pushing audio while the robot speaks (TTS, cue, announcement) and while a person addresses the robot (from the wake word to the end of the reply), then rejoins the live stream after `tools.media.resume_after_s` of silence (the buffered backlog is dropped). A mixer with ducking is a later opt-in slice after robot tests.
- **Self-hearing.** The player publishes "the robot is playing music" on the bus.
  - Dance (owner decision 2026-11-16): the existing microphone detector keeps running on the robot's own music and may start the dance, with its tempo and beat from the microphone as today. Behaviour flag `tools.radio.dance` (default on) lets the admin turn this off; off, the detector ignores music while the player plays. Beat drift from echo and playback latency is a known risk for the robot test.
  - Lyrics: during playback the listener keeps the wake word and addressed speech working; speech that is not addressed is never answered, whatever the conversation policy, while the flag is set.
- **Stations.** A list in config (`tools.radio.stations`: name, URL, genres, language), edited in the Tools tab. The model gets the station names and genres as `choices` (snapshotted per request, SPEC 10). Only listed URLs are ever played: the model cannot pass a URL.
- **Tools.** `radio_play(station)`, `radio_stop()`, `radio_volume(change)`. A stream that fails to start or drops is reported to the model as an error result; a drop during play stops the player and logs the station name only.
- **Limits.** `tools.radio.max_play_s` (auto stop), stop on nap and on platform sleep.
- Off by default; the admin turns it on (`tools.radio.enabled`).
- **Dependency.** A decoder for MP3/AAC/Ogg streams on the CM4 (for example PyAV or an `ffmpeg` subprocess). Builder's call after measuring CPU on the robot; the installer pins it and records it (SPEC 4A, Installer).

### 6. Tools tab (web UI)

A new dashboard tab, `Tools`, PIN session like every page API:

- One card per tool: enabled switch, short status, its settings (weather location with place search; radio station list with add, edit, remove and a test play; scheduler limits under the Advanced fold).
- Active timers and reminders: list with due time, repetition and a cancel button; missed items.
- Now playing: station, volume, stop.
- Strings in Italian and English like the rest of the UI; `scripts/audit_webui.sh` passes.
- Writes go through the existing settings save path (batch save, admin confirmation where the settings editor needs it).

### 7. Configuration (new `tools.*` section, SPEC 11A)

Every number above is a field with default, unit and range. Builder proposes defaults; the doc's names are indicative. Changes that need a restart say so in their description.

## Delivery

One integration branch, `epic/voice-tools`, with short slice PRs; full CI on each; one merge to `main`.

| Slice | Content | Depends on |
| --- | --- | --- |
| T0 | SPEC text (sections 7A, 8, 10, 11A, 12) for this design; Pollen tool-call probe on the robot (manual, owner runs it; does not block T1) | none |
| T1 | `roboemot.skills` group, `ToolHost` protocol, plugin loading, `tools.*` config skeleton, tests | T0 SPEC text |
| T2 | Sleep tool and requested-nap wake rules | T1 |
| T3 | Scheduler store and timer/reminder tools (no announcement yet: items fire to the log and bus) | T1 |
| T4 | Robot-initiated turn on chat and realtime routes; announcement, nod, fallback, repeat | T3 |
| T5 | Weather tool | T1 |
| T6 | Media player, ducking, self-hearing signal, radio tool | T1 |
| T7 | Tools tab | T2–T6 |

Estimate: 8 slices, about 18 agents (one Opus author and one Sonnet reviewer per slice; Opus reviewer for T5 and T6, which add outbound network calls). T4 and T6 carry the most risk.

## Left to the builder

- Exact skill split per tool and the tool descriptions.
- Store file format and atomic write method.
- Decoder choice (after a CPU measurement on the robot).
- Config field names and defaults within the rules above.
- How the robot-initiated turn enters each realtime wire, within SPEC 8's protocol subset.

## Open risks

- Pollen's server may ignore function tools: then no tool works on the cloud-voice profile. The epic proceeds anyway; tools then work on chat, OpenAI-realtime and Gemini Live routes only, and the owner is told. No fallback route is added (my call).
- CPU on the CM4 while decoding a stream and running the listener.
- Echo: the microphone hears the radio. The wake word must still work at the configured music volume; this needs a robot test.
