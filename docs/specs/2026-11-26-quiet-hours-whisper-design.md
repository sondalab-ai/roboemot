# Quiet-hours whisper (design, 2026-11-26)

Status: proposed. SPEC sections touched: 7A (circadian, quiet hours), 8 (voice, speech output, realtime route).

## Intent

Inside the quiet hours (`circadian.quiet_hours`, default 22:30 to 07:00) every sound the robot makes is very soft, close to a whisper, so nobody in the house is disturbed or woken (owner, 2026-11-26).

## Today

- The night volume follows the circadian **state**, not the clock: `circadian.states.resting.volume_db` is -6 dB, `drowsy` -3 dB, `awake` 0 dB (`config/schema.py`). Speech (prosody), cues, clips and self-soothing sounds read it through `effects_now`.
- The robot speaks at night only after a wake (a person, the wake word, an announcement). A wake makes the state `awake`, so the answer plays at full volume.
- -6 dB is about half loudness, not a whisper. `voice.limits.volume_db` also floors speech at -12 dB.
- The radio plays at `tools.media.volume_db` plus its steps and ignores the circadian state. It stops on `resting`, but a woken robot can start it again.
- Every sound goes through one queue, `SpeechOutput` (`voice/speech.py`): `say`, `stream`, `play`, `play_aside`, `play_media`.

## Goals

- Inside the quiet hours, every sound through `SpeechOutput` plays at the night level: speech (Piper and realtime), cues, clips, raw sounds, the radio.
- The clock decides, not the state: a wake inside the quiet hours keeps the night level.
- Announcements a person set (timers, reminders, scheduled items, `tools/announce.py`) keep the day volume (owner decision 2026-11-26).
- The realtime voice is asked to whisper.

## Non-goals

- No change to the robot's hardware speaker volume.
- No whisper timbre for Piper: Piper cannot whisper, it only plays softer.
- No change to what the robot does at night (it still answers every interaction, SPEC 7A).

## Anti-goals

- A sound added later that plays loud at night because it skipped the night stage.
- A night level that stacks with the resting offset and becomes inaudible to the person who spoke to the robot.

## Approaches

1. **One night stage in `SpeechOutput` (chosen by the owner).** Inside the quiet hours the queue applies one night gain to every job; a job marked as an announcement skips it. One place covers every present and future sound. Cost: the queue needs the quiet-hours window and a clock.
2. Night offset at each source (prosody, cues, clips, soothe, radio). Follows today's pattern, but five call sites now and a future source can forget it with no error.
3. Lower the robot's hardware speaker volume. Rejected: robot-specific (SPEC 4A robot hierarchy), and announcements could not pass at day volume.

## Design

- New config fields (SPEC 11A, no magic numbers), in `circadian`:
  - `quiet_hours_volume_db`: the night level, default -18 dB (range -40 to 0; null: no night stage). Placeholder until measured on the robot.
  - `quiet_hours_instructions`: delivery text added to realtime instructions inside the quiet hours, default "whisper, very softly".
- Inside the quiet hours the total night attenuation is the night level, whatever the state: the sources keep applying the state's `volume_db`, and the night stage adds only the difference (`quiet_hours_volume_db` minus the state offset captured when the sound was queued, never above 0). The radio never carries the state offset, so it gets the full night level. Mood offsets still apply, within `voice.limits`.
- The night stage sits after `voice.limits`, so it is not floored at -12 dB.
- Outside the quiet hours, or with `circadian.enabled` false, nothing changes.
- An announcement job carries a flag (for example `day_volume=True`) set by `tools/announce.py` for its speech and its cue; the night stage skips it. The person's reply after the announcement whispers again.
- The night level is reported in the debug report and the voice snapshot (`night_db`, whether the window is on).

## Left to the builder

- Where the clock and window reach `SpeechOutput` (a callable `night_db()` built in `runtime/startup.py` is one option).
- How a gain change mid-stream is handled at the window edge (a step at the next chunk is acceptable).
- Tests: a sound at 23:00 plays at the night level in each state; at 12:00 nothing changes; an announcement at 23:00 plays at day volume; a realtime turn at 23:00 carries the whisper instruction.

## To verify on the robot

- Whether -18 dB is audible from 1 m and inaudible from the next room. Tune the default from that.
