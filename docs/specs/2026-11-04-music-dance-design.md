# Music awareness and dance: design

What this file is: the design for the music detector, the music gate on the wake decision and the
dance motion layer. It feeds new and changed sections of `docs/specs/kickoff.md` (the SPEC), which
stays the source of truth. Status: proposed, not implemented.

## Intent

A song played in the room. The robot heard the lyrics as speech and answered them. The owner wants
two things, in this order (owner, 2026-11-04):

1. **The robot never answers music.** While music plays, every turn needs the wake word.
2. **The robot moves to the rhythm.** It moves to music in the room, and to its own sounds
   (self-soothing hums and blirps).

## Goals

- G1. Detect music in the room from the audio the listener already receives, with a tempo and a
  beat phase.
- G2. While music plays, require the wake word for every utterance, on every path, including the
  realtime route's audio gate.
- G3. Dance to the beat: small, phase-locked moves of abstract roles, scaled by mood, circadian
  state and persona.
- G4. Dance to the robot's own sounds from the beat times the synth planned. No detection is
  needed for them.
- G5. Keep moving lightly while the robot answers (owner, 2026-11-04).
- G6. Keep the CPU cost very low. The robot is at the edge of its CPU budget (owner, 2026-11-04).

## Non-goals

- Genre, mood or song recognition. Lyrics transcription.
- A new audio capture path, a new model or a new dependency.
- Choreography: named dance moves, recorded clips set to music.
- Dancing to room music while voice is off (`voice.enabled: false`). The detector runs on the
  listener's thread (my call, see "Placement").
- Beat tracking good enough for music production. "Close enough to look rhythmic" is the bar.

## Anti-goals (a failure even if it works)

- The robot answers lyrics without the wake word while the detector reports music.
- The gate ever blocks an utterance that holds the wake word.
- The detector raises the listener's realtime load (`listener.load`) enough to cause audio gaps or
  late decodes on the CM4.
- Core names a robot, a joint or an SDK (`tests/test_robot_hierarchy.py`), or a behavioural
  constant lives outside the schema (`tests/test_no_magic_numbers.py`).
- Any audio, spectrum or onset envelope reaches a log or the debug report (SPEC 12).
- The robot's own voice or sounds count as room music.

## Constraints from the code today

- The listener (`roboemot.voice.listener.VoiceListener`) receives mono float32 chunks at
  `voice.sample_rate_hz` (16 kHz) on its own thread. Chunks heard while the robot speaks are dropped
  before ingestion (`_process`, `dropped_speaking`). Self-soothing sounds play through
  `SpeechOutput.play`, so they open the same speaking window and are dropped the same way.
- The listener pauses while the robot sleeps (`runtime.pause_when_asleep`).
- The wake decision lives in one place, `roboemot.voice.wake.WakeGate`: `outcome()` decides,
  `addressable()` predicts at speech start for the realtime audio gate (SPEC 8, Realtime speech
  route). Both share the "required" predicate and `_follow()`.
- The speaking motion (`roboemot.affect.speaking.SpeakingRenderer`) is the pattern to copy. It is an
  additive offset at EXPRESSION priority, fed from the app tick (`motion.app_tick_hz`, 20 Hz). Its
  roles are bound to channels by the manifest's `behaviour.speech_motion_map`. It scales by register,
  circadian state and arousal, and it ramps in and out over `fade_ms`.
- The soothe synth (`roboemot.soothe.synth`) plans every sound before it renders it. A hum plan has
  its note lengths and `beat_s`. A blirp plan has its events and a tempo.
- SPEC 4A (speaking motion on Reachy Mini): min-jerk halves of 0.5 s or more, chosen against the
  fallback caps.

## Design

### 1. Music detector (core, `roboemot.voice.music`)

**Placement.** The listener calls the detector on each chunk it ingests, after the speaking check
in `_process` (my call). So the robot's own voice and sounds never reach it. The detector adds no
thread and no queue. The listener pauses while the robot sleeps, and the detector pauses with it.

**Signal (cheap by construction).**

1. Split each chunk into hops of `music.hop_ms` (for example 32 ms, 512 samples at 16 kHz).
2. Per hop, compute the energy in `music.bands_hz` (eight bands since QA 2026-11-09: SPEC 8, Music
   detector, also for the overlapping windows, the drift removal and the smoothing).
   Use one real FFT per hop, or IIR band filters, whichever measures cheaper on the CM4. The
   builder measures both.
3. The onset strength is the half-wave-rectified log-energy rise per band, summed. This is a
   ~31 Hz envelope, kept in a ring buffer of `music.window_s` (for example 6 s, ~190 values).
4. Every `music.estimate_interval_s` (for example 1 s), and only when the recent level is above
   `music.min_level_db`, autocorrelate the envelope over lags for `music.tempo_bpm` [60, 180].
   Pick the peak with a mild prior toward 100 to 130 BPM to avoid octave errors.
5. Beat phase: the offset that maximises the envelope summed at that period's grid (comb).

**Decision.** "Music" means all of these hold for `music.confirm_s`:

- the tempo peak's strength (normalised autocorrelation) is at least `music.min_periodicity`;
- the tempo is stable within `music.tempo_tolerance` across consecutive estimates;
- the level stays above `music.min_level_db` for at least `music.min_active_share` of the window.

Speech has no stable periodic onset pattern over several seconds, so it fails the first two
conditions. "Music" ends after `music.release_s` without the conditions holding. The release is
long (for example 8 s), so a quiet bar or a pause between songs does not reopen the follow-up path.

**Back-dating (my call).** Music is confirmed only after `music.confirm_s`, but the evidence starts
earlier. On confirmation the detector back-dates the start of the music interval to the start of the
evidence. It keeps a short history of music intervals (`music.history_s`). The gate asks
"was music playing at any time in [utterance start, utterance end]", at the decision. An utterance
that ended before the music was confirmed is not covered. That residual falls back to the
voice-at-face and facing checks.

**Gaps (my call).** A gap in the audio (the robot speaking, a chunk `gap`) longer than
`music.max_gap_s` clears the envelope ring, and the tempo estimate restarts from fresh audio. The
state and the phase keep running on the flywheel (below). Audio replayed by the barge-in `resume()`
path goes to `_ingest` directly and does not reach the detector. That is intended: it is the user's
voice after a cut.

**Blind time.** While the robot speaks, the detector receives nothing. It keeps its last state and
tempo, and its phase runs forward on the clock (a flywheel). When `music.release_s` passes with the
robot still speaking, the state stays "music" until fresh audio arrives, up to
`music.blind_hold_s` (default 30 s, about one long answer). So an answer the robot gives over a song keeps the gate closed, and the dance
keeps the beat.

**Output.**

- `MusicState(playing: bool, tempo_bpm: float | None, beat_phase_s: float | None,
  confidence: float, stamp: float)`, readable by any thread (one immutable object swapped atomically).
- A bus event `music.started` / `music.stopped` (tempo only, no audio).
- A debug section with counts: estimates, starts, stops, current state, tempo, confidence. It also
  holds the cost: the detector's CPU ms per second of audio (min, median, p90, last), in the same
  form as the listener's stage timing. The cost also goes into `listener.load` as its own stage,
  `music`.

**Budget.** Target ≤ 1 % of one CM4 core on average, measured on the robot. The build reports the
number. If the measured cost is above 2 %, slice 1 stops and asks.

### 2. Music gate (core, `WakeGate`)

The gate receives `music: Callable[[float], bool]`, which says whether music plays at a stamp in the
audio clock.

- When music plays at the utterance's `start`, the wake word is **required**, whatever the policy.
- When music plays, the follow-up window does **not** count: `_follow` returns
  `rejected_music` (owner decision 2026-11-04: every turn needs the wake word while music plays).
- `addressable()` uses the same predicates. So the realtime audio gate never sends audio without
  the wake word while music plays. That utterance takes the existing "decode locally and look for
  the wake word" path.
- An utterance with the wake word is always addressed. The music predicate never blocks it.
- The "required" predicate is written twice today (`outcome` and `addressable`). Extract it into one
  function first, then add the music rule there.
- New outcome in `DECISIONS`: `rejected_music`. It shows on the Live tab like the other rejections.

Known limit: over loud music, Vosk may miss the wake word too. Then the robot does not answer at all
until the music is quieter or stops. The owner accepted this (2026-11-04). A miss rate target is
measured in the slice 1 robot check, not fixed in advance.

Failure cost: a false "music" result costs the owner the wake word for one turn. A missed "music"
result falls back to today's voice-at-face and facing checks. So the detector errs toward
"music" when unsure (`music.min_periodicity` is tuned on the robot with that bias).

Side effect: an utterance rejected for music still runs Vosk (it must, to find the wake word). The
wake decode is already capped at `voice.realtime_route.wake_decode_max_s`. With Vosk, the builder
caps the local decode the same way while music plays, if the measured cost shows it is needed.

### 3. Beat source (core, `roboemot.motion.beat`)

One read-only interface gives the dance layer the beat:
`beat_at(now) -> Beat(period_s, phase, strength, source) | None`.

Two providers, picked in this order:

1. **Own sound.** While a self-soothing hum plays, `SootheRunner` publishes the planned beat grid:
   the start stamp of the job and the hum's `beat_s`. A hum is periodic, so it gives a period and a
   phase. Blirps are irregular. They give no beat, and each blirp event plays only the `accent` role
   (my call). The start stamp is the time the speech output really starts the job, not the
   queue time. The builder checks that `SpeechOutput` can report it and adds that if needed.
   `strength` is 1.
2. **Room music.** The detector's `MusicState`, with `strength` = confidence. The dance layer reads
   at least `dance.min_strength` for this source (SPEC 7, Dance motion).

None: no dance.

### 4. Dance layer (core, `roboemot.affect.dance.DanceRenderer`)

An additive offset at EXPRESSION priority, fed from the app tick right after the speaking motion.
It copies the speaking renderer's structure (immutable settings bundle, fade in and out, freeze on
disable, last zero command).

**Roles** (abstract, bound by the manifest's new `behaviour.dance_motion_map`):

- `bob`: one dip per beat (the downbeat of each period), min-jerk.
- `sway`: one side per bar (`music.beats_per_bar`, default 4), slow.
- `accent`: a small flick on strong onsets, at most one per `dance.accent_min_interval_s`.

**Tempo folding.** A move half must last at least `dance.min_half_s` (0.5 s on Reachy, from
SPEC 4A). When the beat period is too short for one full `bob` (2 × `min_half_s`), the layer moves
on every second beat, then every fourth beat (half time, quarter time).

**Latency.** The layer leads the beat by `dance.lead_ms`, which covers capture delay plus command
to motion delay. It is tuned on the robot.

**Amplitude.**
`preset degrees × register.motion_amplitude × circadian factor × (1 + arousal_gain × arousal) ×
beat strength`, capped by `expression.max_amplitude`, clamped to the manifest range about home. No
dance while circadian state is `resting`. The persona traits scale it at the same stage as the other
layers.

**While the robot speaks.** The dance continues at `dance.speaking_gain` (for example 0.3) of its
amplitude (owner, 2026-11-04: keep moving lightly while it answers). On a channel that the speaking
motion also drives (`beat` is on `head.pitch`), the speaking offset wins: the dance fades out of
that channel while the speaking renderer emits on it, and keeps the other channels. My call; it
avoids two rhythms on one joint.

**Gaze and tracking.** The dance offset stacks on gaze and face tracking as an additive offset, as
the speaking motion does. While the robot speaks, `bob` shares `head.pitch` with the speaking `beat`
and fades out, so the dance keeps `sway` and `accent` only (my call).

**Yields.** The layer stops (ramps to zero) for an expression clip on the same channels, external
control (`motion.external.*`), sleep, a disabled layer and a circadian factor of 0. The builder
checks on the robot that the dance offsets do not trip the external-control detector. The pose
must keep following the commands.

**Start and stop.** The dance fades in over `dance.fade_in_s` (for example 2 s) once the beat is
present, and fades out over `dance.fade_out_s` when the beat goes. It never starts with a jump.

### 5. Robot specialisation (Reachy Mini)

Only the manifest changes: a `dance_motion_map` next to `speech_motion_map`. My first guess for
Reachy Mini, to tune on the robot:

- `bob` → `head.pitch` dip.
- `sway` → `head.roll` with a little `head.yaw`.
- `accent` → antennas flick in opposite directions.

No body yaw and no translation, refused at load as for `speech_motion_map`. No new entry point.

### 6. Configuration

New schema sections `music.*` and `dance.*`. Every threshold, time and gain above is a field with a
default, a unit and a range (SPEC 11A). The example values in this file are starting points, not
decisions. `music.enabled` and `dance.enabled` each default to on, so the gate protects the robot
without any setup. The web UI shows them through the existing settings mirror.

## SPEC changes

- Section 7, Speaking motion: a new subsection "Dance motion" (roles, folding, mixing with speaking
  motion, yields).
- Section 7B, Self-soothing: the runner publishes the beat grid of the sound it plays.
- Section 8, stage 2/3: the music detector and the music rule of the wake decision, including the
  realtime audio gate.
- Section 4A, Reachy Mini manifest: `dance_motion_map`.
- Section 11A: the `music` and `dance` config sections.
- Section 12A: the debug sections (counts and cost only).
- QA checklist: the on-robot checks of slice 1 and slice 4.

## Delivery slices

1. **Music detector and gate.** Detector, `WakeGate` rule, `rejected_music`, debug section with
   cost. Ship it, then measure the CPU on the robot with a song playing. This fixes the worse problem
   first and checks the budget before any motion work. The CPU measurement on the robot is an
   owner-gated task. Slice 1 lands on the epic branch only. The epic reaches `main` only after
   the measurement. If the cost is above 2 %, the plan tries a lower hop rate and fewer bands
   before anything else.
2. **Beat source and dance to own sounds.** Beat interface, soothe beat grid, `DanceRenderer`, a
   generic `dance_motion_map` test manifest. The beat is known exactly, so this tests the motion
   side alone.
3. **Dance to room music.** Detector as a beat provider, tempo folding, speaking gain and channel
   mixing.
   Robot check: with the dance on and the song stopped, `music.stopped` comes within
   `music.release_s`. This shows that the robot's own servo noise does not keep the detector
   confirming music. If it does, the detector ignores onsets locked to the dance's own phase.
4. **Reachy Mini map and tuning.** Manifest map, `dance.lead_ms`, amplitudes, detector thresholds,
   external-control check, all on the robot.

## Left to the builder

- FFT versus IIR bands, hop size, band edges: the cheaper one on the CM4, within the budget.
- The exact tempo prior and the octave-error rule.
- Pass criteria without the robot: on synthetic audio, a click track and a drum loop at 70, 100,
  140 and 170 BPM are detected within `music.confirm_s` + one estimate, with the tempo within 4 %
  (or folded by 2). Speech, speech with pauses, a steady fan noise and a 1 Hz ticking clock are not
  detected. The robot check covers TV and real rooms.
- Synthetic test signals: a click track, a drum loop, speech, speech over music. Use generated audio
  only, never recorded rooms in the repo.
- Whether `MusicState` lives on the bus or behind an accessor, as long as the gate reads it without
  a lock on the listener's hot path.

## Open questions for the owner

None at this stage.
