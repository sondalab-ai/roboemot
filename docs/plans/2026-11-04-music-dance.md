# Music awareness and dance: epic plan

> **For agentic workers:** execute slice by slice with superpowers:subagent-driven-development. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** While music plays in the room, the robot answers only to the wake word. It moves to the beat of
room music and of its own self-soothing hums.

**Architecture:** The listener feeds each chunk it ingests to a cheap onset/tempo detector
(`core/voice/music.py`). The wake gate reads the detector's music intervals and requires the wake word over
music. A beat source (`core/motion/beat.py`) merges the soothe hum grid and the detector beat. A dance
renderer (`core/affect/dance.py`) copies the speaking renderer and plays abstract roles that the manifest
binds to channels.

**Tech stack:** Python 3.11+, numpy (already a dependency). No new dependency, no model, no thread.

**Spec:** `docs/specs/2026-11-04-music-dance-design.md` (the design, commit `7c8813e`). It feeds
`docs/specs/kickoff.md` (the SPEC), which each slice updates.

Paths: `core/` = `packages/roboemot-core/src/roboemot/`, `rm/` = `packages/roboemot-reachymini/src/roboemot_reachymini/`,
`tests/` = `packages/roboemot-core/tests/`.

## Status (2026-11-04, end of the slice 1 session)

- **Done.** Slice 1a (https://github.com/sondalab-ai/roboemot/pull/181) and slice 1b
  (https://github.com/sondalab-ai/roboemot/pull/183) are merged into `epic/music-dance`. The epic went to
  `main` with a merge commit in https://github.com/sondalab-ai/roboemot/pull/184 (`bb858c9`). The owner
  asked for this before Q1, so that one robot session covers all recent changes.
- **Waiting for the owner.** Q1 and three extra music checks are in `docs/plans/2026-11-04-qa-handoff.md`,
  section G. Slice 2 starts only after Q1 passes.
- **How to continue.** Keep `epic/music-dance`. Before slice 2, merge the latest `main` into it with a merge
  commit. A slice PR that contains a `main` merge also lands with a merge commit, not a squash.
- **Use these, not the plan text.**
  - The debug path is `roboemot.voice.music`, not `voice.music`.
  - The cost is CPU time of the listener thread (`time.thread_time`).
  - The schema has one more field than the plan: `music.min_band_periodicity` (default 0.35).
  - The line numbers under "Verified facts" are from `69a4440`. They are stale after #180 and #182.
- **Cost so far.** 6 agents of the 10 in the estimate: 2 Opus authors and 4 Sonnet reviewers, the 1a
  re-review included. The controller checked the 1b fix diff. Each slice needed one fix round.

### Rulings made in the slice 1 session

1. 1a fix round 1 also fixed three minor findings: a reader on another thread could see half of an update
   (state and intervals now live in one frozen view), the tempo fields had no unit or range, and the cost
   label said CPU while the timer measured wall time. Cost if wrong: a few extra lines.
2. #180 changed the listener while 1b was open. The 1b author merged `main` into the 1b branch first, and
   #183 landed with a merge commit. Cost if wrong: a larger 1b diff to review.
3. A config with `music.enabled` and a top band edge at or above half of `voice.sample_rate_hz` is refused,
   at load and on reload. Music is not skipped in silence. Cost if wrong: an 8 kHz setup must also lower
   `music.bands_hz`.
4. 1b fix round 1 also added a test that `music.enabled: false` turns the music rule off. Cost if wrong:
   one test.
5. The plan's onset-spread rule (onsets above the band median) passes for noise. The author replaced it
   with a per-band periodicity rule, which the plan allows. The ticking-clock test still passes. A
   broadband tick reads as music.

### Deferred minor findings (for the final review)

- 1a, behaviour:
  - A broadband tick reads as music. The SPEC states it.
  - One very large chunk can turn a "stopped" event into a "started" event in the same feed.
  - The interval stays open for `release_s` after the last beat. This is intended.
  - Back-dating covers the whole window, so noise before a song can enter the interval. This is intended.
- 1a, tests and docs:
  - The flywheel test loop restates the formula.
  - The speech evidence is thin: there is no suite test of regular-rhythm speech through the octave rule.
  - No test ties the clock rejection to `music.min_band_periodicity`.
  - The test helpers in `audio_signals.py` have no docstrings.
  - `music.beats_per_bar` is unused until slice 2.
- 1b:
  - On the realtime path the voice-at-face check still runs over music.
  - A bad-bands reload with no detector left gating off until a good reload. The band validator now
    refuses that reload.
  - No test checks `music.stopped` when a reload turns music off.
  - The gap code that subtracts lost audio needs a why-comment.
  - No Vosk-path test uses the real detector.

---

## Estimate (owner OK needed before any branch or agent)

| Slice | PRs into the epic | Author | Review | Robot gate after it |
| --- | --- | --- | --- | --- |
| 1a detector + load stage + debug section | 1 | Opus | Sonnet | – |
| 1b gate rule + listener wiring + SPEC 8/11A/12A | 1 | Opus | Sonnet | **owner: CPU + gate check (Q1)** |
| 2 beat source + dance to own sounds | 1 | Opus | Sonnet | – |
| 3 dance to room music | 1 | Opus | Sonnet | owner: servo-noise check (Q2) |
| 4 Reachy map + defaults from tuning | 1 | Opus (small) | Sonnet | owner: tuning run (Q3) |

- 5 slice PRs and 1 epic→`main` PR. 10 agents: 5 Opus authors and 5 Sonnet reviewers. Each fix round goes
  to the same author through SendMessage, so it needs no new agent. The main loop runs on Sonnet.
- About 8 agent-hours: 1a 2 h, 1b 2 h, 2 2 h, 3 1.5 h, 4 0.5 h.
- **Blocked on the owner:** slice 2 starts only after Q1 passes (the design checks the budget before any
  motion work). Slice 4 needs Q3 values. After Q1 passes, slice 1 goes to `main` alone (epic → `main`
  merge commit) before the dance work starts. The epic branch then continues from that `main` (owner,
  2026-11-04). The estimate is approved (owner, 2026-11-04).
- Integration branch `epic/music-dance` from `main`. `git branch -r | grep epic/` showed no other epic
  (2026-11-04). Merge `main` into the epic with a merge commit, never a squash.
- Run the epic from a fresh session. This session is already past 2× its budget for today.

## Owner decisions (2026-11-04, binding, from the design)

1. While music plays, every turn needs the wake word. The follow-up window does not count.
2. An utterance with the wake word is always addressed. The music rule never blocks it.
3. Over loud music Vosk may miss the wake word. The owner accepts that the robot then does not answer.
4. Keep moving lightly while the robot answers (`dance.speaking_gain`).
5. CPU budget: ≤ 1 % of one CM4 core on average. Above 2 % slice 1 stops and asks.
6. `music.enabled` and `dance.enabled` default to on.

## Plan decisions (mine, deviations from the design text)

- **Interval predicate.** Design section 2 gives `music: Callable[[float], bool]`, and the back-dating
  paragraph asks "music at any time in [utterance start, utterance end]". The plan uses the interval form:
  `music: Callable[[float, float], bool]`, called as `music(start, end)`. `outcome()` passes
  `(start, last_speech)`. `addressable()` gets an optional `end: float | None = None` (None: `end = start`).
  The listener's `_send_unheard` passes `ev.last_speech_stamp`. `_begin` passes only `start`.
- **Precedence.** With the wake word required and no wake word, music wins over every other reason: the
  outcome is `rejected_music`, never `rejected_no_window` or a face rejection. One reason per utterance keeps
  the Live tab stable.
- **What rejects a 1 Hz clock.** A tick is periodic at 60 BPM, so level, periodicity and stability pass.
  The detector also requires onset spread: at least `music.min_bands_active` of the bands carry onsets
  above their own median within the window. A tick is one narrow band. The builder may replace this rule
  with a measured cheaper one only if the "ticking clock" test still passes.

## Verified facts (code read 2026-11-04 on `69a4440`)

- `WakeGate` (`core/voice/wake.py:45-145`): the "required" predicate is duplicated in `outcome`
  (L99-101) and `addressable` (L129-131). `_follow` (L134) gives `window` / `rejected_no_window` /
  `rejected_other_speaker`. `at_face` runs under `self._lock`, so the music callable does too.
  `DECISIONS` is at L27.
- The listener builds the gate itself (`core/voice/listener.py:225`) and configures it in `apply_config`
  (L286-292). Gate calls: `_begin` → `_addressable` (L616, L641-648), `_send_unheard` (L730-742), the local
  final (L972).
- `_process` (L522-541) drops chunks while `speaking`, then calls `_ingest`. `resume()` (L543) calls `_ingest`
  directly. `chunk.gap` is counted in `_process`.
- `ListenerLoad.STAGES` (`core/voice/listener_load.py:25-35`) is a fixed tuple. `stage()` with an unknown
  name raises `KeyError`.
- The listener snapshot goes into the debug report through `core/runtime/diagnostics.py:48`.
- Soothe raw jobs open the speaking window: `SpeechOutput.play` (`core/voice/speech.py:609`) queues a raw
  job, `_play_raw` (L861) pushes it to the sink, and the Reachy speaker extends `SpeechActivity` on every
  push (`rm/speaker.py:332`). `play()` returns a job id, not a start stamp.
- The soothe synth plans hums with `beat_s` (`core/soothe/synth.py:183-212`) and blirps with events
  (L215-248). `SootheRunner._start` (`core/soothe/runner.py:142`) synthesizes and queues.
- `SpeakingRenderer` (`core/affect/speaking.py`, 529 lines) is built in `core/runtime/assembly.py:281`,
  reloaded in `core/runtime/reload.py:91`, ticked in `core/runtime/loop.py:187`, reported in
  `core/runtime/diagnostics.py:53`.
- `speech_motion_map` is a manifest-owned key (`core/config/stack.py:688`). Its body-yaw refusal is at
  `core/config/stack.py:1299-1305`. The schema class is `SpeechMotionMap` (`core/config/schema.py:4052`).
  The root config lists sections at `core/config/schema.py:5220-5245`.
- `tests/test_no_magic_numbers.py` scans an explicit module list (L18-34).
- The Reachy manifest exists twice: `configs/manifests/reachy-mini-wireless.yaml` and
  `rm/data/reachy-mini-wireless.yaml` (both have `speech_motion_map` at L235).

## Global constraints

- SPEC 11A: every threshold, time and gain is a schema field with a default, a unit and a range. Add every
  new behaviour module to `tests/test_no_magic_numbers.py`.
- SPEC 12: no audio, spectrum, onset envelope or band energy in a log, a bus event or the debug report.
  Counts, tempo, confidence and cost only.
- SPEC 4A/11/12A: core never names a robot, a joint or an SDK. `tests/robot_hierarchy_baseline.yaml` never
  grows. Robot specifics go only into the manifest map.
- No new thread, no queue, no dependency. The detector runs on the listener thread.
- The gate's music read takes no lock or one leaf lock: one immutable `MusicState` and one immutable tuple
  of intervals, each swapped by assignment.
- Units in names (`_s`, `_ms`, `_db`, `_bpm`, `_hz`). Degrees only in YAML.
- Test audio is generated in the test. Never commit a recording.
- Each slice updates `docs/specs/kickoff.md` and cites the SPEC sections in its commits.
- Validation per slice: `ruff check . && ruff format --check .`, focused tests, then the full `pytest`.

## Review focus

1. **Wake word over music is always addressed**, on the local path (L972) and the realtime path
   (`_send_unheard`). Test: `test_wake_word_over_music_is_addressed` in both listener test files.
2. **Music confirmed between speech start and end.** `_begin` skipped Vosk. At the end the gate says
   `rejected_music`, the buffer is decoded locally and no audio is sent. Test:
   `test_music_confirmed_mid_utterance_sends_no_audio` (`tests/realtime/test_realtime_listener.py`).
3. **1 Hz ticking clock and steady fan noise are not music.** Test: `test_clock_tick_is_not_music`,
   `test_fan_noise_is_not_music`.
4. **The robot's own hum never reaches the detector.** Test: `test_detector_gets_no_chunk_while_speaking`
   (listener with a fake `SpeechActivity` that reports speaking).
5. **Vosk cost over lyrics.** Every utterance over music runs Vosk. The detector has no test for this. Q1
   records the `decode` and `asr_*` stages with a song playing, and the cap item below has a measured trigger.

---

## Slice 1a: music detector (PR 1 into `epic/music-dance`)

**Files:**
- Create: `core/voice/music.py`, `tests/test_music_detector.py`, `tests/audio_signals.py` (generators)
- Modify: `core/config/schema.py` (new `MusicConfig`, root field `music`), `core/voice/listener_load.py`
  (`"music"` in `STAGES`), `tests/test_no_magic_numbers.py`

**Interfaces (produces):**
- `MusicConfig(Strict)` with `enabled: bool = True`, `hop_ms`, `bands_hz: list[tuple[float, float]]`,
  `window_s`, `estimate_interval_s`, `min_level_db`, `tempo_bpm: tuple[float, float] = (60, 180)`,
  `prior_bpm: tuple[float, float] = (100, 130)`, `min_periodicity`, `tempo_tolerance`, `min_active_share`,
  `min_bands_active`, `confirm_s`, `release_s`, `max_gap_s`, `blind_hold_s = 30`, `history_s`,
  `beats_per_bar = 4`. Design examples are defaults: hop 32 ms, window 6 s, estimate 1 s, release 8 s.
- `@dataclass(frozen=True) MusicState(playing: bool, tempo_bpm: float | None, beat_phase_s: float | None,
  confidence: float, stamp: float)`.
- `class MusicDetector`:
  - `__init__(cfg: MusicConfig, rate_hz: int, timer: Callable[[], float] = time.perf_counter)`
  - `feed(samples: np.ndarray, start: float) -> str | None` (returns `"started"` / `"stopped"` on a change)
  - `gap(stamp: float) -> None` (a gap or a speaking stretch began at `stamp`)
  - `state(now: float) -> MusicState` (lock-free, flywheel phase and blind hold applied)
  - `during(start: float, end: float) -> bool` (lock-free, over the back-dated interval history)
  - `apply_config(cfg: MusicConfig) -> None`, `snapshot() -> dict[str, Any]` (counts, state, tempo,
    confidence, cost min/median/p90/last in CPU ms per audio second)

**Steps:**
- [ ] Write `tests/audio_signals.py`: `click_track(bpm, seconds, rate)`, `drum_loop(bpm, seconds, rate)`
  (kick low band, hat high band), `speech_like(seconds, rate, pauses: bool)` (syllable-rate noise bursts
  with random gaps), `fan_noise(seconds, rate)`, `clock_tick(hz, seconds, rate)`, `mix(a, b, db)`. Seeded.
- [ ] Write failing tests in `tests/test_music_detector.py`:
  - `test_detects_click_and_drums` parametrised over 70, 100, 140, 170 BPM: `playing` within
    `confirm_s + estimate_interval_s`; `tempo_bpm` within 4 % of the true value or of ×2 / ×½.
  - `test_speech_is_not_music`, `test_speech_with_pauses_is_not_music`, `test_fan_noise_is_not_music`,
    `test_clock_tick_is_not_music`: never `playing` over 20 s.
  - `test_back_dates_the_interval`: after confirmation, `during(evidence_start + 0.5, evidence_start + 1)`
    is True. An interval that ended before the evidence began is False.
  - `test_release_after_music_stops` (stopped within `release_s + estimate_interval_s`).
  - `test_gap_clears_envelope_keeps_state`: a gap longer than `max_gap_s` → state stays playing, the tempo
    estimate restarts from fresh audio.
  - `test_blind_hold`: no `feed` for `release_s + 1` s → still playing. Past `blind_hold_s` → stopped.
  - `test_phase_runs_on_flywheel`: `state(t + k × period).beat_phase_s` stays equal within 1 hop.
  - `test_snapshot_has_no_audio`: snapshot values are numbers, bools or None only.
- [ ] Run `pytest tests/test_music_detector.py -v`. Expected: fail (module missing).
- [ ] Implement the schema and the detector. Algorithm as in design section 1. Measure FFT versus IIR
  bands on the Mac with a 60 s drum loop. Keep the cheaper option and put both numbers in the PR body.
  Timing uses the injected `timer` around `feed`.
- [ ] Add `voice/music.py` to the magic-numbers list and `"music"` to `STAGES`.
- [ ] Run the tests, `pytest tests/test_no_magic_numbers.py tests/test_config.py`, ruff. Expected: pass.
- [ ] Commit `feat(voice): music detector (SPEC 8, 11A)` and open the PR into the epic.

## Slice 1b: music gate and listener wiring (PR 2 into the epic)

**Files:**
- Modify: `core/voice/wake.py`, `core/voice/listener.py`, `docs/specs/kickoff.md` (sections 8, 11A, 12A,
  QA checklist)
- Test: `tests/test_wake_gate.py`, `tests/test_voice_listener.py`, `tests/realtime/test_realtime_listener.py`

**Interfaces:**
- Consumes: `MusicDetector.feed/gap/during/snapshot` from 1a.
- Produces: `WakeGate(policy, window_s, max_s=math.inf, music: Callable[[float, float], bool] | None = None)`.
  `addressable(*, enrolled_in_view, start, speaker=None, end=None)`. `"rejected_music"` in `DECISIONS`.
  `VoiceListener.music: MusicDetector | None` (read by slice 3). Bus events `music.started` and
  `music.stopped` with `{"tempo_bpm": float | None}` only.

**Steps:**
- [ ] Failing gate tests in `tests/test_wake_gate.py`:
  - `test_music_requires_wake_word_with_policy_never`: policy `never`, `music → True`, no wake word →
    `rejected_music`.
  - `test_music_closes_follow_up_window`: open window, then music → `rejected_music`.
  - `test_wake_word_over_music_is_addressed` → `wake_word`, window opens as today.
  - `test_music_wins_over_no_window_and_face`: closed window and a failing `at_face` → `rejected_music`, and
    `at_face` is not called.
  - `test_addressable_false_over_music` with `end` given and with `end=None`.
  - `test_no_music_callable_keeps_today`: existing decision tests pass unchanged.
- [ ] Extract `_required(enrolled_in_view) -> bool` (lock held) and use it in `outcome` and `addressable`.
  Run the existing gate tests first (pure refactor, they must pass). Then add the music rule in one place:
  `required = self._required(...) or music(start, end)`, and `_follow` returns `rejected_music` when music
  holds.
- [ ] Failing listener tests:
  - `test_detector_gets_no_chunk_while_speaking` and `test_resume_audio_skips_detector`.
  - `test_music_stage_in_load` (`load.snapshot()` has a `music` stage > 0 after a drum loop).
  - `test_music_events_published` (`music.started` then `music.stopped` on the bus, tempo only).
  - `test_wake_word_over_music_is_addressed` (local path).
  - `tests/realtime/test_realtime_listener.py`: `test_no_audio_sent_over_music`,
    `test_music_confirmed_mid_utterance_sends_no_audio`, `test_wake_word_over_music_is_addressed`.
- [ ] Wire the listener: build `MusicDetector` when `cfg.music.enabled` (rebuild or `apply_config` on reload).
  In `_process`, call `self.music.gap(start)` on `chunk.gap` and on a dropped speaking chunk. After the
  speaking check, call `self._timed("music", self.music.feed, chunk.samples, start)` before `_ingest`.
  Give the gate `music=self._music_during`. `_send_unheard` passes `end=ev.last_speech_stamp`. Add
  `snap["music"] = self.music.snapshot()`.
- [ ] Check that the Live tab lists `rejected_music` with no web UI change (decisions are rendered from the
  counts). If the UI has a label table, add the label there.
- [ ] Update `docs/specs/kickoff.md`: section 8 stage 2/3 (detector, music rule, realtime gate), 11A
  (`music.*`), 12A (debug section), QA checklist (Q1 below).
- [ ] Full `pytest`, ruff. Commit `feat(voice): wake word required over music (SPEC 8, 11A, 12A)`. Open the PR.
- [ ] After both PRs merge into the epic: hand Q1 to the owner. **Stop until Q1 results arrive.**

**Local decode cap over music (conditional).** If Q1 shows `decode` + `asr_*` above 15 % of one core with a
song playing (threshold confirmed by the owner, 2026-11-04), add a follow-up PR: while music plays, cap the local decode at
`voice.realtime_route.wake_decode_max_s`, as the realtime wake decode does. Otherwise do nothing.

## Slice 2: beat source and dance to own sounds (PR 3)

**Files:**
- Create: `core/motion/beat.py`, `core/affect/dance.py`, `tests/test_beat_source.py`, `tests/test_dance.py`,
  a test manifest with a generic `dance_motion_map` under `tests/data/`
- Modify: `core/config/schema.py` (`DanceConfig`, `DanceMotionMap`, root fields `dance` and
  `dance_motion_map`), `core/config/stack.py` (L688 key set, body-yaw refusal at L1299),
  `core/soothe/runner.py`, `core/voice/speech.py` (job start stamp), `core/runtime/assembly.py`,
  `core/runtime/reload.py`, `core/runtime/loop.py`, `core/runtime/diagnostics.py`,
  `tests/test_no_magic_numbers.py`, `docs/specs/kickoff.md` (sections 7, 7B, 11A, 12A)

**Interfaces:**
- `@dataclass(frozen=True) Beat(period_s: float, phase_s: float, strength: float, source: Literal["own", "room"])`.
  `phase_s` is the stamp of one beat.
- `class BeatSource`: `__init__(own: Callable[[float], Beat | None], room: Callable[[float], Beat | None])`,
  `beat_at(now) -> Beat | None` (own first, then room). Slice 2 passes `room=lambda now: None`.
- `SpeechOutput.started_at(job_id: int) -> float | None`: monotonic stamp when the sink began the job. Add it
  only if no such stamp exists today.
- `SootheRunner.beat_at(now) -> Beat | None` (hum: period `beat_s`, phase = job start, strength 1) and
  `SootheRunner.accents(since: float, now: float) -> list[float]` (blirp event stamps).
- `DanceMotionMap(Strict)`: `bob: dict[str, Literal[-1, 1]]`, `sway: dict[str, Literal[-1, 1]]`,
  `accent: ExpressionEntry | None`, `channels()`.
- `DanceConfig(Strict)`: `enabled = True`, `degrees` per role, `min_half_s = 0.5`, `lead_ms`,
  `arousal_gain`, `speaking_gain = 0.3`, `accent_min_interval_s`, `fade_in_s = 2`, `fade_out_s`.
- `DanceRenderer(cfg, mapping, dofs, *, body_yaw, max_amplitude, beat: BeatSource, speaking_channels:
  Callable[[], set[str]], ducked: Callable[[], set[str]])` with `commands(now) -> list[Command]`,
  `apply_config(...)`, `set_register`, `set_affect`, `circadian_factor`, `snapshot()`. Its structure copies
  `SpeakingRenderer`: one immutable settings bundle, fades, freeze, a last zero command.

**Tests (names and assertions):**
- `test_hum_publishes_grid_from_start_stamp` (period equals plan `beat_s`, phase equals the start stamp,
  not the queue time). `test_blirp_gives_accents_no_beat`.
- `test_bob_dips_once_per_beat`, `test_sway_one_side_per_bar`, `test_accent_rate_limited`.
- `test_fades_in_without_jump` (first command ≤ the fade slope), `test_fades_out_when_beat_goes`,
  `test_last_zero_command`.
- `test_no_dance_while_resting`, `test_yields_to_clip_channels`, `test_disabled_freezes`.
- `test_amplitude_product_capped` (`expression.max_amplitude`, manifest clamp).
- `test_map_with_body_yaw_refused` (load and `apply_config`).
- `test_robot_hierarchy.py` and `test_no_magic_numbers.py` pass with `motion/beat.py` and `affect/dance.py`
  in the list.

Wire the renderer after `p.speaking.commands(now)` in `core/runtime/loop.py:187`. Commit
`feat(motion): dance to own sounds (SPEC 7, 7B, 11A)`.

## Slice 3: dance to room music (PR 4)

**Files:** `core/motion/beat.py`, `core/affect/dance.py`, `core/runtime/assembly.py`, `tests/test_dance.py`,
`tests/test_beat_source.py`, `docs/specs/kickoff.md` (section 7).

**Interfaces:** consumes `VoiceListener.music.state(now)`. The room provider maps `MusicState` to
`Beat(60 / tempo_bpm, beat_phase_s, confidence, "room")`, or None while not playing.

**Tests:**
- `test_own_beat_wins_over_room`.
- `test_tempo_folding`: a full `bob` takes 2 x `min_half_s` (1 s by default). At 170 BPM (period 0.353 s)
  doubling gives 0.706 s and quadrupling 1.41 s, so `bob` moves every fourth beat; at 100 BPM (0.6 s) every
  second beat suffices. Every half lasts ≥ `min_half_s`.
- `test_lead_ms_shifts_peak`.
- `test_speaking_gain_scales_amplitude` while the speech timeline plays.
- `test_speaking_channel_handover`: a channel in the speaking `beat` map fades out of the dance while the
  speaking renderer emits on it. The other channels keep `speaking_gain`.
- `test_room_beat_strength_scales_amplitude`.

Commit `feat(motion): dance to room music (SPEC 7, 8)`. Then hand Q2 to the owner. If Q2 fails, add a PR:
the detector skips onsets within one hop of the dance's own `bob` phase.

## Slice 4: Reachy Mini map and defaults (PR 5)

**Files:** `configs/manifests/reachy-mini-wireless.yaml` and `rm/data/reachy-mini-wireless.yaml` (the same
`dance_motion_map` in both), `core/config/schema.py` (defaults from Q3), `docs/specs/kickoff.md` (section 4A,
QA checklist).

- Map (first guess, design section 5): `bob` → `head.pitch` dip, `sway` → `head.roll` plus a little
  `head.yaw`, `accent` → antennas in opposite directions. No body yaw, no translation.
- Tests: `test_manifest.py` loads the map. `test_golden_sync.py` (or a new check) asserts that the two
  manifests have the same `dance_motion_map`.
- Hand Q3 to the owner before the PR. Put the measured `dance.lead_ms`, amplitudes and `music.*` thresholds
  in the defaults, with the debug report stamps in the PR body.

Commit `feat(reachymini): dance motion map (SPEC 4A)`. Then open epic → `main` (merge commit).

## QA handoff (owner, on the robot)

- **Q1 (after slice 1).** Install the epic build. Play a song for 3 min, talk over it, say the wake word
  10 times. Then 3 min of speech only, 3 min of TV.
  Report with `roboemot debug --json`: `voice.music` cost (median and p90), `voice.realtime` stages
  `music`, `decode`, `asr_*`, wake decisions (`rejected_music`, `wake_word`), and wake hits out of 10.
  Pass: music cost median ≤ 1 % of one core (≤ 2 % with an owner OK), no `rejected_music` during speech
  only, no answer to lyrics. Above 2 %: retry with a longer hop and fewer bands before anything else.
- **Q2 (after slice 3).** Dance on, song on, then stop the song. Pass: `music.stopped` within
  `music.release_s` of the stop.
- **Q3 (slice 4).** Tune `dance.lead_ms` (the bob looks on the beat), amplitudes, and `min_periodicity`
  (biased toward "music"). Check `motion.external` shows no yield while dancing.

## Verification (each slice)

```bash
ruff check . && ruff format --check .
pytest tests/test_music_detector.py tests/test_wake_gate.py tests/test_voice_listener.py tests/realtime -q
pytest tests/test_no_magic_numbers.py tests/test_robot_hierarchy.py -q
pytest -q
```
