# Quiet-hours whisper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inside the quiet hours every sound through `SpeechOutput` plays at one night level, except announcements, and the realtime voice is asked to whisper.

**Architecture:** A small `NightVolume` object (new `runtime/night.py`) answers "how many dB to take off now". `SpeechOutput` applies it at its three push points (`_push`, `play_aside`, `play_media`). Jobs queued while `SpeechOutput.day_volume` is true skip it; the announcer sets that flag around an announcement. The conversation worker appends a whisper text to realtime instructions inside the window.

**Tech Stack:** Python 3.11, numpy, pydantic config schema, pytest.

**Spec:** `docs/specs/2026-11-26-quiet-hours-whisper-design.md`; SPEC sections 7A and 8 of `docs/specs/kickoff.md`.

## Global Constraints

- No magic numbers in behaviour code (SPEC 11A): every constant is a `roboemot.config.schema` field; `tests/test_no_magic_numbers.py` must pass.
- Night level default -18 dB, range -40 to 0, null disables the stage: `circadian.quiet_hours_volume_db`.
- Whisper text default `"whisper, very softly"`: `circadian.quiet_hours_instructions` (empty string: none).
- The night stage never makes a sound louder: the extra attenuation is never above 0 dB.
- Outside the quiet hours, or with `circadian.enabled` false, output samples are bit-identical to today.
- The speech timeline (speaking motion) receives samples before the night gain, like the duck today.
- Never log frames, audio or embeddings (SPEC 12).
- Commit messages cite SPEC 7A and 8. One branch `feat/quiet-hours-whisper` off `main`, one commit per task, one PR (owner rule: single PR for agent-only work).

## Review Focus

1. The window edge mid-reply: a reply that starts at 22:29:58 and runs past 22:30 drops to the night level at the next chunk, not a clip or a jump louder. Test in Task 2 (`test_night_level_read_per_push`).
2. A state change during quiet hours (resting to awake after a wake): total attenuation stays at the night level. Test in Task 1 (`test_extra_db_tracks_state_offset`).
3. Announcement audio queued just before `day_volume` goes false still plays at day volume (the flag is captured at queue time). Test in Task 2 (`test_day_volume_captured_at_queue_time`).
4. Hot reload of `quiet_hours_volume_db` takes effect on the next push without restart. Test in Task 1 (`test_reads_live_config`).
5. An announcement that raises must not leave `day_volume` true forever. Test in Task 3 (`test_day_volume_reset_when_announcement_raises`).

---

### Task 1: Config fields and `NightVolume`

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/config/schema.py` (class `CircadianConfig`, next to `quiet_hours` near line 805)
- Create: `packages/roboemot-core/src/roboemot/runtime/night.py`
- Test: `packages/roboemot-core/tests/test_night_volume.py`

**Interfaces:**
- Produces:
  - `CircadianConfig.quiet_hours_volume_db: float | None` (default -18.0, unit dB, ge -40, le 0)
  - `CircadianConfig.quiet_hours_instructions: str` (default `"whisper, very softly"`, advanced)
  - `class NightVolume` in `roboemot.runtime.night`:
    - `__init__(self, config: Callable[[], RoboEmotConfig], quiet: Callable[[], bool], state_db: Callable[[], float])`
    - `active(self) -> bool`: `quiet()` and `quiet_hours_volume_db is not None`
    - `extra_db(self, *, state_applied: bool) -> float`: 0.0 when not active; else `min(0.0, night - state_db())` when `state_applied`, else `night`

- [ ] **Step 1: Write the failing tests** (fake config via `RoboEmotConfig` defaults with `model_copy`, fakes for `quiet` and `state_db`)

```python
def test_inactive_outside_window():
    nv = NightVolume(lambda: cfg, quiet=lambda: False, state_db=lambda: -6.0)
    assert not nv.active()
    assert nv.extra_db(state_applied=True) == 0.0

def test_extra_db_tracks_state_offset():
    state = [-6.0]
    nv = NightVolume(lambda: cfg, quiet=lambda: True, state_db=lambda: state[0])
    assert nv.extra_db(state_applied=True) == -12.0   # resting: -6 + -12 = -18
    state[0] = 0.0
    assert nv.extra_db(state_applied=True) == -18.0   # woken: still -18 total
    assert nv.extra_db(state_applied=False) == -18.0  # radio

def test_never_louder():
    nv = NightVolume(lambda: cfg_with(night=-3.0), quiet=lambda: True, state_db=lambda: -6.0)
    assert nv.extra_db(state_applied=True) == 0.0

def test_null_disables():
    nv = NightVolume(lambda: cfg_with(night=None), quiet=lambda: True, state_db=lambda: 0.0)
    assert not nv.active() and nv.extra_db(state_applied=False) == 0.0

def test_reads_live_config():
    box = [cfg_with(night=-18.0)]
    nv = NightVolume(lambda: box[0], quiet=lambda: True, state_db=lambda: 0.0)
    box[0] = cfg_with(night=-24.0)
    assert nv.extra_db(state_applied=False) == -24.0
```

- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_night_volume.py -v`. Expected: FAIL, `ModuleNotFoundError: roboemot.runtime.night`.
- [ ] **Step 3: Add the two schema fields** with `param(...)` (SPEC 7A in the description), then implement `NightVolume` (module docstring citing SPEC 7A and 8; ~4-line docstring per method).
- [ ] **Step 4: Run** the test file plus `pytest packages/roboemot-core/tests/test_no_magic_numbers.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(circadian): night volume level for the quiet hours (SPEC 7A)`.

### Task 2: Night stage in `SpeechOutput`

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/voice/speech.py` (`_RawJob`, `_Job`, `SpeechOutput.__init__`, `say`, `stream`, `play`, `play_aside`, `play_media`, `_push`, `snapshot`)
- Test: `packages/roboemot-core/tests/test_speech.py`

**Interfaces:**
- Consumes: `NightVolume.extra_db(*, state_applied: bool) -> float` (Task 1), passed as a plain callable so `speech.py` does not import runtime.
- Produces:
  - `SpeechOutput.night_db: Callable[..., float]`, attribute, default `lambda *, state_applied: 0.0`
  - `SpeechOutput.day_volume: bool`, attribute, default False
  - `_RawJob.day: bool = False`, `_Job.day: bool = False`: set from `self.day_volume` when the job is queued
  - `snapshot()["night_db"]`: the `extra_db(state_applied=False)` value now

- [ ] **Step 1: Write the failing tests** in `test_speech.py`, reusing its fake sink and fake backend:
  - `test_night_gain_on_raw_play`: `night_db` returns -20.0; `play(ones)`; pushed peak equals `db_to_gain(-20)` within 1e-6.
  - `test_night_gain_on_speech_and_stream`: same for a `say` sentence and a `stream` job.
  - `test_night_gain_on_aside_and_media`: `play_aside` and `play_media` push samples scaled by `db_to_gain(-20)`; `play_media` calls `night_db(state_applied=False)`, the others `state_applied=True` (record the keyword in the fake).
  - `test_day_volume_skips_night_gain`: `day_volume = True`, `play(ones)`, then `day_volume = False`; pushed peak 1.0.
  - `test_day_volume_captured_at_queue_time`: queue with `day_volume = True`, set False before the output thread pushes; pushed peak 1.0.
  - `test_night_level_read_per_push`: a `stream` job; `night_db` returns 0.0 for chunk 1, -20.0 for chunk 2; chunk 2 is scaled, chunk 1 is not.
  - `test_timeline_gets_samples_before_night_gain`: the timeline's received samples equal the unattenuated ones.
  - `test_no_night_stage_by_default`: default `SpeechOutput`, pushed samples identical to today's.
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_speech.py -k "night or day_volume" -v`. Expected: FAIL (`AttributeError: night_db`).
- [ ] **Step 3: Implement.** In `_push`, after the duck, multiply by `db_to_gain(self.night_db(state_applied=True))` unless `job.day`, then `np.clip(-1, 1)`; the timeline keeps the pre-gain `samples`. In `play_aside`, multiply under the lock by the same value. In `play_media`, use `state_applied=False`. Skip the multiply when the value is 0.0, so output stays bit-identical. Update the module docstring (night stage paragraph, SPEC 7A).
- [ ] **Step 4: Run** `pytest packages/roboemot-core/tests/test_speech.py packages/roboemot-core/tests/test_speech_timeline.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(voice): night gain stage on the speech output (SPEC 7A, 8)`.

### Task 3: Wiring, announcements at day volume, realtime whisper, SPEC

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/runtime/startup.py` (`start_voice_output`, conversation worker construction near line 180)
- Modify: `packages/roboemot-core/src/roboemot/tools/announce.py` (around `event_turn` / `local_line` / fallback cue, lines ~285-310)
- Modify: `packages/roboemot-core/src/roboemot/conversation/worker.py` (`__init__`, `_rt_instructions` near line 1230)
- Modify: `docs/specs/kickoff.md` (section 7A "Effects by state" / quiet hours text, section 8 "Voice, expressivity and mood" realtime paragraph and the `circadian` config block near line 827)
- Test: `packages/roboemot-core/tests/test_tool_announce.py`, the worker realtime-instruction test file (find with `grep -ln "mood_instructions" packages/roboemot-core/tests`), `packages/roboemot-core/tests/test_night_volume.py`

**Interfaces:**
- Consumes: `NightVolume` (Task 1), `SpeechOutput.night_db` and `SpeechOutput.day_volume` (Task 2), `CircadianRunner.quiet_now() -> bool` (exists, `circadian/runner.py:310`), `effects_now(host, stack, p).volume_db` (exists, `runtime/startup.py:129`).
- Produces: `ConversationWorker(..., night: Callable[[], bool] = lambda: False)`.

- [ ] **Step 1: Write the failing tests:**
  - `test_announcement_plays_at_day_volume` (announce): with a fake output, `day_volume` is True while `event_turn` and the fallback `local_line` and cue run, and False once the announcement audio ended.
  - `test_day_volume_reset_when_announcement_raises`: `event_turn` raises; `day_volume` is False afterwards.
  - `test_realtime_instructions_whisper_at_night` (worker): with `night=lambda: True`, `_rt_instructions` output ends with `cfg.circadian.quiet_hours_instructions`; with `night=lambda: False` it does not contain it; with the text set to `""` nothing is added.
- [ ] **Step 2: Run** those tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - `startup.start_voice_output`: build `NightVolume(stack.effective, quiet=lambda: (r := host.circadian_runner) is not None and r.quiet_now(), state_db=lambda: effects_now(host, stack, p).volume_db)` and set `output.night_db = night.extra_db`. Pass `night=night.active` to `ConversationWorker`.
  - `announce`: set `output.day_volume = True` before `event_turn`; reset it to False in a `finally` once the announcement audio ended (step 7) or on any exception. The follow-up reply runs after that, so it whispers.
  - `worker._rt_instructions`: append `cfg.circadian.quiet_hours_instructions` after the mood text when `self._night()` is true and the text is not empty.
  - SPEC: state that inside the quiet hours every sound plays at `circadian.quiet_hours_volume_db` total (owner decision 2026-11-26), announcements excepted, the radio included, the realtime route gets `quiet_hours_instructions`; add both fields to the `circadian` YAML block; fix the line "the volume ... so the robot stays quiet at night" in section 8.
- [ ] **Step 4: Run** the full suite and lint: `pytest -q` and `ruff check . && ruff format --check .`. Expected: PASS (known Mac-only failure `test_nothing_private_is_in_the_answer_and_nothing_is_logged` may appear; confirm against `main`).
- [ ] **Step 5: Commit** `feat(runtime): whisper in the quiet hours, announcements at day volume (SPEC 7A, 8)`, then push and open the PR with the design doc and this plan in it.

### After merge (owner, on the robot)

- Reinstall, then at 23:00 talk to the robot from 1 m and from the next room. Tune `circadian.quiet_hours_volume_db` from that.
