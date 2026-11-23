# Dropped or slow cloud reply: retry, wait, spoken status. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A cloud reply lost before it said anything is retried up to 2 times within about 15 s,
a slow transcript is waited for longer, and the robot fills the waits with saved clips of its own
cloud voice («Fammi pensare…», «Un attimo, riprovo», «Scusa…»).

**Architecture:** `ConversationWorker._respond` gets a retry loop around `_realtime_turn` (a
`False` return already means: nothing played, no skill ran). A new clip cache
(`voice/clip_cache.py`) stores WAVs keyed by provider, model, voice, language and line; a new
runtime part (`runtime/clips.py`) records missing clips through the worker's existing fixed-line
path (`_rt_line` style, `ResponseRun` with a capturing speech object) while the conversation is
idle, and plays them. The listener and the worker call it for fillers.

**Tech Stack:** Python 3.11, asyncio, numpy, `wave` (stdlib), pytest, the realtime test rig
(`tests/realtime/rt_rig.py`, `turn_server.TurnServer`).

**Spec:** `docs/specs/2026-11-23-dropped-reply-retry-design.md` (read it first), SPEC
`docs/specs/kickoff.md` §8 (Realtime speech route, Non-verbal cues), §11A, §12, §12A.

## Global Constraints

- One branch `feat/reply-retry`, one PR, one commit per task. Worktree with its own venv (CLAUDE.md).
- No magic numbers: every number is a schema `param` with unit and range, and an Italian
  description in `webui/descriptions_it.json` (`tests/test_no_magic_numbers.py`).
- Core stays robot-agnostic; no `getattr` duck typing (`tests/test_robot_hierarchy.py`).
- Privacy: no audio, transcript or line text in logs, report or Live events. Clip files hold
  only the robot's synthetic voice, mode 0600.
- A retry never runs when any reply audio played or a skill ran (`turn.rt_tools`).
- Every task updates SPEC `kickoff.md` §8 (or §12A for report keys) in the same commit and cites
  the sections in the commit message.
- Lint and tests per task: `ruff check . && ruff format --check .` and the focused tests below,
  then `pytest packages/roboemot-core/tests/realtime -q` before the commit.

## Review Focus

1. The person speaks again while a retry waits: the new utterance is answered, the old turn ends
   with no clip and no apology (Task 4 test `..._a_newer_utterance_wins_over_a_retry`).
2. A retry's session is the same session (a stall after `response.created` that did not close
   it): the retry still sends the words as text, never relies on deleted audio items (Task 4
   test `..._a_retry_on_the_same_session_sends_the_words_as_text`).
3. The voice or the language changes while a clip is cached: the old clip never plays (Task 2
   test `..._a_changed_voice_or_language_misses_the_cache`).
4. The recording's audio is never heard and never enters the conversation history (Task 3
   test `..._a_recording_plays_nothing_and_leaves_no_history`).
5. Radio playing during a filler or a retry announcement: no clip plays over audible media
   (Task 3 test `..._no_clip_plays_over_audible_media`).

---

### Task 1: Settings and the longer transcript wait

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/config/schema.py` (`RealtimeRouteConfig`, ~4765)
- Modify: `packages/roboemot-core/src/roboemot/webui/descriptions_it.json`
- Modify: `docs/specs/kickoff.md` §8 Realtime speech route
- Test: `packages/roboemot-core/tests/realtime/test_realtime_retry_config.py` (create)

**Interfaces:**
- Produces: `RetryConfig` at `cfg.voice.realtime_route.retry` with fields
  `enabled: bool = True`, `max_retries: int = 2` (ge 0, le 3), `total_budget_s: float = 15.0`
  (ge 1, le 60), `reconnect_wait_s: float = 5.0` (ge 0.5, le 30), `filler_after_s: float = 2.5`
  (ge 0.5, le 10), `announce_first_retry: bool = True`, `clip_record_attempts: int = 3` (per
  key per day, ge 1, le 20), `retrying_lines: dict[str, str]`, `sorry_lines: dict[str, str]`,
  `thinking_lines: dict[str, str]` (same shape as `conversation.fallback_lines`, `advanced=True`).
  Default lines: it «Un attimo, riprovo.» / en "One moment, let me try again."; it «Scusa, ho
  perso la connessione. Puoi ripetere?» / en "Sorry, I lost the connection. Can you say it
  again?"; it «Fammi pensare…» / en "Let me think…".
- Produces: `transcript_timeout_s` default 9.0 (was 4.0; range unchanged).

- [ ] **Step 1: Write the failing tests**
```python
def test_retry_defaults_match_the_design():
    rt = RoboEmotConfig().voice.realtime_route
    assert (rt.retry.enabled, rt.retry.max_retries, rt.retry.total_budget_s) == (True, 2, 15.0)
    assert rt.retry.filler_after_s == 2.5 and rt.transcript_timeout_s == 9.0
    assert rt.retry.sorry_lines["it"].startswith("Scusa")

def test_every_retry_setting_has_an_italian_description(): ...  # same check the repo uses for other params
```
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/realtime/test_realtime_retry_config.py -v`. Expected: FAIL (no `retry`).
- [ ] **Step 3: Add `RetryConfig` and the default change**; update any test that pins `transcript_timeout_s == 4.0`.
- [ ] **Step 4: Run** the test file, `tests/test_no_magic_numbers.py`, and the realtime folder. Expected: PASS.
- [ ] **Step 5: Commit** `feat(realtime): retry settings and a longer transcript wait (SPEC 8, 11A)`.

### Task 2: Voice clip cache

**Files:**
- Create: `packages/roboemot-core/src/roboemot/voice/clip_cache.py`
- Test: `packages/roboemot-core/tests/test_clip_cache.py`

**Interfaces:**
- Produces:
  - `ClipKind = Literal["retrying", "sorry", "thinking"]`
  - `@dataclass(frozen=True) class ClipKey: provider: str; model: str; voice: str; language: str; line: str` with `def digest(self) -> str` (sha256 hex of all fields; the line text never appears in a file name).
  - `@dataclass(frozen=True) class Clip: samples: np.ndarray; rate_hz: int`
  - `class ClipCache: __init__(self, folder: Path)`, `get(self, kind: ClipKind, key: ClipKey) -> Clip | None`, `put(self, kind: ClipKind, key: ClipKey, clip: Clip) -> None` (writes `<kind>-<digest>.wav` mode 0600, then deletes every other `<kind>-*.wav`), `snapshot(self) -> dict[str, str]` (kind -> `cached`/`missing`).
  - Folder: `paths.data_dir(cfg.app.data_dir) / "voice-clips"` (check the exact config field name passed to `data_dir` in `paths.py:48` callers).

- [ ] **Step 1: Write the failing tests**
```python
def test_a_stored_clip_reads_back_for_the_same_key(tmp_path): ...  # samples equal, rate equal
def test_a_changed_voice_or_language_misses_the_cache(tmp_path):
    c = ClipCache(tmp_path); c.put("sorry", KEY, CLIP)
    assert c.get("sorry", replace(KEY, voice="Puck")) is None
    assert c.get("sorry", replace(KEY, language="en")) is None
def test_a_new_clip_replaces_the_old_file_of_its_kind(tmp_path):  # one sorry-*.wav left
def test_clip_files_are_private_and_hold_no_line_text(tmp_path):  # mode 0o600, line not in name
```
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/test_clip_cache.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** with stdlib `wave` (16-bit mono PCM).
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(voice): a cache of the robot's own voice clips (SPEC 8, 12)`.

### Task 3: Record and play clips

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/conversation/worker.py` (beside `_rt_line`, ~1998)
- Create: `packages/roboemot-core/src/roboemot/runtime/clips.py`
- Modify: `packages/roboemot-core/src/roboemot/runtime/startup.py` (start it after the expressor), `runtime/host.py` (`host.clips`), `runtime/diagnostics.py` (under `roboemot.voice` as `clips`)
- Test: `packages/roboemot-core/tests/realtime/test_realtime_clips.py`

**Interfaces:**
- Consumes: Task 1 `RetryConfig`, Task 2 `ClipCache`, `ClipKey`, `Clip`.
- Produces:
  - `ConversationWorker.record_line(self, text: str, language: str) -> concurrent.futures.Future[Clip | None]`: runs on the worker loop under `_rt_lock`, like `_rt_line`: `line_instructions(...)`, `ResponseRun(route, capture, ...)` with `tool_choice "none"`, where `capture` is a new `CaptureSpeech` class in `runtime/clips.py` that implements the `stream`/`new_utterance` calls `ResponseRun` uses and keeps the PCM instead of playing it. It never calls `_absorb` (no history). Returns `None` unless the reply transcript matches `text` (casefold, punctuation and spaces stripped). It deletes the response's server items afterwards (`route.delete_items`) where the wire supports it.
  - `class VoiceClips` in `runtime/clips.py`: `__init__(self, worker, host, cfg: Callable[[], RoboEmotConfig], cache: ClipCache, clock)`, `start()`, `close()`, `play(self, kind: ClipKind) -> bool` (False when missing, media audible, or the robot speaks; plays through `speech.play(samples, rate_hz)`), `key(self) -> ClipKey | None` (None when no realtime route is up; provider host and model from `route.settings(cfg).route`, the voice that `RealtimeRoute.session_update` sends, `cfg.conversation.language`, the line picked from the kind's `*_lines` for that language), `snapshot() -> dict` (per kind: `cached`, `missing`, `recorded`, `rejected_mismatch`, `played`).
  - Its thread records one missing clip at a time when `conversation_idle(host, now)` holds and no media is audible, at most `clip_record_attempts` per key per day.

- [ ] **Step 1: Write the failing tests** (TurnServer answers the line with audio and a matching reply transcript):
```python
def test_a_missing_clip_is_recorded_while_idle_and_then_plays(rigs): ...
def test_a_recording_plays_nothing_and_leaves_no_history(rigs):
    # sink pushed no samples during the recording; rig.history() unchanged
def test_a_recording_whose_words_differ_is_rejected(rigs):  # snapshot rejected_mismatch == 1, cache empty
def test_no_clip_plays_over_audible_media(rigs):  # play() False while media audible
```
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/realtime/test_realtime_clips.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `record_line`, `CaptureSpeech`, `VoiceClips`, wiring and the report.
- [ ] **Step 4: Run** the file and `tests/test_robot_hierarchy.py`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(runtime): record the robot's status lines in its cloud voice while idle (SPEC 8, 12, 12A)`.

### Task 4: Retry a lost reply

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/conversation/worker.py` (`_respond` ~985, `_answer_server_turn` ~1644, `_realtime_rounds_held` text/audio branch ~1276, `_Turn` ~204)
- Modify: `packages/roboemot-core/src/roboemot/webui/strings.py` (Live events), `docs/specs/kickoff.md` §8
- Test: `packages/roboemot-core/tests/realtime/test_realtime_retry.py`; update `test_realtime_turns.py::test_a_session_lost_before_any_audio_answers_through_the_chat_routes` (it now sets `retry.enabled=False`, plus a twin with retries on)

**Interfaces:**
- Consumes: `RetryConfig`, `VoiceClips.play(kind)` (via `host.clips`, may be None).
- Produces: `_Turn.rt_force_text: bool` (the next attempt sends `turn.utterance.text` as an `input_text` item even when `utterance.realtime` names the current generation); worker counts `retry` = `{attempts, recovered_1, recovered_2, gave_up, skipped_played, skipped_tool_ran, skipped_newer_utterance, skipped_no_words, skipped_budget}` in `snapshot()`; Live events `reply_retry` (attempt n of max), `reply_recovered`, `reply_gave_up`.
- Behaviour in `_respond`, after `answered = await self._realtime_turn(turn)` returns False:
  1. Stop unless `retry.enabled`, `not turn.event`, `not turn.interrupted`, `turn.utterance.text`, no newer utterance pending (`self._pending` non-empty), attempts < `max_retries`, and now − first failure < `total_budget_s`.
  2. Before attempt 1 only, if `announce_first_retry`: `host.clips.play("retrying")`.
  3. Wait up to `min(reconnect_wait_s, budget left)` for `route.ready(cfg)` off the loop.
  4. Set `rt_force_text`, reset the per-attempt state the way `_realtime_lost` leaves it, call `_realtime_turn` again.
  5. When all fail: without a local voice play `sorry` (else today's `_unvoiced` cue); with a local voice run today's chat backup (`_converse`), and only if that also fails say `sorry_lines` with Piper.
  `_answer_server_turn` uses the same loop (retries are robot-asked turns with the attached transcript words; no words: no retry, straight to step 5).

- [ ] **Step 1: Write the failing tests** (TurnServer scripts per request, e.g. `["close_before", "ok"]`):
```python
def test_a_reply_lost_before_any_audio_is_retried_and_answered(rigs):
    # ["close_before", "ok"]: one answer played, retry.recovered_1 == 1, chat_hosts() == []
def test_the_first_retry_is_announced_with_the_retrying_clip(rigs):  # clips.played["retrying"] == 1
def test_two_failed_retries_end_with_the_sorry_clip_on_a_robot_without_local_voice(rigs):
    # ["close_before"]*3: gave_up == 1, played["sorry"] == 1, no chat request
def test_a_robot_with_a_local_voice_tries_the_chat_route_after_the_retries(rigs):
def test_no_retry_after_a_skill_ran(rigs):  # skipped_tool_ran == 1
def test_no_retry_after_audio_played(rigs):  # ["close_mid"]: cut_mid_turn == 1, attempts == 0
def test_a_retry_on_the_same_session_sends_the_words_as_text(rigs):  # stall without close; srv saw an input_text item
def test_a_newer_utterance_wins_over_a_retry(rigs):  # skipped_newer_utterance == 1, no sorry clip
def test_the_total_budget_skips_a_retry_that_cannot_start_in_time(rigs):  # total_budget_s small
```
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/realtime/test_realtime_retry.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the loop, the `rt_force_text` branch, counters, Live strings (it and en in `strings.py`).
- [ ] **Step 4: Run** the realtime folder and `test_conversation_worker.py`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(conversation): retry a cloud reply lost before it spoke (SPEC 8, 11A, 12A)`.

### Task 5: Fillers for a slow transcript and a slow reply

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/voice/listener.py` (the handoff wait ~1795, `_rt_timeout` ~1844)
- Modify: `packages/roboemot-core/src/roboemot/realtime/route.py` (count `transcripts_after_timeout` when an input transcript arrives for a job the listener cancelled on timeout)
- Modify: `packages/roboemot-core/src/roboemot/conversation/worker.py` (first-reply filler), `runtime/startup.py` (wire `listener.on_slow_transcript`)
- Test: `packages/roboemot-core/tests/realtime/test_realtime_fillers.py`

**Interfaces:**
- Consumes: `VoiceClips.play("thinking")`, `retry.filler_after_s`.
- Produces: `VoiceListener.on_slow_transcript: Callable[[], bool]` (default `lambda: False`), called once per addressed handoff whose transcript has not settled `filler_after_s` after its audio was sent; worker plays `thinking` once per turn when no reply audio arrived `filler_after_s` after `turn.stamps["request"]`, not when the listener already filled this utterance (`Utterance` gains `filled: bool`, set by the listener). Counters: listener `fillers.transcript`, worker `fillers.reply`, route `transcripts_after_timeout`.

- [ ] **Step 1: Write the failing tests**
```python
def test_a_transcript_slower_than_the_filler_delay_plays_thinking_once(...):
def test_a_fast_transcript_plays_no_filler(...):
def test_a_slow_first_reply_plays_thinking_once_and_the_reply_follows_it(rigs):
def test_no_second_filler_when_the_transcript_already_had_one(rigs):
def test_a_transcript_after_the_timeout_is_counted(...):  # transcripts_after_timeout == 1
```
- [ ] **Step 2: Run** `pytest packages/roboemot-core/tests/realtime/test_realtime_fillers.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the realtime folder and `tests/test_voice_listener.py`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(voice): fill a slow transcript or reply with the robot's thinking line (SPEC 8, 12A)`.

### Task 6: PR

- [ ] Run the full suite `pytest -q` and `ruff check . && ruff format --check .`. Expected: PASS.
- [ ] Open the PR with the design link, the robot checks (a forced stall: hear «Un attimo, riprovo» then an answer; `transcript_timeouts` drops; `transcripts_after_timeout` tells late from missing), and the attribution line.
