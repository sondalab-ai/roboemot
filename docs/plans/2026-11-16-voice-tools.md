# Voice Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the robot act on free speech (sleep, timers, reminders, weather, radio) through LLM tool calls, with a plugin point for more tools and a Tools tab in the web UI.

**Architecture:** A new entry-point group `roboemot.skills` adds tools to the existing `SkillRegistry` at startup. Core ships five tool groups through that group. Three new core parts serve them: a persistent scheduler with robot-initiated turns, a requested-nap state in the circadian runner, and a media player that pauses while the robot speaks.

**Tech Stack:** Python 3.11, pydantic v2 config (`param(...)`), httpx (already a core dependency), pytest with `asyncio_mode=auto`, vanilla JS web UI with `node` tests in `tests/webui_js/`.

**Spec:** `docs/specs/2026-11-16-voice-tools-design.md` (design) and `docs/specs/kickoff.md` (SPEC; T0 adds the text). Read both.

## Global Constraints

- Every behavioural constant is a `param(...)` field with default, unit and range (SPEC 11A). Add each new module to `BEHAVIOUR_MODULES` in `tests/test_no_magic_numbers.py`.
- Network tools are off by default: `tools.weather.enabled` and `tools.radio.enabled` use `param(False, ..., opt_in=True)` (SPEC 12).
- Logs carry tool names, item ids and counts only. They never carry reminder text, station URLs, coordinates or place names (SPEC 12).
- Core never names a robot. `tests/test_robot_hierarchy.py` stays green, and its baseline stays empty (SPEC 4A, 12A).
- Every new web UI string and every new setting description exists in EN and IT (`webui/strings.py`, `webui/descriptions_it.json`). `tests/test_webui_one_language.py` enforces this.
- Tool parameters use only the types that `skills/schema.py` accepts (`str`, `int`, `float`, `bool`, `Literal`, `X | None`, `Annotated` with `Range`/`MaxLen`/`Desc`).
- Tool descriptions are in English. The model replies in the conversation language.
- Branch: integration branch `epic/voice-tools` from `main`. One PR per slice into it, full CI on each PR, squash-merge. Use a merge commit only when a slice merges `main` in.
- Worktree agents need their own venv. Check `python -c "import roboemot; print(roboemot.__file__)"` before trusting a test run (CLAUDE.md).
- Author briefs: `isolation: worktree` starts at `main`. Each brief says to run `git switch -c <slice> <epic/voice-tools tip sha>` first and never to run git in the main checkout.
- `available=` callables read `host.config()` on each call, never a config captured when the factory ran, so hot reload applies.
- Saved `conversation.llm_skills` overrides: no shipped layer sets it today. A saved web UI override hides the new tools. T1 logs one startup warning that names each core tool missing from the list.
- Validation per slice: `ruff check . && ruff format --check .`, `pytest` (core and reachymini), `node --test tests/webui_js` for T7, and `scripts/audit_webui.sh` for T7.

## Review Focus

1. **Restart during a due window.** RoboEmot restarts 30 s after a timer became due. Expected: the timer announces at start (within `missed_grace_s`) and is not lost or fired twice.
2. **DST and time-zone change.** A daily reminder at 08:00 `Europe/Rome` across the October DST switch. Expected: it fires at 08:00 local on both days. Pin it in T3 with a `zoneinfo` test on 2026-10-25.
3. **Radio heard by the microphone.** The radio plays song lyrics right after a turn, while the follow-up window is open. Expected: no reply, and the radio does not pause itself on its own lyrics. Only the wake word opens a turn while media plays, and the radio pauses for that turn. Pin it in T6 with a test that opens the follow-up window, sets `media.playing`, and feeds a final without the wake word (it is not submitted), plus a player test where the VAD fires on radio sound (pushing continues).
4. **Tool offered while disabled.** An admin turns weather off while a session is open. Expected: the next request no longer offers `weather`, and a call that was already in flight returns `error: unavailable`. Pin it in T1 (`available`) and in T5.
5. **The model passes a time in the past or a malformed date.** Example: "remind me yesterday at 6". Expected: the skill returns an error result that the model can phrase, and the store does not change. Pin it in T3.

---

### Task T0: SPEC text and design doc

**Files:**
- Modify: `docs/specs/kickoff.md`: add a "Voice tools" subsection under section 10, LLM tools. Amend section 7A (requested nap), section 8 (robot-initiated turn, radio pause), section 11A (`tools.*` section) and section 12 (network tools are opt-ins, no logging of tool content).
- Add: `docs/specs/2026-11-16-voice-tools-design.md` (already written, uncommitted).

- [ ] **Step 1:** Write the SPEC text from the design doc. Each owner decision keeps its date (2026-11-16). Every number appears as "the default of `tools.…`".
- [ ] **Step 2:** Run `git diff --stat` and check that only the two docs changed.
- [ ] **Step 3:** Commit as `docs(spec): voice tools (SPEC 7A, 8, 10, 11A, 12)` and open the PR into `epic/voice-tools`.
- [ ] **Step 4 (owner, parallel, does not block T1):** Pollen probe on the robot. Turn the cloud-voice profile on and say "set a timer for one minute" once T3 is installed. Record whether a `function_call` arrives (the log shows `tool_calls` counts). Until T3 lands, use `play` with "nod".

### Task T1: Tool plugin point and `tools` config section

**Files:**
- Create: `src/roboemot/skills/plugins.py`, `src/roboemot/skills/host.py`, `src/roboemot/fileio.py`
- Modify: `src/roboemot/skills/registry.py` (the `available` flag), `src/roboemot/plugins.py` (GROUPS), `src/roboemot/config/schema.py` (`ToolsConfig`, `conversation.llm_skills` default), `src/roboemot/runtime/startup.py:170` (add plugins), `src/roboemot/realtime/check.py:181-196` (the same plugins in `offline_tools`), `src/roboemot/config/stack.py:2016` and `src/roboemot/secret_store.py:151` (use `fileio.replace_file`), `packages/roboemot-core/pyproject.toml` (no entry points yet)
- Test: `tests/test_skill_plugins.py`, `tests/test_robot_hierarchy.py:199` (add `"roboemot.skills"` to the set), `tests/test_skills.py`, `tests/test_fileio.py`

**Interfaces:**
- Produces:
  - `roboemot.fileio.replace_file(path: Path, text: str, mode: int) -> None`. It does what `_replace_file` does today: temp file in the same folder, fsync, `os.replace`, then a best-effort fsync of the folder.
  - `skill(..., available: Callable[[], bool] | None = None)`. `Skill.available` is stored. `SkillRegistry.offered()` also drops a skill whose `available()` is False. `execute()` returns the error code `unavailable` for such a skill. The check runs at call time.
  - `roboemot.skills.host.ToolHost`, a dataclass. T1 fields: `config: Callable[[], RoboEmotConfig]`, `wall_now: Callable[[], datetime]` (tz-aware, `circadian.timezone`, via `timezones.now`), `bus: Bus`. Later tasks add fields with default `None`: `nap` (T2), `scheduler` (T3), `media` (T6), `http` (T5).
  - `roboemot.skills.plugins.SKILLS_GROUP = "roboemot.skills"`.
  - `load_skill_plugins(host: ToolHost) -> list[Callable[..., Any]]`. It loads every entry point in the group, without a `robot.id` key. Each entry point is a factory `(ToolHost) -> list[skill functions]`. A factory that raises or returns a non-skill is logged by entry-point name and skipped. One broken plugin never stops the run. Results are sorted by entry-point name.
  - `GROUPS["roboemot.skills"] = GroupContract("factory (ToolHost) -> list of @skill functions: voice tools, not keyed by robot.id (SPEC 11, Voice tools)", factory_args=1)`. Update the `plugins.py` module docstring: this group is the one group not keyed by `robot.id`.
  - `build_skill_registry(env: SkillEnv, host: ToolHost) -> SkillRegistry` in `skills/plugins.py`: `default_skills(env)` plus every plugin skill (a duplicate name is logged and skipped).
  - `ToolsConfig(Strict)` on `RoboEmotConfig.tools`. In T1 it is empty. Each later task adds its own sub-model (`scheduler`, `weather`, `media`, `radio`). `circadian.requested_nap_min_s` lives in `CircadianConfig` (T2).

- [ ] **Step 1: Write the failing tests**
  - `test_fileio.py::test_replace_file_is_atomic_and_owner_only`: after the write, the mode is `0o600` and the content matches.
  - `test_skills.py::test_unavailable_skill_is_not_offered_and_refused`: with `available=lambda: flag`, the skill is offered while `flag` is True. With `flag=False`, `offered()` drops it and `execute(...)` returns `ToolOutcome(ok=False, code="unavailable")`.
  - `test_skill_plugins.py::test_plugins_load_in_name_order_and_skip_broken`: monkeypatch `importlib.metadata.entry_points` with three entry points (`b` good, `a` good, `c` raises). The result is the skills of `a` then `b`. The log names `c` and not the exception text.
  - `test_skill_plugins.py::test_startup_registry_includes_plugins_named_in_llm_skills`: build the registry the way `startup.py` does, through a new helper `build_skill_registry(env: SkillEnv, host: ToolHost) -> SkillRegistry` in `skills/plugins.py`. A plugin skill is in `.names()`.
  - `test_robot_hierarchy.py`: the expected set includes `"roboemot.skills"`.
- [ ] **Step 2:** Run `pytest tests/test_fileio.py tests/test_skills.py tests/test_skill_plugins.py tests/test_robot_hierarchy.py -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement the interfaces above. `startup.py:170` and `realtime/check.py` call `build_skill_registry`.
- [ ] **Step 4:** Run `pytest -q` and `ruff check . && ruff format --check .`. Expected: all pass.
- [ ] **Step 5:** Commit as `feat(skills): roboemot.skills plugin group and tools config (SPEC 11, 11A)`.

### Task T2: Sleep tool and requested nap

**Files:**
- Modify: `src/roboemot/circadian/nap.py` (`DayNap`), `src/roboemot/circadian/runner.py` (`CircadianRunner`), `src/roboemot/runtime/host.py` (`RuntimeHost.request_nap`), `src/roboemot/config/schema.py` (`CircadianConfig.requested_nap_min_s`)
- Create: `src/roboemot/tools/__init__.py`, `src/roboemot/tools/sleep.py`
- Modify: `packages/roboemot-core/pyproject.toml`: `[project.entry-points."roboemot.skills"] sleep = "roboemot.tools.sleep:skills"`
- Test: `tests/test_circadian_nap.py`, `tests/test_circadian_runner.py`, `tests/test_tool_sleep.py`

**Interfaces:**
- Consumes: `ToolHost` and `load_skill_plugins` (T1).
- Produces:
  - `circadian.requested_nap_min_s: float = param(600.0, "...", unit="s", ge=60, le=7200)`.
  - `DayNap.start_requested(t: float, min_s: float) -> None`. It sets `napping=True` and `requested_until = t + min_s`. `DayNap.requested(t) -> bool` is True while `t < requested_until`.
  - `CircadianRunner.request_nap(now: float) -> Literal["ok", "already_resting", "disabled", "asleep"]`. While `requested(t)` holds, `_activity` and `_stimulus` ignore face, speech, light, motion and loud. Only `wake()` and the wake-word branch of `_on_utterance` (the name and the generic wake phrases) end the nap. After the window, the usual rules apply.
  - `RuntimeHost.request_nap() -> str` arms the nap and returns at once ("ok" when armed). The nap starts when the turn that called the tool has ended: the worker's turn-end hook, after the last reply audio. It does not start when speech is merely idle at call time, because the model usually says good night after the tool result. When the nap starts, the conversation's follow-up window closes, so only the robot's name addresses it.
  - `ToolHost.nap: Callable[[], str] | None`.
  - `roboemot.tools.sleep.skills(host) -> list`: one skill `go_to_sleep()`, `llm_visible=True`, `available=lambda: host.nap is not None`. It returns `SkillResult({"status": status})`. Description: "Take a nap now because the person asked; say a short good night in the same turn; the robot's name wakes you".
  - `conversation.llm_skills` default gains `"go_to_sleep"`.

- [ ] **Step 1: Write the failing tests**
  - `test_circadian_runner.py::test_requested_nap_ignores_face_motion_and_speech_inside_the_window`: request the nap at `t=0`. Face, motion and addressed speech without the wake word at `t=30` do not wake it. A wake word at `t=30` wakes it.
  - `test_circadian_runner.py::test_requested_nap_uses_normal_rules_after_the_window`: with `min_s=600`, a face at `t=601` wakes it.
  - `test_circadian_runner.py::test_requested_nap_refused_while_asleep`: with the platform asleep, the status is `"asleep"`.
  - `test_tool_sleep.py::test_go_to_sleep_starts_after_the_calling_turn_ends`: the tool is called at `t=0` with nothing speaking. The reply speaks from `t=0.5` to `t=2`. `request_nap` is called once, after `t=2`.
  - `test_tool_sleep.py::test_nap_start_closes_the_follow_up_window`: a final without the wake word right after the nap starts is not submitted.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `pytest -q` and ruff. Expected: pass. Add `tools/sleep.py` to `BEHAVIOUR_MODULES`.
- [ ] **Step 5:** Commit as `feat(tools): go_to_sleep and requested nap (SPEC 7A, 10)`.

### Task T3: Scheduler, store, timer and reminder tools

**Files:**
- Create: `src/roboemot/tools/schedule.py` (pure model and recurrence), `src/roboemot/tools/scheduler.py` (store and runner), `src/roboemot/tools/timers.py` (skills)
- Modify: `src/roboemot/paths.py` (`SCHEDULE_FILE = "tools_schedule.json"`, `schedule_path(configured: str = "") -> Path`), `src/roboemot/config/schema.py` (`ToolsConfig.scheduler`), `src/roboemot/runtime/startup.py` (build the scheduler, `host.scheduler`, the runner thread, cleanup), `src/roboemot/bus/__init__.py` (topic `tools.due`), `pyproject.toml` (`timers = "roboemot.tools.timers:skills"`)
- Test: `tests/test_tool_schedule.py`, `tests/test_tool_scheduler.py`, `tests/test_tool_timers.py`

**Interfaces:**
- Produces:
  - `SchedulerConfig`:
    - `max_items` (20, 1–200)
    - `text_max_chars` (200, 10–1000)
    - `timer_max_s` (86400, 10–604800)
    - `missed_grace_s` (900, 0–86400)
    - `tick_s` (1.0, 0.2–10)
    - `announce_timeout_s` (8, 1–60)
    - `announce_max_wait_s` (60, 0–600)
    - `repeat_after_s` (30, 5–600)
    - `repeat_max` (2, 0–10)
  - `Repeat = Literal["once", "daily", "weekdays", "days"]`.
  - `@dataclass(frozen=True) Item(id: str, kind: Literal["timer", "reminder"], text: str, due: datetime, created: datetime, repeat: Repeat = "once", days: tuple[int, ...] = (), local_time: time | None = None)`. `due` is aware UTC. `local_time` is set for repeating reminders and is the anchor for recurrence.
  - `next_due(item: Item, after: datetime, tz: ZoneInfo) -> datetime | None`. It returns the next occurrence of `local_time` in `tz` after `after`, or None for `once`. `days` uses ISO weekdays 1–7.
  - `Scheduler(path: Path, cfg: Callable[[], SchedulerConfig], now: Callable[[], datetime], tz: Callable[[], ZoneInfo])`:
    - `load(started: datetime) -> None`. An item whose due time is older than `missed_grace_s` goes to `missed()`.
    - `add_timer(seconds: int, label: str | None) -> Item`
    - `add_reminder(text: str, due_local: datetime, repeat: Repeat, days: tuple[int, ...]) -> Item`
    - `cancel(item_id: str) -> bool`
    - `items() -> list[Item]` (sorted by `due`)
    - `missed() -> list[Item]`
    - `clear_missed() -> None`
    - `due(now: datetime) -> list[Item]`
    - `fired(item_id: str) -> None`. A `once` item is removed. A repeating item moves to `next_due`.
    - Every change writes the file with `fileio.replace_file(path, json, 0o600)`. It is thread-safe under one lock. Errors raise `ScheduleError(code)`, where code is one of `full`, `too_long`, `in_past`, `too_far`, `bad_days`.
    - The process-restart grace applies only in `load`. A due item never expires while the process runs (design §3).
  - `SchedulerRunner(scheduler, bus, cfg)`: a daemon thread. Every `tick_s` it publishes `tools.due` with payload `[item.id, ...]`, once per item, until `fired` or `defer` is called. T3 marks items fired only when no announcer is attached (log only). T4 replaces this.
  - `ToolHost.scheduler: Scheduler | None`.
  - Skills in `tools/timers.py`, all `llm_visible=True`, `available=lambda: host.scheduler is not None`:
    - `timer_set(seconds: Annotated[int, Range(1, ...)], label: str | None)`
    - `timer_cancel(which: str)`, with `choices` = active timer ids and labels
    - `reminder_set(text: str, date: str | None, time: str, repeat: Repeat, days: str | None)`. `date` is ISO `YYYY-MM-DD`, `time` is `HH:MM`, `days` is like `"1,3,5"`.
    - `reminder_cancel(which: str)`
    - `schedule_list()`
    - `describe=` adds the current local date, weekday and time in `circadian.timezone`, so the model resolves "tomorrow at 6". Results are JSON with `id`, the local due time and `kind`. Errors are `{"error": code}`.
  - `conversation.llm_skills` default gains the five names above. T3 also adds `weather`, `radio_play`, `radio_stop` and `radio_volume` to the default now (ruling 2026-11-16: `available` hides them until T5/T6 land and the admin enables them). The names are offered only once a registered skill has them.

- [ ] **Step 1: Write the failing tests**
  - `test_tool_schedule.py::test_daily_reminder_keeps_local_time_across_dst`: `tz=Europe/Rome`, `local_time=08:00`, after `2026-10-24T08:00+02:00`. The next due is `2026-10-25T08:00+01:00`, then `2026-10-26T08:00+01:00`.
  - `test_tool_schedule.py::test_weekdays_skip_weekend`: from Friday 09:00 with `local_time` 08:00, the next due is Monday 08:00.
  - `test_tool_scheduler.py::test_restart_inside_grace_keeps_item_and_outside_moves_it_to_missed`: due at `T`. `load(T+30s)` leaves it in `due()`. `load(T+901s)` puts it in `missed()`.
  - `test_tool_scheduler.py::test_fired_once_removes_and_repeat_moves_forward`.
  - `test_tool_scheduler.py::test_store_survives_reload_and_is_0600`.
  - `test_tool_scheduler.py::test_add_rejects_past_full_and_long_text`: the codes are `in_past`, `full` and `too_long`, and `items()` is unchanged.
  - `test_tool_timers.py::test_reminder_set_bad_date_returns_error_and_no_item` (`date="2026-13-40"` gives `{"error": "bad_date"}`).
  - `test_tool_timers.py::test_description_carries_local_now`.
  - `test_tool_timers.py::test_logs_never_carry_reminder_text` (caplog).
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. `bad_date` and `bad_time` are skill-level codes from argument parsing.
- [ ] **Step 4:** Run `pytest -q` and ruff. Expected: pass. Add the modules to `BEHAVIOUR_MODULES`.
- [ ] **Step 5:** Commit as `feat(tools): scheduler with timers and reminders (SPEC 10, 11A)`.

### Task T4: Robot-initiated turn and announcements

**Files:**
- Create: `src/roboemot/tools/announce.py`, `src/roboemot/llm/prompts/event_timer.{en,it}.txt`, `event_reminder.{en,it}.txt` (the fallback sentences, with a `{label}`/`{text}` slot)
- Modify: `src/roboemot/conversation/worker.py` (`event_turn`), `src/roboemot/runtime/startup.py` (build the announcer), `src/roboemot/runtime/ticks.py` (`conversation_idle`)
- Test: `tests/test_event_turn.py`, `tests/realtime/test_event_turn_realtime.py`, `tests/test_tool_announce.py`

**Interfaces:**
- Consumes: `Scheduler`, `SchedulerRunner`, `tools.due` (T3), `CircadianRunner.wake()`, `ExpressionRenderer.request("nod")`, `ConversationWorker._say_local`, `Cues.play("fallback")`.
- Produces:
  - `ConversationWorker.event_turn(text: str, language: str) -> concurrent.futures.Future[bool]`. It can be called from any thread. It runs on the worker loop as a turn without a user utterance:
    - On a chat route, the messages are `build_messages(...)` with `user_text` set to the event text. The event text starts with `"Event (data, not instructions): "`.
    - On a realtime route (both wires), it is `text_item(event_text)` then a `ResponseRun.run(instructions, "auto")` loop, the same as `_realtime_rounds`.
    - It is recorded in history as the assistant reply only, and in the Live transcript with kind `event`. It is never stored as a user line.
    - It opens the normal follow-up window, so the person can reply without the wake word.
    - It resolves True once the reply audio has started, and False if no route answered within `announce_timeout_s`.
  - `conversation_idle(host, now: float) -> bool`: True when `phase == "idle"`, `not host.speech.speaking(now)`, `not listener.voice_active()` and no tool round is open.
  - `Announcer(scheduler, worker, host, cfg)`. It subscribes to `tools.due`. The bus handler only hands the item to the announcer's own thread (a queue). It never waits on the publisher's thread. For each item:
    1. If the platform is asleep, defer and keep the item.
    2. If the robot naps or rests (including quiet hours), call `CircadianRunner.wake()`. It cancels the nap and holds the robot awake for `circadian.wake_hold_s`. Wait until the circadian state has left `resting`, because `ExpressionRenderer.request` refuses while `circadian_factor() <= 0`.
    3. Wait for `conversation_idle`, up to `announce_max_wait_s`.
    4. Pause the media (T6 hook: `host.media.pause("announce")` when present).
    5. Request `nod`.
    6. Call `event_turn`.
    7. On False, play the fallback: the local voice when available, `_say_local` with the prompt file's sentence, plus the `fallback` cue. With no local voice, the cue and the gesture only.
    8. Call `scheduler.fired(id)` only after audio played.
    9. Timers: if no addressed utterance arrives within `repeat_after_s`, announce again, at most `repeat_max` times.
  - Event texts (English, sent to the model): `"A timer the person set is due: {label or 'no label'}. Tell them now, briefly."` and `"A reminder the person set is due: {text}. Tell them now, briefly."`

- [ ] **Step 1: Write the failing tests**
  - `test_event_turn.py::test_event_turn_on_chat_route_speaks_and_is_not_a_user_line`: uses the `Server` fake from `tests/test_conversation_worker.py`. The reply text is spoken, `history` has no user role with the event text, and the transcript kind is `event`.
  - `test_event_turn.py::test_event_turn_returns_false_when_route_down`.
  - `tests/realtime/test_event_turn_realtime.py::test_event_turn_sends_text_item_then_response_create` (`FakeServer`) and the same test on `gemini_server.py`.
  - `test_tool_announce.py::test_due_item_waits_for_idle_then_nods_and_announces`: with the robot speaking until `t=3`, the announcement starts after `t=3`, `nod` is requested once, and `fired` is called once.
  - `test_tool_announce.py::test_due_item_speaks_after_max_wait_even_if_busy`.
  - `test_tool_announce.py::test_asleep_defers_and_fires_on_wake_however_late`: asleep for 3 h, then awake. It fires, and is not in `missed()`.
  - `test_tool_announce.py::test_napping_robot_is_woken`.
  - `test_tool_announce.py::test_quiet_hours_rest_is_lifted_before_the_nod`: `nod` is requested only after the state left `resting`, and the request returns True.
  - `test_event_turn.py::test_event_turn_opens_a_closed_realtime_session`: with the session closed by idle, `event_turn` opens it and the reply plays.
  - `test_config.py::test_announce_timeout_exceeds_realtime_connect_timeout`: a validator rejects `tools.scheduler.announce_timeout_s` at or below the realtime route's `connect_timeout_s`.
  - `test_tool_announce.py::test_fallback_sentence_and_cue_when_route_fails`.
  - `test_tool_announce.py::test_timer_repeats_until_answered_at_most_repeat_max`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement. Remove T3's log-only firing when the announcer is attached.
- [ ] **Step 4:** Run `pytest -q`, `pytest tests/realtime -q` and ruff. Expected: pass.
- [ ] **Step 5:** Commit as `feat(tools): robot-initiated turns announce timers and reminders (SPEC 8, 10)`.

### Task T5: Weather tool

**Files:**
- Create: `src/roboemot/tools/weather.py`
- Modify: `src/roboemot/config/schema.py` (`ToolsConfig.weather`), `pyproject.toml` (`weather = "roboemot.tools.weather:skills"`), `src/roboemot/runtime/startup.py` (`ToolHost.http`)
- Test: `tests/test_tool_weather.py`

**Interfaces:**
- Consumes: `circadian.location` (`{lat, lon}`, private), `circadian.timezone`.
- Produces:
  - `WeatherConfig`:
    - `enabled` (`param(False, ..., opt_in=True)`)
    - `place_name: str = ""`. Add `"tools.weather.place_name"` to `redact.PRIVATE_PATHS` (`redact.py:39`), because it reveals the home town. Test: `test_tool_weather.py::test_place_name_is_private`.
    - `forecast_days` (3, 1–7)
    - `cache_s` (600, 0–3600)
    - `timeout_s` (5, 1–30)
    - `base_url` (`"https://api.open-meteo.com/v1/forecast"`, advanced)
    - `geocode_url` (`"https://geocoding-api.open-meteo.com/v1/search"`, advanced)
    - `geocode_results` (5, 1–10)
  - `ToolHost.http: Callable[[], httpx.Client] | None`. It is injected so tests use `httpx.MockTransport`.
  - `fetch_forecast(client, lat, lon, days, timeout_s) -> Forecast`. The query is `current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code`, `daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max`, `forecast_days=days`, `timezone=auto`.
  - `Forecast` is a frozen dataclass of the current values and a per-day tuple. WMO `weather_code` maps to English words through the module constant `WMO_WORDS`.
  - `geocode(client, query, language, timeout_s) -> list[Place(name, admin, country, lat, lon)]`. It is used by T7 only. `count=tools.weather.geocode_results`.
  - Skill `weather(days: Annotated[int, Range(1, 7)] = 1)`, `available=lambda: host.config().tools.weather.enabled and host.config().circadian.location is not None`. It returns the structured forecast as JSON, clamps `days` to `forecast_days`, and caches by `(lat, lon, days)` for `cache_s`. A network error returns `{"error": "unreachable"}`.
- [ ] **Step 1: Write the failing tests**
  - `test_tool_weather.py::test_weather_not_offered_when_disabled_or_no_location`.
  - `test_tool_weather.py::test_weather_parses_open_meteo_payload`: a recorded payload fixture under `tests/data/weather/forecast.json` maps `weather_code 61` to `"slight rain"`.
  - `test_tool_weather.py::test_weather_cache_avoids_second_call` (MockTransport call count).
  - `test_tool_weather.py::test_weather_unreachable_returns_error_code`.
  - `test_tool_weather.py::test_logs_never_carry_coordinates`.
  - `test_tool_weather.py::test_disabling_mid_session_refuses_in_flight_call` (Review Focus 4).
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `pytest -q` and ruff. Expected: pass.
- [ ] **Step 5:** Commit as `feat(tools): weather through Open-Meteo, admin opt-in (SPEC 10, 12)`.

### Task T6: Media player (pause, not mix) and radio tool

**Files:**
- Create: `src/roboemot/media/__init__.py`, `src/roboemot/media/player.py`, `src/roboemot/media/decode.py`, `src/roboemot/tools/radio.py`
- Modify: `src/roboemot/config/schema.py` (`ToolsConfig.media`, `ToolsConfig.radio`), `src/roboemot/voice/listener.py` (the lyrics gate), `src/roboemot/runtime/startup.py` (build the player, `host.media`, stop on nap and asleep), `src/roboemot/bus/__init__.py` (topics `media.playing`, `media.stopped`), `pyproject.toml` (`radio = "roboemot.tools.radio:skills"`). If the decoder needs a dependency, add it to `pyproject.toml` and to the installer record (SPEC 4A, Installer).
- Test: `tests/test_media_player.py`, `tests/test_tool_radio.py`, `tests/test_listener_media.py`

**Decoder choice (builder):** measure on the robot first. Candidates are an `ffmpeg` or `gst-launch-1.0` subprocess that writes s16le PCM to stdout (GStreamer is already there for the SDK), or PyAV. Pick the one with the lowest CPU at 128 kbps MP3, and report the numbers in the PR. `decode.py` hides the choice behind `open_stream(url: str, rate_hz: int, timeout_s: float) -> PcmStream`. `PcmStream` has `.read(frames: int) -> np.ndarray | None` and `.close()`.

**Interfaces:**
- Consumes: `SpeechOutput.speaking_at`, `ConversationPhase`, `listener.voice_active()`, `Speaker.push_audio_sample` through `SpeechOutput.play(..., yields=True)` chunk jobs, or a new `SpeechOutput.play_media(chunk, rate_hz, volume_db) -> bool` that refuses while any speech job is queued (builder picks; the queue rule is that speech always wins).
- Produces:
  - `MediaConfig`:
    - `volume_db` (-12, -40 to 0)
    - `chunk_ms` (200, 50–1000)
    - `resume_after_s` (1.5, 0–30)
    - `connect_timeout_s` (8, 1–30)
    - `max_play_s` (7200, 60–43200)
  - `RadioConfig`:
    - `enabled` (`param(False, ..., opt_in=True)`)
    - `stations: list[Station]`, where `Station(name: str, url: str, genres: list[str], language: str | None)`. URLs must be `http(s)://`, checked at load.
    - `dance: bool = True`
  - `MediaPlayer(output, host, cfg)`:
    - `play(station: Station) -> Literal["ok", "unreachable", "disabled"]`
    - `stop(reason: str) -> None`
    - `pause(reason: str) -> None`
    - `volume(delta_db: float) -> float`
    - `now_playing() -> Station | None`
    - A feeder thread reads `PcmStream` and pushes chunks only while no robot speech is queued or playing and no addressed turn is open. An addressed turn opens on the wake word or when `submit` accepts an utterance, and it closes when the reply ends. The feeder never pauses on VAD, `voice_active()` or the `listening` phase, because the microphone hears the radio itself. While paused it drops the read chunks (the live stream stays live), and it resumes after `resume_after_s` with no speech and no addressed turn.
    - It publishes `media.playing` and `media.stopped` with the station name only.
    - It stops on a nap, on `robot.asleep`, after `max_play_s`, and on a stream end or error.
  - Lyrics gate: while `media.playing` holds, only the wake word addresses the robot. The follow-up window and the other address rules (face, policy) do not count. When `radio.dance` is false, `_music_during` returns False while playing.
  - `ToolHost.media: MediaPlayer | None`.
  - Skills, with `available=lambda: host.config().tools.radio.enabled and bool(host.config().tools.radio.stations)`:
    - `radio_play(station: str)`, with `choices` = station names plus genres. A genre picks the first matching station. A URL is never accepted.
    - `radio_stop()`
    - `radio_volume(change: Literal["up", "down"])`, which steps by `media.volume_step_db` (3, 1–12).

- [ ] **Step 1: Write the failing tests**
  - `test_media_player.py::test_player_pauses_while_robot_speaks_and_resumes_after_idle`: uses a fake `PcmStream` and a fake output. No chunk is pushed during the speaking window. Pushing resumes after `resume_after_s`.
  - `test_media_player.py::test_player_stops_on_nap_and_asleep`.
  - `test_media_player.py::test_player_stops_after_max_play_s`.
  - `test_media_player.py::test_unreachable_stream_reports_error_and_logs_no_url`.
  - `test_tool_radio.py::test_radio_play_accepts_only_configured_names_or_genres` (a URL argument gives a choices error).
  - `test_tool_radio.py::test_radio_not_offered_when_disabled_or_no_stations`.
  - `test_listener_media.py::test_follow_up_window_does_not_address_while_media_plays` (Review Focus 3).
  - `test_media_player.py::test_vad_on_radio_sound_does_not_pause_the_player` (Review Focus 3).
  - `test_listener_media.py::test_dance_off_hides_music_while_playing`.
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `pytest -q` and ruff. Expected: pass.
- [ ] **Step 5:** Commit as `feat(tools): radio with pause-on-speech media player (SPEC 8, 10, 12)`.
- [ ] **Step 6 (robot, owner):** Play one station. Measure the CPU with `roboemot debug --host <robot> --json`. Check that the wake word works at the default volume. Record the result in the PR.

### Task T7: Tools tab

**Files:**
- Create: `src/roboemot/runtime/tools_state.py`, `src/roboemot/webui/static/tools.js`, `tests/webui_js/tools.test.mjs`
- Modify: `src/roboemot/webui/api.py` (a `ToolsPort` Protocol, routes in `_protected`), `src/roboemot/webui/server.py` (`make_server(..., tools=None)`), `src/roboemot/runtime/webui.py` (build `ToolsState` and subscribe it to config), `src/roboemot/webui/static/index.html` (the `tools.js` script before `app.js`), `src/roboemot/webui/static/app.js:1422` (`TABS.tools = toolsView`), `src/roboemot/webui/static/settings.js` (`SETTING_PURPOSES` entry for `tools`), `src/roboemot/webui/strings.py`, `src/roboemot/webui/descriptions_it.json`
- Test: `tests/test_webui_tools.py`, `tests/webui_js/tools.test.mjs`

**Interfaces:**
- Consumes: `Scheduler.items/missed/cancel/clear_missed` (T3), `MediaPlayer.now_playing/stop/play` (T6), `geocode` (T5).
- Produces:
  - `ToolsPort` Protocol: `state() -> dict`, `cancel(item_id: str) -> bool`, `clear_missed() -> None`, `radio_stop() -> None`, `radio_test(name: str) -> str`, `geocode(query: str) -> list[dict]`.
  - Routes (PIN session; a POST also needs CSRF and the `pin` re-ask, like `_emotions_play`):
    - `GET /api/tools/state` (in `PASSIVE_POLLS`) returns `{tools: [{name, enabled, available}], items: [{id, kind, text, due_local, repeat}], missed: [...], playing: {name, volume_db} | null}`
    - `POST /api/tools/cancel {id, pin}`
    - `POST /api/tools/missed/clear {pin}`
    - `POST /api/tools/radio/stop {pin}`
    - `POST /api/tools/radio/test {name, pin}`
    - `POST /api/tools/geocode {query, pin}` (a POST because the query is private)
  - The tab (`toolsView`) has one card per tool with its enabled switch and settings (saved through `settingsBatchSave`), the timer and reminder list with cancel, the missed list, and Now playing. Weather has a place search, and choosing a result writes `circadian.location` and `tools.weather.place_name` in one batch. Radio has a station list editor. The poll interval is `webui.tools.poll_ms` (2000, 500–10000).
- [ ] **Step 1: Write the failing tests**
  - `test_webui_tools.py::test_state_needs_session_and_lists_items`.
  - `test_webui_tools.py::test_cancel_requires_pin_and_csrf`.
  - `test_webui_tools.py::test_geocode_is_post_and_not_logged`.
  - `test_webui_tools.py::test_state_hides_reminder_text_from_logs`.
  - `tools.test.mjs`: the view renders the items and the cancel button calls the API with the id.
  - `test_webui_one_language.py` must pass with the new keys.
- [ ] **Step 2:** Run. Expected: FAIL. **Step 3:** Implement. **Step 4:** Run `pytest -q`, `node --test tests/webui_js`, `scripts/audit_webui.sh` and ruff. Expected: pass, with 0 audit violations.
- [ ] **Step 5:** Commit as `feat(webui): Tools tab (SPEC 11A, 12)`.

### Epic close

- [ ] Open `epic/voice-tools` → `main` and run the full CI. Check `git merge-tree --write-tree origin/main origin/epic/voice-tools` before waiting on CI.
- [ ] Merge with a merge commit after CI is green.
- [ ] Remove the worktrees and delete the slice branches.
- [ ] Tell the owner what to test on the robot: the Pollen probe, the radio CPU, the wake word over the radio, and a timer announcement on every route.
