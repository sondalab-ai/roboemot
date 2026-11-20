# Robot Expression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When something happens around the robot, it reacts in its own words, written by the LLM. Four situations are covered: it did not catch what someone said, an action failed (a radio station dropped), someone shows it an object, and a known person comes back. Two hard rules apply: the robot never talks over a conversation, and it never touches the music.

**Architecture:** A new runtime part, the `Expressor` (`src/roboemot/runtime/expressor.py`), works beside the announcer.
- Triggers call `Expressor.offer(...)` (or one of its thin wrappers). The call is synchronous and cheap: it checks the switch, the cooldown, the music and the rest, then queues the request.
- A single worker thread takes one request at a time. It drops the request when stale or when the robot rests. While music plays it only shows a gesture. Otherwise it waits for an idle conversation, then starts a robot-initiated turn through a new `ConversationWorker.reaction_turn`. That turn re-checks idle on the worker's loop and never jumps ahead of a person.
- Offline, it says a fixed line from `prompts/expression_<situation>.<lang>.txt` with the local voice, or else shows the situation's cue.
- It never calls `media.pause`/`stop`/`volume` and never calls `CircadianRunner.wake`.

**Tech Stack:** Python 3.11, pydantic v2 config (`param(...)`), pytest (`asyncio_mode=auto`), the existing chat `Server` fake (`tests/test_conversation_worker.py`) and realtime `FakeServer` (`tests/realtime/fake_server.py`), vanilla JS Live tab (`webui/static/live.js`).

**Spec:** `docs/specs/2026-11-20-robot-expression-design.md` (the approved design; this plan argues from it) and `docs/specs/kickoff.md` (SPEC; Task 1.1 adds the text).

Paths are relative to `packages/roboemot-core/` unless they start with `docs/` or `configs/`.

## Global Constraints

- No magic numbers (SPEC 11A). Every number is a `param(...)` field with a default, a unit and a range, and has an Italian description in `src/roboemot/webui/descriptions_it.json`. `runtime/expressor.py` goes into `BEHAVIOUR_MODULES` in `tests/test_no_magic_numbers.py`.
- Core stays robot-agnostic (SPEC 4A, 12A). Gestures are named only through the manifest `gesture_map`: a per-situation `gesture` setting whose value is a gesture name, played with `ExpressionRenderer.request`. `tests/test_robot_hierarchy.py` stays green, and its baseline does not grow.
- Privacy (SPEC 12):
  - No heard text and no audio goes into logs, the bus, Live events or the event text.
  - Event texts are built from fixed English templates plus data only: a detector category, a station name, an enrolled first name and a wake flag.
  - Logs carry the situation, the outcome and counts. They never carry a name, a station or a category.
- Hard rule 1, verbatim from the design: "**Never interrupt a conversation.** A reaction speaks only when the conversation is idle (`conversation_idle`): no turn running, no one speaking, no answer playing."
- Hard rule 2, verbatim from the design: "**Never touch the music.** While the robot's radio plays (audible, or held by its own player), a reaction never pauses, lowers or stops it. It shows **only a gesture** and the line is never said later (owner choice "Gesture only"). This differs from the timer and reminder announcer, which pauses the radio."
- Every commit that implements or changes behaviour cites its SPEC sections in the subject, for example `(SPEC 8, 11A)`.
- Every commit ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Branch and PR (owner change, 2026-11-20): one branch, `feat/robot-expression`, off `main`. Each task is its own commit on that branch, in the order below. One PR into `main` at the end. There is no integration branch and there are no per-part PRs.
- A worktree agent needs its own venv. Check that `python -c "import roboemot; print(roboemot.__file__)"` points inside the worktree before trusting a test run (CLAUDE.md).
- Validation per task: `ruff check . && ruff format --check .`, the focused tests named in the task, then `pytest -q` (core and reachymini). A task that touches `live.js` also runs `node --test tests/webui_js`.

## Review Focus

1. **Music starts while a reaction waits.** A shown-object reaction waits for a turn to end. Meanwhile the person asks for the radio, and the station starts before the turn ends. Expected: the reaction is never spoken, and `media.pause`, `media.stop` and `media.volume` are never called. It ends as `gesture_only_music`. Pinned in Task 1.3 by `test_music_starting_during_the_wait_turns_it_gesture_only`.
2. **A conversation starts between the idle check and the turn start.** The expressor saw `conversation_idle` true and called `reaction_turn`. A person's utterance is submitted before the loop takes the reaction. Expected: the reaction returns `"busy"`, the person's turn runs first, and the reaction waits again within its max age. Pinned in Task 1.2 by `test_reaction_turn_refuses_when_an_utterance_is_pending` and in Task 1.3 by `test_busy_result_goes_back_to_waiting`.
3. **The robot falls asleep with a reaction queued.** A request waits for idle, then `robot.asleep` arrives or the quiet hours begin. Expected: it is dropped as `dropped_asleep`, nothing is said, and `CircadianRunner.wake` is never called. Pinned in Task 1.3 by `test_asleep_or_quiet_hours_while_waiting_drops_it`.
4. **A burst of `object.shown`.** Five different things are shown within 2 s while a turn runs. Expected: at most one shown-object request is queued (a newer one replaces the older), a request for a look that ended is dropped as stale, and at most one comment is spoken, about the thing still shown. Pinned in Task 3.2 by `test_burst_of_shown_objects_speaks_at_most_once_about_the_last`.
5. **Returning person: track churn and restart.**
   - A known person sits in view for twice `greet_absence_s`. Their face track drops and comes back every 20 s, so `person.recognized` fires again each time. Expected: no greeting.
   - After an app restart, the first sighting of a person is not a return.
   - Pinned in Task 4.1 by `test_track_churn_in_view_never_greets` and `test_first_sighting_after_restart_does_not_greet`.

---

## Part 1: Expressor core

**Location choice.**
- The announcer lives in `tools/announce.py` because it serves a voice tool: it consumes the scheduler's `tools.due`.
- The expressor is not a tool. It listens to perception, media, circadian and voice events, like `runtime/radio_stop.py` and `runtime/sleep.py`, and the runtime builds and wires it. So it goes in `runtime/expressor.py`.
- It shares the announcer's idle wait through `runtime/ticks.py`, which `tools/announce.py` already imports.
- It shares the turn start through `ConversationWorker`. It does not import the announcer.

### Task 1.1: SPEC text and the `expression.reactions` settings

**Files:**
- Modify: `docs/specs/kickoff.md`:
  - Add a new SPEC 8 subsection "Robot expression" after "Conversation state". It covers the four situations, the two hard rules, the per-request steps, the outcomes and the settings.
  - Amend "Non-verbal cues" (the spoken question takes the `not_heard` sound's place, gesture kept).
  - Amend SPEC 11A (the `expression.reactions` section).
  - Amend SPEC 12 (event texts from templates plus data).
  - Every number appears as "the default of `expression.reactions.…`". Owner decisions keep their date (2026-11-20).
- Modify: `src/roboemot/config/schema.py` (`SituationConfig`, `ExpressionReactionsConfig`, `ExpressionConfig.reactions`), `src/roboemot/webui/descriptions_it.json`
- Test: `tests/test_expression_config.py` (new)

**Interfaces:**
- Produces:
  - `Situation = Literal["didnt_catch", "failed_action", "shown_object", "returning_person"]` and `SITUATIONS: tuple[str, ...] = get_args(Situation)` in `config/schema.py`.
  - `SituationConfig(Strict)`:
    - `enabled: bool = param(True, ...)`
    - `max_age_s: float = param(..., unit="s", ge=1, le=600)`
    - `cooldown_s: float = param(..., unit="s", ge=0, le=86400)`
    - `gesture: str = param("", ...)`: a manifest `gesture_map` name played before the line, and alone while music plays. Empty means none.
  - `ExpressionReactionsConfig(Strict)` on `ExpressionConfig.reactions`:
    - `enabled: bool = param(True, ...)`
    - `model_asks: bool = param(True, ...)`
    - `turn_timeout_s: float = param(5.0, ..., unit="s", ge=1, le=30)`
    - `greet_absence_s: float = param(1800.0, ..., unit="s", ge=60, le=86400)`
    - `didnt_catch: SituationConfig = param({"max_age_s": 10, "cooldown_s": 15}, ...)`
    - `failed_action: SituationConfig = param({"max_age_s": 30, "cooldown_s": 120, "gesture": "puzzled"}, ...)`
    - `shown_object: SituationConfig = param({"max_age_s": 60, "cooldown_s": 120}, ...)`: the description says "and only while the thing is still shown".
    - `returning_person: SituationConfig = param({"max_age_s": 30, "cooldown_s": 0}, ...)`: per person, `greet_absence_s` is the limit.
    - `situation(name: Situation) -> SituationConfig`.

- [ ] **Step 1: Write the failing tests**
  - `test_expression_config.py::test_defaults_match_the_design`:
    ```python
    r = RoboEmotConfig().expression.reactions
    assert r.enabled and r.model_asks
    assert r.didnt_catch.max_age_s == 10 and r.failed_action.max_age_s == 30
    assert r.returning_person.max_age_s == 30 and r.greet_absence_s == 1800
    assert r.failed_action.gesture == "puzzled" and r.shown_object.gesture == ""
    ```
  - `test_expression_config.py::test_ranges_refuse_out_of_bounds`: `max_age_s=0`, `greet_absence_s=59` and `turn_timeout_s=31` each raise `ValidationError`.
  - `test_expression_config.py::test_existing_expression_fields_untouched`: `ExpressionConfig().max_amplitude` keeps its default.
  - `tests/test_webui_one_language.py` (existing) covers the Italian descriptions.
- [ ] **Step 2:** Run `pytest tests/test_expression_config.py tests/test_webui_one_language.py -q`. Expected: FAIL (no `reactions` field, missing IT keys).
- [ ] **Step 3:** Implement the models and the IT descriptions. Write the SPEC text.
- [ ] **Step 4:** Run the same tests plus `pytest -q`. Expected: PASS.
- [ ] **Step 5:** Commit as `feat(config): robot expression settings and SPEC text (SPEC 8, 11A, 12)`.

### Task 1.2: `ConversationWorker.reaction_turn`, a turn that never jumps ahead

`event_turn` puts its turn first in line (`_pending.appendleft`), so an utterance submitted after the expressor's idle check would wait behind the reaction. That would break hard rule 1. `reaction_turn` is queued the same way, but on the loop it re-checks before anything is sent, and it gives way.

**Files:**
- Modify: `src/roboemot/conversation/worker.py`:
  - `_EventTurn` gains `timeout_s: float | None = None`, `reaction: bool = False`, `profile: bool = True` and `quiet: Callable[[], bool] | None = None`.
  - `_watch_event_audio` uses `turn`'s `timeout_s` when set.
  - `_answer_event_turn` gets the refuse check.
  - `_converse` and `_rt_instructions` leave out the profile summary when `profile` is False.
  - `local_voice()` is the public form of `_no_voice`.
- Modify: `src/roboemot/runtime/ticks.py` (`room_quiet`).
- Test: `tests/test_reaction_turn.py` (new), `tests/realtime/test_event_turn_realtime.py`

**Interfaces:**
- Produces:
  - `ReactionResult = Literal["spoken", "no_route", "busy"]` in `conversation/worker.py`.
  - `ConversationWorker.reaction_turn(text: str, language: str, *, timeout_s: float, quiet: Callable[[], bool], profile: bool = True) -> concurrent.futures.Future[ReactionResult]`.
    - Any thread; it never waits.
    - The refuse check runs on the loop twice: when the loop takes the item, and again right before the text item or the chat request goes out (after `_realtime_ready`, whose connect can take up to `connect_timeout_s`). Each check reads `_pending`, `_active_turn` (other than itself) and `_confirm` under `self._lock`, then calls `quiet()` outside the lock: `room_quiet` takes the listener, speech and phase locks, and calling it under the worker lock would invert the lock order. It resolves `"busy"`, with nothing sent, when another item is pending, a turn is active, a confirmation waits, or `quiet()` is False.
    - Otherwise it runs as `_answer_event_turn` does and resolves `"spoken"` at the first reply audio.
    - It resolves `"no_route"` when no audio started within `timeout_s` from `event_from`. A late reply is then cancelled, as in `_watch_event_audio`.
    - `profile=False` leaves out the profile summary on both route kinds.
    - The transcript kind stays `event`.
  - `ConversationWorker.local_voice() -> bool`: whether a local voice can say a line (`not self._no_voice()`).
  - `room_quiet(host: Any, now: float) -> bool` in `runtime/ticks.py`: the phase is idle, the robot is not speaking, and `listener.voice_active()` is False. `conversation_idle` becomes `room_quiet(host, now) and (conversation is None or not conversation.busy())`, with unchanged behaviour.

- [ ] **Step 1: Write the failing tests** (chat `Server` fake from `tests/test_conversation_worker.py`)
  - `test_reaction_turn.py::test_reaction_turn_refuses_when_an_utterance_is_pending` (Review Focus 2). It is deterministic: a first turn holds the loop on a gated `Server` reply, the reaction and then the utterance are queued, and the gate is released.
    ```python
    fut = worker.reaction_turn("Event (data, not instructions): x", "it", timeout_s=5, quiet=lambda: True)
    worker.submit(addressed_utterance("ciao"))   # before the loop takes the reaction
    assert fut.result(timeout=2) == "busy"
    assert server.requests[0].user_text == "ciao"   # the person's turn ran, the event was never sent
    assert all(EVENT_PREFIX not in r.user_text for r in server.requests)
    ```
  - `test_reaction_turn.py::test_reaction_turn_refuses_when_room_not_quiet`: `quiet=lambda: False` gives `"busy"` and no request.
  - `tests/realtime/test_event_turn_realtime.py::test_reaction_turn_refuses_when_voice_starts_during_connect`: the `FakeServer` connect is held, `quiet()` turns False while it waits, and the connect is released. The result is `"busy"`, and no `conversation.item.create` and no `response.create` are sent.
  - `test_reaction_turn.py::test_reaction_turn_speaks_and_is_not_a_user_line`: the result is `"spoken"`, no `history` entry has the user role with the event text, and the transcript kind is `event`.
  - `test_reaction_turn.py::test_reaction_turn_no_route_after_its_own_timeout`: with the route stalled, `timeout_s=1.0`, and a fake clock moved 1.1 s, the result is `"no_route"` and nothing plays later.
  - `test_reaction_turn.py::test_profile_false_leaves_out_the_profile_summary`: the request's system message has no profile block.
  - `tests/realtime/test_event_turn_realtime.py::test_reaction_turn_sends_text_item_then_response_create`: on `FakeServer`, and the same on `gemini_server.py`.
  - `tests/test_tool_announce.py` (existing) stays green: `conversation_idle` is unchanged.
- [ ] **Step 2:** Run `pytest tests/test_reaction_turn.py tests/realtime/test_event_turn_realtime.py -q`. Expected: FAIL (no `reaction_turn`).
- [ ] **Step 3:** Implement the interfaces above.
- [ ] **Step 4:** Run the same tests, then `pytest tests/test_event_turn.py tests/test_tool_announce.py -q` and `pytest -q`. Expected: PASS.
- [ ] **Step 5:** Commit as `feat(conversation): reaction_turn gives way to a person (SPEC 8)`.

### Task 1.3: The expressor (queue, rules, outcomes, report, Live event, wiring)

**Files:**
- Create: `src/roboemot/runtime/expressor.py`.
- Create the prompt files `src/roboemot/llm/prompts/expression_didnt_catch.{en,it}.txt`, `expression_failed_action.{en,it}.txt`, `expression_shown_object.{en,it}.txt` and `expression_returning_person.{en,it}.txt`. These are the offline lines. The slots are the `data` keys, and the caller fills them, then collapses spaces, as `tools/announce.py` `fallback_text` does:
  - `didnt_catch`: «Sorry, I didn't catch that. Can you say it again?» / «Scusa, non ho capito. Puoi ripetere?»
  - `failed_action`: «The radio {station} stopped.» / «La radio {station} si è interrotta.»
  - `shown_object`: «Oh, what is that?» / «Oh, cos'è?»
  - `returning_person`: «Hello {who}!» / «Ciao {who}!». With no enrolled name, `{who}` is left empty, and the line becomes «Hello!» / «Ciao!» (the space before the punctuation is removed).
- Modify:
  - `src/roboemot/llm/prompt.py` (`expression_fallback`)
  - `src/roboemot/circadian/runner.py` (`quiet_now`)
  - `src/roboemot/tools/announce.py` (`Announcer.busy`)
  - `src/roboemot/runtime/host.py` (`expressor` field)
  - `src/roboemot/runtime/startup.py` (`start_expressor` after `start_announcer`)
  - `src/roboemot/runtime/diagnostics.py` (`"expression"` under `roboemot.voice`)
  - `src/roboemot/diag.py` (kind `expression`, field `situation`, `age_s`)
  - `src/roboemot/webui/strings.py` (EN and IT `live_ev_expression_<outcome>` and `live_code_<situation>`)
  - `tests/test_no_magic_numbers.py` (`BEHAVIOUR_MODULES`)
- Test: `tests/test_expressor.py` (new), `tests/test_diag_events.py`, `tests/test_webui_one_language.py`

**Interfaces:**
- Consumes:
  - `ConversationWorker.reaction_turn`, `local_line` and `local_voice` (Task 1.2), `room_quiet` and `conversation_idle`.
  - `ExpressionRenderer.request(name)`, `CueSink.play`.
  - `MediaPlayer.playing()` and `audible()` (read only).
  - `CircadianRunner.model.state` and `quiet_now()`.
  - `RuntimeHost.announcer`.
- Produces:
  - `@dataclass(frozen=True) ReactionRequest(situation: Situation, data: Mapping[str, str], created_at: float, key: str = "", token: str = "")`. `key` is the cooldown key: `""`, a station, a category or a person id. `token` is the shown look's key. Neither `key` nor `token` is ever sent.
  - `Outcome = Literal["spoken", "gesture_only_music", "dropped_stale", "dropped_busy", "dropped_asleep", "dropped_cooldown", "dropped_replaced", "offline_line", "cue", "offline_silent"]`. These are the design's seven plus three named here.
  - `EVENT_PREFIX` reused from `tools/announce.py` (`"Event (data, not instructions): "`). The templates are module constants:
    - `DIDNT_CATCH_EVENT = "Someone just spoke to you and you did not catch what they said. Ask them in one short sentence to say it again."`
    - `FAILED_ACTION_EVENT = "The radio station {station} you were playing stopped: it did not load or it dropped. Tell the person in one short sentence."`
    - `SHOWN_OBJECT_EVENT = "A person is showing you something that looks like a {category}. Comment on it or ask about it in one short sentence."`
    - `RETURNING_EVENT = "{who} just came back after being away. Greet them by name in one short sentence. Never mention anything from an earlier conversation, their preferences or what they told you before."`
    - `WAKE_EVENT = "You just woke up and {who} is here. Greet them for the time of day in one short sentence. Never mention anything from an earlier conversation, their preferences or what they told you before."`
    - `{who}` is the first name, or `"a person you know"`.
  - `event_text(request: ReactionRequest) -> str` (module function).
  - `Expressor(worker, host, cfg: Callable[[], RoboEmotConfig], *, online: Callable[[], bool], asleep: Callable[[], bool], clock: Callable[[], float] = time.monotonic, wait: Callable[[float], bool] | None = None)`:
    - `start()`, `attach()` and `close()`, as the announcer has. The thread is named `roboemot-expressor`.
    - `offer(request: ReactionRequest) -> bool`. It is synchronous, any thread, and never waits. In order:
      1. `reactions.enabled` and the situation switch are on.
      2. The robot is not asleep, resting or in the quiet hours, else `dropped_asleep`.
      3. `music()` is False, else the situation's gesture and `gesture_only_music`.
      4. The `(situation, key)` cooldown has passed, else `dropped_cooldown`. The cooldown starts at acceptance.
      5. The request is queued. The queue holds at most one request per situation, and a newer one replaces the older (`dropped_replaced`).

      It returns True when a spoken attempt is queued.
    - `handle(request: ReactionRequest) -> Outcome`. One request, on the expressor thread. Tests call it directly. The steps follow the design's "Per request" list:
      1. Stale (`clock() - created_at > max_age_s`, or `valid(request)` False) gives `dropped_stale`.
      2. Asleep, resting or quiet hours give `dropped_asleep`.
      3. `music()` gives the gesture and `gesture_only_music`.
      4. While not `idle()`, it waits and re-checks steps 1 to 3 on every poll. Past the max age after having waited, the result is `dropped_busy`.
      5. When `not online()`: with `worker.local_voice()`, `local_line(expression_fallback(...))` gives `offline_line`. Otherwise the situation's offline cue gives `cue` (`didnt_catch` uses `not_heard`, `failed_action` uses `fallback`). The other two situations give `offline_silent`.
      6. Otherwise it requests the situation's gesture (when set) and calls `reaction_turn(event_text(request), cfg.conversation.language, timeout_s=reactions.turn_timeout_s, quiet=lambda: room_quiet(host, clock()), profile=request.situation != "returning_person")`:
         - `"spoken"` gives `spoken`. Like the announcer, it then waits for the reply's audio to end (`worker.after_turn`, then speech no longer queued) and calls `host.voice_listener.open_follow_up()`, so the person can answer without the robot's name. `event_turn` does not open that window itself; only `tools/announce.py:299` does.
         - `"busy"` goes back to step 4.
         - `"no_route"` goes to step 5.
    - `music() -> bool`: `host.media is not None and (host.media.playing() or host.media.audible())`.
    - `idle() -> bool`: `conversation_idle(host, clock())`, and no announcement is being handled (`host.announcer is None or not host.announcer.busy()`).
    - `valid(request) -> bool`: True by default. Task 3.2 adds the shown-object check.
    - `snapshot() -> dict`: `{"enabled": bool, "queued": int, "counts": {situation: {outcome: int}}}`. The keys are codes only.
    - Each outcome publishes `self.events("expression", outcome, situation=request.situation, age_s=...)` and counts it.
  - `Announcer.busy() -> bool`: True while `handle` runs for an item or a repeat.
  - `CircadianRunner.quiet_now() -> bool`: `in_window(self.clock.hour(), self.cfg.circadian.quiet_hours)` under the lock. It is False while the rhythm is off.
  - `expression_fallback(situation: str, language: str, cfg: RoboEmotConfig) -> str` in `llm/prompt.py`. It reads `_prompt_files(f"expression_{situation}")`, falling back to the language, then `conversation.language`, then English.
  - `diag.REASONS["expression"]` holds the `Outcome` values. `diag.FIELDS["situation"]` holds `SITUATIONS` and `diag.FIELDS["age_s"]` is `NUMBER`.
  - `start_expressor(host, stack, p, cleanup) -> None` in `runtime/startup.py`:
    - `online=lambda: (rt is not None and rt.up()) or not p.hub.router.offline`
    - `asleep=lambda: p.sleep.asleep`
    - It is closed before the conversation. It sets `host.expressor`.

- [ ] **Step 1: Write the failing tests** (`tests/test_expressor.py`; fake worker, fake media, fake circadian runner, fake clock and an injected `wait`)
  - `test_spoken_when_idle_online_and_quiet`: the result is `spoken`. `reaction_turn` is called once with a text that starts with `EVENT_PREFIX`, `timeout_s=5.0` and `profile=True`.
  - `test_stale_request_dropped_at_its_max_age`: a `didnt_catch` request created at `t=0` and handled at `t=10.5` gives `dropped_stale`. A `failed_action` request handled at `t=29` is spoken, and one handled at `t=31` is `dropped_stale`.
  - `test_waits_for_idle_and_drops_busy_after_max_age`: the conversation is busy until `t=12`, so a `didnt_catch` request from `t=0` gives `dropped_busy` and `reaction_turn` is never called. Busy until `t=5` gives `spoken`, with `reaction_turn` called after `t=5`.
  - `test_music_at_offer_shows_gesture_only_and_never_touches_the_player`: with `media.playing()` True, `offer(failed_action)` returns False, and `expression.request("puzzled")` is called once. `media.pause`, `stop` and `volume` are never called (spy). No `reaction_turn`, no line is queued for later, and `counts["failed_action"]["gesture_only_music"] == 1`.
  - `test_music_starting_during_the_wait_turns_it_gesture_only` (Review Focus 1): the conversation is busy until `t=4`, and the media starts playing at `t=2`. The result is `gesture_only_music`, there is no `reaction_turn`, and the media spy recorded no calls.
  - `test_busy_result_goes_back_to_waiting` (Review Focus 2): the first `reaction_turn` returns `"busy"`, the conversation is busy again until `t=3`, and the second call returns `"spoken"`. The outcome is `spoken`, with two calls, both inside the max age.
  - `test_asleep_or_quiet_hours_while_waiting_drops_it` (Review Focus 3): three parametrised cases (asleep set at `t=1`, the runner's state `resting` at `t=1`, `quiet_now()` True at `t=1`). Each gives `dropped_asleep`, with no `reaction_turn`, no `local_line` and `runner.wake` never called.
  - `test_offline_with_local_voice_says_the_fixed_line`: with `online=False` and `local_voice()` True, `local_line` gets `expression_fallback("failed_action", "it", cfg)` with `{station}` filled. The outcome is `offline_line`.
  - `test_offline_without_voice_plays_the_situation_cue_or_nothing`: `didnt_catch` plays the `not_heard` cue and gives `cue`. `failed_action` plays `fallback` and gives `cue`. `shown_object` gives `offline_silent` with no cue.
  - `test_no_route_falls_back_like_offline`: `reaction_turn` returns `"no_route"` and `local_voice()` is True, so the outcome is `offline_line`.
  - `test_cooldown_per_situation_and_key`: two `failed_action` offers for station `A` 60 s apart (cooldown 120) give `dropped_cooldown` for the second. An offer for station `B` at 60 s is accepted.
  - `test_newer_request_replaces_queued_one_of_same_situation`: `counts[...]["dropped_replaced"] == 1`, and the queue length is 1.
  - `test_waits_while_an_announcement_is_handled`: while `host.announcer.busy()` is True there is no `reaction_turn`.
  - `test_spoken_reaction_opens_the_follow_up_window`: after `"spoken"` and the reply's end, `listener.open_follow_up` is called once. After `"no_route"` or `offline_line` it is not called.
  - `test_logs_and_events_carry_no_data` (caplog plus a captured `events`): the station name `"Radio Segreta"` and the category `"cup"` appear in no log record and in no event field.
  - `tests/test_diag_events.py::test_expression_event_is_cleaned`: `clean({"situation": "didnt_catch", "age_s": 1.234, "station": "x"})` gives `{"situation": "didnt_catch", "age_s": 1.23}`.
  - `tests/test_circadian_runner.py::test_quiet_now_follows_the_window`.
  - `tests/test_tool_announce.py::test_busy_while_handling_an_item`.
- [ ] **Step 2:** Run `pytest tests/test_expressor.py tests/test_diag_events.py tests/test_circadian_runner.py tests/test_tool_announce.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the interfaces above.
- [ ] **Step 4:** Run the same tests, `tests/test_webui_one_language.py`, `tests/test_no_magic_numbers.py` and `pytest -q`. Expected: PASS.
- [ ] **Step 5:** Commit as `feat(runtime): expressor, LLM-written reactions behind two hard rules (SPEC 8, 11A, 12, 12A)`.

---

## Part 2: Didn't catch it

Triggers:
- `no_transcript`: the audio left, and no transcript came (`listener.py:2103`).
- `decodes_lost`: an addressed utterance whose local decode heard nothing (`listener.py:1783`).
- `no_server_answer`: a streamed utterance no server answer took (`ConversationWorker._unanswered`, `worker.py:1539`).

Not triggers:
- a face wake that heard nothing (`no_audio`, `listener.py:1786`),
- a face wake that was not sent and not addressed (`listener.py:2130`),
- a barge-in segment the server never confirmed (`AUDIO_DROPPED`),
- the worker's `local_only` branch (`worker.py:943`), whose utterance already triggered at `no_transcript`, so one utterance asks once.

### Task 2.1: The spoken question replaces the `not_heard` sound

**Files:**
- Modify:
  - `src/roboemot/voice/cues.py`: `Cues.play`, `NoCues.play` and the `CueSink` protocol gain `sound: bool = True`.
  - `src/roboemot/diag.py`: `CUE_RESULTS` gains `"gesture_only"`.
  - `src/roboemot/voice/listener.py`: `on_not_heard` and `AUDIO_NOT_HEARD_SPOKEN`.
  - `src/roboemot/runtime/startup.py`: `audio_cues` and the wiring.
  - `src/roboemot/conversation/worker.py`: `on_not_heard` and `_unanswered`.
  - `src/roboemot/runtime/expressor.py`: `offer_not_heard`.
  - `src/roboemot/webui/strings.py`: `live_res_gesture_only`, EN and IT.
- Test: `tests/test_cues.py`, `tests/test_expressor_not_heard.py` (new), `tests/realtime/test_realtime_cue_paths.py`

**Interfaces:**
- Produces:
  - `Cues.play(kind: CueKind, reason: str = "", *, sound: bool = True, **detail: Any) -> bool`. With `sound=False` it shows the gesture alone, and the decision event's result is `gesture_only`.
  - `VoiceListener.on_not_heard: Callable[[str], bool]` (default `lambda why: False`). It is called with `"no_transcript"` or `"decodes_lost"` before the audio event, on every robot, with or without a realtime route, and only for an addressed utterance (`p.addressed is True`: the wake decision was taken). The `no_transcript` site is `listener.py:2103`, reached for streams too, so it gets this gate explicitly. The robot never asks a television to repeat itself (design non-goal). When it returns True, the listener emits `AUDIO_NOT_HEARD_SPOKEN = "not_heard_spoken"` instead of `AUDIO_NOT_HEARD`.
  - `audio_cues`: for `AUDIO_NOT_HEARD_SPOKEN` it calls `cues.listening_end(key)` and `cues.play("not_heard", "no_transcript", sound=False)`.
  - `ConversationWorker.on_not_heard: Callable[[str], bool]` (default False). `_unanswered` calls it with `"no_server_answer"` only when `utterance.addressed` is True (a stream the server chose not to answer may be unaddressed speech) and plays `cues.play("not_heard", "no_server_answer", sound=not taken)`.
  - `Expressor.offer_not_heard(why: str) -> bool`:
    - It is `offer(ReactionRequest("didnt_catch", {}, clock()))`, but only when `online()`.
    - Offline or with music it returns False, so the cue stays as today.
    - `why` is never sent or logged beyond its code.
  - `start_expressor` sets `listener.on_not_heard` and `conv.on_not_heard` to `expressor.offer_not_heard`.
  - On a robot without a realtime route, `_on_audio` is a no-op as today: the reaction may speak, and no cue is added there.

- [ ] **Step 1: Write the failing tests**
  - `test_cues.py::test_play_without_sound_shows_gesture_only`: no `output.play` call, one player, and the event result is `"gesture_only"`.
  - `test_expressor_not_heard.py::test_spoken_question_replaces_the_sound_keeps_the_gesture`: with the listener's `on_not_heard` returning True, `audio_cues` receives `"not_heard_spoken"`, the cue is played with `sound=False`, and one `didnt_catch` request is queued.
  - `test_expressor_not_heard.py::test_music_keeps_the_cue_as_today`: with `media.playing()` True, `offer_not_heard` returns False, `cues.play("not_heard", "no_transcript")` keeps its sound, and there is no request.
  - `test_expressor_not_heard.py::test_offline_keeps_the_cue_as_today`: `online=False` gives the same.
  - `test_expressor_not_heard.py::test_decodes_lost_offers_without_realtime_route`: a listener with no `bind_realtime`, fed an addressed utterance whose decode is empty, calls `on_not_heard("decodes_lost")` once.
  - `test_expressor_not_heard.py::test_face_wake_without_audio_does_not_offer`: a `no_audio` face wake that heard nothing makes no `on_not_heard` call.
  - `test_expressor_not_heard.py::test_unaddressed_stream_that_times_out_offers_nothing`: a stream with `p.addressed` not True reaches the `no_transcript` site, so there is no `on_not_heard` call and the cue is as today. The same holds for `_unanswered` with `addressed=False`.
  - `test_expressor_not_heard.py::test_question_opens_the_follow_up_window`: after the spoken question, a final without the wake word is submitted to the worker.
  - `test_expressor_not_heard.py::test_one_utterance_asks_once`: a transcript timeout followed by the worker's `local_only` turn makes exactly one `on_not_heard` call.
  - `tests/realtime/test_realtime_cue_paths.py::test_unanswered_stream_offers_and_drops_sound`: on `FakeServer`, with no server answer within `server_answer_wait_s`, `on_not_heard("no_server_answer")` is called and the cue gets `sound=False`.
- [ ] **Step 2:** Run `pytest tests/test_cues.py tests/test_expressor_not_heard.py tests/realtime/test_realtime_cue_paths.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the interfaces above. Amend SPEC 8 "Non-verbal cues" ("Where each shows": the spoken question takes the sound's place).
- [ ] **Step 4:** Run the same tests, `pytest tests/realtime -q` and `pytest -q`. Expected: PASS.
- [ ] **Step 5:** Commit as `feat(voice): ask again instead of the did-not-get-it sound (SPEC 8)`.

---

## Part 3: Failed action and shown object

### Task 3.1: A station that drops is said, the player untouched

Failed actions are `media.stopped` with reason `dropped` (stream end or stall: a silent station stalls and is stopped after `tools.media.connect_timeout_s`) or `error` (a feeder failure). These reasons are excluded: `replaced`, `voice_stop`, `fist_stop`, `webui`, `asleep`, `nap`, `disabled`, `removed`, `max_play` and `closed`. A `radio_play` that returns `unreachable` happens inside a live answer, which the model already explains (out of scope, per the design).

**Files:**
- Modify: `src/roboemot/runtime/expressor.py` (subscribe `media.stopped`, `FAILED_STOPS = ("dropped", "error")`)
- Test: `tests/test_expressor_failed_action.py` (new)

**Interfaces:**
- Consumes: `media.stopped` (`{"station", "reason"}`, `media/player.py:256`).
- Produces: `Expressor._on_media_stopped(msg: Message) -> None`. For a reason in `FAILED_STOPS` it calls `offer(ReactionRequest("failed_action", {"station": name}, clock(), key=name))`. The bus thread only queues.

- [ ] **Step 1: Write the failing tests**
  - `test_dropped_station_is_said_and_player_never_called`: publish `media.stopped {"station": "Radio Uno", "reason": "dropped"}`, after which `media.playing()` is False. The outcome is `spoken`, and the event text contains `"Radio Uno"`. The `MediaPlayer` spy has no `pause`, `stop`, `play` or `volume` calls.
  - `test_failed_station_is_not_music`: with `playing()` and `audible()` False after the drop, the outcome is not `gesture_only_music`.
  - `test_user_stops_do_not_count`: each excluded reason (parametrised) queues nothing.
  - `test_another_station_playing_by_then_shows_gesture_only`: after the drop, a new `play` makes `playing()` True before `handle`. The outcome is `gesture_only_music` with the gesture `"puzzled"`.
  - `test_station_cooldown`: two drops of `"Radio Uno"` within 120 s give one spoken line and one `dropped_cooldown`.
- [ ] **Step 2:** Run `pytest tests/test_expressor_failed_action.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same tests and `pytest -q`. Expected: PASS.
- [ ] **Step 5:** Commit as `feat(runtime): say when a radio station drops (SPEC 8, 11)`.

### Task 3.2: Shown object, kept while still shown

**Files:**
- Modify: `src/roboemot/runtime/expressor.py` (subscribe `attention.target`, `_shown_key`, `valid`)
- Test: `tests/test_expressor_shown.py` (new)

**Interfaces:**
- Consumes: `attention.target` (`AttentionTarget`, `priority == SHOWN`, `.key`, `.category`), the same new-look rule as `AppraisalAdapter._on_attention` (`affect/appraisal.py:165`). The appraisal adapter's `object.shown` reaction is unchanged.
- Produces:
  - `Expressor._on_attention(msg: Message) -> None`. On a new look it calls `offer(ReactionRequest("shown_object", {"category": target.category}, clock(), key=target.category, token=target.key))`. Every target message updates `_shown_key` (None when the target is not `SHOWN`).
  - `Expressor.valid(request)`: for `shown_object`, `request.token == self._shown_key`. It is checked at step 1, on every wait poll and right before `reaction_turn`.

- [ ] **Step 1: Write the failing tests**
  - `test_shown_object_spoken_with_category`: the event text contains `"cup"` and no entry id.
  - `test_look_ended_before_idle_drops_stale`: the conversation is busy until `t=3`, and the target changes to a person at `t=2`. The outcome is `dropped_stale`.
  - `test_burst_of_shown_objects_speaks_at_most_once_about_the_last` (Review Focus 4): five `SHOWN` targets with keys `e1` to `e5` and categories `cup`, `book`, `cat`, `cup`, `phone`, at `t=0` to `t=2`, while busy until `t=4`, with `e5` still shown. The results are one `reaction_turn`, whose text contains `"phone"`. The fourth offer (`cup` again within 120 s of the first acceptance) is `dropped_cooldown`, and `book`, `cat` and `phone` each replaced the queued one, so `dropped_replaced == 3` and `dropped_cooldown == 1`: five offers in total.
  - `test_same_category_once_per_cooldown`: `cup` shown at `t=0` and `t=60` gives one spoken line.
  - `test_existing_reaction_is_whole_response_under_music`: with music on, the object is shown: no gesture requested by the expressor (`shown_object.gesture == ""`), `gesture_only_music == 1`.
- [ ] **Step 2:** Run `pytest tests/test_expressor_shown.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the same tests and `pytest -q`. Expected: PASS.
- [ ] **Step 5:** Commit as `feat(runtime): comment on a shown object while it is shown (SPEC 5A, 7, 8)`.

---

## Part 4: Returning person

### Task 4.1: Greet a return, never a churn or a restart

Rules:
- Presence is measured from every `percept.face` with an `owner_id`, not from `person.recognized`, because a new track fires `person.recognized` again.
- Sightings while the robot is asleep, resting or in the quiet hours do not count: the robot is not awake.
- A wake is either `robot.awake` after `robot.asleep`, or `circadian.state` leaving `resting` whose reason was `quiet_hours` or `sleepiness` ("the night"). A day `nap` or a `hold` is not a wake. The absence rule covers them.
- Memory is per person and in RAM only.

**Files:**
- Modify:
  - `src/roboemot/runtime/expressor.py` (`ReturnTracker`; subscribe `percept.face`, `robot.asleep`, `robot.awake`, `circadian.state`)
  - `src/roboemot/runtime/startup.py` (pass `registry` to the expressor)
- Test: `tests/test_expressor_returning.py` (new), `tests/test_reaction_turn.py`

**Interfaces:**
- Consumes:
  - `FacePercept.owner_id`
  - `CircadianState.state` and `.reason`
  - `person_name(registry, person_id) -> str | None` (`conversation/sources.py:46`): only active records, the same name the profile summary uses
  - `ConversationWorker.in_session()`
- Produces:
  - `ReturnTracker(absence_s: Callable[[], float])`. Pure, with no locks of its own (the expressor holds its lock).
    - `woke(now: float) -> None` records `last_wake = now`.
    - `seen(person_id: str, now: float, awake: bool) -> Literal["none", "return", "wake"]`:
      - When `awake` is False it returns `"none"` and records nothing.
      - Otherwise, with `prior = last_seen.get(person_id)`:
        - `"wake"` when `last_wake` is set and (`prior is None` or `prior < last_wake`);
        - `"return"` when `prior is not None` and `now - prior >= absence_s()`;
        - else `"none"`.
      - Then it sets `last_seen[person_id] = now`. A fresh tracker (app start) has no `last_wake`, so a first sighting gives `"none"`.
  - `Expressor._on_face(msg)`:
    - `awake = not asleep() and runner state != resting and not quiet_now()`.
    - On `"return"` or `"wake"`, it calls `offer(ReactionRequest("returning_person", {"wake": "1" if wake else ""}, clock(), key=person_id))`.
    - At `handle` (the expressor thread, never the bus thread) the first name is looked up from `key` with `person_name(registry, key)`, then `.split()[0]`, and put in a copy of `data` as `who`. `event_text` uses `WAKE_EVENT` or `RETURNING_EVENT`. The person id itself never reaches the event text.
  - `handle` for `returning_person` gives `dropped_busy` while `worker.in_session()` is True: no greeting into an open session's history. `reaction_turn(..., profile=False)`.

- [ ] **Step 1: Write the failing tests**
  - `test_absence_of_greet_absence_s_greets_once`: seen at `t=0`, then again at `t=1800`, gives `"return"`, then `"none"` at `t=1801`.
  - `test_track_churn_in_view_never_greets` (Review Focus 5): face frames every 1 s for `2 * 1800` s. The `track_id` changes every 20 s and `person.recognized` is published each time. There is no `returning_person` request.
  - `test_first_sighting_after_restart_does_not_greet` (Review Focus 5): a fresh expressor and a first `percept.face` for `"p1"` give no request.
  - `test_wake_from_night_greets_once_awake_and_outside_quiet_hours`:
    - The person is in view through the night: frames arrive while `resting` with reason `quiet_hours`, and they are not recorded.
    - The state goes to `awake` at 07:00, and `quiet_now()` is False.
    - The next frame gives a `"wake"` request whose event text uses `WAKE_EVENT`.
  - `test_woken_inside_quiet_hours_greets_only_after_they_end`: awake at 06:50 with `quiet_now()` True. Frames at 06:55 give no request, and the first frame at 07:00 gives one.
  - `test_day_nap_is_not_a_wake`: `resting`/`nap`, then `awake`, with the person last seen 10 min earlier, gives no request.
  - `test_name_from_registry_first_name_only`: `"Marcello Barile"` (active) gives an event text with `"Marcello"` and without `"Barile"`. A pending record gives `"a person you know"`.
  - `test_greeting_never_carries_profile_or_history`: through the chat `Server` fake, the reaction's request has no profile summary block and no history turns. The event text contains "Never mention anything from an earlier conversation".
  - `test_no_greeting_into_an_open_session`: `in_session()` True gives `dropped_busy`.
  - `test_reaction_turn.py::test_greeting_name_is_aliased_with_pseudonyms`: with `privacy.pseudonyms_to_cloud` on and a cloud chat route, the request body holds the alias and not `"Marcello"`. A cloud realtime route is already `blocked` with `pseudonyms` (`realtime/route.py:679`), so the greeting goes through chat.
  - `test_per_person_memory_in_ram_only`: no file is written under the data folder (tmp_path is empty after the test).
- [ ] **Step 2:** Run `pytest tests/test_expressor_returning.py tests/test_reaction_turn.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the interfaces above.
- [ ] **Step 4:** Run the same tests and `pytest -q`. Expected: PASS.
- [ ] **Step 5:** Commit as `feat(runtime): greet a returning person, never a past conversation (SPEC 6, 7, 8, 12)`.

---

## Part 5: Model asks instead of ignoring (built last)

### Task 5.1: The `ask_unclear` rule behind `expression.reactions.model_asks`

It applies only to realtime robot turns, the only ones that offer `ignore_utterance`, and only for audio that clearly reached the robot: the wake decision (state `ADDRESSED`: a wake phrase or the bare name) or the voice-at-face rule passed (`AT_FACE`). The expressor's limits do not apply inside the live session (design, owner decision).

**Files:**
- Create: `src/roboemot/llm/prompts/ask_unclear.en.txt`. Text: "When the speech is clearly meant for you but you cannot tell what the person wants, do not call ignore_utterance: ask one short question to find out."
- Modify:
  - `src/roboemot/llm/prompt.py` (`ASK_PROMPT`, `ask_prompt()`, `realtime_instructions(..., ask_rule: bool = False)`, appended after the skip rule and before the conversation hint)
  - `src/roboemot/voice/listener.py` (`Utterance.at_face: bool = False`, set in `_finish` when `p.face == AT_FACE` and the utterance is addressed)
  - `src/roboemot/conversation/worker.py` (`_rt_instructions` passes `ask_rule`)
  - `docs/specs/kickoff.md` (SPEC 8, Skip tool: the exception)
- Test: `tests/test_prompt_ask_unclear.py` (new), `tests/realtime/test_realtime_conversation_start.py`

**Interfaces:**
- Produces:
  - `ask_prompt() -> str` (`_instruction("ask_unclear")`).
  - `realtime_instructions(..., ask_rule: bool = False)`: the rule is added only with the skip rule.
  - In `_rt_instructions`: `ask_rule = cfg.expression.reactions.model_asks and (turn.conversation == ADDRESSED or turn.utterance.at_face)`. It is never set for a fixed line or an event turn.
  - Server turns (`turns: server`) get no rule. Their stream's instructions go at the stream's opening, before the wake or face decision exists.

- [ ] **Step 1: Write the failing tests**
  - `test_prompt_ask_unclear.py::test_rule_only_with_skip_rule_and_ask_rule`: `realtime_instructions(cfg, reg, ask_rule=True)` contains `ask_prompt()`. With `skip_tool` off, or `ask_rule=False`, it does not.
  - `tests/realtime/test_realtime_conversation_start.py::test_addressed_turn_carries_ask_rule`: an utterance with `wake_word=True` gives a `response.create` whose instructions contain `"do not call ignore_utterance: ask one short question"`.
  - `...::test_face_rule_turn_carries_ask_rule`: `at_face=True` and `wake_word=False` give the same.
  - `...::test_unaddressed_turn_has_no_ask_rule`: state `none` has no rule.
  - `...::test_model_asks_off_removes_the_rule`: `expression.reactions.model_asks: false` has no rule.
  - `tests/test_voice_listener.py::test_at_face_set_only_when_face_rule_addressed`.
- [ ] **Step 2:** Run `pytest tests/test_prompt_ask_unclear.py tests/realtime/test_realtime_conversation_start.py tests/test_voice_listener.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Write the SPEC 8 text.
- [ ] **Step 4:** Run the same tests, `pytest tests/realtime -q` and `pytest -q`. Expected: PASS.
- [ ] **Step 5:** Commit as `feat(realtime): ask when addressed speech is unclear, behind model_asks (SPEC 8)`.

---

## Close

- [ ] Push `feat/robot-expression` and open one PR into `main`. Its body lists the five parts with their commits and the robot checks below.
- [ ] Check that `gh pr view <n> --json headRefOid` matches the local HEAD before trusting CI. Merge once CI is green.
- [ ] Remove the worktree and delete the local branch after the merge is confirmed (`gh pr view <n> --json state` is `MERGED`).
- [ ] Robot checks for the owner:
  - Say a mumbled request with the cloud up: the robot asks again, with the `cue_not_heard` gesture and no blip.
  - Pull the network during a station: one sentence, and the radio is not paused by it.
  - Show a cup: one comment.
  - Leave for 30 min and come back: one greeting by first name.
  - Repeat each with the radio playing: gesture only, and the music is never dipped.
