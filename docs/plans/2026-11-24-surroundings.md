# Surroundings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With one opt-in route switch, the robot sends sharp camera snapshots (a shown thing, the room) to a chat route that describes them as short text, and that text reaches every answering route through the scene summary.

**Architecture:** A new core package `roboemot.surroundings` holds a picker (blur guards, on the vision thread through a tap), the image shaping (face masking, crop, JPEG), a describer (one chat request with an `image_url` part, via the `Router`) and a service (triggers, gate, state, failures). The scene summary and the expressor read the service. The web UI gets a route-card switch and a Live "Surroundings" card.

**Tech Stack:** Python 3.11, numpy, OpenCV (vision extra) for JPEG, openai `AsyncOpenAI` through `llm/router.py`, pydantic config, vanilla JS web UI with `node:test`.

**Spec:** `docs/specs/2026-11-24-surroundings-design.md` (design) and `docs/specs/kickoff.md` (SPEC). Read both.

## Global Constraints

- SPEC 11A: every behavioural number is a `surroundings.*` (or `llm_routes[].images`) config field with default, unit and range. Add each new module to `tests/test_no_magic_numbers.py::BEHAVIOUR_MODULES`.
- Every new `param` description gets an Italian entry in `webui/descriptions_it.json`. Every new UI string gets `en` and `it` in `webui/strings.py`.
- SPEC 12: frames, crops, JPEGs and description text never go to logs, decision events, the debug report or disk. Logs carry outcome codes, counts, ages, sizes and scores only.
- SPEC 8: describe requests use the common subset plus exactly one addition: a user message whose content is `[{"type":"text",...},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]`. Streaming, `max_tokens`, no `response_format`, no `seed`.
- No image ever enters a conversation request (design 5, owner 2026-11-24).
- At most one route has `images: true`, and only a `kind: chat` route.
- Worktree runs need their own venv: check `python -c "import roboemot; print(roboemot.__file__)"` points inside the worktree before trusting a test run.
- Validation after each task: `ruff check . && ruff format --check .` and the task's focused tests. Run `pytest` (full unit suite) before the PR.
- Commit messages cite SPEC sections, end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Delivery: Task 0 is its own PR off `main`. Tasks 1–10 are one PR off `main` (branch `feat/surroundings`), one commit per task (owner 2026-11-20).

## Deviations from the design doc (decided while planning)

- **PIN on both directions.** Every settings write needs the admin PIN today (`WebApi._check_pin`, `webui/api.py:704`). Turning the switch off also needs the PIN. A no-PIN path would be a new endpoint for a small gain. (Design 5.1 said off needs none.)
- **Description text on the Live card follows `webui.live.video`,** like the JPEG. The text comes from the camera and is as private as the picture.
- **The shown-thing angular speed** needs a new public accessor `AttentionSelector.speed_dps(entry_id)`. Today it is private (`attention/selector.py:119`).
- **The sweep end** needs a new bus topic `world.scan_finished`. No such signal exists (`runtime/ticks.py:170`).

## Review Focus

1. Vision off or no vision extra (no cv2): the switch can be on, but the service reports `no_vision` in the Live card and the report, and it never raises. Test in Task 6.
2. The route key is missing or the route is offline or over budget: no attempt starts. The state shows `route_unavailable`. Test in Task 6.
3. The describer returns empty, multi-line or over-long text: the text is cut to one line and the cap. Empty text counts as a failure, and the room line never reads `Room: ` alone. Test in Task 5.
4. Two shown things in a row: the second attempt replaces the first. The first description never labels the second entry. Test in Task 6.
5. The switch goes off (hot reload) while an attempt is open or a request is in flight: no request starts after the change. The in-flight reply is dropped, and the kept JPEG is dropped. Test in Task 6.

---

### Task 0: Shown-object event names the thing (separate PR)

**Files:**
- Modify: `packages/roboemot-core/src/roboemot/runtime/expressor.py:99-102` (`SHOWN_OBJECT_EVENT`), `:666-682` (`event_text`)
- Modify: `docs/specs/kickoff.md:1047` (shown_object situation text)
- Test: `packages/roboemot-core/tests/test_expressor_shown.py`

**Interfaces:**
- Produces: `SHOWN_OBJECT_EVENT = "A person is showing you {thing}. Say what you think it is and add one short remark or question about it, in one short sentence."` and `SHOWN_THING = "something that looks like {article} {category}"`. `event_text` fills `thing` from `data["description"]` when present (Task 7 sets it), else from `SHOWN_THING` with `a`/`an` from `conversation.summaries._article`.

- [ ] **Step 1: Write the failing tests** `test_shown_object_event_asks_for_a_remark_not_a_hedge` (asserts `"Say what you think it is"` in `rig.worker.turns[0]["text"]`, `"looks like a laptop"` in it, `"or ask about it"` not in it) and `test_shown_object_event_uses_an_before_a_vowel` (category `umbrella` → `"an umbrella"`). Follow `test_shown_object_spoken_with_category`.
- [ ] **Step 2:** `pytest packages/roboemot-core/tests/test_expressor_shown.py -v` → the 2 new tests FAIL.
- [ ] **Step 3:** Change the template and `event_text`. Update `tests/test_reaction_turn.py:13` and `tests/realtime/test_event_turn_realtime.py:120` only if they assert the old wording. Update SPEC 1047: "it names what it thinks the thing is and adds one remark or question".
- [ ] **Step 4:** Focused tests PASS. `ruff` clean.
- [ ] **Step 5:** Branch `fix/shown-object-remark` off `main`. Commit `fix(expression): the shown-object reaction names the thing (SPEC 8)`. Open the PR.

### Task 1: Config, route switch, SPEC text

**Files:**
- Modify: `config/schema.py` (new `SurroundingsConfig`, `SurroundingsShownConfig`, `SurroundingsRoomConfig` near `MusicConfig` :4414. `RoboEmotConfig.surroundings` near :7893. `LLMRoute.images` near :6613. Root validator next to `_route_kinds_fit` :7928)
- Modify: `llm/prompt.py:347` (`system_tokens_worst_case` adds `surroundings.room.max_chars + ROOM_LABEL_CHARS`)
- Modify: `webui/descriptions_it.json`
- Modify: `docs/specs/kickoff.md`: new subsection `### Surroundings snapshots` after Robot expression (section 8), and lines 934, 962, 972, 1420 (`llm_routes` row: `images`), a new 11A row `surroundings`, 1505, 1508, 1515, and the 12A scope list (:1562-1577, `roboemot.surroundings`)
- Test: `tests/test_config_surroundings.py` (new), `tests/test_llm_history.py`

**Interfaces:**
- Produces: `cfg.surroundings.{settle_s, window_s, candidates, min_sharpness, max_side_px, jpeg_quality, max_failures, pause_s, describe_max_tokens}`, `cfg.surroundings.shown.{max_speed_deg_s, margin, wait_s, max_chars}`, `cfg.surroundings.room.{min_interval_s, changed_share, wait_s, max_chars, max_age_s}`. Defaults and ranges are in design section 6. `describe_max_tokens` is 120, range 16–1024. `min_sharpness` is 20.0, range 0–10000, unit "Laplacian variance". It is a placeholder until Task 10.
- Produces: `LLMRoute.images: bool = param(False, ..., opt_in=True)`. Produces `image_route(cfg) -> LLMRoute | None` in `config/schema.py`, next to `active_realtime_route` :6682.

- [ ] **Step 1: Failing tests:**
  - `test_defaults_match_the_design` checks each default from design section 6.
  - `test_two_image_routes_are_rejected` expects a `ValidationError` naming `images`.
  - `test_a_realtime_route_cannot_see_images`.
  - `test_image_route_returns_the_one_route_or_none`.
  - `test_images_is_a_privacy_opt_in`: off→on through `ConfigStack.set` counts as a `privacy` violation, as `webui.live.enabled` does.
  - `test_worst_case_counts_the_room_line` in `test_llm_history.py`.
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement. Validator messages: `"llm_routes: at most one route may have images on"`, `"llm_routes.<name>.images needs kind: chat"`.
- [ ] **Step 4:** Run the new tests, then `tests/test_webui_one_language.py` and `tests/test_no_magic_numbers.py`. All PASS.
- [ ] **Step 5:** Write the SPEC text from design sections 5–6. Replace the "no image path" sentences at 934, 962 and 972 with the opt-in path. Commit `feat(config): surroundings settings and the route image switch (SPEC 8, 11A, 12)`.

### Task 2: Picker (blur guards)

**Files:**
- Create: `surroundings/__init__.py`, `surroundings/types.py`, `surroundings/picker.py`
- Modify: `tests/test_no_magic_numbers.py` (add `surroundings/picker.py`)
- Test: `tests/surroundings/test_picker.py`

**Interfaces:**
- Produces in `surroundings/types.py`:
  - `Kind = Literal["shown", "room"]`
  - `Skip = Literal["head_moving", "object_moving", "too_blurry", "no_frame"]`
  - `@dataclass(frozen=True) class Candidate: image: np.ndarray = field(repr=False); stamp: float; sharpness: float`
- Produces in `surroundings/picker.py`:
  - `sharpness(image: np.ndarray) -> float`: the variance of a 3x3 Laplacian (numpy only, no cv2) on a grey copy. The copy is downscaled by strided slicing to at most `SHARPNESS_SIDE_PX = 256` on its longest side.
  - `class Picker`:
    - `__init__(self, cfg: Callable[[], SurroundingsConfig], wait_s: float, now: float)`
    - `offer(self, image: np.ndarray | None, stamp: float, *, head_moving: bool, object_moving: bool) -> None`
    - `result(self, now: float) -> Candidate | Skip | None`: None while the attempt goes on.
    - `last_skip: Skip | None`
- Rule:
  1. A frame with `head_moving` or `object_moving` is skipped with that reason. A `None` image is `no_frame`.
  2. Other frames join the window. The window holds at most `candidates` frames from the last `window_s`, counted from the first frame in it.
  3. When the window is full or `window_s` has passed, take the sharpest frame. If its score ≥ `min_sharpness`, return it. Else set `last_skip = "too_blurry"`, clear the window and go on.
  4. At `now - start ≥ wait_s`, return `last_skip or "no_frame"`.

- [ ] **Step 1: Failing tests:**
  - `test_a_sharp_frame_scores_higher_than_its_blurred_copy`: a checkerboard against the same board with a 5x5 box blur.
  - `test_the_sharpest_of_the_window_is_chosen`.
  - `test_head_moving_frames_never_become_candidates`.
  - `test_object_moving_frames_never_become_candidates`.
  - `test_all_blurred_ends_too_blurry_at_wait_s`.
  - `test_no_frames_ends_no_frame`.
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests and the no-magic test → PASS.
- [ ] **Step 5:** Commit `feat(surroundings): pick the sharpest still frame (SPEC 5, design 5.3)`.

### Task 3: Image shaping

**Files:**
- Create: `surroundings/image.py`
- Test: `tests/surroundings/test_image.py`

**Interfaces:**
- Consumes: `vision.types.Box` (pixels, `(x1, y1, x2, y2)`).
- Produces:
  - `shrink(image: np.ndarray, boxes: Sequence[Box], max_side_px: int) -> tuple[np.ndarray, list[Box]]`: a scaled copy and the scaled boxes. Never upscales.
  - `mask_faces(image: np.ndarray, faces: Sequence[Box]) -> np.ndarray`: a copy with each box, clipped to the image, filled with `MASK_BGR = (128, 128, 128)`.
  - `crop(image: np.ndarray, box: Box, margin: float) -> np.ndarray`: the box grown by `margin` of its width and height, clipped to the image.
  - `encode_jpeg(image: np.ndarray, quality: int) -> bytes`: `cv2.imencode`. It raises `NoEncoder` (a `RuntimeError`) without cv2.
- Order used by Task 4: `shrink` → `mask_faces` → `crop` (shown only) → `Picker.offer`. `encode_jpeg` runs once, on the chosen candidate only.

- [ ] **Step 1: Failing tests:**
  - `test_every_face_box_is_filled_and_nothing_else_changes`.
  - `test_a_face_overlapping_the_thing_is_masked_inside_the_crop`: the thing's box overlaps the face. The crop keeps the whole thing box, and the face pixels inside it equal `MASK_BGR`.
  - `test_crop_is_clipped_to_the_image`.
  - `test_shrink_scales_boxes_and_never_upscales`.
  - `test_jpeg_round_trips_size`, with `pytest.importorskip("cv2")` inside the test.
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them → PASS.
- [ ] **Step 5:** Commit `feat(surroundings): mask faces, crop and encode a snapshot (SPEC 12, design 5.4)`.

### Task 4: Vision tap and object speed

**Files:**
- Modify: `vision/pipeline.py`. Add a `SnapshotTap` type and the attribute `snapshot_tap: SnapshotTap | None = None` next to `light_tap` (:223, :409). Call it at the end of `_process`, after `_follow_shown`. Wrap the call in try, and log the exception type only, as `_tap_light` does (:524-534).
- Modify: `attention/selector.py`. Add `speed_dps(self, entry_id: str) -> float | None`, read from `_motion` under the selector's lock.
- Test: `tests/test_vision_snapshot_tap.py`, `tests/test_attention_shown.py`

**Interfaces:**
- Produces:
  - `SnapshotTap = Callable[[Frame, list[Box], Box | None], object]`. The arguments are the frame, the face boxes of `_fresh_face_boxes(frame.stamp)`, and the box of the shown track (`follower.target`) or None.
  - `snapshot_wanted: Callable[[], bool] = lambda: False` on the pipeline. The tap runs only when `snapshot_tap` is set and `snapshot_wanted()` returns true, so the idle cost is one call per frame.

- [ ] **Step 1: Failing tests:**
  - `test_tap_gets_frame_faces_and_shown_box_when_wanted`, built with the helpers of `tests/test_vision_object_gate.py` (`scene()`, `drive()`).
  - `test_tap_is_not_called_when_not_wanted`.
  - `test_a_failing_tap_does_not_stop_the_pipeline_and_logs_no_data`.
  - `test_speed_dps_reports_the_measured_speed_of_a_waved_thing`, in `test_attention_shown.py`.
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, and the existing `test_vision_object_gate.py` and `test_attention_shown.py` → PASS.
- [ ] **Step 5:** Commit `feat(vision): snapshot tap and the shown thing's speed (SPEC 5, 5A)`.

### Task 5: Describer

**Files:**
- Modify: `llm/router.py`. Add `describe_image`. Fix `prompt_chars` (:415) so that it counts text parts only; reuse `llm/prompt.py:267` `message_tokens` logic.
- Create: `surroundings/describer.py`, `llm/prompts/surroundings_shown.en.txt`, `llm/prompts/surroundings_room.en.txt`
- Test: `tests/test_llm_router.py`, `tests/surroundings/test_describer.py`

**Interfaces:**
- Produces in `llm/router.py`:
  - `async Router.describe_image(self, route: LLMRoute, prompt: str, jpeg: bytes, max_tokens: int) -> str`. It reuses the client, the secret, the budget charge, the `prompts` ring (the image is already redacted, `prompt_log.py:49`) and the timeouts of `_stream_route`, with `stream=True`. The result is the joined text.
  - It raises `NoRouteAvailable` when the key is unresolved or the budget is spent, `Offline` when offline, and `ImagesUnsupported(code: str)` (new, a `RuntimeError`) on `APIStatusError` with status 400 or 415. Other errors propagate.
- Produces in `surroundings/describer.py`:
  - `describe(router: Router, route: LLMRoute, kind: Kind, jpeg: bytes, hint: str, cfg: RoboEmotConfig) -> str` (async). It fills the prompt file (`{hint}` = category, shown only) and cuts the reply with `summaries.one_line(text, shown.max_chars | room.max_chars)`. It raises `EmptyDescription` on blank text.
- Prompt copy:
  - shown: "Describe the object in the centre of this photo in at most 12 words: what it is, colour, one detail. A detector thinks it is: {hint}. Plain text, no people."
  - room: "Describe this place in at most 25 words: the kind of room, the main things in it, the light. Do not describe people. Plain text."

- [ ] **Step 1: Failing tests** (fake HTTP `Server` of `test_llm_router.py`):
  - `test_describe_image_sends_one_text_and_one_image_part`: asserts the request JSON shape and `stream: true`.
  - `test_a_400_raises_images_unsupported_and_is_not_retried`: exactly one request.
  - `test_a_415_raises_images_unsupported`.
  - `test_describe_charges_the_budget_without_counting_image_bytes`.
  - `test_prompt_chars_ignore_image_parts`.
  - In `test_describer.py`: `test_reply_is_one_line_and_capped` and `test_blank_reply_raises_empty_description`.
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, and all of `test_llm_router.py` and `test_prompt_log.py` → PASS.
- [ ] **Step 5:** Commit `feat(llm): describe one image on the image route (SPEC 8)`.

### Task 6: Surroundings service and runtime wiring

**Files:**
- Create: `surroundings/service.py`
- Modify: `runtime/ticks.py`. Add `robot_resting(host, asleep: Callable[[], bool]) -> bool`, moved out of `Expressor.resting` (`runtime/expressor.py:586`). The expressor calls it.
- Modify: `runtime/ticks.py:170` `scan_tick`. On the end of a `"sweep"` (not a glance), publish `world.scan_finished` (payload None).
- Modify: `runtime/host.py:61`. Add `surroundings: Surroundings | None = None`.
- Modify: `runtime/startup.py`. Add `start_surroundings(host, stack, p, cleanup)`, called before `start_expressor` (:340). Set `p.perception.pipeline.snapshot_tap` and `snapshot_wanted`.
- Modify: `runtime/diagnostics.py:38`. Add `FnSource("roboemot.surroundings", ...)`.
- Modify: `diag.py` `REASONS` and `FIELDS`. Add a `"surroundings"` tuple of outcome codes.
- Test: `tests/surroundings/test_service.py`, `tests/test_scan.py`, `tests/test_expressor.py`

**Interfaces:**
- Produces: `class Surroundings`:
  - `__init__(self, cfg: Callable[[], RoboEmotConfig], router: Callable[[], Router], *, gate: Callable[[Kind], str | None], speed: Callable[[str], float | None], online: Callable[[], bool], clock=time.monotonic)`
  - `attach(bus)`: `robot.awake`, `circadian.state` (rest→awake), `world.scan_finished` → `request("room")`.
  - `request(self, kind: Kind, token: str = "", hint: str = "") -> str | None`: None when it opened an attempt, else the reason code. A new `shown` replaces an open `shown` attempt.
  - `wanted(self) -> bool`: the pipeline's `snapshot_wanted`.
  - `tap(self, frame: Frame, faces: list[Box], shown: Box | None) -> None`, called on the vision thread. It uses `HeadStill` with `vision.object_gate.head_still_deg` and `surroundings.settle_s`. While no attempt is open and the room interval has passed, it also runs the room scene-change test (`thumbnail` and `moved_share` of `vision/scene_change.py`, against `room.changed_share`).
  - `description(self, token: str) -> str | None`
  - `wait_description(self, token: str, timeout_s: float) -> str | None`
  - `room(self, now: float) -> str | None`: None past `room.max_age_s`.
  - `state(self) -> dict`: codes, counts, ages, `bytes`, `sharpness`. It holds no text.
  - `live(self) -> dict`: `state()` plus `text`. Only the Live card uses it.
  - `jpeg(self) -> bytes | None`: the last sent snapshot.
  - `apply_config(self, cfg)`: a stack subscriber. Switch off: close the attempt, drop the JPEG and the texts, and ignore any in-flight reply (a generation counter). It clears `unsupported` only when the image route's `model`, `base_url` or `api_key` changed, or the switch went off and on.
  - `close()`
- The describer runs on its own thread with one `asyncio` loop. At most one request is in flight.
- Outcome codes: `described`, `head_moving`, `object_moving`, `too_blurry`, `no_frame`, `unsupported`, `failed`, `paused`, `offline`, `asleep`, `music`, `busy`, `interval`, `route_unavailable`, `no_vision`, `off`.
- Gate built in `start_surroundings`:
  - `asleep` ← `robot_resting`
  - `music` ← `media_on`
  - `busy` ← `not conversation_idle`, for `room` only
  - `offline` ← `p.network`
  - `no_vision` ← no `p.perception` or no cv2

- [ ] **Step 1: Failing tests** (fake router, fake clock, synthetic frames):
  - `test_switch_off_starts_nothing`.
  - `test_a_shown_request_ends_described_with_the_sharp_crop_sent`.
  - `test_room_on_wake_then_not_again_within_min_interval`.
  - `test_scan_finished_requests_a_room_snapshot`.
  - `test_unsupported_stops_attempts_until_the_route_changes`.
  - `test_max_failures_pauses_for_pause_s`.
  - `test_gate_reasons_asleep_music_busy_offline`.
  - Review Focus 1: `test_no_vision_is_reported_and_never_raises`.
  - Review Focus 2: `test_route_unavailable_when_key_missing_or_budget_spent`.
  - Review Focus 4: `test_second_shown_replaces_first_and_old_text_never_labels_new_entry`.
  - Review Focus 5: `test_switch_off_mid_request_drops_reply_and_jpeg`.
  - `test_state_and_logs_hold_no_text_and_no_image_bytes` (caplog).
  - In `test_scan.py`: `test_a_sweep_end_publishes_scan_finished_and_a_glance_does_not`.
  - In `test_expressor.py`: the existing resting tests still pass through `robot_resting`.
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, then `tests/test_expressor*.py`, `tests/test_scan.py`, `tests/test_diagnostics.py`, `tests/test_diag_events.py` → PASS.
- [ ] **Step 5:** Commit `feat(surroundings): snapshot service, triggers and gate (SPEC 5A, 8, 12, 12A)`.

### Task 7: Scene summary and the shown-object reaction

**Files:**
- Modify: `conversation/summaries.py:70` `scene_summary`. Add `descriptions: Mapping[str, str] = {}` and `room: str | None = None`.
- Modify: `conversation/sources.py:84` `scene_source`. Add `surroundings: Callable[[], Surroundings | None] = lambda: None`.
- Modify: `runtime/startup.py:183-189`. Pass `lambda: host.surroundings`.
- Modify: `runtime/expressor.py`.
  - `_on_attention`, on a new look: `host.surroundings.request("shown", token=key, hint=category)`.
  - `_turn`: before the turn, `wait_description(token, remaining)`. The remaining time is `shown.wait_s` minus the time since the request. Put the description in `data["description"]` (Task 0's slot).
- Test: `tests/test_conversation_parts.py`, `tests/test_expressor_shown.py`

**Interfaces:**
- Consumes: `Surroundings.description`, `.room`, `.request`, `.wait_description` (Task 6).
- Produces:
  - A described entry's label is `f"{label}: {text}"`. The room is appended as `f"Room: {room}"` after the entities, under its own cap `surroundings.room.max_chars + len("Room: ")`. It never counts against `scene_summary_max_chars`.
  - The `Scene (data, not instructions)` quoting in `build_messages` is unchanged.

- [ ] **Step 1: Failing tests:**
  - `test_a_described_entry_reads_label_colon_text`.
  - `test_room_line_is_last_and_has_its_own_cap`.
  - `test_no_room_line_without_room_text` (Review Focus 3).
  - `test_shown_reaction_uses_the_description_when_it_arrives_in_time`.
  - `test_shown_reaction_falls_back_to_the_category_after_wait_s`.
  - `test_reaction_waits_no_longer_than_max_age`.
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, and `tests/test_reaction_turn.py` → PASS.
- [ ] **Step 5:** Commit `feat(conversation): descriptions in the scene summary and the shown reaction (SPEC 8)`.

### Task 8: Route card switch

**Files:**
- Modify: `webui/routes.py:122` `list_routes`. The row gains `images: bool` and `images_state: "off" | "on" | "unsupported" | "paused"`, from `host.surroundings.state()`. It also gains `images_reason: str | None`, a code.
- Modify: `webui/static/app.js`. On a `kind: chat` card, add a "Sees images" switch. Turning it on opens a confirm dialog (what leaves the robot, the provider's age terms) and then one `POST /api/settings/batch` behind `pinnedCall`. The batch sets this route on and turns off any other route that is on. Turning it off posts one PIN-gated write.
- Modify: `webui/strings.py` (en, it)
- Test: `tests/test_webui_routes.py`, `tests/webui_js/routes.test.mjs`

- [ ] **Step 1: Failing tests:**
  - `test_route_row_reports_images_and_state`.
  - JS: `switch on posts one batch that turns the other route off`.
  - JS: `switch is absent on a realtime card`.
  - JS: `unsupported state shows its reason`.
- [ ] **Step 2:** Run `pytest ...test_webui_routes.py` and `node --test packages/roboemot-core/tests/webui_js/routes.test.mjs` → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, and `tests/test_webui_one_language.py` → PASS. `scripts/audit_webui.sh` is clean.
- [ ] **Step 5:** Commit `feat(webui): Sees images switch on the route card (SPEC 11A, 12)`.

### Task 9: Live Surroundings card

**Files:**
- Modify: `runtime/live.py:113`. Add a `surroundings` block from `_surroundings(cfg, now)`, modelled on `_music` (:338). It is None without the service. `text` is present only when `webui.live.video` is on.
- Modify: `webui/api.py`. Add `GET /api/live/surroundings.jpg` (path list :73) with the same checks and codes as `_live_frame` (:921). It also answers 404 `surroundings_off` when the feature is off.
- Modify: `webui/static/live.js:52`. Add `liveSurroundingsCard` to `LIVE_CARDS`. It shows the state, the route, kind and age, bytes and sharpness, the text, the last skip reason, counts per reason, and the image when video is on.
- Modify: `webui/strings.py` (en, it, with `live_code_` sentences for every outcome code)
- Test: `tests/runtime/test_live_state.py`, `tests/test_webui_live.py`, `tests/webui_js/live.test.mjs`

- [ ] **Step 1: Failing tests:**
  - `test_surroundings_block_shape`.
  - `test_surroundings_text_only_with_video_on`.
  - `test_surroundings_jpeg_needs_same_origin_and_video_on`.
  - Extend `test_nothing_private_is_in_the_answer_and_nothing_is_logged` with the feature on and video off.
  - JS: `surroundings card renders each state`.
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, and `tests/test_webui_one_language.py` → PASS.
- [ ] **Step 5:** Commit `feat(webui): Surroundings card in the Live tab (SPEC 11A, 12)`.

### Task 10: Robot measurement and defaults (hil)

**Files:**
- Create: `tests/hil/test_surroundings_sharpness.py` (`@pytest.mark.hil`). It logs the picker scores for: head still, head in a sweep, a thing waved, a thing held still, a dim room.
- Modify: `config/schema.py` `min_sharpness` default; `docs/specs/kickoff.md` (11A row)
- Create: `docs/memory/2026-11-xx-surroundings-sharpness.md`, with the scores, the chosen floor and the date.

- [ ] **Step 1:** Install the branch on the robot. Turn on the switch for a Gemini chat route on Google's OpenAI-compatible endpoint. Run the hil test.
- [ ] **Step 2:** Set `min_sharpness` between the "head in a sweep" and "held still" medians. Record the numbers.
- [ ] **Step 3:** Check by hand that the Live card shows a described laptop and a room line, and that the robot's reaction names the thing.
- [ ] **Step 4:** Run the full `pytest`, `ruff`, and the JS tests → PASS.
- [ ] **Step 5:** Commit `chore(surroundings): sharpness floor from robot scores (SPEC 11A)`. Open the PR.
