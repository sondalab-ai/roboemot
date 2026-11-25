# Surroundings panorama: implementation plan

**Goal:** A room picture is 3 to 5 still views from one look-around, tiled left to right into one
JPEG and sent as one request. The shown-object snapshot is removed.

**Spec:** `docs/specs/2026-11-25-surroundings-panorama-design.md` (the design, binding). SPEC
sections to update: `docs/specs/kickoff.md` 8 "Surroundings snapshots" (line ~1059) and 5A
"Scanning" (line ~523).

**Branch:** `feat/surroundings-panorama`, worktree `../roboemot-panorama`, own `.venv` (both
packages installed). One PR, one commit per task.

## Global constraints

- No magic numbers in behaviour modules: every constant is a `param(...)` in `config/schema.py`
  (`tests/test_no_magic_numbers.py`).
- Every new or changed param has an Italian description in `webui/descriptions_it.json`. Every new
  UI string has en and it in `webui/strings.py`.
- Removed settings go into `config/moved_settings.yaml` under `dropped`, with a reason and the date
  2026-11-25.
- No `getattr` duck typing on the host or on exceptions (`tests/test_robot_hierarchy.py`). Core
  never names a robot or its dofs.
- Privacy (SPEC 12): no frame, JPEG or description text in logs, events, reports or `state`.
- Validation per task: `ruff check . && ruff format --check .`, focused pytest. At the end: full
  `pytest` for both packages and `node --test packages/roboemot-core/tests/webui_js/`.

## Review focus

1. Views sorted by descending azimuth (positive yaw turns left). The home-return waypoint is
   never a view.
2. A photo sweep is not cancelled by engagement, alone, music, glance or object attention. Sleep,
   rest and quiet hours (`ctx.asleep`) cancel it.
3. The head never turns for a picture that cannot be sent: `picture_due` is true only after every
   refusal check passed.
4. Face boxes for a view come from that view's own frames.
5. A robot with saved overrides for removed keys still loads (`dropped`).

---

### Task 1: Scan support for photo sweeps (`world/scan.py`, `runtime/ticks.py`)

- `ScanContext` gets `picture_due: bool = False`, `picture_wait_s: float`,
  `picture_amplitude: float` (share of the circle), `picture_dwell_s: float`. `scan_context(...)`
  takes them from the caller (the loop reads `host.surroundings`).
- `ScanPolicy`: when `picture_due` has been true for `picture_wait_s`, no scan runs and the normal
  rules allow a sweep (not engaged, not alone, not asleep, no music), start a sweep with reason
  `"picture"`. Any sweep that starts while `picture_due` is true is a photo sweep:
  `ScanDecision.photo = True`, amplitude `max(normal, picture_amplitude)`, dwell
  `max(normal, picture_dwell_s)`. Add `"picture"` to `Reason`. The due-since clock resets when
  `picture_due` goes false.
- `ScanController.tick`: a running photo sweep is cancelled only by `ctx.asleep`. It feeds at
  `Priority.EXPRESSION`.
- `ScanController.view() -> tuple[int, float] | None`: the index and azimuth of the coverage
  waypoint where the head dwells now (from arrival until the dwell ends), only during a photo
  sweep. The home-return waypoint gives None. Also `views_planned() -> int`.
- `scan_tick`: during a photo sweep, clear `ATTENTION_SOURCE` even when engaged. At the end of a
  sweep publish `world.scan_finished` with payload `{"photo": bool}`.
- Tests in `tests/` next to the existing scan tests: photo sweep starts after the wait; a normal
  sweep becomes a photo sweep when due; engagement does not cancel a photo sweep; asleep does;
  `view()` is None on the home waypoint; amplitude floor applied at low energy.

### Task 2: Config, moved settings, SPEC

- `SurroundingsConfig`: remove `hold_s`, `shown` (whole group), `room.max_per_day`.
  Add `view_dwell_s` (1.5 s), `room.span_deg` (180), `room.sweep_wait_s` (60 s),
  `room.view_match_deg` (10). Keep `settle_s`, `window_s`, `candidates`, `min_sharpness`; the picker
  now runs per view inside the dwell, so validate `view_dwell_s >= settle_s + window_s`.
- `moved_settings.yaml` `dropped`: `surroundings.hold_s`, `surroundings.shown`,
  `surroundings.room.max_per_day`.
- SPEC 8 and 5A text, the 2026-11-24 design doc gets a one-line pointer to the new design.
- Italian descriptions, strings.

### Task 3: Service rewrite (`surroundings/`)

- Remove the `shown` kind everywhere: `request` kind, `cancel`, `description`, `wait_description`,
  `_shown`, `crop`, the picker's object speed, the `speed` and `shown` arguments.
- Due state: a room trigger (start, switch-on, rest end, scene change) runs the full refusal check
  (`_refusal`). Passed: `_due = True`, else counted as today. `picture_due() -> bool` for the scan
  context. The busy gate stays in the refusal check, so a picture is not due during a conversation.
- `tap(frame, faces)`: while `scan_view()` (injected callable, `ScanController.view`) gives a view,
  feed that view's own `Picker`. Masks use only that frame's faces (plus face tracks of that frame).
- `world.scan_finished` with `photo: True`: if every planned view has a chosen frame, tile and send
  (`_due = False`). Otherwise send nothing and keep `_due`. Gate check before sending drops only on
  asleep or music, not busy (design: the picture goes out even if a conversation started).
- `image.tile(views: list[np.ndarray], separator_px: int, height_px: int) -> np.ndarray`: views
  scaled to one height, joined left to right with a separator. Separator width and height come
  from config (`surroundings.max_side_px` bounds the view height; add params as needed).
- Scene check: keep one thumbnail per sent view, keyed by azimuth. Compare only with the head still
  within `room.view_match_deg` of a stored view.
- Prompt `llm/prompts/surroundings_room.en.txt`: placeholders for the view count and span; ask what
  is on the left, in the centre and on the right. `llm/prompt.py` follows.
- One cap: `max_per_day` only. Remove `_sent_room`.
- Tests: `tests/surroundings/` rewritten for the panorama flow (tile order, missing view, refused
  trigger turns nothing, busy at send still sends, asleep drops, per-view face boxes, scene check
  per view, daily cap, 429).

### Task 4: Remove the shown snapshot and the head hold

- `runtime/expressor.py`: no `request("shown")`, no `cancel`, no `_described`, no
  `SHOWN_DESCRIBED`. The shown reaction keeps `SHOWN_OBJECT_EVENT` (comment, no question) with the
  detector names. `tests/test_expressor_shown.py`, `tests/expressor_rig.py` follow.
- `runtime/ticks.py`: remove `SnapshotHold` and `HOLD_SOURCE`; `runtime/loop.py` drops the hold
  and passes the surroundings due state into `scan_context`. Remove
  `tests/runtime/test_snapshot_hold.py`.
- `vision/pipeline.py`: `snapshot_tap(frame, faces)`; remove `_shown_box` and its helpers if
  unused elsewhere.
- `runtime/startup.py`: wire `scan_view=p.scan.view` (and `views_planned`) into `Surroundings`;
  drop `speed`.
- `diag.py` REASONS: remove shown-only codes, add any new ones (e.g. `views_missing`).

### Task 5: Live card

- `webui/static/live.js` and `runtime/live.py`: the card shows the joined picture, its view count
  and span; the history lists room pictures only; no shown wording. Strings en/it.
- `tests/webui_js/live.test.mjs`, `tests/test_webui_live*.py` follow.
