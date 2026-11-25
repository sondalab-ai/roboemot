# Surroundings panorama: one wide picture of the room (design)

Date: 2026-11-25. Builds on `2026-11-24-surroundings-design.md`. SPEC sections: 8 (Surroundings
snapshots), 5A (Scanning).

## Intent

On the robot, a room snapshot was one frame taken wherever the head pointed. On 2026-11-25 it was a
close-up of a monitor. It told the model nothing about the room. The owner wants broad awareness:
several views across about 180°, sent as one picture. The owner also wants the feature used only for
the room, never for one shown object.

## Goals

- A room picture covers exactly 180° in front of the robot (owner, 2026-11-25).
- One room picture costs one request, as today.
- Room pictures come from views taken while the head is still.

## Non-goals

- Feature-based stitching (blended overlaps).
- Pictures of a shown object. Owner reversal, 2026-11-25: the earlier rule "a shown thing is
  photographed only when the robot reacts" is withdrawn. The shown-object reaction keeps its
  wording rule (a comment, no question) and uses the detector names only.
- Elevation changes. All views stay at the sweep's level pitch.

## Anti-goals

- A photo sweep stops halfway for anything but sleep, rest, quiet hours or external control.
- The head turns for a picture that cannot be sent (cap reached, provider limit, offline, no
  route, the route refuses images, a conversation is running).
- A face from one view masks a spot in another view.

## Approaches considered

1. **Tile the views side by side (chosen).** One JPEG, views in left-to-right order with a thin
   separator. The prompt states the view count, the order and the span. Cheap on the robot CPU.
   It never fails on plain walls.
2. Feature stitching (`cv2.Stitcher`). The sweep overlap is 15°, and low-texture rooms make the
   stitch fail often. It is slow on the robot CPU. Rejected.
3. One request per view. It costs three to four times the quota. Rejected.

When the head turns (owner, 2026-11-25: "1 + 2"):

- A normal look-around takes the views when a room picture is due.
- If no look-around starts within `surroundings.room.sweep_wait_s`, the robot starts its own
  look-around for the picture.

## Design

### Flow

1. **Due.** These triggers make a room picture due: start, switch-on, the end of a rest, and a
   scene change. Every refusal check runs first: off, vision missing, unsupported, paused, daily
   cap, provider limit, offline, route unavailable, asleep, music, busy, interval. A refused
   trigger is counted as today, and nothing turns.
2. **Sweep.** `ScanContext` gets a field `picture_due: bool` and `ScanPolicy` stays pure. The
   policy starts a sweep with reason `picture` when the field has been true for
   `sweep_wait_s` and no sweep runs. Its normal rules apply: no sweep while engaged, alone,
   asleep or during music. The first picture that becomes due after start or switch-on (a
   wake at boot can make it due) is the exception (owner, 2026-11-25):
   `ScanContext.picture_first` starts its photo sweep at once, with no `sweep_wait_s` wait,
   even while engaged or alone; asleep, music and a running sweep still hold it. One attempt
   only, kept by `ScanPolicy`: once that photo sweep started, a picture still due
   (`views_missing`, a failed description, no plan, a planner error) follows the normal rules;
   a cancel by sleep gives the attempt back for the wake. A switch-off and on owes a new first
   picture with a fresh attempt. A sweep that starts while a picture is due is a photo sweep:
   - its amplitude is exactly `surroundings.room.span_deg / 360`, whatever the energy trait says (owner, 2026-11-25);
   - its dwell is at least `surroundings.view_dwell_s`.
3. **Views.** `ScanController` exposes the waypoint where the head dwells now: the waypoint's
   azimuth from the moment the pose arrives until the dwell ends. The sweep's home return
   waypoint is not a view. During each dwell the picker chooses the sharpest still frame, as
   today. The faces come from that frame only.
4. **Compose.** At the sweep's end (`world.scan_finished`), the views are sorted by azimuth,
   left to right. Positive yaw turns left, so the order is descending azimuth. They are scaled to
   one height, joined with a separator, and encoded as one JPEG.
5. **Send.** One request, prompt `surroundings_room` reworded for N views left to right with the
   span in degrees. It asks for the layout: what is on the left, in the centre, on the right.
6. **A photo sweep always completes** (owner, 2026-11-25). A person who engages, a speaker hint,
   attention on an object, a glance or a conversation that starts do not cancel it. The scan
   holds the head for the whole sweep and the attention commands stay hidden until it ends. The
   picture is sent after the sweep even if a conversation started meanwhile. Only these stop it,
   because the head cannot move: sleep or rest, quiet hours, and external control (a hand or a
   platform move). Then nothing is sent and the picture stays due.
7. **Missing views.** A sweep that ends with fewer views than waypoints (a waypoint timed out,
   no still frame) sends nothing. The picture stays due.

### Scene change

Each sent picture keeps one thumbnail per view, keyed by azimuth. A scene check runs only while
the head is still within `room.view_match_deg` of a stored view, and compares with that view's
thumbnail. A changed share of at least `room.changed_share` makes a picture due.

### Removed

- The shown kind:
  - the expressor's request, `cancel`, `SHOWN_DESCRIBED` and the description wait;
  - the crop, `Picker` object speed, the pipeline's `_shown_box`;
  - the `surroundings.shown.*` settings.
- `SnapshotHold` and `surroundings.hold_s`. The sweep's dwell holds the head.
- `surroundings.room.max_per_day`. With one kind, `surroundings.max_per_day` is the only cap.
- Removed settings go into `config/moved_settings.yaml` (`dropped`), so a robot with saved
  overrides still loads.

### New settings (SPEC 11A)

| Path | Default | Meaning |
| --- | --- | --- |
| `surroundings.room.span_deg` | 180 | Smallest span a photo sweep covers |
| `surroundings.room.sweep_wait_s` | 60 | Wait for a normal look-around before the robot starts its own |
| `surroundings.view_dwell_s` | 1.5 | Smallest dwell at each view of a photo sweep |
| `surroundings.room.view_match_deg` | 10 | Head-to-view distance for a scene check |

### Live card

The card shows the joined picture, its view count and span. The history lists room pictures only.

## Facts checked in the code

- Reachy's planner at amplitude 0.5 (180°) with a 78° camera and 15° overlap gives 4 waypoints at
  −90°, −30°, 30° and 90°. The path starts from the end nearest the head. The controller adds a
  home waypoint. Source: `packages/roboemot-reachymini/src/roboemot_reachymini/world.py:68`.
- `head_pose` reads `head.yaw` only (`roboemot_reachymini/gaze.py:13`). The manifest constrains
  `head.yaw` against `body_yaw` (`yaw_delta`), so `head.yaw` is a base-frame angle and the body
  following the head does not turn the camera. Inferred from the manifest, not measured.
- Config models reject unknown keys (`config/schema.py:82`). `moved_settings.yaml` has a
  `dropped` list for removed settings (`config/stack.py:109`).

## Left to the builder

- Separator width and colour, and the shared height of the views.
- The sharpness floor stays a placeholder until measured on the robot.

## Not fixed here

- Google's daily quota may reset at midnight Pacific time, not at local midnight. Then the first
  request after a local midnight can hit the limit again. Noted, not changed.
