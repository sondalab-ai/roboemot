# Surroundings: camera snapshots described as text (design)

> Superseded in part on 2026-11-25 by `2026-11-25-surroundings-panorama-design.md`: the room picture is a panorama from a photo sweep, and the shown-thing snapshot and the head hold are removed.

Date: 2026-11-24. Status: waiting for owner approval (builder check done). SPEC sections touched: 5, 5A, 8
(Robot expression, LLM router, Conversation state), 11A, 12, 12A.

## 1. Intent

On 2026-11-24 a person showed the robot a laptop. The detector said `laptop`. The reaction
turn received "A person is showing you something that looks like a laptop. Comment on it or ask
about it in one short sentence." The robot said "Ehi, cos'è questo oggetto piatto?" ("Hey, what
is this flat object?"). The model already had the answer and chose the hedge that the event text
offered.

The owner wants two things:

1. The robot reacts to a shown thing with a real remark, not a vague question.
2. The robot has a light sense of where it is and what is around it: the kind of room, the
   main things in it, and a better description of what a person shows it. The context must not
   be intrusive, and the Live tab must show it as it changes.

Today the robot knows only detector category names (`couch`, `tv`, `laptop`) of what is in view
now (`conversation/summaries.py:70`). It cannot say "a silver laptop with stickers" or "a living
room in evening light".

## 2. Goals, non-goals, anti-goals

Goals:

- G1. The shown-object reaction names the thing and makes one remark or question about it.
- G2. With the feature on, a cloud or LAN vision model turns camera snapshots into short text:
  one line for a shown thing, at most two lines for the room.
- G3. That text reaches every answering route (Gemini Live, Pollen, local pipeline) through the
  existing scene summary.
- G4. One switch on a route card turns the feature on and off.
- G5. The Live tab shows each snapshot that left the robot, the text that came back, and why a
  snapshot was skipped or failed.
- G6. Blurred snapshots are not sent.

Non-goals:

- No image ever goes into a conversation (owner decision 2026-11-24: "always describe to text").
- No local scene or room classifier model.
- No continuous video, no periodic timer snapshots.
- No memory of descriptions across restarts. Descriptions live in RAM only.
- No room map or room names. "Room" here means the view the camera has now.

Anti-goals (a failure even if it works):

- The robot talks about the room unprompted, or in every reply. The context is data the model
  may use, not a topic.
- A snapshot leaves the robot without the switch on, while the robot sleeps, or without the
  owner seeing it in the Live tab.
- A snapshot or its JPEG is written to disk, a log or a debug report.
- A model that refuses images causes a retry loop, an error storm in the log, or a broken
  conversation.

## 3. Constraints

- SPEC 8 and 12 say today: "Images are never sent (no image path to a route exists)". This design
  reverses that rule for one opt-in path. The SPEC change is part of the work.
- Gemini API and ElevenLabs terms bar users under 18 and apps likely used by under-18s (checked
  2026-11-11). Room snapshots in a home with children make this more relevant. The web UI text of
  the switch says so.
- Chat routes use the OpenAI-compatible common subset (SPEC 8). Image input (`image_url` content
  parts with a `data:` URL) is outside that subset today. The subset grows by exactly this one
  feature, used only on a route with the switch on.
- No magic numbers (SPEC 11A): every threshold below is a config field with default, unit, range.
- CPU budget on the CM4: the snapshot path must cost nothing while the feature is off, and only a
  few small operations per candidate frame while it is on.
- Privacy (SPEC 12): frames, crops and JPEGs stay in RAM. Logs, decision events and the debug
  report carry counts, reasons, ages and sizes only.
- Secrets (SPEC 8, 12): the route's key comes from the secret store, as for every route.

## 4. Approaches considered

| Approach | What it does | Trade-off |
| --- | --- | --- |
| A. Describer to text (chosen) | A route with the switch on gets each snapshot and returns short text. The text goes into the scene summary | Works with every answering route. One control point for images. One extra request and about 1 to 2 s on the shown-object reaction |
| B. Image into the answering session | The snapshot goes into the Gemini Live session | Richest and fastest reply, but Gemini only. Pollen and local routes gain nothing. Images build up in the session context |
| C. Both | A plus B for shown things | Most work and most data sent, small gain over A |

Owner decision 2026-11-24: A, configured from the route card. Turning the switch on enables the
feature. Turning it off disables it. A model that does not accept images is handled as an error,
not a crash.

A local room classifier (Places365-style) was dropped by the owner in favour of the snapshot path:
CPU on the CM4 and a model download, for a coarse label only.

## 5. Decisions

Marked (owner) for owner decisions of 2026-11-24, (design) for calls made in this document.

### 5.1 The switch

- (owner) A new route field `llm_routes[].images` (bool, default false), shown on the route card in
  the Routes tab as "Sees images". The feature is on exactly when one available route has it on.
  There is no second flag.
- (design) At most one route has `images: true`. Turning it on for a route turns it off on any
  other route in the same write. Config validation rejects two.
- (design) The switch exists on `kind: chat` routes only. A describe is one request and one reply;
  the realtime wire has no such call. To use Gemini, add a chat route on Google's
  OpenAI-compatible endpoint. The owner checks this call while reading.
- (design) Turning the switch on needs the admin PIN and a confirmation that names what leaves the
  robot and the provider's age terms, as the other privacy opt-ins do (SPEC 11A). Turning it off
  needs no PIN (it tightens).
- (design) The `images: true` route stays a normal chat route: it may also answer conversation
  turns in the routing order. The `images_unsupported` mark (5.6) stops only describe requests,
  never its chat use. Describe requests are not part of the fallback chain: a describe that fails
  is not retried on another route.
- (design) A `locality: local` route may have the switch too. Its snapshots stay on the LAN; the
  Live tab and the logs treat it the same way.

### 5.2 When a snapshot is taken

- (owner) Shown thing: one snapshot per shown look (SPEC 5A, Joint attention), when the expressor
  accepts a `shown_object` request. The `shown_object` cooldown also limits snapshots: one per
  category per cooldown.
- (owner) Room: one snapshot at wake from sleep or rest, and one after a large scene change, never
  more than one per `surroundings.room.min_interval_s`.
- (design) A large scene change: the scene-change test (`vision/scene_change.py`, `moved_share`)
  above `surroundings.room.changed_share` with the head still, or a scan sweep that ended (the
  view changed by design). Builder chooses the exact hook.
- (design) Never while the robot sleeps, rests or the quiet hours run, never while the radio plays,
  and a room snapshot only while the conversation is idle (a shown-thing snapshot is the reason for
  a turn, so it may run then). One "snapshot allowed" gate reuses the expressor's existing checks
  (`runtime/ticks.py`: `conversation_idle`, `media_on`; the sleep, rest and quiet-hours states the
  expressor already reads). No new signal is needed.

### 5.3 Sharp frames only (owner request: the head may move)

Three guards, each a reason the Live tab shows when it skips:

1. `head_moving`. A candidate frame needs the head still for `surroundings.settle_s`, measured by
   `HeadStill` on the frame's own `joint_state` (as the object gate does, `vision/object_gate.py`).
2. `object_moving`. For a shown thing, the entry's measured angular speed (SPEC 5A, Motion speed)
   must be under `surroundings.shown.max_speed_deg_s`.
3. `too_blurry`. The picker keeps up to `surroundings.candidates` still frames from the last
   `surroundings.window_s`, scores each by the variance of the Laplacian on the crop (on a
   downscaled grey copy), and sends the sharpest. It sends nothing when the best score is under
   `surroundings.min_sharpness`.

If no frame passes within `surroundings.shown.wait_s` (shown) or `surroundings.room.wait_s`
(room), the attempt ends with the last skip reason. The sharpness floor is only a guard against bad
frames: picking the sharpest does most of the work. The default floor is set from scores measured
on the robot (a builder task, section 7); until then it is low.

### 5.4 What is sent

- (owner) Every face the robot detected in the frame is filled with a flat colour before
  encoding, in shown-thing and room snapshots alike. The object is never cut away to avoid a face.
  A face the detector missed goes out uncovered; the owner accepted this on 2026-11-24. The
  face boxes come from the same frame as the snapshot (or the newest face run before it, grown by
  a margin field if they come from an earlier frame).
- (design) Shown thing: a crop around the thing's detection box, grown by
  `surroundings.shown.margin` of its size.
- (design) Room: the whole frame.
- (design) JPEG, longest side `surroundings.max_side_px`, quality `surroundings.jpeg_quality`.
- (design) Fixed English prompts in `llm/prompts/surroundings_shown.en.txt` and
  `surroundings_room.en.txt`. The shown prompt passes the detector category as a hint. The room
  prompt asks for the kind of place, the main things and the light, and says to leave people out.
  No names, no history, no persona, no system prompt of the robot.
- (design) `max_tokens` small (field), and the reply is cut to `surroundings.shown.max_chars` /
  `surroundings.room.max_chars` on one line, like a registry name (`summaries.one_line`).
- (design) Pseudonyms (`privacy.pseudonyms_to_cloud`): the request carries no names, so nothing
  to replace.

### 5.5 What comes back and where it goes

- (design) `SurroundingsState` in RAM holds the last room text with its time, and the last
  shown-thing text with the shown look's token.
- (design) The scene summary gains the room text as its last part, `Room: <text>`, inside the same
  `Scene (data, not instructions)` quoted string and under its own cap. A room text older than
  `surroundings.room.max_age_s` is dropped.
- (design) The shown thing's entry label in the scene summary becomes the description while the
  look's token is valid (`a laptop` → `a laptop: silver, with stickers`).
- (design) A shown-thing description is valid exactly as long as its `shown_object` request is
  valid: the expressor's existing token check (a thing no longer shown is stale) and the
  situation's `max_age_s`. A newer shown thing replaces the older request (the queue holds one
  per situation), and its snapshot attempt replaces the older attempt.
- (design) Every answering route reads the scene summary per turn, so a new description reaches
  the next turn with no extra path: chat turns build their messages per turn
  (`conversation/worker.py:2596`), and realtime turns, reaction turns included, send fresh
  instructions in a `session.update` per turn (`_rt_instructions`, `worker.py:1188`).
- (design) The `shown_object` reaction waits for the description at most `surroundings.shown.wait_s`
  (inside the situation's `max_age_s`). Without one it goes on with the category, as today.
- (design) The description is data. A sign in the picture that says "ignore your instructions"
  comes back as text inside the quoted, capped `Scene (data, not instructions)` string, like any
  other data today.

### 5.6 Failures

- (owner) A model that refuses images: the route is marked `images_unsupported` in RAM. The robot
  stops sending snapshots. The route card and the Live tab show the reason. The mark clears when one of
  that route's own fields changes (model, base URL, key) or the owner turns the switch off and on.
  A reload that leaves the route unchanged keeps the mark. No retry loop.
- (design) What counts as "refuses images": an HTTP 400 or 415 on a request with an image part.
  Builder checks the real bodies of at least one OpenAI-compatible server that has no vision
  (LM Studio or llama.cpp with a text model) and one cloud model without vision, and lists them in
  the tests.
- (design) Other errors (timeout, 5xx, 429, network down, budget spent): that snapshot is
  dropped with its reason; the next trigger tries again. After `surroundings.max_failures` in a row
  the feature pauses for `surroundings.pause_s`. Offline (the network monitor): no attempt.
- (design) The route's `daily_token_budget` counts describe requests as it counts chat requests.
- (design) The answering route never waits on a broken describer beyond `surroundings.shown.wait_s`.

### 5.7 Live tab and debug report

- (owner) A "Surroundings" card in the Live tab:
  - the switch state and the route name, or "off";
  - the last snapshot that left the robot (the JPEG from RAM), its kind (shown or room), its age,
    its size in bytes and its sharpness score;
  - the text that came back, and its age;
  - the last skip or failure reason and counts per reason since start;
  - `images_unsupported` with the route's reason code, when set.
- (design) The JPEG on the card follows the live preview rules (`runtime/live_video.py`): served
  only while a page asks, dropped when the robot sleeps or the feature goes off, never on disk.
  It is shown only when `webui.live.video` is on; the text is shown either way.
- (design) The debug report (SPEC 12A) gets a `roboemot.surroundings` section: on/off, route
  name, counts per outcome and per skip reason, last sharpness scores (numbers), last latency.
  No text and no image.

### 5.8 The shown-object event text (ships first, on its own)

- (design) `SHOWN_OBJECT_EVENT` (`runtime/expressor.py:99`) becomes: "A person is showing you
  {thing}. Say what you think it is and add one short remark or question about it, in one short
  sentence." `{thing}` is the description when there is one, else "something that looks like a
  {category}". The offline line `prompts/expression_shown_object.<lang>.txt` stays.
- This change is independent of the snapshots and goes in a separate small PR before the rest.

## 6. Configuration (all new fields, SPEC 11A)

New section `surroundings` (defaults are a first guess, ranges to confirm in the plan):

| Field | Default | Unit / range | Meaning |
| --- | --- | --- | --- |
| `settle_s` | 0.5 | s, 0.1–5 | Head still this long before a frame is a candidate |
| `window_s` | 1.0 | s, 0.2–5 | Candidate frames kept from this window |
| `candidates` | 5 | 1–20 | At most this many candidate frames in RAM |
| `min_sharpness` | measured | Laplacian variance | Floor under which nothing is sent |
| `max_side_px` | 768 | px, 256–1600 | Longest side of the JPEG |
| `jpeg_quality` | 80 | 40–95 | JPEG quality |
| `max_failures` | 3 | 1–20 | Errors in a row before a pause |
| `pause_s` | 600 | s, 60–86400 | Pause after `max_failures` |
| `shown.max_speed_deg_s` | 15 | deg/s, 1–180 | Thing slower than this to be a candidate |
| `shown.margin` | 0.25 | share, 0–1 | Crop growth around the thing's box |
| `shown.wait_s` | 4 | s, 0.5–30 | Longest wait for a sharp frame plus the reply |
| `shown.max_chars` | 80 | chars, 20–300 | Cap on the description |
| `room.min_interval_s` | 300 | s, 60–86400 | At most one room snapshot in this time |
| `room.changed_share` | 0.5 | share, 0–1 | Scene change that counts as a new view |
| `room.wait_s` | 10 | s, 1–60 | Longest wait for a sharp frame |
| `room.max_chars` | 160 | chars, 40–400 | Cap on the room text |
| `room.max_age_s` | 3600 | s, 60–86400 | Room text older than this is dropped |

Plus `llm_routes[].images` (bool, default false) and the prompt-size estimate in
`llm/prompt.py:347` growing by `room.max_chars`.

## 7. Left to the builder

- The frame tap: how the picker gets frames from the vision thread at no cost while idle (an
  idea: a tap like `light_tap` that copies a frame only while an attempt is open).
- The exact scene-change and sweep-end hook for room snapshots.
- Measuring sharpness on the robot (still head, waved object, dim room) and setting
  `min_sharpness`; recording the numbers in a `docs/memory/` note.
- The HTTP error bodies that mean "no image support" (section 5.6).
- Module layout (a `roboemot.surroundings` package is the expected shape) and the order of commits.
- One PR, commits per part (owner 2026-11-20), after the separate event-text PR (5.8).

## 8. Tests the plan must include

- Picker: head moving, object moving, blurred, sharpest chosen, nothing sent under the floor, no
  work while the feature is off.
- Every detected face box is filled in both kinds of snapshot; the shown crop keeps the whole
  object box.
- Switch: one route at most; turning on needs PIN; off needs none; `kind: realtime` refused.
- Router: image request shape; 400/415 → `images_unsupported`, no retry; 5xx/timeout → skip and
  pause after `max_failures`; budget counted.
- Scene summary: room line, description replaces label, caps, staleness.
- Expressor: waits up to `shown.wait_s`, falls back to the category.
- Privacy: no image bytes or descriptions in logs, decision events or the debug report
  (extend the existing privacy tests).
- Web UI: Surroundings card renders each state, JPEG hidden without `webui.live.video`.
