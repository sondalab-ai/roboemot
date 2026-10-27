# Robot plugin hierarchy (abstract → plugin): epic plan

## Context
Owner rule (2026-10-19): every robot-specific capability sits behind the hierarchy "abstract (core, robot-agnostic)
→ plugin (robot package)". Core consumes an abstract method or protocol defined in core; the robot package
contextualizes it by implementing it, found through an entry point keyed by manifest `robot.id`. This applies to all
existing implementations, not only new ones. RoboEmot is installed on the robot after this epic (owner decision).

Audit on `main` @ 6e67f76 (paths: `core/` = `packages/roboemot-core/src/roboemot/`, `rm/` =
`packages/roboemot-reachymini/src/roboemot_reachymini/`):
- Already compliant: `roboemot.diagnostics`, `roboemot.installers`, `roboemot.world`, `roboemot.settings`
  (with `Restartable`). No core module imports `roboemot_reachymini` or `reachy_mini`.
- Violations:
  1. `core/config/schema.py:377` `EXPRESSION_CHANNELS` whitelists Reachy DOF names. `expression_map`, `gesture_map`,
     `circadian_motion` and `rest_pose_deg` reject any other robot's channels (`configs/manifests/desk-head-v1.yaml`).
  2. `core/health/responder.py:60-76` `puzzled`/`perk_up` hardcode antenna names and signs; `GestureConfig`
     (`schema.py` ~2812) carries `antenna_rad`, `head_roll_rad`, `head_pitch_rad`.
  3. `core/voice/keywords.py:24` `REACHY = "reachy"`; default `name_aliases` `["reati"]`, `["riyadh she"]`
     (`schema.py` ~2361, ~2370). `voice` is not a manifest behaviour key, so a robot cannot supply its own.
  4. `core/affect/expression.py:48` `BODY_CHANNEL = "body_yaw"`, while `GazeMap.body_yaw` exists for exactly this.
  5. Driver, camera, mic, speaker and the gaze/frame/reflex maps have no plugin discovery: they are built by
     concrete type in `rm/app.py` (896 lines, mostly robot-agnostic runtime: pipeline assembly, hot reload fan-out,
     main loop; also `rm/voice.py`, `rm/conversation.py`, `rm/skills.py`).
  6. Three daemon HTTP clients: `rm/diagnostics.py:40`, `rm/settings.py:78-127`, `rm/install.py:55` (`DaemonApi`,
     urllib). Port defined twice; `current-app-status` implemented three times.
- Borderline:
  - undeclared optional capabilities probed by duck typing: `rm/app.py` `hasattr(world_source, "speaking")`,
    `getattr(source, "camera_model")`, `rm/voice.py` `hasattr(asr, "missing_words")`;
  - `rm/app.py` `rate_hz=50.0` and `stop_event.wait(0.05)` bypass `motion.arbiter.loop_hz` (SPEC 11A);
  - `core/webui/robot_section.py:33` `RESTART_EFFECTS` vocabulary is Reachy-shaped;
  - the daemon's 20 s kill is stated twice (`rm/settings.py` `STOP_TIMEOUT_S`, `runtime.shutdown_budget_s` help);
  - `rm/settings.py` imports `APP_NAME` from the PC-side installer module;
  - `core/skills/defaults.py:57` tells the LLM about "antennas and head".

## Slices (integration branch `epic/robot-plugin-hierarchy`, one PR each)
0. **Hierarchy gate (first, before any other slice).** Deterministic checks in the normal `pytest` run (so every CI
   job and every PR runs them). New `packages/roboemot-core/tests/test_robot_hierarchy.py`, AST-based, no regex on
   source text, comments and docstrings excluded:
   - **Import boundary.** No module under `core/` imports a robot package or a robot SDK: any distribution that
     registers a `roboemot.*` entry point other than `roboemot-core`, plus a list of known SDKs (`reachy_mini`).
     Derived from installed entry points, so a new robot package is covered without editing the test.
   - **Robot vocabulary in core.** String literals and identifiers in `core/` behaviour code must not contain robot
     terms. The term list is built from every manifest in `configs/manifests/` (robot ids, product names, DOF and
     joint names, mDNS service names, daemon ports and URL paths), plus a short static list (`reachy`, `pollen`,
     `antenna`, `/api/apps`). A new manifest therefore extends the gate by itself.
   - **Entry-point contract.** A core registry `roboemot.plugins.GROUPS` maps every `roboemot.*` entry-point group
     to its core protocol. Each `roboemot.*` group declared in any installed distribution must be in the registry,
     and each implementation must load and satisfy its protocol (runtime_checkable `isinstance` or a signature
     check). A robot capability without a core abstraction cannot be registered.
   - **No duck typing on core objects in robot packages.** No `hasattr`/`getattr` with a literal attribute name on
     objects typed with a core protocol; optional capabilities must be declared on the protocol.
   - **Robot-agnostic smoke.** Core builds its config, behaviour and renderer with a non-Reachy manifest
     (`desk-head-v1`) without errors.
   - **Ratchet baseline.** Today's known violations (the audit list above) go in
     `packages/roboemot-core/tests/robot_hierarchy_baseline.yaml`, one entry per finding (file, symbol, term,
     audit id). The test fails on any finding not in the baseline and on any baseline entry that no longer occurs,
     so the baseline only shrinks. Slices A–E2 each remove their entries; the epic ends with an empty baseline.
   - **Rule in `CLAUDE.md`** under "Rules that come from the spec": robot-specific behaviour only through a core
     protocol plus a `roboemot.*` entry point keyed by `robot.id`; `test_robot_hierarchy.py` enforces it; the
     baseline may never grow.
   - **SPEC 4A/12A** note on the gate and the registry.
0b. **Sweep (same PR as 0, before it merges).** Anything that landed on `main` or this epic branch after the audit
   commit (`6e67f76`) and before the gate merges is checked by hand and by the gate:
   - `git log --stat 6e67f76..origin/main` and `..origin/epic/robot-plugin-hierarchy`: list every feature commit
     touching `packages/`.
   - Run the gate with the baseline built from `6e67f76` only: any finding not present at `6e67f76` is a new
     violation. It is fixed in this PR, never added to the baseline.
   - Report the swept range and the result in the PR body. Re-run the same sweep at every slice merge until the
     epic lands (the gate itself covers the rest).

A. **One daemon client.** `rm/daemon.py`: base URL from `Target`, port, timeouts, `status()`, `current_app()`,
   `restart_current_app()`, install/start calls. Diagnostics, settings (Restart) and installer use it; `APP_NAME`
   and the stop timeout live there. The plugin declares its supervisor kill timeout through `Restartable` and core
   caps `runtime.shutdown_budget_s` below it instead of the help text naming Reachy.
B. **Joint names from the manifest.** Drop `EXPRESSION_CHANNELS`; validate channel names against the loaded
   manifest's DOFs (`set_manifest_behaviour` / renderer). Body joint from `GazeMap`. Connectivity gestures become
   named `gesture_map` entries (`puzzled`, `perk_up`) in the Reachy manifest, played by `ExpressionRenderer`; the
   antenna/head literals and `GestureConfig.antenna_rad` etc. leave core (fields migrated or removed, SPEC 11A).
   LLM tool text generic or from a manifest body description.
C. **Robot name from the manifest.** Name aliases and the default name move to manifest `behaviour`
   (`voice.keywords.*.name_aliases`, a default name); core compares against that, not `REACHY`. Keeps the M3 wake
   decisions (in-vocab generic phrases stay; parametric name: config > platform name > generic).
D. **Declared capabilities.** Optional members on the core protocols (`WorldSource.speaking`,
   `FrameSource.camera_model`, `ASRBackend.missing_words`) instead of `hasattr`/`getattr`; loop rate and main-loop
   tick from config (`motion.arbiter.loop_hz`, manifest-overridable). Restart effect ids generic, or a plugin
   supplies EN+IT text for its own ids.
E1. **RobotPlatform protocol.** Core `RobotPlatform` (driver, camera, mic, speaker, gaze/frame/reflex maps factories)
   behind a new `roboemot.robots` entry point keyed by `robot.id`, loaded fail-loud (no robot, no run). Reachy
   implements it; `rm/app.py` builds through it. No behaviour change.
E2. **Core Runtime.** Move the robot-agnostic runtime out of `rm/app.py` (pipeline assembly, hot reload, main loop,
   attention/scan/idle helpers) and the robot-agnostic parts of `rm/voice.py`, `rm/conversation.py`, `rm/skills.py`
   into `core/runtime/`. `RoboEmotApp(ReachyMiniApp).run` becomes a thin adapter: manifest → platform via entry
   point → `Runtime.run(stop_event)`. Keep the SPEC 4A rules inside the Reachy driver.

Order: 0 (+0b) → A → B → C → D → E1 → E2 (B–D may run in parallel if files don't overlap; E2 last). Each slice
removes its baseline entries. Estimate: 7 slices, ~16–18 agent runs (Sonnet authors, one Sonnet review each; no security review unless a slice touches secrets/web
auth). Re-confirm past ~27.

### Live view (added 2026-10-21, owner request; slices S, T, U)
Owner decisions (2026-10-21): a "Live" web UI tab; audio as levels + transcript only (no raw audio leaves the
robot); video shown normally whoever is in view (the PIN is the gate); indicator = a web UI banner on every open
page while a video viewer is active, no antenna pose; built inside this epic so the owner's robot run covers it.
Common rules: PIN session + CSRF like every page API; feature flag(s) off by default (SPEC 11A privacy opt-in),
rates and sizes as schema fields; nothing written to disk, logs or the debug report; the live view never wakes a
paused pump or the robot; plain HTTP residual risk recorded in SPEC 12 (TLS is M5).
S. **Inner-state panel.** One cheap endpoint built from the existing snapshot sources (`runtime/diagnostics.py`:
   affect/mood, sleep, network, minor, circadian, reflexes, motion, voice, vision tracks) plus a conversation phase
   (idle / listening / thinking = LLM stream in flight, with route and elapsed / speaking), added where missing.
   Page polls at a config rate; per-source cost measured; no catalogue rebuild per poll.
T. **Audio levels + transcript.** Mic level (the hub/onset RMS already computed) and the robot's output level;
   an in-memory ring of recent utterances and replies (size and age as schema fields), never persisted, logged or in
   the debug report. Opus security review (new exposure: SPEC 8/12A today keep replies out of reports).
U. **Video preview.** JPEG snapshot endpoint the page polls, tapped from the vision pump's latest frame (never opens
   the camera, SPEC 4A); encodes only while a viewer polled within a window; face boxes and names drawn; shows
   "asleep"/"unavailable" (no cv2) instead of failing; CSP `img-src` checked; banner on every open page while a
   viewer is active. Opus security review.
Estimate: 3 slices, ~7–9 agents (Sonnet authors and S review, Opus reviews for T and U, 1–2 fix rounds).

## Rules for authors
- Worktree per slice from `origin/epic/robot-plugin-hierarchy`, own `.venv`, check `roboemot.__file__`.
- Each slice deletes the baseline entries it fixes in `robot_hierarchy_baseline.yaml`; never adds one.
- Cite SPEC sections (4A, 8, 11A, 12A) in commits.

## Verification
Per slice: `ruff check . && ruff format --check .`, full `pytest`, node tests, CI green (incl. `models` and `sim`).
E1/E2: `pytest -m sim` and `pytest -m models` locally too, and a manual run against the sim daemon
(`scripts/sim-daemon.sh`) with the web UI. After the epic: install on the robot (`install_on_robot.py --dry-run`,
then install), then the hardware TODOs (nod/shake, Restart via the real daemon, CM4 CPU of the keyword worker).
