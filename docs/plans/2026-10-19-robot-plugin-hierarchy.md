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

Order: A → B → C → D → E1 → E2 (B–D may run in parallel if files don't overlap; E2 last). Estimate: 6 slices,
~14–16 agent runs (Sonnet authors, one Sonnet review each; no security review unless a slice touches secrets/web
auth). Re-confirm past ~24.

## Rules for authors
- Worktree per slice from `origin/epic/robot-plugin-hierarchy`, own `.venv`, check `roboemot.__file__`.
- Each slice adds a guard test: core must not name Reachy DOFs/products in behaviour code (extend
  `test_no_magic_numbers.py` or a new `test_robot_agnostic.py` with the sweep list: `reachy`, `antenna`,
  `body_yaw`, `head.pitch`, `pollen`, `:8000`, `/api/apps`), comments/docstrings excluded.
- Cite SPEC sections (4A, 8, 11A, 12A) in commits.

## Verification
Per slice: `ruff check . && ruff format --check .`, full `pytest`, node tests, CI green (incl. `models` and `sim`).
E1/E2: `pytest -m sim` and `pytest -m models` locally too, and a manual run against the sim daemon
(`scripts/sim-daemon.sh`) with the web UI. After the epic: install on the robot (`install_on_robot.py --dry-run`,
then install), then the hardware TODOs (nod/shake, Restart via the real daemon, CM4 CPU of the keyword worker).
