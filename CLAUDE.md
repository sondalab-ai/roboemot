# RoboEmot: working notes for coding agents

RoboEmot is a robot-agnostic library for social and emotional robot behaviour.
`docs/specs/kickoff.md` (the SPEC) is the engineering specification and the source of truth for every
design decision. Read the relevant section before changing a module, and cite the
section number in commit messages that implement or change behaviour.

## Repository layout

| Path | What |
| --- | --- |
| `docs/specs/kickoff.md` | Engineering spec (SPEC), the source of truth |
| `packages/roboemot-core/` | Python package `roboemot`: config, manifest, discovery, health (connectivity), diagnostics (debug report), vision (detect, track, embed, replay, model fetcher; ONNX backends need the `vision` extra), motion reference, affect (engine, appraisal, runner, register resolver, expression renderer), circadian, persona (traits), LLM router and system prompt, runtime (`runtime/`: `Runtime` pipeline assembly, hot reload, main loop, debug-report wiring; `RuntimeHost` seams), conversation sources and skill bridge, web UI robot section (`webui/robot_section.py`, `roboemot.settings` plugins), secret store (`secret_store.py`: `${secret:name}` route keys, 0600 file), bus |
| `packages/roboemot-reachymini/` | Python package `roboemot_reachymini`: Reachy Mini driver and SDK bridge; `app.py` is a thin adapter (config + manifest, platform by `robot.id`, then core `Runtime.run`) |
| `packages/roboemot-core/src/roboemot/webui/static/vendor/sondalab-ui-kit/` | Vendored, pinned `@sondalab/ui-kit` (offline web UI). Never edit by hand: change the pin in `scripts/sync_ui_kit.py` and run it; `scripts/audit_webui.sh` runs `sl-audit` on the page (SPEC 12) |
| `rt/` | C++ real-time core (`roboemot-rt`): arbiter and safety limiter, CMake + ctest |
| `configs/` | Library defaults, robot manifests, personality presets |
| `scripts/` | Dev helpers (simulation daemon, `install_on_robot.py`: wrapper over the robot-agnostic installer `roboemot.install`; `robot_ssh.py`: wrapper over `roboemot.install.ssh`; `run_on_robot.py`: wrapper over `roboemot.install.run`) |

## Rules that come from the spec

- **No magic numbers in behaviour code (SPEC 11A).** Every behavioural constant is a
  field in `roboemot.config.schema` with default, unit and range. Code reads it from
  the merged config. `tests/test_no_magic_numbers.py` enforces this for behaviour modules.
- **Safety only gets more conservative at runtime (SPEC 11A, 12).** Runtime overrides of
  `safety.*` may tighten, never loosen. Manifest hardware limits are ceilings.
- **A manifest without limits is rejected (SPEC 12).** `null` dynamics are allowed only
  with `allow_unmeasured_dynamics: true` in dev, and the limiter then uses conservative
  fallbacks.
- **Reachy Mini (SPEC 4A):** stream with `set_target` only; never call `goto_target` in a
  session (it blocks). The daemon's runner connects; the driver calls
  `set_automatic_body_yaw(False)` on start and after a reconnect. Call `set_target` only after
  `enable_motors()`. Antenna order in the SDK is `[right, left]`. One process owns the SDK
  client and its media.
- **External control and sleep (SPEC 4A, 12A):** the motion loop yields (no write, follows the measured
  pose) when the measured pose stops following its commands (`motion.external.*`: a platform move of
  its own, a hand) until the pose has been still for `motion.platform_move_grace_s`. No driver signal
  and no daemon polling for it (the daemon journals every REST call). While the platform does not hold
  the motors enabled the robot is asleep (`robot.asleep` on the bus) and the runtime pauses perception,
  listener and the behaviour tick (`runtime.pause_when_asleep`).
- **LLM routes are OpenAI-compatible only (SPEC 8).** Use the common subset: streaming,
  `tools` with `tool_choice="auto"`, `max_tokens`. Do not rely on `response_format`,
  forced tool choice, `seed` or non-default sampling.
- **Privacy (SPEC 12):** never write frames, audio or embeddings to logs.
- **Secrets (SPEC 8, 12):** API keys come only from the secret store (`roboemot secrets set`, value never an argument). Never log, repr, report or return a key; a route whose `${secret:name}` cannot be resolved is unavailable, never sent with an empty key.
- **Robot hierarchy (SPEC 4A, 11, 12A):** robot-specific behaviour only through a core protocol plus a
  `roboemot.*` entry point keyed by `robot.id`, registered in `roboemot.plugins.GROUPS`. Core never
  names a robot, its DOFs or its SDK. `tests/test_robot_hierarchy.py` enforces it; the baseline
  `tests/robot_hierarchy_baseline.yaml` may only shrink, never grow.
- **Robot platform (SPEC 4A, 11, 12A):** entry-point group `roboemot.robots`, keyed by `robot.id`: a stateless
  `RobotPlatform` class (`roboemot.robot_platform`) with gaze/frame/reflex maps; `bind(client)` gives the per-run
  `RobotSession` that builds the driver, camera, audio hub, speaker and bearing source as core protocols. `load_robot_platform` fails loud (no platform, no run); the app builds the robot
  only through it and core's `roboemot.runtime.Runtime` runs it; the Reachy app only adapts (SPEC 4A rules stay in the driver/session).
- **Diagnostics (SPEC 12A):** robot-specific debug sources are registered in the
  `roboemot.diagnostics` entry-point group under the manifest `robot.id`; core stays robot-agnostic.
- **Web UI robot section (SPEC 11A, 12):** entry-point group `roboemot.settings`, keyed by `robot.id`; the plugin's
  `section(stack, manifest)` returns read-only `robot.<id>.*` rows and the config paths it claims. Core never writes a plugin row.
- **Installer (SPEC 4A, Installer):** same pattern, entry-point group `roboemot.installers`. It
  must leave the robot clean: record only its own changes, pin existing versions, `--no-cache-dir`
  on every pip call, and the dry run never writes.

## Commands

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e "packages/roboemot-core[dev]" -e "packages/roboemot-reachymini[dev]"
# worktree agents need their own venv (editable installs point at the main checkout); check
# `python -c "import roboemot; print(roboemot.__file__)"` is inside the worktree before trusting a test run
pytest                                   # unit tests (no robot, no simulator)
scripts/sim-daemon.sh &                  # MuJoCo Reachy Mini daemon, headless, port 8000
roboemot models fetch --all              # vision weights (~0.8 GB) into the app data folder (~/.local/share/roboemot/models), sha256-verified
roboemot models fetch --voice            # Silero VAD, Vosk it/en, Piper it/en (~0.3 GB) into ~/.local/share/roboemot/voice-models
pytest -m models                         # real ONNX models, needs the vision extra and fetched weights
pytest -m sim                            # integration tests against the simulated robot
cmake -S rt -B rt/build && cmake --build rt/build && ctest --test-dir rt/build
ruff check . && ruff format --check .
python scripts/install_on_robot.py --dry-run   # read-only checks against the real robot
python scripts/install_on_robot.py --uninstall --dry-run   # what uninstall would undo
python scripts/install_on_robot.py --no-vision --no-voice --no-clear-caches   # skip the vision extra and weights; leave voice off (no voice.enabled, Piper and models default off); keep ~/.cache/uv (default: ask)
python scripts/run_on_robot.py   # start RoboEmot on the robot and print the web UI address
python scripts/run_on_robot.py --restart   # restart a running RoboEmot via the daemon (asks; -y skips; the head moves)
python scripts/robot_ssh.py [--host H] [--user U]   # shell on the robot (mDNS pick); `-- cmd` runs one command
roboemot secrets set cloud_key           # API key for ${secret:cloud_key} (read from the prompt/stdin); list/remove too
python scripts/robot_ssh.py --copy-id    # install your SSH key; --debug: roboemot debug --json built on the robot; --dry-run prints the ssh line (discovery still runs, no exec)
roboemot debug --host <robot> --json     # debug report (SPEC 12A); system/battery need it run on the robot
```

## Conventions

- Python 3.11+, type hints everywhere, `ruff` for lint and format, `pytest` for tests.
- Units in names: `_deg`, `_rad`, `_s`, `_ms`, `_hz`, `_db`, `_st` (semitones).
- Angles are radians inside the code; degrees only in YAML files for humans, converted at load.
- Tests that need the simulator are marked `@pytest.mark.sim`; tests that need the real
  robot are marked `@pytest.mark.hil` and never run in CI.
