# RoboEmot

Social and emotional behaviour for robots: recognising people, an empathic register per
person, affect and a simulated circadian rhythm, mimicry, voice and conversation through any
OpenAI-compatible model. v1 targets Pollen Robotics' **Reachy Mini Wireless**, running entirely
on its Raspberry Pi CM4.

The design lives in [`docs/specs/kickoff.md`](docs/specs/kickoff.md).
Notes for coding agents are in [`CLAUDE.md`](CLAUDE.md).

## Status: M0 to M3 built and tested off the robot; first real-robot run in progress

| Area | SPEC | State |
| --- | --- | --- |
| Layered configuration, presets, guardrails, hot reload | 11A | Done, tested. Every change from any layer is checked as a whole candidate config against an admin-approved baseline; runtime approvals lapse with their override |
| Robot manifest, limits, capability negotiation | 4, 12 | Done, tested |
| Robot discovery on the LAN (mDNS, declared per manifest) | 4 | Done, tested; found Reachy Mini Wireless on a real LAN |
| Connectivity: offline detection, backoff, recovery, offline behaviour | 12A | Done, tested; robot-link recovery checked by restarting the simulated daemon's backend; the spoken cue goes through the speech output |
| Debug report: robot state, hardware, resources, system, battery (robot-agnostic, entry-point sources) | 12A | Done, tested; Reachy daemon sections read from a real Reachy Mini Wireless and the simulator; on the CM4 Linux exposes no battery (`/sys/class/power_supply` empty), only `vcgencmd` under-voltage flags and temperature |
| Installer and clean uninstall (robot-agnostic, entry-point platforms; Reachy Mini Wireless) | 4A | Done, tested against a fake robot (install then uninstall restores the exact state), including the voice extra, the voice-model download on the robot, clearing the uv cache and starting through the daemon's autostart; a real install ran on a Reachy Mini Wireless (Debian 13, daemon 1.12.0), including creating the shared apps venv on a robot with no app yet. That first real run found three things now fixed and covered by fake-robot tests only, not yet rerun on the robot: no voice extra and no voice models were installed, the disk was 97% full (536 MB free; the robot image's `~/.cache/uv` holds 5.6 GB), and `run_on_robot.py` collided with the app the daemon auto-starts on wake-up. A real uninstall on the robot is still to run |
| Robot plugins: core never names a robot; joint names, gestures, robot name and capabilities come from the manifest and the robot package's entry points | 4A, 11, 12A | Done, tested; enforced by `tests/test_robot_hierarchy.py` (AST gate with a ratchet baseline). Settings that moved or were renamed (e.g. `reflexes.antenna_ratio` is now `reflexes.startle_lift_ratio`) are migrated on restore and named in `app.yaml` errors |
| Arbiter + safety limiter (Python reference) | 10, 12 | Done; two independent reviews; property and fuzz tests (limits, velocity, acceleration, 65 deg coupling with a 3 deg planning margin) |
| Arbiter + safety limiter in C++ (`rt/`) | 10 | Matches the reference on 600-tick golden vectors (tolerance 1e-9); same input validation; not yet a running process |
| Reachy Mini driver, motion loop, app wrapper | 4A | `pytest -m sim` passes against the MuJoCo simulator (command-to-state 37-55 ms measured on a cloud dev container, not a CM4). The daemon started the app on a real Reachy Mini Wireless in the first real install; its log showed vision unavailable (`opencv` was not installed, the installer now installs it by default, `--no-vision` skips it). Motion, expressions and timing on the real robot are not checked |
| Affect engine, appraisal rules, empathic register, expression maps, reflexes, circadian model with camera light sensing | 7, 7A, 10 | Done, tested; the M2 exit checks pass in the simulator and through the app on a mocked client ([M2 report](docs/eval/m2.md)); expressions, rest posture, light thresholds and reflex timing not yet seen on the real robot |
| OpenAI-compatible LLM router, conversation with skills, API keys from the secret store (`${secret:name}`) | 8, 12 | Done, tested against fake servers and a mocked LAN route (first token 0.55 s in the M3 exit test); no real LLM route used on the robot |
| Voice: capture, VAD, fast path, wake phrases, ASR (Vosk), TTS (Piper in its own process, OpenAI-compatible speech), barge-in | 8, 13 | Done, tested; the M3 exit checks pass on a Mac with the real Silero and Vosk models and Piper in a Linux container, and run in CI (`models` job in `.github/workflows/ci.yml`): first TTS audio 1.34 to 1.43 s after the last speech frame against 2.0 s, barge-in 75 to 163 ms against 200 ms ([M3 report](docs/eval/m3.md)). Not seen on the robot: echo and AEC, CM4 load and latency, Piper install; the wake phrases missed on several synthetic voices |
| Web UI: PIN login, settings editor mirroring the configuration, robot section from the plugin, enrollment, LLM routes and write-only API keys on the Routes tab, Live tab with the robot's inner state, audio levels, an opt-in in-memory transcript and an opt-in still-picture video preview with a banner on every open page while it is watched (off by default, `webui.live.enabled`, `webui.live.transcript`, `webui.live.video`), language as a choice of the installed voice models | 8, 11A, 12 | Done, tested (settings-mirror epic complete, [plan](docs/plans/2026-10-18-webui-settings-mirror.md)); open on hardware per `TODO.md`: nod/shake, Restart through the real daemon. No TLS yet: PIN, cookie and a typed API key travel in clear on the LAN (SPEC 14, M5); `roboemot secrets set` over SSH avoids that |
| Bus transport (Zenoh vs iceoryx2) | 11 | In-process bus only; decision pending |
| Vision, identity, registry, world model, enrollment web UI | 5, 5A, 6 | Done, tested (M1); measured on a Mac, the CM4 is not measured ([M1 report](docs/eval/m1.md)) |
| Mirror (live mimicry, learning from demonstration) | 9 | Not started (M4) |

The affect state (valence, arousal and a label) is an expressive model: it decides which gesture
the robot plays and how large, and shapes the tone of voice and the conversation. It is not a claim
that the robot has feelings, and RoboEmot does not present it as one (SPEC 12, Behaviour and
content).

Known limits: the Reachy Mini manifest has placeholder translation limits and null dynamics
(measure in M0); until then the app starts only when installed with `--dev`. Once they are
measured, re-run the installer with `--no-dev`: it removes the dev key from the app config
(`app.yaml` in the data folder) and keeps your other settings there.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e "packages/roboemot-core[dev]" -e "packages/roboemot-reachymini[dev]"
pytest                                    # unit tests
cmake -S rt -B rt/build && cmake --build rt/build && ctest --test-dir rt/build
```

### Commands

`roboemot` CLI (installed with `roboemot-core`; `roboemot <command> --help` for every flag):

| Command | What it does |
| --- | --- |
| `roboemot config validate --app configs/companion.yaml --preset curious` | Check a layered configuration (defaults, preset, app file) against the schema and guardrails (SPEC 11A) |
| `roboemot config dump ...` | Print the effective merged configuration as YAML |
| `roboemot config schema` | Print the configuration JSON schema |
| `roboemot manifest check configs/manifests/reachy-mini-wireless.yaml --dev` | Validate a robot manifest and list its capabilities (SPEC 4, 12) |
| `roboemot discover [--json]` | Find robots on the LAN by mDNS, matched to the manifests in `configs/manifests` (SPEC 4) |
| `roboemot debug --discover` / `--host <ip>` | Debug report from the robot's platform: daemon, motors, pose, IMU, current app (SPEC 12A) |
| `roboemot debug [--json]` | Run on the robot: adds system, resources, temperatures, battery and the app process |

Scripts (run from the repository root):

| Script | What it does |
| --- | --- |
| `scripts/install_on_robot.py [--dry-run] [--uninstall]` | Install, start, or cleanly uninstall the app on a robot over SSH (see below; SPEC 4A) |
| `scripts/run_on_robot.py [--host H] [--dry-run]` | Start the installed app and print the web UI address; no SSH (SPEC 4A) |
| `scripts/robot_ssh.py [--copy-id] [--debug] [-- cmd]` | SSH into the robot (found by mDNS), install your key, or fetch its debug report built on the robot (SPEC 4A, 12A) |
| `scripts/sim-daemon.sh` | Headless MuJoCo Reachy Mini daemon on `localhost:8000` for `pytest -m sim` |
| `scripts/sync-package-data.sh` | Copy manifest and presets into the Reachy app's package data |
| `scripts/gen_golden.py` | Regenerate the golden vectors the C++ core is tested against |

Simulator (needs GStreamer/GObject system packages, see `.github/workflows/ci.yml`):

```bash
pip install -e "packages/roboemot-reachymini[sim]"
scripts/sim-daemon.sh &
pytest -m sim -s
```

### On the robot

Run the install wizard from a computer on the same network as the robot (not on the robot):

```bash
python scripts/install_on_robot.py --dry-run   # read-only checks; nothing on the robot changes
python scripts/install_on_robot.py             # install, verify, optionally start
python scripts/install_on_robot.py --uninstall --dry-run   # what uninstall would undo
python scripts/install_on_robot.py --uninstall # put the robot back as it was
python scripts/install_on_robot.py --no-vision --no-models --no-clear-caches   # skip optional steps, see below
```

To log in afterwards, `python scripts/robot_ssh.py` finds the robot the same way and opens a shell;
`--copy-id` installs your SSH key, `-- <command>` runs one command, `--debug` prints the debug
report built on the robot (with Wi-Fi signal strength), `--dry-run` shows the `ssh` line (it still runs discovery and only skips the exec).

It lists the robots it finds on the network by mDNS (or asks for a host, default
`reachy-mini.local`) and asks for the SSH user (default `pollen`);
`ssh` asks for the password once per run (`ssh-copy-id pollen@reachy-mini.local` avoids it).
Then it checks the daemon, architecture and Python of the robot's apps venv, builds the wheels
locally, installs them into `/venvs/apps_venv`, and checks that the daemon lists the app. On a
robot where no app was ever installed (a fresh image) that venv does not exist yet: the daemon
creates it only on its first app install. The installer then creates it the same way
(`/venvs/mini_daemon/bin/python -m venv`, plus `reachy-mini` pinned to the daemon's version) and
records that it did. Long steps (venv creation, pip installs, several minutes on the CM4) show a
live status line with the elapsed time and pip's current action. If
asked, it stops the running app (one app at a time), wakes the robot, starts RoboEmot, and makes
it the startup app (the daemon starts it on the robot's first wake-up). Choose dev mode until
the manifest dynamics are measured, or the app refuses to start (SPEC 12). Every prompt has a
flag for unattended runs: `--host`, `--user`, `--robot`, `--dev`, `--start`, `--startup-app`,
`--voice`, `--piper`, `--secret`, `--models`, `--clear-caches`, `--vision`, `--yes` (the boolean ones also
have a `--no-` form).

What goes onto the robot besides the two wheels:

- **Voice is on by default.** The installer sets `voice.enabled: true` (the library default
  stays off), and with `--dev` the dev content, in the app config: `app.yaml` in the data folder
  (`~/.local/share/roboemot/app.yaml`). It changes only those keys and keeps any other key you
  put there, so a reinstall does not undo your edits (comments are not kept); the dry run says
  what it would write. `--no-voice` leaves voice off and makes
  Piper and the models step default to off (they still run when asked with `--piper` or
  `--models`, and the models step still fetches the vision weights while vision is on). With voice
  on, declining Piper or the models warns that the robot will not speak or listen.
- **Voice packages.** `roboemot-core[voice]` (Vosk and ONNX Runtime) is installed by default, and a Piper
  virtualenv in the data folder (`--no-piper` skips it). After Piper the installer runs
  `roboemot models fetch --voice` on the robot (Silero VAD, Vosk it/en, Piper it/en; the prompt
  names the size, from the pinned model manifests) into `~/.local/share/roboemot/voice-models`,
  where the app reads them. If the free space is short it warns and skips, and you can run the
  command later; `--no-models` skips it, and a dry run only reports the sizes and the free space.
  Uninstall removes the models with the data folder (or, when the data folder was already there,
  the model folders the installer created).
- **Vision is on by default.** The installer also installs the `vision` extra (OpenCV, ONNX) and
  fetches the vision weights (about 1.4 GB of disk together with the voice models while downloading, INT8 files included; the figure is computed from the pinned manifests); `--no-vision` skips both,
  and then the app logs "vision unavailable" and runs without it. If the free space is short the
  models step is skipped with a warning that suggests `--no-vision`; a re-run over an install
  made with `--no-vision` adds the extra and the weights.
- **Disk.** The CM4 image has about 0.5 GB free (97% used on the robot of the first run) because
  the robot image's own setup leaves a 5.6 GB uv download cache in `~/.cache/uv`. The installer
  shows its size and asks whether to clear it before the free-space check (default yes on a terminal; an unattended run clears it only with `--clear-caches`, not with `-y` alone; `--no-clear-caches` keeps it).
  uv downloads again whatever it needs; uninstall cannot give the cache back. The dry run says
  exactly what the real run will do with it, and a failed free-space check names the kept cache
  and `--clear-caches`. A cache that a
  venv under `/venvs` links into is kept. The space the install needs is worked out from what it
  installs (the app, the voice and vision extras, and a new apps venv), and a rerun of the models
  step counts only the files not yet downloaded. Other caches
  (`~/.cache/huggingface`, `~/.cache/pip`) are left alone.

The installer leaves the robot as it found it. It keeps the package versions already in the
shared apps venv (it asks before changing one), and it writes no pip cache. It records what it
added or replaced in `/venvs/.roboemot-install.json`. `--uninstall` then removes exactly that:
the app, the dependencies it added (unless another app now needs them), the original versions
of anything it replaced, and the startup-app setting if it still names RoboEmot. It also removes
the app config it wrote (`~/.local/share/roboemot/app.yaml`) whole, including settings saved
from the web UI, and says so (also in the dry run). It removes the
apps venv too when it created it and no other app or package has moved in since. It verifies
the result before deleting the record. The installer is robot-agnostic (`roboemot.install`);
each robot package plugs in its own part (SPEC 4A, Installer).

Afterwards start the app with `run_on_robot.py` (daemon 1.12 retired its web dashboard). It
checks the app is installed, wakes the robot, starts RoboEmot, waits until it runs and prints the
web UI address. When RoboEmot is the startup app the daemon starts it itself on wake-up: the
script then waits for that instance (it says "started by the daemon on wake-up") instead of
sending a second start, which the daemon refuses. The address is `http://<robot>:8042` (the default
`webui.port`). The web UI is served by the app itself, so it is there only while RoboEmot runs.

```bash
python scripts/run_on_robot.py --dry-run   # what it would do
python scripts/run_on_robot.py             # start RoboEmot, print the web UI address
```

The daemon runs one app at a time, and the Reachy mobile app counts as one: while the phone is
connected to the robot, the daemon refuses to start RoboEmot, including as the startup app on
wake-up. Close the mobile app first; `run_on_robot.py` and the installer say so instead of
failing. To stop or restart, use the daemon's REST API:

```bash
curl -X POST http://reachy-mini.local:8000/api/apps/stop-current-app
curl -X POST http://reachy-mini.local:8000/api/apps/restart-current-app
```

While RoboEmot runs, the robot stays awake: idle motion keeps the motors powered, and the
daemon only puts the robot to sleep once the app stops. RoboEmot never wakes the robot itself,
and if the motors are turned off from outside it follows the pose without moving. When the
owner puts the robot to sleep from the app, RoboEmot notices that the robot stopped following its
commands, stops writing until it has been still for `motion.platform_move_grace_s`, and while the
motors stay off pauses the camera perception, the microphone listener and the behaviour tick
(`runtime.pause_when_asleep`). The microphone read loop and the web UI keep running, so this lowers
CPU use but does not stop it. There is no automatic nap on inactivity yet.

Debug report with the robot's own system, temperature and battery data (SPEC 12A):

```bash
ssh pollen@reachy-mini.local /venvs/apps_venv/bin/roboemot debug          # add --json
```

## Licence

Not decided yet (SPEC 15). Note that Piper TTS is GPL-3.0 and is meant to run as a separate
process.
