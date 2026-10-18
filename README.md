# RoboEmot

Social and emotional behaviour for robots: recognising people, an empathic register per
person, affect and a simulated circadian rhythm, mimicry, voice and conversation through any
OpenAI-compatible model. v1 targets Pollen Robotics' **Reachy Mini Wireless**, running entirely
on its Raspberry Pi CM4.

The design lives in [`docs/specs/kickoff.md`](docs/specs/kickoff.md).
Notes for coding agents are in [`CLAUDE.md`](CLAUDE.md).

## Status: M2 affect and expression (work in progress)

| Area | SPEC | State |
| --- | --- | --- |
| Layered configuration, presets, guardrails, hot reload | 11A | Done, tested. Every change from any layer is checked as a whole candidate config against an admin-approved baseline; runtime approvals lapse with their override |
| Robot manifest, limits, capability negotiation | 4, 12 | Done, tested |
| Robot discovery on the LAN (mDNS, declared per manifest) | 4 | Done, tested; found Reachy Mini Wireless on a real LAN |
| Connectivity: offline detection, backoff, recovery, offline behaviour | 12A | Done, tested; robot-link recovery checked by restarting the simulated daemon's backend; spoken cue waits for TTS |
| Debug report: robot state, hardware, resources, system, battery (robot-agnostic, entry-point sources) | 12A | Done, tested; Reachy daemon sections read from a real Reachy Mini Wireless and the simulator; on the CM4 Linux exposes no battery (`/sys/class/power_supply` empty), only `vcgencmd` under-voltage flags and temperature |
| Installer and clean uninstall (robot-agnostic, entry-point platforms; Reachy Mini Wireless) | 4A | Done, tested against a fake robot (install then uninstall restores the exact state); dry runs checked on a real Reachy Mini Wireless leave no file changed; a real install and uninstall cycle on the robot is still to run |
| Arbiter + safety limiter (Python reference) | 10, 12 | Done; two independent reviews; property and fuzz tests (limits, velocity, acceleration, 65 deg coupling with a 3 deg planning margin) |
| Arbiter + safety limiter in C++ (`rt/`) | 10 | Matches the reference on 600-tick golden vectors (tolerance 1e-9); same input validation; not yet a running process |
| Reachy Mini driver, motion loop, app wrapper | 4A | `pytest -m sim` passes against the MuJoCo simulator (command-to-state 37-55 ms measured on a cloud dev container, not a CM4); not yet run on the real robot |
| Affect engine, appraisal rules, empathic register, expression maps, reflexes, circadian model with camera light sensing | 7, 7A, 10 | Done, tested; the M2 exit checks pass in the simulator and through the app on a mocked client ([M2 report](docs/eval/m2.md)); expressions, rest posture, light thresholds and reflex timing not yet seen on the real robot |
| OpenAI-compatible LLM router | 8 | Done, tested against fake servers |
| Voice delivery composition (Piper, speech API) | 8 | Composition only; no TTS engine wired |
| Bus transport (Zenoh vs iceoryx2) | 11 | In-process bus only; decision pending |
| Vision, identity, registry, world model, enrollment web UI | 5, 5A, 6 | Done, tested (M1); measured on a Mac, the CM4 is not measured ([M1 report](docs/eval/m1.md)) |
| ASR/TTS engines, mirror | 8, 9 | Not started (M3-M4) |

The affect state (valence, arousal and a label) is an expressive model: it decides which gesture
the robot plays and how large, and will shape the tone of voice and conversation once M3 lands. It is not a claim
that the robot has feelings, and RoboEmot does not present it as one (SPEC 12, Behaviour and
content).

Known limits: the Reachy Mini manifest has placeholder translation limits and null dynamics
(measure in M0); until then the app starts only when installed with `--dev`.

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
```

To log in afterwards, `python scripts/robot_ssh.py` finds the robot the same way and opens a shell;
`--copy-id` installs your SSH key, `-- <command>` runs one command, `--debug` prints the debug
report built on the robot (with Wi-Fi signal strength), `--dry-run` shows the `ssh` line.

It lists the robots it finds on the network by mDNS (or asks for a host, default
`reachy-mini.local`) and asks for the SSH user (default `pollen`);
`ssh` asks for the password once per run (`ssh-copy-id pollen@reachy-mini.local` avoids it).
Then it checks the daemon, architecture and Python of the robot's apps venv, builds the wheels
locally, installs them into `/venvs/apps_venv`, and checks that the daemon lists the app. If
asked, it stops the running app (one app at a time), wakes the robot, starts RoboEmot, and makes
it the startup app (the daemon starts it on the robot's first wake-up). Choose dev mode until
the manifest dynamics are measured, or the app refuses to start (SPEC 12). Every prompt has a
flag for unattended runs: `--host`, `--user`, `--robot`, `--dev`, `--start`, `--startup-app`, `--yes`.

The installer leaves the robot as it found it. It keeps the package versions already in the
shared apps venv (it asks before changing one), and it writes no pip cache. It records what it
added or replaced in `/venvs/.roboemot-install.json`. `--uninstall` then removes exactly that:
the app, the dependencies it added (unless another app now needs them), the original versions
of anything it replaced, and the startup-app setting if it still names RoboEmot. It verifies
the result before deleting the record. The installer is robot-agnostic (`roboemot.install`);
each robot package plugs in its own part (SPEC 4A, Installer).

Afterwards the app is started and stopped from the Reachy Mini Control desktop app (which
replaced the web dashboard) or through the daemon's REST API:

```bash
curl -X POST http://reachy-mini.local:8000/api/apps/start-app/roboemot
curl -X POST http://reachy-mini.local:8000/api/apps/stop-current-app
```

While RoboEmot runs, the robot stays awake: idle motion keeps the motors powered, and the
daemon only puts the robot to sleep once the app stops. RoboEmot never wakes the robot itself,
and if the motors are turned off from outside it follows the pose without moving. There is no
automatic nap on inactivity yet.

Debug report with the robot's own system, temperature and battery data (SPEC 12A):

```bash
ssh pollen@reachy-mini.local /venvs/apps_venv/bin/roboemot debug          # add --json
```

## Licence

Not decided yet (SPEC 15). Note that Piper TTS is GPL-3.0 and is meant to run as a separate
process.
