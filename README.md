# RoboEmot

Social and emotional behaviour for robots: recognising people, an empathic register per
person, affect and a simulated circadian rhythm, mimicry, voice and conversation through any
OpenAI-compatible model. v1 targets Pollen Robotics' **Reachy Mini Wireless**, running entirely
on its Raspberry Pi CM4.

The design lives in [`SPEC.md`](SPEC.md) (export of the shared spec document).
Notes for coding agents are in [`CLAUDE.md`](CLAUDE.md).

## Status: M0 skeleton (work in progress)

| Area | SPEC | State |
| --- | --- | --- |
| Layered configuration, presets, guardrails, hot reload | 11A | Done, tested. Every change from any layer is checked as a whole candidate config against an admin-approved baseline; runtime approvals lapse with their override |
| Robot manifest, limits, capability negotiation | 4, 12 | Done, tested |
| Arbiter + safety limiter (Python reference) | 10, 12 | Done; two independent reviews; property and fuzz tests (limits, velocity, acceleration, 65 deg coupling with a 3 deg planning margin) |
| Arbiter + safety limiter in C++ (`rt/`) | 10 | Matches the reference on 600-tick golden vectors (tolerance 1e-9); same input validation; not yet a running process |
| Reachy Mini driver, motion loop, app wrapper | 4A | `pytest -m sim` passes against the MuJoCo simulator (command-to-state 37-55 ms measured on a cloud dev container, not a CM4); not yet run on the real robot |
| Affect engine, circadian model, camera light sensing | 7, 7A | Done, tested; no camera wiring yet |
| OpenAI-compatible LLM router | 8 | Done, tested against fake servers |
| Voice delivery composition (Piper, speech API) | 8 | Composition only; no TTS engine wired |
| Bus transport (Zenoh vs iceoryx2) | 11 | In-process bus only; decision pending |
| Vision, identity, registry, ASR/TTS engines, mirror, web UI | 5, 6, 8, 9 | Not started (M1-M4) |

Known limits: the Reachy Mini manifest has placeholder translation limits and null dynamics
(measure in M0); until then the app starts only when installed with `--dev`.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e "packages/roboemot-core[dev]" -e "packages/roboemot-reachymini[dev]"
pytest                                    # unit tests
roboemot config validate --app configs/companion.yaml --preset curious
roboemot manifest check configs/manifests/reachy-mini-wireless.yaml --dev
cmake -S rt -B rt/build && cmake --build rt/build && ctest --test-dir rt/build
```

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
```

It asks for the robot's host (default `reachy-mini.local`) and SSH user (default `pollen`);
`ssh` asks for the password once per run (`ssh-copy-id pollen@reachy-mini.local` avoids it).
Then it checks the daemon, architecture and Python of the robot's apps venv, builds the wheels
locally, installs them into `/venvs/apps_venv`, and checks that the daemon lists the app. If
asked, it stops the running app (one app at a time), wakes the robot, starts RoboEmot, and makes
it the startup app (the daemon starts it on the robot's first wake-up). Choose dev mode until
the manifest dynamics are measured, or the app refuses to start (SPEC 12). Every prompt has a
flag for unattended runs: `--host`, `--user`, `--dev`, `--start`, `--startup-app`, `--yes`.

Afterwards the app is started and stopped from the Reachy Mini Control desktop app (which
replaced the web dashboard) or through the daemon's REST API:

```bash
curl -X POST http://reachy-mini.local:8000/api/apps/start-app/roboemot
curl -X POST http://reachy-mini.local:8000/api/apps/stop-current-app
```

## Licence

Not decided yet (SPEC 15). Note that Piper TTS is GPL-3.0 and is meant to run as a separate
process.
