"""World map in the MuJoCo Reachy Mini simulation (SPEC 5A, 14; M1 exit "keeps a map ... that stays
correct while it scans"). Dev only.

Planted people, animals and objects sit at random base-frame bearings around the robot (at
least 45 degrees apart and 50 degrees off the home view, so a sweep is needed). A stand-in
detector reports each planted thing whenever the measured head pose has it inside the camera
view, like ``test_full_sweep_keeps_the_world_map_correct``; the real app functions run: world
model, attention, scan policy and planner, motion loop and the driver to the simulated daemon.
A planted person is seen while the head is near them and then "leaves" (no more observations)
so that attention on a person does not hold the sweep for the whole run. After the sweep the
observations stop and the world clock is advanced in steps (virtual time, no waiting: the
object half-life is 30 minutes) until every entry has faded.

Pass for a run: every planted thing has an entry of its kind within ``world.match_gate_deg`` of
it, there is no extra entry (no duplicate), and every entry fades after removal.

Needs the daemon: ``scripts/sim-daemon.sh`` (reachy_mini from the main checkout's venv, the
repo packages from this worktree through ``sys.path``). Run without ``-I``:

    python scripts/eval/world_sim.py --runs 5 --out world_sim
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for src in ("roboemot-reachymini", "roboemot-core"):
    sys.path.insert(0, str(ROOT / "packages" / src / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# The main checkout's venv lacks packages added after it was made (defusedxml, SPEC 6); the
# worktree venv's copies are appended last so they never shadow what the main venv has.
sys.path.extend(str(p) for p in (ROOT / ".venv" / "lib").glob("python3.*/site-packages"))

import common  # noqa: E402
import roboemot  # noqa: E402

assert Path(roboemot.__file__).resolve().is_relative_to(ROOT), roboemot.__file__

DEADLINE_S = 200.0
SETTLE_S = 4.0
MIN_SEPARATION_DEG = 45.0
MIN_OFF_HOME_DEG = 50.0
FADE_STEP_S = 5.0
FADE_LIMIT_S = 48 * 3600.0
PERSON_SEEN_S = 2.0
DATA = ROOT / "packages" / "roboemot-reachymini" / "src" / "roboemot_reachymini" / "data"


def plant(rng: random.Random, spec: list[tuple[str, str]]) -> list[tuple[str, str, float]]:
    """Random bearings (radians) for the given (kind, category) list under the spacing rules."""
    while True:
        az = [math.radians(rng.uniform(-175, 175)) for _ in spec]
        degs = [math.degrees(a) for a in az]
        if any(abs(d) < MIN_OFF_HOME_DEG for d in degs):
            continue
        gaps = [abs((a - b + 180) % 360 - 180) for i, a in enumerate(degs) for b in degs[i + 1 :]]
        if all(g >= MIN_SEPARATION_DEG for g in gaps):
            return [(k, c, a) for (k, c), a in zip(spec, az, strict=True)]


def fade_reference(kinds: list[tuple[str, str]], in_view: bool) -> dict[str, float]:
    """Seconds until a fresh entry of each kind is forgotten, one observation, nothing after.

    World model only (no simulation): the same decay code, with the entry in view (full rate)
    or out of view (slower), for comparison with the fade measured after the sweep.
    """
    from roboemot.config.schema import RoboEmotConfig
    from roboemot.world import Observation, WorldModel
    from roboemot_reachymini.gaze import FRAME

    cfg = RoboEmotConfig()
    out: dict[str, float] = {}
    for kind, category in kinds:
        world = WorldModel(cfg.world, FRAME)
        world.observe(Observation(kind, 0.0, "ref", 0.0, 0.0, category))
        t = 0.0
        while world.entries and t < FADE_LIMIT_S:
            t += FADE_STEP_S
            world.tick(t, lambda az, el, v=in_view: v)
        out[kind] = t
    return out


def one_run(mini, seed: int, spec: list[tuple[str, str]]) -> dict:
    from roboemot.attention import AttentionSelector, GazeController
    from roboemot.config import ConfigStack
    from roboemot.config.schema import RoboEmotConfig
    from roboemot.manifest import load_manifest
    from roboemot.motion import MotionLoop
    from roboemot.world import (
        Observation,
        ScanController,
        WorldModel,
        angular_distance,
        camera_in_view,
        from_base_frame,
    )
    from roboemot_reachymini import ReachyMiniDriver
    from roboemot_reachymini.app import attention_tick, scan_context, scan_tick
    from roboemot_reachymini.gaze import FRAME, GAZE
    from roboemot_reachymini.world import ReachyScanPlanner

    rng = random.Random(seed)
    planted = plant(rng, spec)
    cfg = RoboEmotConfig.model_validate(
        {"world": {"scan": {"idle_interval_s": 10, "jitter_s": 2, "dwell_s": 0.4}}}
    )
    manifest = load_manifest(DATA / "reachy-mini-wireless.yaml", allow_unmeasured_dynamics=True)
    loop = MotionLoop(manifest, ReachyMiniDriver(mini), ConfigStack().effective(), rate_hz=50)
    loop.start()
    try:
        world = WorldModel(cfg.world, FRAME)
        selector = AttentionSelector(cfg.attention)
        gaze = GazeController(cfg.attention, GAZE, manifest.dofs, manifest.constraints)
        scan = ScanController(
            cfg.world.scan,
            ReachyScanPlanner(manifest.dofs, manifest.constraints, cfg),
            manifest.dofs,
            random.Random(seed),
        )
        half_h = math.radians(cfg.world.camera_fov_h_deg) / 2 * 0.9
        half_v = math.radians(cfg.world.camera_fov_v_deg) / 2 * 0.9
        person_first: dict[int, float] = {}
        started = time.monotonic()
        gate = math.radians(cfg.world.match_gate_deg)

        def matched() -> int:
            return sum(
                any(
                    e.kind == kind
                    and angular_distance(e.azimuth_rad, e.elevation_rad, az, 0.0) <= gate
                    for e in world.entries
                )
                for kind, _, az in planted
            )

        sweeps, was_scanning, first_sweep_matched = 0, False, None
        while time.monotonic() - started < DEADLINE_S:
            now = time.monotonic()
            state = loop.driver.read_state()
            for n, (kind, category, az) in enumerate(planted):
                s_az, s_el = from_base_frame(az, 0.0, state, FRAME)
                if abs(s_az) > half_h or abs(s_el) > half_v:
                    continue
                if kind == "person":
                    first = person_first.setdefault(n, now)
                    if now - first > PERSON_SEEN_S:
                        continue
                world.observe(Observation(kind, now, "sim-detector", s_az, s_el, category), state)
            attention_tick(now, world, selector, gaze, loop)
            ctx = scan_context(now, cfg, selector, hour=12.0)
            scanning = scan_tick(ctx, scan, loop, cfg)
            if was_scanning and not scanning:
                sweeps += 1
                if first_sweep_matched is None:
                    first_sweep_matched = matched()
                if matched() == len(planted):
                    break
            was_scanning = scanning
            time.sleep(0.05)
        finished = sweeps > 0 and not scan.active
        sweep_s = time.monotonic() - started
        time.sleep(SETTLE_S)
        pose = loop.driver.read_state()
        end = time.monotonic()
        entries = list(world.entries)
        found, errors = 0, []
        for kind, _category, az in planted:
            same = [e for e in entries if e.kind == kind]
            best = min(
                (angular_distance(e.azimuth_rad, e.elevation_rad, az, 0.0) for e in same),
                default=None,
            )
            if best is not None and best <= gate:
                found += 1
            errors.append(None if best is None else math.degrees(best))
        extra = len(entries) - len(planted)
        in_view = camera_in_view(
            pose,
            FRAME,
            math.radians(cfg.world.camera_fov_h_deg),
            math.radians(cfg.world.camera_fov_v_deg),
        )
        fade: list[dict] = []
        alive = {e.entry_id: e for e in entries}
        virtual = 0.0
        while alive and virtual < FADE_LIMIT_S:
            virtual += FADE_STEP_S
            for gone in world.tick(end + virtual, in_view):
                entry = alive.pop(gone.entry_id, None)
                if entry is not None:
                    fade.append(
                        {
                            "kind": entry.kind,
                            "in_view_at_end": bool(in_view(entry.azimuth_rad, entry.elevation_rad)),
                            "seconds_to_forget": virtual,
                        }
                    )
        return {
            "seed": seed,
            "planted": len(planted),
            "found_within_gate": found,
            "extra_entries": max(0, extra),
            "bearing_error_deg": errors,
            "sweep_finished": finished,
            "sweeps": sweeps,
            "found_after_first_sweep": first_sweep_matched,
            "sweep_wall_s": sweep_s,
            "home_yaw_deg": math.degrees(abs(pose["head.yaw"])),
            "all_faded": not alive,
            "fade": fade,
            "pass": found == len(planted) and max(0, extra) == 0 and not alive and finished,
        }
    finally:
        loop.stop()


def main() -> None:
    import reachy_mini

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--set", choices=["things", "with_person"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = [("animal", "cat"), ("object", "cup"), ("object", "book")]
    if args.set == "with_person":
        spec = [("person", "person"), *spec]
    mini = reachy_mini.ReachyMini(
        connection_mode="localhost_only", media_backend="no_media", timeout=10
    )
    runs = []
    for r in range(args.runs):
        runs.append(one_run(mini, 100 + r, spec))
        last = runs[-1]
        common.log(
            f"run {r}: pass={last['pass']} found={last['found_within_gate']}/{last['planted']} "
            f"extra={last['extra_entries']} sweep={last['sweep_wall_s']:.0f}s"
        )
    summary = {
        "set": args.set,
        "planted_per_run": [f"{k}:{c}" for k, c in spec],
        "runs": len(runs),
        "passed": sum(r["pass"] for r in runs),
        "things_planted": sum(r["planted"] for r in runs),
        "things_found_within_gate": sum(r["found_within_gate"] for r in runs),
        "extra_entries": sum(r["extra_entries"] for r in runs),
        "sweeps_finished": sum(r["sweep_finished"] for r in runs),
        "all_faded_runs": sum(r["all_faded"] for r in runs),
        "sweep_wall_s": {
            "min": min(r["sweep_wall_s"] for r in runs),
            "max": max(r["sweep_wall_s"] for r in runs),
        },
        "per_run": runs,
        "fade_reference_one_observation_s": {
            "in_view": fade_reference(spec, True),
            "out_of_view": fade_reference(spec, False),
        },
        "protocol": {
            "gate_deg": 15.0,
            "min_separation_deg": MIN_SEPARATION_DEG,
            "min_off_home_deg": MIN_OFF_HOME_DEG,
            "fade": f"world clock advanced in {FADE_STEP_S:.0f} s steps (virtual time), "
            f"limit {FADE_LIMIT_S / 3600:.0f} h",
            "detector": "stand-in at the observation level (no images, no ONNX)",
            "simulation": "MuJoCo Reachy Mini daemon, headless; the fallback dynamics are slow",
        },
        "machine": common.cpu_label(),
    }
    json.dumps(summary)
    print(common.write_result(args.out, summary))


if __name__ == "__main__":
    main()
