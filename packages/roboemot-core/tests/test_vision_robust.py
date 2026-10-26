"""Vision robustness: track expiry, replay safety, logging, fetcher (SPEC 5, 12)."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import threading
from pathlib import Path

import httpx
import numpy as np
import pytest
from roboemot import cli
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.vision import (
    Detection,
    Frame,
    LatestFrameRunner,
    Tracker,
    VisionPipeline,
)
from roboemot.vision.models import ModelError, ModelFile, check_file, fetch_file
from roboemot.vision.replay import ReplayError, load_session, replay, summarize
from roboemot.vision.runner import pump
from roboemot.world import Observation
from test_vision import (
    CFG,
    CONSENT,
    FACE_MODEL,
    FixedFaceEmbedder,
    ScriptedDetector,
    axis,
    face,
    image,
    registry,
    rig,
    run,
    write_session,
)

SECRET = "SECRETARRAYDATA"
LOST = CFG.vision.tracker.lost_s


# --------------------------------------------------------------------- track expiry


def test_tracker_gap_longer_than_lost_s_gives_a_new_id_at_the_same_spot():
    t = Tracker(CFG.vision.tracker)
    t.update([face(100)], 0.0)
    first = t.update([face(100)], 0.125).visible[0].track_id
    t.update([face(100)], 0.25)
    t.update([face(100)], 0.25 + LOST + 0.5)
    again = t.update([face(100)], 0.25 + LOST + 0.625)
    assert again.visible[0].track_id != first


def test_tracker_expired_ids_are_reported_even_without_a_detection_update():
    t = Tracker(CFG.vision.tracker)
    t.update([face(100)], 0.0)
    tid = t.update([face(100)], 0.125).visible[0].track_id
    assert t.expire(0.125 + LOST - 0.01) == []
    assert t.expire(0.125 + LOST + 0.01) == [tid]


def test_tracker_crossing_faces_keep_their_ids():
    t = Tracker(CFG.vision.tracker)
    ids = {}
    for i in range(24):
        stamp = i * 0.125
        big = Detection((20 + 10 * i, 60, 110 + 10 * i, 150), 0.9)
        small = Detection((260 - 10 * i, 100, 300 - 10 * i, 140), 0.9)
        for tr in t.update([big, small], stamp).visible:
            ids.setdefault("big" if tr.box[2] - tr.box[0] > 70 else "small", set()).add(tr.track_id)
    assert all(len(v) == 1 for v in ids.values()) and len(ids) == 2


def test_returning_face_is_a_new_track_without_old_identity(tmp_path):
    present = lambda i: [face(100)] if (i < 20 or i >= 60) else []  # noqa: E731
    reg, _, pipe, _, got = rig(tmp_path, present)
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT)
    run(pipe, 100)
    faces = [m.payload for m in got["percept.face"]]
    old, new = faces[0].track_id, faces[-1].track_id
    first_new = next(f for f in faces if f.track_id == new)
    assert new != old and first_new.new_track
    assert first_new.owner_id is None or first_new.stamp > 60 * 0.125
    assert old not in pipe._faces


def test_state_is_dropped_by_stamp_while_the_governor_is_off(tmp_path):
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)] if i < 8 else [])
    run(pipe, 8)
    tid = got["percept.face"][0].payload.track_id
    pipe.governor.set_scale("circadian", 0.0)
    pipe.process(Frame(image(), 1.0 + LOST + 1))
    assert tid not in pipe._faces and tid not in pipe.matcher._tracks


# --------------------------------------------------------------- embedding schedule


def test_small_and_weak_faces_are_tracked_but_not_embedded(tmp_path):
    small = CFG.vision.embed_min_face_px - 4
    _, _, pipe, emb, got = rig(tmp_path, lambda i: [face(100, size=small)])
    run(pipe, 8)
    assert got["percept.face"] and emb.stamps == []
    _, _, pipe2, emb2, got2 = rig(
        tmp_path / "w", lambda i: [face(100, score=0.9)] if i < 2 else [face(100, score=0.4)]
    )
    run(pipe2, 12)
    assert len(emb2.stamps) == 1


def test_identity_within_budget_with_one_missed_vote_at_5_hz(tmp_path):
    reg, _, pipe, emb, got = rig(tmp_path, lambda i: [face(100)])
    person = reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT)
    vectors = iter([axis(0), axis(5), axis(0), axis(0)] + [axis(0)] * 20)
    emb.embed = lambda img, det: next(vectors).copy()
    run(pipe, 25, dt=0.2)
    rec = got["person.recognized"]
    assert len(rec) == 1 and rec[0].payload.person_id == person.person_id
    first_embed = got["percept.face"][0].stamp
    assert rec[0].stamp - first_embed <= 2.5
    assert rec[0].stamp == pytest.approx(1.4, abs=0.21)


def test_config_ties_burst_to_votes_and_embed_size_to_detection_size(tmp_path):
    with pytest.raises(ValueError, match="embedding_burst"):
        RoboEmotConfig.model_validate({"vision": {"embedding_burst": 2}})
    with pytest.raises(ValueError, match="embed_min_face_px"):
        RoboEmotConfig.model_validate({"vision": {"embed_min_face_px": 10, "face_min_px": 30}})
    from roboemot.registry import Matcher

    with pytest.raises(ValueError, match="embedding_burst"):
        VisionPipeline(
            CFG.vision.model_copy(update={"embedding_burst": 1}),
            CFG.identity,
            Bus(),
            Matcher(registry(tmp_path), CFG.identity),
            face_detector=ScriptedDetector(lambda i: []),
            face_embedder=FixedFaceEmbedder(axis(0)),
        )


def test_forgotten_person_stops_being_announced_or_reported(tmp_path):
    reg, _, pipe, _, got = rig(tmp_path, lambda i: [face(100)])
    person = reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT)
    run(pipe, 10)
    state = next(iter(pipe._faces.values()))
    assert state.owner_id == person.person_id and state.announced == person.person_id
    reg.forget(person.person_id)
    assert state.owner_id is None and state.announced is None


# ---------------------------------------------------------------------- publishing


def test_raising_subscriber_does_not_abort_the_frame_or_mark_it_reported(tmp_path, caplog):
    _, bus, pipe, _, got = rig(tmp_path, lambda i: [face(100)])
    fail = {"on": True}

    def bad(msg):
        if fail["on"]:
            raise RuntimeError(f"{SECRET}")

    bus.subscribe("percept.face", bad)
    caplog.set_level(logging.DEBUG)
    run(pipe, 3)
    assert len(got["percept.entity"]) == 2
    fail["on"] = False
    pipe.process(Frame(image(), 1.0))
    assert got["percept.face"][-1].payload.new_track
    assert SECRET not in caplog.text


def test_pipeline_uses_the_camera_provider(tmp_path):
    from roboemot.vision import CameraModel

    provider = lambda w, h: CameraModel(w, h, 78, fx=200.0, fy=200.0, cx=0.0, cy=0.0)  # noqa: E731
    _, _, pipe, _, got = rig(tmp_path, lambda i: [face(100, y=100, size=60)], camera_model=provider)
    run(pipe, 4)
    assert got["percept.face"][-1].payload.azimuth_rad < -0.5


# --------------------------------------------------------------------- logging, repr


def test_nothing_with_array_data_is_logged_or_shown_in_reprs(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)

    class Module:
        name = "boom"

        def process(self, frame):
            raise ValueError(f"{SECRET}")

    runner = LatestFrameRunner(Module())
    done = threading.Event()
    orig = Module.process

    def process(self, frame):
        try:
            orig(self, frame)
        finally:
            done.set()

    Module.process = process
    runner.start()
    runner.submit(Frame(image(), 0.0))
    assert done.wait(5)
    runner.stop()
    assert SECRET not in caplog.text

    _, _, pipe, _, _ = rig(tmp_path, lambda i: [face(100)])
    run(pipe, 3)
    state = next(iter(pipe._faces.values()))
    obs = Observation("animal", 0.0, "v", 0.0, 0.0, embedding=np.ones(4))
    for obj in (Frame(image(), 0.0), state, obs, face(1)):
        assert "array(" not in repr(obj) and "[[" not in repr(obj)


# ------------------------------------------------------------------------- replay


def test_replay_forces_refresh_off_and_restores_it(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    write_session(root, 24)
    session = load_session(root)
    near = (axis(0) + 0.3 * axis(1)) / np.linalg.norm(axis(0) + 0.3 * axis(1))
    reg, bus, pipe, _, _ = rig(tmp_path / "a", lambda i: [face(100)], vector=near)
    reg.enroll("Ada", np.stack([axis(0)] * 5), FACE_MODEL, consent=CONSENT)
    before = reg.gallery("face", FACE_MODEL)
    replay(session, pipe, bus)
    assert pipe.refresh is True
    assert {k: v.shape for k, v in reg.gallery("face", FACE_MODEL).items()} == {
        k: v.shape for k, v in before.items()
    }
    run(pipe, 12, start=100.0)
    grown = {k: v.shape for k, v in reg.gallery("face", FACE_MODEL).items()}
    assert grown != {k: v.shape for k, v in before.items()}


def test_session_frames_cannot_escape_the_session_folder(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    np.save(tmp_path / "outside.npy", image())
    (root / "session.json").write_text(
        json.dumps({"version": 1, "frames": [{"file": "../outside.npy", "stamp": 0.0}]})
    )
    with pytest.raises(ReplayError, match="leaves"):
        load_session(root).load(0)


def test_second_replay_needs_no_manual_reset(tmp_path):
    root = tmp_path / "s"
    root.mkdir()
    write_session(root, 30)
    session = load_session(root)
    _, bus, pipe, _, _ = rig(tmp_path / "r", lambda i: [face(100)])
    first = summarize(replay(session, pipe, bus).messages)
    second = summarize(replay(session, pipe, bus).messages)
    assert first and first == second


# --------------------------------------------------------------------- pump pacing


def test_pump_dedupes_identical_frames_and_never_spins():
    frame = Frame(np.zeros((16, 16, 3), np.uint8), 0.0)
    submitted: list[Frame] = []

    class Runner:
        def submit(self, f):
            submitted.append(f)

    class Stop:
        def __init__(self):
            self.waits: list[float] = []

        def is_set(self):
            return len(self.waits) >= 20

        def wait(self, timeout):
            self.waits.append(timeout)

    class Source:
        def read(self):
            return Frame(frame.image.copy(), 0.0)

    stop, now = Stop(), [0.0]
    pump(Source(), Runner(), stop, 10.0, clock=lambda: now[0])
    assert len(submitted) == 1
    assert all(w >= 0.1 - 1e-9 for w in stop.waits)

    stop2 = Stop()
    pump(type("S", (), {"read": lambda self: None})(), Runner(), stop2, 10.0, clock=lambda: 0.0)
    assert all(w > 0 for w in stop2.waits)


# --------------------------------------------------------------------------- fetcher

BODY = b"weights" * 50
FILE = ModelFile(
    path="d/m.onnx",
    url="https://example.test/m.onnx",
    sha256=__import__("hashlib").sha256(BODY).hexdigest(),
    size_bytes=len(BODY),
)


def mock(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_marks_verified_files_and_check_file_detects_changes(tmp_path):
    assert (
        fetch_file(FILE, tmp_path, mock(lambda r: httpx.Response(200, content=BODY)))
        == "downloaded"
    )
    assert check_file(tmp_path, FILE) == (tmp_path / "d/m.onnx").resolve()
    assert oct((tmp_path / "d").stat().st_mode & 0o777) == oct(0o700)
    (tmp_path / "d/m.onnx").write_bytes(BODY + b"x")
    with pytest.raises(ModelError, match="changed"):
        check_file(tmp_path, FILE)


def test_unmarked_files_are_adopted_by_fetch_and_refused_before(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d/m.onnx").write_bytes(BODY)
    with pytest.raises(ModelError, match="not verified"):
        check_file(tmp_path, FILE)
    no_net = mock(lambda r: pytest.fail("no download needed"))
    assert fetch_file(FILE, tmp_path, no_net) == "present"
    check_file(tmp_path, FILE)


def test_fetch_enforces_the_manifest_size(tmp_path):
    with pytest.raises(ModelError, match="larger"):
        fetch_file(FILE, tmp_path, mock(lambda r: httpx.Response(200, content=BODY * 2)))
    with pytest.raises(ModelError, match="size"):
        fetch_file(FILE, tmp_path, mock(lambda r: httpx.Response(200, content=BODY[:-1])))
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_refuses_redirects_to_http_but_follows_https(tmp_path):
    def to_http(req):
        return httpx.Response(302, headers={"location": "http://evil.test/m.onnx"})

    with pytest.raises(ModelError, match="https"):
        fetch_file(FILE, tmp_path, mock(to_http))

    def hop(req):
        if req.url.host == "example.test":
            return httpx.Response(302, headers={"location": "https://cdn.test/m.onnx"})
        return httpx.Response(200, content=BODY)

    assert fetch_file(FILE, tmp_path, mock(hop)) == "downloaded"


def test_stale_part_symlink_is_not_followed(tmp_path):
    (tmp_path / "d").mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    (tmp_path / "d/m.onnx.part").symlink_to(victim)
    fetch_file(FILE, tmp_path, mock(lambda r: httpx.Response(200, content=BODY)))
    assert victim.read_text() == "keep"
    assert (tmp_path / "d/m.onnx").read_bytes() == BODY


def test_os_errors_become_model_errors(tmp_path):
    (tmp_path / "d").write_text("a file where the folder should go")
    with pytest.raises(ModelError):
        fetch_file(FILE, tmp_path, mock(lambda r: httpx.Response(200, content=BODY)))


def test_cli_honours_vision_models_dir(tmp_path, capsys):
    app = tmp_path / "app.yaml"
    app.write_text(f"schema_version: 1\nvision:\n  models_dir: {tmp_path / 'm'}\n")
    assert cli.main(["models", "list", "--app", str(app)]) == 0
    assert str(tmp_path / "m") in capsys.readouterr().out


# ------------------------------------------------------------------- spike parity


def test_align_face_matches_the_spike_implementation():
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision import onnx

    spike = Path(__file__).resolve().parents[3] / "scripts" / "bench" / "bench_lib.py"
    if not spike.exists():
        pytest.skip("spike bench_lib.py not present")
    spec = importlib.util.spec_from_file_location("spike_bench_lib", spike)
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    rng = np.random.default_rng(1)
    img = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    kps = np.array(
        [[120, 100], [180, 102], [150, 130], [128, 160], [172, 161]], np.float32
    ) + rng.normal(0, 1.5, (5, 2)).astype(np.float32)
    assert np.array_equal(onnx.align_face(img, kps), bench.align(img, kps))


def test_load_stages_refuses_unverified_files(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("onnxruntime")
    from roboemot.vision import onnx

    with pytest.raises(ModelError, match="not verified"):
        onnx.load_stages(tmp_path, CFG.vision)


def test_fallback_camera_reads_the_world_fov(tmp_path):
    from roboemot.config.schema import WorldConfig

    world = WorldConfig(camera_fov_h_deg=60, camera_fov_v_deg=40)
    right_edge = lambda i: [face(320 - 60 - 0.01, y=90, size=60)]  # noqa: E731
    _, _, pipe, _, got = rig(tmp_path, right_edge, world=world)
    run(pipe, 4)
    p = got["percept.face"][-1].payload
    assert p.azimuth_rad == pytest.approx(
        -np.arctan(30 / 160 * 1.0 * np.tan(np.radians(30))) * 0
        - np.arctan((290 - 160) / (160 / np.tan(np.radians(30)))),
        abs=0.01,
    )
