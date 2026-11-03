"""SPEC 12A, 13: per-stage timing of the vision pipeline in the debug report."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from roboemot.bus import Bus
from roboemot.config.schema import TIMING_WINDOW_MAX_S, RoboEmotConfig
from roboemot.registry import FileStore, Matcher, Registry
from roboemot.vision import Detection, Frame, VisionPipeline
from roboemot.vision.service import open_perception
from roboemot.vision.stage_timing import (
    APPEARANCE,
    DETECTION,
    EMBEDDING,
    OBJECTS,
    PROCESS,
    STAGES,
    StageTimer,
)
from roboemot.webui.api import holds_vector
from test_perception import Source, fake_stages

CFG = RoboEmotConfig()
W, H = 320, 240


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def timer(window_max_s: float = TIMING_WINDOW_MAX_S) -> tuple[StageTimer, Clock, Clock]:
    wall, cpu = Clock(100.0), Clock(0.0)
    return StageTimer(window_max_s, wall_clock=wall, cpu_clock=cpu), wall, cpu


def run(t: StageTimer, wall: Clock, cpu: Clock, stage: str, wall_s: float, cpu_s: float):
    w0, c0 = t.start()
    wall.t += wall_s
    cpu.t += cpu_s
    t.record(stage, w0, c0)


def test_percentiles_rate_and_cpu_share_from_scripted_durations():
    t, wall, cpu = timer()
    durations_ms = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for ms in durations_ms:
        run(t, wall, cpu, DETECTION, ms / 1000, ms / 2000)  # half of each run on this thread
        wall.t += 1.0 - ms / 1000  # one run per second
    snap = t.snapshot(window_s=30.0)
    det = snap[DETECTION]
    assert det["runs"] == 10
    assert det["wall_ms"]["p50"] == pytest.approx(50)
    assert det["wall_ms"]["p90"] == pytest.approx(90)
    assert det["wall_ms"]["max"] == pytest.approx(100)
    assert det["span_s"] == pytest.approx(10.0)  # less than the window since the timer started
    assert det["runs_per_s"] == pytest.approx(1.0)
    assert det["cpu_share"] == pytest.approx(sum(durations_ms) / 2000 / 10.0)
    assert snap["window_s"] == 30.0


def test_only_runs_inside_the_window_count():
    t, wall, cpu = timer()
    for _ in range(10):
        run(t, wall, cpu, DETECTION, 0.5, 0.5)  # old and slow
        wall.t += 0.5
    wall.t += 20.0
    for _ in range(4):
        run(t, wall, cpu, DETECTION, 0.01, 0.01)
        wall.t += 0.99
    det = t.snapshot(window_s=5.0)[DETECTION]
    assert det["runs"] == 14
    assert det["span_s"] == pytest.approx(5.0)
    assert det["runs_per_s"] == pytest.approx(4 / 5.0)
    assert det["wall_ms"]["max"] == pytest.approx(10)
    assert det["cpu_share"] == pytest.approx(0.04 / 5.0)


def test_a_stage_that_never_ran_reports_zero_runs_and_null_percentiles():
    t, wall, _ = timer()
    wall.t += 3.0
    snap = t.snapshot(window_s=30.0)
    assert set(STAGES) <= set(snap)
    for stage in STAGES:
        entry = snap[stage]
        assert entry["runs"] == 0 and entry["runs_per_s"] == 0.0 and entry["cpu_share"] == 0.0
        assert entry["wall_ms"] == {"p50": None, "p90": None, "max": None}


def test_a_wrapped_ring_reports_the_span_it_covers():
    t, wall, cpu = timer(window_max_s=0.05)  # ring of ceil(0.05 * MAX_RATE_HZ) + 1 samples
    capacity = t._capacity
    for _ in range(capacity * 3):
        run(t, wall, cpu, EMBEDDING, 0.001, 0.001)
        wall.t += 0.009  # one run every 10 ms
    entry = t.snapshot(window_s=30.0)[EMBEDDING]
    assert entry["runs"] == capacity * 3
    assert entry["span_s"] == pytest.approx(capacity * 0.010, rel=1e-6)
    assert entry["runs_per_s"] == pytest.approx(100.0, rel=1e-6)


def test_reset_forgets_every_sample():
    t, wall, cpu = timer()
    run(t, wall, cpu, PROCESS, 0.01, 0.01)
    t.reset()
    assert t.snapshot(window_s=30.0)[PROCESS]["runs"] == 0


def test_window_setting_has_a_default_and_a_range():
    assert CFG.vision.timing_window_s == 30.0
    RoboEmotConfig(vision={"timing_window_s": TIMING_WINDOW_MAX_S})
    for bad in (1.0, TIMING_WINDOW_MAX_S + 1):
        with pytest.raises(ValidationError):
            RoboEmotConfig(vision={"timing_window_s": bad})


# ----------------------------------------------------------------- the pipeline


class Detector:
    def detect(self, img):
        return [Detection((40, 40, 120, 120), 0.95, landmarks=np.zeros((5, 2)))]


class Embedder:
    model_id = "mock-face"

    def embed(self, img, face):
        v = np.zeros(8, np.float32)
        v[0] = 1.0
        return v


def test_the_pipeline_times_each_stage_it_runs(tmp_path):
    reg = Registry(FileStore(tmp_path / "reg", allow_plaintext=True), CFG.identity)
    objects = SimpleNamespace(detect=lambda img: [])
    pipe = VisionPipeline(
        CFG.vision,
        CFG.identity,
        Bus(),
        Matcher(reg, CFG.identity),
        face_detector=Detector(),
        face_embedder=Embedder(),
        object_detector=objects,
    )
    for i in range(4):
        pipe.process(Frame(np.zeros((H, W, 3), np.uint8), i * 1.0))
    timing = pipe.stats()["stage_timing"]
    assert timing[PROCESS]["runs"] == 4
    assert timing[DETECTION]["runs"] == 4
    assert timing[OBJECTS]["runs"] == 4
    assert timing[EMBEDDING]["runs"] >= 1
    assert timing[APPEARANCE]["runs"] == 0
    assert timing[PROCESS]["wall_ms"]["max"] >= timing[DETECTION]["wall_ms"]["p50"]


def _arrays(value, path=""):
    """Paths of every list or tuple in ``value``."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _arrays(v, f"{path}.{k}")]
    if isinstance(value, list | tuple):
        return [path]
    return []


def test_the_debug_report_holds_stage_timing_numbers_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOEMOT_DATA_DIR", str(tmp_path / "data"))
    p = open_perception(CFG, Bus(), Source(), load_stages=fake_stages)
    snap = p.vision_snapshot()
    timing = snap["stage_timing"]
    assert "stage_drops" in snap
    for stage in STAGES:
        assert set(timing[stage]) == {"runs", "runs_per_s", "wall_ms", "cpu_share", "span_s"}
    assert _arrays(timing) == []
    assert not holds_vector(snap)
    json.dumps(snap)
