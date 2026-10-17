"""M2 exit (SPEC 14), second half, app level: a replayed 24-hour light log moves the robot through
awake, drowsy and resting and back, through the real runner and motion loop (mocked SDK client)."""

from __future__ import annotations

import threading
import time

import numpy as np
from roboemot.circadian.motion import FX_SOURCE, REST_SOURCE
from roboemot.circadian.replay import synthetic_day
from roboemot.vision import Frame
from roboemot_reachymini.driver import ReachyMiniDriver
from test_app_circadian import fed  # noqa: F401  (a fixture)
from test_app_perception import (  # noqa: F401  (data is a fixture)
    AppWithFakes,
    FakeDetector,
    data,
    new_app,
    start,
    wait_for,
)

SPEED = 3600.0  # one virtual hour per real second: the 24 h log replays in about 24 s
START_HOUR = 8.0  # synthetic_day starts at 08:00
WALL_BUDGET_S = 60.0
SECONDS_PER_MINUTE = 60


class ReplayedDayClock:
    """Virtual time from 08:00 running SPEED times faster than real time once ``begin`` is
    called (frozen at 08:00 before: the app's start-up must not eat virtual hours)."""

    def __init__(self) -> None:
        self.origin = 0.0
        self.began: float | None = None

    def begin(self) -> None:
        self.began = time.monotonic()

    def virtual_s(self) -> float:
        return 0.0 if self.began is None else (time.monotonic() - self.began) * SPEED

    def monotonic(self) -> float:
        return self.origin + self.virtual_s()

    def hour(self) -> float:
        return (START_HOUR + self.virtual_s() / 3600) % 24


class DayApp(AppWithFakes):
    """The app on the replayed day: the clock is virtual and the camera shows the synthetic
    day's brightness for the virtual minute (``synthetic_day``, SPEC 7A Replay)."""

    clock: ReplayedDayClock
    samples = synthetic_day()

    def make_circadian_clock(self):
        self.clock = ReplayedDayClock()
        return self.clock

    def make_frame_source(self, mini, joint_state):
        app = self

        class DaySource:
            n = 0

            def read(self):
                self.n += 1
                minute = int((app.clock.monotonic() - app.clock.origin) / SECONDS_PER_MINUTE)
                luma = app.samples[min(minute, len(app.samples) - 1)].luma
                img = np.full((120, 160, 3), round(luma * 255), np.uint8)
                img[0, :4, :] = (img[0, :4, :] + self.n) % 250  # a different frame each time
                return Frame(img, time.monotonic(), None)

        return DaySource()


def test_a_replayed_24_hour_light_log_moves_the_robot_through_awake_drowsy_resting_awake(
    data,  # noqa: F811
    fed,  # noqa: F811
    monkeypatch,
):
    """The rest posture is held and the idle motion stopped for as long as the robot rests;
    waking plays the stretch and lets go of the posture."""
    monkeypatch.setattr(FakeDetector, "empty", True)  # nobody around: light and clock decide
    commanded: dict[str, float] = {}
    real_write = ReachyMiniDriver.write_pose_targets

    def write(self, t, values):
        commanded.update(values)
        real_write(self, t, values)

    monkeypatch.setattr(ReachyMiniDriver, "write_pose_targets", write)
    app = new_app(DayApp)
    spans: list[dict] = []  # per state: name, virtual hour of entry, range in the feed log

    def on_state(msg):  # also published periodically: a span starts when the state changes
        if spans and spans[-1]["state"] == msg.payload.state:
            return
        if spans:
            spans[-1]["end"] = len(fed)
        spans.append(
            {"state": msg.payload.state, "hour": msg.payload.hour, "begin": len(fed), "end": None}
        )

    def states() -> list[str]:
        return [s["state"] for s in spans]

    stop, thread, errors = start(app)
    feeder_stop = threading.Event()

    def feed_light() -> None:
        """One logged sample per virtual minute straight into the runner's sampler, as the
        core replay does: the camera path cannot deliver a frame per virtual minute at this
        speed (it would read a frame every few virtual hours)."""
        for sample in app.samples:
            while app.clock.virtual_s() < len(fed_samples) * SECONDS_PER_MINUTE:
                if feeder_stop.wait(0.002):
                    return
            fed_samples.append(sample)
            app.circadian_runner.sampler.offer_stats(sample.luma, sample.noise)

    fed_samples: list = []
    feeder = threading.Thread(target=feed_light, daemon=True)
    try:
        assert wait_for(lambda: app.circadian_runner is not None and app.diagnostics is not None)
        app.bus.subscribe("circadian.state", on_state)
        assert wait_for(lambda: app.circadian_runner.last is not None)  # the loop is ticking
        started = time.monotonic()
        app.clock.begin()
        feeder.start()
        assert wait_for(lambda: "resting" in states(), timeout=45), states()
        assert wait_for(lambda: commanded.get("head.pitch", 0.0) > 0.2, timeout=10)
        rest_pose = dict(commanded)
        assert wait_for(lambda: states()[-1] == "awake" and len(spans) >= 3, timeout=45), states()
        assert wait_for(lambda: commanded["head.pitch"] < 0.1, timeout=10)  # posture let go of
        wall = time.monotonic() - started
        feed_log = list(fed)
    finally:
        feeder_stop.set()
        stop.set()
        thread.join(15)
        feeder.join(5)
    assert not errors and not thread.is_alive()
    print("replayed day:", [(s["state"], round(s["hour"], 2)) for s in spans], f"wall={wall:.1f}s")
    sequence = states()
    assert sequence == ["awake", "drowsy", "resting", "drowsy", "awake"], sequence
    assert wall < WALL_BUDGET_S, wall
    hours = {(i, s["state"]): s["hour"] for i, s in enumerate(spans)}
    assert 21.0 <= hours[(1, "drowsy")] <= 21.6  # SPEC 7A: about 21:15
    assert 22.4 <= hours[(2, "resting")] <= 22.7  # the quiet hours start at 22:30
    assert 6.9 <= hours[(3, "drowsy")] % 24 <= 7.3  # leaves resting at 07:00, awake just after
    assert 6.9 <= hours[(4, "awake")] % 24 <= 7.3
    rest = next(s for s in spans if s["state"] == "resting")
    window = feed_log[rest["begin"] : rest["end"]]
    assert sum(c.source == REST_SOURCE for c in window) >= 50  # the posture is held, not once
    assert not any(c.source in ("idle", "idle-additive") for c in window)  # idle motion stopped
    assert rest_pose["head.pitch"] > 0.2  # head down
    assert rest_pose["antenna_right"] < 0 < rest_pose["antenna_left"]  # antennas droop outward
    assert any(c.source == FX_SOURCE for c in feed_log[rest["end"] :])  # waking: the stretch
