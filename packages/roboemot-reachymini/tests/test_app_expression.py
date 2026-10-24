"""The register and the expression renderer wired into the app (SPEC 7, 10, 12, 12A)."""

from __future__ import annotations

import json
import time

import pytest
from roboemot.motion import MotionLoop, Priority
from roboemot.vision.types import FacePercept
from roboemot_reachymini import app as app_module
from test_app_perception import data, new_app, start, wait_for  # noqa: F401  (data is a fixture)

PERSON = "p-secret-id"


def face(owner=None) -> FacePercept:
    now = time.monotonic()
    return FacePercept(
        track_id=1, stamp=now, box=(0.4, 0.3, 0.6, 0.6), score=0.9, azimuth_rad=0.0,
        elevation_rad=0.0, owner_id=owner, anonymous_id=None if owner else "anon",
        new_track=True,
    )  # fmt: skip


@pytest.fixture()
def fed(monkeypatch):
    """Every command the app feeds its motion loop, by source."""
    log: list = []
    real = MotionLoop.feed

    def feed(self, cmd):
        log.append(cmd)
        real(self, cmd)

    monkeypatch.setattr(MotionLoop, "feed", feed)
    return log


def scenario(data, fed, owner):  # noqa: F811
    """Show one face, let the robot feel joy, return (register, amplitude, expression commands)."""
    app = new_app()
    stop, thread, errors = start(app)
    published: list = []
    try:
        assert wait_for(lambda: app.bus is not None and app.diagnostics is not None)
        app.bus.subscribe("register.active", lambda m: published.append(m.payload))
        assert wait_for(lambda: app.register_tracker.active is not None)
        if owner:  # a record the registry can read
            owner = app.perception.registry.create_pending_person("Kid").person_id
        app.bus.publish("percept.face", face(owner), time.monotonic())
        assert wait_for(lambda: published and published[-1].people == 1), published
        app.affect_runner.engine.impulse(time.monotonic(), 0.3, 0.3, "joy")
        assert wait_for(lambda: app.expression.last_label == "joy"), "no expression started"
        assert wait_for(lambda: any(c.source == "expression" for c in fed))
        report = app.diagnostics.report()["sections"]["roboemot.register"]["data"]
        amplitude = app.expression.last_amplitude
        commands = [c for c in fed if c.source == "expression"]
    finally:
        stop.set()
        thread.join(15)
    assert not errors and not thread.is_alive()
    assert PERSON not in json.dumps(report) and PERSON not in repr(published)
    assert not owner or (owner not in json.dumps(report) and owner not in repr(published))
    return published[-1], report, amplitude, commands


def test_a_face_gets_the_one_register_and_antenna_motion(data, fed):  # noqa: F811
    reg, report, amplitude, commands = scenario(data, fed, PERSON)
    assert reg.params.motion_amplitude == pytest.approx(1.0) and reg.params.approach_allowed
    assert report["people"] == 1 and report["approach_allowed"] is True
    assert "persona_fragment" not in report and "group" not in report
    cmd = commands[0]
    assert cmd.priority == Priority.EXPRESSION and cmd.additive
    assert {"antenna_right", "antenna_left"} <= set(cmd.targets)
    assert amplitude > 0


def test_a_known_person_and_a_stranger_get_the_same_register(data, fed):  # noqa: F811
    """Owner decision 2026-10-23: no age groups, one register and one amplitude for everyone."""
    known = scenario(data, fed, PERSON)
    fed.clear()
    stranger = scenario(data, fed, None)
    assert known[0].params == stranger[0].params
    assert known[2] > 0 and stranger[2] > 0


def test_expression_settings_reload_without_a_restart(data):  # noqa: F811
    app = new_app(app_module.RoboEmotApp)
    stop, thread, errors = start(app)
    try:
        assert wait_for(lambda: app.expression is not None and app.diagnostics is not None)
        assert app.expression.cfg.intensity_factor.high == 1.3
        assert "joy" in app.expression.expression_map
        (data / "app.yaml").write_text(
            (data / "app.yaml").read_text()
            + "expression:\n  intensity_factor: {high: 0.9}\n"
            + "expression_map:\n  joy:\n    channels:\n"
            + "      antenna_right: [[0.0, 0], [0.225, 5], [0.45, 0]]\n"
        )
        assert wait_for(lambda: app.expression.cfg.intensity_factor.high == 0.9)
        assert app.expression.expression_map["joy"].repeat == 3  # merged with the manifest's
        joy = app.expression.expression_map["joy"]
        assert joy.channels["antenna_right"][1] == (0.225, 5)
        assert set(joy.channels) == {"antenna_right", "antenna_left"}
    finally:
        stop.set()
        thread.join(15)
    assert not errors
