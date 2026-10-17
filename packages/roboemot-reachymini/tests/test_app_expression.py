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


def face(group, source, owner=None, minor=True) -> FacePercept:
    now = time.monotonic()
    return FacePercept(
        track_id=1, stamp=now, box=(0.4, 0.3, 0.6, 0.6), score=0.9, azimuth_rad=0.0,
        elevation_rad=0.0, owner_id=owner, anonymous_id=None if owner else "anon",
        age_group=group, age_source=source, gender="unknown", gender_source="none",
        attribute_confidence=0.9, possible_minor=minor, new_track=True,
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


def scenario(data, fed, group, source, owner, minor):  # noqa: F811
    """Show one face, let the robot feel joy, return (register, amplitude, expression commands)."""
    app = new_app()
    stop, thread, errors = start(app)
    published: list = []
    try:
        assert wait_for(lambda: app.bus is not None and app.diagnostics is not None)
        app.bus.subscribe("register.active", lambda m: published.append(m.payload))
        assert wait_for(lambda: app.register_tracker.active is not None)
        app.bus.publish("percept.face", face(group, source, owner, minor), time.monotonic())
        assert wait_for(lambda: published and published[-1].group == group), published
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
    return published[-1], report, amplitude, commands


def test_a_known_child_gets_the_child_register_and_antenna_motion(data, fed):  # noqa: F811
    reg, report, amplitude, commands = scenario(data, fed, "child", "declared", PERSON, True)
    assert reg.group == "child" and reg.params.motion_amplitude == pytest.approx(1.2)
    assert report["group"] == "child" and report["approach_allowed"] is False
    assert "persona_fragment" not in report
    cmd = commands[0]
    assert cmd.priority == Priority.EXPRESSION and cmd.additive
    assert {"antenna_right", "antenna_left"} <= set(cmd.targets)
    assert "body_yaw" not in cmd.targets and not any("body_yaw" in c.targets for c in commands)
    assert amplitude > 0


def test_a_stranger_gets_the_unknown_register_with_a_lower_amplitude(data, fed):  # noqa: F811
    child = scenario(data, fed, "child", "declared", PERSON, True)
    fed.clear()
    stranger = scenario(data, fed, "unknown", "none", None, True)
    fed.clear()
    adult = scenario(data, fed, "adult", "declared", PERSON, False)
    assert stranger[0].group == "unknown" and stranger[0].params.motion_amplitude == 0.7
    assert adult[0].group == "adult"
    assert child[2] > adult[2] > stranger[2] > 0  # amplitude before the minor cap


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
