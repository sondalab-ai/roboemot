"""SPEC 4A, 11, 12A: the Reachy Mini platform behind ``roboemot.robots`` builds core's parts."""

from unittest.mock import MagicMock

import pytest
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.drivers import PoseGroupDriver
from roboemot.robot_platform import (
    AudioHub,
    BearingSource,
    RobotPlatform,
    RobotPlatformError,
    RobotSession,
    Speaker,
    load_robot_platform,
)
from roboemot.voice.activity import SpeechActivity
from roboemot_reachymini import gaze
from roboemot_reachymini.platform import ReachyMiniPlatform, ReachyMiniSession
from test_app_perception import data  # noqa: F401  (data: fixture)


class Mini:
    def set_automatic_body_yaw(self, enabled: bool) -> None:
        self.body_yaw = enabled


def test_the_manifest_robot_id_loads_the_reachy_platform():
    platform = load_robot_platform("reachy-mini-wireless")
    assert isinstance(platform, ReachyMiniPlatform)
    assert isinstance(platform, RobotPlatform)


def test_an_unknown_robot_id_is_an_error_naming_the_reachy_as_installed():
    with pytest.raises(RobotPlatformError, match=r"'ghost'.*reachy-mini-wireless"):
        load_robot_platform("ghost")


def test_the_maps_are_the_reachy_ones():
    platform = ReachyMiniPlatform()
    assert platform.gaze_map() is gaze.GAZE
    assert platform.frame_spec() is gaze.FRAME
    assert platform.reflex_map() is gaze.REFLEX


def test_bind_gives_a_session_that_owns_the_client():
    mini = Mini()
    session = ReachyMiniPlatform().bind(mini)
    assert isinstance(session, ReachyMiniSession)
    assert isinstance(session, RobotSession)
    assert session.client is mini


def test_the_driver_is_a_pose_driver_that_keeps_automatic_body_yaw_off():
    mini = Mini()
    assert isinstance(ReachyMiniSession(mini).driver(), PoseGroupDriver)
    assert mini.body_yaw is False


def test_audio_speaker_and_bearing_take_their_settings_from_the_config():
    cfg = RoboEmotConfig()
    session = ReachyMiniSession(object())
    assert isinstance(session.audio(cfg, None), AudioHub)
    assert isinstance(session.speaker(cfg, SpeechActivity(0.0), None), Speaker)
    assert isinstance(session.bearing_source(Bus()), BearingSource)


def test_require_runtime_fails_loud_without_a_supported_sdk(monkeypatch):
    from roboemot_reachymini import platform as platform_module

    monkeypatch.setattr(platform_module, "installed_version", lambda: None)
    with pytest.raises(RuntimeError, match="not installed"):
        ReachyMiniPlatform().require_runtime()


def test_an_unregistered_robot_id_stops_the_app_before_any_part_is_built(data, monkeypatch):  # noqa: F811
    from roboemot_reachymini import app as app_module
    from roboemot_reachymini import platform as platform_module

    manifest = data / "reachy-mini-wireless.yaml"
    manifest.write_text(manifest.read_text().replace("id: reachy-mini-wireless", "id: ghost", 1))
    built = []
    monkeypatch.setattr(platform_module, "ReachyMiniDriver", lambda client: built.append(client))
    mini = MagicMock()
    with pytest.raises(RobotPlatformError, match=r"'ghost'.*reachy-mini-wireless"):
        app_module.RoboEmotApp().run(mini, None)
    assert built == [] and mini.mock_calls == []
