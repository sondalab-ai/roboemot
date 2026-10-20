"""SPEC 4A, 11, 12A: the Reachy Mini platform behind ``roboemot.robots`` builds core's parts."""

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.drivers import PoseGroupDriver
from roboemot.robot_platform import (
    AudioHub,
    RobotPlatform,
    RobotPlatformError,
    Speaker,
    load_robot_platform,
)
from roboemot.voice.activity import SpeechActivity
from roboemot_reachymini import gaze
from roboemot_reachymini.platform import ReachyMiniPlatform


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


def test_the_driver_is_a_pose_driver_that_keeps_automatic_body_yaw_off():
    mini = Mini()
    assert isinstance(ReachyMiniPlatform().driver(mini), PoseGroupDriver)
    assert mini.body_yaw is False


def test_audio_and_speaker_take_their_settings_from_the_config():
    cfg = RoboEmotConfig()
    platform = ReachyMiniPlatform()
    audio = platform.audio(object(), cfg, None)
    assert isinstance(audio, AudioHub)
    assert isinstance(platform.speaker(object(), cfg, SpeechActivity(0.0), None), Speaker)


def test_require_runtime_fails_loud_without_a_supported_sdk(monkeypatch):
    from roboemot_reachymini import platform as platform_module

    monkeypatch.setattr(platform_module, "installed_version", lambda: None)
    with pytest.raises(RuntimeError, match="not installed"):
        ReachyMiniPlatform().require_runtime()
