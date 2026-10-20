"""SPEC 4A, 11, 12A: core builds every robot part through the ``RobotPlatform`` protocol and loads
the platform of the manifest's ``robot.id`` from the ``roboemot.robots`` entry-point group, failing
loud when there is none. A fake non-Reachy platform (desk-head-v1) proves nothing else is needed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from roboemot import robot_platform
from roboemot.attention import GazeMap
from roboemot.config.schema import RoboEmotConfig
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import load_manifest
from roboemot.plugins import GROUPS, contract_violation
from roboemot.reflex import ReflexMap
from roboemot.robot_platform import (
    AudioHub,
    RobotPlatform,
    RobotPlatformError,
    Speaker,
    load_robot_platform,
)
from roboemot.vision.runner import FrameSource
from roboemot.voice.activity import SpeechActivity
from roboemot.world import FrameSpec

MANIFESTS = Path(__file__).resolve().parents[3] / "configs" / "manifests"
DESK = MANIFESTS / "desk-head-v1.yaml"


class DeskCamera:
    def read(self):
        return None

    def camera_model(self, width, height):
        return None


class DeskAudio:
    def __init__(self, on_level):
        self.on_level = on_level
        self.voice = True

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_voice(self, enabled: bool) -> None:
        self.voice = enabled

    def snapshot(self):
        return {"state": "off"}

    def read(self, timeout: float = 0.0):
        return None


class DeskSpeaker:
    def push_audio_sample(self, samples, rate_hz: int) -> float:
        return 0.0

    def stop(self) -> None: ...
    def start_playing(self) -> bool:
        return True

    def close(self) -> None: ...


class DeskPlatform:
    """A non-Reachy platform: every part comes from core protocols and the desk manifest."""

    def require_runtime(self) -> None: ...

    def driver(self, client):
        return MockPoseDriver(home={"neck_pan": 0.0, "neck_tilt": 0.0})

    def camera(self, client, joint_state):
        return DeskCamera()

    def audio(self, client, cfg, on_level):
        return DeskAudio(on_level)

    def speaker(self, client, cfg, activity, reference):
        return DeskSpeaker()

    def gaze_map(self):
        return GazeMap(head_yaw="neck_pan", head_pitch="neck_tilt")

    def frame_spec(self):
        return FrameSpec(yaw_dofs=("neck_pan",), pitch_dofs=("neck_tilt",), roll_dofs=())

    def reflex_map(self):
        return ReflexMap(head_yaw="neck_pan", head_pitch="neck_tilt")

    def listener(self, *args, **kwargs):
        return None


def entry_point(name: str, loaded):
    return SimpleNamespace(name=name, load=lambda: loaded)


@pytest.fixture
def installed(monkeypatch):
    def install(*eps):
        monkeypatch.setattr(robot_platform, "entry_points", lambda group: list(eps))

    return install


def test_the_group_is_registered_with_the_protocol_as_its_contract():
    assert GROUPS["roboemot.robots"].protocol is RobotPlatform
    assert contract_violation("roboemot.robots", DeskPlatform) is None
    assert contract_violation("roboemot.robots", int) is not None


def test_core_obtains_every_part_from_a_non_reachy_platform(installed):
    installed(entry_point("desk-head-v1", DeskPlatform))
    manifest = load_manifest(DESK)
    platform = load_robot_platform(manifest.robot.id)
    cfg = RoboEmotConfig()
    platform.require_runtime()
    assert platform.driver(object()) is not None
    camera: FrameSource = platform.camera(object(), dict)
    assert camera.read() is None and camera.camera_model(640, 480) is None
    audio = platform.audio(object(), cfg, None)
    assert isinstance(audio, AudioHub)
    assert isinstance(platform.speaker(object(), cfg, SpeechActivity(0.0), None), Speaker)
    dofs = {d for d in manifest.dofs}
    assert {platform.gaze_map().head_yaw, platform.gaze_map().head_pitch} <= dofs
    assert set(platform.frame_spec().yaw_dofs + platform.frame_spec().pitch_dofs) <= dofs


def test_an_unknown_robot_id_names_the_id_and_the_installed_robots(installed):
    installed(entry_point("desk-head-v1", DeskPlatform), entry_point("another", DeskPlatform))
    with pytest.raises(RobotPlatformError, match=r"'ghost'.*installed: another, desk-head-v1"):
        load_robot_platform("ghost")


def test_no_installed_robot_at_all_is_an_error_too(installed):
    installed()
    with pytest.raises(RobotPlatformError, match=r"'ghost'.*installed: none"):
        load_robot_platform("ghost")


def test_a_platform_that_breaks_the_protocol_is_refused(installed):
    installed(entry_point("desk-head-v1", type("Half", (), {"driver": lambda self, client: None})))
    with pytest.raises(RobotPlatformError, match="does not satisfy RobotPlatform"):
        load_robot_platform("desk-head-v1")


def test_a_platform_that_fails_to_load_is_not_swallowed(installed):
    def boom():
        raise ImportError("no sdk")

    installed(SimpleNamespace(name="desk-head-v1", load=boom))
    with pytest.raises(RobotPlatformError, match="cannot be loaded: ImportError: no sdk"):
        load_robot_platform("desk-head-v1")
