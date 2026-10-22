"""SPEC 4A, 11, 12A: core builds every robot part through the ``RobotPlatform`` protocol and loads
the platform of the manifest's ``robot.id`` from the ``roboemot.robots`` entry-point group, failing
loud when there is none. A fake non-Reachy platform (desk-head-v1) proves nothing else is needed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from roboemot import robot_platform
from roboemot.attention import GazeMap
from roboemot.bus import Bus
from roboemot.config.schema import RoboEmotConfig
from roboemot.drivers import MockPoseDriver
from roboemot.manifest import load_manifest
from roboemot.plugins import GROUPS, contract_violation
from roboemot.reflex import ReflexMap
from roboemot.robot_platform import (
    AudioHub,
    BearingSource,
    RobotPlatform,
    RobotPlatformError,
    RobotSession,
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
    def apply_config(self, cfg) -> None:
        pass

    def set_voice(self, enabled: bool) -> None:
        self.voice = enabled

    def snapshot(self):
        return {"state": "off"}

    def read(self, timeout: float = 0.0):
        return None


class DeskSpeaker:
    def push_audio_sample(self, samples, rate_hz: int) -> float:
        return 0.0

    def playing_until(self) -> float:
        return 0.0

    def end_stream(self) -> float:
        return 0.0

    def stop(self) -> None: ...
    def start_playing(self) -> bool:
        return True

    def close(self) -> None: ...


class DeskBearing:
    def get(self, now, max_age_s):
        return None

    def close(self) -> None: ...


class DeskSession:
    def __init__(self, client):
        self.client = client

    def driver(self):
        return MockPoseDriver(home={"neck_pan": 0.0, "neck_tilt": 0.0})

    def camera(self, joint_state):
        return DeskCamera()

    def audio(self, cfg, on_level):
        return DeskAudio(on_level)

    def speaker(self, cfg, activity, reference):
        return DeskSpeaker()

    def bearing_source(self, bus):
        return DeskBearing()


class DeskPlatform:
    """A non-Reachy platform: every part comes from core protocols and the desk manifest."""

    def require_runtime(self) -> None: ...

    def bind(self, client):
        return DeskSession(client)

    def gaze_map(self):
        return GazeMap(head_yaw="neck_pan", head_pitch="neck_tilt")

    def frame_spec(self):
        return FrameSpec(yaw_dofs=("neck_pan",), pitch_dofs=("neck_tilt",), roll_dofs=())

    def reflex_map(self):
        return ReflexMap(head_yaw="neck_pan", head_pitch="neck_tilt")


def entry_point(name: str, loaded, dist: str = "desk-pkg"):
    return SimpleNamespace(name=name, load=lambda: loaded, dist=SimpleNamespace(name=dist))


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
    session = platform.bind(object())
    assert isinstance(session, RobotSession)
    assert session.driver() is not None
    camera: FrameSource = session.camera(dict)
    assert camera.read() is None and camera.camera_model(640, 480) is None
    assert isinstance(session.audio(cfg, None), AudioHub)
    assert isinstance(session.speaker(cfg, SpeechActivity(0.0), None), Speaker)
    assert isinstance(session.bearing_source(Bus()), BearingSource)
    dofs = set(manifest.dofs)
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

    installed(SimpleNamespace(name="desk-head-v1", load=boom, dist=None))
    with pytest.raises(
        RobotPlatformError, match="cannot be loaded: ImportError: no sdk.*SDK extra"
    ):
        load_robot_platform("desk-head-v1")


def test_another_load_failure_gets_no_install_hint(installed):
    def boom():
        raise ValueError("bad")

    installed(SimpleNamespace(name="desk-head-v1", load=boom, dist=None))
    with pytest.raises(RobotPlatformError, match="ValueError: bad$"):
        load_robot_platform("desk-head-v1")


def test_two_packages_registering_the_same_robot_id_are_refused(installed):
    installed(
        entry_point("desk-head-v1", DeskPlatform, dist="zeta-robot"),
        entry_point("desk-head-v1", DeskPlatform, dist="alpha-robot"),
    )
    with pytest.raises(RobotPlatformError, match=r"'desk-head-v1'.*alpha-robot, zeta-robot"):
        load_robot_platform("desk-head-v1")


def test_a_speaker_without_a_playback_timeline_is_not_a_speaker():
    class NoTimeline:
        def push_audio_sample(self, samples, rate_hz: int) -> float:
            return 0.0

        def stop(self) -> None: ...
        def start_playing(self) -> bool:
            return True

        def close(self) -> None: ...

    assert isinstance(DeskSpeaker(), Speaker) and not isinstance(NoTimeline(), Speaker)
