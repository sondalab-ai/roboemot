"""Loop rates come from config, a manifest may set them (SPEC 4A, 11A)."""

from pathlib import Path

import pytest
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import RoboEmotConfig
from roboemot.manifest import load_manifest

REPO = Path(__file__).resolve().parents[3]
REACHY = REPO / "configs" / "manifests" / "reachy-mini-wireless.yaml"


def test_defaults_are_the_library_rates():
    motion = RoboEmotConfig().motion
    assert motion.arbiter.loop_hz == 100.0
    assert motion.app_tick_hz == 20.0


def test_the_reachy_manifest_keeps_its_50_hz_stream():
    manifest = load_manifest(REACHY, allow_unmeasured_dynamics=True)
    stack = ConfigStack()
    stack.set_manifest_behaviour(manifest.model.behaviour, manifest.channels)
    motion = stack.effective().motion
    assert motion.arbiter.loop_hz == 50.0
    assert motion.app_tick_hz == 20.0


def test_a_manifest_may_set_both_rates():
    stack = ConfigStack()
    stack.set_manifest_behaviour(
        {"motion": {"app_tick_hz": 10, "arbiter": {"loop_hz": 200}}}, ("head.yaw",)
    )
    motion = stack.effective().motion
    assert (motion.app_tick_hz, motion.arbiter.loop_hz) == (10.0, 200.0)


@pytest.mark.parametrize(
    "motion", [{"idle": {}}, {"arbiter": {"crossfade_ms": 0}}, {"arbiter": []}, "fast"]
)
def test_a_manifest_may_set_nothing_else_under_motion(motion):
    with pytest.raises(ConfigError, match="manifest behaviour motion"):
        ConfigStack().set_manifest_behaviour({"motion": motion}, ("head.yaw",))


def test_a_rate_outside_the_range_is_refused():
    with pytest.raises(ConfigError):
        ConfigStack().set_manifest_behaviour({"motion": {"app_tick_hz": 0}}, ("head.yaw",))
