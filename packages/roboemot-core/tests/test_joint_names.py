"""SPEC 4A, 7, 11A, 12A: joint and channel names come from the loaded manifest, not from core.

The Reachy Mini numbers below were captured before the names left core; they must not move.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from roboemot.affect import AffectState
from roboemot.affect.clips import load_clip_library
from roboemot.affect.expression import ExpressionRenderer, build_primitive
from roboemot.affect.register import Person, resolve_register
from roboemot.config import ConfigStack
from roboemot.config.schema import ExpressionEntry
from roboemot.config.stack import ConfigError
from roboemot.health.responder import OfflineResponder
from roboemot.manifest import load_manifest

MANIFESTS = Path(__file__).resolve().parents[3] / "configs" / "manifests"
REACHY = MANIFESTS / "reachy-mini-wireless.yaml"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(REACHY, allow_unmeasured_dynamics=True)


@pytest.fixture(scope="module")
def cfg(manifest):
    stack = ConfigStack()
    stack.set_manifest_behaviour(
        manifest.model.behaviour, manifest.channels, load_clip_library(manifest)
    )
    return stack.effective()


def make_responder(cfg, manifest) -> OfflineResponder:
    renderer = ExpressionRenderer(
        cfg.expression,
        cfg.expression_map,
        manifest.dofs,
        0.15,
        cfg.gesture_map,
        body_yaw="body_yaw",
        clips=load_clip_library(manifest),
    )
    return OfflineResponder(cfg.connectivity, renderer, "it")


# ------------------------------------------------------------------ characterization (Reachy)

# (time s, {dof: radians}) of the additive gesture, pinned before the amplitudes moved to the
# manifest. Rise to 30% of 1.2 s, hold to 70%, back to zero at 1.2 s.
PERK_UP = [
    (0.0, {"head.pitch": 0.0, "antenna_right": 0.0, "antenna_left": 0.0}),
    (0.18, {"head.pitch": -0.04, "antenna_right": 0.2, "antenna_left": -0.2}),
    (0.36, {"head.pitch": -0.08, "antenna_right": 0.4, "antenna_left": -0.4}),
    (0.6, {"head.pitch": -0.08, "antenna_right": 0.4, "antenna_left": -0.4}),
    (0.84, {"head.pitch": -0.08, "antenna_right": 0.4, "antenna_left": -0.4}),
    (1.02, {"head.pitch": -0.04, "antenna_right": 0.2, "antenna_left": -0.2}),
]
PUZZLED = [
    (0.0, {"head.roll": 0.0, "antenna_right": 0.0, "antenna_left": 0.0}),
    (0.18, {"head.roll": 0.075, "antenna_right": -0.2, "antenna_left": 0.2}),
    (0.36, {"head.roll": 0.15, "antenna_right": -0.4, "antenna_left": 0.4}),
    (0.6, {"head.roll": 0.15, "antenna_right": -0.4, "antenna_left": 0.4}),
    (0.84, {"head.roll": 0.15, "antenna_right": -0.4, "antenna_left": 0.4}),
    (1.02, {"head.roll": 0.075, "antenna_right": -0.2, "antenna_left": 0.2}),
]


def check_trajectory(responder: OfflineResponder, source: str, expected) -> None:
    for t, want in expected:
        (cmd,) = responder.commands(t)
        assert cmd.source == source and cmd.additive and cmd.priority.name == "EXPRESSION"
        assert cmd.targets == pytest.approx(want, abs=1e-5)
    assert responder.commands(1.2) == []


def test_reachy_perk_up_trajectory_is_unchanged(cfg, manifest):
    r = make_responder(cfg, manifest)
    r.on_event("link.robot.restored", 0.0)
    check_trajectory(r, "connectivity:perk_up", PERK_UP)


def test_reachy_puzzled_trajectory_is_unchanged(cfg, manifest):
    r = make_responder(cfg, manifest)
    r.speech_while_offline(0.0)
    check_trajectory(r, "connectivity:puzzled", PUZZLED)


BODY_SWEEP = ExpressionEntry(
    channels={"body_yaw": [[0, 0], [0.5, 30], [1.0, 0]], "head.yaw": [[0, 0], [0.5, 10], [1.0, 0]]}
)


def play_sweep(cfg, manifest, group: str, **kw):
    reg = resolve_register([Person(group, "declared", {})], cfg.register, cfg.persona)
    r = ExpressionRenderer(cfg.expression, {"joy": BODY_SWEEP}, manifest.dofs, 0.15, **kw)
    r.update(AffectState(0.0, 0.0, "calm", 0.5, 0), reg, 0.0)
    r.request("joy")
    assert r.update(AffectState(0.0, 0.0, "calm", 0.5, 0), reg, 10.0)
    return reg, [r.commands(t)[0].targets for t in (10.25, 10.5, 10.75)]


def test_reachy_body_yaw_expression_is_unchanged_when_approach_is_allowed(cfg, manifest):
    reg, out = play_sweep(cfg, manifest, "adult", body_yaw="body_yaw")
    assert reg.params.approach_allowed
    assert [o["body_yaw"] for o in out] == pytest.approx([0.19634954, 0.39269908, 0.19634954])
    assert [o["head.yaw"] for o in out] == pytest.approx([0.06544985, 0.13089969, 0.06544985])


def test_reachy_body_yaw_is_dropped_when_approach_is_not_allowed(cfg, manifest):
    reg, out = play_sweep(cfg, manifest, "child", body_yaw="body_yaw")
    assert not reg.params.approach_allowed
    assert all("body_yaw" not in o for o in out)
    assert [o["head.yaw"] for o in out] == pytest.approx([0.10210176, 0.20420352, 0.10210176])


# ------------------------------------------------------------------ channels come from the manifest

SWEEP = [[0.0, 0], [0.3, 10], [0.6, 0]]
DESK = ["neck_pan", "neck_tilt"]


def desk_stack(behaviour: dict, **layers) -> ConfigStack:
    stack = ConfigStack(**layers)
    stack.set_manifest_behaviour(behaviour, DESK)
    return stack


def test_the_manifest_dofs_are_the_valid_channels():
    stack = desk_stack(
        {
            "expression_map": {"joy": {"channels": {"neck_pan": SWEEP}}},
            "gesture_map": {"nod": {"channels": {"neck_tilt": SWEEP}}},
            "circadian_motion": {"rest_pose_deg": {"neck_tilt": 10}},
        }
    )
    assert set(stack.effective().expression_map) == {"joy"}


@pytest.mark.parametrize(
    "behaviour",
    [
        {"expression_map": {"joy": {"channels": {"antenna_left": SWEEP}}}},
        {"gesture_map": {"nod": {"channels": {"head.pitch": SWEEP}}}},
        {"circadian_motion": {"rest_pose_deg": {"antenna_left": 10}}},
        {"circadian_motion": {"yawn": {"channels": {"antenna_left": SWEEP}}}},
        {"circadian_motion": {"stretch": {"channels": {"antenna_left": SWEEP}}}},
    ],
)
def test_a_channel_the_manifest_has_no_dof_for_is_refused(behaviour):
    with pytest.raises(ConfigError, match="unknown channel '(antenna_left|head.pitch)'"):
        desk_stack(behaviour)


def test_translation_dofs_are_not_channels(manifest):
    assert "head.x" in manifest.dofs and "head.x" not in manifest.channels
    stack = ConfigStack()
    behaviour = {"expression_map": {"joy": {"channels": {"head.x": SWEEP}}}}
    with pytest.raises(ConfigError, match="unknown channel 'head.x'"):
        stack.set_manifest_behaviour(behaviour, manifest.channels)


def test_a_later_layer_cannot_name_a_missing_channel():
    app = {"expression_map": {"joy": {"channels": {"antenna_left": SWEEP}}}}
    stack = ConfigStack(app=app)
    with pytest.raises(ConfigError, match="unknown channel 'antenna_left'"):
        stack.set_manifest_behaviour({}, DESK)
    assert stack.channels is None
    stack.app = {}
    stack.set_manifest_behaviour({}, DESK)
    stack.app = app
    with pytest.raises(ConfigError, match="unknown channel 'antenna_left'"):
        stack.effective()


def test_a_robot_without_a_body_turn_keeps_every_channel(manifest):
    entry = ExpressionEntry(channels={"body_yaw": SWEEP, "head.yaw": SWEEP})
    prim = build_primitive("x", entry, manifest.dofs, body_yaw=None, body=False)
    assert prim is not None and set(prim.keyframes[1].values) == {"body_yaw", "head.yaw"}
    prim = build_primitive("x", entry, manifest.dofs, body_yaw="body_yaw", body=False)
    assert prim is not None and set(prim.keyframes[1].values) == {"head.yaw"}
