"""Skill packs in the conversation worker's chat turns (SPEC 8; design 2026-11-24 5.4, 5.5)."""

from __future__ import annotations

from pathlib import Path

from roboemot.config.schema import RoboEmotConfig
from roboemot.skill_packs.library import SkillPackLibrary
from roboemot.skill_packs.session import SkillPackSession
from test_conversation_worker import rigs  # noqa: F401  (fixture)


def attach(rig, root: Path, idle_end_s: float = 300.0) -> SkillPackSession:
    """A skill pack session over one enabled skill pack, ``quiz-night``, on ``rig``'s worker."""
    folder = root / "quiz-night"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: quiz-night\ndescription: A quiz.\n---\nQUIZ-BODY")
    cfg = lambda: rig.stack.effective().model_copy(  # noqa: E731
        update={
            "skill_packs": RoboEmotConfig(
                skill_packs={"enabled": ["quiz-night"], "idle_end_s": idle_end_s}
            ).skill_packs
        }
    )
    lib = SkillPackLibrary(root, cfg)
    lib.scan()
    session = SkillPackSession(lib, cfg, lambda: rig.clock[0])
    rig.worker.skill_packs = session
    return session


def test_chat_system_prompt_carries_catalogue_and_active_block(rigs, tmp_path):  # noqa: F811
    rig = rigs()
    session = attach(rig, tmp_path)
    rig.turn("ciao")
    assert "quiz-night: A quiz." in rig.system()
    assert "QUIZ-BODY" not in rig.system()
    session.start("quiz-night")
    rig.turn("giochiamo")
    assert "QUIZ-BODY" in rig.system()
    assert rig.system().index("quiz-night: A quiz.") < rig.system().index("QUIZ-BODY")


def test_no_skill_text_when_session_none(rigs):  # noqa: F811
    rig = rigs()
    rig.turn("ciao")
    assert "skill pack" not in rig.system().lower()


def test_conversation_end_ends_pack(rigs, tmp_path):  # noqa: F811
    rig = rigs()
    session = attach(rig, tmp_path, idle_end_s=3600.0)  # the session times out first
    rig.turn("uno")
    session.start("quiz-night")
    rig.clock[0] += rig.stack.effective().conversation.session_timeout_min * 60 + 1
    assert rig.worker.expire(rig.clock[0])
    assert session.snapshot()["active"] is None
    assert session.snapshot()["counts"]["end_conversation_end"] == 1


def test_addressed_turn_resets_idle_and_idle_ends_before_the_turn(rigs, tmp_path):  # noqa: F811
    rig = rigs()
    session = attach(rig, tmp_path)
    session.start("quiz-night")
    idle = session._config().skill_packs.idle_end_s
    rig.clock[0] += idle - 1
    rig.turn("risposta")
    assert session.snapshot()["active"] == "quiz-night"
    rig.clock[0] += idle + 1
    rig.worker.expire(rig.clock[0])
    assert session.snapshot()["active"] is None


def started_runtime(tmp_path):
    """``start_skill_packs`` over a stack with ``quiz-night`` on, with ``quiz-night`` active."""
    from contextlib import ExitStack
    from types import SimpleNamespace

    from roboemot.bus import Bus
    from roboemot.config import ConfigStack
    from roboemot.runtime.startup import start_skill_packs

    folder = tmp_path / "skill_packs" / "quiz-night"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("---\nname: quiz-night\ndescription: A quiz.\n---\nQUIZ")
    stack = ConfigStack(app={"data_dir": str(tmp_path), "skill_packs": {"enabled": ["quiz-night"]}})
    p = SimpleNamespace(cfg=stack.effective(), bus=Bus())
    host = SimpleNamespace(skill_packs=None)
    cleanup = ExitStack()
    session = start_skill_packs(host, stack, p, cleanup)
    assert host.skill_packs is session
    assert session.start("quiz-night")[0] == "started"
    return session, stack, p.bus, cleanup


def test_asleep_ends_the_active_pack(tmp_path):
    session, _, bus, cleanup = started_runtime(tmp_path)
    with cleanup:
        bus.publish("robot.asleep", {}, 0.0)
        assert session.snapshot()["counts"]["end_asleep"] == 1


def test_resting_ends_the_active_pack(tmp_path):
    from roboemot.circadian.model import CircadianState

    session, _, bus, cleanup = started_runtime(tmp_path)
    with cleanup:
        awake = CircadianState(12.0, 0.0, 0.0, 0.0, 0.0, "awake", None)
        bus.publish("circadian.state", awake, 0.0)
        assert session.snapshot()["active"] == "quiz-night"
        bus.publish(
            "circadian.state", CircadianState(12.0, 0.0, 0.0, 0.0, 0.0, "resting", None), 0.0
        )
        assert session.snapshot()["counts"]["end_resting"] == 1


def test_turning_the_pack_off_ends_it(tmp_path):
    session, stack, _, cleanup = started_runtime(tmp_path)
    with cleanup:
        stack.set("skill_packs.enabled", [], admin=True)
        assert session.snapshot()["counts"]["end_disabled"] == 1
