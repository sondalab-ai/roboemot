"""Skill packs on the realtime route: the active block survives a reconnect and a fallback to a
chat route (SPEC 8; design 2026-11-24-skill-packs 5.5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.skill_packs.library import SkillPackLibrary
from roboemot.skill_packs.session import SkillPackSession
from rt_rig import build_rt, wait_for
from turn_server import TurnServer


@pytest.fixture()
def rigs():
    made = []

    def make(server, **kw):
        rig = build_rt(server, **kw)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


def attach(rig, root: Path) -> SkillPackSession:
    folder = root / "quiz-night"
    folder.mkdir()
    (folder / "SKILL.md").write_text("---\nname: quiz-night\ndescription: A quiz.\n---\nQUIZ-BODY")
    packs = RoboEmotConfig(skill_packs={"enabled": ["quiz-night"]}).skill_packs

    def cfg():
        return rig.stack.effective().model_copy(update={"skill_packs": packs})

    lib = SkillPackLibrary(root, cfg)
    lib.scan()
    session = SkillPackSession(lib, cfg, lambda: 0.0)
    rig.worker.skill_packs = session
    session.start("quiz-night")
    return session


def last_instructions(srv) -> str:
    return srv.of("response.create")[-1]["response"]["instructions"]


def test_realtime_instructions_carry_active_block(rigs, tmp_path):
    with TurnServer() as srv:
        rig = rigs(srv)
        attach(rig, tmp_path)
        rig.turn("giochiamo")
        assert "QUIZ-BODY" in last_instructions(srv)
        assert "quiz-night: A quiz." in last_instructions(srv)


def test_reconnect_replay_still_has_active_block(rigs, tmp_path):
    with TurnServer(["close_mid", "reply"], delta_ms=1500, gap_s=0.3) as srv:
        rig = rigs(srv)
        attach(rig, tmp_path)
        rig.turn("prima")
        rig.turn("seconda")
        assert wait_for(lambda: len(srv.connections) == 2)
        assert "QUIZ-BODY" in last_instructions(srv)


def test_fallback_to_chat_keeps_active_block(rigs, tmp_path):
    with TurnServer(["error"]) as srv:
        rig = rigs(srv, rt={"retry": {"enabled": False}})
        attach(rig, tmp_path)
        rig.turn("giochiamo")
        assert rig.chat_hosts() == ["lan.local"]
        assert "QUIZ-BODY" in rig.chat.requests[-1][1]["messages"][0]["content"]
