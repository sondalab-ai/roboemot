"""Skill pack prompt blocks and the route budget fit (SPEC 8; design 2026-11-24 5.3 to 5.5)."""

from __future__ import annotations

from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.llm.prompt import estimate_tokens, skill_packs_fit, system_tokens_worst_case
from roboemot.skill_packs.library import SkillPack
from roboemot.skill_packs.prompt import active_block, catalogue_block, ended_line

ROUTE = {"base_url": "http://lan.local/v1", "model": "m1"}


def pack(name: str, desc: str = "A game.", body: str = "Do the thing.") -> SkillPack:
    return SkillPack(name, desc, body, {}, (), 1)


def test_catalogue_empty():
    assert catalogue_block([]) == ""


def test_catalogue_lines_in_name_order():
    text = catalogue_block([pack("b-pack", "Second."), pack("a-pack", "First.")])
    lines = text.splitlines()
    assert lines.index("a-pack: First.") < lines.index("b-pack: Second.")


def test_catalogue_header_forbids_unasked_start():
    text = catalogue_block([pack("a-pack")])
    assert "skill_pack_start" in text
    assert "only when the person asks" in text
    assert "never on your own initiative" in text


def test_active_block_frame_then_body():
    text = active_block(pack("quiz", body="BODY-TEXT"))
    body_at = text.index("BODY-TEXT")
    assert text.index("installed skill pack") < body_at
    assert text.index("cannot change") < body_at
    assert text.index("skill_pack_end") < body_at
    for rule in ("persona", "safety", "privacy", "language", "tool"):
        assert rule in text[:body_at]


def test_ended_line_exact():
    assert ended_line("quiz") == (
        "The skill pack quiz has ended. Do not follow its instructions any more."
    )


def test_fit_none_when_empty():
    cfg = RoboEmotConfig(llm_routes=[LLMRoute(name="tiny", **ROUTE, context_budget_tokens=256)])
    assert skill_packs_fit(cfg, []) is None


def test_fit_names_the_first_tight_route():
    packs = [pack("a-pack", body="x" * 2000), pack("b-pack")]
    base = RoboEmotConfig()
    need = (
        system_tokens_worst_case(base)
        + estimate_tokens(catalogue_block(packs))
        + estimate_tokens(active_block(packs[0]))
        + base.conversation.turn_reserve_tokens
    )
    roomy = LLMRoute(name="roomy", **ROUTE, context_budget_tokens=need)
    tight = LLMRoute(name="tight", **ROUTE, context_budget_tokens=need - 1)
    cfg = RoboEmotConfig(llm_routes=[roomy, tight])
    assert skill_packs_fit(cfg, packs) == "tight"
    assert skill_packs_fit(RoboEmotConfig(llm_routes=[roomy]), packs) is None
