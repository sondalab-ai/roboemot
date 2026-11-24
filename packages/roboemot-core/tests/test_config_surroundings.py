"""Surroundings settings and the route image switch (SPEC 8, 11A, 12; design 2026-11-24)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from roboemot.config.schema import LLMRoute, RoboEmotConfig, image_route
from roboemot.config.stack import violations
from roboemot.llm.prompt import system_tokens_worst_case


def chat(name: str, **kw) -> LLMRoute:
    return LLMRoute(name=name, base_url="https://api.example.com/v1", model="m", **kw)


def test_defaults_match_the_design():
    s = RoboEmotConfig().surroundings
    assert (s.settle_s, s.window_s, s.candidates) == (0.5, 1.0, 5)
    assert (s.max_side_px, s.jpeg_quality) == (768, 80)
    assert (s.max_failures, s.pause_s, s.describe_max_tokens) == (3, 600, 120)
    assert s.min_sharpness == 20.0
    sh = s.shown
    assert (sh.max_speed_deg_s, sh.margin, sh.wait_s, sh.max_chars) == (15, 0.25, 4, 80)
    r = s.room
    assert (r.min_interval_s, r.changed_share, r.wait_s, r.max_chars, r.max_age_s) == (
        300,
        0.5,
        10,
        160,
        3600,
    )


def test_two_image_routes_are_rejected():
    with pytest.raises(ValidationError, match="at most one route may have images on"):
        RoboEmotConfig(llm_routes=[chat("a", images=True), chat("b", images=True)])


def test_a_realtime_route_cannot_see_images():
    with pytest.raises(ValidationError, match="images needs kind: chat"):
        chat("rt", kind="realtime", images=True)


def test_image_route_returns_the_one_route_or_none():
    assert image_route(RoboEmotConfig(llm_routes=[chat("a")])) is None
    cfg = RoboEmotConfig(llm_routes=[chat("a"), chat("b", images=True)])
    assert image_route(cfg).name == "b"


def test_images_is_a_privacy_opt_in():
    approved = RoboEmotConfig(llm_routes=[chat("a")])
    candidate = RoboEmotConfig(llm_routes=[chat("a", images=True)])
    found = [v for v in violations(approved, candidate) if v.kind == "privacy"]
    assert found and "images" in found[0].path
    assert not [v for v in violations(candidate, approved) if v.kind == "privacy"]


def test_worst_case_counts_the_room_line():
    off = RoboEmotConfig(llm_routes=[chat("a")], surroundings={"room": {"max_chars": 400}})
    on = RoboEmotConfig(
        llm_routes=[chat("a", images=True)], surroundings={"room": {"max_chars": 400}}
    )
    assert system_tokens_worst_case(on) - system_tokens_worst_case(off) >= 400 // 4
