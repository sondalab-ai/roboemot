"""Surroundings settings and the route image switch (SPEC 8, 11A, 12; design 2026-11-24)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from roboemot.config.schema import LLMRoute, RoboEmotConfig, image_routes
from roboemot.config.stack import violations
from roboemot.llm.prompt import system_tokens_worst_case


def chat(name: str, **kw) -> LLMRoute:
    return LLMRoute(name=name, base_url="https://api.example.com/v1", model="m", **kw)


def test_defaults_match_the_design():
    s = RoboEmotConfig().surroundings
    assert (s.settle_s, s.window_s, s.candidates) == (0.5, 1.0, 5)
    assert (s.max_side_px, s.jpeg_quality) == (768, 80)
    assert (s.max_failures, s.pause_s) == (3, 600)
    # a model that thinks first spends its reply budget on it (gemini-3.5-flash on the robot,
    # 2026-11-25: finish=length with 120 tokens, cut or empty text, answers up to 9.6 s;
    # claude-haiku-5-5: empty under 1024, 313 to 901 completion tokens in a probe)
    assert (s.describe_max_tokens, s.describe_first_token_s) == (4096, 30.0)
    assert s.image_route_order == []
    assert s.min_sharpness == 20.0
    r = s.room
    assert (r.min_interval_s, r.changed_share, r.max_chars, r.max_age_s) == (300, 0.5, 160, 3600)


def test_two_image_routes_load_in_list_order():
    cfg = RoboEmotConfig(llm_routes=[chat("b", images=True), chat("x"), chat("a", images=True)])
    assert [r.name for r in image_routes(cfg)] == ["b", "a"]  # their priority (2026-11-25)


def test_the_image_route_order_comes_before_the_list_order():
    """The robot's case (2026-11-25): a paid route before the free one in llm_routes; the order
    field puts the free one first."""
    routes = [chat("Claude", images=True), chat("gemini_pictures", images=True)]
    cfg = RoboEmotConfig(
        llm_routes=routes, surroundings={"image_route_order": ["gemini_pictures", "Claude"]}
    )
    assert [r.name for r in image_routes(cfg)] == ["gemini_pictures", "Claude"]


def test_routes_the_order_does_not_name_follow_in_list_order_and_unknown_names_are_ignored():
    routes = [chat("a", images=True), chat("x"), chat("b", images=True), chat("c", images=True)]
    cfg = RoboEmotConfig(llm_routes=routes, surroundings={"image_route_order": ["gone", "c", "x"]})
    assert [r.name for r in image_routes(cfg)] == ["c", "a", "b"]


def test_the_image_route_order_refuses_a_name_twice():
    with pytest.raises(ValidationError, match="duplicate route name"):
        RoboEmotConfig(surroundings={"image_route_order": ["a", "a"]})


def test_changing_the_image_route_order_needs_no_approval():
    routes = [chat("a", images=True), chat("b", images=True)]
    approved = RoboEmotConfig(llm_routes=routes)
    moved = RoboEmotConfig(llm_routes=routes, surroundings={"image_route_order": ["b", "a"]})
    assert not [v for v in violations(approved, moved) if v.kind == "privacy"]


def test_a_realtime_route_cannot_see_images():
    with pytest.raises(ValidationError, match="images needs kind: chat"):
        chat("rt", kind="realtime", images=True)


def test_image_routes_is_empty_without_a_route_with_images_on():
    assert image_routes(RoboEmotConfig(llm_routes=[chat("a")])) == []


def test_images_is_a_privacy_opt_in():
    approved = RoboEmotConfig(llm_routes=[chat("a")])
    candidate = RoboEmotConfig(llm_routes=[chat("a", images=True)])
    found = [v for v in violations(approved, candidate) if v.kind == "privacy"]
    assert found and all(v.path == "llm_routes.a.images" for v in found)
    image = [v for v in found if v.check == "image"]
    assert image and "https://api.example.com/v1" in image[0].candidate  # the approval names where
    assert not [v for v in violations(candidate, approved) if v.kind == "privacy"]


def test_worst_case_counts_the_room_line():
    off = RoboEmotConfig(llm_routes=[chat("a")], surroundings={"room": {"max_chars": 400}})
    on = RoboEmotConfig(
        llm_routes=[chat("a", images=True)], surroundings={"room": {"max_chars": 400}}
    )
    assert system_tokens_worst_case(on) - system_tokens_worst_case(off) >= 400 // 4


def test_moving_an_approved_image_route_to_another_host_needs_approval():
    approved = RoboEmotConfig(llm_routes=[chat("a", images=True)])
    moved = RoboEmotConfig(
        llm_routes=[
            LLMRoute(name="a", base_url="https://elsewhere.example.net/v1", model="m", images=True)
        ]
    )
    found = [v for v in violations(approved, moved) if v.kind == "privacy"]
    assert [v.check for v in found] == ["image"]


def test_changing_the_model_of_an_approved_image_route_needs_approval():
    approved = RoboEmotConfig(llm_routes=[chat("a", images=True)])
    other = RoboEmotConfig(
        llm_routes=[
            LLMRoute(name="a", base_url="https://api.example.com/v1", model="m2", images=True)
        ]
    )
    assert [v.check for v in violations(approved, other) if v.kind == "privacy"] == ["image"]


def test_moving_the_switch_to_another_route_needs_approval():
    approved = RoboEmotConfig(llm_routes=[chat("a", images=True), chat("b")])
    swapped = RoboEmotConfig(llm_routes=[chat("a"), chat("b", images=True)])
    found = [v for v in violations(approved, swapped) if v.kind == "privacy"]
    assert {v.path for v in found} == {"llm_routes.b.images"}
    assert "image" in {v.check for v in found}  # bound to b's endpoint and model


def test_reordering_image_routes_needs_no_approval():
    approved = RoboEmotConfig(llm_routes=[chat("a", images=True), chat("b", images=True)])
    reordered = RoboEmotConfig(llm_routes=[chat("b", images=True), chat("a", images=True)])
    assert not [v for v in violations(approved, reordered) if v.kind == "privacy"]


def test_an_unchanged_image_route_needs_nothing():
    cfg = RoboEmotConfig(llm_routes=[chat("a", images=True)])
    assert not [v for v in violations(cfg, cfg) if v.kind == "privacy"]


def test_a_new_path_on_the_same_host_needs_approval():
    approved = RoboEmotConfig(llm_routes=[chat("a", images=True)])
    other = RoboEmotConfig(
        llm_routes=[
            LLMRoute(name="a", base_url="https://api.example.com/other/v1", model="m", images=True)
        ]
    )
    assert [v.check for v in violations(approved, other) if v.kind == "privacy"] == ["image"]


# ---- the panorama (design 2026-11-25)


def test_panorama_defaults_match_the_design():
    s = RoboEmotConfig().surroundings
    assert s.view_dwell_s == 1.5
    r = s.room
    assert (r.span_deg, r.sweep_wait_s, r.view_match_deg) == (180, 60, 10)
    assert r.view_height_px <= s.max_side_px and r.separator_px >= 1


def test_the_view_dwell_leaves_room_for_the_picker():
    with pytest.raises(ValidationError, match="view_dwell_s"):
        RoboEmotConfig(surroundings={"view_dwell_s": 1.0, "settle_s": 0.5, "window_s": 1.0})
