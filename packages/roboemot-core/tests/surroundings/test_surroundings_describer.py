"""The describer: one request for the joined room picture, one line of text (SPEC 8,
Surroundings snapshots; design 2026-11-25)."""

from __future__ import annotations

import pytest
from roboemot.config.schema import LLMRoute, RoboEmotConfig
from roboemot.surroundings.describer import EmptyDescription, describe

ROUTE = LLMRoute(name="eye", base_url="https://api.example.com/v1", model="m", images=True)


class FakeRouter:
    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, []

    async def describe_image(self, route, prompt, jpeg, max_tokens, first_token_s):
        self.calls.append((route.name, prompt, jpeg, max_tokens))
        self.first_token_s = first_token_s
        return self.reply


async def test_reply_is_one_line_and_capped():
    cfg = RoboEmotConfig(surroundings={"room": {"max_chars": 40}})
    r = FakeRouter("a living room\nwith   a couch on the left and a window on the right")
    text = await describe(r, ROUTE, b"jpg", 4, 240, cfg)
    assert text == "a living room with a couch on the left a"
    assert "\n" not in text


async def test_blank_reply_raises_empty_description():
    with pytest.raises(EmptyDescription):
        await describe(FakeRouter("  \n "), ROUTE, b"jpg", 4, 240, RoboEmotConfig())


async def test_the_prompt_states_views_order_and_span_and_leaves_people_out():
    r = FakeRouter("a study")
    cfg = RoboEmotConfig(surroundings={"describe_max_tokens": 99})
    await describe(r, ROUTE, b"jpg", 4, 237, cfg)
    name, prompt, jpeg, max_tokens = r.calls[0]
    assert "4 " in prompt and "237 degrees" in prompt and "left to right" in prompt
    assert "{" not in prompt
    assert all(w in prompt for w in ("left", "centre", "right"))
    assert "people" in prompt.lower()
    assert "overlap" in prompt and "one thing" in prompt  # no double count at a view's edge
    assert (name, jpeg, max_tokens) == ("eye", b"jpg", 99)


async def test_the_describe_request_takes_the_surroundings_first_token_time():
    r = FakeRouter("a red mug")
    cfg = RoboEmotConfig(surroundings={"describe_first_token_s": 25.0})
    await describe(r, ROUTE, b"jpg", 3, 180, cfg)
    assert r.first_token_s == 25.0
