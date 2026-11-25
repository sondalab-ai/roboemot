"""The describer: one request, one line of text (SPEC 8, Surroundings snapshots)."""

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
    cfg = RoboEmotConfig(surroundings={"shown": {"max_chars": 20}})
    r = FakeRouter("a silver laptop\nwith   stickers on the lid and a dent")
    text = await describe(r, ROUTE, "shown", b"jpg", "laptop", cfg)
    assert text == "a silver laptop with"
    assert "\n" not in text


async def test_blank_reply_raises_empty_description():
    with pytest.raises(EmptyDescription):
        await describe(FakeRouter("  \n "), ROUTE, "room", b"jpg", "", RoboEmotConfig())


async def test_the_shown_prompt_carries_the_hint_and_the_token_cap():
    r = FakeRouter("a red mug")
    cfg = RoboEmotConfig(surroundings={"describe_max_tokens": 99})
    await describe(r, ROUTE, "shown", b"jpg", "cup", cfg)
    name, prompt, jpeg, max_tokens = r.calls[0]
    assert "cup" in prompt and "{hint}" not in prompt
    assert (name, jpeg, max_tokens) == ("eye", b"jpg", 99)


async def test_the_room_prompt_asks_to_leave_people_out():
    r = FakeRouter("a living room")
    cfg = RoboEmotConfig(surroundings={"room": {"max_chars": 40}})
    assert await describe(r, ROUTE, "room", b"jpg", "", cfg) == "a living room"
    assert "people" in r.calls[0][1].lower()


async def test_the_describe_request_takes_the_surroundings_first_token_time():
    r = FakeRouter("a red mug")
    cfg = RoboEmotConfig(surroundings={"describe_first_token_s": 25.0})
    await describe(r, ROUTE, "room", b"jpg", "", cfg)
    assert r.first_token_s == 25.0
    assert RoboEmotConfig().surroundings.describe_first_token_s == 20.0
