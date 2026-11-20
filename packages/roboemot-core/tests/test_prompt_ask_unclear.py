"""SPEC 8, Skip tool and Robot expression: with ``expression.reactions.model_asks`` the model
asks a short question about speech clearly meant for the robot instead of ignoring it."""

from __future__ import annotations

from roboemot.config.schema import RoboEmotConfig
from roboemot.llm.prompt import (
    ADDRESSED,
    ask_prompt,
    open_prompt,
    realtime_instructions,
    skip_prompt,
)

TEXT = "do not call ignore_utterance: ask one short question"


def cfg(skip: bool = True) -> RoboEmotConfig:
    return RoboEmotConfig.model_validate({"voice": {"realtime_route": {"skip_tool": skip}}})


def test_the_rule_is_shipped():
    assert TEXT in ask_prompt()


def test_rule_only_with_skip_rule_and_ask_rule():
    on = cfg()
    reg = on.register.params
    text = realtime_instructions(on, reg, ask_rule=True, conversation=ADDRESSED)
    assert ask_prompt() in text
    assert text.index(skip_prompt()) < text.index(ask_prompt()) < text.index(open_prompt())
    assert ask_prompt() not in realtime_instructions(on, reg, ask_rule=False)
    assert ask_prompt() not in realtime_instructions(on, reg, ask_rule=True, skip_rule=False)
    off = cfg(skip=False)
    assert ask_prompt() not in realtime_instructions(off, off.register.params, ask_rule=True)
