"""The robot's body as the LLM is told it: ``persona.body_prompt`` (SPEC 8, 11A)."""

import pytest
from pydantic import ValidationError
from roboemot.config import ConfigStack
from roboemot.config.schema import PersonaConfig, RoboEmotConfig
from roboemot.config.stack import GuardrailError
from roboemot.llm.prompt import (
    CHARS_PER_TOKEN,
    realtime_instructions,
    system_prompt,
    system_tokens_worst_case,
)

BODY = "Your body: a head and a speaker."


def test_body_prompt_follows_the_persona_prompt():
    cfg = RoboEmotConfig(persona={"prompt": "I am Neve.", "body_prompt": BODY})
    for text in (
        system_prompt(cfg, cfg.register.params, language="en"),
        realtime_instructions(cfg, cfg.register.params, language="en"),
    ):
        lines = text.split("\n")
        assert lines[:2] == ["I am Neve.", BODY]


def test_empty_body_prompt_leaves_the_prompt_unchanged():
    cfg = RoboEmotConfig()
    assert cfg.persona.body_prompt == ""
    lines = system_prompt(cfg, cfg.register.params, language="en").split("\n")
    assert "" not in lines
    assert lines[1] == cfg.register.params.persona_fragment


def test_body_prompt_too_long_is_rejected():
    with pytest.raises(ValidationError, match="persona.body_prompt has 61 characters, max 60"):
        PersonaConfig(body_prompt="x" * 61, body_prompt_max_chars=60)


def test_worst_case_counts_the_body_prompt_cap():
    cfg = RoboEmotConfig()
    small = RoboEmotConfig(persona={"body_prompt_max_chars": 50})
    grown = system_tokens_worst_case(cfg) - system_tokens_worst_case(small)
    assert grown >= (cfg.persona.body_prompt_max_chars - 50) // CHARS_PER_TOKEN


def test_body_prompt_change_needs_the_admin():
    stack = ConfigStack()
    with pytest.raises(GuardrailError):
        stack.set("persona.body_prompt", "You have wings.")
    stack.set("persona.body_prompt", BODY, admin=True)
    assert stack.effective().persona.body_prompt == BODY
