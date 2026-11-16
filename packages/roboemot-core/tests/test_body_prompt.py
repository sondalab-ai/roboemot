"""The robot's body as the LLM is told it: ``persona.body_prompt`` (SPEC 8, 11A)."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from roboemot.affect.clips import load_clip_library
from roboemot.config import ConfigError, ConfigStack
from roboemot.config.schema import PersonaConfig, RoboEmotConfig
from roboemot.config.stack import GuardrailError
from roboemot.llm.prompt import (
    CHARS_PER_TOKEN,
    check_route_budgets,
    realtime_instructions,
    system_prompt,
    system_tokens_worst_case,
)
from roboemot.manifest import load_manifest

BODY = "Your body: a head and a speaker."
REPO = Path(__file__).resolve().parents[3]
MANIFESTS = sorted((REPO / "configs" / "manifests").glob("*.yaml"))
REACHY = REPO / "configs" / "manifests" / "reachy-mini-wireless.yaml"
DEFAULT_MAX_CHARS = 200  # the cap for a shipped default (short, to save tokens)


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


# ----------------------------------------------------------------------------- manifest default


def reachy_stack(**kwargs) -> ConfigStack:
    manifest = load_manifest(REACHY, allow_unmeasured_dynamics=True)
    stack = ConfigStack(**kwargs)
    stack.set_manifest_behaviour(
        manifest.model.behaviour,
        manifest.channels,
        load_clip_library(manifest),
        body_yaw=None,
    )
    return stack


def test_manifest_sets_the_body_prompt():
    manifest = load_manifest(REACHY, allow_unmeasured_dynamics=True)
    text = manifest.model.behaviour["persona"]["body_prompt"]
    assert text
    assert reachy_stack().effective().persona.body_prompt == text


def test_manifest_persona_may_set_only_the_body_prompt():
    with pytest.raises(ConfigError, match="body_prompt"):
        ConfigStack().set_manifest_behaviour({"persona": {"prompt": "x"}}, (), body_yaw=None)


def test_manifest_body_prompt_passes_the_manifest_guard_and_owner_edits_need_the_admin():
    stack = ConfigStack()
    stack.set_manifest_behaviour({"persona": {"body_prompt": BODY}}, (), body_yaw=None)
    assert stack.effective().persona.body_prompt == BODY
    with pytest.raises(GuardrailError):  # an owner change of the manifest text needs the admin
        stack.set("persona.body_prompt", "You have wings.")


def test_manifest_body_prompt_over_the_cap_is_refused_at_load():
    too_long = "x" * (PersonaConfig().body_prompt_max_chars + 1)
    with pytest.raises(ConfigError, match="persona.body_prompt has"):
        ConfigStack().set_manifest_behaviour(
            {"persona": {"body_prompt": too_long}}, (), body_yaw=None
        )


def test_manifest_body_prompt_must_be_text():
    with pytest.raises(ConfigError, match="body_prompt"):
        ConfigStack().set_manifest_behaviour({"persona": {"body_prompt": 3}}, (), body_yaw=None)


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: p.stem)
def test_every_shipped_manifest_has_a_short_body_prompt(path):
    behaviour = yaml.safe_load(path.read_text())["behaviour"]
    text = behaviour["persona"]["body_prompt"]
    assert isinstance(text, str) and 0 < len(text) <= DEFAULT_MAX_CHARS


def test_runtime_override_wins_and_clear_restores_the_manifest_text():
    stack = reachy_stack()
    shipped = stack.effective().persona.body_prompt
    stack.set("persona.body_prompt", BODY, admin=True)
    assert stack.effective().persona.body_prompt == BODY
    stack.set("persona.body_prompt", "", admin=True)
    assert stack.effective().persona.body_prompt == ""
    stack.clear("persona.body_prompt")
    assert stack.effective().persona.body_prompt == shipped


def test_shipped_configs_pass_the_route_budget():
    app = yaml.safe_load((REPO / "configs" / "companion.yaml").read_text())
    check_route_budgets(reachy_stack(app=app).effective())
