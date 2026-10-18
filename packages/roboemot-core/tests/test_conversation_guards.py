"""SPEC 8, 11A, 12: guardrails on the child-safety settings, the shared router following the
configuration, logging pins, pseudonym option refusal."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError
from roboemot.config import ConfigStack
from roboemot.config.schema import PrivacyConfig, RoboEmotConfig
from roboemot.config.stack import GuardrailError, field_meta, violations
from roboemot.llm import Router
from roboemot.llm.hub import RouterHub
from roboemot.llm.prompt import max_reply_sentences, system_prompt

ROUTE = {"name": "lan", "base_url": "http://lan.local/v1", "model": "m1"}


def test_admin_only_on_a_container_covers_every_leaf_inside_it():
    assert field_meta("safety.minor_lines.it.filtered").get("admin_only")
    assert field_meta("safety.minor_lines").get("admin_only")
    assert field_meta("safety.minor_output_blocklist.it").get("admin_only")
    assert not field_meta("safety.minor_motion_factor").get("admin_only")


def test_minor_lines_cannot_be_changed_at_runtime_without_admin():
    stack = ConfigStack()
    with pytest.raises(GuardrailError):
        stack.set("safety.minor_lines.it.filtered", "Altro.")
    stack.set("safety.minor_lines.it.filtered", "Altro.", admin=True)
    assert stack.get("safety.minor_lines.it.filtered") == "Altro."


def test_emptying_or_removing_a_blocklist_needs_admin():
    stack = ConfigStack()
    with pytest.raises(GuardrailError):
        stack.set("safety.minor_output_blocklist.en", [])
    approved = RoboEmotConfig()
    shorter = approved.model_copy(deep=True)
    del shorter.safety.minor_output_blocklist["it"]  # a language's list removed outright
    assert any(v.kind == "admin" for v in violations(approved, shorter))


def test_empty_minor_lines_are_refused_even_with_admin():
    stack = ConfigStack()
    with pytest.raises(Exception, match="never silence"):
        stack.set("safety.minor_lines.it.greeting", "", admin=True)


def test_a_reply_cap_can_only_go_down_at_runtime():
    stack = ConfigStack()
    assert stack.get("register.child.max_reply_sentences") == 2
    with pytest.raises(GuardrailError):
        stack.set("register.child.max_reply_sentences", 5)
    stack.set("register.child.max_reply_sentences", 1)
    assert stack.get("register.child.max_reply_sentences") == 1


def test_an_unset_register_cap_counts_as_the_conversation_cap():
    stack = ConfigStack()
    assert stack.get("register.adult.max_reply_sentences") is None
    with pytest.raises(GuardrailError):
        stack.set("register.adult.max_reply_sentences", 10)  # above the conversation's 3
    stack.set("register.adult.max_reply_sentences", 2)


def test_a_per_person_override_cannot_raise_the_cap_and_is_audited():
    stack = ConfigStack()
    cfg = stack.effective({"register": {"child": {"max_reply_sentences": 9}}})
    assert cfg.register.child.max_reply_sentences == 2
    assert any(e.source == "person" and "max_reply_sentences" in str(e.new) for e in stack.audit)


def test_a_register_cap_never_exceeds_the_conversation_cap_in_the_prompt():
    cfg = RoboEmotConfig.model_validate({"register": {"adult": {"max_reply_sentences": 9}}})
    assert max_reply_sentences(cfg, cfg.register.adult) == cfg.conversation.max_reply_sentences
    assert "at most 2 sentences" in system_prompt(cfg, cfg.register.child)
    assert max_reply_sentences(cfg, cfg.register.teen) == cfg.conversation.max_reply_sentences


def test_pseudonyms_to_cloud_is_refused_until_it_is_implemented():
    assert PrivacyConfig().pseudonyms_to_cloud is False
    with pytest.raises(ValidationError, match="not implemented in v1 M3"):
        PrivacyConfig(pseudonyms_to_cloud=True)


def test_the_hub_follows_a_configuration_change_for_every_consumer():
    stack = ConfigStack(app={"llm_routes": [ROUTE]})

    def make(cfg):
        return Router(cfg.llm_routes, cfg.safety)

    hub = RouterHub(make(stack.effective()), make)
    first = hub.router
    first.charge("lan", 5)
    stack.set("llm_routes", [{**ROUTE, "model": "m2"}], layer="app")
    rebuilt = hub.sync(stack.effective())
    assert hub.router is rebuilt and rebuilt is not first
    assert rebuilt.routes[0].model == "m2" and rebuilt.used("lan") == 5
    assert hub.sync(stack.effective()) is rebuilt


def test_library_debug_logging_that_could_carry_text_is_pinned_down():
    assert logging.getLogger("openai").level >= logging.INFO
    assert logging.getLogger("httpx").level >= logging.WARNING
