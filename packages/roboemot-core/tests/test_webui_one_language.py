"""One language per page (SPEC 11A, 12): every page string, every code the server sends for the
page to word and every setting description exists in each page language.

A key added in one language only, a code with no sentence or a new schema description without a
translation fails here, before the page shows it in the other language (or as a raw code).
"""

from __future__ import annotations

import re
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel
from roboemot import diag
from roboemot.config.catalogue import BLANK_SOURCE
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import LAYER_ORDER
from roboemot.registry import enrollment
from roboemot.registry.xmp import CATEGORIES
from roboemot.runtime.live import STATUSES
from roboemot.webui import descriptions
from roboemot.webui.robot_section import RESTART_EFFECTS
from roboemot.webui.strings import DEFAULT_LANGUAGE, STRINGS

SRC = Path(__file__).resolve().parents[1] / "src" / "roboemot"
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
OTHER_LANGUAGES = sorted(set(STRINGS) - {DEFAULT_LANGUAGE})


def keys(lang: str) -> set[str]:
    return set(STRINGS[lang])


@pytest.mark.parametrize("lang", OTHER_LANGUAGES)
def test_every_key_exists_in_every_language(lang):
    en, other = keys(DEFAULT_LANGUAGE), keys(lang)
    assert sorted(en - other) == [], f"missing in {lang}"
    assert sorted(other - en) == [], f"missing in {DEFAULT_LANGUAGE}"


@pytest.mark.parametrize("lang", OTHER_LANGUAGES)
def test_every_translation_has_the_same_placeholders(lang):
    en = STRINGS[DEFAULT_LANGUAGE]
    differ = {
        k: (PLACEHOLDER.findall(en[k]), PLACEHOLDER.findall(v))
        for k, v in STRINGS[lang].items()
        if k in en and sorted(PLACEHOLDER.findall(en[k])) != sorted(PLACEHOLDER.findall(v))
    }
    assert differ == {}


def must_have(prefix: str, codes) -> list[str]:
    """The keys ``prefix + code`` missing from the default language (the parity test covers the
    others); spaces and dashes in a code read as ``_``, as the page's ``word`` does."""
    en = STRINGS[DEFAULT_LANGUAGE]
    wanted = {prefix + re.sub(r"[ -]", "_", c) for c in codes}
    return sorted(k for k in wanted if k not in en)


def schema_descriptions() -> set[str]:
    seen: set[str] = set()
    done: set[type] = set()

    def walk(model: type[BaseModel]) -> None:
        if model in done:
            return
        done.add(model)
        for f in model.model_fields.values():
            if f.description:
                seen.add(f.description)
            visit(f.annotation)

    def visit(t) -> None:
        if isinstance(t, type) and issubclass(t, BaseModel):
            walk(t)
        for arg in typing.get_args(t):
            visit(arg)

    walk(RoboEmotConfig)
    return seen


@pytest.mark.parametrize("lang", OTHER_LANGUAGES)
def test_every_setting_description_is_translated(lang):
    table = descriptions.table(lang)
    wanted = schema_descriptions()
    assert sorted(wanted - set(table)) == [], f"add them to descriptions_{lang}.json"
    assert sorted(set(table) - wanted) == [], f"stale entries in descriptions_{lang}.json"
    assert all(isinstance(v, str) and v for v in table.values())


def test_localize_swaps_descriptions_and_keeps_a_plugins_own_rows():
    text = next(iter(schema_descriptions()))
    rows = [
        {"path": "a.b", "description": text, "entry": [{"path": "x", "description": text}]},
        {"path": "robot.r.c", "description": "Already worded."},
        {"path": "a.d", "description": "Not in the table."},
    ]
    out = descriptions.localize(rows, "it", frozenset({"robot.r.c"}))
    it = descriptions.table("it")[text]
    assert out[0]["description"] == it and out[0]["entry"][0]["description"] == it
    assert out[1]["description"] == "Already worded."
    assert out[2]["description"] == ""  # never shown in English on an Italian page
    assert rows[0]["description"] == text  # the input is not changed
    assert descriptions.localize(rows, DEFAULT_LANGUAGE) is rows


def test_every_live_status_and_event_has_a_sentence():
    assert must_have("live_status_", STATUSES) == []
    events = [
        f"{kind}_{reason}" for kind, reasons in diag.REASONS.items() if kind != "cue"
        for reason in reasons
    ]  # fmt: skip
    assert must_have("live_ev_", events) == []
    assert must_have("live_why_", diag.REASONS["cue"]) == []
    assert must_have("live_ev_cue_", diag.CUE_KINDS) == []
    assert must_have("live_res_", [r for r in diag.CUE_RESULTS if r != "played"]) == []
    assert must_have("live_fail_", diag.FAILURES) == []
    assert must_have("live_conv_", diag.FIELDS["conversation"]) == []
    assert must_have("live_voice_", diag.FIELDS["voice"]) == []


def test_every_code_inside_a_live_sentence_is_worded():
    codes = [
        *diag.ROUTE_REASONS, *diag.EXCEPTIONS.values(), *diag.PATHS,
        *diag.FIELDS["released_by"], diag.OTHER,
    ]  # fmt: skip
    assert must_have("live_code_", codes) == []


def test_default_config_names_are_worded():
    cfg = RoboEmotConfig()
    assert must_have("affect_label_", cfg.affect.labels) == []
    assert must_have("live_reflex_name_", [r.name for r in cfg.reflexes.rules]) == []
    assert must_have("live_circ_", cfg.circadian.states) == []
    assert must_have("category_", CATEGORIES) == []
    assert must_have("settings_restart_effect_", RESTART_EFFECTS) == []


def test_every_setting_source_is_worded():
    """``ConfigStack.source_of`` layers, the catalogue's blank entry and a robot plugin's own
    sources (``manifest``, ``config``)."""
    sources = [*(s for s in LAYER_ORDER if s != "person"), BLANK_SOURCE, "config"]
    assert must_have("settings_source_", sources) == []


def test_enrollment_states_and_route_errors_are_worded():
    states = [
        enrollment.IDLE, enrollment.COLLECTING, enrollment.READY, enrollment.COMPLETING,
        enrollment.TIMED_OUT, enrollment.COMPLETED, enrollment.CANCELLED, enrollment.UNAVAILABLE,
    ]  # fmt: skip
    assert must_have("state_", states) == []
    assert must_have("hint_", ["no_face", "turn_head"]) == []
    errors = ["timeout", "budget", "secret", "failed", "provider", "no_extra"]
    assert must_have("route_error_", errors) == []


def source(*parts: str) -> str:
    return (SRC.joinpath(*parts)).read_text(encoding="utf-8")


def test_every_check_reason_code_is_worded():
    text = source("llm", "conformance.py") + source("realtime", "check.py")
    codes = set(re.findall(r'CheckResult\([^()]*?,\s*"([a-z_]+)"(?:,|\))', text, re.S))
    codes |= set(re.findall(r'_no_text\([^()]*?"([a-z_]+)"\s*\)', text, re.S))
    codes |= {
        "out_of_tokens",
        "latency",
        *(f"item_delete_{v}" for v in ("accepted", "refused", "unconfirmed")),
    }
    assert {"chunks", "latency", "round_trip_ok", "connected", "no_text_after"} <= codes
    assert must_have("check_reason_", codes) == []
    assert must_have("check_why_", ["http", "timeout", "unreachable", "error"]) == []


def test_every_api_error_code_is_worded():
    codes: set[str] = set()
    for path in [*SRC.glob("webui/*.py"), *SRC.glob("config/*.py")]:
        text = path.read_text(encoding="utf-8")
        codes |= set(re.findall(r'Error\(\s*(?:HTTPStatus\.\w+,\s*)?"([a-z_]+)"', text))
    assert len(codes) > 20
    assert must_have("err_", codes) == []
