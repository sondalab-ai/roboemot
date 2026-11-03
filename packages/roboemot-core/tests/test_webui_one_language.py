"""One language per page (SPEC 11A, 12): every page string, every code the server sends for the
page to word and every setting description exists in each page language.

A key added in one language only, a code with no sentence or a new schema description without a
translation fails here, before the page shows it in the other language (or as a raw code).
"""

from __future__ import annotations

import ast
import logging
import re
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel
from roboemot import diag
from roboemot.config.catalogue import BLANK_SOURCE
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import LAYER_ORDER
from roboemot.llm import conformance
from roboemot.realtime import check as realtime_check
from roboemot.registry import enrollment
from roboemot.registry.xmp import CATEGORIES
from roboemot.runtime.live import STATUSES
from roboemot.webui import descriptions
from roboemot.webui.api import CATALOGUE_STATUS, EMOTIONS_STATUS
from roboemot.webui.robot_section import RESTART_EFFECTS
from roboemot.webui.strings import DEFAULT_LANGUAGE, STRINGS

SRC = Path(__file__).resolve().parents[1] / "src" / "roboemot"
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
CODE = re.compile(r"[a-z][a-z_]*")
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


def test_localize_swaps_descriptions_and_keeps_a_plugins_own_rows(caplog):
    text = next(iter(schema_descriptions()))
    rows = [
        {"path": "a.b", "description": text, "entry": [{"path": "x", "description": text}]},
        {"path": "robot.r.c", "description": "Already worded."},
        {"path": "a.d", "description": "Not in the table."},
    ]
    caplog.set_level(logging.DEBUG, logger="roboemot.webui.descriptions")
    out = descriptions.localize(rows, "it", frozenset({"robot.r.c"}))
    it = descriptions.table("it")[text]
    assert out[0]["description"] == it and out[0]["entry"][0]["description"] == it
    assert out[1]["description"] == "Already worded."
    assert out[2]["description"] == ""  # never shown in English on an Italian page
    assert "no it description for a.d" in caplog.text  # the path, never the text
    assert "Not in the table" not in caplog.text
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


def tree(module) -> ast.Module:
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def name_of(func: ast.expr) -> str:
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")


def given_codes(module) -> list[ast.expr]:
    """Every expression the module gives as a check's code: the fourth argument of each
    ``CheckResult(...)``, the last of each ``_no_text(...)`` and each value assigned to ``code``."""
    out: list[ast.expr] = []
    for node in ast.walk(tree(module)):
        if (
            isinstance(node, ast.Call)
            and name_of(node.func) == "CheckResult"
            and len(node.args) > 3
        ):
            out.append(node.args[3])
        elif isinstance(node, ast.Call) and name_of(node.func) == "_no_text":
            out.append(node.args[-1])
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [getattr(t, "id", None) for t in node.targets[0].elts]
            if "code" in names and isinstance(node.value, ast.Tuple):
                out.append(node.value.elts[names.index("code")])
    return out


@pytest.mark.parametrize("module", [conformance, realtime_check])
def test_every_check_reason_code_is_declared_and_worded(module):
    """The module's ``REASON_CODES`` holds every code its calls give (a literal, a parameter
    of ``_no_text`` passed through, or ``"item_delete_" + verdict``), and each is worded."""
    given = given_codes(module)
    assert given, "the syntax walk found the calls"
    for expr in given:
        if isinstance(expr, ast.Constant):
            assert expr.value in module.REASON_CODES, expr.value
        elif isinstance(expr, ast.BinOp):  # "item_delete_" + verdict
            assert isinstance(expr.left, ast.Constant) and expr.left.value == "item_delete_"
        else:
            assert isinstance(expr, ast.Name) and expr.id == "code", ast.dump(expr)
    assert must_have("check_reason_", module.REASON_CODES) == []


def test_every_why_code_is_declared_and_worded():
    whys = {
        v.value
        for node in ast.walk(tree(conformance))
        if isinstance(node, ast.Dict)
        for k, v in zip(node.keys, node.values, strict=True)
        if isinstance(k, ast.Constant) and k.value == "why" and isinstance(v, ast.Constant)
    }
    assert whys == set(conformance.WHY_CODES)
    assert must_have("check_why_", conformance.WHY_CODES) == []


def error_codes(path: Path) -> set[str]:
    """The string codes of every ``...Error(...)`` raised or built in ``path``: the first
    string argument, after an ``HTTPStatus`` when there is one."""
    codes = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and name_of(node.func).endswith("Error"):
            args = node.args
            if args and isinstance(args[0], ast.Attribute):  # HTTPStatus.X, then the code
                args = args[1:]
            first = args[0] if args and isinstance(args[0], ast.Constant) else None
            if first is not None and isinstance(first.value, str) and CODE.fullmatch(first.value):
                codes.add(first.value)
    return codes


def test_every_api_error_code_is_worded():
    codes = set().union(
        *(error_codes(p) for p in [*SRC.glob("webui/*.py"), *SRC.glob("config/*.py")])
    )
    codes |= {*CATALOGUE_STATUS, *EMOTIONS_STATUS}  # codes mapped from other modules' errors
    assert {"bad_pin", "csrf", "unknown_field", "live_disabled"} <= codes
    assert must_have("err_", codes) == []
