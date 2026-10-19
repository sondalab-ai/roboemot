"""SPEC 4A, 11, 12A: core is robot-agnostic; robot specifics enter only through a core protocol
registered in ``roboemot.plugins.GROUPS`` and a ``roboemot.*`` entry point keyed by ``robot.id``.

Checks (all AST-based, comments and docstrings excluded; scanners in ``robot_hierarchy_scan``):

* import boundary: no module under ``roboemot`` imports a package that registers a ``roboemot.*``
  entry point (derived from the installed distributions) or a known robot SDK;
* robot vocabulary: identifiers and string literals in core, built from every manifest in
  ``configs/manifests``. Scope: every ``.py`` file of ``packages/roboemot-core/src/roboemot``, a
  superset of the behaviour modules of ``test_no_magic_numbers.py`` that also covers
  ``config/schema.py``, the installer and the CLI;
* entry-point contract: each installed ``roboemot.*`` group is in ``GROUPS`` and each registered
  object satisfies its group's contract;
* duck typing: no ``hasattr``/``getattr`` with a literal attribute name in robot packages, except
  in the modules that wrap the SDK handle itself (``SDK_ADAPTER_MODULES``);
* a non-Reachy manifest (``desk-head-v1``) works with core's config and renderer.

``robot_hierarchy_baseline.yaml`` lists today's findings (one per check, file, symbol, term, with
the audit id of the epic plan). A finding outside it fails, and so does an entry that no longer
occurs: the baseline only shrinks, and slices remove entries as they fix them.
"""

from __future__ import annotations

from importlib.metadata import distributions
from pathlib import Path

import pytest
import yaml
from roboemot.config.stack import ConfigError
from robot_hierarchy_scan import (
    Finding,
    build_terms,
    diff_baseline,
    duck_findings,
    import_findings,
    normalise,
    vocabulary_findings,
)

ROOT = Path(__file__).resolve().parents[3]
CORE_SRC = ROOT / "packages" / "roboemot-core" / "src" / "roboemot"
MANIFESTS = ROOT / "configs" / "manifests"
BASELINE = Path(__file__).with_name("robot_hierarchy_baseline.yaml")
KNOWN_SDKS = frozenset({"reachy_mini"})
SDK_ADAPTER_MODULES = frozenset(
    {"audio.py", "camera.py", "driver.py", "install.py", "settings.py", "speaker.py"}
)
AUDIT_IDS = frozenset({"1", "2", "3", "4", "5", "6", "B", "D", "U"})


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def roboemot_entry_points():
    """``(distribution name, entry point)`` for every ``roboemot.*`` entry point installed."""
    for dist in distributions():
        for ep in dist.entry_points:
            if ep.group.startswith("roboemot."):
                yield normalise(dist.metadata["Name"]), ep


def robot_packages() -> set[str]:
    """Top-level import names of the non-core distributions that register ``roboemot.*``."""
    return {
        ep.value.split(":")[0].split(".")[0]
        for dist, ep in roboemot_entry_points()
        if dist != "roboemot_core"
    }


def robot_sources() -> list[Path]:
    """Python files of every package under ``packages/`` other than core."""
    out: list[Path] = []
    for pkg in sorted((ROOT / "packages").iterdir()):
        if pkg.name != "roboemot-core":
            out += sorted((pkg / "src").rglob("*.py"))
    return out


def load_baseline() -> dict[Finding, str]:
    entries = yaml.safe_load(BASELINE.read_text()) or []
    out: dict[Finding, str] = {}
    for e in entries:
        key = Finding(e["check"], e["file"], e["symbol"], e["term"])
        assert key not in out, f"duplicate baseline entry {key}"
        out[key] = str(e["audit"])
    return out


def current_findings() -> set[Finding]:
    terms = build_terms(MANIFESTS)
    forbidden = robot_packages() | KNOWN_SDKS
    found: set[Finding] = set()
    for path in sorted(CORE_SRC.rglob("*.py")):
        text = path.read_text()
        found |= vocabulary_findings(text, rel(path), terms)
        found |= import_findings(text, rel(path), forbidden)
    for path in robot_sources():
        if path.name not in SDK_ADAPTER_MODULES:
            found |= duck_findings(path.read_text(), rel(path))
    return found


def test_the_baseline_entries_carry_a_known_audit_id():
    unknown = {k: a for k, a in load_baseline().items() if a not in AUDIT_IDS}
    assert not unknown, f"audit id must be one of {sorted(AUDIT_IDS)}: {unknown}"


def test_the_vocabulary_is_derived_from_the_manifests():
    terms = build_terms(MANIFESTS)
    assert {"reachy", "pollen", "antenna", "/api/apps"} <= set(terms.static)
    assert {"reachy-mini-wireless", "desk-head-v1", "_reachy-mini._tcp"} <= set(terms.static)
    assert {"head.pitch", "body_yaw", "antenna_left", "neck_pan"} <= set(terms.tokens)
    assert not {"head", "body", "voice", "eyes"} & (set(terms.static) | set(terms.tokens))
    assert "head_yaw" not in terms.tokens  # a retargeting source, not a DOF


def test_core_is_robot_agnostic_and_robot_packages_use_declared_capabilities():
    new, stale = diff_baseline(current_findings(), set(load_baseline()))
    assert not new, "new violations (fix them, never baseline them):\n" + "\n".join(
        f"  {f.check} {f.file} {f.symbol} {f.term}" for f in sorted(new)
    )
    assert not stale, "baseline entries that no longer occur (delete them):\n" + "\n".join(
        f"  {f.check} {f.file} {f.symbol} {f.term}" for f in sorted(stale)
    )


def test_the_import_boundary_covers_the_installed_robot_package():
    assert "roboemot_reachymini" in robot_packages()


def test_every_installed_entry_point_group_has_a_core_contract():
    from roboemot.plugins import GROUPS

    groups = {ep.group for _, ep in roboemot_entry_points()}
    assert groups <= set(GROUPS), f"groups without a core contract: {sorted(groups - set(GROUPS))}"


def test_the_registry_covers_the_groups_core_loads():
    from roboemot.plugins import GROUPS

    assert set(GROUPS) == {
        "roboemot.diagnostics",
        "roboemot.installers",
        "roboemot.world",
        "roboemot.settings",
    }


def test_every_registered_object_satisfies_its_group_contract():
    from roboemot.plugins import contract_violation

    broken = {
        f"{ep.group}:{ep.name}": why
        for _, ep in roboemot_entry_points()
        if (why := contract_violation(ep.group, ep.load()))
    }
    assert not broken, broken


def test_an_unregistered_group_or_a_wrong_object_breaks_the_contract():
    from roboemot.plugins import contract_violation

    assert "no core contract" in contract_violation("roboemot.nonsense", object())
    assert contract_violation("roboemot.settings", object()) is not None
    assert contract_violation("roboemot.world", lambda only_one: None) is not None
    assert contract_violation("roboemot.world", lambda a, b, c, d: None) is None
    assert contract_violation("roboemot.installers", int) is not None


@pytest.mark.xfail(
    strict=True,
    raises=ConfigError,
    reason="audit #1: EXPRESSION_CHANNELS whitelists Reachy DOF names (removed in slice B)",
)
def test_core_builds_config_and_renderer_for_a_non_reachy_manifest():
    from roboemot.affect.expression import ExpressionRenderer, build_primitive
    from roboemot.config.stack import ConfigStack
    from roboemot.manifest import load_manifest

    manifest = load_manifest(MANIFESTS / "desk-head-v1.yaml")
    sweep = [[0.0, 0], [0.3, 10], [0.6, 0]]
    stack = ConfigStack()
    stack.set_manifest_behaviour(
        {
            "expression_map": {"joy": {"channels": {"neck_pan": sweep}}},
            "gesture_map": {"nod": {"channels": {"neck_tilt": sweep}}},
            "circadian_motion": {"rest_pose_deg": {"neck_tilt": 10}},
        }
    )
    cfg = stack.effective()
    ExpressionRenderer(cfg.expression, cfg.expression_map, manifest.dofs, 0.15, cfg.gesture_map)
    assert build_primitive("joy", cfg.expression_map["joy"], manifest.dofs)
