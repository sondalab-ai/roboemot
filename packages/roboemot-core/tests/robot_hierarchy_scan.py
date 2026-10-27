"""AST scanners behind ``test_robot_hierarchy.py`` (SPEC 4A, 12A).

Everything works on parsed syntax: comments never reach the AST, and docstrings are skipped
explicitly. A finding is keyed by check, repo-relative file, enclosing symbol and term, never by
line number, so edits elsewhere in a file do not move baseline entries.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

GENERIC_WORDS = frozenset({"head", "body", "voice", "eyes", "base_link"})
STATIC_TERMS = ("reachy", "pollen", "antenna", "/api/apps")


@dataclass(frozen=True, order=True)
class Finding:
    """One violation; ``check`` is vocabulary, import or duck."""

    check: str
    file: str
    symbol: str
    term: str


@dataclass(frozen=True)
class Terms:
    """``static`` terms match as substrings of identifiers and strings, ``tokens`` only as whole
    tokens of strings, ``ports`` only inside URL-shaped strings."""

    static: tuple[str, ...]
    tokens: tuple[str, ...]
    ports: tuple[int, ...] = ()


def normalise(text: str) -> str:
    """Lower case with ``-`` and spaces folded to ``_``, so ids, SDK names and prose compare."""
    return re.sub(r"[-\s]+", "_", text.lower())


def _walk_scoped(tree: ast.Module) -> Iterator[tuple[ast.AST, str]]:
    """Every node with the qualified name of its enclosing class or function (``<module>`` at
    top level). A definition's own name belongs to its own symbol."""

    def visit(node: ast.AST, scope: tuple[str, ...]) -> Iterator[tuple[ast.AST, str]]:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = (*scope, node.name)
        yield node, ".".join(scope) or "<module>"
        for child in ast.iter_child_nodes(node):
            yield from visit(child, scope)

    return visit(tree, ())


def _docstring_ids(tree: ast.Module) -> set[int]:
    """``id`` of every docstring constant node (module, class and function level)."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
        ):
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return ids


def _identifier(node: ast.AST) -> str | None:
    """The name a node introduces or uses, if it is an identifier node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.keyword):
        return node.arg
    return None


def _drop_nested(matched: set[str]) -> set[str]:
    """Remove terms that are a substring of another matched term (keep the most specific)."""
    return {t for t in matched if not any(t != o and t in o for o in matched)}


def _token_pattern(term: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w.])" + re.escape(term) + r"(?!\w)")


def vocabulary_findings(source: str, rel: str, terms: Terms) -> Counter[Finding]:
    """Robot vocabulary in identifiers and string literals of ``source``."""
    tree = ast.parse(source)
    skip = _docstring_ids(tree)
    static = tuple(normalise(t) for t in terms.static)
    patterns = [(t, _token_pattern(t)) for t in terms.tokens]
    found: Counter[Finding] = Counter()
    for node, symbol in _walk_scoped(tree):
        matched: set[str] = set()
        ident = _identifier(node)
        if ident is not None:
            low = normalise(ident)
            matched |= {t for t in static if t in low}
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
        ):
            value = node.value
            low = normalise(value)
            matched |= {t for t in static if t in low}
            matched |= {t for t, pat in patterns if pat.search(value)}
            if "://" in value or "localhost" in value:
                matched |= {f":{p}" for p in terms.ports if re.search(rf":{p}(?!\d)", value)}
        for term in _drop_nested(matched):
            found[Finding("vocabulary", rel, symbol, term)] += 1
    return found


def import_findings(source: str, rel: str, forbidden: set[str]) -> Counter[Finding]:
    """Imports of a module whose top-level package is in ``forbidden`` (relative imports never
    are)."""
    found: Counter[Finding] = Counter()
    for node, symbol in _walk_scoped(ast.parse(source)):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        for module in modules:
            top = module.split(".")[0]
            if top in forbidden:
                found[Finding("import", rel, symbol, top)] += 1
    return found


def duck_findings(source: str, rel: str) -> Counter[Finding]:
    """``hasattr``/``getattr`` calls with a string-literal attribute name."""
    found: Counter[Finding] = Counter()
    for node, symbol in _walk_scoped(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("hasattr", "getattr")
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            found[Finding("duck", rel, symbol, f"{node.func.id}:{node.args[1].value}")] += 1
    return found


def diff_baseline(
    findings: Counter[Finding], baseline: Counter[Finding]
) -> tuple[Counter[Finding], Counter[Finding]]:
    """``(new, stale)`` as surplus counts: occurrences beyond the baseline (a missing key counts
    in full) and baseline occurrences no longer found."""
    return findings - baseline, baseline - findings


def _ports(raw: Any) -> Iterator[int]:
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key == "port" and isinstance(value, int):
                yield value
            else:
                yield from _ports(value)
    elif isinstance(raw, list):
        for item in raw:
            yield from _ports(item)


def _paths(raw: Any) -> Iterator[str]:
    if isinstance(raw, dict):
        for value in raw.values():
            yield from _paths(value)
    elif isinstance(raw, list):
        for item in raw:
            yield from _paths(item)
    elif isinstance(raw, str) and raw.startswith("/api"):
        yield raw


def build_terms(manifest_dir: Path) -> Terms:
    """The robot vocabulary: static terms, plus per manifest the robot id, driver and product
    names (mDNS service and TXT values), DOF and joint names, daemon ports and URL paths.
    Generic words are never terms; retargeting sources are human-pose names, not DOFs."""
    from roboemot.manifest import load_manifest

    static = set(STATIC_TERMS)
    tokens: set[str] = set()
    ports: set[int] = set()
    for path in sorted(manifest_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        robot = raw["robot"]
        static.add(robot["id"])
        if robot.get("driver"):
            static.add(robot["driver"])
        discovery = robot.get("discovery") or {}
        if discovery.get("mdns_service"):
            static.add(discovery["mdns_service"])
        static.update((discovery.get("match_txt") or {}).values())
        manifest = load_manifest(path, allow_unmeasured_dynamics=True)
        tokens.update(manifest.dofs)
        tokens.update(j["name"] for j in raw.get("joints", []))
        for name, group in (raw.get("pose_groups") or {}).items():
            tokens.add(name)
            tokens.update(group.get("names", []))
        ports.update(_ports(raw))
        static.update(_paths(raw))
    static = {t for t in static if normalise(t) not in GENERIC_WORDS}
    tokens = {t for t in tokens if t not in GENERIC_WORDS}
    return Terms(tuple(sorted(static)), tuple(sorted(tokens)), tuple(sorted(ports)))
