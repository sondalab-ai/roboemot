"""SPEC 11A: no behavioural constant hard-coded in behaviour modules.

Numeric literals are allowed only if they are structural (0, 1, -1, 2, 0.5), unit
conversions, or module-level named constants (UPPER_CASE assignments), which document a
fixed fact rather than a tunable behaviour (e.g. luma weights, seconds per hour).
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "roboemot"
BEHAVIOUR_MODULES = [
    "affect/__init__.py",
    "circadian/model.py",
    "circadian/light.py",
    "voice/prosody.py",
    "llm/router.py",
    "motion/loop.py",
    "motion/primitives.py",
    "health/__init__.py",
    "health/network.py",
    "health/responder.py",
]
ALLOWED = {0, 1, -1, 2, 0.5, -1.0}


def literals_outside_constants(tree: ast.Module):
    const_nodes = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if all(isinstance(t, ast.Name) and t.id.isupper() for t in targets):
                const_nodes.update(id(n) for n in ast.walk(node))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and id(node) not in const_nodes
            and node.value not in ALLOWED
        ):
            yield node


def test_behaviour_modules_have_no_magic_numbers():
    offenders = []
    for rel in BEHAVIOUR_MODULES:
        tree = ast.parse((SRC / rel).read_text())
        offenders += [f"{rel}:{n.lineno}: {n.value!r}" for n in literals_outside_constants(tree)]
    assert not offenders, "move these into the config schema (SPEC 11A):\n" + "\n".join(offenders)
