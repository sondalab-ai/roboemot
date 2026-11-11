"""SPEC 11A: a merged tree shares nothing with its layers, and the base is copied once per merge.

``deep_merge`` and ``ConfigStack.merged_tree`` run several times per settings save; the result
must stay a private tree (editing it never reaches a layer or the schema defaults), while a
nested section is not copied again at every level.
"""

from __future__ import annotations

import copy

from roboemot.config import stack as stack_module
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import ConfigStack, deep_merge


def nested(depth: int) -> dict:
    tree: dict = {"leaf": [1, {"x": 2}]}
    for n in range(depth):
        tree = {f"k{n}": tree, f"side{n}": {"v": [n]}}
    return tree


def test_the_result_shares_no_container_with_base_or_over():
    base, over = nested(6), {"k5": {"k4": {"new": {"a": [1]}}, "side4": {"v": [9]}}}
    saved_base, saved_over = copy.deepcopy(base), copy.deepcopy(over)
    out = deep_merge(base, over)
    assert out["k5"]["k4"]["new"] == {"a": [1]} and out["k5"]["side4"] == {"v": [9]}
    out["k5"]["k4"]["new"]["a"].append(2)
    out["k5"]["k4"]["side3"]["v"].append(99)
    out["k5"]["side4"]["v"].append(3)
    assert base == saved_base and over == saved_over


def test_a_nested_base_is_copied_once(monkeypatch):
    calls: list[int] = []
    real = copy.deepcopy

    def counting(value, memo=None):
        if memo is None:  # a top-level call, not deepcopy's own recursion
            calls.append(1)
        return real(value, memo)

    base = nested(8)
    over = {"k7": {"k6": {"k5": {"k4": {"leaf": [0]}}}}}  # four levels merge on the way down
    monkeypatch.setattr(stack_module.copy, "deepcopy", counting)
    deep_merge(base, over)
    assert len(calls) <= 2  # the base once, the replaced leaf once


def test_merged_tree_never_hands_out_the_schema_defaults():
    stack = ConfigStack()
    first = stack.merged_tree()
    first["safety"]["edited"] = True
    first["llm_routes"].append({"name": "leak"})
    second = stack.merged_tree()
    assert second == RoboEmotConfig().model_dump(mode="python")
    assert "edited" not in second["safety"]


def test_merged_tree_shares_nothing_with_a_layer():
    app = {"affect": {"half_life_s": 12.0}, "voice": {"keywords": {"en": {"no": ["nope"]}}}}
    derived = {"voice": {"moods": {"calm": {"instructions": "slow"}}}}
    stack = ConfigStack(app=copy.deepcopy(app), derived=copy.deepcopy(derived))
    tree = stack.merged_tree()
    assert tree["affect"]["half_life_s"] == 12.0
    tree["affect"]["half_life_s"] = 1.0
    tree["voice"]["keywords"]["en"]["no"].append("edited")
    tree["voice"]["moods"]["calm"]["edited"] = True
    assert stack.app == app and stack.derived == derived
    assert stack.merged_tree()["affect"]["half_life_s"] == 12.0


def test_a_merge_of_empty_layers_never_copies_the_whole_tree(monkeypatch):
    stack = ConfigStack()
    calls: list[int] = []
    real = copy.deepcopy

    def counting(value, memo=None):
        if memo is None and isinstance(value, dict) and "safety" in value:
            calls.append(1)
        return real(value, memo)

    monkeypatch.setattr(stack_module.copy, "deepcopy", counting)
    stack.merged_tree()
    assert calls == []
