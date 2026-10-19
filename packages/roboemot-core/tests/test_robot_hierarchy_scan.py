"""Self-tests of the hierarchy gate's scanners (SPEC 4A, 12A) on small synthetic sources."""

import textwrap
from collections import Counter

from robot_hierarchy_scan import (
    Finding,
    Terms,
    diff_baseline,
    duck_findings,
    import_findings,
    vocabulary_findings,
)

TERMS = Terms(
    static=("reachy", "antenna", "reachy_mini_wireless", "_reachy_mini._tcp"),
    tokens=("body_yaw", "head.pitch", "neck_pan"),
    ports=(8000,),
)


def vocab(src: str) -> set[tuple[str, str]]:
    found = vocabulary_findings(textwrap.dedent(src), "x.py", TERMS)
    return {(f.symbol, f.term) for f in found}


def test_a_robot_term_in_a_string_literal_is_flagged():
    assert vocab('NAME = "Reachy Mini"\n') == {("<module>", "reachy")}


def test_comments_and_docstrings_are_not_scanned():
    src = '''
    """Module about the antenna."""
    # reachy lives here
    class C:
        """Reachy docs."""
        def f(self):
            """antenna"""
            return 1
    '''
    assert vocab(src) == set()


def test_a_string_that_is_not_a_docstring_is_flagged_inside_its_symbol():
    src = """
    class C:
        def f(self):
            x = 1
            return "antenna_left"
    """
    assert vocab(src) == {("C.f", "antenna")}


def test_identifiers_match_static_terms_but_not_dof_tokens():
    src = """
    class GazeMap:
        body_yaw: float = 0.0
    def g(m):
        return m.body_yaw
    def antenna_pose():
        return 1
    """
    assert vocab(src) == {("antenna_pose", "antenna")}


def test_a_dof_name_is_flagged_only_as_a_whole_token_in_a_string():
    assert vocab('BODY_CHANNEL = "body_yaw"\n') == {("<module>", "body_yaw")}
    assert vocab('K = "head.pitch"\n') == {("<module>", "head.pitch")}
    assert vocab('K = "body_yaw_deg"\n') == set()
    assert vocab('K = "the head.pitch_x"\n') == set()
    assert vocab('K = "neck_pan, then"\n') == {("<module>", "neck_pan")}


def test_the_longer_term_hides_its_own_substrings():
    assert vocab('K = "reachy-mini-wireless"\n') == {("<module>", "reachy_mini_wireless")}


def test_a_port_is_flagged_only_inside_a_url_shaped_string():
    assert vocab('U = "http://localhost:8000/api"\n') == {("<module>", ":8000")}
    assert vocab("N = 8000\n") == set()
    assert vocab('N = "8000"\n') == set()


def imports(src: str) -> set[str]:
    return {f.term for f in import_findings(textwrap.dedent(src), "x.py", {"roboemot_x", "sdk"})}


def test_imports_of_a_robot_package_or_sdk_are_flagged_everywhere():
    src = """
    import roboemot_x.driver
    from sdk.media import Cam
    def f():
        import sdk
    from . import sdk_like
    import os
    """
    assert imports(src) == {"roboemot_x", "sdk"}


def ducks(src: str) -> set[tuple[str, str]]:
    return {(f.symbol, f.term) for f in duck_findings(textwrap.dedent(src), "x.py")}


def test_hasattr_and_getattr_with_a_literal_name_are_flagged_but_not_dynamic_names():
    src = """
    def f(src, name):
        if hasattr(src, "speaking"):
            return getattr(src, "camera_model", None)
        return getattr(src, name, None)
    """
    assert ducks(src) == {("f", "hasattr:speaking"), ("f", "getattr:camera_model")}


def finding(term: str) -> Finding:
    return Finding("vocabulary", "x.py", "<module>", term)


def test_a_finding_missing_from_the_baseline_is_new_and_an_unused_entry_is_stale():
    base = Counter({finding("a"): 1, finding("b"): 1})
    new, stale = diff_baseline(Counter({finding("a"): 1, finding("c"): 1}), base)
    assert new == Counter({finding("c"): 1})
    assert stale == Counter({finding("b"): 1})


def test_an_extra_occurrence_inside_a_baselined_key_is_new_and_a_lower_count_is_stale():
    base = Counter({finding("a"): 2})
    new, stale = diff_baseline(Counter({finding("a"): 3}), base)
    assert new == Counter({finding("a"): 1}) and not stale
    new, stale = diff_baseline(Counter({finding("a"): 1}), base)
    assert not new and stale == Counter({finding("a"): 1})


def test_an_equal_count_has_no_difference():
    new, stale = diff_baseline(Counter({finding("a"): 2}), Counter({finding("a"): 2}))
    assert not new and not stale


def test_repeated_occurrences_in_one_symbol_are_counted():
    src = """
    def f(x):
        return [hasattr(x, "a"), hasattr(x, "a")]
    """
    found = duck_findings(textwrap.dedent(src), "x.py")
    assert found == Counter({Finding("duck", "x.py", "f", "hasattr:a"): 2})
