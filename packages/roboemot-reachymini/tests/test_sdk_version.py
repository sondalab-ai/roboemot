"""The app refuses an SDK older than the one it was written for (SPEC 4A)."""

import pytest
from roboemot_reachymini import app as app_module
from roboemot_reachymini import platform as platform_module
from roboemot_reachymini import sdk
from test_app_perception import data  # noqa: F401  (data: fixture)


@pytest.mark.parametrize("installed", ["1.12.0", "1.12", "1.12.1", "1.13.0rc1", "2.0.0"])
def test_a_new_enough_sdk_passes(installed):
    sdk.require_sdk(installed)


@pytest.mark.parametrize("installed", ["1.11.0", "1.9.9", "0.9"])
def test_an_older_sdk_fails_loud_with_the_remedy(installed):
    with pytest.raises(RuntimeError, match=rf"reachy_mini {installed} is older than 1\.12.*update"):
        sdk.require_sdk(installed)


def test_a_missing_sdk_fails_too():
    with pytest.raises(RuntimeError, match="reachy_mini is not installed"):
        sdk.require_sdk(None)


def test_installed_version_reads_the_distribution_metadata(monkeypatch):
    monkeypatch.setattr(sdk, "version", lambda name: "1.12.0" if name == "reachy_mini" else "?")
    assert sdk.installed_version() == "1.12.0"


def test_run_checks_the_sdk_before_building_anything(data, monkeypatch):  # noqa: F811
    built = []
    monkeypatch.setattr(platform_module, "installed_version", lambda: "1.11.0")
    monkeypatch.setattr(platform_module, "ReachyMiniDriver", lambda client: built.append(client))
    with pytest.raises(RuntimeError, match="older than 1.12"):
        app_module.RoboEmotApp().run(object(), None)
    assert built == []


@pytest.mark.parametrize(
    ("installed", "newer"),
    [("1.12.0", False), ("1.12.9", False), ("1.11.0", False), ("1.13.0", True), ("2.0", True)],
)
def test_newer_than_tested_compares_major_minor(installed, newer):
    assert sdk.newer_than_tested(installed) is newer


def test_a_newer_sdk_starts_with_a_warning(monkeypatch, caplog):
    """R4 S10: a daemon update past the tested SDK must not stop the app, but it is logged."""
    monkeypatch.setattr(platform_module, "installed_version", lambda: "1.13.0")
    platform_module.ReachyMiniPlatform().require_runtime()
    assert "newer than 1.12" in caplog.text


def test_the_sim_extra_pins_the_tested_sdk():
    """Review S-8: the sim pin and TESTED_SDK are written in two places; keep them together."""
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    (sim,) = project["project"]["optional-dependencies"]["sim"]
    major, minor = map(int, sdk.TESTED_SDK.split("."))
    assert sim.endswith(f">={sdk.MIN_SDK},<{major}.{minor + 1}"), sim
