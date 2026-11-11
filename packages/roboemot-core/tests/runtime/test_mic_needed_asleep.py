"""SPEC 12A, Asleep: who needs the microphone while the platform sleeps.

One row per wake-word policy and voice profile, each with the expected answer, so a later
feature that wakes the platform by sound (a wake word policy, a sound reflex) has to change
``mic_needed_asleep`` and this table together."""

from __future__ import annotations

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.runtime.sleep import mic_needed_asleep

POLICIES = ["always", "never", "unless_enrolled_in_view"]
REALTIME = {"name": "rt", "base_url": "https://rt.example/v1", "model": "m", "kind": "realtime"}

PROFILES = {
    "voice off": {"voice": {"enabled": False}},
    "local voice": {"voice": {"enabled": True, "backend": "piper"}},
    "cloud voice": {
        "llm_routes": [REALTIME],
        "privacy": {"realtime_audio": True},
        "voice": {
            "enabled": True,
            "backend": "none",
            "fallback_backend": "none",
            "realtime_route": {"enabled": True, "route": "rt", "input": "audio"},
        },
    },
}


def config(profile: str, policy: str, pause: bool) -> RoboEmotConfig:
    raw = {**PROFILES[profile], "runtime": {"pause_when_asleep": pause}}
    raw["voice"] = {**raw["voice"], "wake_word_policy": policy}
    return RoboEmotConfig.model_validate(raw)


@pytest.mark.parametrize("profile", sorted(PROFILES))
@pytest.mark.parametrize("policy", POLICIES)
def test_with_the_pause_nothing_needs_the_microphone_while_the_platform_sleeps(profile, policy):
    assert mic_needed_asleep(config(profile, policy, pause=True)) is False


@pytest.mark.parametrize("profile", sorted(PROFILES))
@pytest.mark.parametrize("policy", POLICIES)
def test_without_the_pause_the_listener_runs_asleep_and_keeps_the_microphone(profile, policy):
    assert mic_needed_asleep(config(profile, policy, pause=False)) is True


def test_the_reflexes_alone_never_keep_the_microphone_while_the_pause_applies():
    cfg = RoboEmotConfig.model_validate(
        {"reflexes": {"enabled": True, "audio": {"enabled": True}}, "voice": {"enabled": False}}
    )
    assert mic_needed_asleep(cfg) is False
