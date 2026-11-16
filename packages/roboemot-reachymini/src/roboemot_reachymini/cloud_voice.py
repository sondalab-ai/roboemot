"""Reachy Mini's cloud-voice route (SPEC 4A, Installer; SPEC 8): Pollen's realtime server,
reached through its allocator on Hugging Face in platform mode. A light module: the installer
platform and the web UI settings plugin both read it, and the web UI never loads the installer."""

from __future__ import annotations

from roboemot.cloud_voice import CloudVoiceRoute

CLOUD_VOICE = CloudVoiceRoute(
    name="pollen",
    kind="realtime",
    base_url="https://pollen-robotics-reachy-mini-realtime-url.hf.space/session",
    secret="hf_token",
    model="s2s",
    mode="platform",
    provider="Pollen's server (Hugging Face)",
    # The owner's hand-set value (robot, 2026-10-30): the transcript p50 was 8.3 s before CV-0;
    # lower it to the measured p95 plus about 1 s once CV-0 is measured on the robot.
    transcript_timeout_s=10.0,
)
