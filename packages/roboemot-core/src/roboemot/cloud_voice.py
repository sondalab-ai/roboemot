"""The cloud-voice profile (SPEC 4A, Installer; SPEC 8, Realtime speech route; SPEC 11A): the
platform's realtime route and the keys the profile sets. A light module with no installer
dependency: the installer writes the keys into ``app.yaml`` and the web UI's voice switch offers
them as the platform's choice, from this one source."""

from __future__ import annotations

from dataclasses import dataclass

ROUTES_KEY = "llm_routes"  # a list keyed by route name: merged per entry, never whole


@dataclass(frozen=True)
class CloudVoiceRoute:
    """The realtime speech route a platform offers for the cloud-voice profile (SPEC 4A,
    Installer; SPEC 8, Realtime speech route): the ``llm_routes`` entry (``name``, ``kind``,
    ``base_url``, ``model``; its key is ``${secret:<secret>}`` from the secret store), the
    ``voice.realtime_route.mode`` it needs, ``provider``, who receives the audio, as the
    opt-in question names it, and the transcript timeout that suits that server."""

    name: str
    kind: str
    base_url: str
    secret: str
    model: str
    mode: str
    provider: str
    transcript_timeout_s: float  # voice.realtime_route.transcript_timeout_s the profile writes

    def entry(self) -> dict[str, str]:
        """The ``llm_routes`` entry as the installer writes it into ``app.yaml``."""
        return {
            "name": self.name,
            "kind": self.kind,
            "base_url": self.base_url,
            "api_key": f"${{secret:{self.secret}}}",
            "model": self.model,
        }


def cloud_voice_config(route: CloudVoiceRoute) -> dict[tuple[str, ...], object]:
    """The keys the cloud-voice profile sets in ``app.yaml`` (SPEC 4A, Installer; SPEC 8, 12):
    the platform's route entry; the realtime route through it, with the utterance's audio as its
    input (``privacy.realtime_audio``, the admin opt-in), the skip tool and the tone tool; no local
    voice; wake policy ``never`` with the voice-at-face and facing checks (facing needs
    direction). The one cloud-voice consent also covers audio the robot did not judge addressed
    (``privacy.realtime_audio_unaddressed``, owner decision 2026-11-05), so server turns can be
    switched on after QA; ``voice.realtime_route.turns`` is not the profile's. The transcript
    timeout is the platform's (``CloudVoiceRoute.transcript_timeout_s``)."""
    rt = ("voice", "realtime_route")
    return {
        (ROUTES_KEY, route.name): route.entry(),
        (*rt, "enabled"): True,
        (*rt, "route"): route.name,
        (*rt, "mode"): route.mode,
        (*rt, "input"): "audio",
        (*rt, "skip_tool"): True,
        (*rt, "tone_tool"): True,
        (*rt, "transcript_timeout_s"): route.transcript_timeout_s,
        ("privacy", "realtime_audio"): True,
        ("privacy", "realtime_audio_unaddressed"): True,
        ("voice", "backend"): "none",
        ("voice", "fallback_backend"): "none",
        ("voice", "wake_word_policy"): "never",
        ("voice", "direction", "enabled"): True,
        ("voice", "facing", "enabled"): True,
    }
