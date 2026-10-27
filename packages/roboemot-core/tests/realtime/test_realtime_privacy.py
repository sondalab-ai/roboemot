"""A realtime turn leaves no transcript, reply text or audio in the logs or the debug report
(SPEC 8, Realtime speech route; SPEC 12, 12A; the pattern of ``test_secret_leak.py``)."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from roboemot.manifest import load_manifest
from roboemot.runtime import RuntimeHost
from roboemot.runtime.diagnostics import build_diagnostics
from rt_rig import build_rt, wait_for
from turn_server import TurnServer, pcm

CONFIGS = Path(__file__).resolve().parents[4] / "configs"


def real_report(rig):
    """The runtime's own debug report (``build_diagnostics``) over the rig's parts, with the
    other parts as empty stand-ins."""
    host = RuntimeHost()
    host.conversation, host.speech_output, host.realtime_route = rig.worker, rig.speech, rig.route
    empty = SimpleNamespace(snapshot=lambda *a: {})
    p = SimpleNamespace(
        listener=empty,
        bargein=None,
        runner=SimpleNamespace(engine=rig.worker.affect),
        loop=empty,
        sleep=empty,
        network=rig.network,
        perception=SimpleNamespace(vision_snapshot=dict, registry_snapshot=dict),
        world=SimpleNamespace(entries=[]),
        tracker=SimpleNamespace(active=None),
        circ=empty,
        reflexes=empty,
        speaking=empty,
    )
    manifest = load_manifest(
        CONFIGS / "manifests" / "desk-head-v1.yaml",
        allow_unmeasured_dynamics=True,
        safety=rig.stack.effective().safety,
    )
    report = build_diagnostics(host, rig.stack, manifest, p).report()
    assert report["sections"]["roboemot.realtime_route"]["ok"] is True
    assert report["sections"]["roboemot.voice"]["ok"] is True  # conversation and speech too
    return report


UTTERANCE = "il mio segreto si chiama Zorbaflux"
REPLY = "Zorbaflux risponde con Quintavalle"


def test_a_turn_leaves_no_text_or_audio_in_logs_or_the_report(caplog):
    caplog.set_level(logging.DEBUG)
    with TurnServer(transcript=REPLY) as server:
        rig = build_rt(server)
        try:
            rig.turn(UTTERANCE)
            assert wait_for(lambda: rig.speech.snapshot()["spoken"] == 1)
            rig.worker.mark_interrupting()
            rig.speech.stop()
            rig.worker.interrupt()
            report = real_report(rig)
        finally:
            rig.close()
    assert "realtime route rt: connected" in caplog.text  # the logs were captured
    blob = json.dumps(report)
    audio = pcm(server.delta_ms)
    for text in (caplog.text, blob):
        for secret in ("Zorbaflux", "Quintavalle", "segreto"):
            assert secret not in text
        assert base64.b64encode(audio).decode()[:40] not in text
        assert repr(audio[:16])[2:-1] not in text
        assert "/v1/realtime" not in text  # the endpoint is scheme://host:port only
    section = report["sections"]["roboemot.realtime_route"]
    assert "Zorbaflux" not in json.dumps(section)
