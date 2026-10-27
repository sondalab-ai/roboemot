"""A realtime turn leaves no transcript, reply text or audio in the logs or the debug report
(SPEC 8, Realtime speech route; SPEC 12, 12A; the pattern of ``test_secret_leak.py``)."""

from __future__ import annotations

import base64
import json
import logging

from roboemot.diagnostics import Diagnostics, FnSource
from rt_rig import build_rt, wait_for
from turn_server import TurnServer, pcm

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
            report = (
                Diagnostics()
                .add(
                    FnSource("roboemot.realtime_route", rig.route.snapshot),
                    FnSource("roboemot.network", rig.network.snapshot),
                    FnSource(
                        "roboemot.voice",
                        lambda: {
                            "conversation": rig.worker.snapshot(),
                            "speech": rig.speech.snapshot(),
                        },
                    ),
                )
                .report()
            )
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
