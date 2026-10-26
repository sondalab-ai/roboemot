"""SPEC 7 (Speaking motion), 4A: the assembled runtime gives the speaking motion the speech
output's timeline (built after assembly, read lazily), the listening flag and the expression's
channels, on any robot whose configuration binds the roles to its own dofs."""

from __future__ import annotations

import numpy as np
from roboemot.affect.register import Person, resolve_register
from roboemot.config import ConfigStack
from roboemot.manifest import load_manifest
from roboemot.runtime.assembly import assemble
from roboemot.voice.envelope import SpeechTimeline
from test_runtime import CONFIGS, DaytimeHost, RecordingPlatform

RATE = 16000


def test_the_assembled_speaking_motion_reads_the_speech_output_lazily():
    stack = ConfigStack.from_files(preset="default", preset_dirs=[CONFIGS / "presets"])
    manifest = load_manifest(
        CONFIGS / "manifests" / "desk-head-v1.yaml",
        allow_unmeasured_dynamics=True,
        safety=stack.effective().safety,
    )
    platform = RecordingPlatform()
    host = DaytimeHost()
    p = assemble(host, stack, manifest, platform, platform.bind(object()))
    try:
        assert host.speaking is p.speaking
        assert p.speaking.commands(0.0) == []  # no speech output yet, no map: idle
        stack.set("speech_motion_map.beat", {"neck_tilt": 1}, source="test", layer="app")
        cfg = stack.effective()
        p.speaking.apply_config(
            cfg.speech_motion, cfg.speech_motion_map, cfg.expression.max_amplitude
        )
        p.speaking.set_register(resolve_register([Person({})], cfg.register, cfg.persona))
        tl = SpeechTimeline(cfg.speech_motion, lambda: 0.0)
        quiet, loud = (10 ** (db / 20) * np.ones(int(RATE * s), np.float32)
                       for db, s in ((-26, 1.0), (-14, 0.6)))  # fmt: skip
        x = np.concatenate([quiet, loud, quiet])
        tl.push(1, x, RATE, 0.0, utterance=1, question=False, mood=None, continued=False)
        tl.finish(1, x.size / RATE)

        class Output:
            timeline = tl

        host.speech_output = Output()
        moved, t = [], 0.0
        while t < 1.6:
            moved += [c.targets for c in p.speaking.commands(t)]
            t += 0.05
        assert moved and set().union(*moved) == {"neck_tilt"}
        assert max(m["neck_tilt"] for m in moved) > 0
    finally:
        p.perception.stop()
