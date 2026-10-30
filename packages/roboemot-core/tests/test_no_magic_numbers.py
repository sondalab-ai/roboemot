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
    "affect/appraisal.py",
    "affect/runner.py",
    "affect/register.py",
    "affect/expression.py",
    "affect/speaking.py",
    "circadian/model.py",
    "circadian/light.py",
    "circadian/motion.py",
    "circadian/replay.py",
    "circadian/runner.py",
    "circadian/sampler.py",
    "voice/prosody.py",
    "voice/asr.py",
    "voice/keywords.py",
    "voice/grammar.py",
    "voice/robot_name.py",
    "voice/wake.py",
    "voice/direction.py",
    "voice/facing.py",
    "voice/speech_features.py",
    "voice/fastpath.py",
    "voice/listener.py",
    "voice/listener_load.py",
    "voice/vad.py",
    "voice/models.py",
    "voice/activity.py",
    "voice/bargein.py",
    "voice/duck.py",
    "voice/tts.py",
    "voice/piper_backend.py",
    "voice/piper_worker.py",
    "voice/openai_speech.py",
    "voice/speech.py",
    "voice/envelope.py",
    "voice/yes_no.py",
    "voice/factory.py",
    "voice/pipeline.py",
    "voice/keyword_worker.py",
    "voice/grammar_worker.py",
    "voice/decode_worker.py",
    "voice/cues.py",
    "priority.py",
    "voice/sample.py",
    "llm/router.py",
    "llm/conformance.py",
    "llm/prompt.py",
    "llm/hub.py",
    "skills/schema.py",
    "skills/registry.py",
    "skills/defaults.py",
    "conversation/worker.py",
    "conversation/summaries.py",
    "conversation/sources.py",
    "skills/bridge.py",
    "runtime/assembly.py",
    "runtime/diagnostics.py",
    "runtime/host.py",
    "runtime/levels.py",
    "runtime/live.py",
    "runtime/live_video.py",
    "conversation/transcript.py",
    "conversation/phase.py",
    "runtime/loop.py",
    "runtime/reload.py",
    "runtime/sleep.py",
    "runtime/reports.py",
    "runtime/runtime.py",
    "runtime/startup.py",
    "runtime/ticks.py",
    "runtime/webui.py",
    "conversation/lines.py",
    "conversation/confirm.py",
    "conversation/preferences.py",
    "soothe/__init__.py",
    "soothe/synth.py",
    "soothe/runner.py",
    "persona/__init__.py",
    "persona/llm_traits.py",
    "motion/loop.py",
    "motion/external.py",
    "motion/primitives.py",
    "reflex/__init__.py",
    "reflex/engine.py",
    "attention/__init__.py",
    "attention/selector.py",
    "attention/gaze.py",
    "world/__init__.py",
    "world/frame.py",
    "world/model.py",
    "health/__init__.py",
    "health/network.py",
    "health/responder.py",
    "registry/__init__.py",
    "registry/registry.py",
    "registry/records.py",
    "registry/store.py",
    "registry/matcher.py",
    "registry/enrollment.py",
    "registry/xmp.py",
    "registry/images.py",
    "registry/catalogue.py",
    "registry/tombstones.py",
    "world/autoenroll.py",
    "webui/catalogue.py",
    "world/feed.py",
    "world/scan.py",
    "world/sources.py",
    "webui/robot_section.py",
    "paths.py",
    "vision/__init__.py",
    "vision/facing.py",
    "vision/geometry.py",
    "vision/governor.py",
    "vision/models.py",
    "vision/onnx.py",
    "vision/pipeline.py",
    "vision/replay.py",
    "vision/runner.py",
    "vision/service.py",
    "vision/stages.py",
    "vision/track_stats.py",
    "vision/tracker.py",
    "vision/types.py",
    "webui/api.py",
    "webui/auth.py",
    "webui/server.py",
    "webui/settings.py",
    "webui/setting_clean.py",
    "config/catalogue.py",
    "webui/validation.py",
    "realtime/__init__.py",
    "realtime/protocol.py",
    "realtime/session.py",
    "realtime/client.py",
    "realtime/route.py",
    "realtime/turn.py",
    "realtime/input.py",
    "realtime/check.py",
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
