"""SPEC 11A: a config change reaches the parts that hold a copy (``make_reload``). The pipeline is
a bundle of recording fakes, so the fan-out is observed without starting a runtime."""

from __future__ import annotations

from types import SimpleNamespace

from roboemot.config.schema import RoboEmotConfig
from roboemot.runtime import RuntimeHost
from roboemot.runtime.reload import make_reload


class Recorder:
    """Records every method call as ``(name, args)``; attribute writes stay on the object."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kw: self.calls.append((name, args))


def pipeline() -> SimpleNamespace:
    return SimpleNamespace(
        hub=Recorder(), loop=Recorder(), presence=Recorder(),
        world=SimpleNamespace(cfg=None), selector=SimpleNamespace(cfg=None),
        gaze=SimpleNamespace(cfg=None, margin_rad=0.0), scan=None, plugin=None,
        perception=Recorder(), runner=Recorder(), circ=Recorder(), speech=Recorder(),
        yes_no=Recorder(), listener=None, bargein=None, reflexes=Recorder(),
        tracker=SimpleNamespace(cfg=None), expression=Recorder(),
        responder=SimpleNamespace(cfg=None, language=None), network=Recorder(),
    )  # fmt: skip


def test_a_config_change_reaches_every_part_that_holds_a_copy():
    p, host = pipeline(), RuntimeHost()
    host.speech_output = SimpleNamespace(config=None)
    host.conversation = Recorder()
    cfg = RoboEmotConfig()
    make_reload(host, p)(SimpleNamespace(config=cfg))
    assert p.world.cfg is cfg.world and p.selector.cfg is cfg.attention is p.gaze.cfg
    assert p.tracker.cfg is cfg.expression and p.responder.cfg is cfg.connectivity
    assert p.responder.language == cfg.conversation.language
    assert host.speech_output.config is cfg
    assert ("sync", (cfg,)) in p.hub.calls and ("apply_config", (cfg,)) in p.loop.calls
    assert ("apply_config", (cfg,)) in p.perception.calls
    assert ("apply_config", (cfg,)) in host.conversation.calls
    assert ("set_echo_tail_ms", (cfg.voice.echo_tail_ms,)) in p.speech.calls
    assert ("set_policy", (cfg.connectivity.network,)) in p.network.calls
    assert ("set_routes", (cfg.llm_routes,)) in p.network.calls


def test_optional_parts_that_do_not_exist_are_skipped():
    p, host = pipeline(), RuntimeHost()
    make_reload(host, p)(SimpleNamespace(config=RoboEmotConfig()))  # no listener, scan, speech


def test_the_live_transcript_follows_the_flag_on_reload():
    p, host = pipeline(), RuntimeHost()
    host.transcript = Recorder()
    cfg = RoboEmotConfig()
    make_reload(host, p)(SimpleNamespace(config=cfg))
    assert ("apply_config", (cfg.webui.live,)) in host.transcript.calls
