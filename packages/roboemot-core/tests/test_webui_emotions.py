"""SPEC 7, 11, 11A, 12: the Emotions tab. ``GET /api/emotions/state`` shows the affect, the
register, the persona, the circadian state and the impulses that moved the affect, from parts in
memory; ``POST /api/emotions/play`` and ``/inject`` are writes behind the PIN session, CSRF and
the PIN, refused while the robot is asleep, its motors are off or something else moves it."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from roboemot.affect import AffectEngine, AffectState
from roboemot.affect.clips import load_clip_library
from roboemot.affect.expression import ExpressionRenderer
from roboemot.affect.register import ActiveRegister, Person, resolve_register
from roboemot.config.schema import RegisterParams, RoboEmotConfig
from roboemot.config.stack import ConfigStack
from roboemot.diagnostics import Diagnostics
from roboemot.manifest import load_manifest
from roboemot.runtime.emotions import REFUSALS, EmotionsError, EmotionsState
from roboemot.webui import make_server
from roboemot.webui.api import EMOTIONS_STATUS, PASSIVE_POLLS
from roboemot.webui.settings import SettingsEditor
from roboemot.webui.strings import STRINGS
from test_webui import PIN, Client, FakeService

NOW = 100.0
SECRET_ID = "9d41-registry-private-id"


class Renderer:
    """The part of ``ExpressionRenderer`` the tab uses; records what was asked for."""

    def __init__(self) -> None:
        self.expression_map = {"joy": None, "calm": None}
        self.asked: list[str] = []
        self.accept = True
        self.last_label, self.last_amplitude, self.started = "joy", 0.4321, 3

    def playable(self) -> list[str]:
        return ["joy", "calm", "nod"]

    def request(self, label: str, now: float | None = None) -> bool:
        assert now is None, "the play skill's path: no `now`"
        self.asked.append(label)
        return self.accept


def pipeline(cfg: RoboEmotConfig):
    engine = AffectEngine(cfg.affect, cfg.appraisal)
    register = ActiveRegister(RegisterParams(), people=2)
    return SimpleNamespace(
        runner=SimpleNamespace(engine=engine),
        tracker=SimpleNamespace(active=register),
        circ=SimpleNamespace(
            snapshot=lambda: {"state": "drowsy", "sleepiness": 0.61234},
            amplitude_factor=lambda: 0.5,
        ),
        expression=Renderer(),
        sleep=SimpleNamespace(asleep=False),
        loop=SimpleNamespace(motors_off=False, external=False, unreachable_held=False),
    )


def view(cfg=None, **host):
    cfg = cfg or RoboEmotConfig()
    p = pipeline(cfg)
    return EmotionsState(SimpleNamespace(circadian=object(), **host), p, cfg, clock=lambda: NOW), p


def rule_index(cfg: RoboEmotConfig, event: str) -> int:
    return next(i for i, r in enumerate(cfg.appraisal.rules) if r.event == event)


def test_the_state_holds_codes_and_numbers_only():
    emo, _ = view()
    s = emo.state()
    assert set(s) == {
        "poll_ms", "affect", "register", "persona", "circadian", "expression", "moved",
        "controls", "playable", "rules",
    }  # fmt: skip
    assert s["affect"]["label"] in RoboEmotConfig().affect.labels
    assert s["register"]["people"] == 2 and "persona_fragment" not in s["register"]
    assert s["persona"]["effects"]["motion_amplitude_x"] > 0
    assert s["circadian"] == {"state": "drowsy", "sleepiness": 0.612, "amplitude": 0.5}
    assert s["expression"] == {"last_label": "joy", "last_amplitude": 0.432, "started": 3}
    assert s["playable"] == {"labels": ["joy", "calm"], "gestures": ["nod"]}
    assert s["controls"] == {"available": True, "refused": None}
    assert {"index": 0, "event": "person.known", "label": "calm"} in s["rules"]


def test_without_a_circadian_rhythm_there_is_no_circadian_block():
    cfg = RoboEmotConfig()
    emo = EmotionsState(SimpleNamespace(circadian=None), pipeline(cfg), cfg, clock=lambda: NOW)
    assert emo.state()["circadian"] is None


def test_an_injected_rule_moves_the_affect_and_shows_with_its_delta():
    cfg = RoboEmotConfig()
    emo, p = view(cfg)
    index = rule_index(cfg, "sound.loud")
    assert emo.inject(index) is True
    (moved,) = emo.state()["moved"]
    assert moved["event"] == "sound.loud" and moved["rule"] == index
    assert moved["label"] == "surprise" and moved["dv"] == 0 and moved["da"] == 0.6
    assert emo.inject(index) is False, "the rule's cooldown applies to a test too"
    assert len(emo.state()["moved"]) == 1


def test_the_history_never_holds_a_payload_and_is_bounded():
    cfg = RoboEmotConfig.model_validate({"webui": {"emotions": {"events": 5}}})
    emo, p = view(cfg)
    engine = p.runner.engine
    for n in range(8):
        engine.on_event("person.known", NOW + n * 100, {"person_id": SECRET_ID})
    moved = emo.state()["moved"]
    assert len(moved) == 5 and moved[0]["age_s"] == 0
    assert SECRET_ID not in json.dumps(emo.state())


def test_the_history_is_kept_only_while_the_tab_is_on():
    cfg = RoboEmotConfig.model_validate({"webui": {"emotions": {"enabled": False}}})
    emo, p = view(cfg)
    p.runner.engine.on_event("sound.loud", NOW)
    assert p.runner.engine.history() == [] and emo.state() is None
    emo.apply_config(SimpleNamespace(config=RoboEmotConfig()))
    p.runner.engine.on_event("person.known", NOW)
    assert len(emo.state()["moved"]) == 1


def test_an_llm_impulse_shows_as_the_conversation():
    emo, p = view()
    p.runner.engine.llm_impulse(NOW, 0.1, 0.0)
    (moved,) = emo.state()["moved"]
    assert moved["event"] == "llm" and moved["rule"] is None and moved["dv"] == 0.1


def test_play_takes_the_play_skills_path():
    emo, p = view()
    assert emo.play("nod") is True and p.expression.asked == ["nod"]
    p.expression.accept = False
    assert emo.play("joy") is False
    with pytest.raises(EmotionsError) as err:
        emo.play("cue_listening")
    assert err.value.code == "unknown_expression"


@pytest.mark.parametrize(
    ("part", "code"),
    [("asleep", "robot_asleep"), ("motors_off", "motors_off"), ("external", "external_control")],
)
def test_the_controls_are_refused_while_the_robot_cannot_take_them(part, code):
    emo, p = view()
    target = p.sleep if part == "asleep" else p.loop
    setattr(target, part, True)
    assert emo.state()["controls"] == {"available": False, "refused": code}
    for call in (lambda: emo.play("joy"), lambda: emo.inject(0)):
        with pytest.raises(EmotionsError) as err:
            call()
        assert err.value.code == code
    assert p.expression.asked == [] and p.runner.engine.history() == []


def test_an_unknown_rule_is_refused():
    emo, _ = view()
    with pytest.raises(EmotionsError) as err:
        emo.inject(999)
    assert err.value.code == "unknown_rule"


# ---------------------------------------------------------------- API


@pytest.fixture
def served(tmp_path):
    stack = ConfigStack(
        app={"data_dir": str(tmp_path / "data"), "webui": {"bind": "127.0.0.1", "port": 0}},
        audit_path=tmp_path / "audit.jsonl",
    )
    cfg = stack.effective()
    p = pipeline(cfg)
    emo = EmotionsState(SimpleNamespace(circadian=object()), p, cfg, clock=lambda: NOW)
    stack.subscribe(emo.apply_config)
    editor = SettingsEditor(stack, store=tmp_path / "webui_settings.json")
    srv = make_server(
        FakeService(), Diagnostics(), stack.effective, tmp_path, settings=editor, emotions=emo
    )
    srv.set_pin(PIN)
    srv.start()
    yield SimpleNamespace(client=Client(srv.port), p=p, cfg=cfg)
    srv.stop()


def test_the_state_needs_a_session_and_does_not_extend_it():
    assert "/api/emotions/state" in PASSIVE_POLLS


def test_the_api_serves_the_state_and_turns_off_from_settings(served):
    c = served.client
    assert c.request("GET", "/api/emotions/state").status == 401
    c.login()
    resp = c.request("GET", "/api/emotions/state")
    assert resp.status == 200 and resp.json["poll_ms"] == 1000
    off = {"path": "webui.emotions.enabled", "value": False, "pin": PIN}
    assert c.request("POST", "/api/settings", off).status == 200
    assert c.request("GET", "/api/emotions/state").json == {"error": "emotions_disabled"}
    play = c.request("POST", "/api/emotions/play", {"label": "joy", "pin": PIN})
    assert play.status == 404 and play.json == {"error": "emotions_disabled"}


def test_play_needs_csrf_and_the_pin(served):
    c = served.client
    c.login()
    assert (
        c.request("POST", "/api/emotions/play", {"label": "joy", "pin": PIN}, csrf=False).status
        == 403
    )
    assert served.p.expression.asked == []
    ok = c.request("POST", "/api/emotions/play", {"label": "nod", "pin": PIN})
    assert ok.status == 200 and ok.json == {"queued": True}
    assert served.p.expression.asked == ["nod"]


def test_a_wrong_pin_plays_nothing(served):
    c = served.client
    c.login()
    bad = c.request("POST", "/api/emotions/play", {"label": "joy", "pin": "0000000"})
    assert bad.status == 401 and bad.json["error"] == "bad_pin"
    missing = c.request("POST", "/api/emotions/inject", {"rule": 0})
    assert missing.status == 400
    assert served.p.expression.asked == [] and served.p.runner.engine.history() == []


def test_inject_fires_the_rule_and_bad_bodies_are_refused(served):
    c = served.client
    c.login()
    index = rule_index(served.cfg, "sound.loud")
    ok = c.request("POST", "/api/emotions/inject", {"rule": index, "pin": PIN})
    assert ok.status == 200 and ok.json == {"fired": True}
    assert c.request("POST", "/api/emotions/inject", {"rule": True, "pin": PIN}).status == 400
    extra = c.request("POST", "/api/emotions/inject", {"rule": 1, "pin": PIN, "x": 1})
    assert extra.status == 422 and extra.json["error"] == "unknown_field"
    unknown = c.request("POST", "/api/emotions/inject", {"rule": 999, "pin": PIN})
    assert unknown.status == 422 and unknown.json == {"error": "unknown_rule"}


def test_a_control_is_refused_while_the_robot_is_asleep(served):
    c = served.client
    c.login()
    served.p.sleep.asleep = True
    resp = c.request("POST", "/api/emotions/play", {"label": "joy", "pin": PIN})
    assert resp.status == 409 and resp.json == {"error": "robot_asleep"}
    assert served.p.expression.asked == []


def test_every_refusal_has_a_sentence_in_both_languages():
    codes = {*EMOTIONS_STATUS, *REFUSALS, "emotions_unavailable"}
    for lang, strings in STRINGS.items():
        assert sorted(c for c in codes if f"err_{c}" not in strings) == [], lang


@pytest.mark.parametrize(
    ("label", "status", "code"),
    [("x" * 65, 422, "unknown_expression"), ("", 422, "unknown_expression"), (5, 400, None)],
)
def test_play_refuses_a_label_the_robot_lacks_or_one_that_is_not_text(served, label, status, code):
    c = served.client
    c.login()
    resp = c.request("POST", "/api/emotions/play", {"label": label, "pin": PIN})
    assert resp.status == status
    if code:
        assert resp.json == {"error": code}
    assert served.p.expression.asked == []


def test_inject_refuses_a_negative_index_and_a_second_inject_waits_for_the_cooldown(served):
    c = served.client
    c.login()
    negative = c.request("POST", "/api/emotions/inject", {"rule": -1, "pin": PIN})
    assert negative.status == 422 and negative.json == {"error": "unknown_rule"}
    index = rule_index(served.cfg, "sound.loud")
    first = c.request("POST", "/api/emotions/inject", {"rule": index, "pin": PIN})
    second = c.request("POST", "/api/emotions/inject", {"rule": index, "pin": PIN})
    assert first.json == {"fired": True} and second.json == {"fired": False}
    assert len(served.p.runner.engine.history()) == 1


@pytest.mark.parametrize(
    ("part", "code"), [("motors_off", "motors_off"), ("external", "external_control")]
)
def test_motors_off_and_external_control_are_refused_over_http(served, part, code):
    c = served.client
    c.login()
    setattr(served.p.loop, part, True)
    for path, body in (("play", {"label": "joy"}), ("inject", {"rule": 0})):
        resp = c.request("POST", f"/api/emotions/{path}", {**body, "pin": PIN})
        assert resp.status == 409 and resp.json == {"error": code}
    assert served.p.expression.asked == [] and served.p.runner.engine.history() == []


# ---------------------------------------------------------------- the engine with a history


EVENTS = [("sound.loud", 1.0), ("person.known", 2.5), ("idle.long", 9.0), ("sound.loud", 9.0)]


def replay(history: bool) -> list[tuple]:
    """The same events and LLM impulses at the same times, with or without a history."""
    cfg = RoboEmotConfig()
    engine = AffectEngine(cfg.affect, cfg.appraisal)
    engine.keep_history(cfg.webui.emotions.events if history else 0)
    states = []
    for event, at in EVENTS:
        engine.update(at - 0.5)
        engine.on_event(event, at)
        engine.llm_impulse(at, 0.05, -0.02)
        s = engine.update(at + 0.25)
        states.append((s.valence, s.arousal, s.label, s.intensity, s.impulse))
    return states


def test_the_affect_evolves_the_same_with_and_without_the_history():
    assert replay(history=True) == replay(history=False)


# ---------------------------------------------------------------- contract with the renderer


MANIFEST = (
    Path(__file__).resolve().parents[3] / "configs" / "manifests" / "reachy-mini-wireless.yaml"
)


def test_with_a_real_renderer_play_queues_what_the_play_skill_would():
    """The real ``ExpressionRenderer``: the offer leaves the cue gestures out, a played name
    starts at the next update, an unknown one is refused, and resting plays nothing."""
    manifest = load_manifest(MANIFEST, allow_unmeasured_dynamics=True)
    stack = ConfigStack()
    stack.set_manifest_behaviour(
        manifest.model.behaviour, manifest.channels, load_clip_library(manifest), body_yaw=None
    )
    cfg = stack.effective()
    renderer = ExpressionRenderer(
        cfg.expression, cfg.expression_map, manifest.dofs, 0.15, cfg.gesture_map,
        body_yaw="body_yaw", clips=load_clip_library(manifest),
    )  # fmt: skip
    p = pipeline(cfg)
    p.expression = renderer
    emo = EmotionsState(SimpleNamespace(circadian=object()), p, cfg, clock=lambda: NOW)
    offer = emo.state()["playable"]
    assert "nod" in offer["gestures"] and "joy" in offer["labels"]
    assert not any(n.startswith("cue_") for n in offer["labels"] + offer["gestures"])
    assert emo.play("nod") is True
    register = resolve_register([Person({})], cfg.register, cfg.persona)
    assert renderer.update(AffectState(0.0, 0.0, "calm", 0.5), register, NOW) is True
    assert renderer.last_label == "nod" and renderer.started == 1
    with pytest.raises(EmotionsError):
        emo.play("cue_listening")
    renderer.circadian_factor = lambda: 0.0
    assert emo.play("joy") is False
