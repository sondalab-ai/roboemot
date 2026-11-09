"""SPEC 11A, 12: the Live tab's inner-state payload. State only, and nothing private."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from roboemot.affect import AffectState
from roboemot.affect.dance import DanceLive
from roboemot.config import ConfigStack
from roboemot.conversation import ConversationPhase
from roboemot.runtime.live import LiveState
from roboemot.voice.music import MusicState
from roboemot.world.types import Entry

NOW = 500.0
SECRET_ID = "7f3c9a52-registry-id"
FORBIDDEN = {
    "embedding", "identity", "entry_id", "person_id", "azimuth", "elevation", "distance",
    "bearing", "location", "latitude", "longitude", "image", "frame", "audio", "text",
}  # fmt: skip


def cfg(**live):
    return ConfigStack(app={"webui": {"live": {"enabled": True, **live}}}).effective()


def person(eid, seen=NOW, identity="unknown", kind="person"):
    return Entry(
        entry_id=eid, kind=kind, category="face", identity=identity,
        azimuth_rad=0.3, elevation_rad=0.1, distance_m=1.7, uncertainty_rad=0.1,
        confidence=0.9, first_seen=0.0, last_seen=seen, embedding=np.full(512, 0.123456),
    )  # fmt: skip


class Registry:
    def __init__(self, people):
        self.people = people

    def get_person(self, pid):
        return self.people[pid]

    get_entity = get_person


def record(name, status="active"):
    return SimpleNamespace(display_name=name, status=status)


def parts(entries=(), target=None, registry=None, **over):
    p = SimpleNamespace(
        runner=SimpleNamespace(
            engine=SimpleNamespace(state=lambda now: AffectState(0.25, -0.1, "calm", 0.4))
        ),
        circ=SimpleNamespace(snapshot=lambda: {"state": "awake", "sleepiness": 0.2}),
        sleep=SimpleNamespace(asleep=False, paused_parts=lambda: []),
        alone=SimpleNamespace(alone=False),
        network=SimpleNamespace(online=True),
        tracker=SimpleNamespace(active=SimpleNamespace(people=1, params="PRIVATE-PARAMS")),
        selector=SimpleNamespace(target=target),
        perception=SimpleNamespace(registry=registry),
        world=SimpleNamespace(entries=list(entries)),
        reflexes=SimpleNamespace(last_fired=lambda now: ("startle", 2.34567)),
        loop=SimpleNamespace(motors_off=False, external=False, unreachable_held=False),
    )  # fmt: skip
    for key, value in over.items():
        setattr(p, key, value)
    return p


def live(p=None, config=None, host=None, **kw):
    host = host or SimpleNamespace(
        circadian=SimpleNamespace(state="awake", reason=None),
        phase=ConversationPhase(),
        activity=None,
        realtime_route=None,
        voice_listener=None,
        dance=None,
    )
    return LiveState(
        host, p or parts(), config or cfg(), clock=lambda: NOW, temperature=lambda: None, **kw
    )


def keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from keys(v)


def test_it_is_off_until_enabled_and_follows_a_hot_reload():
    off = ConfigStack().effective()
    state = live(config=off)
    assert state.state() is None and not state.enabled
    state.apply_config(SimpleNamespace(config=cfg()))
    assert state.state() is not None and state.enabled
    state.apply_config(SimpleNamespace(config=off))
    assert state.state() is None


def test_the_payload_shape():
    s = live(config=cfg(poll_ms=750)).state()
    assert set(s) == {
        "poll_ms",
        "mood",
        "modes",
        "conversation",
        "attention",
        "reflexes",
        "motion",
        "music",
        "dance",
        "system",
        "video",
        "audio_egress",
        "activity",
    }
    assert s["audio_egress"] is False  # no realtime route: no audio leaves
    assert s["poll_ms"] == 750
    assert s["mood"] == {
        "label": "calm", "valence": 0.25, "arousal": -0.1, "intensity": 0.4,
        "circadian": {"state": "awake", "reason": None, "sleepiness": 0.2},
    }  # fmt: skip
    assert s["modes"] == {
        "asleep": False, "rest": None, "alone": False, "paused": [], "online": True,
        "people": 1, "persona": "default",
    }  # fmt: skip
    assert s["conversation"] == {"phase": "idle", "route": None, "elapsed_s": None}
    assert s["attention"] == {"count": 0, "faces": []}
    assert s["reflexes"] == {"last": {"name": "startle", "age_s": 2.346}}
    assert s["motion"] == {"state": "writing"}
    assert s["system"] == {"cpu_temp_c": None}
    json.dumps(s, allow_nan=False)


def test_modes_follow_the_parts():
    host = SimpleNamespace(
        circadian=None,
        phase=ConversationPhase(),
        activity=None,
        realtime_route=None,
        voice_listener=None,
        dance=None,
    )
    p = parts(
        sleep=SimpleNamespace(asleep=True, paused_parts=lambda: ["vision", "listener"]),
        network=SimpleNamespace(online=False),
        tracker=SimpleNamespace(active=None),
        reflexes=SimpleNamespace(last_fired=lambda now: None),
        alone=SimpleNamespace(alone=True),
    )
    s = live(p, host=host).state()
    assert s["modes"] == {
        "asleep": True, "rest": None, "alone": True, "paused": ["vision", "listener"],
        "online": False,
        "people": 0, "persona": "default",
    }  # fmt: skip
    assert s["mood"]["circadian"] is None and s["reflexes"] == {"last": None}


def test_the_mood_card_says_why_the_robot_rests():
    p = parts(circ=SimpleNamespace(snapshot=lambda: {"state": "resting", "reason": "nap"}))
    circadian = live(p).state()["mood"]["circadian"]
    assert circadian == {"state": "resting", "reason": "nap", "sleepiness": None}


@pytest.mark.parametrize(
    ("state", "reason", "rest", "status"),
    [
        ("resting", "nap", "napping", "napping"),
        ("resting", "quiet_hours", "resting", "resting"),
        ("drowsy", None, "drowsy", "idle"),
        ("awake", None, None, "idle"),
    ],
)
def test_the_modes_card_names_the_circadian_rest_apart_from_the_motors(state, reason, rest, status):
    host = SimpleNamespace(
        circadian=SimpleNamespace(state=state, reason=reason),
        phase=ConversationPhase(),
        activity=None,
        realtime_route=None,
        voice_listener=None,
        dance=None,
    )
    s = live(host=host).state()
    assert s["modes"]["rest"] == rest
    assert s["modes"]["asleep"] is False  # a nap is not the platform sleep (motors off)
    assert s["activity"]["status"] == status


def test_motors_off_and_a_nap_show_both_and_the_status_says_motors_off():
    host = SimpleNamespace(
        circadian=SimpleNamespace(state="resting", reason="nap"),
        phase=ConversationPhase(),
        activity=None,
        realtime_route=None,
        voice_listener=None,
        dance=None,
    )
    p = parts(sleep=SimpleNamespace(asleep=True, paused_parts=lambda: ["app_tick"]))
    s = live(p, host=host).state()
    assert s["activity"]["status"] == "asleep"  # the platform sleep wins the status line
    assert s["modes"]["asleep"] is True and s["modes"]["rest"] == "napping"


def test_the_conversation_card_is_the_phase():
    host = SimpleNamespace(
        circadian=None,
        phase=ConversationPhase(),
        activity=None,
        realtime_route=None,
        voice_listener=None,
        dance=None,
    )
    host.phase.begin_thinking()
    host.phase.set_route("lan")
    c = live(host=host).state()["conversation"]
    assert c["phase"] == "thinking" and c["route"] == "lan" and c["elapsed_s"] >= 0.0


def test_motion_state():
    for flags, expect in (
        ({"motors_off": False, "external": False}, "writing"),
        ({"motors_off": False, "external": True}, "external"),
        ({"motors_off": True, "external": False}, "motors_off"),
        ({"motors_off": True, "external": True}, "motors_off"),
        ({"motors_off": False, "external": False, "unreachable_held": True}, "held"),
        ({"motors_off": False, "external": True, "unreachable_held": True}, "external"),
        ({"motors_off": True, "external": False, "unreachable_held": True}, "motors_off"),
    ):
        loop = SimpleNamespace(**({"unreachable_held": False} | flags))
        assert live(parts(loop=loop)).state()["motion"] == {"state": expect}


def test_an_asleep_robot_never_reports_writing():
    p = parts(
        loop=SimpleNamespace(motors_off=False, external=False, unreachable_held=False),
        sleep=SimpleNamespace(asleep=True, paused_parts=lambda: []),
    )
    assert live(p).state()["motion"] == {"state": "motors_off"}


def test_faces_are_named_only_from_confirmed_records_and_the_target_comes_first():
    registry = Registry({SECRET_ID: record("Mia"), "pend": record("Nobody", "pending")})
    entries = [
        person("a", seen=NOW - 1, identity=SECRET_ID),
        person("b", seen=NOW),
        person("c", seen=NOW - 2, identity="pend"),
        person("gone", seen=NOW - 99),  # older than attention.stale_s
        person("dog", kind="animal"),
    ]
    target = SimpleNamespace(entry_id="c")
    a = live(parts(entries, target, registry)).state()["attention"]
    assert a == {
        "count": 3,
        "faces": [
            {"name": None, "target": True},  # pending record: not named
            {"name": None, "target": False},  # unknown
            {"name": "Mia", "target": False},
        ],
    }


def test_the_list_is_capped_but_the_count_is_complete():
    entries = [person(f"e{i}", seen=NOW - i * 0.01) for i in range(5)]
    a = live(parts(entries), config=cfg(max_tracked=2)).state()["attention"]
    assert a["count"] == 5 and len(a["faces"]) == 2


def test_a_bearing_only_target_marks_nobody():
    a = live(parts([person("a")], SimpleNamespace(entry_id=None))).state()["attention"]
    assert a["faces"] == [{"name": None, "target": False}]


def test_nothing_private_is_in_the_payload():
    registry = Registry({SECRET_ID: record("Mia")})
    p = parts(
        [person("entry-77", identity=SECRET_ID)], SimpleNamespace(entry_id="entry-77"), registry
    )
    s = live(p).state()
    dump = json.dumps(s)
    assert not (FORBIDDEN & set(keys(s)))
    for secret in (SECRET_ID, "entry-77", "0.123456", "PRIVATE-PARAMS", "0.3", "1.7"):
        assert secret not in dump, secret


def test_the_temperature_is_read_once_per_cache_window():
    reads, clock = [], [NOW]

    def sensor():
        reads.append(1)
        return 61.234

    state = LiveState(
        SimpleNamespace(
            circadian=None,
            phase=ConversationPhase(),
            activity=None,
            realtime_route=None,
            voice_listener=None,
            dance=None,
        ),
        parts(),
        cfg(system_cache_s=5.0),
        clock=lambda: clock[0],
        temperature=sensor,
    )
    assert state.state()["system"] == {"cpu_temp_c": 61.2}
    clock[0] += 4.9
    state.state()
    assert len(reads) == 1
    clock[0] += 0.2
    state.state()
    assert len(reads) == 2


def test_the_default_temperature_reader_survives_an_os_without_sensors(monkeypatch):
    import psutil
    from roboemot.runtime.live import cpu_temperature_c

    monkeypatch.delattr(psutil, "sensors_temperatures", raising=False)
    assert cpu_temperature_c() is None
    monkeypatch.setattr(
        psutil,
        "sensors_temperatures",
        lambda: {"cpu": [SimpleNamespace(current=48.0), SimpleNamespace(current=55.5)]},
        raising=False,
    )
    assert cpu_temperature_c() == 55.5
    monkeypatch.setattr(psutil, "sensors_temperatures", lambda: 1 / 0, raising=False)
    assert cpu_temperature_c() is None


def test_the_state_names_the_video_flag_and_its_poll_interval():
    state = live(config=cfg())
    assert state.state()["video"] == {"enabled": False, "poll_ms": 333, "idle_stop_ms": 600000}
    on = ConfigStack(app={"webui": {"live": {"enabled": True, "video": True, "video_fps": 5.0}}})
    state.apply_config(SimpleNamespace(config=on.effective()))
    assert state.state()["video"] == {"enabled": True, "poll_ms": 200, "idle_stop_ms": 600000}


# ---------------------------------------------------------------- the beat card (SPEC 7, 12)

MUSIC_KEYS = {"playing", "tempo_bpm", "confidence", "beat_age_s"}
DANCE_KEYS = {"state", "source", "bob", "sway", "last_accent_age_s"}


def music_host(detector, frame=None):
    return SimpleNamespace(
        circadian=SimpleNamespace(state="awake", reason=None),
        phase=ConversationPhase(),
        activity=None,
        realtime_route=None,
        voice_listener=SimpleNamespace(music=detector, awaiting_transcript=lambda: False),
        dance=None if frame is None else SimpleNamespace(live_snapshot=lambda now: frame),
    )


def detector(state):
    return SimpleNamespace(state=lambda now: state, snapshot=lambda: {"SPECTRUM": [1.0]})


def test_the_beat_blocks_for_a_detector_and_a_dance():
    playing = MusicState(True, 120.04, 0.12345, 0.87654, NOW)
    frame = DanceLive(True, "room", 0.61234, 0.4, NOW - 3.25, NOW)
    s = live(host=music_host(detector(playing), frame)).state()
    assert s["music"] == {
        "playing": True, "tempo_bpm": 120.04, "confidence": 0.877, "beat_age_s": 0.123,
    }  # fmt: skip
    assert s["dance"] == {
        "state": "active", "source": "room", "bob": 0.612, "sway": 0.4, "last_accent_age_s": 3.25,
    }  # fmt: skip


def test_the_beat_blocks_carry_only_the_listed_keys_and_no_monotonic_stamp():
    playing = MusicState(True, 120.0, 0.1, 0.8, NOW)
    frame = DanceLive(True, "own", 0.5, 0.5, NOW - 1.0, NOW)
    s = live(host=music_host(detector(playing), frame)).state()
    assert set(s["music"]) == MUSIC_KEYS
    assert set(s["dance"]) == DANCE_KEYS
    assert NOW not in [*s["music"].values(), *s["dance"].values()]
    json.dumps(s)


def test_no_detector_and_no_renderer_are_null():
    s = live().state()
    assert s["music"] is None and s["dance"] is None


def test_music_off_is_null_even_with_a_detector():
    off = ConfigStack(app={"webui": {"live": {"enabled": True}}, "music": {"enabled": False}})
    silent = MusicState(False, None, None, 0.0, NOW)
    s = live(host=music_host(detector(silent)), config=off.effective()).state()
    assert s["music"] is None


def test_no_music_and_an_idle_dance_have_no_beat():
    silent = MusicState(False, None, None, 0.2, NOW)
    s = live(host=music_host(detector(silent), DanceLive(stamp=NOW))).state()
    assert s["music"] == {
        "playing": False,
        "tempo_bpm": None,
        "confidence": 0.2,
        "beat_age_s": None,
    }
    assert s["dance"] == {
        "state": "idle", "source": None, "bob": 0.0, "sway": 0.0, "last_accent_age_s": None,
    }  # fmt: skip
