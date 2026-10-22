"""SPEC 11A, 12: the Live tab's inner-state payload. State only, and nothing private."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from roboemot.affect import AffectState
from roboemot.config import ConfigStack
from roboemot.conversation import ConversationPhase
from roboemot.runtime.live import LiveState
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
        network=SimpleNamespace(online=True),
        tracker=SimpleNamespace(active=SimpleNamespace(group="adult", params="PRIVATE-PARAMS")),
        selector=SimpleNamespace(target=target),
        perception=SimpleNamespace(registry=registry),
        world=SimpleNamespace(entries=list(entries)),
        reflexes=SimpleNamespace(last_fired=lambda now: ("startle", 2.34567)),
        loop=SimpleNamespace(motors_off=False, external=False),
    )  # fmt: skip
    for key, value in over.items():
        setattr(p, key, value)
    return p


def live(p=None, config=None, host=None, **kw):
    host = host or SimpleNamespace(
        circadian=object(), minor_present=False, phase=ConversationPhase()
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
        "system",
        "video",
    }
    assert s["poll_ms"] == 750
    assert s["mood"] == {
        "label": "calm", "valence": 0.25, "arousal": -0.1, "intensity": 0.4,
        "circadian": {"state": "awake", "sleepiness": 0.2},
    }  # fmt: skip
    assert s["modes"] == {
        "asleep": False, "paused": [], "online": True, "minor_present": False,
        "register": "adult", "persona": "default",
    }  # fmt: skip
    assert s["conversation"] == {"phase": "idle", "route": None, "elapsed_s": None}
    assert s["attention"] == {"count": 0, "faces": []}
    assert s["reflexes"] == {"last": {"name": "startle", "age_s": 2.346}}
    assert s["motion"] == {"state": "writing"}
    assert s["system"] == {"cpu_temp_c": None}
    json.dumps(s, allow_nan=False)


def test_modes_follow_the_parts():
    host = SimpleNamespace(circadian=None, minor_present=True, phase=ConversationPhase())
    p = parts(
        sleep=SimpleNamespace(asleep=True, paused_parts=lambda: ["vision", "listener"]),
        network=SimpleNamespace(online=False),
        tracker=SimpleNamespace(active=None),
        reflexes=SimpleNamespace(last_fired=lambda now: None),
    )
    s = live(p, host=host).state()
    assert s["modes"] == {
        "asleep": True, "paused": ["vision", "listener"], "online": False, "minor_present": True,
        "register": None, "persona": "default",
    }  # fmt: skip
    assert s["mood"]["circadian"] is None and s["reflexes"] == {"last": None}


def test_the_conversation_card_is_the_phase():
    host = SimpleNamespace(circadian=None, minor_present=False, phase=ConversationPhase())
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
    ):
        assert live(parts(loop=SimpleNamespace(**flags))).state()["motion"] == {"state": expect}


def test_an_asleep_robot_never_reports_writing():
    p = parts(
        loop=SimpleNamespace(motors_off=False, external=False),
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
        SimpleNamespace(circadian=None, minor_present=False, phase=ConversationPhase()),
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
    assert state.state()["video"] == {"enabled": False, "poll_ms": 333}
    on = ConfigStack(app={"webui": {"live": {"enabled": True, "video": True, "video_fps": 5.0}}})
    state.apply_config(SimpleNamespace(config=on.effective()))
    assert state.state()["video"] == {"enabled": True, "poll_ms": 200}
