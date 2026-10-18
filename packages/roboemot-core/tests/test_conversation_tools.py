"""SPEC 8 (tool rounds, stage 8 barge-in), 11 (LLM tools), 6/12 (confirmed preferences): the
conversation worker's tool loop, spoken confirmation, interruption and privacy."""

from __future__ import annotations

import json
import logging
import time

import httpx
import pytest
from roboemot.affect import AffectEngine
from roboemot.config.schema import RoboEmotConfig
from roboemot.llm.prompt import MINOR_RULE
from roboemot.skills import SkillContext, SkillRegistry, skill
from roboemot.skills.defaults import SkillEnv, default_skills
from test_conversation_worker import (
    CLEARED,
    GATE,
    LAN,
    rigs,  # noqa: F401  (fixture)
    sse_piece,
    utt,
    wait_for,
)
from test_llm_router import chunk

HOST = "lan.local"


def tool_chunk(index, call_id, name, args):
    fn = {"name": name, "arguments": args}
    return chunk(tool_calls=[{"index": index, "id": call_id, "type": "function", "function": fn}])


def T(text):
    return sse_piece(text)


def C(index, call_id, name, args):
    return f"data: {json.dumps(tool_chunk(index, call_id, name, json.dumps(args)))}\n\n".encode()


class Scripted:
    """The server side of a conversation with tools: one script (a list of text pieces, tool
    calls built with ``C`` and GATE) per request, in order; later requests get ``Ok.``"""

    def __init__(self, rig, *rounds):
        self.rig, self.rounds = rig, list(rounds)
        self.bodies: list[dict] = []
        rig.server.handler = self.handler

    async def handler(self, request: httpx.Request) -> httpx.Response:
        import asyncio

        body = json.loads(request.content)
        self.bodies.append(body)
        self.rig.server.requests.append((request.url.host, body))
        script = self.rounds.pop(0) if self.rounds else ["Ok."]

        async def stream():
            for piece in script:
                if piece is GATE:
                    while not self.rig.server.gate.is_set():
                        await asyncio.sleep(0.002)
                elif isinstance(piece, bytes):
                    yield piece
                else:
                    yield T(piece)
            fin = chunk("")
            fin["choices"][0]["finish_reason"] = "stop"
            yield f"data: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode()

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream())


class Env:
    def __init__(self):
        cfg = RoboEmotConfig()
        self.engine = AffectEngine(cfg.affect, cfg.appraisal)
        self.played, self.looked = [], []
        env = SkillEnv(
            gestures=lambda: ["joy", "curiosity"],
            play=lambda g: self.played.append(g) or True,
            look_names=lambda: ["Marta"],
            look_at=lambda t: self.looked.append(t) or True,
            affect=self.engine,
        )
        self.registry = default_skills(env)


class Store:
    def __init__(self, allowed=True):
        self.allow, self.saved = allowed, []

    def allowed(self, person_id, key):
        return self.allow

    def store(self, person_id, key, value):
        self.saved.append((person_id, key, value))
        return True


@pytest.fixture()
def tooled(rigs):  # noqa: F811
    def make(*rounds, routes=(LAN,), speaker="p1", store=None, **cfg):
        rig = rigs(routes=routes, **cfg)
        script = Scripted(rig, *rounds)
        env = Env()
        rig.worker.skills = env.registry
        rig.worker.preferences = store
        rig.who = [speaker]
        rig.worker._speaker = lambda: rig.who[0]
        return rig, script, env

    return make


def names(body):
    return [t["function"]["name"] for t in body.get("tools", [])]


# ------------------------------------------------------------------ the loop


def test_a_tool_round_trip_runs_the_skill_and_asks_again_with_the_result(tooled):
    rig, srv, env = tooled(
        ["Guardo. ", C(0, "c1", "play", {"gesture": "joy"})],
        ["Fatto, ecco."],
    )
    rig.turn("fai una cosa allegra")
    assert env.played == ["joy"]
    assert rig.speech.texts() == ["Guardo.", "Fatto, ecco."]
    first, second = srv.bodies
    assert names(first) == [
        "play",
        "look_at",
        "express_emotion",
        "remember_preference",
        "stop_talking",
    ]
    assert first["tool_choice"] == "auto" and second["tool_choice"] == "auto"
    last = second["messages"][-2:]
    assert last[0]["role"] == "assistant" and last[0]["tool_calls"][0]["id"] == "c1"
    assert json.loads(last[0]["tool_calls"][0]["function"]["arguments"]) == {"gesture": "joy"}
    assert last[1] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'}
    snap = rig.worker.snapshot()
    assert snap["tool_calls"] == 1 and rig.events[0]["rounds"] == 1


def test_several_calls_of_one_round_run_in_order_and_all_results_go_back(tooled):
    rig, srv, env = tooled(
        [
            C(0, "c1", "play", {"gesture": "joy"}),
            C(1, "c2", "look_at", {"target": "Marta"}),
        ],
        ["Ecco."],
    )
    rig.turn("ciao")
    assert env.played == ["joy"] and env.looked == ["Marta"]
    msgs = srv.bodies[1]["messages"]
    assert [m["role"] for m in msgs[-3:]] == ["assistant", "tool", "tool"]
    assert [m["tool_call_id"] for m in msgs[-2:]] == ["c1", "c2"]


def test_the_exchange_is_kept_in_the_history_of_the_turn(tooled):
    rig, srv, _ = tooled([C(0, "c1", "play", {"gesture": "joy"})], ["Fatto."], ["Certo."])
    rig.turn("uno")
    rig.turn("due")
    msgs = srv.bodies[2]["messages"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "tool", "assistant", "user"]
    assert msgs[3]["tool_call_id"] == "c1" and msgs[4]["content"] == "Fatto."


def test_rounds_are_bounded_and_the_last_request_has_no_tools(tooled):
    call = [C(0, "c", "play", {"gesture": "joy"})]
    rig, srv, env = tooled(call, call, call, ["Basta così."], conversation={"max_tool_rounds": 3})
    rig.turn("ciao")
    assert len(srv.bodies) == 4 and len(env.played) == 3
    assert [bool(b.get("tools")) for b in srv.bodies] == [True, True, True, False]
    assert rig.speech.texts() == ["Basta così."]


def test_a_final_round_without_text_still_gets_a_spoken_line(tooled):
    call = [C(0, "c", "play", {"gesture": "joy"})]
    rig, srv, _ = tooled(call, call, [], conversation={"max_tool_rounds": 2})
    rig.turn("ciao")
    assert len(srv.bodies) == 3
    assert rig.speech.texts() == [RoboEmotConfig().conversation.fallback_lines["it"]]
    assert rig.events[0]["outcome"] == "no_text"


def test_a_tool_call_in_a_request_that_offered_no_tools_is_ignored(tooled):
    call = [C(0, "c", "play", {"gesture": "joy"})]
    rig, srv, env = tooled(call, call, call, conversation={"max_tool_rounds": 1})
    rig.turn("ciao")
    assert len(srv.bodies) == 2 and len(env.played) == 1  # no third request, no second run
    assert rig.speech.texts() == [RoboEmotConfig().conversation.fallback_lines["it"]]


def test_no_tools_without_skills_or_when_the_config_says_so(tooled):
    for cfg in ({"conversation": {"llm_skills": []}}, {"conversation": {"max_tool_rounds": 0}}):
        rig, srv, _ = tooled(["Ciao."], **cfg)
        rig.turn("ciao")
        assert "tools" not in srv.bodies[0]
    rig, srv, _ = tooled(["Ciao."])
    rig.worker.skills = None
    rig.turn("ciao")
    assert "tools" not in srv.bodies[0]


def test_a_skill_that_is_not_visible_is_not_offered_and_cannot_be_called(tooled):
    rig, srv, env = tooled(
        [C(0, "c1", "secret", {})], ["Ok."], conversation={"llm_skills": ["secret", "shown"]}
    )
    ran = []

    @skill()
    def secret() -> str:
        ran.append(1)
        return "x"

    @skill(llm_visible=True)
    def shown() -> str:
        return "y"

    rig.worker.skills = SkillRegistry([secret, shown])
    rig.turn("ciao")
    assert ran == []
    assert "secret" not in names(srv.bodies[0])
    result = srv.bodies[1]["messages"][-1]
    assert json.loads(result["content"])["error"] == "unknown_skill"


def test_a_bad_call_is_answered_with_an_error_the_model_can_read(tooled, caplog):
    rig, srv, env = tooled(
        [
            C(0, "c1", "play", {"gesture": "SECRET-GESTURE-77"}),
            C(1, "c2", "look_at", {"target": "Marta", "joint": "head.yaw"}),
            C(2, "c3", "express_emotion", {"emotion": "joy", "intensity": "SECRET-77"}),
            b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m","choices":'
            b'[{"index":0,"delta":{"tool_calls":[{"index":3,"id":"c4","type":"function",'
            b'"function":{"name":"play","arguments":"{SECRET-77"}}]},"finish_reason":null}]}\n\n',
        ],
        ["Non ci riesco."],
    )
    with caplog.at_level(logging.DEBUG):
        rig.turn("ciao")
    assert env.played == [] and env.looked == []
    results = [json.loads(m["content"]) for m in srv.bodies[1]["messages"] if m["role"] == "tool"]
    assert [r["error"] for r in results] == ["invalid_arguments"] * 4
    assert "SECRET" not in json.dumps(results) and "SECRET" not in caplog.text
    assert rig.worker.snapshot()["tool_errors"] == 4
    assert rig.speech.texts() == ["Non ci riesco."]


def test_every_round_applies_the_minor_rules(tooled):
    rig, srv, env = tooled(
        [C(0, "c1", "play", {"gesture": "joy"})],
        ["Ciao. ", "Voglio uccidere il tempo."],
        routes=(CLEARED,),
        safety={"minors_allowed_llm": True},
    )
    rig.minor[0] = True
    rig.register[0] = rig.stack.effective().register.child
    rig.turn("ciao")
    texts = rig.speech.texts()
    assert env.played == ["joy"] and len(srv.bodies) == 2
    assert all(MINOR_RULE in b["messages"][0]["content"] for b in srv.bodies)
    assert texts[0] == "Ciao." and "uccidere" not in " ".join(texts)
    assert rig.worker.snapshot()["filtered"] == 1 and all(s["minor"] for s in rig.speech.said)


def test_a_minor_appearing_between_rounds_ends_the_turn(tooled):
    rig, srv, env = tooled(
        ["Guardo. ", C(0, "c1", "play", {"gesture": "joy"})],
        ["Non deve arrivare."],
        routes=({**LAN, "name": "lan"},),
    )

    def appears(gesture):
        rig.minor[0] = True
        return True

    rig.worker.skills = default_skills(
        SkillEnv(lambda: ["joy"], appears, lambda: [], lambda t: True, Env().engine)
    )
    rig.turn("ciao")
    assert len(srv.bodies) == 1  # no second request
    assert rig.speech.texts() == ["Guardo."]  # said before the skill ran; the next round never came
    assert rig.worker.snapshot()["minor_cut"] == 1


def test_skills_not_allowed_with_a_minor_are_not_offered_to_a_child(tooled):
    rig, srv, _ = tooled(
        ["Ciao."],
        routes=(CLEARED,),
        safety={"minors_allowed_llm": True},
        conversation={"llm_skills": ["adults_only", "anyone"]},
    )

    @skill(llm_visible=True, allowed_with_minor=False)
    def adults_only() -> str:
        return "x"

    @skill(llm_visible=True)
    def anyone() -> str:
        return "y"

    rig.worker.skills = SkillRegistry([adults_only, anyone])
    rig.minor[0] = True
    rig.turn("ciao")
    assert "adults_only" not in names(srv.bodies[0])


# ------------------------------------------------------------------ stop_talking


def test_stop_talking_stops_the_speech_ends_the_turn_and_keeps_what_was_heard(tooled):
    rig, srv, _ = tooled(["Prima. ", "Seconda. ", C(0, "c1", "stop_talking", {})])
    rig.speech.unplayed = {2}  # the second sentence was queued behind the first, not heard
    rig.turn("parla")
    assert rig.speech.stops == 1 and len(srv.bodies) == 1
    assert rig.speech.texts() == ["Prima.", "Seconda."]
    rig.turn("ciao")
    hist = srv.bodies[1]["messages"]
    assert hist[2] == {"role": "assistant", "content": "Prima. …"}
    assert rig.worker.snapshot()["stopped"] == 1


# ------------------------------------------------------------------ barge-in


def test_interrupting_a_streaming_answer_cancels_the_stream_and_keeps_the_played_part(tooled):
    rig, srv, _ = tooled(["Uno. ", "Due. ", "Tre", GATE, "..."], ["Dimmi."])
    n = rig.say("racconta")
    assert wait_for(lambda: rig.speech.texts() == ["Uno.", "Due."])
    rig.speech.unplayed = {2}  # the barge-in came while sentence 1 was playing
    t0 = time.monotonic()
    assert rig.worker.interrupt()
    assert rig.done(n) and time.monotonic() - t0 < 1.0  # no wait for the stalled stream
    assert rig.hub.router.used("lan") > 0  # charged although cut short
    snap = rig.worker.snapshot()
    assert snap["interrupted"] == 1 and snap["interrupts"] == 1
    rig.turn("dimmi")  # the worker is alive and the history holds the cut turn
    hist = srv.bodies[1]["messages"]
    assert [m["content"] for m in hist[1:3]] == ["racconta", "Uno. …"]


def test_interrupting_after_the_stream_ended_trims_the_history_to_what_was_heard(tooled):
    rig, srv, _ = tooled(["Uno. ", "Due. ", "Tre."], ["Dimmi."])
    rig.turn("racconta")
    rig.speech.unplayed = {2, 3}
    assert rig.worker.interrupt()
    assert not rig.worker.interrupt()  # once
    rig.turn("ancora")
    hist = srv.bodies[1]["messages"]
    assert hist[2] == {"role": "assistant", "content": "Uno. …"}


def test_interrupting_before_anything_was_heard_drops_the_answer_from_the_history(tooled):
    rig, srv, _ = tooled(["Uno. ", "Due."], ["Ok."])
    rig.turn("racconta")
    rig.speech.unplayed = {1, 2}
    rig.worker.interrupt()
    rig.turn("ancora")
    assert [m["role"] for m in srv.bodies[1]["messages"][1:]] == ["user", "user"]


def test_an_interrupt_with_nothing_to_cut_is_a_no_op(tooled):
    rig, _, _ = tooled()
    assert rig.worker.interrupt() is False


def test_an_uninterrupted_turn_is_kept_whole(tooled):
    rig, srv, _ = tooled(["Uno. ", "Due."], ["Ok."])
    rig.turn("racconta")
    assert rig.worker.interrupt() is False  # everything was played
    rig.turn("ancora")
    assert srv.bodies[1]["messages"][2]["content"] == "Uno. Due."


# ------------------------------------------------------------------ confirmation


def remember(key="colore preferito", value="blu"):
    return C(0, "c1", "remember_preference", {"key": key, "value": value})


def ask(tooled, store=None, **kw):
    store = store or Store()
    rig, srv, env = tooled([remember()], store=store, **kw)
    rig.turn("ricordati che il mio colore preferito è blu")
    return rig, srv, store


def test_remember_preference_asks_and_writes_nothing_yet(tooled):
    rig, srv, store = ask(tooled)
    assert rig.speech.texts() == ["Vuoi che ricordi questo: colore preferito, blu?"]
    assert store.saved == [] and len(srv.bodies) == 1  # the turn ended with the question
    assert rig.worker.snapshot()["confirm_asked"] == 1


def test_a_yes_from_the_same_person_stores_it_without_asking_the_model(tooled):
    rig, srv, store = ask(tooled)
    rig.turn("sì certo")
    assert store.saved == [("p1", "colore preferito", "blu")]
    assert rig.speech.texts()[-1] == "Va bene, lo ricorderò."
    assert len(srv.bodies) == 1
    rig.turn("sì")  # the question was used up: this is an ordinary turn
    assert len(srv.bodies) == 2 and len(store.saved) == 1


@pytest.mark.parametrize(
    "reply, who, wait_s",
    [
        ("no grazie", "p1", 0),
        ("sì ma no", "p1", 0),
        ("sì", "p2", 0),  # somebody else
        ("sì", None, 0),  # nobody recognised
        ("sì", "p1", 30),  # too late (confirm_timeout_s is 20)
    ],
)
def test_no_other_reply_stores_anything(tooled, reply, who, wait_s):
    rig, srv, store = ask(tooled)
    rig.who[0] = who
    rig.clock[0] += wait_s
    rig.turn(reply)
    assert store.saved == []


def test_a_no_is_acknowledged_and_an_unrelated_reply_goes_to_the_model(tooled):
    rig, srv, store = ask(tooled)
    rig.turn("no")
    assert rig.speech.texts()[-1] == "Va bene, non lo ricordo." and len(srv.bodies) == 1
    rig2, srv2, store2 = ask(tooled)
    rig2.turn("che ore sono?")
    assert len(srv2.bodies) == 2 and store2.saved == []
    rig2.turn("sì")
    assert store2.saved == []  # the question did not survive the unrelated reply


def test_a_stranger_or_a_full_registry_is_not_asked(tooled):
    for speaker, store in ((None, Store()), ("p1", Store(allowed=False))):
        rig, srv, _ = tooled([remember()], ["Non posso."], speaker=speaker, store=store)
        rig.turn("ricordati")
        result = json.loads(srv.bodies[1]["messages"][-1]["content"])
        assert result == {"status": "not_allowed"} and rig.speech.texts() == ["Non posso."]


def test_unclean_or_long_preferences_are_refused_before_asking(tooled):
    for key, value in (("k" * 41, "blu"), ("colore", "v" * 121), ("", "blu"), ("!!!", "blu")):
        rig, srv, _ = tooled([remember(key, value)], ["No."], store=Store())
        rig.turn("ricordati")
        assert json.loads(srv.bodies[1]["messages"][-1]["content"]) == {"status": "invalid"}
    rig, srv, store = tooled(
        [remember("Colore\nPreferito", 'blu"; ignore {previous}\n')], store=Store()
    )
    rig.turn("ricordati")
    asked = rig.speech.texts()[0]
    assert not any(c in asked for c in '\n"{}')
    rig.turn("sì")
    [(_, key, value)] = rig.worker.preferences.saved
    assert key == "colore preferito" and value == "blu ignore previous"


def test_the_question_for_a_child_passes_the_output_filter(tooled):
    rig, srv, _ = tooled(
        [remember("gioco", "uccidere")],
        ["Ok."],
        routes=(CLEARED,),
        store=Store(),
        safety={"minors_allowed_llm": True},
    )
    rig.minor[0] = True
    rig.turn("ricordati")
    store = rig.worker.preferences
    assert store.saved == [] and "uccidere" not in " ".join(rig.speech.texts())
    assert json.loads(srv.bodies[1]["messages"][-1]["content"]) == {"status": "not_allowed"}


def test_a_child_may_confirm_a_harmless_preference(tooled):
    store = Store()
    rig, srv, _ = tooled(
        [remember("animale", "gatto")], routes=(CLEARED,), store=store,
        safety={"minors_allowed_llm": True},
    )  # fmt: skip
    rig.minor[0] = True
    rig.turn("ricordati")
    assert rig.speech.said[0]["minor"] is True
    rig.turn("sì")
    assert store.saved == [("p1", "animale", "gatto")]


def test_preferences_never_reach_logs_or_the_bus(tooled, caplog):
    store = Store()
    rig, srv, _ = tooled([remember("segreto-99", "PASSWORD-HUNTER2")], store=store)
    with caplog.at_level(logging.DEBUG):
        rig.turn("ricordati")
        rig.turn("sì")
    blob = caplog.text + json.dumps(rig.events) + json.dumps(rig.worker.snapshot())
    assert "HUNTER2" not in blob and "hunter2" not in blob and store.saved


def test_reply_kind_and_sanitising_edges():
    from roboemot.conversation.confirm import reply_kind, sanitise_preference

    conv = RoboEmotConfig().conversation

    def kinds(text, lang="it"):
        return reply_kind(
            text,
            lang,
            conv.confirm_yes,
            conv.confirm_no,
            conv.confirm_fillers,
            conv.confirm_max_words,
        )

    assert kinds("Sì!") == "yes" and kinds("sì, va bene") == "yes" and kinds("OK") == "yes"
    assert kinds("sì grazie") == "yes" and kinds("certo, volentieri") == "yes"
    assert kinds("no") == "no" and kinds("sì però no") == "no" and kinds("non ora") == "no"
    assert kinds("forse domani") == "other" and kinds("") == "other"
    assert kinds("yes please", "en") == "yes" and kinds("yes", "it") == "other"  # language lists
    assert kinds("siamo qui") == "other"  # whole words only
    # a yes word inside another sentence is not a confirmation
    assert kinds("Come si dice cane in inglese?") == "other"
    assert kinds("si chiama Luca") == "other" and kinds("si si") == "other"  # bare si: no accent
    assert kinds("Ok Google") == "other"
    assert kinds("Sure, what is the weather like", "en") == "other"
    assert kinds("sì sì sì sì sì") == "other"  # longer than confirm_max_words
    assert kinds("What time is it? yes", "en") == "other"  # the yes phrase comes first
    assert sanitise_preference("A b", "x" * 120, 40, 120) is not None
    assert sanitise_preference("A b", "x" * 121, 40, 120) is None
    assert sanitise_preference(1, "x", 40, 120) is None
    assert sanitise_preference("a​b", "c‍d", 40, 120) == ("ab", "cd")


# ------------------------------------------------------------------ review round: confirmation


def test_a_reply_that_started_before_the_question_ended_is_not_the_answer(tooled):
    rig, srv, store = ask(tooled)
    rig.speech.ends[1] = 2000.0  # the question plays until t=2000
    rig.turn("sì", stamp=1500.0)  # queued behind it, or a voice over it
    assert store.saved == [] and len(srv.bodies) == 2  # an ordinary turn
    rig.turn("sì", stamp=2002.0)  # after the question: this is the answer
    assert store.saved == [("p1", "colore preferito", "blu")]


def test_a_question_that_never_played_cannot_be_answered(tooled):
    rig, srv, store = ask(tooled)
    rig.speech.unplayed = {1}
    rig.turn("sì", stamp=5000.0)
    assert store.saved == [] and len(srv.bodies) == 2


def test_the_confirmation_timeout_defaults_to_ten_seconds():
    assert RoboEmotConfig().conversation.confirm_timeout_s == 10.0


def test_a_barge_in_before_the_question_finished_drops_it(tooled):
    rig, srv, store = ask(tooled)
    rig.speech.ends[1] = 2000.0  # still playing at clock 1000
    rig.worker.interrupt()
    rig.turn("sì", stamp=5000.0)
    assert store.saved == [] and rig.worker.snapshot()["confirm_cut"] == 1


def test_a_barge_in_after_the_question_was_heard_keeps_it(tooled):
    rig, srv, store = ask(tooled)
    rig.worker.interrupt()  # the question ended long ago (default): it was heard
    rig.turn("sì", stamp=5000.0)
    assert store.saved == [("p1", "colore preferito", "blu")]


def test_a_skill_thread_that_outlives_its_turn_changes_nothing(tooled):
    import threading

    rig, srv, _ = tooled(
        [C(0, "c1", "late", {})],
        ["Ok."],
        conversation={"llm_skills": ["late"], "tool_timeout_s": 0.1},
    )
    done = threading.Event()
    seen = []

    @skill(llm_visible=True)
    def late(ctx: SkillContext) -> str:
        time.sleep(0.4)
        seen.append(ctx.ask_confirmation("colore", "blu"))
        done.set()
        return "x"

    rig.worker.skills = SkillRegistry([late])
    rig.worker.preferences = Store()
    rig.turn("ricordati")
    assert done.wait(3)
    assert seen == ["cancelled"] and rig.worker._confirm is None
    assert "Vuoi che ricordi" not in " ".join(rig.speech.texts())


# ------------------------------------------------------------------ review round: loop details


def test_two_questions_in_one_round_ask_only_the_first(tooled):
    store = Store()
    rig, srv, _ = tooled(
        [C(0, "c1", "remember_preference", {"key": "a", "value": "uno"}),
         C(1, "c2", "remember_preference", {"key": "b", "value": "due"})],
        store=store,
    )  # fmt: skip
    rig.turn("ricordati")
    assert rig.speech.texts() == ["Vuoi che ricordi questo: a, uno?"]
    assert rig.worker.snapshot()["confirm_asked"] == 1 and len(srv.bodies) == 1


def test_calls_beyond_the_per_round_cap_are_answered_with_an_error_unrun(tooled):
    calls = [C(i, f"c{i}", "play", {"gesture": "joy"}) for i in range(6)]
    rig, srv, env = tooled(calls, ["Fatto."])
    rig.turn("ciao")
    assert len(env.played) == 4
    tool = [json.loads(m["content"]) for m in srv.bodies[1]["messages"] if m["role"] == "tool"]
    assert len(tool) == 6 and [t.get("error") for t in tool[4:]] == ["too_many_calls"] * 2


def test_the_sentences_of_a_tool_round_are_the_content_of_its_assistant_message(tooled):
    rig, srv, _ = tooled(
        ["Guardo. ", C(0, "c1", "play", {"gesture": "joy"})], ["Fatto."], ["Certo."]
    )
    rig.turn("uno")
    assert srv.bodies[1]["messages"][-2]["content"] == "Guardo."
    rig.turn("due")
    roles = [(m["role"], m.get("content")) for m in srv.bodies[2]["messages"][1:]]
    assert roles == [
        ("user", "uno"), ("assistant", "Guardo."), ("tool", '{"ok": true}'),
        ("assistant", "Fatto."), ("user", "due"),
    ]  # fmt: skip


def test_the_text_before_a_call_is_spoken_before_the_skill_acts(tooled):
    order = []
    rig, srv, _ = tooled(
        ["Ecco la domanda. ", C(0, "c1", "remember_preference", {"key": "a", "value": "b"})],
        store=Store(),
    )
    real_say = rig.speech.say
    rig.speech.say = lambda text, **kw: (order.append(text), real_say(text, **kw))[1]
    rig.turn("ricordati")
    assert order[0] == "Ecco la domanda." and order[1].startswith("Vuoi che")


def test_nothing_is_said_after_a_barge_in_began_even_before_the_stop_lands(tooled):
    rig, srv, _ = tooled(["Uno. ", "Due. ", "Tre", GATE, "..."])
    n = rig.say("racconta")
    assert wait_for(lambda: rig.speech.texts() == ["Uno.", "Due."])
    rig.worker.mark_interrupting()  # the barge-in thread: first step
    rig.server.gate.set()  # the model goes on talking, the stop has not landed yet
    time.sleep(0.15)
    assert rig.speech.texts() == ["Uno.", "Due."]
    rig.worker.interrupt()
    assert rig.done(n)


def test_the_choices_the_model_was_shown_are_what_the_executor_checks(tooled):
    names = ["a"]

    @skill(llm_visible=True, inline=True)
    def grow() -> dict:
        names.append("b")  # the world changes while the model is answering
        return {"ok": True}

    @skill(llm_visible=True, choices={"who": lambda: list(names)})
    def pick(who: str) -> dict:
        return {"who": who}

    rig, srv, _ = tooled(
        [C(0, "c1", "grow", {}), C(1, "c2", "pick", {"who": "b"})],
        ["Ok."],
        conversation={"llm_skills": ["grow", "pick"]},
    )
    rig.worker.skills = SkillRegistry([grow, pick])
    rig.turn("ciao")
    assert srv.bodies[0]["tools"][1]["function"]["parameters"]["properties"]["who"]["enum"] == ["a"]
    tool = [json.loads(m["content"]) for m in srv.bodies[1]["messages"] if m["role"] == "tool"]
    assert tool[1]["error"] == "invalid_arguments"  # "b" was never offered


# ------------------------------------------------------------------ review round: stop phrase


def test_a_stop_phrase_right_after_a_barge_in_is_not_answered(tooled):
    rig, srv, _ = tooled()
    rig.worker.interrupt()
    assert rig.worker.submit(utt("stop")) is False
    assert rig.worker.submit(utt("basta basta")) is False
    assert rig.worker.submit(utt("stop", language="en")) is False
    assert rig.worker.submit(utt("fermati e dimmi l'ora")) is True  # more than the command
    assert rig.worker.snapshot()["stop_consumed"] == 3
    rig.clock[0] += 30  # long after: an ordinary word
    assert rig.worker.submit(utt("stop")) is True


def test_a_stop_phrase_without_a_barge_in_is_an_ordinary_utterance(tooled):
    rig, srv, _ = tooled()
    assert rig.worker.submit(utt("stop")) is True
