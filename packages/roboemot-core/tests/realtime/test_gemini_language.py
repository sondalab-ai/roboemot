"""The transcription language on a ``gemini_live`` route (SPEC 8, Gemini Live wire and Realtime
speech route): the setup's ``inputAudioTranscription.languageCodes`` carries the BCP-47 code of
the language the OpenAI wire sends (``transcription_language``, else ``conversation.language``);
a setup the server closes before it is ready is sent again at once without it, and only
``transcription_language_refusals`` such connects in a row remember the route, model and code.
The OpenAI wire and the name hint do not change."""

from __future__ import annotations

import pytest
from gemini_server import GeminiServer
from pydantic import ValidationError
from roboemot.config import ConfigStack
from roboemot.config.schema import RealtimeRouteConfig
from roboemot.realtime.client import RealtimeClient
from roboemot.realtime.gemini_wire import GeminiWire, language_codes
from roboemot.realtime.route import sends_language, session_update, without_language
from rt_rig import build_rt, wait_for
from test_gemini_check import gemini_route as check_gemini_route
from test_gemini_check import make_cfg as check_cfg
from test_gemini_check import run as run_check
from test_gemini_route import gemini_route
from test_voice_preview import gemini_route as preview_gemini_route
from test_voice_preview import make_cfg as preview_cfg
from test_voice_preview import run as run_preview
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

UNKNOWN_FIELD_CLOSE = 1007  # assumed: how Gemini closes a setup with a field it refuses
AUDIO_OPT_IN = {"privacy": {"realtime_audio": True}}


def stack(language: str = "it", **rt) -> ConfigStack:
    return ConfigStack(
        app={
            "conversation": {"language": language},
            "voice": {"realtime_route": {"input": "audio", **rt}},
        }
    )


def setup_of(cfg_stack: ConfigStack, tags: dict[str, str] | None = None) -> dict:
    cfg = cfg_stack.effective()
    rt = cfg.voice.realtime_route
    if tags is None:
        tags = rt.transcription_language_tags if rt.transcription_language_hint else {}
    wire = GeminiWire("gemini-3.8-live", rt.max_message_bytes, language_tags=tags)
    frames, _ = wire.encode(session_update(cfg, "parts 1 to 4", []))
    return frames[0]["setup"]


@pytest.fixture()
def rigs():
    made = []

    def make(srv: GeminiServer, *, rt=None):
        rig = build_rt(
            srv, route=gemini_route(srv), rt={"input": "audio", **(rt or {})}, app=AUDIO_OPT_IN
        )
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.close()


# ------------------------------------------------------------------ what is sent


@pytest.mark.parametrize(("language", "code"), [("it", "it-IT"), ("en", "en-US")])
def test_the_setup_carries_the_conversation_languages_bcp47_code(language, code):
    setup = setup_of(stack(language))
    assert setup["inputAudioTranscription"] == {"languageCodes": [code]}


def test_transcription_language_wins_over_the_conversation_language():
    setup = setup_of(stack("it", transcription_language="en"))
    assert setup["inputAudioTranscription"] == {"languageCodes": ["en-US"]}


def test_no_code_without_a_language_or_a_mapping():
    text = ConfigStack(app={"voice": {"realtime_route": {"input": "text"}}})
    assert setup_of(text)["inputAudioTranscription"] == {}  # text input: no language at all
    assert setup_of(stack("fr"))["inputAudioTranscription"] == {}  # no code for fr
    assert language_codes({}, {"it": "it-IT"}) == []


def test_the_language_is_looked_up_in_lower_case():
    session = {"audio": {"input": {"transcription": {"language": "IT"}}}}
    assert language_codes(session, {"it": "it-IT"}) == ["it-IT"]


def test_the_switch_off_sends_no_code():
    off = stack(transcription_language_hint=False)
    assert setup_of(off)["inputAudioTranscription"] == {}
    cfg = off.effective()
    assert not sends_language(cfg, session_update(cfg, "x", []))


def test_the_wire_counts_the_setups_that_carried_codes():
    cfg = stack().effective()
    wire = GeminiWire("gemini-3.8-live", 65536, language_tags={"it": "it-IT"})
    wire.encode(session_update(cfg, "x", []))
    assert wire.counts["language_codes_sent"] == 1


def test_the_openai_update_is_unchanged_by_the_switch_and_the_codes():
    on = session_update(stack().effective(), "x", [])
    off = session_update(stack(transcription_language_hint=False).effective(), "x", [])
    other = session_update(stack(transcription_language_tags={"it": "it-CH"}).effective(), "x", [])
    assert on == off == other
    assert on["session"]["audio"]["input"]["transcription"] == {"language": "it"}
    assert "languageCodes" not in str(on)


def test_without_language_drops_only_the_language():
    cfg = stack().effective()
    update = {"event_id": "ev_1", **session_update(cfg, "x", [])}
    out = without_language(update)
    assert "transcription" not in out["session"]["audio"]["input"]
    assert "event_id" not in out and update["event_id"] == "ev_1"  # the original is untouched
    assert update["session"]["audio"]["input"]["transcription"] == {"language": "it"}


@pytest.mark.parametrize("tags", [{"IT": "it-IT"}, {"it": "it_IT"}, {"it": ""}])
def test_the_codes_are_validated(tags):
    with pytest.raises(ValidationError):
        RealtimeRouteConfig(transcription_language_tags=tags)


# ------------------------------------------------------------------ end to end


def counts(rig) -> dict:
    return rig.route.snapshot()["transcription_language"]


def stats(sent=0, suspect=0, refused=0, skipped=0) -> dict:
    return {"sent": sent, "suspect": suspect, "refused": refused, "skipped": skipped}


def codes_of(srv: GeminiServer) -> list[dict]:
    return [s["inputAudioTranscription"] for s in srv.setups]


def reconnect(rig) -> bool:
    rig.route.close("conversation_end")
    return rig.route.ready()


IT = {"languageCodes": ["it-IT"]}


def test_an_italian_robots_setup_carries_it_it_and_counts_it(gemini, rigs):
    srv = gemini()
    rig = rigs(srv)
    assert rig.route.ready()
    assert codes_of(srv) == [IT]
    assert counts(rig) == stats(sent=1)
    assert rig.route.snapshot()["wire"]["session"]["language_codes_sent"] == 1


@pytest.mark.parametrize("code", [1008, 1011])
def test_one_transient_close_is_suspect_only_and_the_codes_go_again(gemini, rigs, code):
    srv = gemini(refuse=1, refuse_code=code)  # the first setup only, whatever it carries
    rig = rigs(srv)
    assert rig.route.ready()  # the same connect: no failure, no backoff
    assert codes_of(srv) == [IT, {}]
    snap = rig.route.snapshot()
    assert snap["transcription_language"] == stats(suspect=1)
    assert snap["state"] == "connected" and snap["failures"] == {}
    assert snap["connect_failures"] == 0 and rig.route._refusals == 0
    assert rig.route._resets_in_row == 0
    assert reconnect(rig)
    assert codes_of(srv)[2] == IT  # sent again, and it came up: the streak ends
    assert counts(rig) == stats(sent=1, suspect=1)
    assert not rig.route._language_streak and not rig.route._language_refused


def test_two_closes_in_a_row_remember_the_refusal(gemini, rigs):
    srv = gemini(refuse_language=UNKNOWN_FIELD_CLOSE)
    rig = rigs(srv)
    assert rig.route.ready()
    assert counts(rig) == stats(suspect=1)
    assert not any(d.get("reason") == "language_refused" for d in rig.diag)
    assert reconnect(rig)
    assert codes_of(srv) == [IT, {}, IT, {}]
    assert counts(rig) == stats(suspect=2, refused=1)
    assert any(d.get("reason") == "language_refused" for d in rig.diag)
    assert reconnect(rig)  # the next connect leaves the codes out: one setup
    assert codes_of(srv)[4:] == [{}] and srv.language_closes == 2
    assert counts(rig) == stats(suspect=2, refused=1, skipped=1)
    snap = rig.route.snapshot()  # the wire counts setups sent with codes, closed ones too
    assert snap["wire"]["total"]["language_codes_sent"] == 2


def test_a_language_change_tries_its_own_code(gemini, rigs):
    srv = gemini(refuse_language=UNKNOWN_FIELD_CLOSE)
    rig = rigs(srv, rt={"transcription_language_refusals": 1})
    assert rig.route.ready()
    assert counts(rig)["refused"] == 1
    rig.stack.set("voice.realtime_route.transcription_language", "en", source="test")
    rig.route.apply_config(rig.stack.effective())
    assert reconnect(rig)  # the refusal was it-IT's: en-US is tried
    assert codes_of(srv)[2] == {"languageCodes": ["en-US"]}


def test_new_codes_try_again(gemini, rigs):
    srv = gemini(refuse_language=UNKNOWN_FIELD_CLOSE)
    rig = rigs(srv, rt={"transcription_language_refusals": 1})
    assert rig.route.ready()
    rig.stack.set(
        "voice.realtime_route.transcription_language_tags", {"it": "it-CH"}, source="test"
    )
    rig.route.apply_config(rig.stack.effective())
    assert reconnect(rig)
    assert codes_of(srv)[2] == {"languageCodes": ["it-CH"]}
    assert counts(rig)["refused"] == 2


def test_a_refusal_of_both_setups_is_one_failed_connect_and_changes_nothing(gemini, rigs):
    srv = gemini(refuse=2)  # 1008 on the setup with the codes and on the one without
    rig = rigs(srv, rt={"refusals_to_report": 2})
    assert not rig.route.ready()
    assert len(srv.setups) == 2 and srv.closes == 2
    assert rig.route._refusals == 1 and rig.route.reason == "connect_failed"  # one, not two
    assert rig.route.snapshot()["connect_failures"] == 1
    assert not rig.route._language_refused and not rig.route._language_streak
    assert wait_for(lambda: rig.route.connected, 10.0)  # the background reconnect
    assert codes_of(srv)[2] == IT
    assert counts(rig) == stats(sent=1)


def test_a_close_without_a_close_frame_is_not_reopened(gemini, rigs):
    srv = gemini(drop=1)  # 1006: a network loss, not the server's answer to the setup
    rig = rigs(srv)
    assert not rig.route.ready()
    assert codes_of(srv) == [IT] and rig.route.reason == "connect_failed"
    assert counts(rig) == stats()


def test_a_close_raised_while_the_setup_is_written_is_reopened(gemini, rigs, monkeypatch):
    original = RealtimeClient._send_update
    calls: list[int] = []

    async def closing(self, ws, update):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionClosedError(Close(1008, ""), None)
        await original(self, ws, update)

    monkeypatch.setattr(RealtimeClient, "_send_update", closing)
    srv = gemini()
    rig = rigs(srv)
    assert rig.route.ready()
    assert codes_of(srv) == [{}]  # only the setup without the codes reached the server
    assert counts(rig) == stats(suspect=1)


def test_a_first_close_that_takes_the_budget_ends_as_a_timeout(gemini, rigs):
    # connect_timeout_s 2 (the rig): the first close at 1.7 s, the second setup needs 0.5 s
    srv = gemini(
        refuse_language=UNKNOWN_FIELD_CLOSE, refuse_language_delay_s=1.7, setup_delay_s=0.5
    )
    rig = rigs(srv)
    assert not rig.route.ready()
    assert rig.route.reason == "connect_timeout"
    assert counts(rig) == stats() and not rig.route._language_streak


@pytest.mark.parametrize("refuse_language", [None, UNKNOWN_FIELD_CLOSE])
def test_a_gemini_name_hint_counts_as_before(gemini, rigs, refuse_language):
    srv = gemini(refuse_language=refuse_language)
    rig = rigs(srv)
    rig.route.transcription_name = lambda: "Neve"
    assert rig.route.ready()
    assert rig.route.snapshot()["transcription_hint"] == {"sent": 1, "refused": 0}
    assert rig.route._session_hint == "Neve"


def test_an_idle_close_after_a_refusal_reconnects_without_the_codes(gemini, rigs):
    srv = gemini(refuse_language=UNKNOWN_FIELD_CLOSE)
    rig = rigs(srv, rt={"transcription_language_refusals": 1, "idle_close_min_age_s": 1.0})
    assert rig.route.ready()
    assert counts(rig) == stats(suspect=1, refused=1)
    srv.idle_close_s = 1.2
    assert reconnect(rig)
    assert wait_for(lambda: rig.route.snapshot()["idle_closes"] == 1, 10.0)
    srv.idle_close_s = None
    assert rig.route.ready()  # the next turn's reconnect
    assert codes_of(srv)[-1] == {} and counts(rig)["skipped"] == 2


def test_the_check_sends_the_codes_without_a_fallback_and_reports_it(gemini):
    srv = gemini([["audio"]], refuse_language=UNKNOWN_FIELD_CLOSE)
    cfg = check_cfg(check_gemini_route(srv.base_url()), input="audio", connect_timeout_s=5.0)
    report = run_check(cfg)
    assert report.language_codes_sent and not report.ok
    assert report.close_code == UNKNOWN_FIELD_CLOSE and len(srv.setups) == 1  # shown, not hidden


def test_the_preview_sends_no_codes(gemini):
    srv = gemini([["audio"]], refuse_language=UNKNOWN_FIELD_CLOSE)
    assert run_preview(preview_cfg(preview_gemini_route(srv.base_url()), input="audio"))
    assert srv.setups[0]["inputAudioTranscription"] == {} and srv.language_closes == 0


def test_the_switch_off_sends_no_codes_and_no_reopen(gemini, rigs):
    srv = gemini(refuse=1)
    rig = rigs(srv, rt={"transcription_language_hint": False})
    assert not rig.route.ready()  # a 1008 is a failed connect: no second setup
    assert codes_of(srv) == [{}]
    assert wait_for(lambda: rig.route.connected, 10.0)
    assert counts(rig) == stats()
