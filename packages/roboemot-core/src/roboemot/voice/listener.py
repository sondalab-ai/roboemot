"""The voice listener: microphone chunks to utterances (SPEC 8 stages 2-4).

One thread reads ``source(timeout)`` (the audio hub's voice queue), and for each chunk:

1. drops it while the robot speaks (``SpeechActivity``) after handing it to
   ``on_speaking_chunk`` (the barge-in detector, SPEC 8 stage 8), and resets the pipeline on that
   transition and on a gap, since the audio is no longer contiguous. ``resume`` is the detector's
   way back: it feeds the audio from the barge-in onset and keeps the gate open until the echo
   tail is over, so the user's first words are not lost. When the window opens on a sound of
   the robot's own that is not speech (a self-soothing hum, SPEC 7B) while someone's speech is
   in progress or starting (``voice_active``), ``cut_own_sound`` (``bind_own_sound``) cuts that
   sound and the utterance goes on instead: the gate stays open until the cut's echo tail is
   over;
2. runs the VAD endpointer; the ASR only ever sees speech (pre-roll included);
3. on speech start picks the language, runs the fast path (head turn, ``speech.heard``) and
   starts the recognizer; on each partial runs the keyword spotter;
4. on speech end finishes the ASR and submits the utterance's grammar pass to the grammar
   worker (its own thread, ``roboemot.voice.grammar_worker``), then keeps reading audio;
5. hands the utterance off once the pass is done, its wait (``voice.grammar.handoff_timeout_s``)
   is over (free-text keywords only, counted ``timeouts``). The next utterance starting, or a
   gap, waits for the pass within the same deadline first (on a lagging listener the next
   utterance is already queued); the robot speaking, a barge-in or a pause hand off at once with
   the result so far (counted ``superseded``). A finished utterance is never dropped. The
   hand-off runs the wake gate (with the voice-at-face check, whose ``direction`` samples are
   taken on speech start and on each piece of utterance audio) and builds an ``Utterance``:
   handed to ``on_utterance`` (the conversation, in-process) and announced on the bus as
   ``percept.utterance`` without its text. ``on_activity`` sees the utterance in progress until
   then.

With the realtime route's audio input (``bind_realtime``, SPEC 8, Realtime speech route), an
utterance is decided ``addressable`` when its speech starts, with the wake gate's own predicates
(``WakeGate.addressable``: the wake word not required, or a follow-up in the open window). An
addressable one skips Vosk and is only buffered; at its end the voice-at-face check runs, and
only if the gate then addresses it does its audio go (``RealtimeInput``, off this thread). One
the check rejects is decoded by Vosk from the buffer and goes only if the wake word is in it.
Speech that is not addressable runs Vosk as before; if the wake decision addresses it, its
buffered audio goes. The final of an utterance whose audio went is the server's transcript: it
waits in the same hand-off queue with the deadline ``transcript_timeout_s``, so this thread
never waits on the network. Past the deadline (or when the audio was refused) the utterance is
decoded locally from its buffer and handed off: a refused one for the chat routes, one whose
audio left marked ``local_only`` (the did-not-get-it cue shows at once, through ``on_audio``,
and the conversation does not answer it; its keywords still count). Hand-offs keep the order
of the utterances; each carries the fast path's state of its utterance, so a newer utterance
can begin while an older one waits.

With server turns (``voice.realtime_route.turns: server``) a speech segment streams instead,
while it is heard: at its start the realtime input says whether it may stream at all
(``stream_refusal``), then the server gates (``server_gates``: music, another known speaker,
the voice at a face, that face looking) decide once, at the start or, for the two that need the
voice direction, after ``server_gate_hold_ms`` of speech held on the robot. A segment that
streams skips Vosk, its held audio goes at once and the rest piece by piece; at its end the
stream closes and its final waits for the server's transcript as a burst's does, addressed
(no later check; the wake word policy and the follow-up window do not gate it). A segment a
gate keeps local is heard by Vosk from its start and decided by the wake gate as without server
turns; if addressed it goes as text. A listed gate whose detector is off keeps the segment
local (it fails closed). A segment whose stream never opened (a turn in flight, no session, a
revoke) is decided by the wake gate too. A segment cut part way (the robot speaks, a gap, a
pause) cuts its stream. A barge-in over the robot's voice (S2, ``stream_playback``) may stream
one segment while the robot speaks: chunks then go to the barge-in and to the pipeline, until
that segment ends, the robot stops, or the barge-in releases it (``end_playback``). Its
listening cue (``AUDIO_SENT``) begins only once the server confirmed it and the robot's speech
stopped (S3: no cue over the robot's voice); its counts go when its opening went, as for any
stream. One the server never confirmed by the end of its transcript wait is dropped: never
decoded locally nor answered (heard over the robot's voice, ``playback_unconfirmed``), and the
did-not-get-it cue (``AUDIO_DROPPED``) shows once the robot's speech stopped (owner decision
2026-11-08). The drop is decided with the input, so a confirmation after it confirms nothing.

Music (``music``, a ``MusicDetector`` while ``music.enabled``): every chunk heard while the robot
does not speak is fed to the detector after the speaking check (the ``music`` stage); a dropped
speaking chunk and a chunk gap tell it where the hole began. Audio replayed by ``resume`` is not
fed: it is the user's voice after a cut. The wake gate asks it whether music played at any time
in the utterance (``WakeGate``: the wake word is then required, ``rejected_music`` without it),
also when an utterance that skipped Vosk ends. ``music.started`` and ``music.stopped`` go on the
bus with the tempo only. While the robot's own radio is audible and ``tools.radio.dance`` is on,
the detector and the classifier hear the radio's own pieces instead of the microphone: a robot
with a hardware echo canceller removes its radio from the microphone. The media player hands
each piece that went to the speaker to ``hear_media`` (its feeder thread: one append to a
bounded deque, the detector's window of pieces), stamped when it starts to play; this thread
alone takes them and feeds them, so the detector keeps one feeding thread. The microphone's
chunks then reach neither (no hole either: the radio's stamps carry their own holes); when the
radio is held the microphone feeds them again once the radio's last piece has played, after one
hole.

The robot's own radio (``media_audible``, the media player's ``audible_at``, SPEC 8, Radio
playback): while it can be heard as an utterance starts, only the wake word addresses the robot
(``WakeGate``, ``rejected_media``: the follow-up window, a face and the policy do not count;
the Live tab's ``unanswered``/``media``, with the local final while the live transcript is on),
and with server turns no segment streams (the ``media`` gate, always on: the wake gate decides
each one locally). While the player holds the radio (robot speech, an open addressed turn, an
announcement's reply window, ``open_follow_up``, ``reply_window_open``, and
``tools.media.resume_after_s`` after them) no lyric can be heard and the normal rules apply,
the ``media`` gate and the name-first shortcut included: "Neve" then "stoppa la musica" is a
follow-up. An utterance the wake word addressed over the audible radio tells the player
(``media_hold``) that its turn holds the music from its last speech, so the person's next words
find it held even before the player's feeder saw the turn. Such a wake over the audible radio
must be confirmed by the server's transcript when one comes: a lyric can match the name in the
small local grammar, so a server final that holds neither the name nor a wake phrase (the
server spotter, case-insensitive), nor a close spelling of the name (``FastPath.
server_name_close``: "Nivea" for "Neve", never "nave"; an extra name such as "robot" counts
for none of these over the radio, nor wakes the robot there by itself), is handed off not
addressed, counted
``wake_unconfirmed_media`` (the Live tab's ``unanswered``/``media_unconfirmed``, with the
server's words while the live transcript is on): the conversation deletes
its server items, plays no cue, and the radio rejoins as after any turn; the window it opened
closes. An utterance addressed without the wake word before the verdict (the lyric's next line
starts inside the span the hold back-dates: a follow-up in that window, or ``not_required``
under policy ``never``) goes with it when its own final holds neither either.
Stop trusts the command (owner decision 2026-11-19: a missed stop costs more than a false one):
when such an unconfirmed server final holds a ``stop`` phrase or a ``media_stop`` verb, and a
``media`` word, of its language only (``voice.keywords``, whole words: "Ehi, Alexa, ferma la
musica.", "spegni la radio") while a station is
set (``media_set``) and ``tools.radio.stop_without_name`` is on, the listener stops the radio
(``media_stop``, reason ``voice_stop``), counts ``voice_stop_media`` instead of
``wake_unconfirmed_media`` and publishes ``media``/``voice_stop`` instead of
``media_unconfirmed``; the turn is still handed off not addressed, so nothing else is answered
without the name. A local rejection (``rejected_media``) never reaches this rule.
A face over the radio (owner decision 2026-11-20, ``tools.radio.face_addresses``): when the switch
is on, the policy does not require the wake word (``never``, or ``unless_enrolled_in_view`` with an
enrolled person in view; under ``always`` Vosk listens for the name as before), the voice-at-face
check is on and reads a real sound direction over the radio (``direction_over_media``: the host
sets it from the manifest's hardware echo cancellation, whose readings over the radio reach only
this check) and the utterance's audio can go to a realtime route with robot turns
(``_wants_audio``), an utterance that starts while the radio is audible skips the local speech
check (``_face_media_at``: no Vosk, no grammar pass, no local decode; counted ``vosk_skipped``) and
the voice-at-face check alone decides at its end (``_face_media_end``), on direction readings only
(a degraded check, or audio that can no longer go, decodes the buffer for the wake word instead):
passed, the wake gate's ``face_media``, its audio goes as a media wake the server's words must
confirm (counted ``face_wake_media``, the Live tab's ``media``/``face_wake``), held as a wake, and
the rules above decide it (answered with the name, the radio stopped by a stop with a media word,
else dropped); failed, ``rejected_media`` with no words and nothing sent. A face wake whose server
final never comes is dropped, never decoded (``face_wake_unheard``); one refused before it left
(its session never came) is decoded for the wake word (``face_wake_unsent``). With server turns
no burst can go, so the ``media`` gate's local path stays as it is.
Otherwise a wake in a follow-up window or a reply window (the radio held) needs no
confirmation. With ``tools.radio.dance`` off the
music detector ignores the radio while it is audible: its chunks are holes to it, and
``music_state`` and the wake gate's music predicate say no music.

Connect on speech (SPEC 8, Realtime speech route; owner decision 2026-11-20): when speech
starts with no session open and the utterance could go as audio without its wake word
(``_addressable_rule``, ``_face_media_rule``), the realtime input has the route reopen one it
closed on demand (a conversation's end, Gemini's idle close: ``_connect_on_speech``), before the
speech start's checks read ``_wants_audio``, which counts that connect as a session; so the
Vosk skip, the face rule over the radio and the decision at the end agree, and the burst waits
for the connect within its deadline less ``connect_wait_reserve_s``. One that needs the wake
word reopens it at its decision. Each decision that sends audio tries again (an idle close
during the utterance). A connect that fails or comes too late refuses the burst: the local path
decodes it and it is answered with the robot's own words.

The duck over speech (SPEC 8, Radio playback, owner decision 2026-11-18): each utterance's start
and end also go to the player: ``media_speech_start`` at the VAD start, ``media_speech_end``
once no utterance is in progress or waiting for its decision (the local final, the grammar pass,
the server's transcript), so the duck never ends before an addressed turn's hold takes over.
The player decides: it ducks the audible music only on a robot with hardware echo cancellation.

Decision events (``roboemot.diag``, SPEC 12A): an utterance with words that the wake gate does
not address publishes ``unanswered`` with the rule that rejected it and the voice-at-face
check's numbers (the facing check's median head yaw and its limit); a facing check that did not
trust the yaw on a busy CPU publishes ``facing``/``cpu_high``; a transcript that did not come
publishes ``audio``/``transcript_timeout``. No speaker id, no text, except a recognizer's words
over the radio as ``heard``, only with the live transcript on: the server's of an unconfirmed
wake (``_heard``, a server final only) and the local final of a ``rejected_media``
(``_live_words``); no other rejection carries words.

Chunk stamps are the monotonic time the chunk *ends*; the endpointer wants its start. The work
per utterance is bounded by ``voice.vad.max_utterance_s``. The ``realtime`` block of the report
(``ListenerLoad``) says whether the listener keeps up: its time per second of audio by stage, its
lag behind the microphone and the audio lost before it read it (SPEC 13). Audio and text are
never logged.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .. import shutdown
from ..bus import Bus
from ..config.schema import RoboEmotConfig
from ..diag import EventSink, no_events
from .activity import SpeechActivity
from .asr import ASRBackend, ASREvent, resolve_language
from .decode_worker import DecodeJob, DecodeWorker
from .direction import HEARD, VoiceDirection
from .fastpath import NAME_FIRST, WAKE, FastPath, GrammarPass, UtteranceState
from .grammar_worker import GrammarJob, GrammarWorker
from .keywords import holds_any
from .listener_load import ListenerLoad
from .music import STARTED, MusicDetector, MusicState
from .music_classifier import MusicClassifier
from .speech_features import Prosody
from .vad import Endpointer, SpeechAudio, SpeechEnd, SpeechStart
from .wake import AT_FACE, NO_FACE_AT_VOICE, NO_FACING_DATA, NOT_FACING, AtFace, WakeGate

log = logging.getLogger(__name__)
TOPIC = "percept.utterance"
POLL_S = 0.1
JOIN_S = 2.0
MS_PER_S = 1000.0
# what became of an utterance's audio (``bind_realtime``'s ``on_audio``, the listening cue)
AUDIO_SENT, AUDIO_UNSENT, AUDIO_NOT_HEARD = "sent", "unsent", "not_heard"
# a streamed segment the server answered on its own and whose transcript never came: the
# listening cue ends, nothing more (the answer played; server turns, SPEC 8)
AUDIO_ANSWERED = "answered"
# a barge-in's stream the server never confirmed (S3, owner decision 2026-11-08): dropped, not
# answered; the did-not-get-it cue shows once the robot's speech stopped (no listening cue began)
AUDIO_DROPPED = "dropped"
# an addressed utterance not heard (no transcript, or a local decode with no word) that the
# robot asks about in its own words (``on_not_heard``, SPEC 8, Robot expression): the
# did-not-get-it cue shows its gesture without its sound
AUDIO_NOT_HEARD_SPOKEN = "not_heard_spoken"
# the server gates (voice.realtime_route.server_gates) that need the voice direction's samples:
# they decide after server_gate_hold_ms of speech
HELD_GATES = frozenset({"at_face", "facing"})
SERVER_GATES = ("music", "other_speaker", "at_face", "facing")  # decided by the listener
OFF_GATES = ("music", "at_face", "facing")  # gates with a detector that can be off
NO_UNADDRESSED = "no_unaddressed_opt_in"  # turns: server without its opt-in (robot turns run)
# what in the server's transcript confirms a wake over the audible radio (SPEC 8, Radio playback)
WAKE_CONFIRMS = frozenset({"name", "wake"})  # not ``extra_name``: a lyric can say "robot"
# the wake gate's outcomes that open a window as a wake (``face_media``: the voice-at-face check
# stood in for the wake word over the audible radio, SPEC 8, Radio playback)
WAKES = frozenset({"wake_word", "face_media"})
FACE_WAKE = "face_wake"  # the ``media`` event of a face wake sent for the server to confirm
VOICE_STOP = "voice_stop"  # ``media.stopped``'s reason, and the ``media`` event's (SPEC 8, 11)


class VoiceChunkLike(Protocol):
    samples: np.ndarray
    rate_hz: int
    stamp: float
    gap: bool


@dataclass(frozen=True)
class Utterance:
    """A finished utterance. ``text`` is the user's speech: in-process only, hidden from repr.

    Stamps are monotonic seconds; latencies count from ``last_speech_stamp``. ``addressed`` is
    the wake-word decision: the conversation answers only addressed utterances.
    """

    text: str = field(repr=False)
    language: str
    start_stamp: float
    last_speech_stamp: float
    endpoint_stamp: float
    addressed: bool
    wake_word: bool
    keywords: frozenset[str]
    prosody: Prosody
    final: bool = True
    # The realtime route's audio input (SPEC 8, Realtime speech route): the server session
    # (generation) and its user items that hold this utterance's words; None when its audio
    # did not go. ``local_only``: its audio went but no transcript came; it is not answered
    # (the did-not-get-it cue showed, SPEC 8, Non-verbal cues).
    realtime: tuple[int, tuple[str, ...]] | None = field(default=None, repr=False)
    local_only: bool = False
    # with server turns, the live stream that carried it (a ``RealtimeInput`` job): the server
    # answered it on its own, and the conversation joins these words to that turn's history
    stream: Any = field(default=None, repr=False, compare=False)
    # the voice-at-face rule addressed it (the voice came from a face): speech clearly meant
    # for the robot, which the model asks about when unclear (SPEC 8, Robot expression)
    at_face: bool = False


@dataclass(frozen=True)
class UtteranceInfo:
    """The ``percept.utterance`` payload: everything about an utterance except what was said."""

    language: str
    chars: int
    words: int
    start_stamp: float
    last_speech_stamp: float
    endpoint_stamp: float
    addressed: bool
    wake_word: bool
    rms_db: float
    pitch_hz: float | None
    words_per_s: float | None


@dataclass
class _Handoff:
    """A finished utterance waiting until ``deadline`` for its grammar pass (``job``), for
    the realtime route's transcript of its audio (``rt``) or for its local decode (``decode``,
    then its grammar pass). ``state`` is the fast path's state
    of the utterance (held at its end); ``addressed`` the wake decision once made (None
    before); ``face`` the voice-at-face check's verdict once asked (None: not asked);
    ``heard`` what its grammar pass heard."""

    ev: SpeechEnd
    text: str = field(repr=False)
    language: str
    grammar_pass: GrammarPass | None
    job: GrammarJob | None
    ended: float
    deadline: float
    state: UtteranceState | None = None
    rt: Any = None  # the RealtimeInput job (realtime.input.InputJob)
    addressed: bool | None = None
    face: str | None = None
    face_info: dict[str, Any] = field(default_factory=dict)  # that check's numbers
    face_seen: str | None = None  # the check's verdict when the wake gate asked it at its end
    heard: frozenset[str] = field(default=frozenset(), repr=False)
    decode: DecodeJob | None = None  # its local decode, off the listener thread
    local_only: bool = False  # its audio left without a transcript: not answered (not_heard)
    # a face wake whose audio was refused before it left, decoded instead: never offered again,
    # and its listening cue ends at the decision (the did-not-get-it cue unless addressed)
    no_audio: bool = False
    progress: _DecodeProgress | None = field(default=None, repr=False)  # its local decode's
    decoded_end: int | None = None  # index in ``ev.audio`` the local text was decoded up to
    stream: bool = False  # its audio is a live stream's (turns: server)
    announced: bool = False  # its stream's opening went, and that was counted
    cued: bool = False  # ... and its listening cue began (AUDIO_SENT; a barge-in's waits, S3)
    media_wake: bool = False  # only the wake word addressed it, over the audible radio
    media: bool = False  # the radio was audible as it started: extra names do not wake (SPEC 8)
    # a follow-up in the window a media wake opened, decided before that wake's verdict
    media_parent: _Handoff | None = field(default=None, repr=False)
    unconfirmed: bool = False  # the server's words did not confirm its wake: not answered
    # the voice-at-face check stood in for the wake word over the radio (``face_media``): only a
    # server final may confirm it, never a local one (Vosk never heard it)
    face_wake: bool = False


@dataclass
class _Stream:
    """A speech segment with server turns (SPEC 8, Realtime speech route): its audio is held
    (``held``, the pre-roll first) until the gates decide, then it streams (``job``, a
    ``RealtimeInput`` stream) or Vosk hears it (``gate``: the gate that kept it local)."""

    start: float
    held: list[np.ndarray] = field(repr=False)
    speech_s: float = 0.0  # speech heard after the start (the pre-roll not counted)
    job: Any = None
    gate: str | None = None
    face: str | None = None  # the voice-at-face verdict the gates asked for (asked once)
    face_info: dict[str, Any] = field(default_factory=dict)
    playback: bool = False  # a barge-in's segment over the robot's speech (S2): gates decided


class _DecodeProgress:
    """What a local decode has heard so far, written by the decode worker after each piece and
    read by the listener at the deadline (one tuple: a single store). ``stop`` asks the worker
    to end the decode after its current piece."""

    def __init__(self) -> None:
        self.heard: tuple[str, int] = ("", 0)  # the words so far, and the audio index up to them
        self.stop = False


class VoiceListener:
    """See the module docstring. ``process`` is the synchronous step; ``start`` runs it.
    ``grammar_background`` runs the grammar pass on its own thread (the app); without it the
    pass runs inline at the end of the utterance (tests, deterministic)."""

    def __init__(
        self,
        cfg: RoboEmotConfig,
        source: Callable[[float], VoiceChunkLike | None],
        endpointer: Endpointer,
        asr: ASRBackend,
        speech: SpeechActivity,
        fastpath: FastPath,
        on_utterance: Callable[[Utterance], None],
        bus: Bus | None = None,
        *,
        speaker_language: Callable[[], str | None] = lambda: None,
        speaker_id: Callable[[], str | None] = lambda: None,
        on_speaking_chunk: Callable[[VoiceChunkLike], None] | None = None,
        on_speaking_end: Callable[[], None] | None = None,
        before_asr_reset: Callable[[], None] | None = None,
        on_activity: Callable[[bool], None] | None = None,
        enrolled_in_view: Callable[[float], bool] = lambda now: False,
        direction: VoiceDirection | None = None,
        clock: Callable[[], float] = time.monotonic,
        timer: Callable[[], float] = time.perf_counter,
        grammar_background: bool = False,
    ) -> None:
        self._source = source
        self._endpointer = endpointer
        self._asr = asr
        self._speech = speech
        self._fast = fastpath
        self._on_utterance = on_utterance
        self._bus = bus
        self._speaker_language = speaker_language
        self._speaker_id = speaker_id
        self._on_speaking_chunk = on_speaking_chunk
        self._on_speaking_end = on_speaking_end
        self._before_asr_reset = before_asr_reset
        self._on_activity = on_activity
        self._enrolled_in_view = enrolled_in_view
        self._clock = clock
        self._timer = timer
        self._direction = direction
        # whether the robot's own radio could be heard at a stamp, and the hold of a turn the
        # wake word opened over it (the media player's ``audible_at`` and ``hold_from``, set at
        # start)
        self.media_audible: Callable[[float], bool] = lambda at: False
        self.media_hold: Callable[[float], None] = lambda at: None
        self.media_set: Callable[[], bool] = lambda: False  # a station is set (held or not)
        # stop the station by voice without the name (``voice_stop``: the player's stop and a nod)
        self.media_stop: Callable[[str], None] = lambda reason: None
        # an utterance's VAD start, and the moment every utterance is decided, for the
        # player's duck (``duck_from`` and ``release_from``; the player applies the guard)
        self.media_speech_start: Callable[[float], None] = lambda at: None
        self.media_speech_end: Callable[[float], None] = lambda at: None
        # whether the voice direction is read over the robot's own radio for the voice-at-face
        # check (a hardware echo canceller; the host sets it from the manifest): without it the
        # face rule over the radio cannot pass, so the local check runs (SPEC 8, Radio playback)
        self.direction_over_media = False
        # the radio's pieces as they play (``hear_media``, the player's thread), taken by this
        # thread only; sized in apply_config to the detector's window
        self._media_pieces: collections.deque[tuple[np.ndarray, float]] = collections.deque()
        self._music_from_media = False  # the detector's last audio was the radio's pieces
        self._media_until = -math.inf  # where the last radio piece fed to it ends
        self._radio_music = False  # the detector's music started on the radio's pieces
        self._media_wake: _Handoff | None = None  # the media wake whose verdict is still due
        self._gate = WakeGate(  # see apply_config
            "always", 0.0, music=self._music_during, media=self._media_at
        )
        self.music: MusicDetector | None = None  # built by apply_config while music.enabled
        self.music_classifier: MusicClassifier | None = None  # set by the pipeline
        self.events: EventSink = no_events  # the Live tab's decision events (wired at start)
        self._face_info: dict[str, Any] = {}  # the last voice-at-face check's numbers
        self.load = ListenerLoad(cfg.voice.listener)
        self._last_end: float | None = None  # stamp of the end of the last chunk read
        self._grammar = GrammarWorker(
            background=grammar_background,
            on_spent=lambda seconds: self.load.stage("grammar", seconds),
            timer=timer,
            clock=clock,
        )
        self._decoder = DecodeWorker(
            background=grammar_background,
            on_spent=lambda seconds: self.load.stage("decode", seconds),
            timer=timer,
            clock=clock,
        )
        # finished utterances waiting for their final, in order (the head goes first)
        self._queue: collections.deque[_Handoff] = collections.deque()
        self._handoff = {"waits": 0, "timeouts": 0, "superseded": 0, "dropped_on_stop": 0}
        self._cut_own_sound: Callable[[], bool] | None = None  # bind_own_sound
        self._realtime: Any = None  # the realtime route's audio input (bind_realtime)
        self._on_audio: Callable[[str, float], None] = lambda event, key: None  # bind_realtime
        # an addressed utterance was not heard (``no_transcript`` or ``decodes_lost``): whether
        # the robot asks about it in its own words (SPEC 8, Robot expression; the expressor)
        self.on_not_heard: Callable[[str], bool] = lambda why: False
        self._audio = {
            "vosk_skipped": 0,
            "sent": 0,
            "transcribed": 0,
            "wake_word_heard": 0,  # the server's transcript held a wake phrase or the name
            # a wake over the audible radio whose server transcript held neither: not answered
            "wake_unconfirmed_media": 0,
            # one that said stop the music instead: the radio stopped, still not answered
            "voice_stop_media": 0,
            # over the audible radio, a voice from a face sent for the server to confirm
            # (tools.radio.face_addresses), one of those whose server final never came:
            # dropped, never decoded locally, and one refused before it left (its session never
            # came): decoded for the wake word
            "face_wake_media": 0,
            "face_wake_unheard": 0,
            "face_wake_unsent": 0,
            "transcript_timeouts": 0,
            "local_fallbacks": 0,
            "decodes": 0,
            "decode_timeouts": 0,
            "decodes_capped": 0,
            "decodes_lost": 0,  # an addressed utterance whose decode heard nothing: not_heard cue
            # turns: server (SPEC 8): segments streamed, not streamed (no session, a turn in
            # flight, the budget...), and kept local by each server gate
            "streamed": 0,
            "streams_refused": 0,
            "turns_fallbacks": 0,  # turns: server without privacy.realtime_audio_unaddressed
            # a streamed segment the server answered on its own, its transcript never in
            "answered_untranscribed": 0,
            # S2: barge-ins over the robot's speech that streamed (their opening went), and
            # those whose stream was refused (the barge-in then cut the robot locally)
            "streamed_during_playback": 0,
            "playback_refused": 0,
            "playback_unsent": 0,  # its stream's opening was refused later: dropped
            "playback_unconfirmed": 0,  # it went, the server never confirmed it: dropped (S3)
            **{f"stream_gated_{gate}": 0 for gate in SERVER_GATES},
            "stream_gated_media": 0,  # the robot's own radio plays: always kept local
            # a listed gate whose detector is off or missing: it fails closed (kept local)
            **{f"stream_gated_{gate}_off": 0 for gate in OFF_GATES},
        }
        self._stream: _Stream | None = None  # the segment in progress with server turns
        # S2: a barge-in's stream over the robot's speech (``stream_playback``): chunks are
        # heard while the robot speaks; the next segment is the barge-in's, and only it streams
        self._playback = False
        self._playback_segment = False
        self._playback_ended = False  # the barge-in's segment ended: back to dropping
        self._playback_until = -math.inf  # its stream ends here unless the server confirmed
        self._playback_job: Any = None  # its stream's job, once it opened
        # start stamps of barge-in segments dropped unconfirmed, cued once the robot is silent
        self._unheard: collections.deque[float] = collections.deque()
        self._last_streamed: tuple[str | None, float] = (None, -math.inf)  # speaker, last speech
        self._vosk = True  # the utterance in progress runs the recognizer
        # the utterance in progress started over the audible radio with the face rule on: no
        # local recognizer, the voice-at-face check decides at its end (``_face_media_end``)
        self._face_media = False
        self._wait_last_s = 0.0
        self._wait_max_s = 0.0
        self.apply_config(cfg)
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._counts = {
            "utterances": 0,
            "empty": 0,
            "gaps": 0,
            "dropped_speaking": 0,
            "barge_ins": 0,
            "errors": 0,
            "name_first_no_face": 0,  # the name opened a long final, no face at the voice
            "media_rate_mismatch": 0,  # a radio piece at another rate than the detector's
        }
        # Written by the listener thread, read by snapshot(): single-word stores, atomic in
        # CPython, so no lock; the counters have their own.
        self._active = False  # an utterance is in progress or waiting for its hand-off
        self._recording = False  # the recognizer is taking the utterance's audio
        self._hearing = False  # speech in progress, from its start to its end (``hearing``)
        self._muted = False
        self._open_until = -math.inf  # barge-in handoff: chunks starting before this pass the gate
        self._state = "off"
        self._language = ""
        self._last_error = ""

    def apply_config(self, cfg: RoboEmotConfig) -> None:
        """Hot reload: keyword lists, wake policy and window, default language, VAD settings,
        the music detector (built when ``music.enabled`` turns on, reconfigured after that:
        turned off, it ends music at its next chunk)."""
        self._cfg = cfg
        self._endpointer.apply_config(cfg.voice.vad)
        self._fast.apply_config(cfg.voice)
        fp = cfg.voice.fastpath
        self._gate.apply(cfg.voice.wake_word_policy, fp.addressed_window_s, fp.window_max_s)
        self.load.apply_config(cfg.voice.listener)
        if self._direction is not None:
            self._direction.apply_config(cfg.voice.direction, cfg.voice.facing)
        pieces = max(1, round(cfg.music.window_s * MS_PER_S / cfg.tools.media.chunk_ms))
        if self._media_pieces.maxlen != pieces or not (cfg.music.enabled and cfg.tools.radio.dance):
            # a fresh deque, never a copy of the live one: the player's thread appends to it
            # meanwhile; turned off, nothing stale is fed after it is turned on again
            self._media_pieces = collections.deque(maxlen=pieces)
        if self.music is not None:  # last: a band above the Nyquist frequency raises here
            self.music.apply_config(cfg.music)
        elif cfg.music.enabled:
            self.music = MusicDetector(cfg.music, cfg.voice.sample_rate_hz)
            self.music.speech_heard = self.voice_active

    def bind_own_sound(self, cut: Callable[[], bool]) -> None:
        """``cut()`` cuts a sound of the robot's own that is not speech, when only such a sound
        plays, and says whether it did: someone's speech that is in progress as the sound starts
        then goes on (SPEC 7B, 8)."""
        self._cut_own_sound = cut

    def bind_realtime(
        self, realtime: Any, on_audio: Callable[[str, float], None] | None = None
    ) -> None:
        """Send addressed utterances' audio to the realtime route through ``realtime`` (a
        ``RealtimeInput``; None stops it). Bound after the route exists (start-up).
        ``on_audio(event, key)`` hears what became of an utterance's audio (``key``: its start
        stamp, the ``Utterance.start_stamp`` it is handed off with), for the listening cue
        (SPEC 8, Non-verbal cues): ``AUDIO_SENT`` when it is handed off to leave,
        ``AUDIO_UNSENT`` when it did not leave after all (refused, or dropped on stop) and
        ``AUDIO_NOT_HEARD`` when it left and no transcript came (timeout or failure)."""
        self._realtime = realtime
        self._on_audio = on_audio or (lambda event, key: None)

    def _not_heard(self, p: _Handoff, why: str) -> None:
        """``p``'s audio was not heard (``why``: ``no_transcript`` or ``decodes_lost``): for an
        addressed utterance (the wake decision taken: never a television) the robot may ask
        about it in its own words (``on_not_heard``), and then the did-not-get-it cue keeps its
        gesture without its sound (``AUDIO_NOT_HEARD_SPOKEN``). A streamed segment (server
        turns) is never asked about here: it streamed on the server gates, with no wake
        decision, so it may be a television."""
        spoken = False
        if p.addressed is True and not p.stream:
            try:
                spoken = bool(self.on_not_heard(why))
            except Exception as exc:  # a reaction must not stop the listener
                log.warning("voice listener: not-heard consumer raised %s", type(exc).__name__)
        self._audio_event(AUDIO_NOT_HEARD_SPOKEN if spoken else AUDIO_NOT_HEARD, p)

    def _audio_event(self, event: str, p: _Handoff) -> None:
        try:
            self._on_audio(event, p.ev.start_stamp)
        except Exception as exc:  # a cue must not stop the listener
            log.warning("voice listener: audio event consumer raised %s", type(exc).__name__)

    @property
    def config(self) -> RoboEmotConfig:
        """The config of the last ``apply_config``: the live values for the voice wiring."""
        return self._cfg

    @property
    def realtime_input(self) -> Any:
        """The realtime route's audio input (``bind_realtime``), None without one."""
        return self._realtime

    # ---- lifecycle

    def start(self) -> None:
        """Read the audio in a daemon thread (no-op when running). The thread loads the ASR
        models first, so a slow disk never holds up the caller; a model that fails to load is
        logged (type only) and leaves the listener ``unavailable``."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        thread = threading.Thread(target=self._run, name="roboemot-listener", daemon=True)
        thread.start()
        self._thread = thread

    def stop(self) -> None:
        """Stop the thread (waits briefly for the chunk in hand)."""
        self._stop.set()
        if self._thread is not None:
            shutdown.join(self._thread, JOIN_S)
            if not self._thread.is_alive():
                self._drop_pending()
            self._thread = None
        else:
            self._drop_pending()
        self._grammar.close()
        self._decoder.close()

    def pause_listening(self) -> None:
        """Run no VAD or recognizer until :meth:`resume_listening` (the robot is asleep, SPEC
        12A). The thread drops the utterance in hand, then keeps emptying the audio queue (so
        stale audio is not recognised on wake) and idles."""
        self._paused.set()

    def resume_listening(self) -> None:
        """Listen again after :meth:`pause_listening`, on the audio that arrives from now."""
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def close_follow_up(self) -> None:
        """Close the wake gate's follow-up window: a requested nap started (SPEC 7A), so only the
        robot's name or a wake phrase addresses the robot until a new one opens it."""
        self._gate.close_window()

    def open_follow_up(self) -> None:
        """Open the wake gate's follow-up window now: the robot's announcement ended (SPEC 8,
        robot-initiated turn), so the person can reply without the wake word. Any thread."""
        self._gate.open_window(self._clock())

    def reply_window_open(self) -> bool:
        """Whether the reply window an announcement opened (:meth:`open_follow_up`) is open
        now: the media player keeps the radio paused while it is (SPEC 8, Radio playback).
        Any thread."""
        return self._gate.reply_window(self._clock())

    def snapshot(self) -> dict[str, Any]:
        """Counters and fast-path latencies for the debug report: numbers only, no content."""
        with self._lock:
            snap: dict[str, Any] = {
                "state": self._state,
                "paused": self._paused.is_set(),
                **self._counts,
            }
        snap["latency"] = self._fast.stats.snapshot()
        snap["wake"] = self._gate.snapshot(self._clock())
        if self._direction is not None:
            snap["direction"] = self._direction.snapshot()
        snap["wake_phrases_out_of_vocab"] = self._fast.wake_out_of_vocab()
        grammar = self._fast.grammar_snapshot()
        if grammar is not None:
            snap["grammar"] = grammar
        snap["vad"] = self._endpointer.snapshot()
        snap["realtime"] = {**self.load.snapshot(), "recognizers": self._asr.stats()}
        if self.music is not None:
            snap["music"] = self.music.snapshot()
        if self.music_classifier is not None:
            snap["music_classifier"] = self.music_classifier.snapshot()
        with self._lock:
            snap["realtime_audio"] = dict(self._audio)
            snap["grammar_handoff"] = {
                **self._handoff,
                "pending": bool(self._queue),
                "replaced": self._grammar.replaced,
                "skipped_stale": self._grammar.skipped_stale,
                "wait_ms": {
                    "last": self._wait_last_s * MS_PER_S,
                    "max": self._wait_max_s * MS_PER_S,
                },
            }
        return snap

    def _run(self) -> None:
        models = list(self._cfg.voice.asr.models)
        primary = resolve_language(models, None, self._cfg.conversation.language)
        try:
            self._asr.load([primary])
        except Exception as exc:
            self._state = "unavailable"
            log.warning("voice listener: models: %s", type(exc).__name__)
            self._discard()
            return
        self._name_vocabulary()
        self._state = "running"
        for language in models:  # the others while already listening; utterances before that
            if language != primary and not self._stop.is_set():  # use the languages loaded so far
                try:
                    self._asr.load([language])
                    self._name_vocabulary()
                except Exception as exc:
                    log.warning("voice listener: model %s: %s", language, type(exc).__name__)
        was_paused = False
        while not self._stop.is_set():
            if self._paused.is_set():
                if not was_paused:  # on this thread: the endpointer and recognizer are ours
                    was_paused = True
                    try:
                        self._abort()
                    except Exception as exc:
                        log.warning("voice listener: pause: %s", type(exc).__name__)
                while (chunk := self._source(0.0)) is not None:
                    self._last_end = chunk.stamp  # a pause is not lost audio
                try:  # an utterance whose audio went still gets its final
                    self.poll_handoff()
                except Exception as exc:
                    log.warning("voice listener: paused hand-off: %s", type(exc).__name__)
                self._stop.wait(POLL_S)
                continue
            was_paused = False
            head = self._queue[0] if self._queue else None
            wait = POLL_S if head is None else min(POLL_S, head.deadline - self._clock())
            chunk = self._source(max(0.0, wait))
            try:
                if chunk is not None:
                    self.process(chunk)
                t0 = self._timer()
                self.poll_handoff()  # a hand-off here is listener work outside ``process``
                self.load.busy(self._timer() - t0)
            except Exception as exc:  # one bad chunk must not end the listener
                self._count("errors")
                name = type(exc).__name__
                if name != self._last_error:
                    self._last_error = name
                    log.warning("voice listener: %s", name)
                try:
                    self._abort()
                except Exception as again:  # nor a recognizer that fails to reset
                    log.warning("voice listener: abort: %s", type(again).__name__)

    def _name_vocabulary(self) -> None:
        """A model loaded: check the robot's name against its vocabulary now (logs once)."""
        self._fast.names.refresh()
        self._fast.apply_config(self._cfg.voice)

    def _discard(self) -> None:
        """Without a working recognizer keep draining the hub's queue so it never backs up."""
        while not self._stop.is_set():
            self._source(POLL_S)

    def _count(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    def _count_audio(self, key: str) -> None:
        with self._lock:
            self._audio[key] += 1

    def _set_active(self, active: bool) -> None:
        """Track "an utterance is in progress" and tell ``on_activity`` (the Live tab's phase) when
        it changes. A raising hook is logged, never allowed to break the audio path."""
        changed = self._active != active
        self._active = active
        if changed and not active:  # every utterance is decided (handed off or dropped)
            self._media_speech(False, self._clock())
        if changed and self._on_activity is not None:
            try:
                self._on_activity(active)
            except Exception as exc:
                log.warning("voice listener: activity hook: %s", type(exc).__name__)

    # ---- one chunk

    def _abort(self, wait: bool = False) -> None:
        """Drop the utterance in progress and every buffered sample; one that already ended is
        handed off first (``wait``: for its grammar pass, within its deadline). The barge-in gives
        the recognizer back first (``before_asr_reset``): its keyword worker may be using it."""
        self._settle(wait)
        self._cut_stream()
        self._recording = self._hearing = False
        if self._before_asr_reset is not None:
            try:
                self._before_asr_reset()
            except Exception as exc:
                log.warning("voice listener: asr hand-back: %s", type(exc).__name__)
        self._endpointer.reset()
        try:
            self._asr.reset()
        except Exception as exc:
            log.warning("voice listener: asr reset: %s", type(exc).__name__)
        self._fast.reset()
        self._set_active(False)

    def process(self, chunk: VoiceChunkLike) -> None:
        """Handle one chunk (see the module docstring) and account for it (``realtime``)."""
        t0 = self._timer()
        lag = self._clock() - chunk.stamp
        duration = chunk.samples.size / chunk.rate_hz
        lost = 0.0
        if chunk.gap and self._last_end is not None:
            lost = max(0.0, chunk.stamp - duration - self._last_end)
        if chunk.gap:  # the hole began where the last chunk ended
            self._music_gap(chunk.stamp - duration - lost)
        self._last_end = chunk.stamp
        try:
            self._process(chunk, chunk.stamp - duration)
        finally:
            self.load.chunk(duration, self._timer() - t0, lag, lost)

    def _timed(self, stage: str, fn: Callable[..., Any], *args: Any) -> Any:
        """``fn(*args)``, its wall time added to ``stage`` of the realtime account."""
        t0 = self._timer()
        try:
            return fn(*args)
        finally:
            self.load.stage(stage, self._timer() - t0)

    def _process(self, chunk: VoiceChunkLike, start: float) -> None:
        if chunk.gap:
            self._count("gaps")
            self._abort(wait=True)
            self._playback_ended = self._playback_ended or self._playback  # its segment is gone
        speaking = self._speech.speaking(start) or self._speech.speaking(chunk.stamp)
        if self._playback and speaking and not self._playback_ended:
            self._music_gap(start)  # the robot's own sound is never music
            if start >= self._playback_until and not self._playback_confirmed():
                # its own bound, whatever the barge-in does (S2: confirm_ms, then it ends)
                self.end_playback()
            self._speaking_hook(chunk)  # the duck, the confirm and the release (barge-in)
            if self._playback:  # still: the barge-in's segment is heard over the robot (S2)
                self._ingest(chunk.samples, start, chunk.stamp)
            else:  # released: it ended the stream
                self._count("dropped_speaking")
            return
        if self._playback:  # the robot stopped (the confirm, or its answer ended), or the
            self._playback = self._playback_segment = False  # segment ended: as without it
            if not speaking:
                self._speaking_ended()
        if speaking and start >= self._open_until and self._own_sound_cut():
            self._open_until = self._speech.until()  # the cut's echo tail: the utterance goes on
        if speaking and start >= self._open_until:
            if not self._muted:
                self._muted = True
                self._abort()
            self._count("dropped_speaking")
            self._music_gap(start)
            self._speaking_hook(chunk)
            return
        if self._muted and not speaking:
            self._muted = False
            self._speaking_ended()
        self._hear_music(chunk, start, speaking)
        self._ingest(chunk.samples, start, chunk.stamp)

    def _speaking_hook(self, chunk: VoiceChunkLike) -> None:
        """Hand a chunk heard while the robot speaks to the barge-in detector."""
        if self._on_speaking_chunk is not None:
            try:  # its hand-back (resume) replays audio: counted here and in its stages
                self._timed("bargein", self._on_speaking_chunk, chunk)
            except Exception as exc:
                log.warning("voice listener: speaking-chunk hook: %s", type(exc).__name__)

    def _hear_music(self, chunk: VoiceChunkLike, start: float, speaking: bool) -> None:
        """Feed the music detector one chunk the listener takes (the ``music`` stage) and
        publish a change of state. A chunk inside the robot's speaking window (the echo tail a
        cut left open) is a hole to it: the robot's own sound must never count as music. The
        classifier gets the same chunks (the ``music_classifier`` stage), and while it decides
        (``music.classifier.decides``) its scores go to the detector. While the robot's own
        radio is audible with ``tools.radio.dance`` off, every chunk is a hole as well; with it
        on, the detector hears the radio's own pieces instead (``_hear_media``) and the
        microphone's chunk not at all: behind a hardware echo canceller it holds the room
        without the radio, and its stamps would mix two timelines."""
        if self._media_feeds(start):
            self._hear_media()
            return
        if self._music_from_media:
            self._media_pieces.clear()
            if start < self._media_until:  # the radio's last piece is still playing
                return
            self._music_from_media = False
            if not self._media_set():  # the station stopped: room music counts again
                self._radio_music = False
            self._music_gap(self._media_until)  # one hole, from the radio's last piece
        speaking = speaking or self._music_hidden(start)
        if speaking:
            self._music_gap(start)
        self._music_audio(chunk.samples, start, chunk.stamp, heard=not speaking)

    def _music_audio(self, samples: np.ndarray, start: float, end: float, heard: bool) -> None:
        """One piece of audio for the classifier and the detector (``heard`` False: the
        classifier skips it; the caller told the detector the hole). Publishes a change, at
        ``end``: for a radio piece that is when it will have played, so the music state may
        lead the sound by up to one piece (the player pushes the next piece while the speaker
        still holds one)."""
        classifier, scores = self.music_classifier, None
        if heard and classifier is not None:
            scores = self._timed("music_classifier", classifier.feed, samples, start)
        music = self.music
        if music is None:
            return
        music.classifier_primary = (
            classifier is not None and classifier.running and self._cfg.music.classifier.decides
        )
        if not heard:
            return
        if scores is not None:
            self._publish_music(music, music.classified(*scores, start), end)
        event = self._timed("music", music.feed, samples, start)
        self._publish_music(music, event, end)

    def _media_set(self) -> bool:
        """Whether a station is set, held or not (the media player's ``playing``): music the
        detector started on the radio's pieces is ignored only while it is."""
        try:
            return bool(self.media_set())
        except Exception as exc:
            log.warning("voice listener: media player: %s", type(exc).__name__)
            return False

    def _music_gap(self, stamp: float) -> None:
        """A hole in the microphone's audio at ``stamp`` for the detector; none while it hears
        the radio's pieces, whose own stamps show their holes."""
        if self.music is not None and not self._music_from_media:
            self.music.gap(stamp)

    def _media_feeds(self, at: float) -> bool:
        """Whether the detector hears the radio's pieces now: a detector, ``music.enabled``,
        ``tools.radio.dance``, and the radio audible at ``at`` (SPEC 8, Radio playback). At the
        switch back to the microphone the detector sees one hole, from the end of the radio's
        last piece to the first microphone chunk after it (microphone chunks before it are
        skipped: they overlap the radio's timeline); a short hole keeps its envelope, a long
        one clears it, as any hole."""
        cfg = self._cfg
        if self.music is None or not cfg.music.enabled or not cfg.tools.radio.dance:
            return False
        return self._media_at(at)

    def hear_media(self, samples: np.ndarray, rate_hz: int, start: float) -> None:
        """A piece of the robot's own radio that went to the speaker and starts to play at
        ``start`` (the media player's ``on_piece``, its feeder thread): kept for the music
        detector while it may dance to the radio (``_media_feeds``), at most the detector's
        window of pieces. Never blocks: one append to a bounded deque (thread-safe), taken
        by the listener thread only. A piece at another rate than the detector's is dropped
        (the player decodes at ``voice.sample_rate_hz``, as the detector runs)."""
        cfg = self._cfg
        if self.music is None or not cfg.music.enabled or not cfg.tools.radio.dance:
            return
        if rate_hz != cfg.voice.sample_rate_hz:
            self._count("media_rate_mismatch")
            return
        self._media_pieces.append((samples, start))

    def _hear_media(self) -> None:
        """Feed the detector (and the classifier) the radio's pieces waiting, in play order,
        on this thread: the same analysis as for a microphone chunk, never a second decode."""
        self._music_from_media = True
        rate = self._cfg.voice.sample_rate_hz
        while self._media_pieces:
            try:
                samples, start = self._media_pieces.popleft()
            except IndexError:  # emptied by a reload meanwhile
                return
            end = start + samples.size / rate
            if end <= self._media_until:  # out of order: never feed the detector backwards
                continue
            self._media_until = end
            self._music_audio(samples, start, end, heard=True)

    def _publish_music(self, music: MusicDetector, event: str | None, stamp: float) -> None:
        """Put a change of the music state on the bus; no change publishes nothing. A start
        notes whether the detector heard it on the radio's pieces (``_music_during``)."""
        if event is not None:
            self._radio_music = event == STARTED and self._music_from_media
        if event is None or self._bus is None:
            return
        topic = "music.started" if event == STARTED else "music.stopped"
        payload = {"tempo_bpm": music.state(stamp).tempo_bpm}
        try:
            self._bus.publish(topic, payload, stamp)
        except Exception as exc:
            log.warning("voice listener: %s subscriber raised %s", topic, type(exc).__name__)

    def _music_during(self, start: float, end: float) -> bool:
        """The wake gate's music predicate: whether music played at any time in [start, end]
        (lock-free, one frozen view of the detector). False without a detector, while
        ``music.enabled`` is off, while the radio is audible with ``tools.radio.dance`` off, and
        while a station is set and the music the detector holds started on the radio's own
        pieces: while the radio is audible the media rule covers it, and while the radio is
        held no lyric can be heard, though the detector carries that music through the hold for
        the dance (SPEC 8, Radio playback). Music it heard on the microphone counts as ever, and
        so does music once the station stopped (room music right after the radio)."""
        music = self.music
        if music is None or not self._cfg.music.enabled or self._music_hidden(start):
            return False
        return not (self._radio_music and self._media_set()) and music.during(start, end)

    def music_state(self, now: float) -> MusicState | None:
        """The detector's state at ``now`` for the dance's room beat (lock-free, no blocking);
        None without a detector or while ``music.enabled`` is off; no music while the radio
        plays with ``tools.radio.dance`` off (the dance does not start)."""
        music = self.music
        if music is None or not self._cfg.music.enabled:
            return None
        if self._music_hidden(now):
            return MusicState(False, None, None, 0.0, now)
        return music.state(now)

    def _media_at(self, at: float) -> bool:
        """Whether the robot's own radio could be heard at ``at`` (the media player's
        ``audible_at``): the wake gate's media rule, the ``media`` stream gate, the name-first
        shortcut, the dance switch. Takes only the player's leaf lock, never the wake gate's,
        so the gate may ask it under its own lock."""
        try:
            return bool(self.media_audible(at))
        except Exception as exc:
            log.warning("voice listener: media player: %s", type(exc).__name__)
            return False

    def _music_hidden(self, at: float) -> bool:
        """Whether the detector ignores music at ``at``: the radio is audible and
        ``tools.radio.dance`` is off (SPEC 8, Radio playback)."""
        return not self._cfg.tools.radio.dance and self._media_at(at)

    def _hold_media(self, at: float) -> None:
        """Tell the media player an addressed turn holds the radio from ``at``."""
        try:
            self.media_hold(at)
        except Exception as exc:
            log.warning("voice listener: media player: %s", type(exc).__name__)

    def _media_speech(self, started: bool, at: float) -> None:
        """Tell the media player an utterance started (``started``) or ended at ``at``: it
        ducks the music over the speech where the robot's echo canceller allows it."""
        try:
            (self.media_speech_start if started else self.media_speech_end)(at)
        except Exception as exc:
            log.warning("voice listener: media player: %s", type(exc).__name__)

    def _own_sound_cut(self) -> bool:
        """Someone's speech is in progress or starting as the speaking window opens: cut the
        robot's own sound if only such a sound plays (``bind_own_sound``); True when it did."""
        if self._muted or self._cut_own_sound is None or not self.voice_active():
            return False
        try:
            return bool(self._cut_own_sound())
        except Exception as exc:
            log.warning("voice listener: own-sound cut: %s", type(exc).__name__)
            return False

    def resume(self, audio: np.ndarray, start: float) -> None:
        """Barge-in handoff (SPEC 8 stage 8), called from the speaking-chunk hook after the
        speech was stopped: ``audio`` runs from the user's speech onset (pre-roll included)
        and starts at monotonic ``start``. The pipeline restarts on it, and the gate stays open
        until the speaking window the stop left behind (the echo tail) is over."""
        self._abort()
        self._muted = False
        self._open_until = self._speech.until()
        self._ingest(audio, start, start + audio.size / self._cfg.voice.sample_rate_hz)
        self._count("barge_ins")

    def stream_playback(self, audio: np.ndarray, start: float) -> bool:
        """Server turns, S2 (SPEC 8, 12): the barge-in heard the person over the robot's voice
        (its echo guard passed) and asks to let the server confirm it instead of cutting the
        robot off: from the speaking-chunk hook, ``audio`` from the onset (pre-roll included)
        at monotonic ``start``. True when the segment may stream: the input's checks for a
        stream (both opt-ins, ``audio_gate``, awake, a session, no realtime turn in flight,
        the budget; not the robot's silence) and the server gates now (``music``,
        ``other_speaker``; a listed gate that needs the voice direction refuses: the direction
        is muted while the robot speaks). Then the pipeline restarts on ``audio``, chunks are
        heard while the robot speaks, and the next segment streams (only it). False (counted
        ``playback_refused`` with server turns): nothing changes, the barge-in cuts locally."""
        why = self._playback_refusal(start)
        if why is not None:
            if why not in ("turns", NO_UNADDRESSED):
                self._count_audio("playback_refused")
            return False
        self._abort()
        self._muted = False
        self._playback = self._playback_segment = True
        self._playback_ended = False
        self._playback_job = None
        end = start + audio.size / self._cfg.voice.sample_rate_hz
        # counted from the end of the handed audio, at or after the barge-in's cut-off point
        # (where its own wait starts): the barge-in releases first, so a confirmation it takes
        # as on time always finds this stream (a late one resumes the listener)
        self._playback_until = end + self._cfg.voice.barge_in.duck.confirm_ms / MS_PER_S
        self._ingest(audio, start, end)
        return True

    def end_playback(self) -> None:
        """The barge-in's stream got no confirmation in time (S2, from the speaking-chunk
        hook): the segment in progress is dropped (its stream cut, its items deleted) and
        chunks are dropped again while the robot speaks."""
        if not self._playback:
            return
        self._playback = self._playback_segment = False
        self._muted = True
        self._abort()

    def playback_job(self) -> Any:
        """The stream job of the last barge-in's segment (S2), None before it opened: the
        barge-in takes the server's confirmation of that stream only."""
        return self._playback_job

    def _playback_confirmed(self) -> bool:
        job = self._playback_job
        return job is not None and job.confirmed

    def server_turns(self) -> bool:
        """Whether a barge-in would be confirmed by the server now (``turns: server`` with its
        opt-in): the barge-in keys its counters by it."""
        rt = self._realtime
        if rt is None:
            return False
        try:
            return rt.stream_refusal(self._cfg, playback=True) not in ("turns", NO_UNADDRESSED)
        except Exception:
            return False

    def _playback_refusal(self, start: float) -> str | None:
        """Why a barge-in's segment may not stream over the robot's speech, None when it may
        (``stream_playback``)."""
        rt = self._realtime
        if rt is None:
            return "turns"
        try:
            why = rt.stream_refusal(self._cfg, playback=True)
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)
            return "error"
        if why is not None:
            return why
        held = [g for g in self._cfg.voice.realtime_route.server_gates if g in HELD_GATES]
        if held:
            return held[0]
        return self._stream_gate(_Stream(start, []), self._clock())

    def _speaking_ended(self) -> None:
        if self._on_speaking_end is not None:
            try:
                self._on_speaking_end()
            except Exception as exc:
                log.warning("voice listener: speaking-end hook: %s", type(exc).__name__)

    def _ingest(self, samples: np.ndarray, start: float, audio_end: float) -> None:
        """Run ``samples`` (starting at ``start``) through the endpointer and the recognizer."""
        buffered: list[np.ndarray] = []
        for ev in self._timed("vad", self._endpointer.feed, samples, start):
            if isinstance(ev, SpeechAudio):
                buffered.append(ev.samples)
                continue
            self._feed(buffered, audio_end)
            buffered = []
            if isinstance(ev, SpeechStart):
                self._begin(ev, audio_end)
            elif isinstance(ev, SpeechEnd):
                self._end(ev)
        self._feed(buffered, audio_end)

    def _feed(self, parts: list[np.ndarray], audio_end: float) -> None:
        if parts and self._recording:
            if self._vosk:
                self._partial(
                    self._timed("asr_accept", self._asr.accept, np.concatenate(parts)), audio_end
                )
            self._sample_direction(audio_end)
            if self._stream is not None:  # after the direction sample: a held gate reads it
                self._stream_feed(np.concatenate(parts), audio_end)

    def _partial(self, event: ASREvent | None, audio_end: float) -> None:
        if event is not None and event.text:
            self._fast.partial(event.text, audio_end)

    # ---- utterance edges

    def awaiting_transcript(self) -> bool:
        """Whether an utterance's audio went to the realtime route and its transcript has not
        come yet (the Live tab's "sending to the cloud"). Safe from any thread: a copy of the
        queue, nothing waits."""
        try:
            return any(p.rt is not None for p in tuple(self._queue))
        except RuntimeError:  # the queue changed during the copy: ask again at the next poll
            return False

    def hearing(self) -> bool:
        """Whether someone's speech is in progress now (from its start to its end): the
        listening blip waits for a quiet moment (SPEC 8, Non-verbal cues). Safe from any thread
        (a single flag)."""
        return self._hearing

    def voice_active(self) -> bool:
        """Whether someone's speech is in progress or may be starting (the VAD is counting
        frames above its threshold towards a start): no self-soothing sound starts then, and no
        one-shot cue sound plays (SPEC 7B, 8). Safe from any thread: two single-word reads
        without the listener's lock. A read racing the listener thread can be one chunk stale,
        which costs at most one tick of hold or one cue sound; it never blocks the audio path."""
        return self._hearing or self._endpointer.onset

    def _begin(self, ev: SpeechStart, audio_end: float) -> None:
        self._hearing = True  # before the hand-offs below: their blip must not cover this speech
        self._settle(wait=True)  # the per-utterance state below is the pending one's until then
        try:
            speaker = self._speaker_language()
        except Exception:
            speaker = None
        self._language = resolve_language(
            self._asr.languages, speaker, self._cfg.conversation.language
        )
        self._fast.begin(ev.stamp, audio_end, self._language)
        self._cut_stream()  # none should be left: a segment's stream ends with it
        # asked once: could it go as audio without the wake word once a session is up? Only
        # then does its start reopen a session (SPEC 8, Connect on speech), before the checks
        # below read whether audio would go, so they agree on the session
        bound = self._realtime is not None
        addressable = bound and self._addressable_rule(ev.stamp)
        if addressable or (bound and self._face_media_rule(ev.stamp)):
            self._connect_on_speech()
        self._stream = self._stream_start(ev)
        self._face_media = self._stream is None and self._face_media_at(ev.stamp)
        self._vosk = self._stream is None and not self._face_media
        self._vosk = self._vosk and not (addressable and self._wants_audio())
        if self._vosk:
            self._timed("asr_start", self._asr.start, self._language)
        else:
            self._count_audio("vosk_skipped")  # its audio goes if the gate addresses it
        self._recording = True
        self._set_active(True)
        self._media_speech(True, ev.stamp)  # after the settle: its end must not release this
        if self._direction is not None:
            self._direction.begin()
        self._sample_direction(audio_end)
        if self._vosk:
            self._partial(self._timed("asr_accept", self._asr.accept, ev.pre_roll), audio_end)
        elif self._stream is not None and not self._held_gates():
            self._stream_decide(audio_end)  # no gate needs the voice direction: decided now

    # ---- server turns (SPEC 8, Realtime speech route): live streams

    def _stream_start(self, ev: SpeechStart) -> _Stream | None:
        """A segment starts with server turns: its audio is held for the gates, unless it
        cannot stream now (counted ``streams_refused``: no session, a turn in flight, the
        budget, the robot asleep or speaking; it then runs as without server turns)."""
        rt = self._realtime
        playback, self._playback_segment = self._playback_segment, False
        if rt is None:
            return None
        try:
            why = rt.stream_refusal(self._cfg, playback=playback)
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)
            return None
        if why == "turns":  # robot turns: the burst path
            return None
        if why == NO_UNADDRESSED:  # turns: server without its opt-in: robot turns run
            self._count_audio("turns_fallbacks")
            return None
        if why is not None:
            self._count_audio("streams_refused")
            if why == "no_session":  # this segment runs locally; the next one may stream
                self._reopen_for_streams(rt)
            return None
        return _Stream(ev.stamp, [ev.pre_roll], playback=playback)

    def _reopen_for_streams(self, rt: Any) -> None:
        """A segment could not stream for want of a session (server turns): the route reopens
        one it closed on demand (``connect_on_speech``), for the next segment. Not with the
        upload budget spent (``stream_refusal`` asks for the session first): nothing could go."""
        rt_cfg = self._cfg.voice.realtime_route
        try:
            if rt.route.upload_left_s(self._cfg) <= rt_cfg.close_silence_ms / MS_PER_S:
                return
            rt.route.connect_on_speech()
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)

    def _held_gates(self) -> bool:
        """Whether a listed gate needs the voice direction's samples, and it can take them."""
        gates = self._cfg.voice.realtime_route.server_gates
        d = self._direction
        held = any(g in HELD_GATES for g in gates) and d is not None and d.enabled
        return held and not (self._stream is not None and self._stream.playback)

    def _stream_feed(self, samples: np.ndarray, audio_end: float) -> None:
        """One piece of the segment's audio: streamed, held for the gates, or (kept local)
        already Vosk's."""
        s = self._stream
        assert s is not None
        if s.job is not None:
            self._realtime.stream(s.job, samples)
            return
        if s.gate is not None:
            return
        s.held.append(samples)
        s.speech_s += samples.size / self._cfg.voice.sample_rate_hz
        if s.speech_s * MS_PER_S >= self._cfg.voice.realtime_route.server_gate_hold_ms:
            self._stream_decide(audio_end)

    def _stream_decide(self, now: float) -> None:
        """The server gates decide, once, whether the segment streams: then its held audio
        goes at once and the rest live; else (counted ``stream_gated_<gate>``) Vosk hears it
        from its start, and the wake gate decides at its end as without server turns."""
        s = self._stream
        assert s is not None
        gate = None if s.playback else self._stream_gate(s, now)  # decided at the barge-in
        held, s.held = np.concatenate(s.held), []
        if gate is None:
            rate = self._cfg.voice.sample_rate_hz
            s.job = self._realtime.open_stream(held, rate, clock=self._clock, playback=s.playback)
            if s.playback:
                self._playback_job = s.job
            return  # counted once its opening went (_announce)
        s.gate = gate
        self._count_audio(f"stream_gated_{gate}")
        self._vosk = True
        self._timed("asr_start", self._asr.start, self._language)
        self._partial(self._timed("asr_accept", self._asr.accept, held), now)

    def _stream_gate(self, s: _Stream, now: float) -> str | None:
        """The first server gate (``voice.realtime_route.server_gates``) that keeps the segment
        from streaming, None when none does. Asleep is the input's own check, always. A listed
        gate whose detector is off or missing fails closed (``<gate>_off``): ``music`` without
        the music detector (``music.enabled``), ``at_face`` and ``facing`` without the voice
        direction, ``facing`` without ``voice.facing.enabled``. The verdict of the voice-at-face
        check is kept, so the wake gate does not ask it again. While the robot's own radio
        can be heard as the segment starts, ``media`` keeps every segment local whatever is
        listed (SPEC 8, Radio playback: only the wake word addresses the robot then)."""
        if self._media_at(s.start):
            return "media"
        gates = self._cfg.voice.realtime_route.server_gates
        if "music" in gates:
            if self.music is None or not self._cfg.music.enabled:
                return "music_off"
            if self._music_during(s.start, now):
                return "music"
        if "other_speaker" in gates:
            speaker = self._speaker()
            last, last_end = self._last_streamed
            window = self._cfg.voice.fastpath.addressed_window_s
            if speaker and last and speaker != last and s.start - last_end <= window:
                return "other_speaker"
        held = [g for g in gates if g in HELD_GATES]
        if held:
            d = self._direction
            if d is None or not d.enabled:
                return f"{held[0]}_off"
            if "facing" in gates and not self._cfg.voice.facing.enabled:
                return "facing_off"
            verdict = s.face = self._face()
            s.face_info = dict(self._face_info)
            if verdict == NO_FACE_AT_VOICE and ("at_face" in gates or "facing" in gates):
                return "at_face" if "at_face" in gates else "facing"
            if verdict in (NOT_FACING, NO_FACING_DATA) and "facing" in gates:
                return "facing"
        return None

    def _stream_end(self, ev: SpeechEnd, s: _Stream) -> None:
        """A streamed segment ended: its stream closes (the closing silence goes) and its final
        waits for the server's transcript until ``transcript_timeout_s``, as a burst's does.
        The listening cue begins now, not while the person still speaks."""
        rate = self._cfg.voice.sample_rate_hz
        face, info = s.face, s.face_info
        if face is None and self._direction is not None and self._direction.enabled:
            # asked now, on this segment's own samples: should its opening be refused, the wake
            # gate decides later, after the next segment's start replaced them
            face, info = self._face(), dict(self._face_info)
        p = self._handoff_for(ev, face)
        p.face_info = info
        p.addressed = True  # the gates let it stream: no later check, once its opening went
        p.stream = True
        p.deadline = self._clock() + self._cfg.voice.realtime_route.transcript_timeout_s
        p.rt = s.job
        # the server may answer before its transcript: the turn starts from these (latency)
        s.job.start_stamp, s.job.speech_end = ev.start_stamp, ev.last_speech_stamp
        self._realtime.close_stream(s.job, self._last_sample(ev) / rate * MS_PER_S, p.deadline)
        self._queue.append(p)
        self.poll_handoff()

    def _announce(self, p: _Handoff) -> None:
        """A streamed segment's opening went (the stream thread may send it after the segment
        ended): only now is it counted (``streamed``, ``sent``), does it count as the last
        streamed one (``other_speaker``) and does the listening cue begin (``_cue``)."""
        if not p.stream or p.rt is None or not p.rt.sent:
            return
        if not p.announced:
            p.announced = True
            self._last_streamed = (self._speaker(), p.ev.last_speech_stamp)
            self._count_audio("streamed")
            if p.rt.playback:
                self._count_audio("streamed_during_playback")
            self._count_audio("sent")
        self._cue(p)

    def _drop_unconfirmed(self, job: Any) -> bool:
        """Drop a barge-in's stream unless the server confirmed it, decided with the input
        (``drop_unconfirmed``) so a confirmation racing the drop is one or the other."""
        try:
            return bool(self._realtime.drop_unconfirmed(job))
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)
            return not job.confirmed

    def _cue_unheard(self) -> None:
        """The did-not-get-it cue of the barge-in segments dropped unconfirmed, once the
        robot's speech (echo tail included) stopped: never over its voice."""
        if not self._unheard or self._speech.speaking(self._clock()):
            return
        while self._unheard:
            key = self._unheard.popleft()
            try:
                self._on_audio(AUDIO_DROPPED, key)
            except Exception as exc:  # a cue must not stop the listener
                log.warning("voice listener: audio event consumer raised %s", type(exc).__name__)

    def _cue(self, p: _Handoff) -> None:
        """Begin ``p``'s listening cue, once. A barge-in's stream (S2) waits until the server
        confirmed it and the robot's own speech stopped, and never begins once its answer was
        taken (S3: the cue never shows over the robot's voice; ``poll_handoff`` asks again)."""
        job = p.rt
        if p.cued:
            return
        if job.playback and (
            not job.confirmed or job.answered or self._speech.speaking(self._clock())
        ):
            return
        p.cued = True
        self._audio_event(AUDIO_SENT, p)

    def _cut_stream(self) -> None:
        """The segment in progress is dropped part way: its stream is cut (the closing silence
        goes, its items are deleted)."""
        s, self._stream = self._stream, None
        if s is not None and s.job is not None and self._realtime is not None:
            try:
                self._realtime.cut_stream(s.job)
            except Exception as exc:
                log.warning("voice listener: realtime input: %s", type(exc).__name__)

    def _connect_on_speech(self) -> None:
        """Speech starts, or an utterance is about to be decided: with no session open, the
        realtime input has the route start the connect a turn would make (SPEC 8, owner
        decision 2026-11-20: ``RealtimeInput.speech_started``, never blocks, one at a time), so
        ``_wants_audio`` counts it and the utterance's audio waits for it."""
        rt = self._realtime
        if rt is None:
            return
        try:
            rt.speech_started(self._cfg)
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)

    def _wants_audio(self) -> bool:
        """Whether an addressed utterance's audio would go to the realtime route now (cheap:
        no network, no lock held across it): a session open, or a connect in flight that its
        burst waits for."""
        rt = self._realtime
        if rt is None:
            return False
        try:
            return bool(rt.wants(self._cfg))
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)
            return False

    def _face_media_at(self, start: float) -> bool:
        """Whether the utterance starting at ``start`` is left to the voice-at-face check over
        the audible radio (``tools.radio.face_addresses``, owner decision 2026-11-20): the
        switch on, a policy that does not require the wake word now (``never``, or
        ``unless_enrolled_in_view`` with an enrolled person in view), the radio audible, the
        check on and the utterance's audio able to go to a realtime route with robot turns,
        whose transcript then confirms it, and a live sound direction over the radio
        (``direction_over_media``: the robot reads it behind a hardware echo canceller; else the
        check could only take a degraded path, never trusted here). It skips the local
        recognizer, its grammar pass and any local decode (CPU)."""
        return self._face_media_rule(start) and self._wants_audio()

    def _face_media_rule(self, start: float) -> bool:
        """``_face_media_at`` but for the audio path: the switch, the policy, the radio audible
        and a live sound direction over it."""
        if not self._cfg.tools.radio.face_addresses or self._at_face() is None:
            return False
        d = self._direction
        if not (self.direction_over_media and d is not None and d.doa_available):
            return False  # no reading over the radio: Vosk listens for the name as today
        if self._gate.policy_requires(self._enrolled(start)):
            return False  # the policy needs the wake word: Vosk listens for it as today
        return self._media_at(start)

    def _addressable_rule(self, start: float) -> bool:
        """Whether the wake gate would address the utterance starting at ``start`` without the
        wake word (its voice-at-face check aside), whatever the audio path. With audio that
        would go (``_wants_audio``) the utterance skips Vosk; either this or the face rule
        (``_face_media_rule``) lets its start reopen a session (SPEC 8, Connect on speech).
        One that needs the wake word reopens it at its decision (``_deliver``), so what leaves
        the robot is what left before."""
        return self._gate.addressable(
            enrolled_in_view=self._enrolled(start), start=start, speaker=self._speaker()
        )

    def _end(self, ev: SpeechEnd) -> None:
        """The utterance ended: the recognizer's final (seconds on a loaded robot) and the
        grammar pass submitted. The Live phase stays ``listening`` until it is handed off or
        dropped."""
        if not self._recording:
            return
        self._recording = self._hearing = False
        if self._playback:  # the barge-in's segment ended: back to dropping (_process)
            self._playback_ended = True
        s, self._stream = self._stream, None
        if s is not None and s.job is None and s.gate is None:
            self._stream = s  # shorter than the hold: the gates decide now
            self._stream_decide(ev.last_speech_stamp)
            self._stream = None
        if s is not None and s.job is not None:
            self._stream_end(ev, s)
            return
        if self._face_media:  # over the radio: the voice-at-face check alone decides
            self._face_media = False
            self._face_media_end(ev)
            return
        if not self._vosk:  # buffered only: addressed now, or decoded from the buffer
            face = self._face()
            if not self._send_unheard(ev, face):
                p = self._handoff_for(ev, face)
                self._decode(p, wake=True)
                self._queue.append(p)
                self.poll_handoff()
            return
        try:
            text = self._timed("asr_finish", self._asr.finish).strip()
        except BaseException:
            self._set_active(bool(self._queue))
            raise
        if not text:
            self._count("empty")
            self._fast.reset()
            self._set_active(bool(self._queue))
            return
        grammar_pass = self._fast.grammar_pass(text, ev.audio[: self._last_sample(ev)])
        now = self._clock()
        deadline = now + self._cfg.voice.grammar.handoff_timeout_s
        job = None if grammar_pass is None else self._grammar.submit(grammar_pass.run, deadline)
        held = self._fast.hold()
        p = _Handoff(ev, text, self._language, grammar_pass, job, now, deadline, held)
        if s is not None and s.face is not None:  # the gates asked the voice-at-face check
            p.face, p.face_info = s.face, s.face_info
        self._queue.append(p)
        self.poll_handoff()

    def _handoff_for(self, ev: SpeechEnd, face: str | None) -> _Handoff:
        """The hand-off of an utterance that skipped Vosk, its final still to come."""
        now = self._clock()
        held = self._fast.hold()
        info = self._face_info if face is not None else {}
        return _Handoff(
            ev, "", self._language, None, None, now, now, held, face=face, face_info=info
        )

    def _face(self) -> str | None:
        """The voice-at-face check's verdict, asked once for an utterance; None without one."""
        check = self._at_face()
        return None if check is None else check()

    def _decided(self, p: _Handoff, outcome: str, text: str, *, wordless: bool = False) -> None:
        """Publish the wake decision of an utterance with words: ``unanswered`` with the rule
        when it was rejected, and the facing check's CPU fallback when it took one. ``text`` is
        the local final; a rejection over the radio (``rejected_media``) carries it as ``heard``
        under the live transcript's gate (``_live_words``), no other rejection carries words.
        ``wordless``: an utterance no local recognizer heard (the face rule over the radio) is
        published all the same, with no words."""
        info = p.face_info
        if info.get("facing") == "cpu_high":
            self.events(
                "facing",
                "cpu_high",
                path=info.get("path"),
                cpu_percent=info.get("cpu_percent"),
                cpu_max_percent=info.get("cpu_max_percent"),
            )
        if not (text.strip() or wordless) or not outcome.startswith("rejected_"):
            return
        words = {}
        if outcome == "rejected_media":
            words = {"heard": self._live_words(text) if text.strip() else None}
        self.events(
            "unanswered",
            outcome.removeprefix("rejected_"),
            policy=self._cfg.voice.wake_word_policy,
            facing=info.get("facing"),
            yaw_deg=info.get("yaw_deg"),
            limit_deg=info.get("limit_deg"),
            **words,
        )

    def _face_media_end(self, ev: SpeechEnd) -> None:
        """An utterance that started over the audible radio with the face rule on ended
        (``_face_media_at``): no local words yet. When the radio still counts at its start, the
        voice-at-face check ran on direction readings (``HEARD``, never a degraded path) and
        its audio can still go (``_wants_audio``), the wake gate decides: ``face_media`` sends
        it as a wake the server's transcript must confirm (``_finish``); a voice the readings
        put at no face (or a face looking away) is ``rejected_media``, nothing sent, no words.
        Anything else (a degraded check, the audio no longer able to go, the radio no longer
        counting at its start) is decoded from its buffer for the wake word, as
        ``_send_unheard`` does with a reject, and decided on those words (``_deliver``)."""
        face = self._face()
        d = self._direction
        heard = d is not None and d.last_path == HEARD
        start = ev.start_stamp
        if heard and self._media_at(start) and self._realtime:
            self._connect_on_speech()  # a session closed since its start (an idle close)
        if heard and self._media_at(start) and self._wants_audio() and self._realtime:
            p = self._handoff_for(ev, face)
            outcome = self._gate.outcome(
                wake_word=False,
                enrolled_in_view=self._enrolled(start),
                start=start,
                last_speech=ev.last_speech_stamp,
                speaker=self._speaker(),
                at_face=None if face is None else (lambda: face),
                face_media=True,
            )
            if outcome == "face_media":
                p.addressed = p.media = p.media_wake = p.face_wake = True
                self._decided(p, outcome, "")  # its CPU fallback
                self._hold_media(ev.last_speech_stamp)  # its turn holds the radio, as a wake's
                self._count_audio("face_wake_media")
                self.events("media", FACE_WAKE)
                self._note_wake(p, outcome)
                self._send(p)  # no wake word: its burst does not say the conversation is open
                self._queue.append(p)
                self.poll_handoff()
                return
            if outcome == "rejected_media":
                p.addressed = False
                self._decided(p, outcome, "", wordless=True)
                self._set_active(self._recording or bool(self._queue))
                return
            # the policy changed meanwhile: the gate decided already, so decode, not re-decide
            p.addressed = not outcome.startswith("rejected")
            self._note_wake(p, outcome)
            self._decode(p)
            self._queue.append(p)
            self.poll_handoff()
            return
        p = self._handoff_for(ev, face)
        self._decode(p, wake=True)
        self._queue.append(p)
        self.poll_handoff()

    def _send_unheard(self, ev: SpeechEnd, face: str | None) -> bool:
        """An utterance that skipped Vosk ended: if the wake gate addresses it without the
        wake word (its voice-at-face check passed) its audio goes and its final waits for the
        server. False: it is decoded locally (the check rejected it, the window closed, or music
        played at any time in it)."""
        if self._realtime is None:  # unbound meanwhile
            return False
        start, speaker = ev.start_stamp, self._speaker()
        enrolled = self._enrolled(start)
        if face not in (None, AT_FACE) or not self._gate.addressable(
            enrolled_in_view=enrolled, start=start, speaker=speaker, end=ev.last_speech_stamp
        ):
            return False
        outcome = self._gate.outcome(
            wake_word=False,
            enrolled_in_view=enrolled,
            start=start,
            last_speech=ev.last_speech_stamp,
            speaker=speaker,
            at_face=None if face is None else (lambda: face),
        )
        addressed = not outcome.startswith("rejected")
        p = self._handoff_for(ev, face)
        p.addressed = addressed
        self._note_wake(p, outcome)
        if addressed:
            self._decided(p, outcome, "")  # its CPU fallback; words come with its final
        if addressed:
            self._connect_on_speech()  # a session closed since its start (an idle close)
            self._send(p)
        else:  # decided already (a reload changed the policy meanwhile): decoded, not re-decided
            self._decode(p)
        self._queue.append(p)
        self.poll_handoff()
        return True

    def _send(self, p: _Handoff, *, wake_word: bool = False) -> None:
        """Hand ``p``'s audio to the realtime input (never blocks); its final now waits for the
        server's transcript until ``transcript_timeout_s``. ``wake_word``: the wake gate's wake
        decision held for it (its burst then says the conversation is open)."""
        rate = self._cfg.voice.sample_rate_hz
        p.deadline = self._clock() + self._cfg.voice.realtime_route.transcript_timeout_s
        # past its deadline it is never sent: the listener has decoded it locally by then
        p.rt = self._realtime.submit(
            self._speech_audio(p.ev),
            rate,
            deadline=p.deadline,
            clock=self._clock,
            wake_word=wake_word,
        )
        self._count_audio("sent")
        self._audio_event(AUDIO_SENT, p)

    def _speech_audio(self, ev: SpeechEnd) -> np.ndarray:
        """The utterance's audio from its first to its last speech frame, pre-roll included."""
        return ev.audio[: self._last_sample(ev)]

    def _decode(self, p: _Handoff, *, wake: bool = False) -> None:
        """Decode ``p``'s buffer for its local final, off this thread (the decode worker), on a
        recognizer of its own, in one pass, waited for until ``local_decode_timeout_s``; its
        grammar pass follows (``poll_handoff``). The audio goes in pieces of
        ``wake_decode_max_s`` of speech (the first with the pre-roll before it), and the words
        so far are kept after each (``_DecodeProgress``), so a decode out of time still gives
        them. ``wake``: the decode looks for the wake word (a voice-at-face reject); it stops
        after the first piece unless that piece wakes the robot (counted ``decodes_capped``)."""
        rt = self._cfg.voice.realtime_route
        rate = self._cfg.voice.sample_rate_hz
        language, audio, state = p.language, self._speech_audio(p.ev), p.state
        first = max(0, round((p.ev.start_stamp - p.ev.audio_stamp) * rate))
        step = max(1, round(rt.wake_decode_max_s * rate))
        bounds = list(range(first + step, audio.size, step)) + [audio.size]
        progress = p.progress = _DecodeProgress()

        def run() -> str:
            dec = self._asr.decoder(language)
            if dec is None:
                return ""
            start = 0
            for end in bounds:
                dec.accept(audio[start:end])
                start = end
                progress.heard = (dec.text().strip(), end)
                if end == audio.size or progress.stop:
                    break
                woke = self._fast.wakes(progress.heard[0], state=state)
                if wake and end == bounds[0] and not woke:
                    self._count_audio("decodes_capped")
                    break
            text = dec.finish()
            progress.heard = (text, start)
            return text

        p.deadline = self._clock() + rt.local_decode_timeout_s
        p.decode = self._decoder.submit(run, p.deadline)
        self._count_audio("decodes")

    def _decoded(self, p: _Handoff) -> None:
        """``p``'s local decode is over (or out of time): its text, then its grammar pass
        (submitted like any utterance's, under ``voice.grammar.handoff_timeout_s``)."""
        job, p.decode = p.decode, None
        progress = p.progress
        if job is None or not job.ran:  # out of time: the words heard so far, if any
            self._count_audio("decode_timeouts")
            if progress is not None:
                progress.stop = True
        p.text, p.decoded_end = progress.heard if progress is not None else ("", 0)
        if not p.text:
            return
        audio = p.ev.audio[: p.decoded_end]  # the grammar pass hears what the text came from
        p.grammar_pass = self._fast.grammar_pass(p.text, audio, p.language)
        now = self._clock()
        p.deadline = now + self._cfg.voice.grammar.handoff_timeout_s
        if p.grammar_pass is not None:
            p.job = self._grammar.submit(p.grammar_pass.run, p.deadline)

    def _last_sample(self, ev: SpeechEnd) -> int:
        """Index in ``ev.audio`` just past the last speech frame (the end silence off)."""
        rate = self._cfg.voice.sample_rate_hz
        return round((ev.last_speech_stamp - ev.audio_stamp + self._endpointer.frame_s) * rate)

    def poll_handoff(self) -> None:
        """Hand waiting utterances off, in order, while the first one's final is in: its
        grammar pass is done or its wait is over, or the realtime route's transcript came or
        its deadline passed (the listener thread calls this after every chunk and every empty
        read)."""
        for waiting in self._queue:
            self._announce(waiting)
        self._cue_unheard()
        while self._queue:
            p = self._queue[0]
            if p.decode is not None:
                if not p.decode.done and self._clock() < p.deadline:
                    return
                self._decoded(p)
                if not p.text:  # nothing heard, or out of time
                    self._queue.popleft()
                    self._count("empty")
                    if p.addressed and not p.local_only:  # never dropped without a word
                        self._count_audio("decodes_lost")
                        self._not_heard(p, "decodes_lost")
                    elif p.no_audio:  # a decoded face wake that heard nothing (it looked)
                        self._audio_event(AUDIO_NOT_HEARD, p)
                    self._set_active(self._recording or bool(self._queue))
                    continue
            if p.rt is not None:
                if not self._rt_settled(p):
                    if self._clock() < p.deadline:
                        return
                    if p.rt.sent:
                        self._count_audio("transcript_timeouts")
                        self._rt_timeout(p)
                    else:  # it never left (a hold, a session that never came): no transcript
                        self._cancel_job(p.rt)  # was due, so no timeout; decoded locally
                self._hand_off(frozenset())
            elif p.job is None or (p.job.done and p.job.ran):
                self._hand_off(p.job.result if p.job is not None else frozenset())
            elif p.job.done or self._clock() >= p.deadline:  # skipped as stale, or out of time
                self._count_handoff("timeouts")
                self._hand_off(frozenset())
            else:
                return

    def _rt_settled(self, p: _Handoff) -> bool:
        try:
            return bool(self._realtime.settled(p.rt))
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)
            return True  # handled as a failure: decoded locally

    def _cancel_job(self, job: Any) -> None:
        """Cancel ``job`` (its server items are deleted); a failing input never stops us."""
        try:
            self._realtime.cancel(job)
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)

    def _rt_timeout(self, p: _Handoff) -> None:
        """No transcript within ``transcript_timeout_s``: the job is cancelled (its server
        items are deleted) and the utterance is decoded locally."""
        self.events(
            "audio",
            "transcript_timeout",
            timeout_s=self._cfg.voice.realtime_route.transcript_timeout_s,
        )
        try:
            self._realtime.cancel(p.rt)
            self._realtime.route.count("transcript_timeouts")
        except Exception as exc:
            log.warning("voice listener: realtime input: %s", type(exc).__name__)

    def _settle(self, wait: bool) -> None:
        """Hand the waiting utterances off now, in order; with ``wait``, after waiting for a
        grammar pass up to its deadline (else a timeout), without, with its result only if it
        is in (else superseded). An utterance waiting for the realtime route's transcript is
        never waited for (that is the network): it and those behind it wait on, handed off by
        ``poll_handoff``."""
        while self._queue:
            p = self._queue[0]
            if p.rt is not None:
                self.poll_handoff()
                if self._queue and self._queue[0] is p:
                    return
                continue
            if wait and p.job is not None and not p.job.done:
                self._timed("grammar_wait", self._wait_for, p)
                if self._stop.is_set() and not p.job.done:
                    self._drop_pending()
                    return
            if p.job is not None and not p.job.done:
                self._count_handoff("timeouts" if wait else "superseded")
                self._hand_off(frozenset())
            else:
                self.poll_handoff()
                if self._queue and self._queue[0] is p:
                    return

    def _wait_for(self, p: _Handoff) -> None:
        """Wait for ``p``'s pass until its deadline (as wall time from now), or ``stop``."""
        assert p.job is not None
        end = self._timer() + max(0.0, p.deadline - self._clock())
        while not p.job.done and not self._stop.is_set():
            left = end - self._timer()
            if left <= 0:
                return
            p.job.wait(min(POLL_S, left))

    def _drop_pending(self) -> None:
        """The listener stops with utterances still waiting: drop them, counted (a realtime
        job is cancelled: its server items are deleted)."""
        if not self._queue:
            return
        while self._queue:
            p = self._queue.popleft()
            if p.rt is not None and self._realtime is not None:
                try:
                    self._realtime.cancel(p.rt)
                except Exception as exc:
                    log.warning("voice listener: realtime input: %s", type(exc).__name__)
                if not p.stream or p.cued:
                    self._audio_event(AUDIO_UNSENT, p)
            self._count_handoff("dropped_on_stop")
        self._set_active(False)

    def _count_handoff(self, key: str) -> None:
        with self._lock:
            self._handoff[key] += 1

    def _hand_off(self, heard: frozenset[str]) -> None:
        if not self._queue:
            return
        p = self._queue.popleft()
        if p.rt is None:
            waited = self._clock() - p.ended
            with self._lock:
                self._handoff["waits"] += 1
                self._wait_last_s = waited
                self._wait_max_s = max(self._wait_max_s, waited)
        try:
            again = self._timed("handoff", self._deliver, p, heard)
            if again is not None:  # its audio went: it waits for the server's final first
                self._queue.appendleft(again)
        finally:
            self._set_active(self._recording or bool(self._queue))

    def _deliver(self, p: _Handoff, heard: frozenset[str]) -> _Handoff | None:
        """Decide and hand ``p`` off; returns it instead when its audio just went to the
        realtime route (its final then comes from the server)."""
        ev = p.ev
        if p.rt is not None:
            return self._deliver_heard(p)
        text = p.text
        if p.addressed is None:
            # over the audible radio an extra name ("robot", a lyric word) does not wake the
            # robot by itself: its own name or a wake phrase does (owner decision 2026-11-18)
            media = p.media = self._media_at(ev.start_stamp)
            kind = self._fast.wake_kind(
                text, grammar_pass=p.grammar_pass, heard=heard, state=p.state, extras=not media
            )
            face = p.face
            at_face = self._checked(p) if face is None else (lambda: face)
            wake = kind == WAKE  # also True below for NAME_FIRST over the radio (SPEC 8)
            if kind == NAME_FIRST:
                # while the robot's own radio is audible the wake word is required anyway and
                # the sound direction is muted (own sound): the name opening the text wakes it
                # without the voice-at-face check (controller ruling, SPEC 8 Radio playback)
                if media:
                    wake = True
                else:
                    wake, at_face = self._name_first_at_face(at_face)
            outcome = self._gate.outcome(
                wake_word=wake,
                enrolled_in_view=self._enrolled(ev.start_stamp),
                start=ev.start_stamp,
                last_speech=ev.last_speech_stamp,
                speaker=self._speaker(),
                at_face=at_face,
            )
            p.addressed = not outcome.startswith("rejected")
            self._decided(p, outcome, text)
            if p.addressed and media:  # its turn holds the radio from its last speech
                self._hold_media(ev.last_speech_stamp)
            p.media_wake = media and outcome == "wake_word"
            self._note_wake(p, outcome)
            self._face_unsent_cue(p)
            if p.addressed and not p.no_audio:
                self._connect_on_speech()  # a session closed since its start (an idle close)
            if p.addressed and not p.no_audio and self._wants_audio():
                p.heard = heard
                self._send(p, wake_word=wake)
                return p
        if not text:  # decoded to nothing after the decision
            self._count("empty")
            return None
        self._finish(p, text, p.grammar_pass, heard, local_only=p.local_only)
        return None

    def _note_wake(self, p: _Handoff, outcome: str) -> None:
        """Keep the wake decision's media context: a wake opens a new window (the pending media
        wake is ``p`` when it was one over the audible radio, else none), and any utterance
        addressed without the wake word while a media wake's server verdict is still due (a
        window follow-up, or ``not_required`` under policy ``never``) remembers that wake."""
        if outcome in WAKES:
            self._media_wake = p if p.media_wake else None
        elif not outcome.startswith("rejected") and self._media_wake is not None:
            # a window follow-up, or (policy never) one the gate addressed without the wake
            # word: the media wake's hold made the radio inaudible at its start either way
            p.media_parent = self._media_wake

    def _confirms(self, fast: Any, text: str, p: _Handoff, server: bool) -> bool:
        """Whether ``p``'s final confirms a wake over the radio: a wake phrase or the robot's own
        name in it, or, in a server's transcript, a close spelling of that name (owner decision
        2026-11-17: the server wrote "Nivea" for "Neve"). An extra name ("robot") alone never
        confirms it (owner decision 2026-11-18: a lyric can say "robot")."""
        if fast.keywords & WAKE_CONFIRMS:
            return True
        return server and self._fast.server_name_close(text, p.language, extras=False)

    def _stops_media(self, fast: Any, text: str, p: _Handoff, server: bool) -> bool:
        """Whether ``p``'s unconfirmed server final asks to stop the radio (owner decision
        2026-11-19): ``tools.radio.stop_without_name`` on, a station set (held or not: the
        turn holds it), a ``stop`` phrase (the server spotter) or a ``media_stop`` verb, and a
        ``media`` word in the text, whole words all, of ``p``'s language only (the owner does
        not want languages mixed)."""
        if not (server and self._cfg.tools.radio.stop_without_name and self._media_set()):
            return False
        words = self._cfg.voice.keywords.get(p.language)
        if words is None or not holds_any(text, words.media):
            return False
        return "stop" in fast.keywords or holds_any(text, words.media_stop)

    def _stop_media(self) -> None:
        """Ask the media player to stop the station for ``voice_stop`` (``media_stop``)."""
        try:
            self.media_stop(VOICE_STOP)
        except Exception as exc:
            log.warning("voice listener: media player: %s", type(exc).__name__)

    def _heard(self, text: str, server: bool) -> str | None:
        """The server's words of an unconfirmed wake for the Live tab (SPEC 12): only a server
        final, only while the Live tab and its transcript are on, cut as a transcript turn;
        None otherwise. They go to the in-memory event ring only, never to a log."""
        return self._live_words(text) if server else None

    def _live_words(self, text: str) -> str | None:
        """``text`` for a decision event's ``heard`` (SPEC 12): only while the Live tab and its
        transcript are on, stripped and cut to ``webui.live.transcript_max_chars``; None
        otherwise. The caller picks which recognizer's words may go (``_heard``, ``_decided``)."""
        live = self._cfg.webui.live
        if not (live.enabled and live.transcript):
            return None
        return text.strip()[: live.transcript_max_chars]

    def _unconfirmed(self, p: _Handoff, server: bool) -> bool:
        """Whether ``p``'s wake over the radio is not confirmed (its final holds neither the
        name nor a wake phrase, checked by the caller): its own server final, or the media wake
        whose window it followed in was not confirmed (hand-offs keep their order, so that
        verdict is in)."""
        if server and p.media_wake:
            return True
        parent = p.media_parent
        return parent is not None and parent.unconfirmed

    def _name_first_at_face(self, at_face: AtFace | None) -> tuple[bool, AtFace | None]:
        """The name opened a long final (``NAME_FIRST``): it wakes the robot only when the
        voice came from a visible face, whether or not that face looks at the robot (SPEC 8:
        a television saying "Neve fino a 1000 metri" must not wake it). Runs the voice-at-face
        check once; returns the wake and the check with its verdict kept for the gate."""
        if at_face is None:
            return True, None
        verdict = at_face()
        if verdict == NO_FACE_AT_VOICE:
            self._count("name_first_no_face")
            return False, lambda: verdict
        return True, lambda: verdict

    def _deliver_heard(self, p: _Handoff) -> _Handoff | None:
        """``p``'s audio went to the realtime route: its final is the server's transcript
        (keywords, the wake word and the speaking rate come from it), else a local decode of
        its buffer: handed off ``local_only`` when the audio did leave (not answered, the
        did-not-get-it cue shows now), for the chat routes when it did not."""
        job = p.rt
        result = job.result
        if result is not None:
            self._count_audio("transcribed")
            self._realtime.route.note_transcript(
                (self._clock() - p.ev.last_speech_stamp) * MS_PER_S,
                turns="server" if p.stream else "robot",  # a stream's, or a burst's
            )
            heard = (result.generation, result.items)
            stream = job if p.stream else None
            self._finish(p, result.text, None, frozenset(), heard, server=True, stream=stream)
            return None
        if p.face_wake and not job.sent:
            # refused before it left (the session it waited for never came, the budget,
            # asleep): no audio path after all, so its buffer is decoded for the wake word, as
            # a face wake whose audio could not go at its end (``_face_media_end``)
            return self._face_wake_unsent(p)
        if p.face_wake:
            # no server final for a face wake over the radio: no local words exist to decide
            # it (Vosk never ran), so it is dropped, never decoded nor answered; its items go
            self._count_audio("face_wake_unheard")
            p.addressed, p.unconfirmed = False, True
            if self._media_wake is p:
                self._gate.close_window()
                self._media_wake = None
            self._audio_event(AUDIO_UNSENT, p)  # its listening cue ends silently
            self._cancel_job(job)
            return None
        if p.stream and job.answered:
            # the server answered it on its own, and its transcript never came: the answer
            # played, so neither a local decode nor the did-not-get-it cue (its history keeps
            # no words of the person); its listening cue ends here only if it began (a
            # barge-in's waits for the robot to stop, S3, and its answer may come first)
            self._count_audio("answered_untranscribed")
            if p.cued:
                self._audio_event(AUDIO_ANSWERED, p)
            return None
        if p.stream and job.playback and not job.sent:
            # a barge-in's segment whose stream never opened: it was heard over the robot's
            # (ducked) voice, so it is never decoded or given to the wake gate (S2)
            self._count_audio("playback_unsent")
            return None
        if p.stream and job.playback and self._drop_unconfirmed(job):
            # a barge-in's segment that went but the server never confirmed: heard over the
            # robot's voice, so never decoded nor answered (the echo risk); the did-not-get-it
            # cue shows once the robot's speech stopped (S3, owner decision 2026-11-08)
            self._count_audio("playback_unconfirmed")
            self._unheard.append(p.ev.start_stamp)
            self._cue_unheard()
            return None
        self._count_audio("local_fallbacks")
        self._announce(p)
        p.rt, p.local_only = None, job.sent
        if p.stream and not job.sent:
            # its stream never opened (a turn in flight, no session, a revoke): nothing left,
            # so nothing was decided either; the wake gate decides, as with robot turns
            p.addressed = None
        if not p.stream or p.cued:
            if job.sent:
                self._not_heard(p, "no_transcript")
            else:
                self._audio_event(AUDIO_UNSENT, p)
        if not p.text:  # Vosk never ran: decoded off this thread, then handed off
            self._decode(p)
            return p
        self._finish(p, p.text, p.grammar_pass, p.heard, local_only=p.local_only)
        return None

    def _face_wake_unsent(self, p: _Handoff) -> _Handoff:
        """A face wake over the radio whose audio was refused before it left (SPEC 8, owner
        decision 2026-11-20: the session it waited for never came, the budget, asleep): no
        utterance is lost, so its buffer is decoded for the wake word and the wake gate decides
        again on those words (``_deliver``: over the radio only the name addresses it), and its
        audio is never offered again (``no_audio``). The window its face wake opened closes.
        Its listening cue ends at that decision (``_face_unsent_cue``): silently when the name
        addressed it, else with the did-not-get-it cue (the person looked at the robot)."""
        self._count_audio("face_wake_unsent")
        p.rt, p.addressed, p.no_audio = None, None, True
        p.face_wake = p.media_wake = False
        if self._media_wake is p:
            self._gate.close_window()
            self._media_wake = None
        self._decode(p, wake=True)
        return p

    def _face_unsent_cue(self, p: _Handoff) -> None:
        """End the listening cue of a decoded face wake (``_face_wake_unsent``) once decided."""
        if p.no_audio:
            self._audio_event(AUDIO_UNSENT if p.addressed else AUDIO_NOT_HEARD, p)

    def _finish(
        self,
        p: _Handoff,
        text: str,
        grammar_pass: GrammarPass | None,
        heard: frozenset[str],
        realtime: tuple[int, tuple[str, ...]] | None = None,
        *,
        local_only: bool = False,
        server: bool = False,
        stream: Any = None,
    ) -> None:
        """The utterance with its final ``text`` (its wake decision made): keywords, prosody
        and impulses, then the hand-off to the conversation and the bus."""
        ev = p.ev
        rate = self._cfg.voice.sample_rate_hz
        first = max(0, round((ev.start_stamp - ev.audio_stamp) * rate))
        end = self._last_sample(ev) if server or p.decoded_end is None else p.decoded_end
        fast = self._fast.end(
            text,
            ev.audio[first:end],
            rate,
            last_speech_stamp=ev.last_speech_stamp,
            endpoint_stamp=ev.endpoint_stamp,
            grammar_pass=grammar_pass,
            heard=heard,
            state=p.state,
            server=server,
            extras=not p.media,
        )
        addressed = bool(p.addressed)
        if server and fast.wake_word:
            self._count_audio("wake_word_heard")
        if addressed and self._unconfirmed(p, server) and not self._confirms(fast, text, p, server):
            # the local wake over the radio was a lyric the recognizer took for the name: the
            # server's words hold no name and no wake phrase, so it is not answered (SPEC 8)
            addressed = p.addressed = False
            p.unconfirmed = True
            if self._stops_media(fast, text, p, server):
                # stop trusts the command (owner decision 2026-11-19): only the radio stops
                self._stop_media()
                self._count_audio("voice_stop_media")
                self.events("media", VOICE_STOP)
            else:
                self._count_audio("wake_unconfirmed_media")
                # the Live tab: the server did not hear the name, and what it heard instead
                self.events("unanswered", "media_unconfirmed", heard=self._heard(text, server))
            if self._media_wake is p:  # no wake since: the window it opened was a lyric's
                self._gate.close_window()
        if self._media_wake is p:  # its verdict is in
            self._media_wake = None
        self._fast.impulses(fast, addressed, state=p.state)
        utterance = Utterance(
            text,
            p.language,
            ev.start_stamp,
            ev.last_speech_stamp,
            ev.endpoint_stamp,
            addressed,
            fast.wake_word,
            fast.keywords,
            fast.prosody,
            realtime=realtime,
            local_only=local_only,
            stream=stream,
            at_face=addressed and AT_FACE in (p.face, p.face_seen),
        )
        self._fast.stats.record("final_vs_last_speech", self._clock() - ev.last_speech_stamp)
        self._count("utterances")
        self._publish(utterance, len(text), len(text.split()))
        try:
            self._on_utterance(utterance)
        except Exception as exc:
            log.warning("voice listener: utterance consumer raised %s", type(exc).__name__)

    def _sample_direction(self, stamp: float) -> None:
        """One voice-at-face sample at ``stamp`` (skipped while the check is off: it scans the
        world model). Muted while the robot speaks: its sound direction is off then (barge-in).
        The listening blip never mutes a sample: it is short, plays only when nobody speaks, and
        its own window is not speech."""
        d = self._direction
        if d is not None and d.enabled:
            d.sample(stamp, muted=self._speech.speaking(stamp))

    def _at_face(self) -> AtFace | None:
        """The voice-at-face check for the wake gate, None when there is none or it is off.
        Each check keeps its numbers in ``_face_info`` (listener thread only)."""
        d = self._direction
        if d is None or not d.enabled:
            return None

        def check() -> str:
            verdict = d.check()
            self._face_info = dict(d.last)
            return verdict

        return check

    def _checked(self, p: _Handoff) -> AtFace | None:
        """The voice-at-face check for ``p``'s decision: its numbers land in ``p.face_info``."""
        check = self._at_face()
        if check is None:
            return None

        def run() -> str:
            verdict = check()
            p.face_info, p.face_seen = self._face_info, verdict
            return verdict

        return run

    def _speaker(self) -> str | None:
        try:
            return self._speaker_id()
        except Exception:
            return None

    def _enrolled(self, stamp: float) -> bool:
        try:
            return self._enrolled_in_view(stamp)
        except Exception:
            return False

    def _publish(self, u: Utterance, chars: int, words: int) -> None:
        if self._bus is None:
            return
        info = UtteranceInfo(
            u.language,
            chars,
            words,
            u.start_stamp,
            u.last_speech_stamp,
            u.endpoint_stamp,
            u.addressed,
            u.wake_word,
            u.prosody.rms_db,
            u.prosody.pitch_hz,
            u.prosody.words_per_s,
        )
        try:
            self._bus.publish(TOPIC, info, u.endpoint_stamp)
        except Exception as exc:
            log.warning(
                "voice listener: percept.utterance subscriber raised %s", type(exc).__name__
            )
