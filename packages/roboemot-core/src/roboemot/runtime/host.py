"""What the owner of a runtime can override and observe (SPEC 4A, 7A, 8).

``RuntimeHost`` carries the seams the runtime calls (the LLM router, the utterance sink, the
circadian clock, the frame source, the vision backends) and the parts it publishes while it runs
(the bus, the world model, the diagnostics registry, the voice and conversation workers). The
robot app inherits from it, so a test overrides a seam by subclassing the app and reads the
running parts off the app. A bare ``RuntimeHost`` is the default owner. One ``Runtime.run`` per
host: the parts are not reset between runs.
"""

from __future__ import annotations

import math
from typing import Any

from roboemot.affect.expression import ExpressionRenderer
from roboemot.affect.register import RegisterTracker
from roboemot.affect.runner import AffectRunner
from roboemot.affect.speaking import SpeakingRenderer
from roboemot.attention import AttentionSelector
from roboemot.bus import Bus
from roboemot.circadian import CircadianModel, CircadianRunner, IdleLightReader, SystemClock
from roboemot.conversation import ConversationPhase, ConversationWorker, TranscriptRing
from roboemot.diagnostics import Diagnostics
from roboemot.health.network import NetworkMonitor
from roboemot.llm import Router
from roboemot.llm.hub import RouterHub
from roboemot.llm.pseudonyms import NameSource
from roboemot.persona.llm_traits import TraitsWorker
from roboemot.realtime.route import RealtimeRoute
from roboemot.reflex import ReflexEngine
from roboemot.robot_platform import AudioHub, RobotSession, Speaker
from roboemot.runtime.live_video import LiveVideo
from roboemot.secret_store import SecretStore
from roboemot.soothe import SootheRunner
from roboemot.vision import FrameSource
from roboemot.vision.service import Perception, default_stages
from roboemot.voice.activity import SpeechActivity
from roboemot.voice.bargein import BargeIn
from roboemot.voice.listener import Utterance, VoiceListener
from roboemot.voice.robot_name import RobotName
from roboemot.voice.speech import SpeechOutput
from roboemot.voice.yes_no import YesNoGestures
from roboemot.webui import Server
from roboemot.world import WorldModel
from roboemot.world.autoenroll import AutoEnroller
from roboemot.world.sources import JointStateFn

from .levels import AudioLevels


class RuntimeHost:
    """Overridable seams of a runtime and the parts it exposes while it runs (see the module
    docstring). Attributes are None until the runtime builds them."""

    webui: Server | None = None
    diagnostics: Diagnostics | None = None
    bus: Bus | None = None
    world: WorldModel | None = None  # vision (M1) and the speaker hint (M3) feed these
    selector: AttentionSelector | None = None
    circadian: CircadianModel | None = None  # the circadian runner's model; None while disabled
    circadian_runner: CircadianRunner | None = None  # light, presence, state, motion (SPEC 7A)
    light_reader: IdleLightReader | None = None  # reads a frame when vision gave none
    affect_runner: AffectRunner | None = None  # owns the one affect engine
    register_tracker: RegisterTracker | None = None  # people in view -> register.active
    yes_no: YesNoGestures | None = None  # spoken yes / no -> nod / shake
    expression: ExpressionRenderer | None = None  # affect label -> additive motion
    speaking: SpeakingRenderer | None = None  # speech timeline -> additive motion (SPEC 7)
    reflexes: ReflexEngine | None = None  # runs on the motion loop's tick (SPEC 10)
    audio: AudioHub | None = None  # the one reader of the microphone (reflex level, voice)
    speaker: Speaker | None = None  # playback; only when voice.enabled
    speech: SpeechActivity | None = None  # "the robot is speaking" gate (reflex, DoA)
    bargein: BargeIn | None = None  # user speech over the robot's voice (SPEC 8 stage 8)
    voice_listener: VoiceListener | None = None  # mic -> VAD -> ASR -> utterances; voice.enabled
    speech_output: SpeechOutput | None = None  # TTS queue onto the speaker; only when voice.enabled
    soothe_runner: SootheRunner | None = None  # idle self-soothing sound (SPEC 7B); needs voice
    traits_worker: TraitsWorker | None = None  # rates persona traits when traits_source is llm
    name_source: NameSource | None = None  # registry names for pseudonyms outside the conversation
    router_hub: RouterHub | None = None  # the one router: traits, conversation, speech, checks
    conversation: ConversationWorker | None = None  # answers addressed utterances (voice.enabled)
    realtime_route: RealtimeRoute | None = None  # the realtime speech route, when enabled
    phase: ConversationPhase | None = None  # idle / listening / thinking / speaking (Live tab)
    live_video: LiveVideo | None = None  # the Live tab's preview (set when the web UI starts)
    transcript: TranscriptRing | None = None  # opt-in in-memory turns (Live tab, SPEC 12)
    levels: AudioLevels | None = None  # microphone and output levels in dBFS (Live tab)
    names: RobotName | None = None  # the robot's name for the wake word and addressing
    perception: Perception | None = None
    enrollment: Any = None  # EnrollmentService: the web UI enrolls through it (SPEC 6)
    autoenroll: AutoEnroller | None = None  # animals and objects that keep coming back (SPEC 5A)
    catalogue: Any = None  # CatalogueService: XMP import, confirm "this is Mia" (SPEC 6)

    def make_router(self, cfg: Any, monitor: NetworkMonitor) -> Router:
        """The LLM router over the configured routes (overridable: tests inject fakes)."""
        return Router(
            cfg.llm_routes,
            cfg.safety,
            secrets=SecretStore.for_data_dir(cfg.data_dir).get,
            monitor=monitor,
        )

    def on_utterance(self, utterance: Utterance) -> None:
        """Where finished utterances go (SPEC 8): the conversation worker, which answers the
        addressed ones; overridable, tests inject a collector."""
        if self.conversation is not None:
            self.conversation.submit(utterance)

    def make_circadian_clock(self) -> Any:
        """The clock the circadian rhythm runs on: monotonic seconds and the local hour of day
        (overridable: tests inject a fixed or accelerated clock)."""
        return SystemClock()

    def make_frame_source(self, session: RobotSession, joint_state: JointStateFn) -> FrameSource:
        """The camera the vision pump reads (overridable: tests inject a planted scene)."""
        return session.camera(joint_state)

    def load_vision_stages(self, cfg: Any, profile: str | None) -> Any:
        """The vision backends (overridable: tests inject fakes)."""
        return default_stages(cfg, profile)

    def route_check_router(self) -> Router | None:
        """The router a web UI route check is charged to: the shared one (traits, conversation,
        speech), None before it exists, which gives the checker a private one."""
        return self.router_hub.router if self.router_hub else None

    def forget_circadian(self) -> None:
        """The runtime stopped: hold and wake report that it is not running."""
        self.circadian_runner = None

    def hold(self, state: str, minutes: float) -> None:
        """``robot.circadian.hold``: keep the robot ``awake`` (or ``drowsy`` or ``resting``) for
        ``minutes``, e.g. during a party (SPEC 7A, API)."""
        runner = self.circadian_runner
        if runner is None:
            raise RuntimeError("the runtime is not running")
        runner.hold(state, minutes)

    def wake(self) -> None:
        """Wake the robot as the wake word or a direct address does (SPEC 7A, Waking)."""
        runner = self.circadian_runner
        if runner is None:
            raise RuntimeError("the runtime is not running")
        runner.wake()

    def detection_period(self) -> float:
        """Seconds between face-stage runs now (infinite while vision is off)."""
        pipeline = self.perception.pipeline if self.perception is not None else None
        return math.inf if pipeline is None else pipeline.governor.period("detection")
