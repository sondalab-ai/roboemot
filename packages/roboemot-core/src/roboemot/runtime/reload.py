"""Hot reload fan-out (SPEC 11A): hands a changed config to every part that holds a copy."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from roboemot.attention import AttentionSelector, GazeController
from roboemot.world import WorldModel

from .assembly import MS_PER_S, Pipeline
from .host import RuntimeHost


def apply_attention_config(
    cfg: Any, world: WorldModel, selector: AttentionSelector, gaze: GazeController
) -> None:
    """Hot reload (SPEC 11A): hand the new world and attention config to their consumers."""
    world.cfg = cfg.world
    selector.cfg = gaze.cfg = cfg.attention
    gaze.margin_rad = math.radians(cfg.safety.constraint_margin_deg)


def make_reload(host: RuntimeHost, p: Pipeline) -> Callable[[Any], None]:
    """The ``ConfigStack`` subscriber that pushes each effective-config change into the parts."""

    def reload(change: Any) -> None:
        p.hub.sync(change.config)  # the web UI check and the speech backend see it at once
        p.loop.apply_config(change.config)
        p.presence.configure(change.config.safety)
        apply_attention_config(change.config, p.world, p.selector, p.gaze)
        if p.scan is not None:
            p.scan.apply_config(change.config.world.scan)
            p.plugin.planner.apply_config(change.config)
        p.perception.apply_config(change.config)
        if host.autoenroll is not None:
            host.autoenroll.apply_config(change.config.world.autoenroll)
        if host.catalogue is not None:
            host.catalogue.cfg = change.config.catalogue
        p.runner.apply_config(change.config.affect, change.config.appraisal)
        p.circ.apply_config(change.config)
        p.speech.set_echo_tail_ms(change.config.voice.echo_tail_ms)
        p.yes_no.apply_config(change.config.voice.fastpath)
        if p.listener is not None:
            p.listener.apply_config(change.config)
        if p.bargein is not None:
            p.bargein.apply_config(change.config)
        if host.speech_output is not None:
            host.speech_output.apply_config(change.config)
        if host.conversation is not None:
            host.conversation.apply_config(change.config)
        if host.transcript is not None:
            host.transcript.apply_config(change.config.webui.live)
        p.reflexes.apply_config(
            change.config.reflexes, math.radians(change.config.safety.constraint_margin_deg)
        )
        p.tracker.cfg = change.config.expression
        p.expression.apply_config(
            change.config.expression,
            change.config.expression_map,
            change.config.motion.arbiter.crossfade_ms / MS_PER_S,
            change.config.gesture_map,
        )
        p.responder.cfg = change.config.connectivity
        p.responder.language = change.config.conversation.language
        p.network.set_policy(change.config.connectivity.network)
        p.network.set_routes(change.config.llm_routes)  # new routes are probed, removed dropped

    return reload
