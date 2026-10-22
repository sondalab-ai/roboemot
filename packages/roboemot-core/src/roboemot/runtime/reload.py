"""Hot reload fan-out (SPEC 11A): hands a changed config to every part that holds a copy."""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable
from typing import Any

from roboemot.attention import AttentionSelector, GazeController
from roboemot.world import WorldModel

from .assembly import MS_PER_S, Pipeline
from .host import RuntimeHost

log = logging.getLogger(__name__)


def apply_attention_config(
    cfg: Any, world: WorldModel, selector: AttentionSelector, gaze: GazeController
) -> None:
    """Hot reload (SPEC 11A): hand the new world and attention config to their consumers."""
    world.cfg = cfg.world
    selector.cfg = gaze.cfg = cfg.attention
    gaze.margin_rad = math.radians(cfg.safety.constraint_margin_deg)


def make_reload(host: RuntimeHost, p: Pipeline) -> Callable[[Any], None]:
    """The ``ConfigStack`` subscriber that pushes each effective-config change into the parts.

    Each part is its own step: one that raises does not skip the others, and the safety steps
    (motion limits, minor caps, reflex and gaze margins) run first. The first failure is raised
    again once every step ran, so the stack audits it and the web UI reports the change as not
    applied (SPEC 11A, 12)."""

    def steps(cfg: Any) -> list[tuple[str, Callable[[], None]]]:
        margin_rad = math.radians(cfg.safety.constraint_margin_deg)
        out: list[tuple[str, Callable[[], None]]] = [
            ("loop", lambda: p.loop.apply_config(cfg)),
            ("presence", lambda: p.presence.configure(cfg.safety)),
            ("reflexes", lambda: p.reflexes.apply_config(cfg.reflexes, margin_rad)),
            ("attention", lambda: apply_attention_config(cfg, p.world, p.selector, p.gaze)),
            ("hub", lambda: p.hub.sync(cfg)),  # the web UI check and the speech backend
        ]
        audio_reload = getattr(p.audio, "apply_config", None)  # optional for a robot package
        if audio_reload is not None:
            out.append(("audio", lambda: audio_reload(cfg)))
        if p.scan is not None:
            scan = p.scan
            out.append(("scan", lambda: scan.apply_config(cfg.world.scan)))
            out.append(("planner", lambda: p.plugin.planner.apply_config(cfg)))
        out += [
            ("perception", lambda: p.perception.apply_config(cfg)),
            ("affect", lambda: p.runner.apply_config(cfg.affect, cfg.appraisal)),
            ("circadian", lambda: p.circ.apply_config(cfg)),
            ("speech", lambda: p.speech.set_echo_tail_ms(cfg.voice.echo_tail_ms)),
            ("yes_no", lambda: p.yes_no.apply_config(cfg.voice.fastpath)),
        ]
        optional = [
            ("autoenroll", host.autoenroll, lambda part: part.apply_config(cfg.world.autoenroll)),
            ("listener", p.listener, lambda part: part.apply_config(cfg)),
            ("bargein", p.bargein, lambda part: part.apply_config(cfg)),
            ("speech_output", host.speech_output, lambda part: part.apply_config(cfg)),
            ("conversation", host.conversation, lambda part: part.apply_config(cfg)),
            ("transcript", host.transcript, lambda part: part.apply_config(cfg.webui.live)),
        ]
        out += [
            (name, functools.partial(apply, part))
            for name, part, apply in optional
            if part is not None
        ]
        if host.catalogue is not None:
            catalogue = host.catalogue
            out.append(("catalogue", lambda: setattr(catalogue, "cfg", cfg.catalogue)))
        out += [
            ("register", lambda: setattr(p.tracker, "cfg", cfg.expression)),
            (
                "expression",
                lambda: p.expression.apply_config(
                    cfg.expression,
                    cfg.expression_map,
                    cfg.motion.arbiter.crossfade_ms / MS_PER_S,
                    cfg.gesture_map,
                ),
            ),
            ("responder", lambda: _apply_responder(p, cfg)),
            ("network", lambda: _apply_network(p, cfg)),
        ]
        return out

    def reload(change: Any) -> None:
        failed: Exception | None = None
        for name, step in steps(change.config):
            try:
                step()
            except Exception as exc:
                log.warning("config reload of %s failed: %s", name, type(exc).__name__)
                failed = failed or exc
        if failed is not None:
            raise failed

    return reload


def _apply_responder(p: Pipeline, cfg: Any) -> None:
    p.responder.cfg = cfg.connectivity
    p.responder.language = cfg.conversation.language


def _apply_network(p: Pipeline, cfg: Any) -> None:
    p.network.set_policy(cfg.connectivity.network)
    p.network.set_routes(cfg.llm_routes)  # new routes are probed, removed dropped
