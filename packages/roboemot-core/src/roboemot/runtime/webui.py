"""The runtime's web UI server (SPEC 4A, 6, 11A, 12)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from roboemot import paths
from roboemot.config import ConfigStack
from roboemot.realtime.check import offline_tools
from roboemot.secret_store import SecretStore
from roboemot.webui import Server, make_server
from roboemot.webui.routes import RouteChecker
from roboemot.webui.secrets_admin import SecretsAdmin
from roboemot.webui.settings import SettingsEditor

from .emotions import EmotionsState
from .host import RuntimeHost
from .live import LiveState

log = logging.getLogger(__name__)


@dataclass
class WebAdmin:
    """The web UI's settings editor and secrets admin, built before the pipeline."""

    secrets: SecretsAdmin
    settings: SettingsEditor


def restore_settings(
    stack: ConfigStack, manifest: Any, key_hosts: Callable[[], Iterable[str]]
) -> WebAdmin | None:
    """Build the web UI's settings editor and replay the edits saved from the page into
    ``stack`` (SPEC 11A). The runtime calls it before it builds the pipeline, so an edit that
    takes effect at a restart is in the config the parts are built from. ``key_hosts`` gives the
    realtime provider's key hosts; it is read only by page requests, after the pipeline exists.
    A failure is logged by type and the runtime starts without the edits."""
    try:
        data_dir = stack.effective().data_dir
        secrets = SecretsAdmin(stack.effective, store=SecretStore.for_data_dir(data_dir))
        settings = SettingsEditor(
            stack,
            store=paths.settings_overrides_path(data_dir),
            manifest=manifest,
            known_secrets=secrets.names,
            realtime_key_hosts=key_hosts,
        )
        settings.restore()
    except Exception as exc:
        log.error("web UI settings not restored: %s", type(exc).__name__)
        return None
    return WebAdmin(secrets, settings)


def check_tools(stack: ConfigStack, p: Any) -> Callable[[], list[dict[str, Any]]]:
    """The realtime check's setup tools on the robot (SPEC 8, Realtime check): the offline
    tools under the committed config with the robot's playable gestures (motion names, never a
    name in view, SPEC 12)."""

    def tools() -> list[dict[str, Any]]:
        return offline_tools(stack.effective(), p.expression.playable)

    return tools


def route_status(host: RuntimeHost) -> Callable[[], dict[str, Any] | None]:
    """The running realtime route's debug snapshot for the Routes tab (SPEC 12A), or None
    when no realtime route was built."""

    def status() -> dict[str, Any] | None:
        route = host.realtime_route
        return route.snapshot() if route is not None else None

    return status


def realtime_key_hosts(p: Any) -> list[str]:
    """The key hosts of the pipeline's realtime session provider (none without a provider)."""
    return list(provider.key_hosts) if (provider := p.realtime_provider) is not None else []


def start_webui(
    host: RuntimeHost,
    stack: ConfigStack,
    cfg: Any,
    manifest: Any,
    cleanup: ExitStack,
    p: Any,
    admin: WebAdmin | None = None,
) -> Server | None:
    """Serve the web UI over the runtime's enrollment service and diagnostics registry.

    The server is stopped by ``cleanup`` (registered before it starts). A port that is
    taken or an unusable data folder is logged (type only) and the runtime goes on without the
    page: enrollment is a side channel, not a reason to stop the robot. ``p`` is the pipeline the
    Live tab reads its inner state from (SPEC 11A). ``admin`` is the editor the runtime restored
    before it built ``p``; without one it is built and restored here.
    """
    if admin is None:
        admin = restore_settings(stack, manifest, lambda: realtime_key_hosts(p))
        if admin is None:
            return None
    secrets, settings = admin.secrets, admin.settings
    undo: list[Callable[[], None]] = []  # run once the page stops, or at once if it never starts

    def stop() -> None:
        while undo:
            undo.pop()()

    cleanup.callback(stop)
    try:
        live = LiveState(
            host,
            p,
            stack.effective(),
            clock=p.loop.clock,
            audio_egress=lambda: (r := host.realtime_route) is not None and r.audio_egress,
        )
        undo.append(stack.subscribe(live.apply_config))
        live.apply_config(
            SimpleNamespace(config=stack.effective())
        )  # catches a change since the read
        if host.bus is not None:  # the preview never outlives the robot's waking
            host.bus.subscribe("robot.asleep", lambda msg: live.video.sleep())
        cleanup.callback(live.video.drop)
        host.live_video = live.video
        emotions = EmotionsState(host, p, stack.effective(), clock=p.loop.clock)
        undo.append(emotions.close)  # the engine stops keeping the tab's history
        undo.append(stack.subscribe(emotions.apply_config))
        emotions.apply_config(SimpleNamespace(config=stack.effective()))
        server = make_server(
            host.perception.enrollment,
            host.diagnostics,
            stack.effective,
            paths.data_dir(cfg.data_dir),
            robot_hosted=manifest.robot.profile is not None,
            catalogue=host.catalogue,
            settings=settings,
            pseudonyms=host.name_source,
            secrets=secrets,
            routes=RouteChecker(
                stack.effective,
                paths.route_checks_path(cfg.data_dir),
                router=host.route_check_router,
                names=host.name_source,
                realtime_provider=lambda: p.realtime_provider,  # the runtime's own (SPEC 8)
                realtime_tools=check_tools(stack, p),
                realtime_status=route_status(host),
            ),
            live=live,
            emotions=emotions,
        )
    except Exception as exc:
        stop()
        log.error("web UI not started: %s", type(exc).__name__)
        return None
    cleanup.callback(server.stop)
    server.start()
    host.webui = server
    log.info("web UI on %s:%d", server.host, server.port)
    return server
