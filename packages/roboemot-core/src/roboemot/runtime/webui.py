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
from roboemot.secret_store import SecretStore
from roboemot.webui import Server, make_server
from roboemot.webui.routes import RouteChecker
from roboemot.webui.secrets_admin import SecretsAdmin
from roboemot.webui.settings import SettingsEditor

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
    unsubscribe = None
    try:
        live = LiveState(
            host,
            p,
            stack.effective(),
            clock=p.loop.clock,
            audio_egress=lambda: (r := host.realtime_route) is not None and r.audio_egress,
        )
        unsubscribe = stack.subscribe(live.apply_config)
        live.apply_config(
            SimpleNamespace(config=stack.effective())
        )  # catches a change since the read
        if host.bus is not None:  # the preview never outlives the robot's waking
            host.bus.subscribe("robot.asleep", lambda msg: live.video.sleep())
        cleanup.callback(live.video.drop)
        host.live_video = live.video
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
            ),
            live=live,
        )
    except Exception as exc:
        if unsubscribe is not None:
            unsubscribe()
        log.error("web UI not started: %s", type(exc).__name__)
        return None
    cleanup.callback(server.stop)
    server.start()
    host.webui = server
    log.info("web UI on %s:%d", server.host, server.port)
    return server
