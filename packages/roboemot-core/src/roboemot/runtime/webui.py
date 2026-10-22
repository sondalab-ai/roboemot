"""The runtime's web UI server (SPEC 4A, 6, 11A, 12)."""

from __future__ import annotations

import logging
from contextlib import ExitStack
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


def start_webui(
    host: RuntimeHost,
    stack: ConfigStack,
    cfg: Any,
    manifest: Any,
    cleanup: ExitStack,
    p: Any,
) -> Server | None:
    """Serve the web UI over the runtime's enrollment service and diagnostics registry.

    The server is stopped by ``cleanup`` (registered before it starts). A port that is
    taken or an unusable data folder is logged (type only) and the runtime goes on without the
    page: enrollment is a side channel, not a reason to stop the robot. ``p`` is the pipeline the
    Live tab reads its inner state from (SPEC 11A).
    """
    unsubscribe = None
    try:
        live = LiveState(host, p, stack.effective(), clock=p.loop.clock)
        unsubscribe = stack.subscribe(live.apply_config)
        live.apply_config(
            SimpleNamespace(config=stack.effective())
        )  # catches a change since the read
        if host.bus is not None:  # the preview never outlives the robot's waking
            host.bus.subscribe("robot.asleep", lambda msg: live.video.drop())
        cleanup.callback(live.video.drop)
        secrets = SecretsAdmin(stack.effective, store=SecretStore.for_data_dir(cfg.data_dir))
        settings = SettingsEditor(
            stack,
            store=paths.settings_overrides_path(cfg.data_dir),
            manifest=manifest,
            known_secrets=secrets.names,
        )
        settings.restore()
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
