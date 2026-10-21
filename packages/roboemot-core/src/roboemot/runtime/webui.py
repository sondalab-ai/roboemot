"""The runtime's web UI server (SPEC 4A, 6, 11A, 12)."""

from __future__ import annotations

import logging
from contextlib import ExitStack
from typing import Any

from roboemot import paths
from roboemot.config import ConfigStack
from roboemot.webui import Server, make_server
from roboemot.webui.routes import RouteChecker
from roboemot.webui.secrets_admin import SecretsAdmin
from roboemot.webui.settings import SettingsEditor

from .host import RuntimeHost

log = logging.getLogger(__name__)


def start_webui(
    host: RuntimeHost, stack: ConfigStack, cfg: Any, manifest: Any, cleanup: ExitStack
) -> Server | None:
    """Serve the web UI over the runtime's enrollment service and diagnostics registry.

    The server is stopped by ``cleanup`` (registered before it starts). A port that is
    taken or an unusable data folder is logged (type only) and the runtime goes on without the
    page: enrollment is a side channel, not a reason to stop the robot.
    """
    try:
        settings = SettingsEditor(
            stack, store=paths.settings_overrides_path(cfg.data_dir), manifest=manifest
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
            secrets=SecretsAdmin(stack.effective),
            routes=RouteChecker(
                stack.effective,
                paths.route_checks_path(cfg.data_dir),
                router=host.route_check_router,
                names=host.name_source,
            ),
        )
    except Exception as exc:
        log.error("web UI not started: %s", type(exc).__name__)
        return None
    cleanup.callback(server.stop)
    server.start()
    host.webui = server
    log.info("web UI on %s:%d", server.host, server.port)
    return server
