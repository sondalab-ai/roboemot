"""HTTP transport of the web UI: static files, size limits, security headers (SPEC 12).

Standard library only. Static files come from a fixed allow-list read at start-up, so no
request path ever reaches the file system.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from roboemot.config.schema import RoboEmotConfig

from .api import DebugReport, EnrollmentPort, Request, Response, WebApi
from .auth import PinStore, check_pin_format

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/static/style.css": ("style.css", "text/css; charset=utf-8"),
}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}
SERVER_NAME = "roboemot"


class _Httpd(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    api: WebApi
    statics: dict[str, tuple[bytes, str]]
    max_body: int
    request_timeout_s: float

    def handle_error(self, request: Any, client_address: Any) -> None:
        log.error("web ui connection failed")


class _Handler(BaseHTTPRequestHandler):
    server: _Httpd
    server_version = SERVER_NAME
    sys_version = ""

    def setup(self) -> None:
        self.timeout = self.server.request_timeout_s
        super().setup()

    def log_message(self, format: str, *args: Any) -> None:
        """Silence the default access log: request lines can carry data."""

    def do_GET(self) -> None:
        self._serve()

    def do_POST(self) -> None:
        self._serve()

    def do_PUT(self) -> None:
        self._serve()

    def do_DELETE(self) -> None:
        self._serve()

    def do_PATCH(self) -> None:
        self._serve()

    def do_HEAD(self) -> None:
        self._serve()

    def do_OPTIONS(self) -> None:
        self._serve()

    def _serve(self) -> None:
        path = urlsplit(self.path).path
        if self.command == "GET" and path in self.server.statics:
            body, ctype = self.server.statics[path]
            self._send(HTTPStatus.OK, body, ctype)
            return
        if self.command == "GET" and not path.startswith("/api/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        body = self._read_body()
        if body is None:
            return
        headers = {k.lower(): v for k, v in self.headers.items()}
        resp = self.server.api.handle(
            Request(self.command, path, headers, body, self.client_address[0])
        )
        self._send_response(resp)

    def _read_body(self) -> bytes | None:
        """The body, or None after an error response; the size is checked before reading."""
        if "Transfer-Encoding" in self.headers:
            self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "bad_request"})
            return None
        raw = self.headers.get("Content-Length", "0")
        if not raw.isascii() or not raw.isdigit():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_request"})
            return None
        length = int(raw)
        if length > self.server.max_body:
            self.close_connection = True
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body_too_large"})
            return None
        return self.rfile.read(length) if length else b""

    def _send_response(self, resp: Response) -> None:
        try:
            payload = json.dumps(resp.data, allow_nan=False).encode()
        except (TypeError, ValueError):
            log.error("web ui response is not JSON-serialisable")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "generic"})
            return
        self._send(resp.status, payload, "application/json; charset=utf-8", resp)

    def _send_json(self, status: HTTPStatus, data: Any) -> None:
        self._send(status, json.dumps(data).encode(), "application/json; charset=utf-8")

    def _send(
        self, status: HTTPStatus, body: bytes, ctype: str, resp: Response | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        if resp is not None:
            for name, value in resp.headers.items():
                self.send_header(name, value)
            for cookie in resp.cookies:
                self.send_header("Set-Cookie", cookie)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


class Server:
    """The running web UI. ``start()`` serves on a background thread, ``stop()`` ends it."""

    def __init__(self, httpd: _Httpd, pins: PinStore, cfg: RoboEmotConfig) -> None:
        self._httpd = httpd
        self._pins = pins
        self._cfg = cfg
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return str(self._httpd.server_address[0])

    @property
    def port(self) -> int:
        """The bound port (the real one when ``webui.port`` is 0)."""
        return int(self._httpd.server_address[1])

    def set_pin(self, pin: str) -> None:
        """The admin path: store a new operator PIN after the format check (raises PinError)."""
        self._pins.set(check_pin_format(pin, self._cfg.webui))

    def start(self) -> Server:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever, name="roboemot-webui", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving, close the socket and join the thread; safe to call twice."""
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join()
            self._thread = None
        self._httpd.server_close()


def make_server(
    service: EnrollmentPort,
    diagnostics: DebugReport,
    cfg: RoboEmotConfig | Callable[[], RoboEmotConfig],
    data_dir: Path,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> Server:
    """Build the web UI on ``cfg.webui.bind:port``; nothing is served until ``start()``.

    ``cfg`` may be a callable returning the effective config, so the page follows
    ``conversation.language`` changes; bind, port and limits are read once here.
    """
    current = cfg() if callable(cfg) else cfg
    web = current.webui
    pins = PinStore(Path(data_dir))
    api = WebApi(service, diagnostics, cfg, pins, clock=clock)
    httpd = _Httpd((web.bind, web.port), _Handler)
    httpd.api = api
    httpd.max_body = web.max_body_bytes
    httpd.request_timeout_s = web.request_timeout_s
    httpd.statics = {
        route: ((STATIC_DIR / name).read_bytes(), ctype)
        for route, (name, ctype) in STATIC_FILES.items()
    }
    return Server(httpd, pins, current)
