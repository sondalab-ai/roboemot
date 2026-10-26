"""HTTP transport of the web UI: static files, size limits, security headers (SPEC 12).

Standard library only. Static files come from a fixed allow-list read at start-up, so no
request path ever reaches the file system. The allow-list is the first-party files plus the
files listed in the vendored kit's ``VENDORED.json``; those are checked against their sha256 at
start-up and served with ``Cache-Control: no-cache`` and a strong ETag (304 on a match), while
every other response stays ``no-store``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import socket
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from roboemot.config.schema import RoboEmotConfig

from .. import shutdown
from .api import (
    CataloguePort,
    DebugReport,
    EnrollmentPort,
    LivePort,
    PseudonymsPort,
    Request,
    Response,
    RoutesPort,
    SecretsPort,
    SettingsPort,
    WebApi,
)
from .auth import PinStore, check_pin_format
from .guard import hostname_of, is_trusted_host

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/static/settings.js": ("settings.js", "text/javascript; charset=utf-8"),
    "/static/live.js": ("live.js", "text/javascript; charset=utf-8"),
    "/static/style.css": ("style.css", "text/css; charset=utf-8"),
    "/static/theme.js": ("theme.js", "text/javascript; charset=utf-8"),
    "/static/fx.js": ("fx.js", "text/javascript; charset=utf-8"),
}
VENDOR_ROUTE = "/static/vendor/sondalab-ui-kit/"
VENDOR_DIR = STATIC_DIR / "vendor" / "sondalab-ui-kit"
VENDOR_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}
VENDOR_CACHE_CONTROL = "no-cache"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors {ancestors}"
)
SERVER_NAME = "roboemot"
BUSY_DRAIN = 4096
SERVE_POLL_S = 0.5  # how often serve_forever checks for a stop; tests shorten it
ANY_INTERFACE = "0.0.0.0"
LOOPBACK_INTERFACE = "127.0.0.1"
BUSY_REPLY = b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nRetry-After: 1\r\n\r\n"


Static = tuple[bytes, str, str | None]  # body, content type, strong ETag (vendored files only)


def load_vendor() -> dict[str, Static]:
    """The vendored kit files named in ``VENDORED.json``, keyed by URL path.

    Each file must match the sha256 in the manifest (a changed or missing file stops the
    start-up); the hash is the ETag. Files not in the manifest are never served.
    """
    manifest = json.loads((VENDOR_DIR / "VENDORED.json").read_text())
    out: dict[str, Static] = {}
    root = VENDOR_DIR.resolve()
    for name, digest in manifest["files"].items():
        path = (VENDOR_DIR / name).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"vendored file {name} is outside the vendor folder")
        ctype = VENDOR_TYPES.get(path.suffix)
        if ctype is None:
            raise RuntimeError(f"vendored file {name} has no known content type")
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != digest:
            raise RuntimeError(f"vendored file {name} does not match VENDORED.json")
        out[VENDOR_ROUTE + name] = (body, ctype, f'"{digest}"')
    return out


def etag_matches(header: str | None, etag: str) -> bool:
    """Whether an ``If-None-Match`` value names ``etag`` (weak prefix ignored, ``*`` matches)."""
    if not header:
        return False
    tags = [t.strip() for t in header.split(",")]
    return "*" in tags or etag in [t.removeprefix("W/") for t in tags]


class _DeadlineRaw(io.RawIOBase):
    """Raw socket reader with one overall deadline (slowloris guard).

    Before every ``recv`` the socket timeout is set to what is left of the deadline (at most
    the per-read timeout), so a client dripping one byte at a time is cut off on schedule.
    """

    def __init__(
        self, sock: Any, deadline: float, per_read_s: float, clock: Callable[[], float]
    ) -> None:
        super().__init__()
        self._sock, self._deadline = sock, deadline
        self._per_read_s, self._clock = per_read_s, clock

    def arm(self) -> None:
        """Set the socket timeout to the time left; raises ``TimeoutError`` once it is gone."""
        remaining = self._deadline - self._clock()
        if remaining <= 0:
            raise TimeoutError("request deadline passed")
        self._sock.settimeout(min(self._per_read_s, remaining))

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        self.arm()
        return self._sock.recv_into(buffer)


class _Httpd(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    api: WebApi
    statics: dict[str, Static]
    slots: threading.BoundedSemaphore

    def process_request(self, request: Any, client_address: Any) -> None:
        """Serve on a new thread, or refuse with 503 when ``max_connections`` are in use."""
        if not self.slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(BUSY_REPLY)
                request.shutdown(socket.SHUT_WR)
                request.recv(BUSY_DRAIN)  # read what the client sent, so close does not reset
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        log.error("web ui connection failed")


class _Handler(BaseHTTPRequestHandler):
    server: _Httpd
    server_version = SERVER_NAME
    sys_version = ""

    def setup(self) -> None:
        web = self.server.api.web  # per connection: a runtime change applies to the next one
        self.timeout = web.request_timeout_s
        super().setup()
        self.deadline = _DeadlineRaw(
            self.connection,
            time.monotonic() + web.request_deadline_s,
            web.request_timeout_s,
            time.monotonic,
        )
        self.rfile = io.BufferedReader(self.deadline)

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

    def _host_ok(self) -> bool:
        """Exactly one Host naming a trusted host, at most one Origin (DNS-rebinding guard)."""
        hosts, origins = self.headers.get_all("Host", []), self.headers.get_all("Origin", [])
        if len(hosts) != 1 or len(origins) > 1:
            return False
        allowed = self.server.api.web.allowed_hosts
        return is_trusted_host(hostname_of(hosts[0]), allowed)

    def _serve(self) -> None:
        if not self._host_ok():
            self.close_connection = True
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_host"})
            return
        path = urlsplit(self.path).path
        if self.command in ("GET", "HEAD") and path in self.server.statics:
            body, ctype, etag = self.server.statics[path]
            if etag is not None and etag_matches(self.headers.get("If-None-Match"), etag):
                self._send(HTTPStatus.NOT_MODIFIED, b"", ctype, etag=etag)
            else:
                self._send(HTTPStatus.OK, body, ctype, etag=etag)
            return
        if self.command in ("GET", "HEAD") and not path.startswith("/api/"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        headers = {k.lower(): v for k, v in self.headers.items()}
        limit = self.server.api.body_limit(self.command, path, headers)
        body = self._read_body(limit)
        if body is None:
            return
        resp = self.server.api.handle(
            Request(
                self.command, path, headers, body, self.client_address[0], urlsplit(self.path).query
            )
        )
        sent = self._send_response(resp)
        if sent and resp.after is not None:
            self.wfile.flush()
            resp.after()

    def _read_body(self, limit: int) -> bytes | None:
        """The body, or None after an error response; the size is checked before reading."""
        if "Transfer-Encoding" in self.headers:
            self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "bad_request"})
            return None
        raw = self.headers.get("Content-Length", "0")
        if not raw.isascii() or not raw.isdigit():
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_request"})
            return None
        length = int(raw)
        if length > limit:
            self.close_connection = True
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body_too_large"})
            return None
        return self.rfile.read(length) if length else b""

    def _send_response(self, resp: Response) -> bool:
        """Send ``resp``; False when it was replaced by an error (it was not JSON-serialisable).
        A failed socket write raises."""
        if isinstance(resp.data, bytes):
            self._send(
                resp.status, resp.data, resp.content_type or "application/octet-stream", resp
            )
            return True
        try:
            payload = json.dumps(resp.data, allow_nan=False).encode()
        except (TypeError, ValueError):
            log.error("web ui response is not JSON-serialisable")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "generic"})
            return False
        self._send(resp.status, payload, "application/json; charset=utf-8", resp)
        return True

    def _send_json(self, status: HTTPStatus, data: Any) -> None:
        self._send(status, json.dumps(data).encode(), "application/json; charset=utf-8")

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        ctype: str,
        resp: Response | None = None,
        etag: str | None = None,
    ) -> None:
        self.deadline.arm()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if status != HTTPStatus.NOT_MODIFIED:
            self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Security-Policy", CSP.format(ancestors=self.server.api.web.frame_ancestors)
        )
        for name, value in SECURITY_HEADERS.items():
            if etag is not None and name == "Cache-Control":
                continue
            self.send_header(name, value)
        if etag is not None:
            self.send_header("Cache-Control", VENDOR_CACHE_CONTROL)
            self.send_header("ETag", etag)
        if resp is not None:
            for name, value in resp.headers.items():
                self.send_header(name, value)
            for cookie in resp.cookies:
                self.send_header("Set-Cookie", cookie)
        self.end_headers()
        if self.command != "HEAD" and status != HTTPStatus.NOT_MODIFIED:
            self.wfile.write(body)


class Server:
    """The running web UI. ``start()`` serves on a background thread, ``stop()`` ends it."""

    def __init__(self, httpd: _Httpd, pins: PinStore) -> None:
        self._httpd = httpd
        self._pins = pins
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._serving = threading.Event()  # serve_forever was entered
        self._gate = threading.Lock()  # orders _serve's entry against stop()

    @property
    def host(self) -> str:
        return str(self._httpd.server_address[0])

    @property
    def port(self) -> int:
        """The bound port (the real one when ``webui.port`` is 0)."""
        return int(self._httpd.server_address[1])

    def set_pin(self, pin: str) -> None:
        """The admin path: store a new operator PIN after the format check (raises PinError)."""
        self._pins.set(check_pin_format(pin, self._httpd.api.web))

    def start(self) -> Server:
        if self._thread is None:
            with self._gate:
                self._stopped = False
                self._serving.clear()
            thread = threading.Thread(
                target=self._serve,
                name="roboemot-webui",
                daemon=True,
            )
            thread.start()
            self._thread = thread
        return self

    def _serve(self) -> None:
        """The serving thread; does nothing when ``stop`` ran before the thread got going."""
        with self._gate:
            if self._stopped:
                return
            self._serving.set()
        self._httpd.serve_forever(poll_interval=SERVE_POLL_S)

    def stop(self) -> None:
        """Stop serving, close the socket and join the thread; safe to call twice."""
        with self._gate:
            self._stopped = True
            entered = self._serving.is_set()
        if self._thread is not None:
            if entered:  # shutdown() would wait for a serve_forever that never began
                self._httpd.shutdown()
            shutdown.join(self._thread, None)
            self._thread = None
        self._httpd.server_close()


def make_server(
    service: EnrollmentPort,
    diagnostics: DebugReport,
    cfg: RoboEmotConfig | Callable[[], RoboEmotConfig],
    data_dir: Path,
    *,
    clock: Callable[[], float] = time.monotonic,
    robot_hosted: bool = False,
    catalogue: CataloguePort | None = None,
    routes: RoutesPort | None = None,
    settings: SettingsPort | None = None,
    pseudonyms: PseudonymsPort | None = None,
    secrets: SecretsPort | None = None,
    live: LivePort | None = None,
) -> Server:
    """Build the web UI on ``cfg.webui.bind:port``; nothing is served until ``start()``.

    The interface is ``webui.bind``; unset, it is every interface when ``robot_hosted`` (the
    manifest names a robot profile, the app runs on the robot and the LAN must reach it) and
    loopback otherwise (sim and dev). ``cfg`` may be a callable returning the effective config,
    so the page and its limits follow runtime changes; bind, port and ``max_connections`` are read
    once here (restart fields).
    """
    current = cfg() if callable(cfg) else cfg
    web = current.webui
    pins = PinStore(Path(data_dir))
    api = WebApi(
        service,
        diagnostics,
        cfg,
        pins,
        clock=clock,
        catalogue=catalogue,
        routes=routes,
        settings=settings,
        pseudonyms=pseudonyms,
        secrets=secrets,
        live=live,
    )
    bind = web.bind or (ANY_INTERFACE if robot_hosted else LOOPBACK_INTERFACE)
    httpd = _Httpd((bind, web.port), _Handler)
    httpd.slots = threading.BoundedSemaphore(web.max_connections)  # restart field
    httpd.api = api
    httpd.statics = {
        route: ((STATIC_DIR / name).read_bytes(), ctype, None)
        for route, (name, ctype) in STATIC_FILES.items()
    }
    httpd.statics.update(load_vendor())
    return Server(httpd, pins)
