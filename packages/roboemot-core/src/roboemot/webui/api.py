"""The web UI's JSON API: routing, auth, CSRF and handlers, independent of the HTTP server.

SPEC 6 (Enrollment, registry management), 11A (admin confirmation), 12 (Data, Behaviour,
privacy) and 12A (the debug report as JSON).

The web UI never edits configuration, so it never passes ``admin=True`` to the config stack;
the operator PIN gates the registry and enrollment only. Responses carry summaries and
whitelisted status fields, never images, crops or embeddings; a last-line guard refuses any
response that holds a long list of numbers.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from typing import Any, NoReturn, Protocol

from roboemot.config.schema import ROUTE_NAME_MAX_CHARS, RoboEmotConfig, WebUIConfig
from roboemot.llm.conformance import RouteBudgetExhausted
from roboemot.registry.records import Consent, ConsentError

from . import validation as v
from .auth import LoginLimiter, PinStore, Sessions
from .guard import same_origin
from .routes import RouteBusy
from .strings import STRINGS, language_for

log = logging.getLogger(__name__)

COOKIE = "roboemot_session"
CSRF_HEADER = "x-csrf-token"
VECTOR_GUARD_LEN = 32
MAX_PIN_CHARS = 64
ENROLLMENT_ID_BYTES = 8
STATE_PATTERN = re.compile(r"[a-z_]{1,32}")
ENTITY_KINDS = frozenset({"animal", "object"})
JSON_TYPE = "application/json; charset=utf-8"
CATALOGUE_IMPORT = "/api/catalogue/import"
BASE64_GROUP_IN = 3
BASE64_GROUP_OUT = 4
CATALOGUE_STATUS = {
    "import_busy": HTTPStatus.SERVICE_UNAVAILABLE,
    "not_pending": HTTPStatus.CONFLICT,
    "entry_gone": HTTPStatus.CONFLICT,
    "no_appearance": HTTPStatus.CONFLICT,
    "kind_mismatch": HTTPStatus.UNPROCESSABLE_ENTITY,
    "category_mismatch": HTTPStatus.UNPROCESSABLE_ENTITY,
}


class EnrollmentPort(Protocol):
    """What the web UI needs from ``roboemot.registry.enrollment.EnrollmentService``.

    ``status()`` returns a mapping with ``state``, ``collected``, ``target``, ``can_complete``,
    ``face_visible`` and ``hint``; the UI copies only those, as codes and counts. ``delete``
    erases a person (``forget``), an animal, an object or a quarantined key; an unknown id
    raises ``KeyError``.
    """

    def start(self, session_id: str) -> Any: ...
    def status(self) -> Any: ...
    def complete(
        self,
        name: str,
        consent: Consent,
        declared_age_group: str | None,
        declared_gender: str | None,
        pending_id: str | None = None,
    ) -> Any: ...
    def cancel(self) -> Any: ...
    def list_records(self) -> list[Any]: ...
    def rename(self, record_id: str, display_name: str) -> Any: ...
    def merge(self, keep_id: str, drop_id: str) -> Any: ...
    def delete(self, record_id: str) -> Any: ...
    def quarantined(self) -> list[str]: ...


class RoutesPort(Protocol):
    """What the web UI needs from ``roboemot.webui.routes.RouteChecker`` (SPEC 8).

    ``start`` returns at once and raises ``KeyError`` for an unknown route, ``RouteBusy`` or
    ``RouteBudgetExhausted``; the outcome shows up in ``list_routes``. The UI copies what
    these return as is: they hold no key, prompt or reply text.
    """

    def list_routes(self) -> list[dict[str, Any]]: ...
    def start(self, name: str) -> None: ...


class CataloguePort(Protocol):
    """What the web UI needs from ``roboemot.registry.catalogue.CatalogueService``.

    ``import_sidecar`` raises an exception with a ``code`` attribute for refused input;
    ``confirm`` raises ``KeyError`` for an unknown record. The UI copies only whitelisted
    fields of what these return: never embeddings, images or paths.
    """

    def import_sidecar(self, xmp: bytes | str, image: bytes | None = None) -> Any: ...
    def list_entities(self) -> list[Any]: ...
    def live_entries(self) -> list[Any]: ...
    def confirm(self, record_id: str, entry_id: str | None = None) -> Any: ...
    def purge_candidates(self) -> int: ...
    def tombstone_count(self) -> int: ...
    def clear_tombstones(self) -> int: ...


class DebugReport(Protocol):
    """``roboemot.diagnostics.Diagnostics``: ``report()`` is the SPEC 12A JSON."""

    def report(self) -> dict[str, Any]: ...


@dataclass
class Request:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes
    client_ip: str


@dataclass
class Response:
    status: HTTPStatus
    data: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    cookies: list[str] = field(default_factory=list)


class ApiError(Exception):
    def __init__(
        self,
        status: HTTPStatus,
        code: str,
        retry_after: float | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.retry_after = retry_after
        self.field = field


def holds_vector(value: Any) -> bool:
    """True if ``value`` contains a list of ``VECTOR_GUARD_LEN`` or more numbers."""
    if isinstance(value, dict):
        return any(holds_vector(x) for x in value.values())
    if isinstance(value, list | tuple):
        numbers = sum(isinstance(x, int | float) and not isinstance(x, bool) for x in value)
        return numbers >= VECTOR_GUARD_LEN or any(holds_vector(x) for x in value)
    return False


def parse_cookie(header: str | None) -> str | None:
    if not header:
        return None
    jar: SimpleCookie = SimpleCookie()
    try:
        jar.load(header)
    except CookieError:
        return None
    morsel = jar.get(COOKIE)
    return morsel.value if morsel else None


class WebApi:
    """Handles one request at a time, thread-safe through its collaborators."""

    def __init__(
        self,
        service: EnrollmentPort,
        diagnostics: DebugReport,
        cfg: RoboEmotConfig | Callable[[], RoboEmotConfig],
        pin_store: PinStore,
        *,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        catalogue: CataloguePort | None = None,
        routes: RoutesPort | None = None,
    ) -> None:
        self.service = service
        self.catalogue = catalogue
        self.routes = routes
        self.diagnostics = diagnostics
        self._cfg = cfg
        self.pins = pin_store
        self.clock = clock
        self.now = now
        web = self.config.webui
        self.sessions = Sessions(web)
        self.limiter = LoginLimiter(web, pin_store.version())
        self._hashing = threading.Semaphore(1)

    @property
    def config(self) -> RoboEmotConfig:
        return self._cfg() if callable(self._cfg) else self._cfg

    @property
    def web(self) -> WebUIConfig:
        return self.config.webui

    # ------------------------------------------------------------------ entry point

    def handle(self, req: Request) -> Response:
        """Route ``req``; every failure becomes a JSON error with a stable code."""
        try:
            resp = self._dispatch(req)
        except ApiError as exc:
            body: dict[str, Any] = {"error": exc.code}
            if exc.field:
                body["field"] = exc.field
            resp = Response(exc.status, body)
            if exc.retry_after is not None:
                resp.headers["Retry-After"] = str(int(exc.retry_after) + 1)
        except v.ValidationError as exc:
            resp = Response(
                HTTPStatus.UNPROCESSABLE_ENTITY, {"error": exc.code, "field": exc.field}
            )
        except Exception as exc:
            log.error("web ui handler failed: %s", type(exc).__name__)
            resp = Response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "generic"})
        if holds_vector(resp.data):
            log.error("web ui response blocked: it held a long list of numbers")
            return Response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "vector_leak"})
        return resp

    def body_limit(self, method: str, path: str, headers: dict[str, str]) -> int:
        """Largest body to read for this request.

        The catalogue import carries a photo, so it gets ``webui.catalogue_max_body_bytes``,
        but only when the request already holds a live session: an anonymous client never
        makes the server read more than ``max_body_bytes``.
        """
        web = self.web
        if (method, path) == ("POST", CATALOGUE_IMPORT):
            token = parse_cookie(headers.get("cookie"))
            if self.sessions.get(token, self.clock(), self.pins.version()) is not None:
                return max(web.max_body_bytes, web.catalogue_max_body_bytes)
        return web.max_body_bytes

    def _dispatch(self, req: Request) -> Response:
        method, path = req.method, req.path
        if method not in ("GET", "POST"):
            raise ApiError(HTTPStatus.METHOD_NOT_ALLOWED, "bad_request")
        if method == "POST":
            self._check_origin(req)
        public = {
            ("GET", "/api/strings"): self._strings,
            ("GET", "/api/session"): self._session,
            ("POST", "/api/login"): self._login,
        }
        if (method, path) in public:
            return public[(method, path)](req)
        token = parse_cookie(req.headers.get("cookie"))
        session = self.sessions.get(token, self.clock(), self.pins.version())
        if session is None:
            if not path.startswith("/api/"):
                raise ApiError(HTTPStatus.NOT_FOUND, "not_found")
            raise ApiError(HTTPStatus.UNAUTHORIZED, "unauthorized")
        if method == "POST":
            sent = req.headers.get(CSRF_HEADER, "")
            if not hmac.compare_digest(sent.encode(), session.csrf.encode()):
                raise ApiError(HTTPStatus.FORBIDDEN, "csrf")
        return self._protected(req, token)

    def _protected(self, req: Request, token: str | None) -> Response:
        method, path = req.method, req.path
        if (method, path) == ("POST", "/api/logout"):
            self.sessions.drop(token)
            return Response(HTTPStatus.OK, {"ok": True}, cookies=[self._cookie("", expire=True)])
        simple: dict[tuple[str, str], Callable[[Request], Response]] = {
            ("POST", "/api/enrollment/start"): self._enroll_start,
            ("GET", "/api/enrollment/status"): self._enroll_status,
            ("POST", "/api/enrollment/complete"): self._enroll_complete,
            ("POST", "/api/enrollment/cancel"): self._enroll_cancel,
            ("GET", "/api/registry"): self._registry_list,
            ("POST", "/api/registry/merge"): self._merge,
            ("GET", "/api/registry/quarantined"): self._quarantined,
            ("POST", "/api/registry/quarantined/purge"): self._purge,
            ("GET", "/api/debug"): self._debug,
            ("GET", "/api/routes"): self._routes,
            ("GET", "/api/catalogue"): self._catalogue,
            ("POST", CATALOGUE_IMPORT): self._catalogue_import,
            ("POST", "/api/catalogue/candidates/clear"): self._catalogue_clear,
            ("POST", "/api/catalogue/tombstones/clear"): self._tombstones_clear,
        }
        if (method, path) in simple:
            return simple[(method, path)](req)
        check = re.fullmatch(
            rf"/api/routes/([A-Za-z0-9_-]{{1,{ROUTE_NAME_MAX_CHARS}}})/check", path
        )
        if method == "POST" and check:
            return self._route_check(check.group(1))
        confirm = re.fullmatch(r"/api/catalogue/([^/]+)/confirm", path)
        if method == "POST" and confirm:
            return self._catalogue_confirm(req, v.record_id(confirm.group(1)))
        match = re.fullmatch(r"/api/registry/([^/]+)/(rename|forget|delete)", path)
        if method == "POST" and match:
            rid = v.record_id(match.group(1))
            return {
                "rename": self._rename,
                "forget": self._forget,
                "delete": self._delete,
            }[match.group(2)](req, rid)
        raise ApiError(HTTPStatus.NOT_FOUND, "not_found")

    # ------------------------------------------------------------------ helpers

    def _check_origin(self, req: Request) -> None:
        origin = req.headers.get("origin")
        if origin is not None and not same_origin(origin, req.headers.get("host", "")):
            raise ApiError(HTTPStatus.FORBIDDEN, "origin")

    def _json_body(self, req: Request) -> dict[str, Any]:
        if not req.body:
            return {}
        if req.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            raise ApiError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "bad_request")
        try:
            body = json.loads(req.body, parse_constant=_reject_constant)
        except (ValueError, RecursionError, UnicodeDecodeError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_request") from exc
        if not isinstance(body, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_request")
        return body

    def _cookie(self, token: str, expire: bool = False) -> str:
        """A session cookie (no Max-Age: it dies with the browser); ``expire`` clears it."""
        base = f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/"
        return f"{base}; Max-Age=0" if expire else base

    def _open_session(self) -> Response:
        token, session = self.sessions.create(self.clock(), self.pins.version())
        return Response(
            HTTPStatus.OK,
            {"ok": True, "csrf": session.csrf},
            cookies=[self._cookie(token)],
        )

    def _kind_of(self, rid: str) -> str:
        for row in self.service.list_records():
            if row.id == rid:
                return str(row.kind)
        raise ApiError(HTTPStatus.NOT_FOUND, "not_found")

    # ------------------------------------------------------------------ open routes

    def _strings(self, req: Request) -> Response:
        lang = language_for(self.config.conversation.language)
        strings = dict(STRINGS[lang])
        cloud = (
            "consent_cloud_pseudonyms"
            if self.config.privacy.pseudonyms_to_cloud
            else ("consent_cloud_names")
        )
        strings["consent_cloud"] = strings[cloud]
        return Response(HTTPStatus.OK, {"language": lang, "strings": strings})

    def _session(self, req: Request) -> Response:
        session = self.sessions.get(
            parse_cookie(req.headers.get("cookie")), self.clock(), self.pins.version()
        )
        data: dict[str, Any] = {
            "authenticated": session is not None,
            "pin_set": self.pins.is_set(),
            "status_poll_ms": self.web.status_poll_ms,
            "name_max_chars": self.web.name_max_chars,
            "pin_min_length": self.web.pin_min_length,
        }
        if session is not None:
            data["csrf"] = session.csrf
        return Response(HTTPStatus.OK, data)

    def _login(self, req: Request) -> Response:
        body = self._json_body(req)
        v.only_keys(body, {"pin"})
        self._check_pin(req, body.get("pin"))
        return self._open_session()

    def _check_pin(self, req: Request, pin: object) -> None:
        """Verify the operator PIN under the guess limits (login and destructive re-auth).

        The attempt is reserved and counted before the hash runs, only one hash runs at a
        time, and a right PIN refunds the attempt.
        """
        if not isinstance(pin, str) or not 0 < len(pin) <= MAX_PIN_CHARS:
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_request")
        if not self.pins.is_set():
            raise ApiError(HTTPStatus.CONFLICT, "pin_not_set")
        self.limiter.sync(self.pins.version())
        wait = self.limiter.reserve(req.client_ip, self.clock())
        if wait > 0:
            raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, "locked", retry_after=wait)
        if not self._hashing.acquire(blocking=False):
            self.limiter.cancel(req.client_ip)
            raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, "busy", retry_after=0)
        try:
            ok = self.pins.verify(pin)
        finally:
            self._hashing.release()
        if not ok:
            raise ApiError(HTTPStatus.UNAUTHORIZED, "bad_pin")
        self.limiter.succeeded(req.client_ip)

    # ------------------------------------------------------------------ enrollment

    def _enroll_start(self, req: Request) -> Response:
        v.only_keys(self._json_body(req), set())
        try:
            self.service.start(secrets.token_hex(ENROLLMENT_ID_BYTES))
        except (RuntimeError, ValueError) as exc:
            raise ApiError(HTTPStatus.CONFLICT, "enrollment_not_ready") from exc
        return Response(HTTPStatus.OK, self._status_view())

    def _enroll_status(self, req: Request) -> Response:
        return Response(HTTPStatus.OK, self._status_view())

    def _status_view(self) -> dict[str, Any]:
        """The enrollment status as counts, flags and hint codes; nothing else is copied."""
        raw = self.service.status()
        raw = raw if isinstance(raw, dict) else {}
        state, collected, target = raw.get("state"), raw.get("collected"), raw.get("target")
        state = str(state) if STATE_PATTERN.fullmatch(str(state)) else "idle"
        hints = []
        if state == "collecting" and raw.get("face_visible") is False:
            hints.append("no_face")
        if raw.get("hint"):
            hints.append("turn_head")
        return {
            "state": state,
            "frames_collected": collected if isinstance(collected, int) else 0,
            "frames_needed": target if isinstance(target, int) else 0,
            "can_complete": raw.get("can_complete") is True,
            "hints": hints,
        }

    def _enroll_complete(self, req: Request) -> Response:
        form = v.enrollment_form(self._json_body(req), self.web)
        pending = form["pending_id"]
        if pending is not None and not any(
            r.id == pending and r.kind == "person" and r.status == "pending"
            for r in self.service.list_records()
        ):
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "pending_invalid", field="pending_id")
        try:
            consent = Consent(
                who=form["who"],
                when=self.now(),
                scope=form["scope"],
                guardian=form["guardian"],
            )
            self.service.complete(
                form["display_name"],
                consent,
                form["age_group"],
                form["gender"],
                form["pending_id"],
            )
        except ConsentError as exc:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "consent_refused") from exc
        except (RuntimeError, ValueError) as exc:
            raise ApiError(HTTPStatus.CONFLICT, "enrollment_not_ready") from exc
        return Response(HTTPStatus.OK, {"ok": True})

    def _enroll_cancel(self, req: Request) -> Response:
        v.only_keys(self._json_body(req), set())
        self.service.cancel()
        return Response(HTTPStatus.OK, {"ok": True})

    # ------------------------------------------------------------------ registry

    def _registry_list(self, req: Request) -> Response:
        rows = [
            {
                "id": str(r.id),
                "display_name": str(r.display_name),
                "status": str(r.status),
                "kind": str(r.kind),
                "last_seen": None if r.last_seen is None else r.last_seen.isoformat(),
                "age_group": getattr(r, "age_group", None),
            }
            for r in self.service.list_records()
        ]
        return Response(HTTPStatus.OK, {"records": rows})

    def _rename(self, req: Request, rid: str) -> Response:
        body = self._json_body(req)
        v.only_keys(body, {"display_name"})
        display = v.name(body.get("display_name"), "display_name", self.web)
        self._kind_of(rid)
        self._call(self.service.rename, rid, display)
        return Response(HTTPStatus.OK, {"ok": True})

    def _merge(self, req: Request) -> Response:
        body = self._json_body(req)
        v.only_keys(body, {"keep_id", "drop_id"})
        keep, drop = (
            v.record_id(body.get("keep_id"), "keep_id"),
            v.record_id(body.get("drop_id"), "drop_id"),
        )
        if keep == drop:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "id_invalid", field="drop_id")
        if self._kind_of(keep) not in ENTITY_KINDS or self._kind_of(drop) not in ENTITY_KINDS:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "wrong_kind")
        self._call(self.service.merge, keep, drop)
        return Response(HTTPStatus.OK, {"ok": True})

    def _forget(self, req: Request, rid: str) -> Response:
        body = self._json_body(req)
        v.only_keys(body, {"confirm", "pin"})
        if self._kind_of(rid) != "person":
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "wrong_kind")
        if not v.confirmed(body):
            raise ApiError(HTTPStatus.CONFLICT, "confirm_required")
        self._check_pin(req, body.get("pin"))
        self._erase(self.service.delete, rid)
        return Response(HTTPStatus.OK, {"ok": True})

    def _delete(self, req: Request, rid: str) -> Response:
        body = self._json_body(req)
        v.only_keys(body, {"confirm"})
        if self._kind_of(rid) not in ENTITY_KINDS:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "wrong_kind")
        if not v.confirmed(body):
            raise ApiError(HTTPStatus.CONFLICT, "confirm_required")
        self._call(self.service.delete, rid)
        return Response(HTTPStatus.OK, {"ok": True})

    def _quarantined(self, req: Request) -> Response:
        return Response(HTTPStatus.OK, {"keys": [str(k) for k in self.service.quarantined()]})

    def _purge(self, req: Request) -> Response:
        body = self._json_body(req)
        v.only_keys(body, {"key", "confirm", "pin"})
        key = v.quarantine_key(body.get("key"))
        if key not in self.service.quarantined():
            raise ApiError(HTTPStatus.NOT_FOUND, "not_found")
        if not v.confirmed(body):
            raise ApiError(HTTPStatus.CONFLICT, "confirm_required")
        self._check_pin(req, body.get("pin"))
        self._erase(self.service.delete, key)
        return Response(HTTPStatus.OK, {"ok": True})

    def _call(self, fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return fn(*args)
        except KeyError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, "not_found") from exc
        except ValueError as exc:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "bad_request") from exc

    def _erase(self, fn: Callable[[str], Any], key: str) -> None:
        """Synchronous erasure; a partial failure is reported so the operator retries."""
        try:
            found = fn(key)
        except KeyError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, "not_found") from exc
        except Exception as exc:
            log.error("erasure incomplete: %s", type(exc).__name__)
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "forget_incomplete") from exc
        if found is False:
            raise ApiError(HTTPStatus.NOT_FOUND, "not_found")

    # ------------------------------------------------------------------ other

    def _debug(self, req: Request) -> Response:
        """The 12A report; a section that holds a long list of numbers is withheld, not the rest."""
        report = self.diagnostics.report()
        sections = report.get("sections")
        if isinstance(sections, dict):
            for name, section in list(sections.items()):
                if holds_vector(section):
                    log.error("debug section %r withheld: it held a long list of numbers", name)
                    sections[name] = {
                        "scope": section.get("scope") if isinstance(section, dict) else None,
                        "ok": False,
                        "error": "withheld",
                    }
        return Response(HTTPStatus.OK, report)

    # ------------------------------------------------------------------ llm routes

    def _routes_port(self) -> RoutesPort:
        if self.routes is None:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "routes_unavailable")
        return self.routes

    def _routes(self, req: Request) -> Response:
        """Routes by name, model, host and minors flag, with the last conformance report."""
        return Response(HTTPStatus.OK, {"routes": self._routes_port().list_routes()})

    def _route_check(self, name: str) -> Response:
        """Start the conformance test of one route in the background (it outlives the request
        deadline); advisory, it does not enable or gate the route. The list shows the outcome."""
        port = self._routes_port()
        try:
            port.start(name)
        except KeyError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, "route_unknown") from exc
        except RouteBusy as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "route_busy") from exc
        except RouteBudgetExhausted as exc:
            raise ApiError(HTTPStatus.CONFLICT, "route_budget") from exc
        return Response(HTTPStatus.ACCEPTED, {"running": True})

    # ------------------------------------------------------------------ catalogue

    def _port(self) -> CataloguePort:
        if self.catalogue is None:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "catalogue_unavailable")
        return self.catalogue

    def _catalogue(self, req: Request) -> Response:
        """Pending and auto-enrolled records plus the animals and objects seen right now."""
        port = self._port()
        records = [
            {
                "id": str(r.id),
                "display_name": str(r.display_name),
                "kind": str(r.kind),
                "category": str(r.category),
                "status": str(r.status),
                "origin": str(r.origin),
                "has_appearance": r.has_appearance is True,
                "last_seen": None if r.last_seen is None else r.last_seen.isoformat(),
            }
            for r in port.list_entities()
        ]
        live = [
            {
                "entry_id": str(e.entry_id),
                "kind": str(e.kind),
                "category": str(e.category),
                "age_s": round(float(e.age_s), 1),
                "identified": e.identified is True,
            }
            for e in port.live_entries()
        ]
        cat = self.config.catalogue
        limits = {
            "max_xmp_bytes": cat.max_xmp_bytes,
            "max_image_bytes": cat.max_image_bytes,
            "max_items": cat.max_items,
        }
        return Response(
            HTTPStatus.OK,
            {
                "records": records,
                "live": live,
                "limits": limits,
                "tombstones": int(port.tombstone_count()),
            },
        )

    def _catalogue_import(self, req: Request) -> Response:
        """One XMP sidecar, optionally with its photo (base64), per request (SPEC 6)."""
        port = self._port()
        body = self._json_body(req)
        v.only_keys(body, {"xmp", "image_b64"})
        cat = self.config.catalogue
        xmp = body.get("xmp")
        if not isinstance(xmp, str) or not xmp.strip():
            raise v.ValidationError("xmp_required", "xmp")
        if len(xmp) > cat.max_xmp_bytes:
            raise v.ValidationError("xmp_too_large", "xmp")
        image = self._image_bytes(body.get("image_b64"), cat.max_image_bytes)
        try:
            report = port.import_sidecar(xmp, image)
        except Exception as exc:
            self._catalogue_error(exc)
        return Response(
            HTTPStatus.OK,
            {
                "items": [
                    {
                        "id": str(i.id),
                        "display_name": str(i.display_name),
                        "kind": str(i.kind),
                        "category": str(i.category),
                        "created": i.created is True,
                        "embeddings": int(i.embeddings),
                    }
                    for i in report.items
                ],
                "conflicts": [{"name": str(c.name), "code": str(c.code)} for c in report.conflicts],
                "skipped": {str(k): int(n) for k, n in report.skipped.items()},
                "photo": str(report.photo),
                "truncated": report.truncated is True,
            },
        )

    @staticmethod
    def _image_bytes(value: object, max_bytes: int) -> bytes | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise v.ValidationError("image_corrupt", "image_b64")
        if len(value) > -(-max_bytes // BASE64_GROUP_IN) * BASE64_GROUP_OUT:
            raise v.ValidationError("image_too_large", "image_b64")
        try:
            return base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise v.ValidationError("image_corrupt", "image_b64") from exc

    def _catalogue_error(self, exc: Exception) -> NoReturn:
        """Map a refused import or confirmation to a stable code; anything else is a 500."""
        code = getattr(exc, "code", None)
        if isinstance(exc, KeyError):
            raise ApiError(HTTPStatus.NOT_FOUND, "not_found") from exc
        if isinstance(code, str) and re.fullmatch(r"[a-z_]{1,48}", code):
            status = CATALOGUE_STATUS.get(code, HTTPStatus.UNPROCESSABLE_ENTITY)
            raise ApiError(status, code) from exc
        raise exc

    def _catalogue_confirm(self, req: Request, rid: str) -> Response:
        """ "This is Mia": activate a pending record from a live entry or its photo rows."""
        port = self._port()
        body = self._json_body(req)
        v.only_keys(body, {"entry_id"})
        entry = body.get("entry_id")
        entry_id = None if entry is None else v.record_id(entry, "entry_id")
        try:
            port.confirm(rid, entry_id)
        except Exception as exc:
            self._catalogue_error(exc)
        return Response(HTTPStatus.OK, {"ok": True})

    def _catalogue_clear(self, req: Request) -> Response:
        """Forget what auto-enrollment has collected so far (PIN again)."""
        port = self._port()
        body = self._json_body(req)
        v.only_keys(body, {"confirm", "pin"})
        if not v.confirmed(body):
            raise ApiError(HTTPStatus.CONFLICT, "confirm_required")
        self._check_pin(req, body.get("pin"))
        return Response(HTTPStatus.OK, {"cleared": int(port.purge_candidates())})

    def _tombstones_clear(self, req: Request) -> Response:
        """Let a catalogue import bring back records the owner deleted (PIN again)."""
        port = self._port()
        body = self._json_body(req)
        v.only_keys(body, {"confirm", "pin"})
        if not v.confirmed(body):
            raise ApiError(HTTPStatus.CONFLICT, "confirm_required")
        self._check_pin(req, body.get("pin"))
        return Response(HTTPStatus.OK, {"cleared": int(port.clear_tombstones())})


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name}")
