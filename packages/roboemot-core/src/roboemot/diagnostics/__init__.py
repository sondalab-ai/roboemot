"""Debug report: robot state, hardware, resources, system and battery (SPEC 12A, Diagnostics).

A report is a JSON-safe dict built from named :class:`Source` objects. Each source fills one
section; a source that fails or hangs past ``SECTION_TIMEOUT_S`` becomes an error section and
the others still report. The same dict serves the CLI (``roboemot debug``) and, from M1, the
app's web page; :class:`Diagnostics` is the registry a running app fills with its live objects.

Sources are robot-agnostic. Generic ones (system, temperatures, battery, process) live in
:mod:`roboemot.diagnostics.system`. A platform package adds its own by registering a factory in
the ``roboemot.diagnostics`` entry-point group under the manifest's ``robot.id``; the factory
takes a :class:`Target` (``None`` host means "this machine") and returns a list of sources.

Every section has a ``scope``: ``local`` describes the machine that built the report, ``robot``
comes from the robot's platform, ``runtime`` from the running app. Local sections only describe
the robot when the report is built on the robot.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import math
import socket
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Literal, Protocol

from roboemot.redact import REDACTED, is_secret_key, secret_shaped

SCHEMA_VERSION = 1
SECTION_TIMEOUT_S = 5.0
ENTRY_POINT_GROUP = "roboemot.diagnostics"
LOOPBACK = {"localhost", "127.0.0.1", "::1"}

Scope = Literal["local", "robot", "runtime"]


class Source(Protocol):
    name: str
    scope: Scope

    def collect(self) -> Any: ...


@dataclass
class FnSource:
    """A source from a plain callable, for live objects (``loop.snapshot``) and tests."""

    name: str
    fn: Callable[[], Any]
    scope: Scope = "runtime"

    def collect(self) -> Any:
        return self.fn()


@dataclass(frozen=True)
class Target:
    """Where the robot's platform answers; ``host=None`` is this machine."""

    host: str | None = None
    port: int | None = None

    @property
    def remote(self) -> bool:
        return self.host is not None and self.host not in LOOPBACK | {socket.gethostname()}


PlatformFactory = Callable[[Target], list[Source]]


def platform_sources(manifest_id: str, target: Target) -> list[Source]:
    """Sources the platform package registered for ``manifest_id``, or one error section."""
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == manifest_id:
            try:
                return list(ep.load()(target))
            except Exception as exc:  # a broken plugin must not take the report (or app) down
                return [FnSource("platform", _raise(exc), scope="robot")]
    msg = f"no diagnostics source installed for {manifest_id}"
    return [FnSource("platform", _raise(LookupError(msg)), scope="robot")]


def installed_platforms() -> list[str]:
    return sorted({ep.name for ep in entry_points(group=ENTRY_POINT_GROUP)})


def _raise(exc: Exception) -> Callable[[], Any]:
    def fn() -> Any:
        raise exc

    return fn


def collect_section(source: Source, timeout_s: float = SECTION_TIMEOUT_S) -> dict[str, Any]:
    """Run one source in a worker thread so a hung call cannot stall the report."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["data"] = source.collect()
        except Exception as exc:
            box["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=run, name=f"diag-{source.name}", daemon=True)
    worker.start()
    worker.join(timeout_s)
    section: dict[str, Any] = {"scope": source.scope}
    if worker.is_alive():
        return section | {"ok": False, "error": f"timed out after {timeout_s:g} s"}
    if "error" in box:
        return section | {"ok": False, "error": box["error"]}
    return section | {"ok": True, "data": jsonable(box.get("data"), bare_user=False)}


def collect(
    sources: Iterable[Source], *, timeout_s: float = SECTION_TIMEOUT_S, notes: Iterable[str] = ()
) -> dict[str, Any]:
    """Build the report; sections run in parallel, so it takes about the slowest one."""
    sources = list(sources)
    results: dict[str, dict[str, Any]] = {}

    def one(s: Source) -> None:
        results[s.name] = collect_section(s, timeout_s)

    threads = [threading.Thread(target=one, args=(s,), daemon=True) for s in sources]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "notes": list(notes),
        "sections": {s.name: results[s.name] for s in sources},
    }


@dataclass
class Diagnostics:
    """Registry a running app fills; ``report()`` is what a web route returns as JSON."""

    sources: list[Source] = dataclasses.field(default_factory=list)
    timeout_s: float = SECTION_TIMEOUT_S

    def add(self, *sources: Source) -> Diagnostics:
        self.sources.extend(sources)
        return self

    def report(self) -> dict[str, Any]:
        return collect(self.sources, timeout_s=self.timeout_s)


def jsonable(value: Any, *, bare_user: bool = True) -> Any:
    """Plain JSON types only: no numpy scalars, enums, datetimes, NaN; secrets redacted.
    ``bare_user`` False (debug report) keeps ``scheme://user@host``; ``user:pass@`` stays hidden."""
    if isinstance(value, str):
        return REDACTED if secret_shaped(value, bare_user=bare_user) else value
    if isinstance(value, bool | int) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, enum.Enum):
        return jsonable(value.value, bare_user=bare_user)
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    elif hasattr(value, "_asdict"):  # namedtuple, e.g. psutil results
        value = value._asdict()
    elif hasattr(value, "model_dump"):  # pydantic
        value = value.model_dump(mode="json")
    elif hasattr(value, "tolist"):  # numpy scalar or array
        return jsonable(value.tolist(), bare_user=bare_user)
    if isinstance(value, dict):
        return {
            str(k): REDACTED if is_secret_key(str(k)) else jsonable(v, bare_user=bare_user)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(v, bare_user=bare_user) for v in value]
    return str(value)


def _printable(text: str) -> str:
    """Escape control characters: report strings come from devices and must not drive the
    terminal (ANSI/OSC escape sequences)."""
    return "".join(c if c.isprintable() else repr(c)[1:-1] for c in text)


def render_text(report: dict[str, Any]) -> str:
    """Human-readable form of a report, for the terminal."""
    lines = [f"roboemot debug report  host={report['host']}  at={report['generated_at']}"]
    lines += [f"note: {_printable(n)}" for n in report.get("notes", [])]
    for name, sec in report["sections"].items():
        lines.append("")
        lines.append(f"[{_printable(name)}] ({sec['scope']})")
        if not sec["ok"]:
            lines.append(f"  error: {_printable(sec['error'])}")
            continue
        lines += _render_value(sec["data"], 1)
    return "\n".join(lines)


def _render_value(value: Any, depth: int) -> list[str]:
    pad = "  " * depth
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            if isinstance(v, dict | list) and v:
                out.append(f"{pad}{_printable(k)}:")
                out += _render_value(v, depth + 1)
            else:
                out.append(f"{pad}{_printable(k)}: {_scalar(v)}")
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, dict | list):
                out.append(f"{pad}-")
                out += _render_value(v, depth + 1)
            else:
                out.append(f"{pad}- {_scalar(v)}")
        return out
    return [f"{pad}{_scalar(value)}"]


def _scalar(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list | dict):
        return "[]" if isinstance(v, list) else "{}"
    return _printable(str(v))


__all__ = [
    "ENTRY_POINT_GROUP",
    "SCHEMA_VERSION",
    "SECTION_TIMEOUT_S",
    "Diagnostics",
    "FnSource",
    "PlatformFactory",
    "Scope",
    "Source",
    "Target",
    "collect",
    "collect_section",
    "installed_platforms",
    "REDACTED",
    "is_secret_key",
    "jsonable",
    "secret_shaped",
    "platform_sources",
    "render_text",
]
