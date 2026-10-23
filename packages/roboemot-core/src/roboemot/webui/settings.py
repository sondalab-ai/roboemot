"""The web-editable settings: the generic write path over the field catalogue (SPEC 11A, 12).

Every catalogue leaf that is not a secret can be written (``fields()`` marks it ``writable``);
a path that is not such a leaf is refused. A value is cleaned from its descriptor (kind and
limits, ``setting_clean.clean_value``) or by the per-path hook that owns it (wake word, wake
phrases: ``setting_clean.HOOKS``), then written through
``ConfigStack.set``, so pydantic validation, the guardrails, audit (source ``webui:<ip>``) and hot
reload apply unchanged. A secret field and a secret-shaped value (also inside an added entry) are
never accepted.

Layers: runtime by default. When the stack refuses a change only because it loosens a safety
value (a field's own direction; never the reflex, locality or secret rules, which stay refused
on both layers), it is retried once on the app layer, which writes the app YAML and the audit.
Entries of a merged list are addressed by identity; entries can be added, and removed when only the
runtime layer holds them (a lower layer cannot be masked by a merge, so that is refused). Each
accepted runtime edit is kept in a 0600 overrides file in the data folder and replayed as one batch
at startup (``restore``). The PIN is the caller's job (``WebApi``); this module never sees a PIN
or a secret.
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import secrets
import socket
import stat
import tempfile
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlsplit

import httpx

from roboemot import paths
from roboemot.config.catalogue import (
    CONTAINERS,
    ENTRY_KINDS,
    FieldDescriptor,
    build_catalogue,
    write_only,
)
from roboemot.config.schema import RoboEmotConfig
from roboemot.config.stack import (
    DROP,
    KEYED_LISTS,
    NO_KEY,
    REMOVED_SETTINGS,
    RENAMED_SETTINGS,
    ConfigError,
    GuardrailError,
    deep_merge,
    dropped_entry,
    endpoint_key,
    entry_identity,
    get_path,
    ident_fields,
    keyed_index,
    keyed_ref,
    log_dropped,
    set_path,
    without_dropped,
)
from roboemot.redact import REDACTED, secret_shaped
from roboemot.redact import origin as plain_origin
from roboemot.secret_store import NAME as SECRET_NAME
from roboemot.secret_store import SECRET_REF
from roboemot.voice.models import ModelError, supported_languages

from .languages import LANGUAGE_PATH, installed_languages
from .robot_section import (
    RobotBlock,
    SettingsPlugin,
    build_section,
    claimed,
    load_settings_plugin,
    restart_hook,
    vet_claims,
)
from .setting_clean import (
    clean_hooked,
    clean_key,
    clean_value,
    hook_for,
)
from .validation import ValidationError

if TYPE_CHECKING:
    from roboemot.manifest import Manifest

log = logging.getLogger(__name__)
SOURCE = "webui"
RESTORE_SOURCE = "webui:restore"
FILE_VERSION = 1
MAX_FILE_BYTES = 1_000_000  # bounds what a damaged or hostile file can make the app read
FILE_MODE = 0o600
GROUP_OTHER_BITS = 0o077
NOT_WRITABLE = frozenset({"schema_version"})
# keyed by language: a new key must be one the voice stack supports
LANGUAGE_MAPS = frozenset({"voice.keywords"})
# whole-value maps keyed by language (free text per key): a key that is new must be a language the
# voice stack supports, as for the keyed language maps above
LANGUAGE_DICTS = frozenset(
    {
        "voice.voices",
        "voice.asr.models",
        "conversation.fallback_lines",
        "privacy.pseudonym_templates",
        "connectivity.offline_phrases",
    }
)
REFUSED_CHECKS = frozenset({"reflex", "locality", "secret"})  # never granted, on any layer
APP_FILE = "app"
NO_KEY_REF = "none"  # the api_key of a route without a key


@dataclass(frozen=True)
class Move:
    """An accepted endpoint change of a route with an API key: origins, route name and its key."""

    old: str
    new: str
    route: str
    api_key: str


def fingerprint(api_key: str) -> str:
    """``salt:digest`` of an ``api_key`` (reference text or literal), stored with an acknowledged
    endpoint so the ack holds only for the key it was given for. The key itself is never stored."""
    salt = secrets.token_hex()
    return f"{salt}:{hashlib.sha256((salt + api_key).encode()).hexdigest()}"


def key_matches(stored: str, api_key: str) -> bool:
    """Whether ``api_key`` is the one ``fingerprint`` was made from."""
    salt, _, digest = stored.partition(":")
    return secrets.compare_digest(digest, hashlib.sha256((salt + api_key).encode()).hexdigest())


class StackPort(Protocol):
    """The part of ``roboemot.config.ConfigStack`` this module uses."""

    files: dict[str, Path]

    listener_failures: int  # reload listeners that raised so far (R3 SF3)

    def effective(self) -> RoboEmotConfig: ...

    def merged_tree(self, *, include_runtime: bool = True) -> dict[str, Any]: ...

    def source_of(self, path: str, tree: dict[str, Any] | None = None) -> str: ...

    def set(self, key: str, value: Any, **kwargs: Any) -> RoboEmotConfig: ...

    def set_many(self, changes: dict[str, Any], **kwargs: Any) -> RoboEmotConfig: ...

    def preview(self, changes: dict[str, Any]) -> RoboEmotConfig: ...

    def clear(self, key: str, **kwargs: Any) -> bool: ...


def languages() -> frozenset[str]:
    """The languages a new language key may name; none when the voice manifest cannot be read."""
    try:
        return frozenset(supported_languages())
    except ModelError:
        return frozenset()


class SettingNotAllowed(LookupError):
    """The path is not a writable catalogue field (or entry container)."""


def _writable(path: str) -> bool:
    return path not in NOT_WRITABLE


def plain(value: Any) -> Any:
    """``value`` as plain JSON types (models dumped, tuples as lists)."""
    return json.loads(json.dumps(value, default=lambda o: o.model_dump(mode="json")))


def loosening_only(exc: GuardrailError) -> bool:
    """Whether every refusal is a field's own safety direction: the only case that may be retried
    on the app layer. Reflex, locality and secret findings, privacy and admin ones, and an error
    without structured findings are not."""
    return bool(exc.violations) and all(
        v.kind == "safety" and v.check == "field" for v in exc.violations
    )


class OverridesFile:
    """The persisted edits, ``{"v": 1, "overrides": {path: value}}``, mode 0600 in the data folder.

    Written like the secret store (temp file in the same folder, fsync, rename, folder fsync); a
    read refuses a file that is not a regular file owned by this user or that group or others
    can access, or that is oversized or not the expected shape. Nothing here logs a value.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, str]]:
        """The stored edits, the acknowledged route endpoints (``{file key: {"origin", "key"}}``,
        see ``SettingsEditor.change``) and the keys bound to routes from the web (``{route name:
        "none" or "${secret:name}"}``); all {} when there is no file. Raises ``ValueError``
        (reason only, no content) when the file is unsafe or malformed."""
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except FileNotFoundError:
            return {}, {}, {}
        except OSError as exc:
            raise ValueError(f"cannot open ({exc.strerror})") from exc
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("not a regular file owned by this user")
            if stat.S_IMODE(info.st_mode) & GROUP_OTHER_BITS:
                raise ValueError("readable by group or others")
            if info.st_size > MAX_FILE_BYTES:
                raise ValueError("too large")
            raw = handle.read(MAX_FILE_BYTES + 1)
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ValueError("not valid JSON") from exc
        found = data.get("overrides") if isinstance(data, dict) else None
        if not isinstance(data, dict) or data.get("v") != FILE_VERSION:
            raise ValueError("unexpected shape")
        acks, keys = data.get("acks", {}), data.get("keys", {})
        if not isinstance(found, dict) or not isinstance(acks, dict) or not isinstance(keys, dict):
            raise ValueError("unexpected shape")
        good = {
            k: v
            for k, v in acks.items()
            if isinstance(v, dict) and all(isinstance(v.get(f), str) for f in ("origin", "key"))
        }
        bound = {k: v for k, v in keys.items() if isinstance(k, str) and is_key_ref(v)}
        return found, good, bound

    def save(
        self, overrides: dict[str, Any], acks: dict[str, dict[str, str]], keys: dict[str, str]
    ) -> None:
        """Replace the file atomically; the data folder is created private when missing."""
        folder = paths.data_dir(str(self.path.parent))
        fd, name = tempfile.mkstemp(dir=folder, prefix=self.path.name + ".")  # 0600
        tmp = Path(name)
        try:
            with os.fdopen(fd, "w") as handle:
                os.fchmod(handle.fileno(), FILE_MODE)
                handle.write(
                    json.dumps(
                        {
                            "v": FILE_VERSION,
                            "overrides": overrides,
                            **({"acks": acks} if acks else {}),
                            **({"keys": keys} if keys else {}),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
            dir_fd = os.open(folder, os.O_RDONLY | os.O_CLOEXEC)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise


def origin(url: str) -> str:
    """``scheme://host[:port]`` as the web UI names and acknowledges it: the host as the HTTP client
    resolves it (``httpx.URL(...).raw_host``: IDNA 2008, UTS 46 non-transitional), so a look-alike
    letter cannot pass for another host and the acknowledged host is the one the key goes to.
    '' when unparsable or the host cannot be encoded."""
    found = plain_origin(url)
    if found.isascii():
        return found
    try:
        host = httpx.URL(url).raw_host.decode("ascii")
        parts = urlsplit(url)
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme.lower()}://{'[' + host + ']' if ':' in host else host}{port}"
    except (httpx.InvalidURL, ValueError, UnicodeError):
        return ""


# IPv6 forms that carry an IPv4 address in their low 32 bits: IPv4-compatible (``::a.b.c.d``),
# IPv4-mapped, IPv4-translated (``::ffff:0:a.b.c.d``) and the NAT64 well-known prefix
EMBEDS_IPV4 = tuple(
    ipaddress.IPv6Network(net)
    for net in ("::/96", "::ffff:0:0/96", "::ffff:0:0:0/96", "64:ff9b::/96")
)
IPV4_BITS = 0xFFFFFFFF


def this_machine(url: object) -> bool:
    """Whether ``url`` points at this machine or a link-local address (SPEC 8, 12): loopback,
    ``localhost`` and ``*.localhost``, the unspecified address (it reaches loopback on Linux),
    169.254/16 and fe80::/10, in any spelling (``127.1``, ``0x7f000001``, ``2130706433``, an IPv6
    address that carries one of them, ``EMBEDS_IPV4``). The host is read by httpx, the client
    that connects. Names are not resolved: a public name that resolves to loopback is not caught
    here. Private LAN addresses stay allowed, a LAN model server is the usual local route."""
    try:
        host = httpx.URL(str(url)).host.lower().rstrip(".")
    except (httpx.InvalidURL, ValueError, TypeError):
        return False  # unparsable: ``LLMRoute`` refuses it (``route_url_ok``, same parser)
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(host)
    except ValueError:
        try:  # the shorthand IPv4 forms the socket layer accepts
            ip = ipaddress.IPv4Address(socket.inet_aton(host))
        except OSError:
            return False
    if ip.is_loopback or ip.is_link_local or ip.is_unspecified:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and any(ip in net for net in EMBEDS_IPV4):
        carried = ipaddress.IPv4Address(int(ip) & IPV4_BITS)
        return carried.is_loopback or carried.is_link_local or carried.is_unspecified
    return False


def check_web_route_url(url: object, old: object = None) -> None:
    """The web UI's rules for a route ``base_url`` it writes (SPEC 8, 12), on top of the schema's:
    no query or fragment (``url_query``: the SDK appends its path, so either would sit inside the
    request path), and not this machine or a link-local address (``url_local_refused``) unless the
    origin stays the one the route already has (a path edit of a route of the app config)."""
    if isinstance(url, str) and ("?" in url or "#" in url):
        raise ValidationError("url_query", "value")
    if this_machine(url) and not (isinstance(old, str) and origin(old) == origin(str(url))):
        raise ValidationError("url_local_refused", "value")


def is_key_ref(value: object) -> bool:
    """Whether ``value`` is a route key a web edit may bind: ``none`` or one whole
    ``${secret:name}`` reference (never a literal key)."""
    return isinstance(value, str) and (value == NO_KEY_REF or SECRET_REF.match(value) is not None)


def okey(tree: dict[str, Any], path: str) -> str:
    """The key an edit has in the overrides file: ``path``, except that an index into a merged
    list becomes ``list[{identity}].rest``, so a replay finds the same entry wherever it sits."""
    ref = keyed_ref(tree, path, named=True)
    if ref is None:
        return path
    return entry_key(*ref)


def entry_key(list_path: str, ident: dict[str, Any], rest: str = "") -> str:
    """``list[{identity}]`` plus ``.rest`` when given: the file key of a merged-list entry."""
    text = json.dumps(ident, sort_keys=True, separators=(",", ":"))
    return f"{list_path}[{text}]" + (f".{rest}" if rest else "")


def parse_okey(key: str) -> tuple[str, dict[str, Any], str] | None:
    """Inverse of ``okey`` for a merged-list key: ``(list path, identity, rest)``, else ``None``."""
    for list_path in KEYED_LISTS:
        if not key.startswith(list_path + "["):
            continue
        try:
            ident, end = json.JSONDecoder().raw_decode(key, len(list_path) + 1)
        except ValueError:
            return None
        if not isinstance(ident, dict) or key[end : end + 1] != "]":
            return None
        rest = key[end + 1 :]
        if rest and not rest.startswith("."):
            return None
        return list_path, ident, rest[1:]
    return None


class SettingsEditor:
    """Lists the writable settings and applies validated changes: one value, an entry added or
    removed, or a runtime value reset."""

    def __init__(
        self,
        stack: StackPort,
        *,
        store: Path | None = None,
        manifest: Manifest | None = None,
        plugin: SettingsPlugin | None = None,
        known_secrets: Callable[[], Iterable[str]] | None = None,
        installed: Callable[[RoboEmotConfig], Iterable[str]] = installed_languages,
    ):
        self.stack = stack
        self._installed = installed
        self.manifest = manifest
        self._known_secrets = known_secrets
        self._plugin = plugin
        if plugin is None and manifest is not None:
            self._plugin = load_settings_plugin(manifest.robot.id)
        self._file = OverridesFile(store) if store else None
        self._overrides: dict[str, Any] = {}
        # file key of a moved route base_url -> acknowledged origin and key fingerprint
        self._acks: dict[str, dict[str, str]] = {}
        # route name -> key bound from the web (``none`` or ``${secret:name}``), SPEC 8 Secrets
        self._keys: dict[str, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ reading

    def fields(self) -> list[FieldDescriptor]:
        """The full field catalogue, ``writable`` true for every non-secret leaf. Private values
        (privacy, location) are hidden."""
        return self._catalogue()

    def robot_split(
        self, fields: list[FieldDescriptor]
    ) -> tuple[list[FieldDescriptor], RobotBlock | None]:
        """Split a ``fields()`` result into the core rows and the robot block. The plugin's
        read-only rows come first, then the rows of the config paths it claims (same descriptors
        as before, still written by path); core rows lose the claimed ones. With no plugin, or a
        failing one, the rows are returned whole and the block is ``None``."""
        built = build_section(self._plugin, self.stack, self.manifest)
        if built is None:
            return fields, None
        section, robot_id = built
        claims = vet_claims(section.claims, fields)
        mine = [f for f in fields if claimed(f["path"], claims)]
        rest = [f for f in fields if not claimed(f["path"], claims)]
        block: RobotBlock = {
            "id": robot_id,
            "title": section.title,
            "actions": list(section.actions),
            "restart_effects": list(section.restart_effects),
            "fields": [*section.fields, *mine],
        }
        return rest, block

    def restart_hook(self) -> Callable[[], None] | None:
        """The robot plugin's restart, or ``None`` when it has none or cannot restart now."""
        return restart_hook(self._plugin)

    def reveal(self) -> list[FieldDescriptor]:
        """Only the private fields, with their real values (secrets stay redacted). The caller
        has verified the PIN."""
        return [f for f in self._catalogue(reveal_private=True) if f["private"]]

    # ------------------------------------------------------------------ writing

    def change(
        self, path: object, value: object, source: str = SOURCE, ack: object = None
    ) -> dict[str, Any]:
        """Validate ``value`` for ``path``, write it and keep it in the overrides file; the
        stack's guardrails have the last word. A route with an API key may change its endpoint
        (scheme, host, port) only when ``ack`` equals the new endpoint as the server writes it (the
        ``endpoint_ack_required`` error carries it); the move is then audited by host.

        Raises ``SettingNotAllowed``, ``ValidationError`` (code, no value echoed) and the
        stack's ``GuardrailError`` / ``ConfigError``. An unchanged value writes nothing. The result
        names the ``layer`` written (``runtime``, or ``app`` for a safety loosening), whether
        the field ``restart_required`` and ``persisted`` (False when the file could not be
        written: the edit is live until restart).
        """
        with self._lock:
            before = self._failures()
            result = self._apply(path, value, source, self._rows(), self.stack.effective(), ack)
            key = result.pop("okey", None)
            acked = result.pop("acked", None)
            if result["changed"] and result["layer"] == "runtime":
                self._remember(key, result["value"])
                self._keep_ack(key, result["value"], acked)
                self._rebase_binding(path, acked)
                result["persisted"] = self._persist()
            elif result["changed"] and self._overrides.pop(key, None) is not None:
                self._acks.pop(key, None)
                result["persisted"] = self._persist()  # the app value replaced a runtime edit
            return self._applied(result, before)

    def add_entry(
        self, path: object, key: object, value: object, source: str = SOURCE
    ) -> dict[str, Any]:
        """Add an entry to the collection at ``path`` (a dict of models by ``key``, a keyed list
        by the identity inside ``value``) on the runtime layer. A new language key of a language
        map (``LANGUAGE_MAPS``) must be one the voice stack supports (``supported_languages``).
        A route added here has no key (``add_route``).

        Raises like ``change``; ``entry_exists`` when the key or identity is taken,
        ``entry_invalid`` for a field the entry type does not have or a secret field.
        """
        with self._lock:
            before = self._failures()
            return self._applied(self._add_entry(path, key, value, source), before)

    def _add_entry(
        self,
        path: object,
        key: object,
        value: object,
        source: str,
        bind: tuple[str, str | None] | None = None,
    ) -> dict[str, Any]:
        """``add_entry`` under the lock. ``bind`` (a route only) is ``(reference, ack)``: the
        entry is written with that ``api_key`` once its endpoint is acknowledged."""
        rows, cfg = self._rows(), self.stack.effective()
        box = rows.get(path) if isinstance(path, str) else None
        if box is None or box["secret"] or box["kind"] not in ENTRY_KINDS:
            raise SettingNotAllowed("path")
        url = value.get("base_url") if isinstance(value, dict) else None
        if box["path"] == "llm_routes" and secret_shaped(url):
            raise ValidationError("url_credentials", "value")
        if box["path"] == "llm_routes":
            check_web_route_url(url)
        if secret_shaped([key, value]):
            raise ValidationError("value_type", "value")
        name = clean_key(key, cfg) if box["kind"] != "keyed_list" else None
        if name is not None and box["path"] in LANGUAGE_MAPS and name not in languages():
            raise ValidationError("language_unsupported", "key")
        entry = self._clean_entry(box, value, cfg, name)
        tree = self.stack.merged_tree()
        exempt: frozenset[str] = frozenset()
        if name is not None:
            if name in get_path(tree, box["path"]):
                raise ValidationError("entry_exists", "key")
            target, stored, written = f"{box['path']}.{name}", f"{box['path']}.{name}", entry
        else:
            target, stored, written = self._list_add(box, entry, tree, cfg)
            if bind is not None and bind[0] != NO_KEY_REF:
                self._check_endpoint_ack(entry.get("base_url"), bind[1])
                written = [{**entry, "api_key": bind[0]}]
                exempt = frozenset({f"llm_routes.{entry['name']}.api_key"})
        self.stack.set(
            target,
            written,
            source=source,
            admin=True,
            layer="runtime",
            refuse=REFUSED_CHECKS,
            exempt=exempt,
        )
        self._overrides[stored] = entry
        if bind is not None and bind[0] != NO_KEY_REF:
            self._remember_key(entry["name"], bind[0])
            self.stack.note(
                source, f"llm_routes.{entry['name']}.endpoint", "", origin(entry["base_url"])
            )
        return {
            "path": box["path"],
            "changed": True,
            "layer": "runtime",
            "restart_required": self._restart(box, entry_rows=box["entry"]),
            "persisted": self._persist(),
        }

    def remove_entry(self, path: object, source: str = SOURCE) -> dict[str, Any]:
        """Remove the entry at ``path`` (``container.key`` or a list entry) if the catalogue lists
        it as ``removable``: only the runtime layer holds it. A lower layer's entry cannot be
        masked (the merge only adds and overrides), so that raises ``entry_in_lower_layer``; an
        unknown entry raises ``SettingNotAllowed``."""
        with self._lock:
            rows, tree = self._rows(), self.stack.merged_tree()
            box_path, _, name = path.rpartition(".") if isinstance(path, str) else ("", "", "")
            box = rows.get(box_path)
            if box is None or box["kind"] not in ENTRY_KINDS or name not in box["value"]:
                raise SettingNotAllowed("path")
            if name not in box["removable"]:
                raise ValidationError("entry_in_lower_layer", "path")
            key = okey(tree, path)
            before = self._failures()
            self.stack.clear(path, source=source, subtree=True, admin=True, refuse=REFUSED_CHECKS)
            for stored in [k for k in self._overrides if k == key or k.startswith(key + ".")]:
                del self._overrides[stored]
            for stored in [k for k in self._acks if k == key or k.startswith(key + ".")]:
                del self._acks[stored]
            if box_path == "llm_routes":
                self._keys.pop(name, None)
            result = {
                "path": path,
                "changed": True,
                "layer": "runtime",
                "restart_required": self._restart(box, entry_rows=box["entry"]),
                "persisted": self._persist(),
            }
            return self._applied(result, before)

    # ------------------------------------------------------------------ route keys (SPEC 8, 11A)

    def add_route(
        self, value: object, secret: object = None, ack: object = None, source: str = SOURCE
    ) -> dict[str, Any]:
        """Add an LLM route (``value``: name, base_url, model, locality, ...) on
        the runtime layer, with the key ``secret`` (a name in the secret store) or none. A route
        with a key needs ``ack`` equal to the origin of its ``base_url`` (``endpoint_ack_required``
        carries it): the stack's ``secret`` rule is passed for that one ``api_key`` only, the
        locality and reflex rules stay refused. A key typed into ``value`` is never accepted."""
        with self._lock:
            ref = self._key_ref(secret)
            name = value.get("name") if isinstance(value, dict) else None
            if isinstance(name, str) and name.isdigit():  # the catalogue would read it as an index
                raise ValidationError("key_invalid", "name")
            before = self._failures()
            result = self._add_entry("llm_routes", None, value, source, bind=(ref, ack))
            return self._applied(result, before)

    def set_route_key(
        self, name: object, secret: object, ack: object = None, source: str = SOURCE
    ) -> dict[str, Any]:
        """Point route ``name`` at the stored secret ``secret`` (``None`` for no key). Unlike a
        field edit, a key other than none needs ``ack`` equal to the origin of the route's
        ``base_url``: that is where the key will be sent. Raises ``SettingNotAllowed`` (unknown
        route), ``ValidationError`` (``secret_name_invalid``, ``secret_unknown``,
        ``endpoint_ack_required``) and the stack's refusals. The value is kept on the runtime layer
        and in the overrides file (``keys``) with the acknowledgement that replays it."""
        with self._lock:
            route = next((r for r in self.stack.effective().llm_routes if r.name == name), None)
            if route is None:
                raise SettingNotAllowed("path")
            ref = self._key_ref(secret)
            if ref == route.api_key:
                return {"name": route.name, "changed": False}
            path = f"llm_routes.{route.name}.api_key"
            if ref != NO_KEY_REF:
                self._check_endpoint_ack(route.base_url, ack)
            before = self._failures()
            self.stack.set(
                path,
                ref,
                source=source,
                admin=True,
                must_exist=True,
                refuse=REFUSED_CHECKS,
                exempt=frozenset({path}) if ref != NO_KEY_REF else frozenset(),
            )
            self._remember_key(route.name, ref)
            if ref != NO_KEY_REF:
                self.stack.note(
                    source, f"llm_routes.{route.name}.endpoint", "", origin(route.base_url)
                )
            result = {"name": route.name, "changed": True, "persisted": self._persist()}
            return self._applied(result, before)

    def removable_routes(self) -> list[str]:
        """Names of the routes only the runtime layer holds (``remove_entry`` accepts them)."""
        row = self._rows().get("llm_routes")
        return list(row["removable"] or []) if row else []

    def _key_ref(self, secret: object) -> str:
        """``none`` for no secret, else ``${secret:name}`` for a name in the secret store; any
        other text (a key typed in) is refused and never echoed."""
        if secret is None:
            return NO_KEY_REF
        cap = self.stack.effective().webui.secret_name_max_chars
        if not isinstance(secret, str) or len(secret) > cap or SECRET_NAME.match(secret) is None:
            raise ValidationError("secret_name_invalid", "secret")
        if self._known_secrets is None or secret not in set(self._known_secrets()):
            raise ValidationError("secret_unknown", "secret")
        return f"${{secret:{secret}}}"

    @staticmethod
    def _check_endpoint_ack(base_url: object, ack: object) -> None:
        """The key goes to the origin of ``base_url``; ``ack`` must be exactly that origin."""
        target = origin(base_url) if isinstance(base_url, str) else ""
        if not target:
            raise ValidationError("value_type", "value")
        if ack != target:
            raise ValidationError("endpoint_ack_required", "value", {"endpoint": target})

    def _remember_key(self, name: str, ref: str) -> None:
        """Keep the key bound to route ``name`` for the next start, with the acknowledged origin and
        a salted fingerprint of the reference in ``acks`` under the route's ``api_key`` file key (a
        record apart from the one of a route move, which names the key the move was made with).
        ``_moved_endpoint`` reads both, so later ``base_url`` edits at that origin keep working."""
        route = next(r for r in self.stack.effective().llm_routes if r.name == name)
        ack_key = okey(self.stack.merged_tree(), f"llm_routes.{name}.api_key")
        self._keys[name] = ref
        if ref == NO_KEY_REF:
            self._acks.pop(ack_key, None)
        else:
            self._acks[ack_key] = {"origin": origin(route.base_url), "key": fingerprint(ref)}

    def _restore_keys(self, keys: dict[str, str], acks: dict[str, dict[str, str]]) -> None:
        """Replay the keys bound from the web, after the other edits. A key other than none is set
        past the secret rule (for that route alone) only when the stored acknowledgement still
        names the route's origin and the fingerprint matches the reference; anything else is
        dropped with a warning that names the route only."""
        self._keys = {}
        for name, ref in keys.items():
            route = next((r for r in self.stack.effective().llm_routes if r.name == name), None)
            path = f"llm_routes.{name}.api_key"
            ack_key = okey(self.stack.merged_tree(), f"llm_routes.{name}.api_key")
            record = acks.get(ack_key)
            if route is None:
                log.warning("web UI key of route %s dropped: no such route", name)
                continue
            if ref != NO_KEY_REF and not (
                record
                and record["origin"] == origin(route.base_url)
                and key_matches(record["key"], ref)
            ):
                log.warning("web UI key of route %s dropped: its endpoint changed", name)
                continue
            try:
                self.stack.set(
                    path,
                    ref,
                    source=RESTORE_SOURCE,
                    admin=True,
                    must_exist=True,
                    refuse=REFUSED_CHECKS,
                    exempt=frozenset({path}) if ref != NO_KEY_REF else frozenset(),
                )
            except (GuardrailError, ConfigError):
                log.warning("web UI key of route %s dropped: no longer acceptable", name)
                continue
            self._keys[name] = ref
            if ref != NO_KEY_REF and record:
                self._acks[ack_key] = record

    def reset(self, path: object, source: str = SOURCE) -> dict[str, Any]:
        """Drop the runtime value of the leaf ``path`` so the lower layer shows again, and its
        entry in the overrides file. A value of the app layer is not touched. ``changed`` is False
        when there was no runtime value; a field inside an entry added at runtime cannot be reset
        (``not_resettable``)."""
        with self._lock:
            rows = self._rows()
            row = rows.get(path) if isinstance(path, str) else None
            if row is None or row["kind"] in CONTAINERS or not row["writable"]:
                raise SettingNotAllowed("path")
            key = okey(self.stack.merged_tree(), path)
            if self._added_parent(key) is not None:  # an added entry's field has no lower value
                raise ValidationError("not_resettable", "path")
            before = self._failures()
            changed = self.stack.clear(path, source=source, admin=True, refuse=REFUSED_CHECKS)
            stale = self._overrides.pop(key, None) is not None
            self._acks.pop(key, None)
            result = {
                "path": path,
                "changed": changed,
                "value": self._shown(get_path(self.stack.effective(), path)),
                "layer": "runtime",
                "restart_required": self._restart(row),
                "persisted": self._persist() if changed or stale else True,
            }
            return self._applied(result, before)

    def restore(self) -> None:
        """Replay the overrides file at startup as one batch: every entry is cleaned against the
        configuration the whole file would give and the stack validates the final state once, so a
        rule that spans fields does not depend on their order.

        An unsafe or malformed file is ignored. An entry that is not writable, no longer valid,
        refused by a guardrail or for a language that is gone is dropped; warnings name the path
        only. If the batch is refused as a whole, the entries are tried one by one. The file is
        rewritten without dropped entries.
        """
        if self._file is None:
            return
        try:
            stored, acks, keys = self._file.load()
        except ValueError as exc:
            log.warning("web UI settings file ignored: %s", exc)
            return
        migrated, stripped = _migrated(stored, str(self._file.path))
        with self._lock:
            tree, rows = self.stack.merged_tree(), self._rows()
            planned: dict[str, tuple[str, Any, bool]] = {}  # file key -> (stack path, value, add)
            for key, value in migrated.items():
                found = self._plan(key, value, tree, rows)
                if found is None:
                    log.warning("web UI setting %s dropped: no longer acceptable", _shown(key))
                    continue
                planned[key] = found
            try:
                cfg = self.stack.preview(combine((p, v) for p, v, _ in planned.values()))
            except (ConfigError, ValueError):
                cfg = self.stack.effective()
            cleaned: dict[str, tuple[str, Any]] = {}  # file key -> (stack path, stack value)
            for key, (path, value, add) in planned.items():
                try:
                    cleaned[key] = self._clean_planned(key, path, value, add, rows, cfg, tree)
                except (ValidationError, SettingNotAllowed, GuardrailError):
                    log.warning("web UI setting %s dropped: no longer acceptable", _shown(key))
            self._overrides, self._acks = self._commit_batch(cleaned, acks)
            self._warn_local_routes()
            self._restore_keys(keys, acks)
            if stripped or set(self._overrides) != set(stored) or self._keys != keys:
                self._persist()

    def _warn_local_routes(self) -> None:
        """A route added or moved from the web before the address rule (``this_machine``) that
        points at this machine or a link-local address is kept, so startup never fails on it,
        with a warning naming the route (never the URL)."""
        for key, value in self._overrides.items():
            found = parse_okey(key)
            if found is None or found[0] != "llm_routes":
                continue
            _, ident, rest = found
            url = value.get("base_url") if not rest and isinstance(value, dict) else None
            url = value if rest == "base_url" else url
            if this_machine(url):
                log.warning(
                    "web UI route %s points at this machine or a link-local address: kept, but "
                    "the web UI refuses such an address now; move it to the app config",
                    ident.get("name"),
                )

    # ------------------------------------------------------------------ write internals

    def _rows(self) -> dict[str, FieldDescriptor]:
        return {row["path"]: row for row in self._catalogue(reveal_private=True)}

    def _catalogue(self, **kwargs: Any) -> list[FieldDescriptor]:
        """The field catalogue with ``conversation.language`` as a choice: the languages whose voice
        models are installed (``languages.installed_languages``) plus the current value, so any
        other value is refused by the enum cleaner, on every write path."""
        rows = build_catalogue(self.stack, _writable, **kwargs)
        current = self.stack.effective()
        offered = sorted({*self._installed(current), current.conversation.language})
        return [
            cast(FieldDescriptor, {**row, "kind": "enum", "choices": offered})
            if row["path"] == LANGUAGE_PATH
            else row
            for row in rows
        ]

    def _apply(
        self,
        path: object,
        value: object,
        source: str,
        rows: dict[str, FieldDescriptor],
        cfg: RoboEmotConfig,
        ack: object = None,
    ) -> dict[str, Any]:
        row = rows.get(path) if isinstance(path, str) else None
        if row is None or not isinstance(path, str) or not self._leaf_writable(rows, row):
            raise SettingNotAllowed("path")
        if secret_shaped(value):
            raise ValidationError("url_credentials" if write_only(path) else "value_type", "value")
        hook = hook_for(path)
        clean = clean_hooked(hook, path, value, cfg) if hook else clean_value(row, value, cfg)
        old = plain(get_path(cfg, path))
        if clean == old:
            return {"path": path, "changed": False, "value": old}
        if write_only(path):
            check_web_route_url(clean, old)
        if (
            path in LANGUAGE_DICTS
            and isinstance(clean, dict)
            and set(clean) - set(old or {}) - languages()
        ):
            raise ValidationError("language_unsupported", "value")
        # a runtime dict merges, so a value that would drop a key cannot be expressed there
        if isinstance(clean, dict) and deep_merge(old or {}, clean) != clean:
            raise ValidationError("value_merged", "value")
        tree = self.stack.merged_tree()
        stored = okey(tree, path)
        bound = okey(tree, f"{path.rpartition('.')[0]}.api_key") if write_only(path) else stored
        moved = self._moved_endpoint(
            path, old, clean, cfg, ack, (self._acks.get(stored), self._acks.get(bound))
        )
        layer = self._write(
            path,
            clean,
            source,
            retry_app=self._added_parent(stored) is None,
            exempt=frozenset({f"llm_routes.{moved.route}.api_key"}) if moved else frozenset(),
        )
        if moved and moved.old != moved.new:
            self.stack.note(source, f"{path.rpartition('.')[0]}.endpoint", moved.old, moved.new)
        return {
            "path": path,
            "changed": True,
            "value": clean,
            "layer": layer,
            "restart_required": self._restart(row),
            "okey": stored,
            "acked": {"origin": moved.new, "key": fingerprint(moved.api_key)} if moved else None,
        }

    @staticmethod
    def _moved_endpoint(
        path: str,
        old: Any,
        new: Any,
        cfg: RoboEmotConfig,
        ack: object,
        records: Iterable[dict[str, str] | None],
    ) -> Move | None:
        """The ``Move`` when ``path`` is the ``base_url`` of a route with an API key and the
        endpoint changes, or stays at the origin acknowledged earlier for this very key (any of
        ``records``: the route's move record and its key-binding record, so a later path edit
        there is not refused again); else ``None``. Raises ``endpoint_ack_required`` (carrying
        the new origin) unless ``ack`` is that origin."""
        if not write_only(path) or not isinstance(old, str) or not isinstance(new, str):
            return None
        route = plain(get_path(cfg, path.rpartition(".")[0]))
        target = origin(new)
        acked = (
            target
            if any(
                r and r["origin"] == target and key_matches(r["key"], route["api_key"])
                for r in records
            )
            else None
        )
        same = endpoint_key(old) == endpoint_key(new)
        # a path-only edit at the origin acknowledged for this key still needs the secret exemption:
        # the approved baseline holds the old origin
        if route["api_key"] in NO_KEY or (same and acked != target):
            return None
        if not target:
            raise ValidationError("value_type", "value")
        if ack != target and acked != target:
            raise ValidationError("endpoint_ack_required", "value", {"endpoint": target})
        return Move(origin(old), target, route["name"], route["api_key"])

    @staticmethod
    def _leaf_writable(rows: dict[str, FieldDescriptor], row: FieldDescriptor) -> bool:
        """A non-secret leaf (or a route ``base_url``, replaced whole) that is not the identity of
        a keyed-list entry (renaming one would move it to another entry)."""
        if row["kind"] in CONTAINERS or not row["writable"]:
            return False
        if row["secret"] and not write_only(row["path"]):
            return False
        entry, _, name = row["path"].rpartition(".")
        box = rows.get(entry.rpartition(".")[0])
        return not (
            box is not None
            and box["kind"] == "keyed_list"
            and name in (box["key"], "when" if box["key"] == "event" else box["key"])
        )

    def _write(
        self,
        key: str,
        value: Any,
        source: str,
        *,
        retry_app: bool = True,
        exempt: frozenset[str] = frozenset(),
    ) -> str:
        """Write on the runtime layer; if the stack refuses it only for a safety loosening, write
        it on the app layer instead (app YAML and audit) and drop the runtime value that would
        shadow it. Returns the layer. Every other refusal propagates, and so does that one for a
        field of an entry added at runtime (``retry_app`` False): the app file has no such entry.
        """
        try:
            self.stack.set(
                key,
                value,
                source=source,
                admin=True,
                must_exist=True,
                refuse=REFUSED_CHECKS,
                exempt=exempt,
            )
            return "runtime"
        except GuardrailError as exc:
            if not retry_app or not loosening_only(exc):
                raise
        self.stack.set(
            key,
            value,
            source=source,
            admin=True,
            layer="app",
            must_exist=True,
            refuse=REFUSED_CHECKS,
            exempt=exempt,
            drop_runtime=True,  # one transaction: a refusal leaves the YAML and the runtime as were
        )
        return "app"

    def _remember(self, key: str, value: Any) -> None:
        """Keep a runtime edit for the next start; inside an entry added at runtime it joins the
        entry's own value."""
        parent = self._added_parent(key)
        if parent is None:
            self._overrides[key] = value
            return
        entry = json.loads(json.dumps(self._overrides[parent]))
        set_path(entry, key[len(parent) + 1 :], value)
        self._overrides[parent] = entry

    def _added_parent(self, key: str) -> str | None:
        """The file key of an entry added at runtime that holds ``key``."""
        return next(
            (
                k
                for k, v in self._overrides.items()
                if key.startswith(k + ".") and isinstance(v, dict)
            ),
            None,
        )

    def _rebase_binding(self, path: object, acked: dict[str, str] | None) -> None:
        """After an acknowledged move of a route ``base_url``, the key-binding record of that route
        (if any) names the new origin: the owner acknowledged it for this very key."""
        if not acked or not isinstance(path, str) or not write_only(path):
            return
        parent = path.rpartition(".")[0]
        bound = okey(self.stack.merged_tree(), f"{parent}.api_key")
        if bound in self._acks:
            self._acks[bound] = {**self._acks[bound], "origin": acked["origin"]}

    def _keep_ack(self, key: str, value: Any, acked: dict[str, str] | None) -> None:
        """Record the origin (and key fingerprint) the owner acknowledged for a moved route
        ``base_url``; a later value at another origin forgets it, so a replay never trusts more
        than was confirmed."""
        if acked:
            self._acks[key] = acked
        elif key in self._acks and origin(str(value)) != self._acks[key]["origin"]:
            del self._acks[key]

    def _failures(self) -> int:
        """The stack's count of reload listeners that raised (R3 SF3)."""
        return self.stack.listener_failures

    def _applied(self, result: dict[str, Any], before: int) -> dict[str, Any]:
        """``result`` with ``applied``: False when a reload listener raised during the write, so
        a part of the robot may still run the old value although the config holds the new one.
        A failure on another thread in the same moment also counts: it errs towards a warning."""
        result["applied"] = self._failures() == before
        return result

    def _persist(self) -> bool:
        if self._file is None:
            return True
        try:
            self._file.save(self._overrides, self._acks, self._keys)
        except OSError as exc:
            log.warning("web UI settings not saved (%s)", type(exc).__name__)
            return False
        return True

    @staticmethod
    def _shown(value: Any) -> Any:
        """``value`` as plain JSON for a response; a credential-shaped one never leaves."""
        value = plain(value)
        return REDACTED if secret_shaped(value) else value

    @staticmethod
    def _restart(row: FieldDescriptor, entry_rows: list[FieldDescriptor] | None = None) -> bool:
        """Whether the change needs a restart: the field's ``restart`` flag, for a collection any
        field of its entries."""
        return row["restart"] or any(r["restart"] for r in entry_rows or [])

    # ------------------------------------------------------------------ entries

    def _clean_entry(
        self, box: FieldDescriptor, value: object, cfg: RoboEmotConfig, label: str | None = None
    ) -> dict[str, Any]:
        """An entry for ``box`` cleaned field by field from the blank entry's descriptors, with the
        per-path hook of the field when the entry has a key (``label``: the lines and phrases of a
        new language get the same checks as an existing one). A field the entry type does not
        have, or a secret field with a value, is ``entry_invalid``."""
        if not isinstance(value, dict):
            raise ValidationError("value_type", "value")
        rows = {r["path"]: r for r in box["entry"] or []}
        out: dict[str, Any] = {}

        def walk(node: dict[str, Any], prefix: str, into: dict[str, Any]) -> None:
            for name, item in node.items():
                path = f"{prefix}{name}"
                row = rows.get(path)
                if row is not None and row["kind"] not in CONTAINERS:
                    if row["secret"]:
                        if item not in (None, "", row["default"]):
                            raise ValidationError("entry_invalid", "value")
                        continue
                    full = f"{box['path']}.{label}.{path}" if label else ""
                    hook = hook_for(full) if label else None
                    into[name] = (
                        clean_hooked(hook, full, item, cfg) if hook else clean_value(row, item, cfg)
                    )
                elif isinstance(item, dict) and any(r.startswith(path + ".") for r in rows):
                    into[name] = {}
                    walk(item, path + ".", into[name])
                else:
                    raise ValidationError("entry_invalid", "value")

        walk(value, "", out)
        return out

    def _list_add(
        self, box: FieldDescriptor, entry: dict[str, Any], tree: dict[str, Any], cfg: RoboEmotConfig
    ) -> tuple[str, str, Any]:
        """``(stack path, file key, stack value)`` for a new keyed-list entry. A merged list takes
        just the entry; any other list is replaced by the old entries plus the new one."""
        id_field = box["key"]
        ident = entry.get(id_field)
        clean_key(ident, cfg, dots=id_field == "event")
        path = box["path"]
        current = get_path(tree, path)
        if any(entry_identity(path, e) == entry_identity(path, entry) for e in current):
            raise ValidationError("entry_exists", "value")
        if path in KEYED_LISTS:
            return path, entry_key(path, ident_fields(path, entry)), [entry]
        return path, path, [*current, entry]

    # ------------------------------------------------------------------ restore internals

    def _plan(
        self, key: str, value: Any, tree: dict[str, Any], rows: dict[str, FieldDescriptor]
    ) -> tuple[str, Any, bool] | None:
        """Where a stored edit lands now: ``(stack path, stack value, is an added entry)``, or
        ``None`` when its field, entry or language is gone."""
        if secret_shaped(value):
            return None
        parsed = parse_okey(key)
        base = self.stack.merged_tree(include_runtime=False)
        if parsed is not None:
            list_path, ident, rest = parsed
            if not rest:  # an added entry: its identity is its own, and no lower layer has it
                if not isinstance(value, dict) or value.get(KEYED_LISTS[list_path]) is None:
                    return None
                if (
                    ident != ident_fields(list_path, value)
                    or keyed_index(base, list_path, ident) is not None
                ):
                    return None
                return (list_path, [value], True)
            at = keyed_index(tree, list_path, ident)
            if at is None:
                return None
            name = ident.get(KEYED_LISTS[list_path])
            for segment in (name, str(at)):  # the catalogue names an entry by its name when it can
                path = f"{list_path}.{segment}.{rest}"
                if isinstance(segment, str) and path in rows:
                    return path, value, False
            return None
        parent = rows.get(key.rpartition(".")[0])
        if parent is not None and parent["kind"] == "dict_of_model" and isinstance(value, dict):
            absent = key.rpartition(".")[2] not in get_path(base, parent["path"])
            return (key, value, True) if absent else None
        return (key, value, False) if key in rows else None

    def _clean_planned(
        self,
        key: str,
        path: str,
        value: Any,
        add: bool,
        rows: dict[str, FieldDescriptor],
        cfg: RoboEmotConfig,
        tree: dict[str, Any],
    ) -> tuple[str, Any]:
        """Clean one planned edit with the live-edit rules, against ``cfg`` (the whole file
        applied). Returns the stack path and value."""
        if add:
            parsed = parse_okey(key)
            box = rows.get(parsed[0] if parsed else key.rpartition(".")[0])
            if box is None or box["kind"] not in ENTRY_KINDS:
                raise SettingNotAllowed("path")
            language = key.rpartition(".")[2]
            if not parsed and box["path"] in LANGUAGE_MAPS and language not in languages():
                raise SettingNotAllowed("path")
            entry = self._clean_entry(
                box, value[0] if parsed else value, cfg, None if parsed else language
            )
            return path, ([entry] if parsed else entry)
        row = rows.get(path)
        if row is None or not self._leaf_writable(rows, row):
            raise SettingNotAllowed("path")
        hook = hook_for(path)
        return path, clean_hooked(hook, path, value, cfg) if hook else clean_value(row, value, cfg)

    def _commit_batch(
        self, cleaned: dict[str, tuple[str, Any]], acks: dict[str, dict[str, str]]
    ) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
        """Apply the cleaned edits as one change; if the stack refuses the whole, one by one.
        A route ``base_url`` whose origin and key still equal the ones acknowledged when it was
        saved (``acks``) is replayed past the secret check of that route alone. Returns the file
        entries that were applied and the acks that still hold."""
        if not cleaned:
            return {}, {}

        def file_value(key: str) -> Any:
            parsed = parse_okey(key)
            value = cleaned[key][1]
            return value[0] if parsed and not parsed[2] else value

        def route_of(path: str) -> dict[str, Any] | None:
            if not write_only(path):
                return None
            try:
                return plain(get_path(self.stack.effective(), path.rpartition(".")[0]))
            except (KeyError, IndexError, ValueError, TypeError):
                return None

        def holds(key: str) -> dict[str, Any] | None:
            """The route of ``key`` when its ack matches the replayed origin and the current key."""
            path, value = cleaned[key]
            route, record = route_of(path), acks.get(key)
            if route and record and record["origin"] == origin(str(value)):
                return route if key_matches(record["key"], route["api_key"]) else None
            return None

        def standing(kept: dict[str, Any]) -> dict[str, dict[str, str]]:
            return {k: acks[k] for k in kept if k in acks and holds(k)}

        try:
            self.stack.set_many(
                combine(cleaned.values()),
                source=RESTORE_SOURCE,
                admin=True,
                refuse=REFUSED_CHECKS,
            )
            done = {k: file_value(k) for k in cleaned}
            return done, standing(done)
        except (GuardrailError, ConfigError):
            pass
        kept: dict[str, Any] = {}
        for key, (path, value) in cleaned.items():
            route = holds(key)
            try:
                self.stack.set_many(
                    {path: value},
                    source=RESTORE_SOURCE,
                    admin=True,
                    refuse=REFUSED_CHECKS,
                    exempt=frozenset({f"llm_routes.{route['name']}.api_key"})
                    if route
                    else frozenset(),
                )
            except (GuardrailError, ConfigError):
                log.warning("web UI setting %s dropped: no longer acceptable", _shown(key))
                continue
            kept[key] = file_value(key)
        return kept, standing(kept)


def combine(edits: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    """``{stack path: value}`` of a batch; entries added to one merged list share its path, so
    their lists are joined rather than the last one winning."""
    out: dict[str, Any] = {}
    for path, value in edits:
        if path in KEYED_LISTS and isinstance(value, list):
            out.setdefault(path, []).extend(value)
        else:
            out[path] = value
    return out


def _migrated(stored: dict[str, Any], source: str) -> tuple[dict[str, Any], bool]:
    """The stored overrides with renamed settings under their new path (a value already stored
    under the new path wins) and removed ones dropped with a log line naming the key only. A
    dropped setting (``DROPPED_SETTINGS``) is taken out of a stored value too, e.g. an added
    route, and logged once under ``source`` (the file); the flag says whether a value changed
    that way, so the file is rewritten."""
    out: dict[str, Any] = {}
    stripped = False
    for key, value in stored.items():
        if any(key == old or key.startswith(old + ".") for old in REMOVED_SETTINGS):
            log.warning("web UI setting %s dropped: removed from the configuration", _shown(key))
            continue
        parsed = parse_okey(key)
        if parsed is not None and dropped_entry(parsed[0], parsed[1]):
            log_dropped(source, [f"{parsed[0]}[{parsed[1][KEYED_LISTS[parsed[0]]]}]"])
            stripped = True
            continue
        value, found = without_dropped(_pattern_path(key), value)
        log_dropped(source, found)
        if value is DROP:
            continue
        stripped = stripped or bool(found)
        new = RENAMED_SETTINGS.get(key, key)
        if new != key and new in stored:
            continue
        out[new] = value
    return out, stripped


def _pattern_path(key: str) -> str:
    """A file key as a stack path, a merged-list entry's identity as ``*``."""
    parsed = parse_okey(key)
    if parsed is None:
        return key
    list_path, _, rest = parsed
    return f"{list_path}.*" + (f".{rest}" if rest else "")


def _shown(key: str) -> str:
    """A file key for a log line: the path only (a merged-list entry shows its list)."""
    return key.split("[", 1)[0]
