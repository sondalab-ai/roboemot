"""Write-only secret entry for the web UI (SPEC 8 Secrets, 12 Data).

Lists secret names (never values, lengths or prefixes) with the routes that use each, sets or
replaces a value and removes a name. The PIN is the caller's job (``WebApi``). A value is
checked here (printable ASCII, no spaces, within ``MAX_VALUE_CHARS``: it ends up in an
``Authorization`` header) and handed to the store; it is never returned, logged, put in an
exception message or kept. A secret a route uses cannot be removed.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from roboemot.config.schema import RoboEmotConfig
from roboemot.secret_store import MAX_VALUE_CHARS, NAME, SECRET_REF, SecretStore

from .validation import ValidationError

VALUE_PATTERN = re.compile(r"[\x21-\x7e]+")  # printable ASCII, no space: a header-safe token


class SecretInUse(RuntimeError):
    """The secret is the key of a route; ``routes`` names them (no value)."""

    def __init__(self, routes: list[str]) -> None:
        super().__init__("secret in use")
        self.routes = routes


def secret_name(value: object, field: str = "name") -> str:
    """``value`` as a store name, else ``ValidationError("secret_name_invalid")``."""
    if not isinstance(value, str) or NAME.match(value) is None:
        raise ValidationError("secret_name_invalid", field)
    return value


def secret_value(value: object) -> str:
    """The value to store: surrounding whitespace (a pasted newline) dropped, then 1 to
    ``MAX_VALUE_CHARS`` printable ASCII characters without spaces. The error never echoes it."""
    if not isinstance(value, str):
        raise ValidationError("secret_value_invalid", "value")
    text = value.strip()
    if not text or len(text) > MAX_VALUE_CHARS or VALUE_PATTERN.fullmatch(text) is None:
        raise ValidationError("secret_value_invalid", "value")
    return text


class SecretsAdmin:
    """What the web UI needs from the secret store: ``listing``, ``names``, ``set``, ``remove``.

    ``cfg`` returns the effective config (routes follow a hot reload, and ``data_dir`` says where
    the store is). Store errors (``SecretUnavailable``) propagate; their messages name a path
    or a secret, never a value, and the API maps them to a bare code.
    """

    def __init__(self, cfg: Callable[[], RoboEmotConfig], store: SecretStore | None = None) -> None:
        self._cfg, self._store = cfg, store

    @property
    def store(self) -> SecretStore:
        return self._store or SecretStore.for_data_dir(self._cfg().data_dir)

    def names(self) -> list[str]:
        """Sorted names in the store."""
        return self.store.names()

    def used_by(self) -> dict[str, list[str]]:
        """Route names by the secret name their ``api_key`` references."""
        out: dict[str, list[str]] = {}
        for route in self._cfg().llm_routes:
            ref = SECRET_REF.match(route.api_key)
            if ref:
                out.setdefault(ref.group(1), []).append(route.name)
        return out

    def listing(self) -> list[dict[str, object]]:
        """``[{"name", "used_by": [route names]}]`` for every name in the store, sorted. A name a
        route references but the store lacks is listed with ``set`` false."""
        used = self.used_by()
        have = self.names()
        return [
            {"name": n, "set": n in have, "used_by": used.get(n, [])}
            for n in sorted({*have, *used})
        ]

    def set(self, name: str, value: str) -> None:
        """Add or replace ``name`` (already validated by ``secret_name`` / ``secret_value``)."""
        self.store.set(name, value)

    def remove(self, name: str) -> bool:
        """Delete ``name``; True when it was there. Raises :class:`SecretInUse` while a route
        uses it (nothing is deleted)."""
        routes = self.used_by().get(name, [])
        if routes:
            raise SecretInUse(routes)
        return self.store.remove(name)
