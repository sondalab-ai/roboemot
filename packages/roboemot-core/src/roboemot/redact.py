"""What counts as a secret, shared by the debug report and the web UI field catalogue (SPEC 8, 12).

Two rules, both fail-closed: a key name that suggests a credential (``is_secret_key``) and a value
that looks like one whatever its key (``secret_shaped``): a ``${secret:name}`` reference, a URL
with user info, or a URL whose query or fragment carries a credential-named parameter.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit

REDACTED = "***"
SECRET_WORDS = (
    "key",
    "token",
    "secret",
    "password",
    "credential",
    "auth",
    "bearer",
    "cookie",
    "session",
)
SECRET_REF = re.compile(r"\$\{secret:")
URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+")
URL_USER_INFO = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^/?#\s@]*@")


def is_secret_key(key: str) -> bool:
    """Whether a field or dict key name marks its value as a secret (substring match)."""
    k = key.lower()
    return any(w in k for w in SECRET_WORDS)


def _url_secret(url: str) -> bool:
    """A URL with user info, or a query or fragment parameter named like a credential."""
    if URL_USER_INFO.match(url):
        return True
    try:
        parts = urlsplit(url)
        params = parse_qsl(parts.query, keep_blank_values=True)
        params += parse_qsl(parts.fragment, keep_blank_values=True)
    except ValueError:
        return True
    return any(is_secret_key(name) for name, _ in params)


def secret_shaped(value: Any) -> bool:
    """Whether a value looks like a secret whatever its key. Looks inside lists, tuples and dicts
    (values and keys): one secret-shaped element makes the whole value secret."""
    if isinstance(value, str):
        return bool(SECRET_REF.search(value)) or any(_url_secret(m) for m in URL.findall(value))
    if isinstance(value, dict):
        return any(secret_shaped(k) or secret_shaped(v) for k, v in value.items())
    if isinstance(value, list | tuple | set | frozenset):
        return any(secret_shaped(v) for v in value)
    return False
