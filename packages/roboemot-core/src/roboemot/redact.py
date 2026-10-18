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
SECRET_REF_FULL = re.compile(r"\$\{secret:[^}]*\}?")
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


def endpoint(url: str) -> str:
    """``host`` or ``host:port`` of a URL: no credentials, path or query ('' when unparsable)."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return ""
    host = parts.hostname or ""
    return f"{host}:{port}" if port else host


def scrub(key: str, value: Any) -> Any:
    """``value`` with whatever is secret replaced by ``REDACTED``, for an audit entry or a message:
    a secret-shaped value, a string under a credential-named key (the last segment of ``key``, or
    an inner dict key). Anything else comes back unchanged, so a normal value stays comparable."""
    if secret_shaped(value):
        return REDACTED
    leaf = key.rsplit(".", 1)[-1]
    if isinstance(value, str):
        return REDACTED if value and is_secret_key(leaf) else value
    if isinstance(value, dict):
        return {k: scrub(str(k), v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [scrub(key, v) for v in value]
    return value


def shown(key: str, value: Any) -> str:
    """Text for ``value`` in an error message or log line: ``repr`` of ``scrub``, and a
    ``base_url`` shows its host only."""
    if key.rsplit(".", 1)[-1] == "base_url" and isinstance(value, str):
        return repr(endpoint(value))
    clean = scrub(key, value)
    return repr(scrub_text(clean) if isinstance(clean, str) else clean)


def scrub_text(text: str) -> str:
    """``text`` with secret references and credential-bearing URLs replaced by ``REDACTED``."""
    text = SECRET_REF_FULL.sub(REDACTED, text)
    return URL.sub(lambda m: REDACTED if _url_secret(m.group(0)) else m.group(0), text)
