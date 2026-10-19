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
USERNAME = re.compile(r"[a-z_][a-z0-9_.-]{0,31}", re.IGNORECASE)
TOKEN_PREFIXES = ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_", "glpat-", "xox", "sk-")


def is_secret_key(key: str) -> bool:
    """Whether a field or dict key name marks its value as a secret (substring match)."""
    k = key.lower()
    return any(w in k for w in SECRET_WORDS)


def _bare_user_only(url: str) -> bool:
    """Whether the user info of ``url`` is one plain name: parsed by ``urlsplit`` (not by a
    prefix match, which could disagree with the HTTP client), with no password, ``@``, ``%`` or
    ``:`` in it. A URL that does not parse cleanly is not."""
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018  (raises ValueError on a bad port)
        if parts.password is not None:
            return False
    except ValueError:
        return False
    info = parts.netloc.rpartition("@")[0]
    return not any(c in info for c in "@%:")


def _url_secret(url: str, bare_user: bool = True) -> bool:
    """A URL with user info, or a query or fragment parameter named like a credential.
    ``bare_user`` False lets a user name without a password through (``ssh://user@host``); any
    other user info, and any that is hard to parse, stays secret."""
    has_info = "@" in urlsplit_netloc(url) or URL_USER_INFO.match(url)
    if has_info and (bare_user or not _bare_user_only(url)):
        return True
    try:
        parts = urlsplit(url)
        params = parse_qsl(parts.query, keep_blank_values=True)
        params += parse_qsl(parts.fragment, keep_blank_values=True)
    except ValueError:
        return True
    return any(is_secret_key(name) for name, _ in params)


def urlsplit_netloc(url: str) -> str:
    """The authority of ``url`` as ``urlsplit`` sees it ('' when it cannot be parsed)."""
    try:
        return urlsplit(url).netloc
    except ValueError:
        return ""


def mask_bare_users(text: str) -> str:
    """``text`` with the single user name of a URL replaced by ``***`` unless it is a short plain
    name that is no known token prefix (``https://ghp_...@github.com`` -> ``https://***@github.com``),
    so the host stays visible. Other user info is left for ``secret_shaped`` to reject."""

    def one(m: re.Match[str]) -> str:
        url = m.group(0)
        if not _bare_user_only(url) or "@" not in urlsplit_netloc(url):
            return url
        info = urlsplit_netloc(url).rpartition("@")[0]
        plain = USERNAME.fullmatch(info) and not info.lower().startswith(TOKEN_PREFIXES)
        return url if plain else url.replace("://" + info + "@", "://***@", 1)

    return URL.sub(one, text)


def secret_shaped(value: Any, *, bare_user: bool = True) -> bool:
    """Whether a value looks like a secret whatever its key. Looks inside lists, tuples and dicts
    (values and keys): one secret-shaped element makes the whole value secret. The debug report
    passes ``bare_user=False``: a user name without a password is not hidden there."""
    if isinstance(value, str):
        return bool(SECRET_REF.search(value)) or any(
            _url_secret(m, bare_user) for m in URL.findall(value)
        )
    if isinstance(value, dict):
        return any(
            secret_shaped(k, bare_user=bare_user) or secret_shaped(v, bare_user=bare_user)
            for k, v in value.items()
        )
    if isinstance(value, list | tuple | set | frozenset):
        return any(secret_shaped(v, bare_user=bare_user) for v in value)
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
