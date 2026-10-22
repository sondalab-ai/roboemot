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
SCHEME_RELATIVE_INFO = re.compile(r"(?<![:\w/])//[^/?#\s@]*@")
USERNAME = re.compile(r"[a-z_][a-z0-9_.-]{0,19}", re.IGNORECASE)
USERNAME_MIXED_MAX = 16  # a longer name of letters and digits looks like a generated token
TOKEN_PREFIXES = (
    "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_", "glpat-", "xox", "sk-", "sk_live_",
    "sk_test_", "hf_", "npm_", "pypi-", "akia", "ya29.",
)  # fmt: skip


PRIVATE_PATHS = (
    "circadian.location",
    "privacy",
    "safety.minor_motion_factor",
    "safety.minor_hold_s",
    "safety.minor_hold_periods",
    "safety.minor_output_blocklist",
    "safety.minor_lines",
    "register.child.minor_content_policy",
    "register.teen.minor_content_policy",
    "register.adult.minor_content_policy",
    "register.senior.minor_content_policy",
    "register.unknown.minor_content_policy",
    "reflexes.rules.*.suppress_with_minor",
)
PATH_WILDCARD = "*"


def is_private(path: str) -> bool:
    """Whether ``path`` is one of ``PRIVATE_PATHS`` or inside one (a ``*`` matches one segment)."""
    parts = path.split(".")
    for pattern in PRIVATE_PATHS:
        want = pattern.split(".")
        if len(parts) >= len(want) and all(
            w in (PATH_WILDCARD, p) for w, p in zip(want, parts, strict=False)
        ):
            return True
    return False


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
    """A URL with user info anywhere in it (also one nested after a comma or in a query value), or
    a query or fragment parameter named like a credential. ``bare_user`` False lets one plain user
    name without a password through (``ssh://user@host``) when the text holds exactly one URL and
    one ``@``; any other user info, and any that is hard to parse, stays secret."""
    has_info = "@" in urlsplit_netloc(url) or URL_USER_INFO.search(url)
    if has_info and (
        bare_user or url.count("://") != 1 or url.count("@") != 1 or not _bare_user_only(url)
    ):
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


def _plain_user(name: str) -> bool:
    """A short user name that is no known token prefix and no long mix of letters and digits."""
    mixed = len(name) > USERNAME_MIXED_MAX and any(c.isdigit() for c in name)
    return (
        bool(USERNAME.fullmatch(name)) and not mixed and not name.lower().startswith(TOKEN_PREFIXES)
    )


def mask_bare_users(text: str) -> str:
    """``text`` with the single user name of a URL replaced by ``***`` unless it is a plain name
    (``_plain_user``): ``https://ghp_...@github.com`` -> ``https://***@github.com``, so the host
    stays visible. Other user info (several URLs or ``@``, a password) is left for
    ``secret_shaped`` to reject."""

    def one(m: re.Match[str]) -> str:
        url = m.group(0)
        if url.count("://") != 1 or url.count("@") != 1 or not _bare_user_only(url):
            return url
        info = urlsplit_netloc(url).rpartition("@")[0]
        if not info:
            return url
        return url if _plain_user(info) else url.replace("://" + info + "@", "://***@", 1)

    return URL.sub(one, text)


def secret_shaped(value: Any, *, bare_user: bool = True) -> bool:
    """Whether a value looks like a secret whatever its key. Looks inside lists, tuples and dicts
    (values and keys): one secret-shaped element makes the whole value secret. The debug report
    passes ``bare_user=False``: a user name without a password is not hidden there."""
    if isinstance(value, str):
        return (
            bool(SECRET_REF.search(value))
            or bool(SCHEME_RELATIVE_INFO.search(value))
            or any(_url_secret(m, bare_user) for m in URL.findall(value))
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
    if ":" in host:
        host = f"[{host}]"  # an IPv6 literal keeps its brackets, so the port stays unambiguous
    return f"{host}:{port}" if port else host


def origin(url: str) -> str:
    """``scheme://host[:port]`` of a URL, lower case, no credentials, path or query ('' when
    unparsable): where a request would go."""
    try:
        scheme = urlsplit(url).scheme.lower()
    except ValueError:
        return ""
    where = endpoint(url)
    return f"{scheme}://{where}" if scheme and where else ""


def scrub(key: str, value: Any) -> Any:
    """``value`` with whatever is secret replaced by ``REDACTED``, for an audit entry or a message:
    a secret-shaped value, any value at a private path (``is_private``: minors, privacy, location;
    the audit file still says that the path changed), a string under a credential-named key (the
    last segment of ``key``, or an inner dict key). Anything else comes back unchanged, so a
    normal value stays comparable."""
    if secret_shaped(value) or (value is not None and is_private(key)):
        return REDACTED
    leaf = key.rsplit(".", 1)[-1]
    if isinstance(value, str):
        return REDACTED if value and is_secret_key(leaf) else value
    if isinstance(value, dict):
        return {k: scrub(f"{key}.{k}" if key else str(k), v) for k, v in value.items()}
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
    text = SCHEME_RELATIVE_INFO.sub(REDACTED + "@", text)
    return URL.sub(lambda m: REDACTED if _url_secret(m.group(0)) else m.group(0), text)
