"""Pseudonyms for routes that are not local (SPEC 8, 12 Data; ``privacy.pseudonyms_to_cloud``).

The registry's display names are swapped for neutral aliases ("Person-A", "Persona-A") in
everything a cloud route receives, and swapped back in what it sends back:

- out: the system prompt (profile and scene summaries), the history of past turns (the user's own
  words included), the tool schemas (the ``look_at`` enum) and the tool calls and results of
  earlier rounds (their JSON is parsed, the strings aliased and the JSON written again without
  ``\\u`` escapes);
- back: the streamed reply text, before the sentence splitter and the speech, and the arguments
  of every tool call, before the executor runs. The history kept in RAM holds the real names.

Matching. Text is NFC-normalised, then compared with case folding, accents and other combining
marks removed (``Ömer``, ``omer`` and an NFD ``Ömer`` are one name; ``Strauß`` matches
``STRAUSS``; ``İlker`` matches ``ilker``), as whole words, with any run of spaces between the
words of a name. The JSON-escaped spellings of a name are matched too. A check over the finished
request, in both JSON spellings (``ensure_ascii`` on and off), refuses to send a body that still
holds a name (``Pseudonyms.leaks``).

Restoring. Only the aliases that the outgoing request really contained are restored, in their
exact spelling (case-sensitive, whole token), and never an alias that already appeared literally
in the text the router was given (the user said "Person-A" aloud): the model cannot use the map
to ask about names, and ordinary words ("persona a cui") are never turned into names. The map
is rebuilt from the registry before every request, so a forgotten or renamed record is dropped
and can no longer be restored.

The map lives in RAM, is never written or logged (this module logs nothing). A label (A, B, ...
Z, AA, ...) is handed out the first time a name is seen, in random order, so it says nothing
about the person; labels are stable for the process and reassigned after a restart. Limits:
nicknames, a first name when only the full name is registered, inflections, misspellings and
personal details that are not registry names are not caught; a name that is also a common word is
replaced wherever it appears.
"""

from __future__ import annotations

import copy
import functools
import ipaddress
import json
import random
import re
import threading
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

import httpx

from ..config.schema import PSEUDONYM_LABEL, LLMRoute

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOOPBACK_NAMES = ("localhost",)
MDNS_SUFFIX = ".local"
NUMERIC_HOST = re.compile(r"^(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+))*$", re.IGNORECASE)
DOTTED_QUAD = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


LAN_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "127.0.0.0/8",  # loopback
        "::1/128",
        "10.0.0.0/8",  # RFC 1918
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",  # link-local
        "fe80::/10",
        "fc00::/7",  # unique local
    )
)


def route_is_local(route: LLMRoute) -> bool:
    """Whether ``route`` is trusted to see real names (SPEC 8, 12). ``locality`` says
    ``local`` or ``cloud``; ``auto`` (the default) means this machine and the home network:
    ``localhost``, an address in an explicit allowlist (loopback, RFC 1918, ``fc00::/7``,
    ``fe80::/10``, 169.254/16), a ``*.local`` name or a name without a dot. Other addresses
    (6to4, 198.18/15, 240/4, carrier-grade NAT, IPv4-mapped IPv6) and every other name are
    cloud. The host says nothing about where a gateway forwards the data: a LAN proxy to a cloud
    provider (LiteLLM) needs ``locality: cloud``. A host written as a number (``134744072``,
    ``0x7f.1``) or a URL whose parts parsers read differently (backslash, user info) is cloud
    whatever ``locality`` says. The host is parsed by httpx, the library that connects."""
    url = route.base_url
    if "\\" in url or any(c.isspace() or ord(c) < ord(" ") for c in url):
        return False
    try:
        parsed = httpx.URL(url)
        host = parsed.host.lower().rstrip(".")
        userinfo = parsed.userinfo
    except (httpx.InvalidURL, ValueError):
        return False
    if not host or userinfo:
        return False
    if NUMERIC_HOST.match(host) and not DOTTED_QUAD.match(host):
        return False
    if route.locality != "auto":
        return route.locality == "local"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host in LOOPBACK_NAMES or host.endswith(MDNS_SUFFIX) or "." not in host
    return any(ip in net for net in LAN_NETWORKS if net.version == ip.version)


def _label(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (spreadsheet columns)."""
    out = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, len(LETTERS))
        out = LETTERS[rem] + out
    return out


@functools.lru_cache(maxsize=8192)
def _fold_char(c: str) -> str:
    """Case-folded character without accents or other combining marks (may be several
    characters: ß gives ss, İ gives i)."""
    decomposed = unicodedata.normalize("NFKD", c.casefold())
    return "".join(x for x in decomposed if unicodedata.category(x) != "Mn")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _fold(text: str) -> tuple[str, list[int], list[int], str]:
    """``(folded, start, end, nfc)``: the folded text and, for each of its characters, the
    span of the NFC character it came from."""
    nfc = _nfc(text)
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, c in enumerate(nfc):
        for x in _fold_char(c):
            chars.append(x)
            starts.append(i)
            ends.append(i + 1)
    return "".join(chars), starts, ends, nfc


def _key(text: str) -> str:
    """The comparison form of a name: folded, spaces collapsed."""
    return " ".join(_fold(text)[0].split())


def _variants(name: str) -> set[str]:
    """The spellings of ``name`` in a JSON string, with and without ``\\u`` escapes."""
    out = {name}
    for ascii_only in (False, True):
        out.add(json.dumps(name, ensure_ascii=ascii_only)[1:-1])
    return out


def _compile(keys: Iterable[str]) -> re.Pattern[str] | None:
    """One pattern for folded names (longest first): whole word, spaces flexible."""
    parts = []
    for key in sorted({k for k in keys if k}, key=len, reverse=True):
        body = r"\s+".join(re.escape(w) for w in key.split())
        before = r"(?<!\w)" if re.match(r"\w", key) else ""
        after = r"(?!\w)" if re.search(r"\w$", key) else ""
        parts.append(before + body + after)
    return re.compile("|".join(parts)) if parts else None


def _spans(pattern: re.Pattern[str] | None, text: str) -> list[tuple[int, int, str]]:
    """``(start, end, key)`` of each name in ``text`` (the NFC text's indices)."""
    if pattern is None or not text:
        return []
    folded, starts, ends, _ = _fold(text)
    return [
        (starts[m.start()], ends[m.end() - 1], " ".join(m.group(0).split()))
        for m in pattern.finditer(folded)
        if m.end() > m.start()
    ]


def replace_names(text: str, names: Mapping[str, str]) -> str:
    """``text`` with each key of ``names`` replaced by its value. A fixed map, for callers that
    bring their own aliases."""
    table = {_key(v): alias for n, alias in names.items() if n.strip() for v in _variants(n)}
    pattern = _compile(table)
    nfc = _nfc(text)
    out, pos = [], 0
    for start, end, key in _spans(pattern, nfc):
        out += [nfc[pos:start], table[key]]
        pos = end
    return "".join(out) + nfc[pos:]


class Restoration:
    """The aliases one request really carried and what they stand for. Case-sensitive, whole
    token; a held-back tail in a stream is kept by :class:`Restorer`."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self.mapping = dict(mapping)
        ordered = sorted(self.mapping, key=len, reverse=True)
        self._pattern = (
            re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in ordered) + r")(?!\w)")
            if ordered
            else None
        )

    def text(self, text: str) -> str:
        """``text`` with each alias of the request turned into its name."""
        if self._pattern is None or not text:
            return text
        return self._pattern.sub(lambda m: self.mapping[m.group(0)], text)

    def arguments(self, value: Any) -> Any:
        """A tool call's arguments with the aliases of the request mapped back (keys stay)."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.arguments(v) for v in value]
        if isinstance(value, dict):
            return {k: self.arguments(v) for k, v in value.items()}
        return value

    def restorer(self) -> Restorer:
        return Restorer(self)


class Outgoing:
    """What goes to a cloud route: the scrubbed messages and tools, and how to restore."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        restoration: Restoration,
    ) -> None:
        self.messages, self.tools, self.restoration = messages, tools, restoration


class Pseudonyms:
    """Name to alias map and the replacements built on it. Thread-safe; ``repr`` shows nothing."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._alias: dict[str, str] = {}  # comparison key of a name -> alias
        self._real: dict[str, str] = {}  # comparison key -> the name as registered
        self._variants: dict[str, str] = {}  # folded spelling (also JSON-escaped) -> alias
        self._pattern: re.Pattern[str] | None = None
        self._used = 0
        self._rng = random.SystemRandom()

    def __repr__(self) -> str:
        return "Pseudonyms(<map not shown>)"

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._alias)

    def sync(
        self,
        names: Iterable[str],
        templates: Mapping[str, str],
        language: str,
        default_language: str,
    ) -> None:
        """Make the map match the registry as it is now. Names no longer there are dropped (they
        can no longer be restored); a name with an alias keeps it; a new one gets the next label,
        new names in random order. The template is the one for ``language``, else for
        ``default_language``, else the first. An alias that equals a real name (a record
        registered after the alias was given) is replaced."""
        current: dict[str, str] = {}
        for raw in names:
            name = " ".join(_nfc(raw).split())
            if name and (key := _key(name)):
                current.setdefault(key, name)
        if not templates:
            return
        template = (
            templates.get(language)
            or templates.get(default_language)
            or next(iter(templates.values()))
        )
        with self._lock:
            kept = {k: a for k, a in self._alias.items() if k in current}
            reserved = {_key(v) for n in current.values() for v in _variants(n)}
            taken = set(reserved)
            for key in list(kept):
                alias_key = _key(kept[key])
                if alias_key in taken:
                    del kept[key]  # collides with a real name: hand out another one
                else:
                    taken.add(alias_key)
            fresh = [k for k in current if k not in kept]
            self._rng.shuffle(fresh)
            for key in fresh:
                while True:
                    alias = template.replace(PSEUDONYM_LABEL, _label(self._used))
                    self._used += 1
                    if _key(alias) not in taken:
                        break
                taken.add(_key(alias))
                kept[key] = alias
            self._alias, self._real = kept, {k: current[k] for k in kept}
            self._variants = {_key(v): kept[k] for k, n in self._real.items() for v in _variants(n)}
            self._pattern = _compile(self._variants)

    # ------------------------------------------------------------------ text

    def aliases(self) -> list[str]:
        """The aliases in use."""
        with self._lock:
            return list(self._alias.values())

    def to_alias(self, text: str, used: set[str] | None = None) -> str:
        """``text`` (NFC) with every known name replaced by its alias; the aliases put in are
        added to ``used``."""
        with self._lock:
            pattern, variants = self._pattern, dict(self._variants)
        nfc = _nfc(text)
        out, pos = [], 0
        for start, end, key in _spans(pattern, nfc):
            alias = variants[key]
            out += [nfc[pos:start], alias]
            pos = end
            if used is not None:
                used.add(alias)
        return "".join(out) + nfc[pos:]

    def safe_cut(self, text: str, cap: int) -> str:
        """``text`` cut to at most ``cap`` characters, but never inside a name: a cut that falls
        in the middle of one moves back to its start, so no partial name is ever sent."""
        nfc = _nfc(text)
        if len(nfc) <= cap:
            return nfc
        with self._lock:
            pattern = self._pattern
        cut = cap
        for start, end, _ in _spans(pattern, nfc):
            if start < cut < end:
                cut = start
        return nfc[:cut]

    def mentions(self, text: str) -> bool:
        """Whether ``text`` holds a registered name."""
        with self._lock:
            pattern = self._pattern
        return bool(_spans(pattern, text))

    def leaks(self, body: Any) -> bool:
        """Whether a name survives in ``body`` (messages, tools, the request) once written as
        JSON, with ``ensure_ascii`` on or off, which is how it would go on the wire."""
        with self._lock:
            pattern = self._pattern
        if pattern is None:
            return False
        return any(
            _spans(pattern, json.dumps(body, ensure_ascii=ascii_only))
            for ascii_only in (False, True)
        )

    def _aliased(self, value: Any, used: set[str]) -> Any:
        if isinstance(value, str):
            return self.to_alias(value, used)
        if isinstance(value, list):
            return [self._aliased(v, used) for v in value]
        if isinstance(value, dict):
            return {k: self._aliased(v, used) for k, v in value.items()}
        return value

    def _aliased_json(self, text: str, used: set[str]) -> str:
        """A JSON text with its strings aliased and written back without ``\\u`` escapes; text
        that is not a JSON object or list is aliased as plain text."""
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):
            return self.to_alias(text, used)
        if isinstance(data, (dict, list)):
            return self.to_alias(json.dumps(self._aliased(data, used), ensure_ascii=False), used)
        return self.to_alias(text, used)

    # ------------------------------------------------------------------ requests

    def prepare(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> Outgoing:
        """The request for a cloud route: copies of ``messages`` and ``tools`` with the names
        replaced (the input, the RAM history, is not changed), and the restoration for exactly
        the aliases put in, less any alias that was already in the text (said literally)."""
        used: set[str] = set()
        literal = self._literal_aliases(messages, tools)
        out = [self._scrub_message(m, used) for m in messages]
        scrubbed_tools = self._scrub_tools(tools, used)
        with self._lock:
            by_alias = {a: self._real[k] for k, a in self._alias.items()}
        mapping = {a: by_alias[a] for a in used if a in by_alias and a not in literal}
        return Outgoing(out, scrubbed_tools, Restoration(mapping))

    def _literal_aliases(self, messages: Any, tools: Any) -> set[str]:
        blob = json.dumps([messages, tools], ensure_ascii=False, default=str)
        return {a for a in self.aliases() if a in blob}

    def _scrub_message(self, m: dict[str, Any], used: set[str]) -> dict[str, Any]:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, str):
            m["content"] = (
                self._aliased_json(content, used)
                if m.get("role") == "tool"
                else self.to_alias(content, used)
            )
        elif isinstance(content, list):
            m["content"] = [
                {**p, "text": self.to_alias(p["text"], used)}
                if isinstance(p, dict) and isinstance(p.get("text"), str)
                else p
                for p in content
            ]
        if m.get("tool_calls"):
            calls = copy.deepcopy(m["tool_calls"])
            for call in calls:
                fn = call.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = self._aliased_json(fn["arguments"], used)
            m["tool_calls"] = calls
        return m

    def _scrub_tools(
        self, tools: list[dict[str, Any]] | None, used: set[str]
    ) -> list[dict[str, Any]] | None:
        """Tool schemas with names replaced in the descriptions and parameters (the ``look_at``
        enum); a tool's own name stays."""
        if not tools:
            return tools
        out = []
        for tool in tools:
            fn = dict(tool.get("function", {}))
            for key in ("description", "parameters"):
                if key in fn:
                    fn[key] = self._aliased(fn[key], used)
            out.append({**tool, "function": fn})
        return out


class Restorer:
    """Turns the aliases of one request back into names in streamed text. An alias can arrive
    split across deltas ("Person" + "-A"), so a tail that could still become one is held back
    until the next delta (or ``flush``) settles it; a stream that is cut loses the held tail."""

    def __init__(self, restoration: Restoration) -> None:
        self._r = restoration
        self._held = ""

    def _hold_from(self, buf: str) -> int:
        aliases = list(self._r.mapping)
        for i in range(len(buf)):
            if (i == 0 or not (buf[i - 1].isalnum() or buf[i - 1] == "_")) and any(
                a.startswith(buf[i:]) for a in aliases
            ):
                return i
        return len(buf)

    def feed(self, delta: str) -> str:
        """The part of the text that is settled, with aliases replaced."""
        buf = self._held + delta
        cut = self._hold_from(buf)
        self._held = buf[cut:]
        return self._r.text(buf[:cut])

    def flush(self) -> str:
        """Whatever is held, at the end of the text."""
        buf, self._held = self._held, ""
        return self._r.text(buf)
