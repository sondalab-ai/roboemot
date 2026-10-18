"""Pseudonyms for routes that are not local (SPEC 8, 12 Data; ``privacy.pseudonyms_to_cloud``).

The registry's display names are swapped for neutral aliases ("Person-A", "Persona-A") in
everything a cloud route receives, and swapped back in what it sends back:

- out: the system prompt (profile and scene summaries), the history of past turns (the user's own
  words included), the tool schemas (the ``look_at`` enum) and the tool calls and results of
  earlier rounds (their JSON is parsed, the string values aliased, never the keys, and the
  JSON written again without ``\\u`` escapes);
- back: the streamed reply text, before the sentence splitter and the speech, and the arguments
  of every tool call, before the executor runs. The history kept in RAM holds the real names.

Matching. Text is NFC-normalised, then compared with case folding, accents and other combining
marks removed (``Ömer``, ``omer`` and an NFD ``Ömer`` are one name; ``Strauß`` matches
``STRAUSS``; ``İlker`` matches ``ilker``), as whole words, with any run of spaces between the
words of a name. The JSON-escaped spellings of a name are matched too, and so is a name
written with ``\\uXXXX`` escapes in text that is not JSON (an escape counts as its character).
A check over the text of the finished request (as it is, as JSON strings, and decoded where it
is JSON) refuses to send a body that still holds a name (``Pseudonyms.leaks``); the request's
structure (roles, types, tool names, schema keywords) is not text and is never aliased or
checked, except that a JSON key other than a tool property name is checked. A registry name that
cannot be aliased (no letter or digit, a word of an alias template, a value of the request
format such as "user") is refused and counted, and reaches a cloud route as typed.

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
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import httpx

from ..config.schema import PSEUDONYM_LABEL, LLMRoute, RoboEmotConfig

UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
UNICODE_ESCAPE_LEN = 6  # characters of "\\uXXXX"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOOPBACK_NAMES = ("localhost",)
MDNS_SUFFIX = ".local"
NUMERIC_HOST = re.compile(r"^(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+))*$")  # host is lowercased
DOTTED_QUAD = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
# Names that are values of the request format itself (roles, part and call types): a registry
# entry with one of these names is not aliased (SPEC 12), it would rewrite the wire format.
STRUCTURAL_WORDS = frozenset(
    {"user", "assistant", "system", "tool", "function", "developer", "text", "image_url"}
)
# Request fields whose values are structure, not text: never scrubbed, never leak-checked.
ENVELOPE_KEYS = frozenset({"model", "stream", "max_tokens", "tool_choice"})
MESSAGE_FIXED = frozenset({"role", "tool_call_id"})
CALL_FIXED = frozenset({"id", "type"})
SCHEMA_FIXED = frozenset({"type", "format", "required"})
SCHEMA_TEXT = ("description", "const")


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
    whatever ``locality`` says. A fully qualified name with a trailing dot (``ai.``) does not
    count as a name without a dot. The host is parsed by httpx, the library that connects."""
    url = route.base_url
    if "\\" in url or any(c.isspace() or ord(c) < ord(" ") for c in url):
        return False
    try:
        parsed = httpx.URL(url)
        qualified = parsed.host.endswith(".")
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
        dotless = "." not in host and not qualified
        return host in LOOPBACK_NAMES or host.endswith(MDNS_SUFFIX) or dotless
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
    characters: ß gives ss, İ gives i). Compatibility forms are decomposed before the case is
    folded (ℌ gives h) and again after, as case folding can compose."""
    decomposed = unicodedata.normalize("NFKD", unicodedata.normalize("NFKD", c).casefold())
    return "".join(x for x in decomposed if unicodedata.category(x) != "Mn")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _fold(text: str) -> tuple[str, list[int], list[int], str]:
    """``(folded, start, end, nfc)``: the folded text and, for each of its characters, the
    span of the NFC text it came from. A ``\\uXXXX`` escape (any hex case, not a surrogate)
    counts as the character it writes, so a name spelled with escapes in plain text is matched
    and replaced over its six-character spans."""
    nfc = _nfc(text)
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i = 0
    while i < len(nfc):
        c, width = nfc[i], 1
        if (m := UNICODE_ESCAPE.match(nfc, i)) and not 0xD800 <= (code := int(m[1], 16)) <= 0xDFFF:
            c, width = chr(code), UNICODE_ESCAPE_LEN
        for x in _fold_char(c):
            chars.append(x)
            starts.append(i)
            ends.append(i + width)
        i += width
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


def _strings(value: Any) -> Iterator[str]:
    """Every string value in ``value`` (keys are not values)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)


def _property_names(body: Any) -> frozenset[str]:
    """The property names of the tool schemas in a request body (``properties`` keys, at any
    depth). They are sent as typed, so a JSON key that equals one exposes nothing new."""
    names: set[str] = set()
    tools = body.get("tools") if isinstance(body, dict) else None
    stack: list[Any] = list(tools) if isinstance(tools, list) else []
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    names.update(k for k in value if isinstance(k, str))
                stack.append(value)
    return frozenset(names)


def _without_keys(text: str, exempt: frozenset[str]) -> str:
    """``text`` as it is, unless it is a JSON object or list holding an ``exempt`` key: then
    the text is written again with those keys left out (their values stay)."""
    if not exempt or not text.lstrip().startswith(("{", "[")):
        return text
    try:
        data = json.loads(text)

        def strip(node: Any) -> Any:
            if isinstance(node, list):
                return [strip(v) for v in node]
            if isinstance(node, dict):
                return {k: strip(v) for k, v in node.items() if k not in exempt}
            return node

        return json.dumps(strip(data), ensure_ascii=False)
    except (ValueError, RecursionError):
        return text


def _decoded(text: str, exempt: frozenset[str] = frozenset()) -> Iterator[str]:
    """The strings and keys of ``text`` when it is a JSON object or list, else nothing. A key
    in ``exempt`` (a property name of a tool schema in the same request) is not yielded."""
    if not text.lstrip().startswith(("{", "[")):
        return
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            yield node
        elif isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key not in exempt:
                    yield key
                stack.append(value)


def _schema_text(node: Any) -> Iterator[str]:
    """The string values of a JSON schema, less its keywords and property names."""
    if isinstance(node, list):
        for v in node:
            yield from _schema_text(v)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key not in SCHEMA_FIXED:
                yield from _schema_text(value)
    elif isinstance(node, str):
        yield node


def _message_text(m: Any) -> Iterator[str]:
    if not isinstance(m, dict):
        yield from _strings(m)
        return
    for key, value in m.items():
        if key in MESSAGE_FIXED:
            continue
        if key == "content" and isinstance(value, list):
            for part in value:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    continue
                yield from (
                    _strings({k: v for k, v in part.items() if k != "type"})
                    if isinstance(part, dict)
                    else _strings(part)
                )
        elif key == "tool_calls" and isinstance(value, list):
            for call in value:
                if not isinstance(call, dict):
                    yield from _strings(call)
                    continue
                for ck, cv in call.items():
                    if ck in CALL_FIXED:
                        continue
                    if ck == "function" and isinstance(cv, dict):
                        yield from _strings({k: v for k, v in cv.items() if k != "name"})
                    else:
                        yield from _strings(cv)
        else:
            yield from _strings(value)


def _tool_text(tool: Any) -> Iterator[str]:
    if not isinstance(tool, dict):
        yield from _strings(tool)
        return
    for key, value in tool.items():
        if key == "type":
            continue
        if key == "function" and isinstance(value, dict):
            for fk, fv in value.items():
                if fk == "name":
                    continue
                yield from _schema_text(fv) if fk == "parameters" else _strings(fv)
        else:
            yield from _strings(value)


def _request_text(body: Any) -> Iterator[str]:
    """The text of a request (``{"messages", "tools", ...}``) or of a list of messages: what a
    name could be hidden in, without the request's structure."""
    if isinstance(body, list):
        for m in body:
            yield from _message_text(m)
        return
    if not isinstance(body, dict):
        yield from _strings(body)
        return
    for key, value in body.items():
        if key in ENVELOPE_KEYS:
            continue
        if key == "messages" and isinstance(value, list):
            for m in value:
                yield from _message_text(m)
        elif key == "tools" and isinstance(value, list):
            for t in value:
                yield from _tool_text(t)
        else:
            yield from _strings(value)


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
        self.refused = 0  # registry names that cannot be aliased (a count, never the names)

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
        ``default_language``, else the first. An alias that equals a real name, or holds one
        (a record "A" inside "Person-A", registered after the alias was given), is replaced.

        A name that cannot be aliased is refused and only counted (``refused``): one without a
        letter or digit, one with a backslash (JSON's escape character), one found inside a
        template's prefix ("Person", "Persona-") and a value of the request format
        (``STRUCTURAL_WORDS``). It reaches the cloud as typed."""
        if not templates:
            raise ValueError("privacy.pseudonym_templates is empty")
        prefixes = [t.replace(PSEUDONYM_LABEL, "") for t in templates.values()]
        current: dict[str, str] = {}
        refused: set[str] = set()
        for raw in names:
            # Double quotes around a name ('"Nonna"') are JSON's string delimiters: matched
            # without them, so a quote next to an alias in a JSON text never reads as a name.
            name = " ".join(_nfc(raw).split()).strip('" ')
            if not name or not (key := _key(name)):
                continue
            own = _compile(_key(v) for v in _variants(name))
            if (
                not re.search(r"\w", key)
                or "\\" in name  # the JSON escape character: no name has one
                or key in STRUCTURAL_WORDS
                or any(_spans(own, prefix) for prefix in prefixes)
            ):
                refused.add(key)
            else:
                current.setdefault(key, name)
        template = (
            templates.get(language)
            or templates.get(default_language)
            or next(iter(templates.values()))
        )
        pattern = _compile(_key(v) for n in current.values() for v in _variants(n))
        # A name's core without its edge punctuation ("A'" gives "a"): an alias followed by
        # that punctuation in a text ("Person-A'") would read as the name.
        cores = _compile(re.sub(r"^\W+|\W+$", "", k) for k in current)

        def clashes(alias: str) -> bool:
            return _key(alias) in taken or bool(_spans(pattern, alias) or _spans(cores, alias))

        with self._lock:
            kept = {k: a for k, a in self._alias.items() if k in current}
            taken = {_key(v) for n in current.values() for v in _variants(n)}
            for key in list(kept):
                if clashes(kept[key]):
                    del kept[key]  # collides with a real name: hand out another one
                else:
                    taken.add(_key(kept[key]))
            fresh = [k for k in current if k not in kept]
            self._rng.shuffle(fresh)
            for key in fresh:
                # Each name can block at most one label (it is not inside the prefix), so this
                # ends; the bound only turns a broken invariant into an error, not a hang.
                for _ in range(len(current) + len(taken) + 1):
                    alias = template.replace(PSEUDONYM_LABEL, _label(self._used))
                    self._used += 1
                    if not clashes(alias):
                        break
                else:
                    raise RuntimeError("pseudonyms: no free alias")
                taken.add(_key(alias))
                kept[key] = alias
            self._alias, self._real = kept, {k: current[k] for k in kept}
            self._variants = {_key(v): kept[k] for k, n in self._real.items() for v in _variants(n)}
            self._pattern = _compile(self._variants)
            self.refused = len(refused)

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
        """Whether a name survives in the text of ``body`` (a request with ``messages`` and
        ``tools``, or a list of messages). Each string is checked as it is, written as a JSON
        string (escaped quotes) and, when it is JSON itself (tool arguments and results), with
        its strings and keys decoded (``\\u`` escapes). The ``ensure_ascii`` spelling adds
        nothing (every escaped spelling of a name is also caught decoded) and would see word
        boundaries that are not in the text ("eß" as "e\\u00df"). The structure of the
        request is not text and is not checked: roles, ids, types, tool names, JSON schema
        ``type``, ``format`` and ``required``, image parts (a name there is not a leak, and
        checking it would let a registry name like "User" or "Target" disable every cloud
        route). The one JSON key that is not checked is a property name of a tool schema in the
        same request (a record "Target" and the ``target`` argument of ``look_at``): the schema
        already sends it as typed. Any other key, such as a name used as a key in a tool result,
        is checked and blocks the request."""
        with self._lock:
            pattern = self._pattern
        if pattern is None:
            return False
        exempt = _property_names(body)
        return any(
            _spans(pattern, form)
            for text in _request_text(body)
            for shown in (_without_keys(text, exempt),)
            for form in (
                shown,
                json.dumps(shown, ensure_ascii=False)[1:-1],
                *_decoded(text, exempt),
            )
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
        """A JSON text with its string values aliased (keys stay as they are) and written back
        without ``\\u`` escapes; text that is not a JSON object or list is aliased as plain
        text."""
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):
            return self.to_alias(text, used)
        if isinstance(data, (dict, list)):
            return json.dumps(self._aliased(data, used), ensure_ascii=False)
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
        """The aliases already in the request as whole tokens (said or written literally)."""
        blob = json.dumps([messages, tools], ensure_ascii=False, default=str)
        return {a for a in self.aliases() if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", blob)}

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
        """Tool schemas with names replaced in the descriptions and in the parameters' text
        (``description``, ``const``, the ``look_at`` enum); a tool's own name, property names
        and keywords (``type``, ``format``, ``required``) stay."""
        if not tools:
            return tools
        out = []
        for tool in tools:
            fn = dict(tool.get("function", {}))
            if isinstance(fn.get("description"), str):
                fn["description"] = self.to_alias(fn["description"], used)
            if "parameters" in fn:
                fn["parameters"] = self._aliased_schema(fn["parameters"], used)
            out.append({**tool, "function": fn})
        return out

    def _aliased_schema(self, node: Any, used: set[str]) -> Any:
        """A JSON schema with names replaced in its text values only."""
        if isinstance(node, list):
            return [self._aliased_schema(v, used) for v in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in SCHEMA_TEXT and isinstance(value, str):
                out[key] = self.to_alias(value, used)
            elif key == "enum" and isinstance(value, list):
                out[key] = [self.to_alias(v, used) if isinstance(v, str) else v for v in value]
            elif key == "properties" and isinstance(value, dict):
                out[key] = {k: self._aliased_schema(v, used) for k, v in value.items()}
            elif key in SCHEMA_FIXED:
                out[key] = value
            else:
                out[key] = self._aliased_schema(value, used)
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


class NameSource:
    """The pseudonym map for callers of the router outside the conversation (persona traits,
    route checks). ``names`` lists the registry's display names; ``for_config`` brings the map
    in line with them and returns it, or None when ``privacy.pseudonyms_to_cloud`` is off. An
    error from ``names`` propagates, so nothing is sent without the names it should replace."""

    def __init__(self, names: Callable[[], Iterable[str]]) -> None:
        self._names = names
        self._map = Pseudonyms()

    def for_config(self, cfg: RoboEmotConfig) -> Pseudonyms | None:
        """The synced map for ``cfg``, in the configured conversation language."""
        if not cfg.privacy.pseudonyms_to_cloud:
            return None
        language = cfg.conversation.language
        self._map.sync(self._names(), cfg.privacy.pseudonym_templates, language, language)
        return self._map
