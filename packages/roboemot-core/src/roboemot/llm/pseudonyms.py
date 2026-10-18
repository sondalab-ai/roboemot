"""Pseudonyms for routes that are not local (SPEC 8, 12 Data; ``privacy.pseudonyms_to_cloud``).

The registry's display names are swapped for neutral aliases ("Person A", "Persona A") in
everything a cloud route receives, and swapped back in what it sends back:

- out: the system prompt (profile and scene summaries), the history of past turns (the user's own
  words included), the tool schemas (the ``look_at`` choices) and the tool calls and results of
  earlier rounds;
- back: the streamed reply text, before the sentence splitter and the speech, and the arguments
  of every tool call, before the executor runs. The history kept in RAM holds the real names.

The map lives in RAM for the life of the process and is never written or logged (this module logs
nothing). An alias is a counter letter handed out in sorted name order the first time a name is
seen, so it is stable for the process and says nothing about the person; after a restart the
order can differ if the registry changed. Names are matched as whole words, ignoring case and
spacing. Limits: nicknames, first names of a multi-word name, inflections and misspellings are
not caught, and a name that is also a common word is replaced wherever it appears.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import re
import threading
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

from ..config.schema import PSEUDONYM_LABEL, LLMRoute

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MDNS_SUFFIXES = (".local", ".localhost")


def route_is_local(route: LLMRoute) -> bool:
    """Whether ``route`` stays inside the household: ``locality`` when set, else judged from the
    host of ``base_url`` (``auto``). Unknown means cloud, so a doubtful route gets aliases."""
    if route.locality != "auto":
        return route.locality == "local"
    try:
        host = (urlsplit(route.base_url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost" or host.endswith(MDNS_SUFFIXES) or "." not in host
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _label(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (spreadsheet columns)."""
    out = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, len(LETTERS))
        out = LETTERS[rem] + out
    return out


def _norm(text: str) -> str:
    return " ".join(text.split())


def _pattern(words: str) -> str:
    return r"\s+".join(re.escape(w) for w in words.split())


def _compile(terms: Iterable[str]) -> re.Pattern[str] | None:
    ordered = sorted({t for t in terms if t}, key=len, reverse=True)  # longest name first
    if not ordered:
        return None
    return re.compile(
        r"(?<!\w)(?:" + "|".join(_pattern(t) for t in ordered) + r")(?!\w)", re.IGNORECASE
    )


def replace_names(text: str, names: Mapping[str, str]) -> str:
    """``text`` with each key of ``names`` (a whole word or phrase, any case) replaced by its
    value. A fixed map, for callers that bring their own aliases."""
    folded = {_norm(k).casefold(): v for k, v in names.items() if _norm(k)}
    pattern = _compile(folded)
    if pattern is None or not text:
        return text
    return pattern.sub(lambda m: folded.get(_norm(m.group(0)).casefold(), m.group(0)), text)


class Pseudonyms:
    """Name to alias map and the replacements built on it. Thread-safe; ``repr`` shows nothing."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._alias: dict[str, str] = {}  # normalised casefolded name -> alias
        self._real: dict[str, str] = {}  # casefolded alias -> name as registered
        self._variants: dict[str, str] = {}  # the same, plus the JSON-escaped spelling
        self._used = 0
        self._forward: re.Pattern[str] | None = None
        self._back: re.Pattern[str] | None = None

    def __repr__(self) -> str:
        return "Pseudonyms(<map not shown>)"

    def __bool__(self) -> bool:
        return bool(self._alias)

    def learn(
        self,
        names: Iterable[str],
        templates: Mapping[str, str],
        language: str,
        default_language: str,
    ) -> None:
        """Give an alias to each name not seen before, in sorted order. The template is the one
        for ``language``, else for ``default_language``, else the first. Aliases that equal a
        real name are skipped, so the way back is never ambiguous."""
        fresh = {_norm(n).casefold(): _norm(n) for n in names if _norm(n)}
        if not templates:
            return
        template = (
            templates.get(language)
            or templates.get(default_language)
            or next(iter(templates.values()))
        )
        with self._lock:
            new = sorted(k for k in fresh if k not in self._alias)
            if not new:
                return
            taken = set(self._alias) | set(self._real) | set(fresh)
            for key in new:
                while True:
                    alias = template.replace(PSEUDONYM_LABEL, _label(self._used))
                    self._used += 1
                    if alias.casefold() not in taken:
                        break
                taken.add(alias.casefold())
                self._alias[key] = alias
                self._real[alias.casefold()] = fresh[key]
            self._variants = {}
            for key, alias in self._alias.items():
                self._variants[key] = alias
                self._variants[_norm(json.dumps(key, ensure_ascii=False)[1:-1])] = alias  # JSON
            self._forward = _compile(self._variants)
            self._back = _compile(self._real)

    # ------------------------------------------------------------------ text

    def aliases(self) -> list[str]:
        """The aliases in use, lower-cased (for the stream restorer)."""
        with self._lock:
            return list(self._real)

    def to_alias(self, text: str) -> str:
        """``text`` with every known name replaced by its alias."""
        with self._lock:
            pattern, aliases = self._forward, dict(self._variants)
        if pattern is None or not text:
            return text
        return pattern.sub(lambda m: aliases.get(_norm(m.group(0)).casefold(), m.group(0)), text)

    def to_real(self, text: str) -> str:
        """``text`` with every alias replaced by the name it stands for."""
        with self._lock:
            pattern, real = self._back, dict(self._real)
        if pattern is None or not text:
            return text
        return pattern.sub(lambda m: real.get(_norm(m.group(0)).casefold(), m.group(0)), text)

    def _restored(self, value: Any) -> Any:
        """``value`` with the strings of lists and dicts mapped back to real names (keys are
        left alone)."""
        if isinstance(value, str):
            return self.to_real(value)
        if isinstance(value, list):
            return [self._restored(v) for v in value]
        if isinstance(value, dict):
            return {k: self._restored(v) for k, v in value.items()}
        return value

    def _aliased(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.to_alias(value)
        if isinstance(value, list):
            return [self._aliased(v) for v in value]
        if isinstance(value, dict):
            return {k: self._aliased(v) for k, v in value.items()}
        return value

    # ------------------------------------------------------------------ requests

    def scrub_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """A copy of ``messages`` with names replaced in the text of every message and in the
        arguments of earlier tool calls. The input is not changed (it is the RAM history)."""
        out = []
        for m in messages:
            m = dict(m)
            content = m.get("content")
            if isinstance(content, str):
                m["content"] = self.to_alias(content)
            elif isinstance(content, list):
                m["content"] = [
                    {**p, "text": self.to_alias(p["text"])}
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                    else p
                    for p in content
                ]
            if m.get("tool_calls"):
                calls = copy.deepcopy(m["tool_calls"])
                for call in calls:
                    fn = call.get("function", {})
                    if isinstance(fn.get("arguments"), str):
                        fn["arguments"] = self.to_alias(fn["arguments"])
                m["tool_calls"] = calls
            out.append(m)
        return out

    def scrub_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """A copy of the tool schemas with names replaced in the descriptions and parameters
        (the ``look_at`` enum); a tool's own name stays."""
        if not tools:
            return tools
        out = []
        for tool in tools:
            fn = dict(tool.get("function", {}))
            for key in ("description", "parameters"):
                if key in fn:
                    fn[key] = self._aliased(fn[key])
            out.append({**tool, "function": fn})
        return out

    def real_arguments(self, arguments: Any) -> Any:
        """A tool call's arguments with aliases mapped back to names."""
        return self._restored(arguments)

    def restorer(self) -> Restorer:
        """A fresh stream restorer for one request."""
        return Restorer(self)


class Restorer:
    """Turns aliases in streamed text back into names. An alias can arrive split across
    deltas ("Pers" + "on A"), so a tail that could still become one is held back until the next
    delta (or ``flush``) settles it."""

    def __init__(self, pseudonyms: Pseudonyms) -> None:
        self._p = pseudonyms
        self._held = ""

    def _hold_from(self, buf: str) -> int:
        aliases = self._p.aliases()
        if not aliases:
            return len(buf)
        low = buf.casefold()
        if len(low) != len(buf):  # case folding changed the length: hold nothing rather than guess
            return len(buf)
        for i in range(len(buf)):
            if (i == 0 or not (buf[i - 1].isalnum() or buf[i - 1] == "_")) and any(
                a.startswith(low[i:]) for a in aliases
            ):
                return i
        return len(buf)

    def feed(self, delta: str) -> str:
        """The part of the text that is settled, with aliases replaced."""
        buf = self._held + delta
        cut = self._hold_from(buf)
        self._held = buf[cut:]
        return self._p.to_real(buf[:cut])

    def flush(self) -> str:
        """Whatever is held, at the end of the text."""
        buf, self._held = self._held, ""
        return self._p.to_real(buf)
