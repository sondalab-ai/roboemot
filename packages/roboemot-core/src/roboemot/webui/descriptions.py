"""Setting descriptions in the page language (SPEC 11A, 12).

The schema describes every field in English (``config.schema.param``). The Settings page shows
them in the language of the page: ``descriptions_<lang>.json`` maps each English description to
its translation, and :func:`localize` swaps them in a copy of the catalogue rows (and of the rows
of an entry form). English is the source and needs no table. A description the table lacks is
dropped rather than shown in English; ``tests/test_webui_one_language.py`` keeps the table complete.
A robot plugin's own rows are worded by the plugin (it reads ``conversation.language``): their
descriptions are not in the table and pass through unchanged.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any

from .strings import DEFAULT_LANGUAGE

log = logging.getLogger(__name__)
HERE = Path(__file__).parent


@functools.cache
def table(language: str) -> dict[str, str]:
    """English description -> its ``language`` wording; empty for English or a language
    without a table."""
    path = HERE / f"descriptions_{language}.json"
    if language == DEFAULT_LANGUAGE or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def localize(rows: list[Any], language: str, own: frozenset[str] = frozenset()) -> list[Any]:
    """Copies of ``rows`` with each description in ``language``. ``own`` are paths whose rows
    come worded already (a robot plugin's), kept as they are."""
    words = table(language)
    if not words:
        return rows
    out = []
    for row in rows:
        if row["path"] in own:
            out.append(row)
            continue
        text = row.get("description") or ""
        if text and text not in words:  # the path only, never the text
            log.debug("no %s description for %s", language, row["path"])
        copy = {**row, "description": words.get(text, "") if text else ""}
        if isinstance(row.get("entry"), list):
            copy["entry"] = localize(row["entry"], language)
        out.append(copy)
    return out
