"""The station catalogue shipped with the library (SPEC 11, Voice tools, Radio; 11A).

``radio_catalogue.yaml`` lists checked stations grouped by genre. The web UI's Tools tab offers
them, and a person adds one to ``tools.radio.stations`` through the settings entry route. Nothing
here plays: the radio plays only listed stations.

Each entry's ``name``, ``url``, ``genres`` and ``language`` are validated by
``config.schema.Station``, the same validator as a configured station, so a catalogue URL obeys
the same rules (http or https, no quote, backquote, exclamation mark, space, backslash or user
info). The rest (``operator``, ``policy_url``, ``official``, ``bitrate_kbps``) is catalogue-only.
A broken file fails at load and in the tests, never later in the page.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, PositiveInt, field_validator, model_validator

from roboemot.config.schema import Station, Strict, route_url_ok

CATALOGUE_FILE = Path(__file__).with_name("radio_catalogue.yaml")
# The genre groups in display order; their labels are web UI strings (tools_catalogue_group_<id>).
GROUP_IDS = (
    "jazz",
    "ambient_chillout",
    "funk_soul_rnb",
    "hiphop_rap",
    "rock",
    "metal_progressive",
)
# The fields an added station carries: exactly a ``Station``'s.
STATION_FIELDS = tuple(Station.model_fields)


class CatalogueStation(Strict):
    """One catalogue entry: a ``Station`` plus where it comes from."""

    name: str
    url: str
    genres: list[str]
    language: str | None
    operator: str
    policy_url: str
    official: bool
    bitrate_kbps: PositiveInt

    @model_validator(mode="before")
    @classmethod
    def _a_station(cls, data: Any) -> Any:
        """The station part goes through ``Station``'s own validators (name, URL rules). A
        missing station field is reported as missing, by name, before ``Station`` sees it."""
        if isinstance(data, dict):
            missing = [k for k in STATION_FIELDS if k not in data]
            if missing:
                raise ValueError(f"field required: {', '.join(missing)}")
            station = Station.model_validate({k: data[k] for k in STATION_FIELDS})
            return {**data, **station.model_dump()}
        return data

    @field_validator("operator")
    @classmethod
    def _operator(cls, value: str) -> str:
        """An operator is named: the page shows who runs the stream."""
        if not value.strip():
            raise ValueError("a catalogue station needs an operator")
        return value

    @field_validator("policy_url")
    @classmethod
    def _policy(cls, value: str) -> str:
        """The operator's page the tab links to: https only."""
        if not route_url_ok(value, ("https",)) or httpx.URL(value).userinfo:
            raise ValueError("must be an https URL with a host and no user info")
        return value

    def station(self) -> dict[str, Any]:
        """The entry as ``tools.radio.stations`` takes it (the add route's ``value``)."""
        return {k: getattr(self, k) for k in STATION_FIELDS}


class CatalogueGroup(Strict):
    """One genre group: its id (one of ``GROUP_IDS``) and its stations."""

    id: str
    stations: list[CatalogueStation]

    @field_validator("id")
    @classmethod
    def _known(cls, value: str) -> str:
        """Only the known groups: each has a label in both page languages."""
        if value not in GROUP_IDS:
            raise ValueError("unknown group")
        return value


class Catalogue(BaseModel):
    """The whole file. A station name or URL appears once in the catalogue, and so does a
    group id."""

    model_config = {"extra": "forbid", "frozen": True}

    groups: list[CatalogueGroup]

    @model_validator(mode="after")
    def _unique(self) -> Catalogue:
        """Duplicate group ids, station names or URLs are refused."""
        ids = [g.id for g in self.groups]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate group id")
        stations = [s for g in self.groups for s in g.stations]
        for field in ("name", "url"):
            values = [getattr(s, field) for s in stations]
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate station {field}")
        return self


def parse_catalogue(raw: Any) -> Catalogue:
    """Validate a catalogue's data (the YAML as loaded); raises ``pydantic.ValidationError``."""
    return Catalogue.model_validate(raw)


@cache
def radio_catalogue() -> Catalogue:
    """The shipped catalogue, read and validated once."""
    return parse_catalogue(yaml.safe_load(CATALOGUE_FILE.read_text(encoding="utf-8")))
