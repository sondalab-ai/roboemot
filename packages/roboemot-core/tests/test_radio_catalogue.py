"""SPEC 11 (Voice tools, Radio): the station catalogue shipped with the library.

Every shipped entry is a valid ``Station`` (the same URL rules as a configured station), group
ids are the known ones, and a duplicate name or URL is refused at load.
"""

from __future__ import annotations

import copy

import pytest
import yaml
from pydantic import ValidationError
from roboemot.config.schema import RoboEmotConfig, Station
from roboemot.tools.radio_catalogue import (
    CATALOGUE_FILE,
    GROUP_IDS,
    parse_catalogue,
    radio_catalogue,
)
from roboemot.webui.setting_clean import clean_label
from roboemot.webui.strings import STRINGS

SOMAFM_TERMS = "https://somafm.com/contact/tos.html"


def raw() -> dict:
    return yaml.safe_load(CATALOGUE_FILE.read_text(encoding="utf-8"))


def test_every_shipped_entry_is_a_valid_station():
    cfg = RoboEmotConfig()
    stations = [s for g in radio_catalogue().groups for s in g.stations]
    assert stations
    for s in stations:
        assert Station.model_validate(s.station()).model_dump() == s.station()
        assert clean_label(s.name, cfg) == s.name, "the add route's name rule"
        assert s.genres and s.operator and s.policy_url.startswith("https://")


def test_the_groups_are_the_owner_s_six_in_order():
    assert [g.id for g in radio_catalogue().groups] == list(GROUP_IDS)
    assert all(g.stations for g in radio_catalogue().groups)


def test_every_group_has_a_label_in_both_page_languages():
    for language in ("en", "it"):
        for group in GROUP_IDS:
            assert f"tools_catalogue_group_{group}" in STRINGS[language], (language, group)


def test_radio_france_and_somafm_are_marked_unofficial():
    for s in (s for g in radio_catalogue().groups for s in g.stations):
        third_party = s.operator.startswith("Radio France") or s.operator == "SomaFM"
        assert s.official is not third_party, s.name
        if s.operator == "SomaFM":
            assert s.policy_url == SOMAFM_TERMS


@pytest.mark.parametrize("field", ["name", "url"])
def test_a_duplicate_name_or_url_is_refused(field):
    data = raw()
    twin = copy.deepcopy(data["groups"][1]["stations"][0])
    twin.update(name="Fresh", url="https://example.org/fresh.mp3")
    twin[field] = data["groups"][0]["stations"][0][field]
    data["groups"][1]["stations"].append(twin)
    with pytest.raises(ValidationError, match=f"duplicate station {field}"):
        parse_catalogue(data)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.org/a.mp3",
        "https://example.org/a b.mp3",
        'https://example.org/a".mp3',
        "https://example.org/a'.mp3",
        "https://example.org/a!.mp3",
        "https://example.org/a`.mp3",
        "https://example.org/a\\b.mp3",
        "https://user:pw@example.org/a.mp3",
    ],
)
def test_a_url_breaking_the_station_rules_is_refused(url):
    data = raw()
    data["groups"][0]["stations"][0]["url"] = url
    with pytest.raises(ValidationError):
        parse_catalogue(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_url", "http://example.org/terms"),
        ("operator", " "),
        ("bitrate_kbps", 0),
        ("official", "maybe"),
        ("extra", 1),
    ],
)
def test_catalogue_only_fields_are_checked(field, value):
    data = raw()
    data["groups"][0]["stations"][0][field] = value
    with pytest.raises(ValidationError):
        parse_catalogue(data)


def test_an_unknown_or_repeated_group_is_refused():
    data = raw()
    data["groups"][0]["id"] = "polka"
    with pytest.raises(ValidationError):
        parse_catalogue(data)
    data = raw()
    data["groups"][1]["id"] = data["groups"][0]["id"]
    with pytest.raises(ValidationError, match="duplicate group id"):
        parse_catalogue(data)


@pytest.mark.parametrize("field", ["name", "url", "genres", "language"])
def test_a_missing_station_field_is_reported_as_required(field):
    data = raw()
    del data["groups"][0]["stations"][0][field]
    with pytest.raises(ValidationError, match=f"field required: {field}"):
        parse_catalogue(data)
