"""SPEC 4, Discovery: manifests declare an mDNS service; answering robots are matched to them."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from roboemot.discovery import FoundRobot, ServiceRecord, discover, load_hints, match, resolve_host
from roboemot.manifest import RobotInfo

REPO = Path(__file__).resolve().parents[3]
WIRELESS = {"model": "Reachy Mini Wireless", "robot_name": "Neve", "version": "1.11.0"}


def rec(name="Neve", txt=WIRELESS, addrs=("fe80::1", "192.168.88.198"), stype="_reachy-mini._tcp"):
    return ServiceRecord(stype, name, "reachy-mini.local", addrs, 8000, dict(txt))


def hint(id_, service="_reachy-mini._tcp", **txt) -> RobotInfo:
    return RobotInfo.model_validate(
        {"id": id_, "discovery": {"mdns_service": service, "match_txt": txt}}
    )


def test_wireless_manifest_declares_discovery():
    hints = load_hints(sorted((REPO / "configs/manifests").glob("*.yaml")))
    assert [h.id for h in hints] == ["reachy-mini-wireless"]
    assert hints[0].discovery.match_txt == {"model": "Reachy Mini Wireless"}


def test_match_requires_service_and_txt():
    hints = [hint("wireless", model="Reachy Mini Wireless"), hint("lite", model="Reachy Mini Lite")]
    found = match(
        hints,
        [
            rec("Neve"),
            rec("Lu", txt={"model": "Reachy Mini Lite"}),
            rec("X", txt={"model": "Other"}),
            rec("Head", stype="_desk-head._tcp"),
        ],
    )
    assert [(f.manifest_id, f.record.name) for f in found] == [("lite", "Lu"), ("wireless", "Neve")]


def test_address_prefers_ipv4_then_any_then_host():
    assert FoundRobot("w", rec()).address == "192.168.88.198"
    assert FoundRobot("w", rec(addrs=("fe80::1",))).address == "fe80::1"
    assert FoundRobot("w", rec(addrs=())).address == "reachy-mini.local"


def test_discover_browses_only_declared_service_types():
    asked = []

    def browse(types, timeout_s):
        asked.append((types, timeout_s))
        return [rec()]

    hints = [hint("a"), hint("b", service="_desk-head._tcp")]
    found = discover(hints, timeout_s=0.5, browse=browse)
    assert asked == [(["_desk-head._tcp", "_reachy-mini._tcp"], 0.5)]
    assert [f.manifest_id for f in found] == ["a"]
    assert discover([RobotInfo(id="none")], browse=browse) == [] and len(asked) == 1


def test_bad_service_type_is_rejected():
    with pytest.raises(ValueError, match="mdns_service"):
        hint("x", service="reachy-mini")


def test_resolve_host_orders_ipv4_first_and_strips_scope(monkeypatch):
    infos = [
        (socket.AF_INET6, 1, 6, "", ("fe80::1%en0", 8000, 0, 14)),
        (socket.AF_INET, 1, 6, "", ("192.168.88.198", 8000)),
        (socket.AF_INET, 1, 6, "", ("192.168.88.198", 8000)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: infos)
    assert resolve_host("reachy-mini.local", 8000) == ("192.168.88.198", "fe80::1")


def test_resolve_host_unknown_is_empty(monkeypatch):
    def fail(*a, **k):
        raise socket.gaierror("nope")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    assert resolve_host("nowhere.local", 8000) == ()


def test_cli_discover_json(monkeypatch, capsys, tmp_path):
    import json

    from roboemot import cli, discovery

    monkeypatch.setattr(discovery, "zeroconf_browse", lambda types, t: [rec()])
    rc = cli.main(["discover", "--manifest-dir", str(REPO / "configs/manifests"), "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert rc == 0 and rows[0]["manifest"] == "reachy-mini-wireless"
    assert rows[0]["address"] == "192.168.88.198" and rows[0]["name"] == "Neve"
    assert cli.main(["discover", "--manifest-dir", str(tmp_path)]) == 1
