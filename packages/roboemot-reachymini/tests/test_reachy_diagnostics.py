"""SPEC 12A: Reachy Mini debug sources against a mocked daemon."""

from __future__ import annotations

import httpx
from roboemot.diagnostics import Target, collect
from roboemot_reachymini import diagnostics
from roboemot_reachymini.daemon import DaemonClient

STATUS = {
    "mode": "wlan",
    "known_networks": ["Casa Barile", "Ufficio"],
    "connected_network": "Casa Barile",
}


def _sections(handler):
    client = DaemonClient(Target("robot"), transport=httpx.MockTransport(handler))
    return collect(diagnostics.sources(Target("robot"), client))["sections"]


def test_wifi_reports_mode_and_network_and_only_counts_saved_networks():
    seen = []

    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json=STATUS if req.url.path == "/wifi/status" else {})

    wifi = _sections(handler)["robot.wifi"]
    assert "/wifi/status" in seen and "/api/wifi/status" not in seen
    assert wifi == {
        "scope": "robot",
        "ok": True,
        "data": {
            "available": True,
            "mode": "wlan",
            "connected_network": "Casa Barile",
            "known_networks": 2,
        },
    }
    assert "Ufficio" not in str(wifi)


def test_wifi_says_so_on_a_daemon_without_the_router():
    wifi = _sections(lambda req: httpx.Response(404 if "wifi" in req.url.path else 200, json={}))[
        "robot.wifi"
    ]
    assert wifi["ok"] and wifi["data"]["available"] is False


def test_wifi_failure_is_one_error_section_not_a_broken_report():
    sections = _sections(
        lambda req: httpx.Response(500 if "wifi" in req.url.path else 200, json={})
    )
    assert sections["robot.wifi"]["ok"] is False
    assert sections["robot.daemon"]["ok"] is True


def test_wifi_summary_tolerates_unexpected_shapes():
    odd = diagnostics.wifi_summary(
        {"mode": 3, "connected_network": 7, "known_networks": "Casa,Ufficio"}
    )
    assert odd == {
        "available": True,
        "mode": None,
        "connected_network": None,
        "known_networks": None,
    }
    assert diagnostics.wifi_summary(["x"])["known_networks"] is None
    assert diagnostics.wifi_summary({"known_networks": [], "connected_network": None}) == {
        "available": True,
        "mode": None,
        "connected_network": None,
        "known_networks": 0,
    }


def test_the_daemon_section_says_whether_there_is_a_hardware_id_never_its_value():
    """SR-4 fix round 1, I2: the identifier is data behind privacy.realtime_platform_id, so the
    debug report shows only that the daemon has one (not the value, not a hash; SPEC 12, 12A)."""
    hw = "c0ffee5e171e1d42"

    def handler(req):
        if req.url.path == "/api/daemon/status":
            return httpx.Response(200, json={"version": "1.12.0", "hardware_id": hw})
        return httpx.Response(404, json={"detail": "Not Found"})

    sections = _sections(handler)
    daemon_section = sections["robot.daemon"]
    data = daemon_section.get("data", daemon_section)
    assert "hardware_id" not in data and data["hardware_id_present"] is True
    assert data["version"] == "1.12.0"
    assert hw not in repr(sections)


def test_without_a_hardware_id_the_daemon_section_says_so():
    def handler(req):
        return httpx.Response(200, json={"version": "1.12.0", "hardware_id": None})

    section = _sections(handler)["robot.daemon"]
    assert section.get("data", section)["hardware_id_present"] is False
