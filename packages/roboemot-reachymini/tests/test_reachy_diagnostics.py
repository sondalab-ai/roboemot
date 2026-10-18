"""SPEC 12A: Reachy Mini debug sources against a mocked daemon."""

from __future__ import annotations

import httpx
from roboemot.diagnostics import Target, collect
from roboemot_reachymini import diagnostics

STATUS = {
    "mode": "wlan",
    "known_networks": ["Casa Barile", "Ufficio"],
    "connected_network": "Casa Barile",
}


def _sections(handler):
    client = httpx.Client(base_url="http://robot:8000", transport=httpx.MockTransport(handler))
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
