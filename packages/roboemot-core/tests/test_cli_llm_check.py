"""`roboemot llm check <route>`: output and exit codes (SPEC 8)."""

from __future__ import annotations

import json

import httpx
import pytest
import yaml
from roboemot import cli
from test_llm_conformance import ENGLISH, Fake, reply, text_chunks


@pytest.fixture
def app_yaml(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "llm_routes": [
                    {
                        "name": "lan",
                        "base_url": "http://lan.local/v1",
                        "model": "m",
                        "first_token_timeout_s": 0.5,
                        "daily_token_budget": 1_000,
                    }
                ]
            }
        )
    )
    return path


def serve(monkeypatch, fake: Fake) -> None:
    monkeypatch.setattr(
        cli,
        "HTTP_CLIENT_FACTORY",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
    )


def argv(app_yaml, *extra: str) -> list[str]:
    return ["llm", "check", "lan", "--app", str(app_yaml), *extra]


def test_exit_zero_when_every_check_passes(app_yaml, monkeypatch, capsys):
    serve(monkeypatch, Fake())
    assert cli.main(argv(app_yaml)) == 0
    out = capsys.readouterr().out
    assert "pass  streaming" in out and "route lan (m): ok" in out


def test_exit_one_on_a_failed_check_with_json(app_yaml, monkeypatch, capsys):
    serve(monkeypatch, Fake(plain=reply(text_chunks(ENGLISH))))
    assert cli.main(argv(app_yaml, "--json")) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert {c["name"]: c["status"] for c in report["checks"]}["italian"] == "fail"
    assert "api_key" not in json.dumps(report)


def test_unknown_route_is_an_error(app_yaml, capsys):
    assert cli.main(["llm", "check", "nope", "--app", str(app_yaml)]) == 1
    assert "no route named 'nope'" in capsys.readouterr().err


def test_spent_budget_is_refused(app_yaml, monkeypatch, capsys):
    app_yaml.write_text(app_yaml.read_text().replace("1000", "0"))
    fake = Fake()
    serve(monkeypatch, fake)
    assert cli.main(argv(app_yaml)) == 1
    assert "budget" in capsys.readouterr().err and fake.bodies == []


def test_the_check_sends_the_configured_system_prompt(app_yaml, monkeypatch):
    data = yaml.safe_load(app_yaml.read_text())
    app_yaml.write_text(yaml.safe_dump({**data, "persona": {"prompt": "PERSONA-4c1d"}}))
    fake = Fake()
    serve(monkeypatch, fake)
    cli.main(argv(app_yaml))
    assert "PERSONA-4c1d" in fake.bodies[0]["messages"][0]["content"]
