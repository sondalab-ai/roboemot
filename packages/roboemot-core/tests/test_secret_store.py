"""Secret store, ``${secret:name}`` route keys and the CLI (SPEC 8 Routes, 12 Data)."""

from __future__ import annotations

import io
import json
import logging
import os
import stat
import threading

import httpx
import pytest
import yaml
from roboemot import cli, paths
from roboemot.config import ConfigStack, GuardrailError
from roboemot.config.schema import LLMRoute, RoboEmotConfig, SafetyConfig
from roboemot.llm import NoRouteAvailable, Router, TextDelta
from roboemot.llm.conformance import RouteKeyUnavailable, check_route
from roboemot.llm.hub import RouterHub
from roboemot.secret_store import SECRET_REF, SecretStore, SecretUnavailable
from roboemot.webui.routes import RouteChecker
from test_llm_conformance import Fake
from test_llm_router import Server, collect, route

KEY = "sk-planted-1f9d3c"


@pytest.fixture
def store(tmp_path) -> SecretStore:
    return SecretStore(tmp_path / "data" / "secrets.json")


def mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def cloud(**kw):
    return route("cloud", "cloud.example", api_key="${secret:cloud_key}", **kw)


def mock_client(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------ the store


def test_set_get_list_remove_round_trip(store):
    store.set("cloud_key", KEY)
    store.set("other.key-2", "v2")
    assert store.get("cloud_key") == KEY
    assert store.names() == ["cloud_key", "other.key-2"] and store.has("cloud_key")
    assert store.remove("cloud_key") is True and store.remove("cloud_key") is False
    assert store.names() == ["other.key-2"]


def test_file_is_0600_in_a_private_folder_and_no_temp_file_is_left(store):
    store.set("cloud_key", KEY)
    store.set("cloud_key", KEY + "2")
    assert mode(store.path) == 0o600 and mode(store.path.parent) == 0o700
    assert [p.name for p in store.path.parent.iterdir()] == ["secrets.json"]


def test_the_file_stays_0600_with_a_permissive_umask(store):
    old = os.umask(0)
    try:
        store.set("cloud_key", KEY)
    finally:
        os.umask(old)
    assert mode(store.path) == 0o600


def test_a_failed_write_keeps_the_old_file_and_leaves_no_temp(store, monkeypatch):
    store.set("cloud_key", KEY)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.set("cloud_key", "new")
    monkeypatch.undo()
    assert store.get("cloud_key") == KEY
    assert [p.name for p in store.path.parent.iterdir()] == ["secrets.json"]


def test_the_last_removal_deletes_the_file(store):
    store.set("a", "1")
    store.remove("a")
    assert not store.path.exists() and store.names() == []


@pytest.mark.parametrize("bad", ["", "a b", "a/b", "${x}", "é"])
def test_names_follow_the_route_key_pattern(store, bad):
    with pytest.raises(ValueError):
        store.set(bad, "v")


@pytest.mark.parametrize("bad", ["", "x" * 5000, "a\x00b"])
def test_values_are_bounded(store, bad):
    with pytest.raises(ValueError):
        store.set("n", bad)


@pytest.mark.parametrize("bits", [0o640, 0o604, 0o660, 0o644, 0o666])
def test_a_group_or_world_accessible_file_is_refused_without_the_value(store, bits):
    store.set("cloud_key", KEY)
    store.path.chmod(bits)
    for call in (lambda: store.get("cloud_key"), store.names, lambda: store.set("x", "y")):
        with pytest.raises(SecretUnavailable) as err:
            call()
        assert KEY not in str(err.value) and "readable by group or others" in str(err.value)
    assert store.has("cloud_key") is False
    store.path.chmod(0o600)
    assert store.get("cloud_key") == KEY


def test_a_file_owned_by_someone_else_is_refused(store, monkeypatch):
    store.set("cloud_key", KEY)
    owner = os.stat(store.path).st_uid
    monkeypatch.setattr(os, "getuid", lambda: owner + 1)
    with pytest.raises(SecretUnavailable, match="not owned by this user"):
        store.get("cloud_key")


def test_a_symlink_is_refused(store, tmp_path):
    store.set("cloud_key", KEY)
    link = tmp_path / "link.json"
    link.symlink_to(store.path)
    with pytest.raises(SecretUnavailable):
        SecretStore(link).get("cloud_key")


@pytest.mark.parametrize("content", ["not json " + KEY, "[1]", '{"a": 1}'])
def test_a_malformed_file_is_refused_without_its_content(store, content):
    store.set("x", "y")
    store.path.write_text(content)
    with pytest.raises(SecretUnavailable) as err:
        store.get("a")
    assert KEY not in str(err.value) and KEY not in repr(err.value)


def test_missing_secret_message_names_it_and_the_fix(store):
    with pytest.raises(SecretUnavailable, match="cloud_key.*roboemot secrets set cloud_key"):
        store.get("cloud_key")


def test_store_follows_the_data_dir(tmp_path, monkeypatch):
    assert SecretStore.for_data_dir(str(tmp_path)).path == tmp_path / paths.SECRETS_FILE
    monkeypatch.setenv(paths.ENV_DIR, str(tmp_path / "env"))
    assert SecretStore.for_data_dir().path == tmp_path / "env" / "secrets.json"


# ------------------------------------------------------------------ routes


async def test_missing_secret_makes_the_route_unavailable_with_one_warning(store, caplog):
    """Regression: the router used to send ``Bearer none`` for a secret that was not there."""
    srv = Server()
    with caplog.at_level(logging.DEBUG):
        r = Router(
            [cloud()],
            SafetyConfig(),
            secrets=store.get,
            http_client_factory=mock_client(srv.handler),
        )
        assert r.eligible(False) == []
        with pytest.raises(NoRouteAvailable):
            await collect(r)
    assert srv.requests == []
    warned = [x.getMessage() for x in caplog.records if x.levelno == logging.WARNING]
    assert len(warned) == 1 and "cloud" in warned[0] and "cloud_key" in warned[0]


async def test_a_router_without_a_store_treats_secret_routes_as_unavailable():
    assert Router([cloud()], SafetyConfig()).eligible(False) == []


async def test_an_empty_secret_value_is_unavailable_too(store):
    store.set("cloud_key", "x")
    store.path.write_text(json.dumps({"cloud_key": ""}))
    assert Router([cloud()], SafetyConfig(), secrets=store.get).eligible(False) == []


async def test_an_unsafe_store_makes_the_route_unavailable_and_the_next_route_answers(store):
    store.set("cloud_key", KEY)
    store.path.chmod(0o644)
    srv = Server()
    r = Router(
        [cloud(), route("lan", "lan.local")],
        SafetyConfig(),
        secrets=store.get,
        http_client_factory=mock_client(srv.handler),
    )
    evs = await collect(r)
    assert [h for h, _ in srv.requests] == ["lan.local"]
    assert any(isinstance(e, TextDelta) for e in evs)


async def test_a_set_secret_goes_out_as_the_bearer_token(store):
    store.set("cloud_key", KEY)
    seen = []
    srv = Server()

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return await srv.handler(request)

    r = Router(
        [cloud()], SafetyConfig(), secrets=store.get, http_client_factory=mock_client(handler)
    )
    await collect(r)
    assert seen == [f"Bearer {KEY}"]


def test_the_hub_rebuilds_when_a_secret_changes_on_reload(store):
    cfg = RoboEmotConfig(llm_routes=[cloud().model_dump()])

    def make(c):
        return Router(c.llm_routes, c.safety, secrets=store.get)

    hub = RouterHub(make(cfg), make)
    first = hub.router
    assert hub.sync(cfg) is first and first.eligible(False) == []
    store.set("cloud_key", KEY)
    second = hub.sync(cfg)
    assert second is not first and [r.name for r in second.eligible(False)] == ["cloud"]
    assert hub.sync(cfg) is second
    store.set("cloud_key", KEY + "2")
    assert hub.sync(cfg) is not second


async def test_a_rotated_secret_is_used_by_the_next_sync_without_a_config_change(store):
    """SPEC 8 Secrets: the hub is synced each turn, so a key rotated on disk
    goes out with the next request, with no reload event and no restart; budgets are kept."""
    cfg = RoboEmotConfig(llm_routes=[cloud().model_dump()])
    seen = []
    srv = Server()

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return await srv.handler(request)

    def make(c):
        return Router(
            c.llm_routes, c.safety, secrets=store.get, http_client_factory=mock_client(handler)
        )

    store.set("cloud_key", KEY)
    hub = RouterHub(make(cfg), make)
    await collect(hub.sync(cfg))
    spent = hub.router.used("cloud")
    assert spent > 0
    store.set("cloud_key", KEY + "2")
    await collect(hub.sync(cfg))
    assert seen == [f"Bearer {KEY}", f"Bearer {KEY}2"]
    assert hub.router.used("cloud") > spent


def test_a_check_probe_does_not_hide_a_later_secret_from_the_reload(store):
    """The web UI check resolves the key on the shared router; the next reload must still
    rebuild it, or the route stays unavailable until a restart."""
    cfg = RoboEmotConfig(llm_routes=[cloud().model_dump()])

    def make(c):
        return Router(c.llm_routes, c.safety, secrets=store.get)

    hub = RouterHub(make(cfg), make)
    store.set("cloud_key", KEY)
    assert hub.router.key_problem(hub.router.routes[0]) is None
    assert hub.router.eligible(False) == []
    assert [r.name for r in hub.sync(cfg).eligible(False)] == ["cloud"]


@pytest.mark.parametrize("bad", ["-h", "--x", "-"])
def test_names_cannot_start_with_a_dash(store, bad):
    with pytest.raises(ValueError):
        store.set(bad, "v")
    assert SECRET_REF.match("${secret:-h}") is None


@pytest.mark.parametrize(
    "ref", ["${secret: x}", "${SECRET:x}", "${secret:a/b}", "${secret:x", "${x}"]
)
def test_a_malformed_secret_reference_is_rejected_by_the_config(ref):
    with pytest.raises(ValueError):
        LLMRoute(name="r", base_url="http://x/v1", model="m", api_key=ref)
    assert LLMRoute(name="r", base_url="http://x/v1", model="m", api_key="${secret:ok}")
    assert LLMRoute(name="r", base_url="http://x/v1", model="m", api_key="sk-literal")


def test_concurrent_writers_lose_no_update(store):
    names = [f"k{i}" for i in range(16)]
    threads = [threading.Thread(target=store.set, args=(n, f"v{n}")) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.names() == sorted(names)
    assert [p.name for p in store.path.parent.iterdir()] == ["secrets.json"]


def test_a_plain_http_key_to_a_non_local_host_warns_once(store, caplog):
    store.set("cloud_key", KEY)
    with caplog.at_level(logging.WARNING):
        Router([cloud()], SafetyConfig(), secrets=store.get)
    warned = [r.getMessage() for r in caplog.records if "plain http" in r.getMessage()]
    assert len(warned) == 1 and KEY not in warned[0]


def test_the_web_ui_private_router_follows_the_secret(store, tmp_path):
    cfg = RoboEmotConfig(data_dir=str(store.path.parent), llm_routes=[cloud().model_dump()])
    checker = RouteChecker(lambda: cfg)
    first = checker._private_router(cfg)
    assert first.eligible(False) == []
    store.set("cloud_key", KEY)
    second = checker._private_router(cfg)
    assert second is not first and [r.name for r in second.eligible(False)] == ["cloud"]
    assert checker._private_router(cfg) is second


SECRET_ROUTE = {
    "name": "cloud",
    "base_url": "https://api.example.com/v1",
    "api_key": "${secret:cloud_key}",
    "model": "m",
}


def secret_stack() -> ConfigStack:
    return ConfigStack(app={"llm_routes": [SECRET_ROUTE]})


@pytest.mark.parametrize(
    "routes",
    [
        [{**SECRET_ROUTE, "base_url": "https://evil.example.net/v1"}],  # key sent elsewhere
        [SECRET_ROUTE, {**SECRET_ROUTE, "name": "extra", "base_url": "https://evil.net/v1"}],
        [SECRET_ROUTE, {**SECRET_ROUTE, "name": "extra"}],  # a new route reusing the secret
        [{**SECRET_ROUTE, "api_key": "${secret:other}"}],
    ],
)
def test_pointing_a_secret_somewhere_new_needs_admin(routes):
    stack = secret_stack()
    with pytest.raises(GuardrailError):
        stack.set("llm_routes", routes)
    assert stack.effective().llm_routes[0].base_url == SECRET_ROUTE["base_url"]
    stack.set("llm_routes", routes, admin=True)


def test_an_api_key_changing_to_a_reference_needs_admin():
    plain = {**SECRET_ROUTE, "api_key": "none"}
    stack = ConfigStack(app={"llm_routes": [plain]})
    with pytest.raises(GuardrailError):
        stack.set("llm_routes", [SECRET_ROUTE])
    stack.set("llm_routes", [SECRET_ROUTE], admin=True)


def test_unrelated_route_edits_keep_a_secret_route_free_of_admin():
    stack = secret_stack()
    stack.set(
        "llm_routes",
        [{**SECRET_ROUTE, "model": "m2"}, {**SECRET_ROUTE, "name": "p", "api_key": "none"}],
    )


def test_router_repr_has_no_key(store):
    store.set("cloud_key", KEY)
    assert KEY not in repr(Router([cloud()], SafetyConfig(), secrets=store.get))


async def test_check_route_refuses_before_sending_when_the_secret_is_missing(store):
    fake = Fake()
    with pytest.raises(RouteKeyUnavailable, match="cloud_key") as err:
        await check_route(
            cloud(),
            SafetyConfig(),
            secrets=store.get,
            http_client_factory=mock_client(fake.handler),
        )
    assert fake.bodies == [] and KEY not in str(err.value)


async def test_check_route_uses_the_secret_store(store):
    store.set("cloud_key", KEY)
    fake = Fake()
    report = await check_route(
        cloud(), SafetyConfig(), secrets=store.get, http_client_factory=mock_client(fake.handler)
    )
    assert report.ok and set(fake.auth) == {f"Bearer {KEY}"}


# ------------------------------------------------------------------ CLI


@pytest.fixture
def app_yaml(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(tmp_path / "data"),
                "llm_routes": [
                    {
                        "name": "cloud",
                        "base_url": "http://cloud.example/v1",
                        "model": "m",
                        "api_key": "${secret:cloud_key}",
                        "first_token_timeout_s": 0.5,
                    }
                ],
            }
        )
    )
    return path


def test_cli_set_reads_the_value_from_stdin_and_never_prints_it(app_yaml, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(KEY + "\n"))
    assert cli.main(["secrets", "set", "cloud_key", "--app", str(app_yaml)]) == 0
    assert cli.main(["secrets", "list", "--app", str(app_yaml)]) == 0
    out = capsys.readouterr()
    assert out.out.split() == ["secret", "cloud_key", "set", "cloud_key"]
    assert KEY not in out.out + out.err
    path = app_yaml.parent / "data" / "secrets.json"
    assert mode(path) == 0o600 and json.loads(path.read_text()) == {"cloud_key": KEY}


def test_cli_takes_no_value_argument(app_yaml, capsys):
    assert cli.main(["secrets", "set", "a", KEY, "--app", str(app_yaml)]) == 1
    assert KEY not in capsys.readouterr().out
    assert not (app_yaml.parent / "data" / "secrets.json").exists()


def test_cli_remove_and_list(app_yaml, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("v\n"))
    cli.main(["secrets", "set", "a", "--app", str(app_yaml)])
    capsys.readouterr()
    assert cli.main(["secrets", "remove", "a", "b", "--app", str(app_yaml)]) == 0
    assert capsys.readouterr().out.split("\n")[:2] == ["removed: a", "not set: b"]
    assert cli.main(["secrets", "list", "--app", str(app_yaml)]) == 0
    assert capsys.readouterr().out == ""


def test_cli_data_dir_flag_needs_no_config(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("v\n"))
    assert cli.main(["secrets", "set", "k", "--data-dir", str(tmp_path / "d")]) == 0
    assert (tmp_path / "d" / "secrets.json").exists()


def test_cli_refuses_an_unsafe_store(app_yaml, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(KEY + "\n"))
    cli.main(["secrets", "set", "cloud_key", "--app", str(app_yaml)])
    (app_yaml.parent / "data" / "secrets.json").chmod(0o644)
    capsys.readouterr()
    assert cli.main(["secrets", "list", "--app", str(app_yaml)]) == 1
    err = capsys.readouterr().err
    assert "readable by group or others" in err and KEY not in err


def serve(monkeypatch, fake):
    monkeypatch.setattr(cli, "HTTP_CLIENT_FACTORY", mock_client(fake.handler))


def test_llm_check_with_a_missing_secret_fails_and_sends_nothing(app_yaml, monkeypatch, capsys):
    fake = Fake()
    serve(monkeypatch, fake)
    assert cli.main(["llm", "check", "cloud", "--app", str(app_yaml)]) == 1
    assert "cloud_key" in capsys.readouterr().err and fake.bodies == []


def test_llm_check_sends_the_stored_secret(app_yaml, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(KEY + "\n"))
    cli.main(["secrets", "set", "cloud_key", "--app", str(app_yaml)])
    fake = Fake()
    serve(monkeypatch, fake)
    assert cli.main(["llm", "check", "cloud", "--app", str(app_yaml), "--json"]) == 0
    out = capsys.readouterr()
    assert set(fake.auth) == {f"Bearer {KEY}"} and KEY not in out.out + out.err
