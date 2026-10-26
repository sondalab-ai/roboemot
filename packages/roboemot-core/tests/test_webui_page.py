"""The served page: kit wiring, offline links, no colour literals, a string for every key."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from roboemot.config.schema import RoboEmotConfig
from roboemot.diagnostics import Diagnostics
from roboemot.webui import make_server
from roboemot.webui import server as server_module
from roboemot.webui.strings import EN, IT
from test_webui import PIN, Client, FakeService

STATIC = server_module.STATIC_DIR
KIT = server_module.VENDOR_ROUTE


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    cfg = RoboEmotConfig(webui={"bind": "127.0.0.1", "port": 0})
    srv = make_server(FakeService(), Diagnostics(), cfg, tmp_path_factory.mktemp("page"))
    srv.set_pin(PIN)
    srv.start()
    yield Client(srv.port)
    srv.stop()


class _Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.scripts: list[tuple[str, str | None]] = []
        self.root: dict[str, str | None] = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.root = a
        if tag == "link" and a.get("rel") == "stylesheet":
            self.links.append(a["href"])
        if tag == "script":
            self.scripts.append((a["src"], a.get("type")))


def _page(client) -> _Refs:
    refs = _Refs()
    refs.feed(client.request("GET", "/").body.decode())
    return refs


def test_root_carries_the_product_and_a_theme(client):
    root = _page(client).root
    assert root["data-sl-product"] == "roboemot"
    assert root["data-sl-theme"] in {"light", "dark", "high-contrast"}


def test_stylesheets_load_kit_then_themes_then_local(client):
    links = _page(client).links
    order = [
        "tokens.css",
        "components.css",
        "themes/light.css",
        "themes/dark.css",
        "themes/high-contrast.css",
        "themes/products.css",
        "fonts.css",
        "effects.css",
    ]
    assert links == [KIT + name for name in order] + ["/static/style.css"]


def test_every_linked_file_is_served_from_this_origin(client):
    refs = _page(client)
    for path in refs.links + [src for src, _ in refs.scripts]:
        assert path.startswith("/static/") and "//" not in path, path
        assert client.request("GET", path).status == 200, path


def test_scripts_are_external_and_only_the_effects_loader_is_a_module(client):
    scripts = dict(_page(client).scripts)
    assert scripts == {
        "/static/theme.js": None,
        "/static/settings.js": None,
        "/static/live.js": None,
        "/static/app.js": None,
        "/static/fx.js": "module",
    }


def test_effects_are_wired_from_the_vendored_runtime():
    fx = (STATIC / "fx.js").read_text()
    assert 'from "./vendor/sondalab-ui-kit/effects.js"' in fx
    assert "mount()" in fx
    assert not re.search(r"glass", fx)


def test_local_stylesheet_has_only_the_registered_product_accent():
    css = (STATIC / "style.css").read_text()
    assert not re.search(r"\brgba?\(|\bhsla?\(|\boklch\(", css)
    # the roboemot accent per theme, from the kit's accent-registry.json
    assert sorted(re.findall(r"#[0-9a-fA-F]{3,8}\b", css)) == [
        "#f726b4",
        "#f726b4",
        "#ff48c4",
        "#ff48c4",
    ]
    assert "url(" not in css and "@import" not in css


def test_first_party_scripts_have_no_inline_style_or_html_injection():
    for name in ("app.js", "settings.js", "live.js", "theme.js", "fx.js"):
        text = (STATIC / name).read_text()
        for banned in (
            'setAttribute("style"',
            "innerHTML",
            "insertAdjacentHTML",
            "eval(",
            "document.write",
        ):
            assert banned not in text, (name, banned)
        assert "http://" not in text and "https://" not in text


def test_no_native_confirm_prompt_or_page_level_pin_field_is_left():
    js = (STATIC / "app.js").read_text()
    assert not re.search(r"\b(confirm|prompt|alert)\(", js)
    assert "pinField" not in js and "takePin" not in js
    assert "ask({" in js


def _page_js() -> str:
    return "".join((STATIC / n).read_text() for n in ("app.js", "settings.js", "live.js"))


def _keys_used_by_app() -> set[str]:
    js = _page_js()
    return set(re.findall(r"""\bt\(\s*"([a-z0-9_]+)"\s*[,)]""", js))


def test_every_string_key_the_page_reads_exists_in_both_languages():
    used = _keys_used_by_app() - {
        "consent_cloud"
    }  # filled in by /api/strings from the privacy mode
    assert {"tab_settings", "dialog_cancel", "settings_save", "pin"} <= used
    for lang in (EN, IT):
        assert sorted(k for k in used if k not in lang) == []


@pytest.mark.parametrize(
    "prefix",
    [
        "tab_", "state_", "kind_", "status_", "origin_", "theme_", "settings_source_", "check_",
        "route_error_", "scope_", "conflict_", "photo_", "hint_",
    ],
)  # fmt: skip
def test_dynamic_string_families_are_complete(prefix):
    en = {k for k in EN if k.startswith(prefix)}
    assert en and en == {k for k in IT if k.startswith(prefix)}


def test_every_restart_effect_core_accepts_has_text_in_both_languages():
    from roboemot.webui.robot_section import RESTART_EFFECTS

    for lang in (EN, IT):
        assert [e for e in RESTART_EFFECTS if not lang.get("settings_restart_effect_" + e)] == []
    assert "{timeout_s}" in EN["settings_restart_effect_kill_on_timeout"]
    assert "{timeout_s}" in IT["settings_restart_effect_kill_on_timeout"]
    assert not any("Reachy" in v for v in (*EN.values(), *IT.values()) if "restart" in v.lower())
    assert {k for k in EN if k.startswith("settings_restart_effect_")} == {
        "settings_restart_effect_" + e for e in RESTART_EFFECTS
    }


def test_tabs_are_the_strings_that_exist():
    js = (STATIC / "app.js").read_text()
    block = js[js.index("const TABS = {") :]
    names = re.findall(r"^\s+([a-z]+): [a-zA-Z]+View,", block, re.M)
    assert names == ["enroll", "registry", "catalogue", "routes", "settings", "live", "debug"]
    for name in names:
        assert EN["tab_" + name] and IT["tab_" + name]


def test_settings_client_prechecks_use_server_error_codes():
    js = (STATIC / "settings.js").read_text()
    for code in ("value_type", "value_range", "key_invalid", "key_duplicate"):
        assert f'"{code}"' in js
        assert EN["err_" + code] and IT["err_" + code]


def test_settings_posts_path_value_and_pin_to_the_api():
    js = (STATIC / "settings.js").read_text()
    for route in ("", "/reset", "/reveal", "/entry", "/entry/remove"):
        assert f'"/api/settings{route}"' in js
    assert re.search(r"path: row\.path, value: out\.value, pin", js)
    assert "localStorage" not in js and "sessionStorage" not in js


def test_theme_script_guards_storage_and_reloads_for_high_contrast():
    js = (STATIC / "theme.js").read_text()
    assert js.count("try {") >= 2 and "location.reload()" in js
    assert "prefers-contrast: more" in js and "prefers-color-scheme: dark" in js


def test_fonts_the_page_needs_are_served(client):
    for name in ("geist-latin.woff2", "geist-mono-latin.woff2"):
        assert client.request("GET", f"{KIT}assets/fonts/{name}").status == 200


def test_package_data_lists_the_static_tree():
    assert Path(server_module.__file__).parent.joinpath("static/vendor").is_dir()


def test_effects_follow_the_agreed_families_only():
    js = (STATIC / "app.js").read_text()
    for banned in ("sl-fx-glass", "sl-fx-liquid", "sl-fx-goo", "data-sl-halftone", "sl-fx-glitch"):
        assert banned not in js, banned
    for used in (
        "sl-fx-press",
        "sl-fx-aurora--field",
        "sl-fx-aurora--linear",
        "sl-fx-aurora--wide",
    ):
        assert used in js, used


def test_a_busy_aurora_always_carries_aria_busy():
    js = (STATIC / "app.js").read_text()
    body = js[js.index("function busy(") :].split("\n}\n", 1)[0]
    assert 'data-sl-fx-live", "busy"' in body and 'aria-busy", "true"' in body


def test_only_a_saved_change_fires_the_glitch_burst():
    js = (STATIC / "settings.js").read_text()
    assert js.count("roboemotFx.burst") == 1
    assert "result.changed ? row.path : null" in js


def test_vendored_javascript_is_the_only_place_for_effect_internals():
    server_text = Path(server_module.__file__).read_text()
    assert (
        "img-src 'self' blob:;" in server_text
    )  # blob: for the live preview only; no data: images


def test_dialogs_cover_merge_and_guard_the_pending_request():
    js = (STATIC / "app.js").read_text()
    assert "confirm_merge" in js and EN["confirm_merge"] and IT["confirm_merge"]
    assert 'addEventListener("cancel"' in js and "if (inflight) ev.preventDefault()" in js
    assert not re.search(r"const confirm\b", js)


def test_error_states_are_programmatic_and_tabs_are_linked_to_the_panel():
    js = (STATIC / "app.js").read_text()
    assert js.count('"aria-invalid"') >= 2 and js.count("aria-describedby") >= 3
    assert '"aria-controls": "view"' in js and '"aria-labelledby", "tab-"' in js


def test_string_templates_use_a_replacer_function():
    js = (STATIC / "app.js").read_text()
    assert 'text.replace("{" + k + "}", () => String(v))' in js


def test_retry_hint_strings_exist():
    for lang in (EN, IT):
        assert "{n}" in lang["dialog_retry_in"]


def test_every_error_code_the_server_sends_has_a_string_in_both_languages():
    root = Path(server_module.__file__).parent
    codes: set[str] = set()
    for source in root.glob("*.py"):
        text = source.read_text()
        codes |= set(re.findall(r'ValidationError\(\s*"([a-z_]+)"', text))
        codes |= set(re.findall(r'_fail\(\s*"([a-z_]+)"', text))
        codes |= set(re.findall(r'ApiError\(\s*HTTPStatus\.\w+,\s*"([a-z_]+)"', text))
    assert {"setting_guardrail", "entry_in_lower_layer", "value_range"} <= codes
    for lang in (EN, IT):
        assert sorted(c for c in codes if "err_" + c not in lang) == []


def test_settings_page_strings_are_chrome_only_and_the_legacy_ones_are_gone():
    for lang in (EN, IT):
        assert not [k for k in lang if k.startswith("settings_group_")]
        assert "settings_edit" not in lang and "settings_platform_name" not in lang
    js = _page_js()
    assert "SETTING_GROUPS" not in js and "settingEditor" not in js
