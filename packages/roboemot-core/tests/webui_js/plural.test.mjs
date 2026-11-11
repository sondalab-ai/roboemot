// One and many (SPEC 11A, 12): a count in a sentence takes the `_one` text when it is 1, in English
// and Italian alike ("1 setting shown", "1 impostazione mostrata"). Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage } from "./harness.mjs";

const TEXTS = {
  en: { settings_matches: "{n} settings shown", settings_matches_one: "{n} setting shown", live_more: "and {n} more", live_more_one: "and one more" },
  it: { settings_matches: "{n} impostazioni mostrate", settings_matches_one: "{n} impostazione mostrata", live_more: "e altri {n}", live_more_one: "e un altro" },
};

for (const [language, strings] of Object.entries(TEXTS)) {
  test(`tn picks the singular for 1 only (${language})`, async () => {
    const page = await loadPage({ "GET /api/strings": { language, strings } });
    const tn = (key, n) => page.run(`tn(${JSON.stringify(key)}, ${n})`);
    assert.equal(tn("settings_matches", 1), strings.settings_matches_one.replace("{n}", "1"));
    for (const n of [0, 2, 11]) assert.equal(tn("settings_matches", n), strings.settings_matches.replace("{n}", n));
    assert.equal(tn("live_more", 1), strings.live_more_one);
    assert.equal(tn("live_more", 3), strings.live_more.replace("{n}", "3"));
    assert.equal(page.run('tn("live_reflex_last", 1, { name: "x" })'), "live_reflex_last", "a key with no singular stays as it is");
  });
}

function d(path, kind, extra = {}) {
  return {
    path, kind, nullable: false, value: null, default: null, source: "schema", ge: null, le: null, unit: null,
    restart: false, admin_only: false, opt_in: false, conservative: null, description: "", choices: null,
    item_kind: null, length: null, key: null, entry: null, secret: false, private: false, writable: true, removable: null, ...extra,
  };
}

test("the settings count says 1 setting, then 2 settings", async () => {
  const page = await loadPage({
    "GET /api/strings": { language: "it", strings: TEXTS.it },
    "GET /api/settings": { fields: [d("voice.a", "number"), d("other.b", "number")], pin_required: true },
  });
  await page.run("settingsView()");
  const count = page.document.querySelector("#settings-count");
  assert.equal(count.textContent, "2 impostazioni mostrate");
  page.run("settingsState.query = 'voice'; refreshSettings()");
  assert.equal(count.textContent, "1 impostazione mostrata");
});
