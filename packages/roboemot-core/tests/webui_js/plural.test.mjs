// One and many (SPEC 11A, 12): a count in a sentence takes the `_one` text when it is 1, in English
// and Italian alike ("1 setting shown", "1 impostazione mostrata"). Run with `npm test`.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
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

// The page strings as Python holds them (as in language.test.mjs): the real texts, in both languages.
function pageStrings() {
  const path = fileURLToPath(new URL("../../src/roboemot/webui/strings.py", import.meta.url));
  const dump = "import json, runpy, sys; d = runpy.run_path(sys.argv[1]); print(json.dumps(d['STRINGS']))";
  return JSON.parse(execFileSync(process.env.PYTHON || "python3", ["-c", dump, path], { encoding: "utf8" }));
}

const REAL = pageStrings();
const ONE = {
  en: { mem: "RoboEmot memory: 150 MB, 1 thread", report: "1 created, 1 already known.", cleared: "Forgotten: 1.", tombstones: "Cleared: 1.",
    chunks: "chunks: 1", broke: "stream broke; chunks received: 1", short: "reply too short to judge (words: 1)" },
  it: { mem: "Memoria di RoboEmot: 150 MB, 1 thread", report: "1 creato, 1 già noto.", cleared: "Dimenticato: 1.", tombstones: "Rimosso: 1.",
    chunks: "blocchi: 1", broke: "streaming interrotto; blocchi ricevuti: 1", short: "risposta troppo corta per valutarla (parole: 1)" },
};
const MANY = {
  en: { mem: "RoboEmot memory: 150 MB, 21 threads", report: "2 created, 3 already known.", cleared: "Forgotten: 2.", tombstones: "Cleared: 2." },
  it: { mem: "Memoria di RoboEmot: 150 MB, 21 thread", report: "2 creati, 3 già noti.", cleared: "Dimenticati: 2.", tombstones: "Rimossi: 2." },
};

for (const language of ["en", "it"]) {
  test(`a count of 1 reads right in every counted sentence (${language})`, async () => {
    const page = await loadPage({ "GET /api/strings": { language, strings: REAL[language] } });
    const text = (src) => page.run(src);
    const mem = (threads) => text(`liveSystemProcess({ cpu_percent: null, rss_mb: 150, threads: ${threads} })[1][0].textContent`);
    const report = (created, existing) => text(`importReport({ items: ${JSON.stringify([
      ...Array(created).fill({ created: true }), ...Array(existing).fill({ created: false })])}, photo: "none", truncated: false, conflicts: [] })[0].textContent`);
    assert.equal(mem(1), ONE[language].mem);
    assert.equal(mem(21), MANY[language].mem);
    assert.equal(report(1, 1), ONE[language].report);
    assert.equal(report(2, 3), MANY[language].report);
    assert.equal(text('tn("catalogue_cleared", 1)'), ONE[language].cleared);
    assert.equal(text('tn("catalogue_cleared", 2)'), MANY[language].cleared);
    assert.equal(text('tn("tombstones_cleared", 1)'), ONE[language].tombstones);
    assert.equal(text('tn("tombstones_cleared", 2)'), MANY[language].tombstones);
    assert.equal(text('checkReason({ code: "chunks", params: { chunks: 1 } })'), ONE[language].chunks);
    assert.equal(text('checkReason({ code: "stream_broke", params: { chunks: 1 } })'), ONE[language].broke);
    assert.equal(text('checkReason({ code: "too_short", params: { count: 1 } })'), ONE[language].short);
  });
}

test("a missing thread count reads as before and a missing memory draws no line", async () => {
  const page = await loadPage({ "GET /api/strings": { language: "en", strings: REAL.en } });
  assert.equal(page.run("liveSystemProcess({ cpu_percent: null, rss_mb: 150, threads: null })[1][0].textContent"), "RoboEmot memory: 150 MB, null threads");
  assert.equal(page.run("liveSystemProcess({ cpu_percent: null, rss_mb: null, threads: 3 })[1].length"), 0);
});
