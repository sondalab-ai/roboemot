// Numbers in the page language (SPEC 11A, 12): the Live and Emotions tabs write their numbers with
// the page's decimal separator ("0,62" on an Italian page), a signed value keeps its sign rule.
// Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

const BASE = {
  "GET /api/session": { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 },
  "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
  "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
};

test("num: the page language's separator, fixed decimals, and the sign rule of a signed value", async () => {
  const page = await loadPage({ ...BASE, "GET /api/strings": { language: "it", strings: {} } });
  assert.equal(page.run("num(0.6234, 2)"), "0,62");
  assert.equal(page.run("num(-12.46, 1)"), "-12,5");
  assert.equal(page.run("num(3, 1)"), "3,0");
  assert.equal(page.run("num(0.6, 2, true)"), "+0,60");
  assert.equal(page.run("num(-0.6, 2, true)"), "-0,60");
  assert.equal(page.run("num(0, 2, true)"), "0,00", "zero has no sign");
  page.run("state.lang = 'en'");
  assert.equal(page.run("num(0.6234, 2)"), "0.62", "a formatter per language");
  assert.equal(page.run("num(0.6, 2, true)"), "+0.60");
});

const EMOTIONS = {
  poll_ms: 60000,
  affect: { label: "joy", valence: 0.4, arousal: -0.25, intensity: 0.3, baseline_valence: 0, baseline_arousal: -0.1 },
  register: null,
  persona: { name: "default", effects: {} },
  circadian: null,
  expression: { last_label: null, last_amplitude: null, started: 0 },
  moved: [{ age_s: 1.5, event: "sound.loud", rule: 3, label: null, dv: 0, da: 0.6 }],
  controls: { available: true, refused: null },
  playable: { labels: [], gestures: [] },
  rules: [],
};

test("an Italian page writes the Emotions numbers with a decimal comma", async (t) => {
  const page = await loadPage({
    ...BASE,
    "GET /api/strings": { language: "it", strings: { emo_delta: "valenza {dv}, attivazione {da}", emo_ago: "{n} s fa" } },
    "GET /api/emotions/state": EMOTIONS,
  });
  t.after(() => page.window.close());
  page.document.querySelector('#tabs [data-tab="emotions"]').click();
  await settle();
  await settle();
  const text = page.document.querySelector("#view").textContent;
  assert.ok(text.includes("0,40") && text.includes("-0,25"), text);
  const moved = page.document.querySelector("#emo-moved li").textContent;
  assert.ok(moved.includes("1,5 s fa") && moved.includes("valenza 0,00, attivazione +0,60"), moved);
  const bar = page.document.querySelector('[role=progressbar][aria-label="live_valence"]');
  assert.equal(bar.getAttribute("aria-valuetext"), "0,40");
});

test("an Italian page writes the audio levels with a decimal comma", async (t) => {
  const page = await loadPage({
    ...BASE,
    "GET /api/strings": { language: "it", strings: { live_db: "{n} dBFS", live_floor: "Rumore di fondo: {n} dBFS" } },
    "GET /api/live/state": { poll_ms: 60000, mood: { label: "joy", valence: 0.25, arousal: 0, intensity: 0.4, circadian: null },
      modes: { asleep: false, paused: [], online: true, people: 0, persona: "p" }, conversation: { phase: "idle", route: null, elapsed_s: null },
      attention: { count: 0, faces: [] }, reflexes: { last: null }, motion: { state: "writing" }, system: { cpu_temp_c: null }, video: { enabled: false, poll_ms: 333 } },
    "GET /api/live/audio": { audio_poll_ms: 60000, levels: { mic: { state: "ok", db: -23.4 }, out: { state: "ok", db: -40 }, floor_db: -55.25 },
      transcript: { enabled: false, epoch: "", rev: 0, min_seq: 0, max_turns: 0, turns: [] } },
  });
  t.after(() => page.window.close());
  page.document.querySelector('#tabs [data-tab="live"]').click();
  await settle();
  await settle();
  const levels = page.document.querySelector("#live-levels").textContent;
  assert.ok(levels.includes("-23,4 dBFS") && levels.includes("-40,0 dBFS") && levels.includes("Rumore di fondo: -55,3 dBFS"), levels);
  assert.ok(page.document.querySelector("#live-cards").textContent.includes("0,25"), "the mood card too");
});
