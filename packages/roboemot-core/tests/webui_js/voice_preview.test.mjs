// The voice picker and its live preview (SPEC 8, Voice preview; SPEC 12): the route form and the
// Settings voice row offer the protocol's known voices, a play button posts the route and the voice
// and plays the WAV with Web Audio, a failure is worded from its code, and nothing plays on load.
// Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

const VOICES = { gemini_live: ["Kore", "Puck"], openai_realtime: ["Serena", "Aiden"] };
const ROUTE = {
  name: "gemini", model: "gemini-3.8-live", endpoint: "generativelanguage.googleapis.com", base_url: null,
  locality: "cloud", kind: "realtime", protocol: "gemini_live", voice: "Puck", key: "secret",
  secret: { name: "gemini_key", set: true }, running: false, last_check: null, session: null, removable: true,
};
const POLLEN = { ...ROUTE, name: "pollen", protocol: undefined, voice: undefined };
const STRINGS = {
  language: "en",
  strings: { voice_preview_err_no_audio: "No sample: no audio.", voice_preview_err_1008: "Refused (1008).", voice_preview_failed: "Failed ({code})." },
};
const PATH = "/api/routes/voice_preview";
const WAV = new Uint8Array([82, 73, 70, 70]).buffer;

// A Web Audio stand-in: records decodes and started sources.
function fakeAudio(window, played) {
  window.AudioContext = class {
    constructor() { this.state = "running"; this.destination = {}; played.contexts += 1; }
    async decodeAudioData(buffer) { played.decoded.push(buffer.byteLength); return { duration: 1 }; }
    createBufferSource() {
      return { connect() {}, start() { played.started += 1; }, stop() { played.stopped += 1; } };
    }
  };
}

async function page(routes, extra = {}) {
  const played = { contexts: 0, decoded: [], started: 0, stopped: 0 };
  const p = await loadPage({
    "GET /api/strings": STRINGS,
    "GET /api/routes": { routes, voices: VOICES, name_max_chars: 32 },
    "GET /api/secrets": { secrets: [{ name: "gemini_key", set: true, used_by: ["gemini"] }], max_value_chars: 4096 },
    ...extra,
  }, { before: (w) => fakeAudio(w, played) });
  return { ...p, played };
}

const wav = () => ({ ok: true, status: 200, headers: { get: () => "audio/wav" }, arrayBuffer: async () => WAV, json: async () => ({}) });
const posts = (p) => p.calls.filter((c) => c.method === "POST" && c.path === PATH);
const options = (list) => [...list.querySelectorAll("option")].map((o) => o.value);

test("the route form's voice is a picker of the protocol's voices that still takes any name", async () => {
  const p = await page([ROUTE, POLLEN]);
  await p.run("routesView()");
  const gemini = p.document.querySelector("#route-gemini-voice");
  assert.equal(gemini.getAttribute("list"), "route-gemini-voices");
  assert.deepEqual(options(p.document.querySelector("#route-gemini-voices")), VOICES.gemini_live);
  assert.deepEqual(options(p.document.querySelector("#route-pollen-voices")), VOICES.openai_realtime);
  assert.equal(gemini.tagName, "INPUT", "free text stays allowed");
  const added = p.document.querySelector("#route-new-protocol");
  assert.deepEqual(options(p.document.querySelector("#route-new-voices")), VOICES.openai_realtime);
  added.value = "gemini_live";
  added.dispatchEvent(new p.window.Event("change"));
  assert.deepEqual(options(p.document.querySelector("#route-new-voices")), VOICES.gemini_live);
  assert.equal(p.document.querySelector("#route-new-voice-preview"), null, "a route not added yet has no preview");
});

test("nothing plays on load; the play button posts the route and the typed voice and plays the WAV", async () => {
  const p = await page([ROUTE], { [`POST ${PATH}`]: () => wav() });
  await p.run("routesView()");
  assert.equal(posts(p).length, 0);
  assert.equal(p.played.contexts, 0, "no audio context before a press");
  p.document.querySelector("#route-gemini-voice").value = " Kore ";
  const play = [...p.document.querySelectorAll("button")].find((b) => b.getAttribute("aria-describedby") === "route-gemini-voice-preview");
  play.click();
  assert.ok(play.disabled, "busy while the preview runs");
  assert.match(p.document.querySelector("#route-gemini-voice-preview").textContent, /voice_preview_busy/);
  await settle();
  await settle();
  assert.deepEqual(posts(p)[0].body, { route: "gemini", voice: "Kore" });
  assert.deepEqual(p.played.decoded, [4]);
  assert.equal(p.played.started, 1);
  assert.match(p.document.querySelector("#route-gemini-voice-preview").textContent, /voice_preview_played/);
  assert.equal(play.disabled, false);
});

test("a failed preview is worded by its code, a 1008 close as a refusal, an API error as every call", async () => {
  const answers = [
    reply(502, { error: "preview_failed", code: "no_audio", close_code: null }),
    reply(502, { error: "preview_failed", code: "closed", close_code: 1008 }),
    reply(502, { error: "preview_failed", code: "weird_code", close_code: null }),
    reply(503, { error: "route_busy" }),
  ];
  const p = await page([ROUTE], { [`POST ${PATH}`]: () => answers.shift() });
  await p.run("routesView()");
  const status = p.document.querySelector("#route-gemini-voice-preview");
  const play = [...p.document.querySelectorAll("button")].find((b) => b.getAttribute("aria-describedby") === status.id);
  const said = [];
  for (let i = 0; i < 4; i++) {
    play.click();
    await settle();
    await settle();
    said.push(status.textContent);
  }
  assert.deepEqual(said.slice(0, 3), ["No sample: no audio.", "Refused (1008).", "Failed (weird_code)."]);
  assert.match(said[3], /err_route_busy|err_generic/);
  assert.equal(p.played.started, 0, "no audio on a failure");
});

test("the Settings voice row of a realtime route gets the picker and the play button", async () => {
  const d = (path, kind, extra = {}) => ({
    path, kind, nullable: false, value: null, default: null, source: "default", gt: null, ge: null, le: null,
    unit: null, restart: false, admin_only: false, opt_in: false, conservative: null, description: "", choices: null,
    item_kind: null, length: null, key: null, entry: null, secret: false, private: false, writable: true, removable: null, ...extra,
  });
  const fields = [d("llm_routes.gemini.voice", "text", { value: "Puck", default: "" })];
  const p = await page([ROUTE], { "GET /api/settings": { fields, pin_required: true }, [`POST ${PATH}`]: () => wav() });
  await p.run("settingsView()");
  p.document.querySelector("#grp-core-llm_routes").click();
  await settle();
  await settle();
  const row = p.document.querySelector('[data-path="llm_routes.gemini.voice"]');
  const input = row.querySelector("input");
  const list = row.querySelector("datalist");
  assert.equal(input.getAttribute("list"), list.id);
  assert.deepEqual(options(list), VOICES.gemini_live);
  const play = [...row.querySelectorAll("button")].find((b) => b.getAttribute("aria-describedby") === input.id + "-preview");
  assert.ok(play, "the preview button is in the row");
  play.click();
  await settle();
  await settle();
  assert.deepEqual(posts(p)[0].body, { route: "gemini", voice: "Puck" });
  assert.equal(p.played.started, 1);
});
