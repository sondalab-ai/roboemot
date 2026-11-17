// Behaviour of the Live tab's LLM prompts panel (SPEC 11A, 12): a collapsed terminal log of the
// cloud LLM requests, polled from /api/live/prompts with a since-cursor only while it is open and
// the flag is on. Text goes through textContent only. Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { loadPage, settle } from "./harness.mjs";

const POLL_MS = 15;
const WAIT_MS = POLL_MS * 8;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const STATE = {
  poll_ms: 1000,
  mood: { label: "joy", valence: 0.25, arousal: -0.1, intensity: 0.4, circadian: null },
  modes: { asleep: false, paused: [], online: true, people: 0, persona: "default" },
  conversation: { phase: "idle", route: null, elapsed_s: null },
  attention: { count: 0, faces: [] },
  reflexes: { last: null },
  motion: { state: "writing" },
  system: { cpu_temp_c: null },
  video: { enabled: false, poll_ms: 333 },
};
const AUDIO = {
  audio_poll_ms: 1000,
  levels: { mic: { state: "no_data", db: null }, out: { state: "no_data", db: null }, floor_db: null },
  transcript: { enabled: false, epoch: "", rev: 0, min_seq: 0, max_turns: 0, turns: [] },
};
const STRINGS = {
  live_prompts: "LLM prompts",
  live_prompts_off: "Off: webui.live.prompts",
  live_prompts_empty: "No request sent yet.",
  live_prompts_clear: "Clear",
};

const entry = (seq, extra = {}) => ({
  seq, t: 1000 + seq, route: "cloud_chat", via: "gpt-x", label: "chat.completions",
  text: "[system] be kind\n  [user] hi", ...extra,
});
const OFF = { enabled: false, epoch: "", rev: 0, min_seq: 0, max_entries: 0, entries: [] };

// A server that, like the real one, returns only the entries newer than the `after` cursor.
function ringServer(ring) {
  return (body, path) => {
    const q = new URL(path, "http://x").searchParams;
    const after = Number(q.get("after"));
    const stale = after > ring.rev || (q.get("epoch") && q.get("epoch") !== ring.epoch);
    const cursor = stale ? 0 : after;
    return {
      poll_ms: POLL_MS,
      prompts: { enabled: ring.enabled, epoch: ring.epoch, rev: ring.rev, min_seq: ring.all[0]?.seq ?? 0,
        max_entries: 50, entries: ring.all.filter((e) => e.rev > cursor) },
    };
  };
}

async function livePage(t, prompts) {
  const page = await loadPage({
    "GET /api/session": { authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 1000 },
    "GET /api/strings": { language: "en", strings: STRINGS },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/state": STATE,
    "GET /api/live/audio": AUDIO,
    "GET /api/live/prompts": prompts,
    "POST /api/live/prompts/clear": { ok: true },
  });
  t.after(() => page.window.close());
  return page;
}

const openLive = async (page) => {
  page.document.querySelector('#tabs button[data-tab="live"]').click();
  await settle();
};
const toggle = async (page) => {
  page.document.querySelector("#live-prompts-toggle").click();
  await settle();
};
const promptCalls = (page) => page.calls.filter((c) => c.path.startsWith("/api/live/prompts?"));
const rows = (page) => [...page.document.querySelectorAll("#live-prompts-log .live-prompt")];

test("the panel starts collapsed and asks nothing until it is opened", async (t) => {
  const ring = { enabled: true, epoch: "a.0", rev: 1, all: [{ ...entry(1), rev: 1 }] };
  const page = await livePage(t, ringServer(ring));
  await openLive(page);
  const button = page.document.querySelector("#live-prompts-toggle");
  assert.equal(button.getAttribute("aria-expanded"), "false");
  assert.ok(page.document.querySelector("#live-prompts-body").hidden);
  await sleep(WAIT_MS);
  assert.equal(promptCalls(page).length, 0);
  await toggle(page);
  assert.equal(button.getAttribute("aria-expanded"), "true");
  assert.equal(promptCalls(page).length, 1);
});

test("an entry is a terminal header line and its body, whitespace kept", async (t) => {
  const ring = { enabled: true, epoch: "a.0", rev: 1, all: [{ ...entry(1), rev: 1 }] };
  const page = await livePage(t, ringServer(ring));
  await openLive(page);
  await toggle(page);
  const [row] = rows(page);
  const head = row.querySelector(".live-prompt__head").textContent;
  assert.match(head, /^\$ .+ → cloud_chat \(gpt-x\) chat\.completions$/);
  assert.equal(row.querySelector(".live-prompt__body").textContent, "[system] be kind\n  [user] hi");
  const log = page.document.querySelector("#live-prompts-log");
  assert.equal(log.getAttribute("role"), "log");
  assert.ok(log.getAttribute("aria-label"));
});

test("the poll sends the cursor, appends new entries and updates a grown audio summary", async (t) => {
  const ring = { enabled: true, epoch: "a.0", rev: 1, all: [{ ...entry(1), rev: 1 }] };
  const page = await livePage(t, ringServer(ring));
  await openLive(page);
  await toggle(page);
  ring.all.push({ ...entry(2, { label: "audio", text: "[audio 10 bytes in 1 frame]" }), rev: 2 });
  ring.rev = 2;
  await sleep(WAIT_MS);
  ring.all[1] = { ...ring.all[1], text: "[audio 20 bytes in 2 frames]", rev: 3 };
  ring.rev = 3;
  await sleep(WAIT_MS);
  const texts = rows(page).map((r) => r.querySelector(".live-prompt__body").textContent);
  assert.deepEqual(texts, ["[system] be kind\n  [user] hi", "[audio 20 bytes in 2 frames]"]);
  const later = promptCalls(page).at(-1).path;
  assert.ok(later.includes("after=3") && later.includes("epoch=a.0"), later);
});

test("hostile text is shown as text", async (t) => {
  const evil = '<img src=x onerror="window.hacked=1">';
  const ring = { enabled: true, epoch: "a.0", rev: 1, all: [{ ...entry(1, { text: evil, route: evil }), rev: 1 }] };
  const page = await livePage(t, ringServer(ring));
  await openLive(page);
  await toggle(page);
  const log = page.document.querySelector("#live-prompts-log");
  assert.equal(log.querySelector("img"), null);
  assert.ok(log.textContent.includes(evil));
  assert.equal(page.window.hacked, undefined);
});

test("the page script never builds markup from prompt text", () => {
  const src = readFileSync(new URL("../../src/roboemot/webui/static/live.js", import.meta.url), "utf8");
  assert.ok(!/innerHTML|outerHTML|insertAdjacentHTML/.test(src));
});

test("with the flag off the panel names the setting, hides Clear and stops polling", async (t) => {
  const page = await livePage(t, { poll_ms: POLL_MS, prompts: OFF });
  await openLive(page);
  await toggle(page);
  await sleep(WAIT_MS);
  assert.equal(promptCalls(page).length, 1, "no retry while the flag is off");
  const off = [...page.document.querySelectorAll("#live-prompts-body .hint")].find((p) => !p.hidden && p.textContent.includes("webui.live.prompts"));
  assert.ok(off, "the hint names the setting");
  assert.ok(page.document.querySelector("#live-prompts-clear").hidden);
  await toggle(page); // close
  await toggle(page); // reopen: asks again
  assert.equal(promptCalls(page).length, 2);
});

test("closing the panel stops the poll and drops the drawn text", async (t) => {
  const ring = { enabled: true, epoch: "a.0", rev: 1, all: [{ ...entry(1), rev: 1 }] };
  const page = await livePage(t, ringServer(ring));
  await openLive(page);
  await toggle(page);
  assert.equal(rows(page).length, 1);
  await toggle(page);
  const n = promptCalls(page).length;
  await sleep(WAIT_MS);
  assert.equal(promptCalls(page).length, n);
  assert.equal(rows(page).length, 0);
});

test("leaving the tab stops the poll", async (t) => {
  const ring = { enabled: true, epoch: "a.0", rev: 1, all: [{ ...entry(1), rev: 1 }] };
  const page = await livePage(t, ringServer(ring));
  await openLive(page);
  await toggle(page);
  page.document.querySelector('#tabs button[data-tab="debug"]').click();
  await settle();
  const n = promptCalls(page).length;
  await sleep(WAIT_MS);
  assert.equal(promptCalls(page).length, n);
});

test("the Clear button posts and empties the log", async (t) => {
  const ring = { enabled: true, epoch: "a.0", rev: 1, all: [{ ...entry(1), rev: 1 }] };
  const page = await livePage(t, ringServer(ring));
  await openLive(page);
  await toggle(page);
  page.document.querySelector("#live-prompts-clear").click();
  await settle();
  const post = page.calls.find((c) => c.method === "POST" && c.path === "/api/live/prompts/clear");
  assert.ok(post, "posted");
  assert.equal(rows(page).length, 0);
});

test("a new entry scrolls to the bottom only when the log was at the bottom", async (t) => {
  const ring = { enabled: true, epoch: "a.0", rev: 1, all: [{ ...entry(1), rev: 1 }] };
  const page = await livePage(t, ringServer(ring));
  await openLive(page);
  await toggle(page);
  const log = page.document.querySelector("#live-prompts-log");
  Object.defineProperty(log, "scrollHeight", { configurable: true, get: () => 500 });
  Object.defineProperty(log, "clientHeight", { configurable: true, get: () => 100 });
  log.scrollTop = 100; // the reader scrolled up
  ring.all.push({ ...entry(2), rev: 2 });
  ring.rev = 2;
  await sleep(WAIT_MS);
  assert.equal(log.scrollTop, 100);
  log.scrollTop = 400; // back at the bottom
  ring.all.push({ ...entry(3), rev: 3 });
  ring.rev = 3;
  await sleep(WAIT_MS);
  assert.equal(log.scrollTop, 500);
});
