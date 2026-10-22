// Behaviour of the Live tab's video card and the viewing banner (SPEC 11A, 12): the card polls
// /api/live/frame.jpg at the rate the state names, only while the tab is open and visible and the
// card is not paused; every picture becomes an object URL on the <img>, the previous one revoked at
// once and all of them when polling stops. The banner polls /api/live/viewers on every tab while
// logged in. Run with `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

const POLL_MS = 15;
const WAIT_MS = POLL_MS * 8;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const STATE = {
  poll_ms: 1000,
  mood: { label: "joy", valence: 0.25, arousal: -0.1, intensity: 0.4, circadian: null },
  modes: { asleep: false, paused: [], online: true, minor_present: false, register: null, persona: "default" },
  conversation: { phase: "idle", route: null, elapsed_s: null },
  attention: { count: 0, faces: [] },
  reflexes: { last: null },
  motion: { state: "writing" },
  system: { cpu_temp_c: null },
  video: { enabled: true, poll_ms: POLL_MS },
};
const AUDIO = {
  audio_poll_ms: 1000,
  levels: { mic: { state: "ok", db: -30 }, out: { state: "ok", db: -60 }, floor_db: null },
  transcript: { enabled: false, epoch: "", rev: 0, min_seq: 0, max_turns: 0, turns: [] },
};
const STRINGS = {
  live_video_pause: "Pause",
  live_video_resume: "Resume",
  live_video_paused: "Paused.",
  live_video_off: "The camera preview is off.",
  live_video_banner: "Live video view active",
  err_video_asleep: "The robot is asleep, so the camera is off.",
  err_video_no_frame: "Waiting for the next camera picture.",
  err_generic: "Something went wrong.",
};
const JPEG = () => reply(200, null, { "Content-Type": "image/jpeg" });

async function videoPage(t, extra = {}) {
  let loggedIn = true;
  const page = await loadPage({
    "GET /api/session": () => ({ authenticated: loggedIn, csrf: "tok", pin_set: true, status_poll_ms: 1000 }),
    "GET /api/strings": { language: "en", strings: STRINGS },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/state": STATE,
    "GET /api/live/audio": AUDIO,
    "GET /api/live/frame.jpg": JPEG,
    "GET /api/live/viewers": { video_viewers_active: false, count: 0, poll_ms: POLL_MS },
    "POST /api/logout": () => { loggedIn = false; return { ok: true }; },
    ...extra,
  });
  t.after(() => page.window.close());
  return page;
}

const frames = (page) => page.calls.filter((c) => c.path === "/api/live/frame.jpg").length;
const viewers = (page) => page.calls.filter((c) => c.path === "/api/live/viewers").length;
const img = (page) => page.document.querySelector("#live-video-img");
const openLive = async (page) => {
  page.document.querySelector('#tabs button[data-tab="live"]').click();
  await settle();
};
const visibility = (page, value) => {
  Object.defineProperty(page.document, "visibilityState", { configurable: true, get: () => value });
  page.document.dispatchEvent(new page.window.Event("visibilitychange"));
};

test("the video card polls at the rate the state names and swaps and revokes object URLs", async (t) => {
  const page = await videoPage(t);
  await sleep(WAIT_MS);
  assert.equal(frames(page), 0, "not before the Live tab is open");
  await openLive(page);
  await sleep(WAIT_MS * 2);
  assert.ok(frames(page) >= 3, "polls repeatedly: " + frames(page));
  const { created, revoked } = page.urls;
  assert.equal(img(page).getAttribute("src"), created.at(-1));
  assert.equal(img(page).hidden, false);
  assert.deepEqual(revoked, created.slice(0, -1), "every earlier URL revoked, the current one kept");
});

test("an error code hides the picture and shows its text", async (t) => {
  const page = await videoPage(t, { "GET /api/live/frame.jpg": () => reply(503, { error: "video_asleep" }) });
  await openLive(page);
  await sleep(WAIT_MS);
  assert.equal(img(page).hidden, true);
  assert.equal(img(page).hasAttribute("src"), false);
  assert.equal(page.document.querySelector("#live-video-text").textContent, STRINGS.err_video_asleep);
  assert.ok(frames(page) <= 2, "a slow retry, not the picture rate: " + frames(page));
});

test("a good frame after waiting for the first one brings the picture and clears the text", async (t) => {
  let ready = false;
  const page = await videoPage(t, {
    "GET /api/live/frame.jpg": () => (ready ? JPEG() : reply(503, { error: "video_no_frame" })),
  });
  await openLive(page);
  await sleep(WAIT_MS);
  assert.equal(img(page).hidden, true);
  ready = true;
  await sleep(WAIT_MS);
  assert.equal(img(page).hidden, false);
  assert.equal(page.document.querySelector("#live-video-text").hidden, true);
});

test("no frame yet retries at the poll rate", async (t) => {
  const page = await videoPage(t, { "GET /api/live/frame.jpg": () => reply(503, { error: "video_no_frame" }) });
  await openLive(page);
  await sleep(WAIT_MS);
  assert.ok(frames(page) >= 3);
  assert.equal(page.document.querySelector("#live-video-text").textContent, STRINGS.err_video_no_frame);
});

test("the video flag off in the state shows the off text and polls no frame", async (t) => {
  const page = await videoPage(t, { "GET /api/live/state": { ...STATE, video: { enabled: false, poll_ms: POLL_MS } } });
  await openLive(page);
  await sleep(WAIT_MS);
  assert.equal(frames(page), 0);
  assert.equal(page.document.querySelector("#live-video-text").textContent, STRINGS.live_video_off);
  assert.equal(page.document.querySelector("#live-video-pause").hidden, true);
});

test("the pause button stops the polling and drops the picture; resume starts again", async (t) => {
  const page = await videoPage(t);
  await openLive(page);
  await sleep(WAIT_MS);
  const pause = page.document.querySelector("#live-video-pause");
  pause.click();
  await settle();
  const at = frames(page);
  await sleep(WAIT_MS);
  assert.equal(frames(page), at, "no polls while paused");
  assert.equal(img(page).hidden, true);
  assert.deepEqual(page.urls.revoked, page.urls.created, "all URLs revoked");
  assert.equal(pause.textContent, "Resume");
  assert.equal(page.document.querySelector("#live-video-text").textContent, "Paused.");
  pause.click();
  await sleep(WAIT_MS);
  assert.ok(frames(page) > at);
  assert.equal(pause.textContent, "Pause");
});

test("a state refresh while paused does not restart the polling", async (t) => {
  const page = await videoPage(t, { "GET /api/live/state": { ...STATE, poll_ms: POLL_MS } });
  await openLive(page);
  await sleep(WAIT_MS);
  page.document.querySelector("#live-video-pause").click();
  await settle();
  const at = frames(page);
  await sleep(WAIT_MS * 2); // several state polls redraw the cards
  assert.equal(frames(page), at);
});

test("hiding the page stops the polling and revokes the URLs; showing it resumes", async (t) => {
  const page = await videoPage(t, { "GET /api/live/state": { ...STATE, poll_ms: POLL_MS } });
  await openLive(page);
  await sleep(WAIT_MS);
  visibility(page, "hidden");
  await settle();
  const at = frames(page);
  await sleep(WAIT_MS);
  assert.equal(frames(page), at);
  assert.deepEqual(page.urls.revoked.sort(), [...page.urls.created].sort());
  visibility(page, "visible");
  await sleep(WAIT_MS * 2);
  assert.ok(frames(page) > at);
  assert.equal(img(page).hidden, false);
});

test("leaving the tab stops the polling and revokes the URLs", async (t) => {
  const page = await videoPage(t);
  await openLive(page);
  await sleep(WAIT_MS);
  page.document.querySelector('#tabs button[data-tab="debug"]').click();
  await settle();
  const at = frames(page);
  await sleep(WAIT_MS);
  assert.equal(frames(page), at);
  assert.deepEqual(page.urls.revoked.sort(), [...page.urls.created].sort());
});

test("logging out stops the polling, revokes the URLs and hides the banner", async (t) => {
  const page = await videoPage(t, { "GET /api/live/viewers": { video_viewers_active: true, count: 1, poll_ms: POLL_MS } });
  await openLive(page);
  await sleep(WAIT_MS);
  assert.equal(page.document.querySelector("#live-video-banner").hidden, false);
  page.document.querySelector("#logout").click();
  await sleep(WAIT_MS);
  const at = frames(page);
  const seen = viewers(page);
  await sleep(WAIT_MS);
  assert.equal(frames(page), at);
  assert.equal(viewers(page), seen);
  assert.deepEqual(page.urls.revoked.sort(), [...page.urls.created].sort());
  assert.equal(page.document.querySelector("#live-video-banner").hidden, true);
});

test("an expired session on a frame goes back to the login", async (t) => {
  let n = 0;
  const page = await videoPage(t, {
    "GET /api/live/frame.jpg": () => (n++ < 1 ? JPEG() : reply(401, { error: "unauthorized" })),
    "GET /api/session": () => ({ authenticated: n < 2, csrf: "tok", pin_set: true, status_poll_ms: 1000 }),
  });
  await openLive(page);
  await sleep(WAIT_MS);
  assert.equal(page.document.querySelector("#tabs").hidden, true);
});

// ---------------------------------------------------------------- banner

test("the banner shows on any tab while someone watches and hides when they stop", async (t) => {
  let active = false;
  const page = await videoPage(t, { "GET /api/live/viewers": () => ({ video_viewers_active: active, count: active ? 1 : 0, poll_ms: POLL_MS }) });
  const banner = page.document.querySelector("#live-video-banner");
  assert.ok(banner.classList.contains("sl-callout"));
  await sleep(WAIT_MS);
  assert.equal(page.document.querySelector("#tabs button[aria-selected=true]").dataset.tab, "enroll");
  assert.equal(banner.hidden, true);
  assert.ok(viewers(page) >= 2, "polled without opening the Live tab");
  active = true;
  await sleep(WAIT_MS);
  assert.equal(banner.hidden, false);
  assert.equal(banner.textContent.trim(), "Live video view active");
  active = false;
  await sleep(WAIT_MS);
  assert.equal(banner.hidden, true);
});

test("the banner poll stops while the page is hidden and restarts when it is shown", async (t) => {
  const page = await videoPage(t);
  await sleep(WAIT_MS);
  visibility(page, "hidden");
  await settle();
  const at = viewers(page);
  await sleep(WAIT_MS);
  assert.equal(viewers(page), at);
  visibility(page, "visible");
  await sleep(WAIT_MS);
  assert.ok(viewers(page) > at);
});

test("no banner poll before login", async (t) => {
  const page = await loadPage({
    "GET /api/session": { authenticated: false, pin_set: true },
    "GET /api/strings": { language: "en", strings: STRINGS },
    "GET /api/live/viewers": { video_viewers_active: true, count: 1, poll_ms: POLL_MS },
  });
  t.after(() => page.window.close());
  await sleep(WAIT_MS);
  assert.equal(viewers(page), 0);
  assert.equal(page.document.querySelector("#live-video-banner").hidden, true);
});
