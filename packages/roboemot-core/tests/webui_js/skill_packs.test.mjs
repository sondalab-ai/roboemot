// The Skills section of the Tools tab (SPEC 11A, 12; design 2026-11-24-skill-packs 5.6): the
// skill packs with their states and reasons, turning one on after reading its text (PIN and
// confirm), the refusal that names the route, the upload as base64 and the End button.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage, reply, settle } from "./harness.mjs";

const STATE = { poll_ms: 60000, tools: [], items: [], missed: [], playing: null };

const PACKS = {
  packs: [
    { name: "bedtime", state: "on", reason: null, route: null, description: "A bedtime story.", meta: {}, ignored_files: [], size_bytes: 10 },
    { name: "broken", state: "invalid", reason: "no_front_matter", route: null, enabled: true },
    { name: "huge", state: "too_large", reason: null, route: "lan", description: "Big.", meta: {}, ignored_files: [], size_bytes: 9000 },
    { name: "quiz", state: "off", reason: null, route: null, description: "A quiz.", meta: { author: "me" }, ignored_files: ["run.py"], size_bytes: 50 },
  ],
  active: "bedtime",
  tools_listed: true,
};

const EN = {
  tab_tools: "Tools", skills_title: "Skills", skills_state_on: "On", skills_state_off: "Off",
  skills_state_invalid: "Invalid", skills_state_too_large: "Too large", skills_too_large: "Too large for route {route}.",
  skills_reason_no_front_matter: "No front matter.", skills_turn_on: "Turn on", skills_turn_off: "Turn off",
  skills_end: "End", err_invalid: "The skill file is not valid.", skills_active: "Active now: {name}", skills_ignored: "not used: {files}",
  err_too_large_for_route: "Too large for route {route}.", pin: "PIN",
};

async function page(t, routes = {}) {
  const p = await loadPage({
    "GET /api/session": () => ({ authenticated: true, csrf: "tok", pin_set: true, status_poll_ms: 60000 }),
    "GET /api/strings": { language: "en", strings: EN },
    "GET /api/enrollment/status": { state: "idle", frames_collected: 0, frames_needed: 0, hints: [] },
    "GET /api/live/viewers": { video_viewers_active: false, poll_ms: 60000 },
    "GET /api/settings": { fields: [], robot: null, pin_required: true },
    "GET /api/settings/voice": reply(404, { error: "not_found" }),
    "GET /api/tools/state": STATE,
    "GET /api/tools/radio/catalogue": reply(404, { error: "not_found" }),
    "GET /api/tools/skill_packs": PACKS,
    "GET /api/tools/skill_packs/quiz": { name: "quiz", description: "A quiz.", body: "QUIZ-BODY-TEXT" },
    ...routes,
  });
  t.after(() => p.window.close());
  p.document.querySelector('#tabs button[data-tab="tools"]').click();
  for (let i = 0; i < 6; i++) await settle();
  return p;
}

async function submitDialog(p, pin) {
  const dialog = p.document.querySelector("dialog[open]");
  const field = dialog.querySelector("#dialog-pin");
  if (field) field.value = pin;
  dialog.querySelector("form").dispatchEvent(new p.window.Event("submit", { cancelable: true }));
  for (let i = 0; i < 6; i++) await settle();
}

const rowOf = (p, name) => p.document.querySelector(`#skills-list li[data-skill="${name}"]`);
const buttonIn = (node, label) => [...node.querySelectorAll("button")].find((b) => b.textContent === label);

test("renders states and reasons", async (t) => {
  const p = await page(t);
  assert.match(rowOf(p, "bedtime").textContent, /On/);
  assert.match(rowOf(p, "broken").textContent, /Invalid.*No front matter\./);
  assert.match(rowOf(p, "huge").textContent, /Too large for route lan\./);
  assert.match(rowOf(p, "quiz").textContent, /not used: run\.py/);
  assert.equal(buttonIn(rowOf(p, "broken"), "Turn on"), undefined, "an invalid skill cannot be turned on");
});

test("turning on shows the text and asks for PIN and confirm", async (t) => {
  let sent = null;
  const p = await page(t, { "POST /api/tools/skill_packs/quiz/enable": (body) => { sent = body; return { ok: true }; } });
  buttonIn(rowOf(p, "quiz"), "Turn on").click();
  for (let i = 0; i < 6; i++) await settle();
  const dialog = p.document.querySelector("dialog[open]");
  assert.ok(dialog, "a dialog opens");
  assert.match(dialog.textContent, /QUIZ-BODY-TEXT/);
  await submitDialog(p, "1234567");
  assert.deepEqual(sent, { on: true, pin: "1234567", confirm: true });
});

test("shows refusal message with route name", async (t) => {
  const p = await page(t, {
    "POST /api/tools/skill_packs/quiz/enable": reply(409, { error: "too_large_for_route", route: "cloud" }),
  });
  buttonIn(rowOf(p, "quiz"), "Turn on").click();
  for (let i = 0; i < 6; i++) await settle();
  await submitDialog(p, "1234567");
  const error = p.document.querySelector("#dialog-error");
  assert.equal(error.textContent, "Too large for route cloud.");
});

test("upload sends filename and base64", async (t) => {
  let sent = null;
  const p = await page(t, { "POST /api/tools/skill_packs/upload": (body) => { sent = body; return { name: "quiz" }; } });
  const file = new p.window.File(["---\nname: quiz\n---\nx"], "quiz.md", { type: "text/markdown" });
  p.run("skillsUploadFile")(file);
  for (let i = 0; i < 10; i++) await settle();
  await submitDialog(p, "1234567");
  assert.equal(sent.filename, "quiz.md");
  assert.equal(p.window.atob(sent.data_b64), "---\nname: quiz\n---\nx");
  assert.equal(sent.replace, false);
  assert.equal(sent.confirm, true);
});

test("active pack shows End", async (t) => {
  let ended = false;
  const p = await page(t, { "POST /api/tools/skill_packs/end": () => { ended = true; return { ended: true }; } });
  const active = p.document.querySelector("#skills-active");
  assert.match(active.textContent, /Active now: bedtime/);
  buttonIn(active, "End").click();
  for (let i = 0; i < 6; i++) await settle();
  await submitDialog(p, "1234567");
  assert.ok(ended);
});

test("an enabled invalid row offers Turn off", async (t) => {
  const p = await page(t);
  assert.ok(buttonIn(rowOf(p, "broken"), "Turn off"));
});

test("a refused upload says why", async (t) => {
  const p = await page(t, {
    "POST /api/tools/skill_packs/upload": reply(422, { error: "invalid", reason: "no_front_matter" }),
  });
  const file = new p.window.File(["x"], "quiz.md", { type: "text/markdown" });
  p.run("skillsUploadFile")(file);
  for (let i = 0; i < 10; i++) await settle();
  await submitDialog(p, "1234567");
  assert.equal(p.document.querySelector("#dialog-error").textContent, "The skill file is not valid. No front matter.");
});
