// The page's feedback (SPEC 11A, 12): `notice()` shows one kit toast in the toast stack that is
// always in the page, a success in its polite region and an error in its alert region. Run with
// `npm test` in this folder.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage } from "./harness.mjs";

const toasts = (page) => [...page.document.querySelectorAll("#notice .sl-toast")];

test("both live regions are in the page, empty, before the first toast", async () => {
  const page = await loadPage();
  const regions = [...page.document.querySelectorAll("#notice > .sl-toast-stack__live")];
  assert.deepEqual(regions.map((r) => [r.getAttribute("aria-live"), r.getAttribute("role")]), [["polite", null], [null, "alert"]]);
  assert.ok(regions.every((r) => r.childNodes.length === 0));
  assert.ok(!page.document.querySelector("main").contains(page.document.getElementById("notice")), "not at the top of the page");
});

test("a success goes to the polite region, an error to the alert region, and \"\" clears", async () => {
  const page = await loadPage();
  page.run('notice("Saved.")');
  assert.equal(page.document.querySelector("#notice-polite .sl-toast--success").textContent, "Saved.");
  page.run('notice("Bad key.", true)');
  assert.equal(toasts(page).length, 1, "a later message replaces the toast");
  assert.equal(page.document.querySelector("#notice-alert .sl-toast--danger").textContent, "Bad key.");
  page.run('notice("")');
  assert.equal(toasts(page).length, 0);
});

test("an error that repeats keeps its toast and is not announced again, even after a dismiss", async () => {
  const page = await loadPage();
  page.run('notice("Cannot reach the robot.", true)');
  const first = toasts(page)[0];
  page.run('notice("Cannot reach the robot.", true)');
  assert.equal(toasts(page)[0], first, "the same node: the alert region did not change");
  first.querySelector(".sl-toast__dismiss").click();
  assert.equal(toasts(page).length, 0);
  page.run('notice("Cannot reach the robot.", true)');
  assert.equal(toasts(page).length, 0, "the poll that keeps failing does not bring it back");
  page.run('notice("Another error.", true)');
  assert.equal(toasts(page)[0].textContent, "Another error.");
  page.run('notice("")');
  page.run('notice("Cannot reach the robot.", true)');
  assert.equal(toasts(page).length, 1, "after a clear the same error is news again");
});

test("a success leaves after its reading time, an error stays", async () => {
  const page = await loadPage();
  const timers = [];
  const real = page.window.setTimeout.bind(page.window);
  page.window.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return real(() => {}, 1e9); };
  page.run('notice("Bad key.", true)');
  assert.equal(timers.length, 0, "no timer for an error");
  page.run('notice("Saved.")');
  assert.deepEqual(timers.map((x) => x.ms), [8000]);
  timers[0].fn();
  assert.equal(toasts(page).length, 0);
});

test("a success the user has focused or points at waits for them", async () => {
  const page = await loadPage();
  const timers = [];
  const real = page.window.setTimeout.bind(page.window);
  page.window.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return real(() => {}, 1e9); };
  page.run('notice("Saved.")');
  const dismiss = page.document.querySelector("#notice .sl-toast__dismiss");
  dismiss.focus();
  timers.shift().fn();
  assert.equal(toasts(page).length, 1, "kept while it has the focus");
  assert.equal(page.document.activeElement, dismiss);
  assert.deepEqual(timers.map((x) => x.ms), [8000], "asked again later");
  dismiss.blur();
  timers.shift().fn();
  assert.equal(toasts(page).length, 0);
});

test("the dismiss button has an accessible name in the page language", async () => {
  const page = await loadPage({ "GET /api/strings": { language: "it", strings: { notice_dismiss: "Chiudi" } } });
  page.run('notice("Salvato.")');
  const dismiss = page.document.querySelector("#notice .sl-toast__dismiss");
  assert.equal(dismiss.getAttribute("aria-label"), "Chiudi");
  assert.equal(dismiss.textContent, "", "an icon only: the name is the label");
  dismiss.click();
  assert.equal(toasts(page).length, 0);
});
