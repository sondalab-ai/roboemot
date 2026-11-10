// The sticky kit header (SPEC 11A, 12): the page publishes its height as --app-header-h on the
// root, so the scroll padding and the sticky settings nav sit below it. The header wraps on a
// narrow screen, so the height follows a ResizeObserver. jsdom has no layout and applies no
// stylesheet: the media rule that hides the settings nav below 48rem is not testable here.
// Run with `npm test`.
import assert from "node:assert/strict";
import { test } from "node:test";
import { loadPage } from "./harness.mjs";

const height = (page) => page.document.documentElement.style.getPropertyValue("--app-header-h");

// Gives the header a height the test sets (jsdom lays nothing out).
function headerHeight(window, box) {
  const header = window.document.querySelector(".sl-header");
  header.getBoundingClientRect = () => ({ height: box.h, width: 0, top: 0, left: 0, right: 0, bottom: box.h });
}

test("the header height is published on the root and follows the header's resizes", async () => {
  const box = { h: 96 };
  const observed = [];
  const page = await loadPage({}, {
    before: (window) => {
      headerHeight(window, box);
      window.ResizeObserver = class {
        constructor(fn) { this.fn = fn; }
        observe(node) { observed.push({ node, fn: this.fn }); }
      };
    },
  });
  assert.equal(observed.length, 1);
  assert.equal(observed[0].node, page.document.querySelector(".sl-header"));
  assert.equal(height(page), "96px");
  box.h = 152; // the header wrapped on a narrow screen
  observed[0].fn([]);
  assert.equal(height(page), "152px");
});

test("without ResizeObserver the height is published at start and on a window resize", async () => {
  const box = { h: 80 };
  const page = await loadPage({}, {
    before: (window) => { headerHeight(window, box); delete window.ResizeObserver; },
  });
  assert.equal(height(page), "80px");
  box.h = 120;
  page.window.dispatchEvent(new page.window.Event("resize"));
  assert.equal(height(page), "120px");
});
