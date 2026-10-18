// Loads the served page (index.html + theme.js + app.js, unmodified) into jsdom with a stubbed
// `fetch`, so the page script runs as in a browser. The vendored kit is not loaded: it only styles.
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { JSDOM } from "jsdom";

const STATIC = new URL("../../src/roboemot/webui/static/", import.meta.url);
const read = (name) => readFileSync(new URL(name, STATIC), "utf8");

/** A fetch-like reply: `status` and a JSON `body`, optional response `headers`. */
export function reply(status, body, headers = {}) {
  return { ok: status >= 200 && status < 300, status, headers: { get: (k) => headers[k] ?? null }, json: async () => body };
}

/**
 * Boots the page. `routes` maps "METHOD /path" to a body, a `reply(...)`, or a function of the
 * parsed request body. Returns the window, the recorded calls, and `run(src)` to evaluate in the page.
 */
export async function loadPage(routes = {}) {
  const html = read("index.html").replace(/<script[\s\S]*?<\/script>/g, "");
  const dom = new JSDOM(html, { url: "http://localhost/", runScripts: "outside-only", pretendToBeVisual: true });
  const { window } = dom;
  const calls = [];
  const table = {
    "GET /api/strings": { language: "en", strings: {} },
    "GET /api/session": { authenticated: false, pin_set: true },
    ...routes,
  };
  window.fetch = async (path, init = {}) => {
    const body = init.body === undefined ? undefined : JSON.parse(init.body);
    calls.push({ method: init.method, path, body });
    const hit = table[`${init.method} ${path}`];
    if (hit === undefined) return reply(404, { error: "not_found" });
    const out = typeof hit === "function" ? await hit(body) : hit;
    return out && out.ok !== undefined && out.json ? out : reply(200, out);
  };
  window.CSS = { escape: (s) => s };
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  if (!window.HTMLDialogElement.prototype.showModal) polyfillDialog(window);
  const ctx = dom.getInternalVMContext();
  const run = (src) => new vm.Script(src).runInContext(ctx);
  run(read("theme.js"));
  run(read("app.js"));
  await settle();
  return { window, document: window.document, calls, run };
}

/** jsdom has no modal dialog: open/close and the `cancel` and `close` events behave like a browser's. */
function polyfillDialog(window) {
  const proto = window.HTMLDialogElement.prototype;
  proto.showModal = function () { this.setAttribute("open", ""); };
  proto.close = function () {
    if (!this.hasAttribute("open")) return;
    this.removeAttribute("open");
    this.dispatchEvent(new window.Event("close"));
  };
}

/** Lets pending promise callbacks and zero-delay timers run. */
export const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/** The WebStorage contents of a window as one string, to look for a secret in. */
export function storageDump(window) {
  const out = [];
  for (const store of [window.localStorage, window.sessionStorage]) {
    for (let i = 0; i < store.length; i++) out.push(store.key(i), store.getItem(store.key(i)));
  }
  out.push(window.document.cookie);
  return out.join("\n");
}
