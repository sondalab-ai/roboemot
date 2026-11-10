// Turns on the kit's effects layer (aurora, press, glitch). Off in high contrast and under
// reduced motion (mount() declines there); the page is complete without it. The runtime and the
// modules it imports load once the browser is idle after the first paint, not before it: until
// then `burst` does nothing. Views render later: mount() watches for them.
const FX_IDLE_TIMEOUT_MS = 3000; // load by then even if the browser never goes idle
const fx = { burst() {} };
window.roboemotFx = fx;

// Loads the kit's effects runtime and mounts it; a failure leaves the page without effects.
async function loadEffects() {
  try {
    const kit = await import("./vendor/sondalab-ui-kit/effects.js");
    kit.mount();
    fx.burst = kit.burst;
  } catch (e) {
    /* effects are optional */
  }
}

if (typeof window.requestIdleCallback === "function") {
  window.requestIdleCallback(loadEffects, { timeout: FX_IDLE_TIMEOUT_MS });
} else {
  window.addEventListener("load", () => setTimeout(loadEffects, 0), { once: true });
}
