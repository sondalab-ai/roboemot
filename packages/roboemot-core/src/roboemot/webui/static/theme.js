"use strict";
// Picks the kit theme (light, dark, high-contrast) before first paint. The choice lives in
// localStorage when it works; otherwise, and for "auto", the OS preferences decide.
// Entering or leaving high contrast reloads the page: the effects runtime refuses to arm there.
(function () {
  var KEY = "roboemot.theme";
  var MODES = ["auto", "light", "dark", "high-contrast"];
  var root = document.documentElement;
  var applied = null; // the theme this page was loaded with; a later flip to or from contrast reloads

  function stored() {
    try {
      var value = localStorage.getItem(KEY);
      return MODES.indexOf(value) >= 0 ? value : "auto";
    } catch (e) {
      return "auto";
    }
  }

  function resolve(mode) {
    if (mode !== "auto") return mode;
    if (window.matchMedia("(prefers-contrast: more)").matches) return "high-contrast";
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function apply() {
    var next = resolve(stored());
    root.setAttribute("data-sl-theme", next);
    var flipped = applied !== null && applied !== next;
    var contrast = applied === "high-contrast" || next === "high-contrast";
    applied = next;
    if (flipped && contrast) window.location.reload();
  }

  window.roboemotTheme = {
    modes: MODES,
    mode: stored,
    set: function (mode) {
      if (MODES.indexOf(mode) < 0) return;
      try {
        localStorage.setItem(KEY, mode);
      } catch (e) {
        /* storage blocked: the choice lasts for this page only */
      }
      apply();
    },
  };
  apply();
  ["(prefers-contrast: more)", "(prefers-color-scheme: dark)"].forEach(function (query) {
    window.matchMedia(query).addEventListener("change", apply);
  });
})();
