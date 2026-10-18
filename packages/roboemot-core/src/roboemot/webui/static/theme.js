"use strict";
// Picks the kit theme (light, dark, high-contrast) before first paint. The choice lives in
// localStorage when it works; otherwise, and for "auto", the OS preferences decide.
// Entering or leaving high contrast reloads the page: the effects runtime refuses to arm there.
(function () {
  var KEY = "roboemot.theme";
  var MODES = ["auto", "light", "dark", "high-contrast"];
  var root = document.documentElement;

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
    var before = root.getAttribute("data-sl-theme");
    var next = resolve(stored());
    root.setAttribute("data-sl-theme", next);
    if (before && before !== next && (before === "high-contrast" || next === "high-contrast")) {
      window.location.reload();
    }
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
