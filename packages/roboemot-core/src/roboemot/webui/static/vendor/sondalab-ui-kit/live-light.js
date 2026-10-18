/* @sondalab/ui-kit — live-light.js
   Re-scope R3 (docs/specs/2027-improve-aesthetics.md), owner's pick A,
   2026-09-25; cards pulsing-border and edge-breath. The aurora's two work
   readings, drawn as light with body instead of a 1px arc:

     data-sl-fx-live="run"   an agent is working HERE — two bodies of light
                             travel the edge, the accent receding to its depth
                             companion, with a slow smoke in them
     data-sl-fx-live="busy"  something is loading — the edge breathes light
                             inward, its reach set by four waves at co-prime
                             harmonics so it never visibly loops

   One WebGL canvas per live host (a child, under the label), one frame loop
   for all of them. Without WebGL, past MAX_LIVE hosts, on a --linear or --rim
   aurora, and on a determinate one (an inline --slfx-progress), nothing is
   attached and effects.css keeps the CSS busy arc: that is the fallback, and
   it is complete on its own. Under reduced motion one still frame is drawn
   and the loop never starts. High contrast takes every canvas away.

   STILL: data-sl-fx-still on <html> holds the light where it is. The frame
   that sees the flag is drawn and none is asked for after it; canvases and
   contexts stay, showing that frame. A host that attaches or resizes meanwhile
   is drawn once, at the frozen instant. Clearing the flag restarts the loop
   with the time origin moved by the pause, so the light resumes where it
   stopped (as animation-play-state does) instead of jumping ahead.

   COLOUR: no literal. The accent and its depth companion are read off the
   host through a probe element whose colour is the role variable, so a band
   or a product override that remaps them is followed. */

import { roleColour } from "./colour.js";

const HOSTS = '.sl-fx-aurora:is([data-sl-fx-live="busy"], [data-sl-fx-live="run"])';
/** Canvases beyond this fall back to the CSS arc: a WebGL context each, and
 *  browsers cap how many a page may hold. */
const MAX_LIVE = 8;
/** How far the light may reach past the host's box, in CSS px. At 6 the run
 *  light still carried 18% alpha where the canvas ended and stopped on a hard
 *  rectangle; by 16 its bleed is under 0.5%, and FS_RUN fades it to zero here. */
const PAD = 16;
const REDUCE = "(prefers-reduced-motion: reduce)";
const STILL = "data-sl-fx-still";

const VS = "attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}";
const HEAD = `precision highp float;
uniform vec2 uSize; uniform float uR,uT,uPad,uDpr,uDark; uniform vec3 uA,uB;
float sd(vec2 p){vec2 q=abs(p)-uSize*.5+uR;return length(max(q,0.))+min(max(q.x,q.y),0.)-uR;}
float hash(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
float vn(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+1.),f.x),f.y);}
vec2 P(){vec2 fc=gl_FragCoord.xy/uDpr; vec2 full=uSize+2.*uPad; return vec2(fc.x-full.x*.5, full.y*.5-fc.y);}
float perim(vec2 p){ float a=atan(p.y/uSize.y, p.x/uSize.x); return fract(a/6.2831853+.625); }
`;
/** run: two bodies of light travelling the edge, accent receding to depth. */
const FS_RUN = HEAD + `void main(){
  vec2 p=P(); float d=sd(p);
  float band=exp(-abs(d+1.)/3.);
  float bleed=d<0.?exp(d/9.)*.35:exp(-d/3.)*.5;
  float s=perim(p); float light=0.; vec3 acc=vec3(0.);
  for(int k=0;k<2;k++){
    float c=fract(uT*.11+float(k)*.5); float dd=abs(s-c); dd=min(dd,1.-dd);
    float body=exp(-dd*dd/.006);
    light+=body; acc+=body*mix(uA,uB,smoothstep(0.,.08,dd));
  }
  float smoke=.8+.4*vn(p*.022+vec2(uT*.25,-uT*.2));
  float a=clamp((band+bleed)*light*smoke,0.,1.)*(1.-smoothstep(uPad*.5,uPad,d))*(uDark>.5?.95:.8);
  vec3 c=light>0.?acc/max(light,1e-3):uA;
  float dz=(hash(gl_FragCoord.xy+uT)-.5)/255.;
  gl_FragColor=vec4(c*a+dz,a);
}`;
/** busy: the edge breathes light inward, never visibly looping. */
const FS_BUSY = HEAD + `void main(){
  vec2 p=P(); float d=sd(p); if(d>1.){gl_FragColor=vec4(0);return;}
  float s=perim(p)*6.2831853;
  float reach=6.+3.*sin(2.*s+uT*.9)+2.2*sin(3.*s-uT*1.3)+1.4*sin(5.*s+uT*1.7)+1.*sin(7.*s-uT*2.3);
  float x=max(-d,0.)/max(reach,1.);
  float a=exp(-x*x*2.2)*(uDark>.5?.32:.28)*(.75+.25*sin(uT*.9));
  float dz=(hash(gl_FragCoord.xy+uT)-.5)/255.;
  gl_FragColor=vec4(uA*a+dz,a);
}`;

const live = new Map();
let rafId = 0;
let t0 = 0;
/** The frame time the light froze at, or null while it moves. */
let stillAt = null;
let motion = null;
const UNIFORMS = ["uSize", "uR", "uT", "uPad", "uDpr", "uDark", "uA", "uB"];
/** Same margin as effects.js's near marking: a light is drawing before it
 *  crosses the fold, never after. */
const NEAR_MARGIN = "200px";
let nearIO = null;
let sizeRO = null;

/** Whether motion is welcome. The query list is made once: matchMedia builds
 *  a new one on every call, and this is read every frame. */
function moving() {
  motion ??= globalThis.matchMedia?.(REDUCE) ?? { matches: false };
  return !motion.matches;
}

/** The observers every host shares. Off-screen hosts are not drawn at all,
 *  and a host's box and radius are read when it resizes, never per frame.
 *  Without IntersectionObserver every host stays near: fail open. */
function observers() {
  if (!nearIO && typeof IntersectionObserver === "function") {
    nearIO = new IntersectionObserver((entries) => {
      for (const e of entries) {
        const it = live.get(e.target);
        if (it) it.near = e.isIntersecting;
      }
      ensureLoop();
    }, { rootMargin: NEAR_MARGIN });
  }
  if (!sizeRO && typeof ResizeObserver === "function") {
    sizeRO = new ResizeObserver((entries) => {
      for (const e of entries) {
        const it = live.get(e.target);
        if (!it) continue;
        const b = e.borderBoxSize?.[0];
        if (b) { it.w = b.inlineSize; it.h = b.blockSize; }
        it.dirty = true;
      }
      ensureLoop();
    });
  }
}

function compile(gl, fs) {
  const prog = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, VS], [gl.FRAGMENT_SHADER, fs]]) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) return null;
    gl.attachShader(prog, sh);
  }
  gl.linkProgram(prog);
  return gl.getProgramParameter(prog, gl.LINK_STATUS) ? prog : null;
}

/** Whether a host is one this module draws, or one the CSS arc keeps. */
function wanted(el) {
  if (!el.matches(HOSTS) || !el.isConnected) return false;
  if (el.matches(".sl-fx-aurora--linear, .sl-fx-aurora--rim")) return false;
  if (el.getAttribute("style")?.includes("--slfx-progress")) return false;
  if (el.closest('[data-sl-fx="off"]')) return false;
  return el.ownerDocument.documentElement.getAttribute("data-sl-theme") !== "high-contrast";
}

function attach(el) {
  if (live.has(el) || live.size >= MAX_LIVE) return;
  const canvas = el.ownerDocument.createElement("canvas");
  canvas.className = "sl-fx-live__canvas";
  canvas.setAttribute("aria-hidden", "true");
  const gl = canvas.getContext("webgl", { premultipliedAlpha: true, alpha: true });
  if (!gl) return;
  const mode = el.getAttribute("data-sl-fx-live");
  const prog = compile(gl, mode === "run" ? FS_RUN : FS_BUSY);
  if (!prog) return;
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "a");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  el.append(canvas);
  el.classList.add("is-live-lit");
  const u = Object.fromEntries(UNIFORMS.map((n) => [n, gl.getUniformLocation(prog, n)]));
  live.set(el, {
    canvas, gl, u, mode,
    a: roleColour(el, mode === "run" ? "--slfx-light-a" : "--slc-accent"),
    b: roleColour(el, mode === "run" ? "--slfx-light-b" : "--slfx-depth"),
    near: true, dirty: true, w: el.offsetWidth, h: el.offsetHeight, dpr: 0,
  });
  observers();
  nearIO?.observe(el);
  sizeRO?.observe(el);
  ensureLoop();
}

function detach(el) {
  const it = live.get(el);
  if (!it) return;
  nearIO?.unobserve(el);
  sizeRO?.unobserve(el);
  it.canvas.remove();
  it.gl.getExtension("WEBGL_lose_context")?.loseContext();
  el.classList.remove("is-live-lit");
  live.delete(el);
}

/** Reconcile one host: attach, detach, or re-attach when its mode changed. */
function reconcile(el) {
  const it = live.get(el);
  if (it && (!wanted(el) || it.mode !== el.getAttribute("data-sl-fx-live"))) detach(el);
  if (wanted(el)) attach(el);
}

/** Upload what changes only on a resize, a DPR change or a theme switch. */
function setup(el, it, dpr) {
  const { gl, u } = it;
  const w = Math.round((it.w + PAD * 2) * dpr);
  const h = Math.round((it.h + PAD * 2) * dpr);
  if (it.canvas.width !== w || it.canvas.height !== h) {
    it.canvas.width = w;
    it.canvas.height = h;
    gl.viewport(0, 0, w, h);
  }
  const dark = el.ownerDocument.documentElement.getAttribute("data-sl-theme") !== "light";
  gl.uniform2f(u.uSize, it.w, it.h);
  gl.uniform1f(u.uR, Math.min(parseFloat(getComputedStyle(el).borderTopLeftRadius) || 0, it.h / 2, it.w / 2));
  gl.uniform1f(u.uPad, PAD);
  gl.uniform1f(u.uDpr, dpr);
  gl.uniform1f(u.uDark, dark ? 1 : 0);
  gl.uniform3f(u.uA, it.a[0], it.a[1], it.a[2]);
  gl.uniform3f(u.uB, it.b[0], it.b[1], it.b[2]);
  it.dpr = dpr;
  it.dirty = false;
}

/** Draw every near host; per frame that is one uniform and one triangle. */
function draw(now) {
  const t = (now - t0) / 1000;
  const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
  let drawn = 0;
  for (const [el, it] of live) {
    if (!el.isConnected) { detach(el); continue; }
    if (!it.near) continue;
    if (it.dirty || it.dpr !== dpr) setup(el, it, dpr);
    const { gl } = it;
    gl.uniform1f(it.u.uT, t);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    drawn++;
  }
  return drawn;
}

/** Whether the document asks the light to hold still. One attribute read per
 *  frame; every live host shares the loop, so the first one's document says. */
function stilled() {
  for (const el of live.keys()) return el.ownerDocument.documentElement.hasAttribute(STILL);
  return false;
}

function frame(now) {
  rafId = 0;
  const hold = stilled();
  // Held: every frame from here draws the instant it froze at. Released: the
  // origin moves by the pause, so time picks up where it stopped.
  if (hold) stillAt ??= now;
  else if (stillAt !== null) { t0 += now - stillAt; stillAt = null; }
  // Nothing near, nothing to draw: the loop stops, and the observers start it
  // again when a host comes near or resizes. Held, or under reduced motion,
  // the frame is drawn and none follows.
  if (draw(hold ? stillAt : now) && moving() && !hold) rafId = requestAnimationFrame(frame);
}

function ensureLoop() {
  if (rafId) return;
  if (!t0) t0 = performance.now();
  // Reduced motion: one still frame, and no loop.
  rafId = requestAnimationFrame(frame);
}

/**
 * Give every live aurora host under `root` its light, and keep following:
 * hosts added later, a data-sl-fx-live that changes, a theme switch to or
 * from high contrast, and data-sl-fx-still on the root, which holds the light
 * in place while set.
 * @param {Document|Element} root
 * @returns {{stop(): void}}
 */
export function liveLight(root) {
  const doc = root.ownerDocument || root;
  const scan = (n) => {
    if (n instanceof Element && n.matches(".sl-fx-aurora")) reconcile(n);
    for (const el of n.querySelectorAll?.(".sl-fx-aurora") ?? []) reconcile(el);
  };
  scan(root);
  const watch = new MutationObserver((records) => {
    let removed = false;
    for (const r of records) {
      // effects.js writes inline style on lit hosts every frame; only an
      // aurora's own attributes matter here.
      if (r.type === "attributes") { if (r.target.classList.contains("sl-fx-aurora")) reconcile(r.target); continue; }
      for (const n of r.addedNodes) if (n instanceof Element) scan(n);
      removed ||= r.removedNodes.length > 0;
    }
    // A host that left the DOM gives its WebGL context back now, not on a
    // frame that may never run once the loop has stopped.
    if (removed) for (const el of [...live.keys()]) if (!el.isConnected) detach(el);
  });
  watch.observe(root, { subtree: true, childList: true, attributes: true, attributeFilter: ["data-sl-fx-live", "style"] });
  const theme = new MutationObserver((records) => {
    // The still flag changes nothing but whether the loop runs: no host to
    // reconcile, no colour to re-read. Set, one more frame is drawn and the
    // loop stops; cleared, the loop starts again.
    if (records.every((r) => r.attributeName === STILL)) { ensureLoop(); return; }
    for (const el of [...live.keys()]) reconcile(el);
    scan(root);
    // A theme switch re-reads the colours on the next attach.
    for (const [el, it] of live) {
      const mode = it.mode;
      it.a = roleColour(el, mode === "run" ? "--slfx-light-a" : "--slc-accent");
      it.b = roleColour(el, mode === "run" ? "--slfx-light-b" : "--slfx-depth");
      it.dirty = true;
    }
    ensureLoop();
  });
  theme.observe(doc.documentElement, { attributes: true, attributeFilter: ["data-sl-theme", "style", STILL] });
  return {
    stop() {
      watch.disconnect();
      theme.disconnect();
      for (const el of [...live.keys()]) detach(el);
    },
  };
}
