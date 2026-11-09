// The preview: fits the frame to the chosen screen shape, pans / zooms / turns the view,
// and moves / resizes layers directly on the picture.

import { html, useEffect, useRef, useState } from "../vendor/preact-htm.js";
import { Icon } from "./icons.js";
import {
  S, commit, currentAspect, emit, layerLabel, layerValue, resetView, scheduleRender, selectLayer, setPath, setPreviewSize,
  startCompare, stopCompare, useStore,
} from "./store.js";
import { clamp } from "./util.js";

// ---------------------------------------------------------------- view math ----
// Frame axes in plane coordinates: x runs along (cos θ, sin θ), "up" along (-sin θ, cos θ),
// θ = view rotation (counter-clockwise). Screen y points down. `S.view.extent` holds the plane
// half-extents of the image on screen, rendered for `S.view.shownView`.

function frameExtents() {
  const hw = S.state.view?.half_width ?? 1;
  const ext = S.view.extent, shown = S.view.shownView;
  if (ext && shown?.half_width) {
    const k = hw / shown.half_width;
    return { halfW: ext.halfW * k, halfH: ext.halfH * k };
  }
  return { halfW: hw, halfH: hw * (S.state.height / S.state.width) };
}

const axes = (view) => {
  const th = (view.rotation || 0) * Math.PI / 180;
  return { cos: Math.cos(th), sin: Math.sin(th) };
};

function planeAt(rect, clientX, clientY) {
  const u = (clientX - rect.left) / rect.width;
  const v = (clientY - rect.top) / rect.height;
  const { halfW, halfH } = frameExtents();
  const x = (2 * u - 1) * halfW, y = (1 - 2 * v) * halfH;
  const { cos, sin } = axes(S.state.view);
  const [cx, cy] = S.state.view.center;
  return { re: cx + x * cos - y * sin, im: cy + x * sin + y * cos };
}

/** Plane point under a client (x, y), for tests and the console. */
export function planeFromClient(x, y) {
  const p = planeAt(document.querySelector(".frame").getBoundingClientRect(), x, y);
  return [p.re, p.im];
}

const layersShown = () => (S.state.layers || []).some((l) => l.enabled !== false);

/** CSS transform that moves the last image to match the current view until a new render lands. */
function imageTransform(rectW) {
  const s = S.view.shownView, c = S.state.view;
  if (!s || !c || !S.view.extent || !s.half_width || !c.half_width || layersShown() || S.ui.compare) return "";
  const k = s.half_width / c.half_width;
  const upp = (2 * frameExtents().halfW) / (rectW || S.state.width);
  const { cos, sin } = axes(c);
  const dx = s.center[0] - c.center[0], dy = s.center[1] - c.center[1];
  const tx = (dx * cos + dy * sin) / upp;
  const ty = (-dx * sin + dy * cos) / upp;
  const rot = (c.rotation || 0) - (s.rotation || 0);
  return `translate(${tx}px, ${-ty}px) rotate(${rot}deg) scale(${k})`;
}

// ---------------------------------------------------------------- stage ----

const layerMode = () => S.ui.tab === "layers" || !S.state.view;

export function Stage() {
  useStore();
  const stageRef = useRef(null), frameRef = useRef(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const drag = useRef(null);

  useEffect(() => {
    const fit = () => {
      const el = stageRef.current;
      if (!el) return;
      const asp = currentAspect();
      const sw = el.clientWidth - 64, sh = el.clientHeight - 64; // room for the crop marks
      if (sw < 10 || sh < 10 || !asp) return;
      const ar = asp.w / asp.h;
      setBox(sw / sh > ar ? { w: sh * ar, h: sh } : { w: sw, h: sw / ar });
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(stageRef.current);
    return () => ro.disconnect();
  }, [S.aspect]);

  if (!S.state) return html`<div class="stage" ref=${stageRef}><div class="loading">Loading…</div></div>`;

  const rect = () => frameRef.current.getBoundingClientRect();
  const layers = S.state.layers || [];
  const boxes = S.view.boxes || [];

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    const r = rect();
    const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
    if (layerMode() && layers.length) {
      // Topmost layer under the pointer, else keep the selected one if we are inside its box.
      let hit = null;
      for (let i = layers.length - 1; i >= 0; i--) {
        const b = boxes[i];
        if (layers[i].enabled !== false && b && fx >= b[0] && fx <= b[2] && fy >= b[1] && fy <= b[3]) { hit = layers[i]; break; }
      }
      if (hit) {
        if (S.ui.selected !== hit.id) selectLayer(hit.id);
        e.currentTarget.setPointerCapture(e.pointerId);
        drag.current = { mode: "layer", id: hit.id, x: e.clientX, y: e.clientY,
                         lx: Number(layerValue(hit, "x")), ly: Number(layerValue(hit, "y")), moved: false };
        return;
      }
      if (!S.state.view) { selectLayer(null); return; }
      if (S.ui.selected) selectLayer(null);
    }
    if (!S.state.view) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const { halfW, halfH } = frameExtents();
    drag.current = { mode: e.altKey ? "rotate" : "pan", x: e.clientX, y: e.clientY, center: [...S.state.view.center],
                     rotation: S.state.view.rotation || 0, uppX: (2 * halfW) / r.width, uppY: (2 * halfH) / r.height };
  };

  const onPointerMove = (e) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x, dy = e.clientY - d.y;
    if (d.mode === "layer") {
      const r = rect();
      const layer = layers.find((l) => l.id === d.id);
      if (!layer) return;
      layer.x = clamp(d.lx + dx / r.width, -0.25, 1.25);
      layer.y = clamp(d.ly + dy / r.height, -0.25, 1.25);
      d.moved = true;
    } else if (d.mode === "rotate") {
      S.state.view.rotation = d.rotation + dx * 0.4;
    } else {
      const { cos, sin } = axes(S.state.view);
      const fx = -dx * d.uppX, fy = dy * d.uppY;
      S.state.view.center = [d.center[0] + fx * cos - fy * sin, d.center[1] + fx * sin + fy * cos];
    }
    scheduleRender(120);
    emit();
  };

  const endDrag = () => {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    if (d.mode === "layer") {
      if (d.moved) commit(`layer.${d.id}.move`);
    } else {
      commit(d.mode === "rotate" ? "view.rotation" : "view.center");
    }
  };

  const onWheel = (e) => {
    e.preventDefault();
    const sel = layers.find((l) => l.id === S.ui.selected);
    if (layerMode() && sel) {
      if (e.shiftKey) sel.rotate = (Number(layerValue(sel, "rotate")) || 0) + e.deltaY * 0.1;
      else sel.size = clamp(Number(layerValue(sel, "size")) * Math.pow(1.1, -e.deltaY / 100), 0.02, 1.5);
      commit(`layer.${sel.id}.${e.shiftKey ? "rotate" : "size"}`);
      return;
    }
    if (!S.state.view) return;
    if (e.shiftKey) {
      S.state.view.rotation = (S.state.view.rotation || 0) + e.deltaY * 0.1;
      commit("view.rotation");
      return;
    }
    const p = planeAt(rect(), e.clientX, e.clientY);
    const factor = Math.pow(1.15, e.deltaY / 100);
    const [cx, cy] = S.state.view.center;
    S.state.view.half_width *= factor;
    S.state.view.center = [p.re + (cx - p.re) * factor, p.im + (cy - p.im) * factor];
    commit("view.half_width");
  };

  // Two-finger pinch zoom on touch screens.
  const pinch = useRef(null);
  const onTouchStart = (e) => {
    if (!S.state.view || e.touches.length !== 2) { pinch.current = null; return; }
    const [a, b] = e.touches;
    pinch.current = { d0: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY), hw0: S.state.view.half_width,
                      c0: [...S.state.view.center] };
  };
  const onTouchMove = (e) => {
    const p0 = pinch.current;
    if (!p0 || e.touches.length !== 2) return;
    e.preventDefault();
    const [a, b] = e.touches;
    const factor = p0.d0 / Math.max(Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY), 1);
    const p = planeAt(rect(), (a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2);
    S.state.view.half_width = p0.hw0 * factor;
    S.state.view.center = [p.re + (p0.c0[0] - p.re) * factor, p.im + (p0.c0[1] - p.im) * factor];
    scheduleRender(120);
    emit();
  };
  const onTouchEnd = () => {
    if (pinch.current) { pinch.current = null; commit("view.half_width"); }
  };

  const pos = S.view.boxPos || [];
  const showBoxes = layerMode() && layers.length > 0 && !S.ui.compare;
  const img = S.ui.compare ? S.compareImg || S.view.img : S.view.img;
  const cursor = layerMode() && layers.length ? "default" : S.state.view ? "grab" : "default";

  return html`
    <div class="stage" ref=${stageRef}>
      <div class="frame-wrap" style=${{ width: `${box.w}px`, height: `${box.h}px` }}>
      <div class=${`frame${S.view.transparent ? " transparent" : ""}`} ref=${frameRef}
        style=${{ width: `${box.w}px`, height: `${box.h}px` }}>
        ${img ? html`<img src=${img} alt=${`Preview of ${S.state.recipe || S.state.generator}`}
          style=${{ transform: imageTransform(box.w) }} draggable="false" />` : null}
        <div class="overlay" style=${{ cursor }}
          onPointerDown=${onPointerDown} onPointerMove=${onPointerMove} onPointerUp=${endDrag} onPointerCancel=${endDrag}
          onWheel=${onWheel} onTouchStart=${onTouchStart} onTouchMove=${onTouchMove} onTouchEnd=${onTouchEnd}
          onDblClick=${() => { if (!layerMode()) resetView(); }}>
          ${showBoxes ? layers.map((l, i) => {
            const b = boxes[i];
            if (!b || l.enabled === false) return null;
            const sel = l.id === S.ui.selected;
            const p0 = pos[i];
            const off = p0 && pos.length === layers.length
              ? [Number(layerValue(l, "x")) - p0[0], Number(layerValue(l, "y")) - p0[1]] : [0, 0];
            return html`<div class=${`layer-box${sel ? " sel" : ""}`} key=${l.id}
              style=${{ left: `${(b[0] + off[0]) * 100}%`, top: `${(b[1] + off[1]) * 100}%`,
                        width: `${(b[2] - b[0]) * 100}%`, height: `${(b[3] - b[1]) * 100}%` }}>
              ${sel ? html`<span class="box-label">${layerLabel(l)}</span>` : null}
            </div>`;
          }) : null}
        </div>
        ${S.ui.compare ? html`<div class="compare-badge">Original look</div>` : null}
      </div>
      </div>
      ${S.view.hints?.length && !S.view.busy ? html`<${Hints} hints=${S.view.hints} />` : null}
      <${StageTools} />
      <${StatusPill} />
      ${S.view.error ? html`<${RenderError} err=${S.view.error} />` : null}
    </div>`;
}

const PREVIEW_QUALITY = [
  { px: 480, label: "Fast" },
  { px: 960, label: "Good" },
  { px: 1600, label: "Sharp" },
];

function PreviewQuality() {
  const frame = document.querySelector(".frame");
  // "Full": one render pixel per screen pixel of the preview frame.
  const full = frame ? Math.min(2000, Math.round(Math.max(frame.clientWidth, frame.clientHeight) * (window.devicePixelRatio || 1))) : 2000;
  const opts = [...PREVIEW_QUALITY, { px: full, label: "Full" }];
  const cur = opts.find((o) => o.px === S.previewSize) ? S.previewSize : "custom";
  return html`<select class="tool-select" title="Preview quality: render size of the live preview (the saved wallpaper is always full size)"
    value=${cur} onChange=${(e) => setPreviewSize(Number(e.currentTarget.value))}>
    ${opts.map((o) => html`<option value=${o.px}>${o.label} preview (${o.px}px)</option>`)}
    ${cur === "custom" ? html`<option value="custom" disabled>${S.previewSize}px preview</option>` : null}
  </select>`;
}

function StageTools() {
  const v = S.state.view;
  const zoom = (f) => { v.half_width *= f; commit("view.half_width"); };
  return html`
    <div class="stage-tools">
      <${PreviewQuality} />
      ${v ? html`
        <button type="button" class="tool" title="Zoom in" onClick=${() => zoom(0.5)}><${Icon} name="plus" /></button>
        <button type="button" class="tool" title="Zoom out" onClick=${() => zoom(2)}><${Icon} name="minus" /></button>
        <button type="button" class="tool" title="Reset framing (R)" onClick=${resetView}><${Icon} name="reset-view" /></button>` : null}
      <button type="button" class="tool wide" title="Hold to see the look as it was before your changes (\\)"
        onPointerDown=${startCompare} onPointerUp=${stopCompare} onPointerLeave=${() => S.ui.compare && stopCompare()}>
        Compare
      </button>
    </div>`;
}

/** Suggestions the last render made about its own settings (e.g. more detail on a deep zoom). */
function Hints({ hints }) {
  return html`<div class="hints">${hints.map((h) => html`
    <div class="hint-card" key=${h.key}>
      <span>${h.text}</span>
      ${h.param ? html`<button type="button" class="btn small accent"
        onClick=${() => setPath(`params.${h.param}`, h.value)}>${h.label || "Apply"}</button>` : null}
    </div>`)}</div>`;
}

function StatusPill() {
  const v = S.view;
  const secs = (performance.now() - v.started) / 1000;
  if (v.busy && secs > 0.25) {
    const lbl = v.waiting ? "Waiting for another render" : v.label;
    const pct = v.pct != null && !v.waiting ? ` ${Math.round(v.pct * 100)}%` : "";
    return html`<div class="status busy" style=${{ "--p": v.pct || 0 }}>${lbl}${pct}</div>`;
  }
  const hint = !S.state.view ? "" : layerMode() && S.state.layers.length
    ? "Drag a layer to move it. Scroll over it to resize, or hold Shift to turn it."
    : "Drag to move around and scroll to zoom. Hold Shift while scrolling to turn.";
  return html`<div class="status">${hint ? html`<span class="hint">${hint}</span>` : null}</div>`;
}

function RenderError({ err }) {
  const [open, setOpen] = useState(false);
  return html`
    <div class="render-error">
      <strong>This didn't render.</strong> ${err.message}
      ${err.trace ? html`<button type="button" class="link" onClick=${() => setOpen(!open)}>${open ? "Hide" : "Details"}</button>` : null}
      ${open ? html`<pre>${err.trace}</pre>` : null}
    </div>`;
}
