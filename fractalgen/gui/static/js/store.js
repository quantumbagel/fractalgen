// App state, history, URL persistence, the preview render loop and every edit action.
// Components read `S` directly and re-render on `emit()` (see useStore).

import { useEffect, useState } from "../vendor/preact-htm.js";
import {
  apiJson, apiRender, b64UrlDecode, b64UrlEncode, clamp, deepClone, deepEq, finiteNum, getAtPath, setAtPath, uid,
} from "./util.js";

export const S = {
  catalog: null,
  state: null, // the look being edited (a server "state")
  baseline: null, // the recipe as loaded: Reset / modified marks / Compare
  aspects: [],
  aspect: "screen", // default: a wallpaper for the screen the GUI is open on
  previewSize: 720,
  modified: new Set(),
  ui: {
    tab: "look",
    selected: null, // selected layer id
    overlay: null, // "browse" | "export" | "save" | "variations" | "keys" | null
    menu: false,
    help: true,
    filter: "",
    showMore: false,
    compare: false,
    toasts: [],
    sheet: false, // phone layout: panel open
  },
  view: {
    img: null, extent: null, shownView: null, boxes: null, transparent: false, ms: null,
    busy: false, pct: null, label: "", started: 0, waiting: false, error: null, size: null,
  },
  compareImg: null,
  variations: [],
  exportJob: null,
  history: { undo: [], redo: [] },
};

const subs = new Set();
let emitQueued = false;
export function emit() {
  if (emitQueued) return;
  emitQueued = true;
  queueMicrotask(() => {
    emitQueued = false;
    for (const f of [...subs]) f();
  });
}

export function useStore() {
  const [, bump] = useState(0);
  useEffect(() => {
    const f = () => bump((v) => v + 1);
    subs.add(f);
    return () => subs.delete(f);
  }, []);
  return S;
}

// ---------------------------------------------------------------- toasts ----

export function toast(text, kind = "info", detail = null, ms = 3500) {
  const t = { id: uid(), text, kind, detail };
  S.ui.toasts = [...S.ui.toasts, t];
  emit();
  if (kind !== "error") setTimeout(() => dismissToast(t.id), ms);
  return t.id;
}

export function dismissToast(id) {
  S.ui.toasts = S.ui.toasts.filter((t) => t.id !== id);
  emit();
}

export function showError(e, prefix = "") {
  toast(`${prefix}${e.message || e}`, "error", e.trace || null);
}

// ---------------------------------------------------------------- catalog helpers ----

export const genSpec = (s = S.state) => S.catalog?.generators[s?.generator];
export const effectSpec = (op) => S.catalog?.effects.find((e) => e.name === op);
export const layerTypeSpec = (t) => S.catalog?.layer_types.find((l) => l.name === t);
export const recipeInfo = (name) => S.catalog?.recipes.find((r) => r.name === name);

export function layerKey(layer, name) {
  return layerTypeSpec(layer.type)?.keys.find((k) => k.name === name);
}

export function layerValue(layer, name) {
  if (layer[name] !== undefined) return layer[name];
  return layerKey(layer, name)?.neutral;
}

export function isCutout(layer) {
  return ["fractal", "self"].includes(String(layerValue(layer, "fill")).toLowerCase());
}

export function layerLabel(layer) {
  if (layer.name) return layer.name;
  if (layer.type === "text") {
    const t = String(layer.text || "").replace(/\\n|\n/g, " ").trim();
    return t ? `“${t.slice(0, 22)}”` : "Text";
  }
  if (layer.type === "shape") {
    const s = layer.shape || "circle";
    return s[0].toUpperCase() + s.slice(1);
  }
  if (layer.type === "image") return String(layer.image || "Image").split("/").pop() || "Image";
  return layer.type;
}

// ---------------------------------------------------------------- frame / aspects ----

/** This device's screen in physical pixels, in its current orientation. */
export function screenPixels() {
  const dpr = window.devicePixelRatio || 1;
  return { w: Math.round(window.screen.width * dpr), h: Math.round(window.screen.height * dpr) };
}

export function buildAspects() {
  const scr = screenPixels();
  const list = [
    { id: "screen", label: `This screen (${scr.w}×${scr.h})`, w: scr.w, h: scr.h, size: [scr.w, scr.h],
      group: scr.h > scr.w ? "Phone" : "Desktop" },
    { id: "desktop", label: "Desktop 16:9", w: 16, h: 9, size: [3840, 2160], group: "Desktop" },
  ];
  for (const p of S.catalog.presets || []) {
    list.push({
      id: `preset:${p.name}`, label: `${p.name} (${p.width}×${p.height})`, w: p.width, h: p.height,
      size: [p.width, p.height], format: p.format, group: p.height > p.width ? "Phone" : "Desktop",
    });
  }
  const phone = list.find((a) => a.group === "Phone");
  list.splice(2, 0, { id: "phone", label: "Phone 9:19.5", w: 9, h: 19.5, size: phone ? phone.size : [1290, 2796],
                      group: "Phone" });
  S.aspects = list;
}

export const currentAspect = () => S.aspects.find((a) => a.id === S.aspect) || S.aspects[0];

export function previewDims(asp = currentAspect(), longEdge = S.previewSize) {
  const portrait = asp.h > asp.w;
  return portrait
    ? { w: Math.max(16, Math.round(longEdge * asp.w / asp.h)), h: longEdge }
    : { w: longEdge, h: Math.max(16, Math.round(longEdge * asp.h / asp.w)) };
}

function syncDims() {
  const { w, h } = previewDims();
  S.state.width = w;
  S.state.height = h;
}

export const orient = () => (S.state && S.state.height > S.state.width ? "port" : "land");

// ---------------------------------------------------------------- state shape ----

function withIds(list) {
  return (list || []).map((it) => (it.id ? it : { ...it, id: uid() }));
}

export function normalizeState(st) {
  st.params ||= {};
  st.tone ||= {};
  st.post = withIds(Array.isArray(st.post) ? st.post : []);
  st.layers = withIds(st.layers);
  st.background ||= {};
  delete st.mask;
  return st;
}

function sanitizeView(view, fallback) {
  if (!view || typeof view !== "object") return view;
  const fb = fallback || {};
  const src = Array.isArray(view.center) ? view.center : null;
  const fbc = Array.isArray(fb.center) ? fb.center : [0, 0];
  const re = finiteNum(src?.[0]) ?? finiteNum(fbc[0]) ?? 0;
  const im = finiteNum(src?.[1]) ?? finiteNum(fbc[1]) ?? 0;
  let hw = finiteNum(view.half_width) ?? finiteNum(fb.half_width);
  hw = hw == null || hw <= 0 ? 2 : Math.max(1e-14, hw);
  const rot = finiteNum(view.rotation) ?? finiteNum(fb.rotation) ?? 0;
  return { ...view, center: [re, im], half_width: hw, rotation: rot, fit: view.fit || fb.fit || "fill" };
}

let seq = 0;
export const clientId = uid();

export function stateForApi(s = S.state, extra = {}) {
  const o = deepClone(s);
  o.seq = seq;
  o.client = clientId;
  const tone = {};
  for (const op of S.catalog.tone_ops) {
    const v = o.tone?.[op.name];
    if (v != null && !deepEq(v, op.neutral)) tone[op.name] = v;
  }
  o.tone = tone;
  if (o.view) o.view = sanitizeView(o.view, S.baseline?.view);
  return Object.assign(o, extra);
}

// ---------------------------------------------------------------- modified marks ----

function recomputeModified() {
  const m = new Set();
  const s = S.state, b = S.baseline;
  if (s && b) {
    if (!deepEq(s.palette, b.palette)) m.add("palette");
    for (const k of Object.keys(s.params || {})) {
      if (!deepEq(s.params[k], b.params?.[k])) m.add(`params.${k}`);
    }
    if (s.view && b.view) {
      for (const k of ["center", "half_width", "rotation", "fit"]) {
        if (!deepEq(s.view[k], b.view[k])) m.add(`view.${k}`);
      }
    }
    for (const op of S.catalog.tone_ops) {
      if (!deepEq(s.tone?.[op.name] ?? op.neutral, b.tone?.[op.name] ?? op.neutral)) m.add(`tone.${op.name}`);
    }
    const strip = (l) => (l || []).map(({ id, ...rest }) => rest);
    if (!deepEq(strip(s.layers), strip(b.layers)) || !deepEq(s.background, b.background)) m.add("layers");
    if (!deepEq(strip(s.post), strip(b.post))) m.add("post");
  }
  S.modified = m;
}

export const isModified = (path) => S.modified.has(path);

// ---------------------------------------------------------------- history ----

const HISTORY_KEYS = ["recipe", "generator", "palette", "params", "view", "tone", "post", "layers", "background"];
let committed = null, lastKey = null, lastTime = 0;

function snapshot() {
  const o = { baseline: S.baseline, aspect: S.aspect, selected: S.ui.selected };
  for (const k of HISTORY_KEYS) o[k] = deepClone(S.state[k] ?? null);
  return o;
}

const sameLook = (a, b) => HISTORY_KEYS.every((k) => deepEq(a[k], b[k])) && a.aspect === b.aspect;

export function resetHistory() {
  committed = snapshot();
  lastKey = null;
  S.history = { undo: [], redo: [] };
}

function checkpoint(key = null) {
  const cur = snapshot();
  if (committed && !sameLook(cur, committed)) {
    const now = performance.now();
    const merge = key !== null && key === lastKey && now - lastTime < 1200;
    if (!merge) {
      S.history.undo.push(committed);
      if (S.history.undo.length > 150) S.history.undo.shift();
    }
    S.history.redo = [];
    lastKey = key;
    lastTime = now;
  }
  committed = cur;
}

function restore(snap) {
  for (const k of HISTORY_KEYS) S.state[k] = deepClone(snap[k]);
  S.baseline = snap.baseline;
  if (snap.aspect !== S.aspect) S.aspect = snap.aspect;
  S.ui.selected = S.state.layers?.some((l) => l.id === snap.selected) ? snap.selected : null;
  syncDims();
  committed = snapshot();
  lastKey = null;
  afterChange();
}

export function undo() {
  checkpoint();
  if (!S.history.undo.length) return;
  S.history.redo.push(committed);
  restore(S.history.undo.pop());
}

export function redo() {
  checkpoint();
  if (!S.history.redo.length) return;
  S.history.undo.push(committed);
  restore(S.history.redo.pop());
}

function afterChange() {
  seq++;
  recomputeModified();
  persistHash();
  scheduleRender();
  emit();
}

/** Record an edit already made to S.state (rapid edits with the same key merge into one undo step). */
export function commit(key = null) {
  checkpoint(key);
  afterChange();
}

export function setPath(path, value, key = path) {
  setAtPath(S.state, path, value);
  commit(key);
}

export function resetPath(path) {
  setAtPath(S.state, path, deepClone(getAtPath(S.baseline, path)));
  commit(path);
}

export function resetAll() {
  const { width, height } = S.state;
  S.state = normalizeState(deepClone(S.baseline));
  Object.assign(S.state, { width, height });
  S.ui.selected = null;
  commit("reset");
}

export function resetView() {
  if (S.baseline?.view && S.state.view) {
    S.state.view = deepClone(S.baseline.view);
    commit("view");
  }
}

// ---------------------------------------------------------------- URL hash ----

function persistHash() {
  try {
    const s = S.state;
    const payload = {
      recipe: s.recipe, generator: s.generator, palette: s.palette, params: s.params,
      view: s.view ? sanitizeView(s.view, S.baseline?.view) : s.view,
      tone: s.tone, post: s.post, layers: s.layers, background: s.background,
      aspect: S.aspect, previewSize: S.previewSize,
    };
    history.replaceState(null, "", "#" + b64UrlEncode(payload));
  } catch (_) { /* too long for the URL: keep the old one */ }
}

// ---------------------------------------------------------------- loading ----

let loadToken = 0;

async function fetchRecipeState(name) {
  const { w, h } = previewDims();
  return normalizeState(await apiJson(`/api/state?recipe=${encodeURIComponent(name)}&w=${w}&h=${h}`));
}

export function blankState(genName) {
  const g = S.catalog.generators[genName];
  const params = {};
  let palette = null;
  for (const p of g.params) {
    if (p.name === "palette") palette = p.default;
    else if (p.default !== null) params[p.name] = deepClone(p.default);
  }
  const { w, h } = previewDims();
  return normalizeState({
    recipe: null, generator: genName, palette, params,
    view: g.uses_view ? { center: [0, 0], half_width: 2, rotation: 0, fit: "fill", ...deepClone(g.default_view) } : null,
    tone: {}, post: [], layers: [], background: {}, width: w, height: h,
  });
}

function adopt(st, key) {
  const first = !S.state;
  S.state = st;
  S.baseline = deepClone(st);
  S.ui.selected = null;
  S.compareImg = null;
  S.variations = [];
  syncDims();
  if (first) resetHistory();
  commit(key);
}

export async function loadRecipe(name) {
  const token = ++loadToken;
  try {
    const st = await fetchRecipeState(name);
    if (token !== loadToken) return;
    adopt(st, "recipe");
  } catch (e) {
    showError(e, `Could not load ${name}: `);
  }
}

export function loadGenerator(name) {
  loadToken++;
  adopt(blankState(name), "recipe");
}

export function stepRecipe(delta) {
  const names = S.catalog.recipes.map((r) => r.name).sort((a, b) => a.localeCompare(b));
  let i = names.indexOf(S.state.recipe);
  if (i < 0) i = 0;
  loadRecipe(names[(i + delta + names.length) % names.length]);
}

export function surprise() {
  const pool = S.catalog.recipes.filter((r) => !(r.tags || []).includes("draft") && r.name !== S.state.recipe);
  if (pool.length) loadRecipe(pool[Math.floor(Math.random() * pool.length)].name);
}

/** Switch the frame shape; the recipe's own per-orientation framing applies to whatever you have not edited. */
export async function setAspect(id) {
  const token = ++loadToken;
  S.aspect = id;
  syncDims();
  if (S.state.recipe) {
    try {
      const fresh = await fetchRecipeState(S.state.recipe);
      if (token !== loadToken) return;
      const s = S.state, b = S.baseline;
      if (s.view && fresh.view) {
        for (const k of ["center", "half_width", "rotation", "fit"]) {
          if (deepEq(s.view[k], b.view?.[k])) s.view[k] = deepClone(fresh.view[k]);
        }
      }
      for (const k of Object.keys(fresh.params)) {
        if (deepEq(s.params[k], b.params?.[k])) s.params[k] = deepClone(fresh.params[k]);
      }
      if (deepEq(s.palette, b.palette)) s.palette = deepClone(fresh.palette);
      for (const op of S.catalog.tone_ops) {
        if (deepEq(s.tone[op.name], b.tone?.[op.name])) {
          if (fresh.tone[op.name] === undefined) delete s.tone[op.name];
          else s.tone[op.name] = fresh.tone[op.name];
        }
      }
      const strip = (l) => (l || []).map(({ id, ...rest }) => rest);
      // Same stack as the recipe: each key you haven't touched follows the new orientation.
      const sameShape = (a, c) => a.length === c.length && a.every((l, i) => l.type === c[i].type);
      if (sameShape(s.layers, b.layers || []) && sameShape(fresh.layers, b.layers || [])) {
        s.layers = s.layers.map((l, i) => {
          const old = b.layers[i], next = { ...l };
          for (const k of new Set([...Object.keys(old), ...Object.keys(fresh.layers[i])])) {
            if (k === "id" || !deepEq(l[k], old[k])) continue;
            if (fresh.layers[i][k] === undefined) delete next[k];
            else next[k] = deepClone(fresh.layers[i][k]);
          }
          return next;
        });
      }
      for (const k of new Set([...Object.keys(b.background || {}), ...Object.keys(fresh.background)])) {
        if (!deepEq(s.background[k], b.background?.[k])) continue;
        if (fresh.background[k] === undefined) delete s.background[k];
        else s.background[k] = fresh.background[k];
      }
      if (deepEq(strip(s.post), strip(b.post))) s.post = fresh.post;
      S.baseline = fresh;
    } catch (e) {
      showError(e);
    }
  }
  S.compareImg = null;
  commit("aspect");
}

export async function restoreFromHash(h) {
  S.aspect = S.aspects.some((a) => a.id === h.aspect) ? h.aspect : "screen";
  if (h.previewSize) S.previewSize = clamp(Number(h.previewSize) || 720, 240, 2000);
  let st;
  if (h.recipe) {
    st = await fetchRecipeState(h.recipe);
  } else if (S.catalog.generators[h.generator]) {
    st = blankState(h.generator);
  } else {
    throw new Error(`unknown generator ${h.generator}`);
  }
  const base = deepClone(st);
  if ("palette" in h) st.palette = h.palette;
  st.params = { ...st.params, ...(h.params || {}) };
  if (h.view && st.view) st.view = sanitizeView({ ...st.view, ...h.view }, st.view);
  st.tone = h.tone || {};
  if (Array.isArray(h.post)) st.post = h.post;
  if (Array.isArray(h.layers)) st.layers = h.layers;
  if (h.background) st.background = h.background;
  S.state = normalizeState(st);
  S.baseline = base;
  syncDims();
  resetHistory();
  afterChange();
}

export async function boot() {
  try { S.ui.help = localStorage.getItem("fractalgen.help") !== "0"; } catch (_) { /* private mode */ }
  S.catalog = await apiJson("/api/catalog");
  buildAspects();
  if (location.hash.length > 1) {
    try {
      await restoreFromHash(b64UrlDecode(location.hash.slice(1)));
      return;
    } catch (e) {
      showError(new Error(`Could not restore the saved link (${e.message}); loading the default look.`));
    }
  }
  await loadRecipe("julia_dragon");
  resetHistory();
  emit();
}

export async function reloadCatalog() {
  S.catalog = await apiJson("/api/catalog");
  buildAspects();
  emit();
}

// ---------------------------------------------------------------- render loop ----

let renderToken = 0, debounce = null, objectUrl = null;

export function scheduleRender(delay = 60) {
  clearTimeout(debounce);
  debounce = setTimeout(renderPass, delay);
}

function parseExtent(hdr) {
  if (!hdr) return null;
  const [hw, hh] = hdr.split(",").map(parseFloat);
  return Number.isFinite(hw) && Number.isFinite(hh) ? { halfW: hw, halfH: hh } : null;
}

function applyImage(res, rendered) {
  const url = URL.createObjectURL(res.blob);
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = url;
  const v = S.view;
  v.img = url;
  v.extent = parseExtent(res.extent);
  v.shownView = rendered.view ? deepClone(rendered.view) : null;
  v.boxes = res.boxes;
  v.boxPos = (rendered.layers || []).map((l) => [l.x ?? 0.5, l.y ?? 0.5]);
  v.ms = res.ms;
  v.size = [rendered.width, rendered.height];
  v.error = null;
  const bg = String(rendered.background?.fill ?? "").toLowerCase();
  v.transparent = (rendered.layers || []).some((l) => l.enabled !== false && isCutout(l)) &&
    ["transparent", "none", "alpha"].includes(bg);
  emit();
}

async function renderPass() {
  if (!S.state) return;
  const token = ++renderToken;
  syncDims();
  const full = stateForApi();
  const half = { ...full, width: Math.max(16, Math.round(full.width / 2)), height: Math.max(16, Math.round(full.height / 2)),
                 extent_size: [full.width, full.height] };
  const v = S.view;
  Object.assign(v, { busy: true, pct: null, label: "Quick preview", started: performance.now(), waiting: false });
  emit();
  let polling = false;
  const tick = setInterval(() => {
    if (token !== renderToken) { clearInterval(tick); return; }
    emit();
    if (!polling && performance.now() - v.started > 300) {
      polling = true;
      apiJson(`/api/progress?client=${clientId}`).then((p) => {
        if (token !== renderToken) return;
        v.waiting = Boolean(p.waiting);
        if (p.progress != null) v.pct = p.progress;
      }).catch(() => {}).finally(() => { polling = false; });
    }
  }, 150);
  try {
    const draft = await apiRender(half);
    if (token !== renderToken || draft.stale) return;
    applyImage(draft, half);
    Object.assign(v, { label: "Rendering", pct: null });
    const fin = await apiRender(full);
    if (token !== renderToken || fin.stale) return;
    applyImage(fin, full);
  } catch (e) {
    if (token === renderToken) {
      v.error = e;
      emit();
    }
  } finally {
    clearInterval(tick);
    if (token === renderToken) {
      v.busy = false;
      emit();
    }
  }
}

export function setPreviewSize(px) {
  S.previewSize = clamp(Math.round(px), 240, 2000);
  syncDims();
  persistHash();
  scheduleRender(150);
  emit();
}

// ---------------------------------------------------------------- compare ----

let compareKey = null;

export async function startCompare() {
  S.ui.compare = true;
  emit();
  const st = stateForApi(S.baseline, { width: S.state.width, height: S.state.height, client: `${clientId}-cmp`, seq: 0 });
  st.view = S.baseline.view ? sanitizeView(S.baseline.view) : null;
  const key = JSON.stringify(st);
  if (compareKey === key && S.compareImg) return;
  try {
    const res = await apiRender(st);
    if (res.stale) return;
    if (S.compareImg) URL.revokeObjectURL(S.compareImg);
    S.compareImg = URL.createObjectURL(res.blob);
    compareKey = key;
    emit();
  } catch (e) {
    showError(e);
  }
}

export function stopCompare() {
  S.ui.compare = false;
  emit();
}

// ---------------------------------------------------------------- layers ----

export function selectLayer(id) {
  S.ui.selected = id;
  if (id) S.ui.tab = "layers";
  emit();
}

export function addLayer(type, init = {}) {
  const lt = layerTypeSpec(type);
  const layer = { id: uid(), type, ...deepClone(lt?.start || {}), ...init };
  S.state.layers.push(layer);
  S.ui.selected = layer.id;
  S.ui.tab = "layers";
  commit(`layer-add-${layer.id}`);
  return layer;
}

const findLayer = (id) => S.state.layers.find((l) => l.id === id);

export function setLayer(id, key, value, commitKey = `layer.${id}.${key}`) {
  const layer = findLayer(id);
  if (!layer) return;
  const spec = layerKey(layer, key);
  if (value === undefined || value === null || (spec && deepEq(value, spec.neutral) && key !== "fill")) delete layer[key];
  else layer[key] = value;
  commit(commitKey);
}

export function patchLayer(id, patch, commitKey = `layer.${id}`) {
  const layer = findLayer(id);
  if (!layer) return;
  for (const [k, v] of Object.entries(patch)) {
    if (v === undefined || v === null) delete layer[k];
    else layer[k] = v;
  }
  commit(commitKey);
}

export function removeLayer(id) {
  S.state.layers = S.state.layers.filter((l) => l.id !== id);
  if (S.ui.selected === id) S.ui.selected = null;
  commit(`layer-rm-${id}`);
}

export function duplicateLayer(id) {
  const i = S.state.layers.findIndex((l) => l.id === id);
  if (i < 0) return;
  const copy = { ...deepClone(S.state.layers[i]), id: uid() };
  if (typeof copy.x === "number" || copy.x === undefined) copy.x = clamp((copy.x ?? 0.5) + 0.04, 0, 1);
  if (typeof copy.y === "number" || copy.y === undefined) copy.y = clamp((copy.y ?? 0.5) + 0.04, 0, 1);
  S.state.layers.splice(i + 1, 0, copy);
  S.ui.selected = copy.id;
  commit(`layer-dup-${copy.id}`);
}

/** Move a layer up (+1, toward the front) or down (-1) the stack. */
export function moveLayer(id, delta) {
  const L = S.state.layers;
  const i = L.findIndex((l) => l.id === id);
  const j = i + delta;
  if (i < 0 || j < 0 || j >= L.length) return;
  [L[i], L[j]] = [L[j], L[i]];
  commit(`layer-move-${id}`);
}

export function reorderLayer(id, toIndex) {
  const L = S.state.layers;
  const i = L.findIndex((l) => l.id === id);
  if (i < 0) return;
  const [item] = L.splice(i, 1);
  L.splice(clamp(toIndex, 0, L.length), 0, item);
  commit(`layer-move-${id}`);
}

export function setBackground(key, value) {
  const spec = S.catalog.background_keys.find((k) => k.name === key);
  if (value === null || value === undefined || (spec && deepEq(value, spec.neutral) && key !== "fill")) {
    delete S.state.background[key];
  } else {
    S.state.background[key] = value;
  }
  commit(`background.${key}`);
}

export const STYLE_PRESETS = {
  Clean: { outline: null, glow: null, shadow: null, bevel: null, feather: null },
  Glow: { glow: 0.8, glow_radius: 60, outline: null, shadow: null, bevel: null },
  Neon: { glow: 1.2, glow_radius: 40, outline: 3, outline_color: "ffffff", bevel: null, shadow: null },
  Stamp: { outline: 8, outline_color: "ffffff", shadow: 0.5, glow: null, bevel: null },
  Emboss: { bevel: 0.8, shadow: 0.6, glow: null, outline: null },
  Soft: { feather: 12, glow: 0.4, glow_radius: 80, outline: null, bevel: null },
};

export function applyStylePreset(id, name) {
  patchLayer(id, STYLE_PRESETS[name], `layer.${id}.style`);
}

// ---------------------------------------------------------------- effects ----

export function addEffect(op, init = {}) {
  const spec = effectSpec(op);
  const item = { id: uid(), op, ...deepClone(spec?.start || {}), ...init };
  S.state.post.push(item);
  commit(`fx-add-${item.id}`);
  return item;
}

const findEffect = (id) => S.state.post.find((e) => e.id === id);

export function setEffect(id, key, value) {
  const it = findEffect(id);
  if (!it) return;
  if (value === undefined || value === null) delete it[key];
  else it[key] = value;
  if (key === "mix" && value >= 1) delete it.mix;
  if (key === "enabled" && value) delete it.enabled;
  commit(`fx.${id}.${key}`);
}

export function removeEffect(id) {
  S.state.post = S.state.post.filter((e) => e.id !== id);
  commit(`fx-rm-${id}`);
}

export function duplicateEffect(id) {
  const i = S.state.post.findIndex((e) => e.id === id);
  if (i < 0) return;
  S.state.post.splice(i + 1, 0, { ...deepClone(S.state.post[i]), id: uid() });
  commit(`fx-dup-${id}`);
}

export function moveEffect(id, delta) {
  const P = S.state.post;
  const i = P.findIndex((e) => e.id === id);
  const j = i + delta;
  if (i < 0 || j < 0 || j >= P.length) return;
  [P[i], P[j]] = [P[j], P[i]];
  commit(`fx-move-${id}`);
}

export function reorderEffect(id, toIndex) {
  const P = S.state.post;
  const i = P.findIndex((e) => e.id === id);
  if (i < 0) return;
  const [item] = P.splice(i, 1);
  P.splice(clamp(toIndex, 0, P.length), 0, item);
  commit(`fx-move-${id}`);
}

/** The value of a single-value effect as a quick adjustment (first effect of that kind). */
export function quickValue(op) {
  const it = S.state.post.find((e) => e.op === op && e.enabled !== false);
  const spec = effectSpec(op);
  return it ? it.value : spec?.params[0].neutral;
}

export function setQuick(op, value) {
  const spec = effectSpec(op);
  const neutral = spec?.params[0].neutral;
  const i = S.state.post.findIndex((e) => e.op === op);
  if (deepEq(value, neutral)) {
    if (i >= 0 && Object.keys(S.state.post[i]).every((k) => ["id", "op", "value"].includes(k))) S.state.post.splice(i, 1);
    else if (i >= 0) S.state.post[i].value = value;
  } else if (i >= 0) {
    S.state.post[i].value = value;
  } else {
    S.state.post.push({ id: uid(), op, value });
  }
  commit(`quick.${op}`);
}

export const EFFECT_PRESETS = {
  Film: [{ op: "contrast", value: 1.1 }, { op: "grain", amount: 0.06, size: 2.5 }, { op: "vignette", value: 0.35 }],
  Dreamy: [{ op: "bloom", amount: 0.8, threshold: 0.45, radius: 90 }, { op: "saturation", value: 1.15 }],
  Crisp: [{ op: "sharpen", amount: 0.7, radius: 3 }, { op: "contrast", value: 1.15 }],
  Retro: [{ op: "posterize", value: 6 }, { op: "grain", amount: 0.05, size: 3 }, { op: "vignette", value: 0.3 }],
  Lens: [{ op: "chromatic_aberration", amount: 10 }, { op: "vignette", value: 0.45 }, { op: "bloom", amount: 0.4 }],
  Sunset: [{ op: "gradient", palette: "ember", angle: 90, blend: "soft_light" }, { op: "vignette", value: 0.3 }],
};

export function applyEffectPreset(name) {
  S.state.post = EFFECT_PRESETS[name].map((e) => ({ id: uid(), ...deepClone(e) }));
  commit("fx-preset");
}

export function clearEffects() {
  S.state.post = [];
  commit("fx-clear");
}

// ---------------------------------------------------------------- palette ----

export function setPalette(v) {
  S.state.palette = v;
  commit("palette");
}

// ---------------------------------------------------------------- variations ----

let varToken = 0;

function rand(a, b) { return a + Math.random() * (b - a); }

function mutate(src, strength) {
  const st = deepClone(src);
  const g = genSpec(st);
  for (const p of g.params) {
    const cur = st.params[p.name];
    if (p.advanced && Math.random() < 0.7) continue;
    if (["int", "float"].includes(p.kind) && p.range && typeof cur === "number") {
      const [lo, hi] = p.range;
      if (p.name === "seed" || p.name.endsWith("_seed")) { st.params[p.name] = Math.floor(rand(0, 9999)); continue; }
      if (/max_iter|samples|steps|n_points|particles|quality|batch/.test(p.name)) continue; // cost, not look
      const v = clamp(cur + rand(-1, 1) * strength * (hi - lo), lo, hi);
      st.params[p.name] = p.kind === "int" ? Math.round(v) : v;
    } else if (p.kind === "complex" && Array.isArray(cur)) {
      const m = Math.max(0.02, Math.hypot(cur[0], cur[1]) * 0.08) * strength * 4;
      st.params[p.name] = [cur[0] + rand(-m, m), cur[1] + rand(-m, m)];
    }
  }
  if (st.view) {
    st.view.half_width *= Math.exp(rand(-0.5, 0.5) * strength * 2);
    st.view.rotation = (st.view.rotation || 0) + rand(-25, 25) * strength * 2;
  }
  if (Math.random() < 0.45 && st.palette !== null && st.palette !== undefined) {
    const pals = S.catalog.palettes.map((p) => p.name);
    st.palette = pals[Math.floor(Math.random() * pals.length)] + (Math.random() < 0.3 ? "_r" : "");
  }
  return st;
}

export async function makeVariations(n = 8, strength = 0.12) {
  const token = ++varToken;
  const asp = currentAspect();
  const { w, h } = previewDims(asp, 360);
  S.variations = Array.from({ length: n }, () => ({ state: mutate(S.state, strength), url: null, error: null }));
  S.ui.overlay = "variations";
  emit();
  for (let i = 0; i < n; i++) {
    if (token !== varToken || S.ui.overlay !== "variations") return;
    const v = S.variations[i];
    try {
      const res = await apiRender(stateForApi(v.state, { width: w, height: h, client: `${clientId}-var${i}`, seq: 0 }));
      if (token !== varToken) return;
      if (!res.stale) v.url = URL.createObjectURL(res.blob);
    } catch (e) {
      v.error = e.message;
    }
    emit();
  }
}

export function pickVariation(i) {
  const v = S.variations[i];
  if (!v) return;
  for (const k of ["palette", "params", "view"]) S.state[k] = deepClone(v.state[k]);
  S.ui.overlay = null;
  commit("variation");
}

export function closeOverlay() {
  if (S.ui.overlay === "variations") {
    varToken++;
    for (const v of S.variations) if (v.url) URL.revokeObjectURL(v.url);
    S.variations = [];
  }
  S.ui.overlay = null;
  emit();
}

// ---------------------------------------------------------------- export / save ----

let pollTimer = null;

export async function startExport({ width, height, format, name }) {
  clearTimeout(pollTimer);
  S.exportJob = { status: "starting", progress: 0, width, height, format };
  emit();
  try {
    const { job } = await apiJson("/api/export", { state: stateForApi(), width, height, format, name: name || undefined });
    S.exportJob = { ...S.exportJob, id: job, status: "running" };
    poll();
  } catch (e) {
    S.exportJob = { ...S.exportJob, status: "error", error: e.message };
    emit();
  }
}

function poll() {
  const job = S.exportJob;
  if (!job?.id) return;
  apiJson(`/api/export/${job.id}`).then((st) => {
    if (S.exportJob?.id !== job.id) return;
    S.exportJob = { ...job, ...st };
    emit();
    if (st.status === "running") pollTimer = setTimeout(poll, 400);
    else if (st.status === "done" && S.ui.overlay !== "export") toast("Wallpaper saved", "ok");
  }).catch(() => { pollTimer = setTimeout(poll, 900); });
}

export async function revealExport() {
  try { await apiJson("/api/reveal", { path: S.exportJob.path }); } catch (e) { showError(e); }
}

export async function setAsWallpaper() {
  try {
    await apiJson("/api/wallpaper", { path: S.exportJob.path });
    toast("Desktop wallpaper set", "ok");
  } catch (e) {
    showError(e, "Could not set the wallpaper: ");
  }
}

export async function saveRecipe(name, overwrite) {
  const info = await apiJson("/api/save", { name, state: stateForApi(), overwrite });
  await reloadCatalog();
  const st = await fetchRecipeState(name);
  S.state = st;
  S.baseline = deepClone(st);
  syncDims();
  resetHistory();
  afterChange();
  return info;
}

export async function copyCli() {
  try {
    const { command } = await apiJson("/api/cli", { state: stateForApi(), width: S.state.width, height: S.state.height });
    await navigator.clipboard.writeText(command);
    toast("Command copied to the clipboard", "ok");
  } catch (e) {
    showError(e);
  }
}

export function setHelp(on) {
  S.ui.help = on;
  try { localStorage.setItem("fractalgen.help", on ? "1" : "0"); } catch (_) { /* private mode */ }
  emit();
}
