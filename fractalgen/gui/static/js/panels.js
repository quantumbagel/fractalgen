// The side panel's tabs: Look, Frame, Color, Layers, Effects.

import { html, useRef, useState } from "../vendor/preact-htm.js";
import { Icon } from "./icons.js";
import {
  Control, Field, FillPicker, PaletteGrid, Section, Segmented, Select, Slider, Toggle,
} from "./controls.js";
import {
  EFFECT_PRESETS, S, STYLE_PRESETS, addEffect, addLayer, applyEffectPreset, applyStylePreset, clearEffects, commit,
  duplicateEffect, duplicateLayer, effectSpec, genSpec, isCutout, isModified, layerKey, layerLabel, layerTypeSpec,
  layerValue, loadGenerator, makeVariations, moveEffect, moveLayer, quickValue, recipeInfo, removeEffect, removeLayer,
  reorderEffect, reorderLayer, resetPath, resetView, selectLayer, setBackground, setEffect, setLayer, setPalette,
  setPath, setQuick, showError, surprise,
} from "./store.js";
import { Thumb } from "./overlays.js";
import { apiUpload, deepEq, fmtNum } from "./util.js";

// ---------------------------------------------------------------- Look ----

export function LookPanel() {
  const st = S.state;
  const g = genSpec();
  const info = st.recipe ? recipeInfo(st.recipe) : null;
  const coeffInfo = S.catalog.coefficients[st.generator];
  const skip = new Set(["palette", ...(coeffInfo ? ["coeffs"] : [])]);
  const params = (g?.params || []).filter((p) => !skip.has(p.name));
  const basic = params.filter((p) => !p.advanced);
  const q = S.ui.filter.trim().toLowerCase();
  const advanced = params.filter((p) => p.advanced && (!q || `${p.name} ${p.label} ${p.doc}`.toLowerCase().includes(q)));
  const nAdv = params.filter((p) => p.advanced).length + (coeffInfo ? 1 : 0);

  const row = (p) => html`<${Control} key=${p.name} spec=${p} value=${st.params[p.name]}
    modified=${isModified(`params.${p.name}`)} onReset=${() => resetPath(`params.${p.name}`)}
    onChange=${(v) => {
      if (coeffInfo && p.name === coeffInfo.via) st.params.coeffs = null; // a different map has its own coefficients
      setPath(`params.${p.name}`, v);
    }} />`;

  return html`
    <div class="panel">
      <div class="look-card">
        <div class="look-thumb">${st.recipe ? html`<${Thumb} recipe=${st.recipe} orient="land" key=${st.recipe} />` : null}</div>
        <div class="look-text">
          <div class="look-name">${(st.recipe || `New ${st.generator}`).replace(/_/g, " ")}</div>
          <div class="look-desc">${info?.description || g?.description || ""}</div>
          <div class="look-meta">Made with ${st.generator.replace(/_/g, " ")}</div>
        </div>
      </div>
      <div class="btn-row">
        <button type="button" class="btn" onClick=${() => { S.ui.overlay = "browse"; commit(null); }}>Browse looks</button>
        <button type="button" class="btn" title="Random look (.)" onClick=${surprise}>Surprise me</button>
        <button type="button" class="btn accent" title="Eight random variations of this look" onClick=${() => makeVariations()}>
          Variations
        </button>
      </div>
      <${Section} title="Main settings" count=${basic.length || null}>
        ${basic.length ? basic.map(row) : html`<div class="muted">Everything for this look is under More settings.</div>`}
      </${Section}>
      ${nAdv ? html`<${Section} title="More settings" count=${nAdv} open=${false}>
        <input type="search" class="filter" placeholder="Filter settings…" value=${S.ui.filter}
          onInput=${(e) => { S.ui.filter = e.currentTarget.value; commit(null); }} />
        ${advanced.map(row)}
        ${coeffInfo ? html`<${Coefficients} info=${coeffInfo} />` : null}
      </${Section}>` : null}
      <${Section} title="Start from scratch" open=${false}>
        <p class="muted">Every look starts from a generator. Pick one to start fresh from its defaults.</p>
        <${GeneratorPicker} />
      </${Section}>
    </div>`;
}

function Coefficients({ info }) {
  const st = S.state;
  const defaults = info.defaults[st.params[info.via]] || {};
  const merged = { ...defaults, ...(st.params.coeffs || {}) };
  return html`<div class="subgroup"><div class="subgroup-title">Coefficients (${st.params[info.via]})</div>
    ${Object.keys(defaults).sort().map((k) => {
      const mag = Math.max(1, 3 * Math.abs(defaults[k]));
      return html`<${Field} key=${k} label=${k} modified=${merged[k] !== defaults[k]}
        onReset=${() => { const c = { ...(st.params.coeffs || {}) }; delete c[k]; setPath("params.coeffs", c, "coeffs"); }}>
        <${Slider} value=${merged[k]} min=${-mag} max=${mag}
          onInput=${(v) => setPath("params.coeffs", { ...(st.params.coeffs || {}), [k]: v }, `coeffs.${k}`)} /></${Field}>`;
    })}</div>`;
}

function GeneratorPicker() {
  const byCat = {};
  for (const [name, g] of Object.entries(S.catalog.generators)) (byCat[g.category] ||= []).push(name);
  return html`<select value="" onChange=${(e) => { if (e.currentTarget.value) loadGenerator(e.currentTarget.value); }}>
    <option value="">Choose a generator…</option>
    ${Object.keys(byCat).sort().map((cat) => html`<optgroup label=${cat}>
      ${byCat[cat].sort().map((n) => html`<option value=${n}>${n}</option>`)}</optgroup>`)}
  </select>`;
}

// ---------------------------------------------------------------- Frame ----

const FIT = [
  { value: "fill", label: "Fill width", title: "Keeps the same width on every screen. Taller screens show more above and below." },
  { value: "cover", label: "Cover", title: "Keeps the same height as a 16:9 screen. Phones crop off the sides." },
  { value: "contain", label: "Fit all", title: "Always shows everything a 16:9 screen would, with extra space around it on other shapes." },
];

export function FramePanel() {
  const v = S.state.view;
  if (!v) {
    return html`<div class="panel"><div class="empty">
      <p>This look has no camera, so it always fills the screen the same way.</p>
      <p class="muted">You can still add text and shapes in Layers, or finish it in Effects.</p></div></div>`;
  }
  return html`
    <div class="panel">
      <p class="lead">Drag the picture to move around and scroll to zoom. Hold Shift while scrolling to turn it.</p>
      <${Field} label="Zoom" doc="Smaller numbers zoom in further." modified=${isModified("view.half_width")}
        onReset=${() => resetPath("view.half_width")}
        extra=${html`<span class="grow"></span>
          <button type="button" class="icon-btn" title="Zoom in 2×" onClick=${() => setPath("view.half_width", v.half_width / 2)}><${Icon} name="plus" /></button>
          <button type="button" class="icon-btn" title="Zoom out 2×" onClick=${() => setPath("view.half_width", v.half_width * 2)}><${Icon} name="minus" /></button>`}>
        <${Slider} value=${v.half_width} min=${1e-13} max=${10} log=${true}
          onInput=${(x) => setPath("view.half_width", Math.max(1e-14, x))} />
      </${Field}>
      <${Field} label="Rotation" modified=${isModified("view.rotation")} onReset=${() => resetPath("view.rotation")}
        extra=${html`<span class="grow"></span><button type="button" class="icon-btn" title="Level (0°)"
          onClick=${() => setPath("view.rotation", 0)}>0°</button>`}>
        <${Slider} value=${v.rotation || 0} min=${-180} max=${180} step=${0.5} unit="°"
          onInput=${(x) => setPath("view.rotation", x)} />
      </${Field}>
      <${Field} label="On other screen shapes" doc=${FIT.find((f) => f.value === (v.fit || "fill"))?.title}
        modified=${isModified("view.fit")} onReset=${() => resetPath("view.fit")}>
        <${Segmented} value=${v.fit || "fill"} options=${FIT} onChange=${(x) => setPath("view.fit", x)} />
      </${Field}>
      <${Section} title="Exact position" open=${false}>
        <${Field} label="Center (real)" modified=${isModified("view.center")} onReset=${() => resetPath("view.center")}>
          <input type="number" step="any" value=${v.center[0]}
            onChange=${(e) => setPath("view.center", [parseFloat(e.currentTarget.value) || 0, v.center[1]])} />
        </${Field}>
        <${Field} label="Center (imaginary)">
          <input type="number" step="any" value=${v.center[1]}
            onChange=${(e) => setPath("view.center", [v.center[0], parseFloat(e.currentTarget.value) || 0])} />
        </${Field}>
        <div class="muted">Half width ${fmtNum(v.half_width, 6)}</div>
      </${Section}>
      <div class="btn-row"><button type="button" class="btn" onClick=${resetView}>Reset framing</button></div>
    </div>`;
}

// ---------------------------------------------------------------- Color ----

const QUICK = [
  { op: "exposure", label: "Brightness", min: -2, max: 2, step: 0.01, unit: " stops" },
  { op: "contrast", label: "Contrast", min: 0.3, max: 2.5, step: 0.01 },
  { op: "saturation", label: "Saturation", min: 0, max: 2.5, step: 0.01 },
  { op: "hue_shift", label: "Hue", min: -180, max: 180, step: 1, unit: "°" },
  { op: "vignette", label: "Vignette", min: 0, max: 1, step: 0.01 },
];

export function ColorPanel() {
  const st = S.state;
  const g = genSpec();
  const hasPalette = g?.params.some((p) => p.name === "palette");
  return html`
    <div class="panel">
      ${hasPalette ? html`<${Section} title="Colors">
        <${PaletteGrid} value=${st.palette} onChange=${setPalette} />
        ${isModified("palette") ? html`<button type="button" class="link" onClick=${() => resetPath("palette")}>
          Back to the look's colors</button>` : null}
      </${Section}>` : html`<p class="muted">This look picks its own colors. To change them, add the Recolor effect.</p>`}
      <${Section} title="Adjust">
        ${QUICK.filter((q) => effectSpec(q.op)).map((q) => {
          const val = quickValue(q.op);
          const neutral = effectSpec(q.op).params[0].neutral;
          return html`<${Field} key=${q.op} label=${q.label} doc=${effectSpec(q.op).doc} modified=${!deepEq(val, neutral)}
            onReset=${() => setQuick(q.op, neutral)}>
            <${Slider} value=${val} min=${q.min} max=${q.max} step=${q.step} onInput=${(v) => setQuick(q.op, v)} />
          </${Field}>`;
        })}
        <div class="muted small">These are added as effects, so you'll also see them in the Effects tab.</div>
      </${Section}>
      <${Section} title="Exposure" open=${Object.keys(st.tone || {}).length > 0}>
        <p class="muted small">Adjusts how the look spreads its brightness across its colors.</p>
        ${S.catalog.tone_ops.map((op) => html`<${Control} key=${op.name} spec=${op}
          value=${st.tone?.[op.name] ?? op.neutral} modified=${isModified(`tone.${op.name}`)}
          onReset=${() => resetPath(`tone.${op.name}`)}
          onChange=${(v) => { if (deepEq(v, op.neutral)) delete st.tone[op.name]; else st.tone[op.name] = v; commit(`tone.${op.name}`); }} />`)}
      </${Section}>
    </div>`;
}

// ---------------------------------------------------------------- Layers ----

const KIND = { text: "Text", shape: "Shape", image: "Image" };

export function LayersPanel() {
  const layers = S.state.layers;
  const sel = layers.find((l) => l.id === S.ui.selected);
  const anyCut = layers.some((l) => l.enabled !== false && isCutout(l));
  const fileRef = useRef(null);
  const addImage = async (file) => {
    if (!file) return;
    try {
      const { path } = await apiUpload(file);
      addLayer("image", { image: path });
    } catch (e) {
      showError(e, "Upload failed: ");
    }
  };
  return html`
    <div class="panel">
      <div class="add-row">
        <button type="button" class="btn add" onClick=${() => addLayer("text")}>Add text</button>
        <button type="button" class="btn add" onClick=${() => addLayer("shape")}>Add shape</button>
        <button type="button" class="btn add" onClick=${() => fileRef.current.click()}>Add image</button>
        <input ref=${fileRef} type="file" accept="image/*" hidden onChange=${(e) => addImage(e.currentTarget.files?.[0])} />
      </div>
      ${layers.length ? html`<${LayerList} layers=${layers} />` : html`
        <div class="empty">
          <p>Put words, shapes or a logo on top of the look, or cut the look out of them.</p>
          <p class="muted small">A layer can show the fractal through it, or be filled with a color or a gradient.</p>
        </div>`}
      ${sel ? html`<${LayerEditor} layer=${sel} key=${sel.id} />` : layers.length
        ? html`<p class="muted small">Pick a layer here, or click it on the picture, to edit it.</p>` : null}
      ${anyCut ? html`<${BackgroundEditor} />` : null}
    </div>`;
}

function LayerList({ layers }) {
  const [dragId, setDragId] = useState(null);
  const shown = [...layers].map((l, i) => ({ l, i })).reverse(); // top of the stack first
  return html`
    <ul class="item-list" role="listbox" aria-label="Layers, top first">
      ${shown.map(({ l, i }) => html`
        <li key=${l.id} class=${`item${l.id === S.ui.selected ? " sel" : ""}${l.enabled === false ? " off" : ""}${dragId === l.id ? " dragging" : ""}`}
          draggable="true" role="option" aria-selected=${l.id === S.ui.selected}
          onDragStart=${(e) => { setDragId(l.id); e.dataTransfer.effectAllowed = "move"; }}
          onDragEnd=${() => setDragId(null)}
          onDragOver=${(e) => e.preventDefault()}
          onDrop=${(e) => { e.preventDefault(); if (dragId && dragId !== l.id) reorderLayer(dragId, i); setDragId(null); }}
          onClick=${() => selectLayer(l.id === S.ui.selected ? null : l.id)}>
          <span class="grip" title="Drag to reorder"></span>
          <input type="checkbox" class="eye" title=${l.enabled === false ? "Show this layer" : "Hide this layer"}
            checked=${l.enabled !== false} onClick=${(e) => e.stopPropagation()}
            onChange=${() => setLayer(l.id, "enabled", l.enabled === false ? null : false, `layer.${l.id}.eye`)} />
          <span class="item-name">${layerLabel(l)}</span>
          <span class="kind">${KIND[l.type] || l.type}${isCutout(l) ? ", cut-out" : ""}</span>
          <span class="item-actions" onClick=${(e) => e.stopPropagation()}>
            <button type="button" class="icon-btn" title="Bring forward" disabled=${i === layers.length - 1} onClick=${() => moveLayer(l.id, 1)}><${Icon} name="arrow-up" /></button>
            <button type="button" class="icon-btn" title="Send backward" disabled=${i === 0} onClick=${() => moveLayer(l.id, -1)}><${Icon} name="arrow-down" /></button>
            <button type="button" class="icon-btn" title="Duplicate" onClick=${() => duplicateLayer(l.id)}><${Icon} name="copy" /></button>
            <button type="button" class="icon-btn danger" title="Delete" onClick=${() => removeLayer(l.id)}><${Icon} name="x" /></button>
          </span>
        </li>`)}
    </ul>`;
}

const CONTENT_KEYS = ["text", "font", "shape", "image", "image_channel"];
const PLACE_KEYS = ["size", "x", "y", "rotate"];
const STYLE_KEYS = ["opacity", "blend", "outline", "outline_color", "glow", "glow_radius", "shadow", "bevel", "feather", "invert"];
const ALIGN = [[0.12, 0.12], [0.5, 0.12], [0.88, 0.12], [0.12, 0.5], [0.5, 0.5], [0.88, 0.5], [0.12, 0.88], [0.5, 0.88], [0.88, 0.88]];

function LayerEditor({ layer }) {
  const lt = layerTypeSpec(layer.type);
  const ctl = (name, label = null) => {
    const spec = layerKey(layer, name);
    if (!spec) return null;
    if (name === "shape") return html`<${ShapePicker} layer=${layer} spec=${spec} />`;
    return html`<${Control} key=${name} spec=${spec} label=${label} value=${layerValue(layer, name)}
      modified=${layer[name] !== undefined && name !== "text" && name !== "image"} onReset=${() => setLayer(layer.id, name, null)}
      onChange=${(v) => setLayer(layer.id, name, v)} />`;
  };
  const extraKeys = lt.keys.map((k) => k.name).filter((n) => !CONTENT_KEYS.includes(n) && !lt.common.includes(n));
  const x = Number(layerValue(layer, "x")), y = Number(layerValue(layer, "y"));
  return html`
    <div class="editor">
      <div class="editor-title">${layerLabel(layer)}</div>
      ${CONTENT_KEYS.map((n) => ctl(n))}
      ${extraKeys.map((n) => ctl(n))}
      <${Field} label="Fill" doc=${layerKey(layer, "fill")?.doc}>
        <${FillPicker} value=${layerValue(layer, "fill")} mode=${layer.type === "image" ? "image-layer" : "layer"}
          onChange=${(v) => setLayer(layer.id, "fill", v)} />
      </${Field}>
      <${Section} title="Position">
        <div class="align-row">
          <div class="align-grid" title="Snap to a spot">
            ${ALIGN.map(([ax, ay]) => html`<button type="button" class=${Math.abs(ax - x) < 0.01 && Math.abs(ay - y) < 0.01 ? "on" : ""}
              onClick=${() => { layer.x = ax; layer.y = ay; commit(`layer.${layer.id}.align`); }}></button>`)}
          </div>
          <div class="muted small">Click a spot, or drag the layer on the picture. Scroll over it to resize.</div>
        </div>
        ${PLACE_KEYS.map((n) => ctl(n))}
      </${Section}>
      <${Section} title="Style">
        <div class="chips">
          ${Object.keys(STYLE_PRESETS).map((p) => html`<button type="button" class="chip-btn" onClick=${() => applyStylePreset(layer.id, p)}>${p}</button>`)}
        </div>
        ${STYLE_KEYS.map((n) => ctl(n))}
      </${Section}>
      ${ctl("name", "Name")}
    </div>`;
}

function ShapePicker({ layer, spec }) {
  const cur = layerValue(layer, "shape");
  return html`<${Field} label="Shape">
    <div class="shape-grid">${(spec.choices || []).filter(Boolean).map((s) => html`
      <button type="button" class=${s === cur ? "on" : ""} onClick=${() => setLayer(layer.id, "shape", s)} title=${s}>
        <span class=${`shape-icon shape-${s}`}></span><span>${s}</span>
      </button>`)}</div>
  </${Field}>`;
}

function BackgroundEditor() {
  const bg = S.state.background;
  const spec = (n) => S.catalog.background_keys.find((k) => k.name === n);
  const fill = bg.fill ?? spec("fill").neutral;
  const image = /^(self|fractal|recipe:)/i.test(String(fill));
  return html`<${Section} title="Behind cut-outs">
    <${Field} label="Background" doc=${spec("fill").doc}>
      <${FillPicker} value=${fill} mode="background" onChange=${(v) => setBackground("fill", v)} />
    </${Field}>
    ${image ? ["dim", "blur"].map((n) => html`<${Control} key=${n} spec=${spec(n)} value=${bg[n] ?? spec(n).neutral}
      modified=${bg[n] !== undefined} onReset=${() => setBackground(n, null)} onChange=${(v) => setBackground(n, v)} />`) : null}
  </${Section}>`;
}

// ---------------------------------------------------------------- Effects ----

const CATEGORY_LABEL = { color: "Color", light: "Light", texture: "Texture", transform: "Transform" };

export function EffectsPanel() {
  const post = S.state.post;
  const [menu, setMenu] = useState(false);
  const byCat = {};
  for (const e of S.catalog.effects) (byCat[e.category] ||= []).push(e);
  return html`
    <div class="panel">
      <div class="add-row">
        <div class="menu-wrap">
          <button type="button" class="btn add" onClick=${() => setMenu(!menu)}>+ Add effect</button>
          ${menu ? html`<div class="dropdown" onMouseLeave=${() => setMenu(false)}>
            ${Object.keys(byCat).map((cat) => html`<div class="dd-group">${CATEGORY_LABEL[cat] || cat}</div>
              ${byCat[cat].map((e) => html`<button type="button" class="dd-item" onClick=${() => { addEffect(e.name); setMenu(false); }}>
                <span>${e.label}</span><span class="muted small">${e.doc}</span></button>`)}`)}
          </div>` : null}
        </div>
        ${post.length ? html`<button type="button" class="btn" onClick=${clearEffects}>Clear all</button>` : null}
      </div>
      <div class="chips"><span class="muted small">Start from a preset</span>
        ${Object.keys(EFFECT_PRESETS).map((p) => html`<button type="button" class="chip-btn" onClick=${() => applyEffectPreset(p)}>${p}</button>`)}
      </div>
      ${post.length ? html`<p class="muted small">Effects apply from top to bottom. Drag them to change the order.</p>
        <${EffectList} post=${post} />` : html`<div class="empty">
          <p>Finish the look with glow, film grain, color grading and more.</p>
          <p class="muted small">Start from a preset above, or add effects one at a time.</p></div>`}
    </div>`;
}

function EffectList({ post }) {
  const [dragId, setDragId] = useState(null);
  return html`<ul class="item-list effects">
    ${post.map((it, i) => {
      const spec = effectSpec(it.op);
      return html`<li key=${it.id} class=${`item card${it.enabled === false ? " off" : ""}${dragId === it.id ? " dragging" : ""}`}
        draggable="true" onDragStart=${(e) => { setDragId(it.id); e.dataTransfer.effectAllowed = "move"; }}
        onDragEnd=${() => setDragId(null)} onDragOver=${(e) => e.preventDefault()}
        onDrop=${(e) => { e.preventDefault(); if (dragId && dragId !== it.id) reorderEffect(dragId, i); setDragId(null); }}>
        <div class="card-head">
          <span class="grip" title="Drag to reorder"></span>
          <input type="checkbox" class="eye" title=${it.enabled === false ? "Turn this effect on" : "Turn this effect off"}
            checked=${it.enabled !== false} onChange=${() => setEffect(it.id, "enabled", it.enabled === false)} />
          <span class="item-name" title=${spec?.doc}>${spec?.label || it.op}</span>
          <span class="item-actions">
            <button type="button" class="icon-btn" title="Earlier" disabled=${i === 0} onClick=${() => moveEffect(it.id, -1)}><${Icon} name="arrow-up" /></button>
            <button type="button" class="icon-btn" title="Later" disabled=${i === post.length - 1} onClick=${() => moveEffect(it.id, 1)}><${Icon} name="arrow-down" /></button>
            <button type="button" class="icon-btn" title="Duplicate" onClick=${() => duplicateEffect(it.id)}><${Icon} name="copy" /></button>
            <button type="button" class="icon-btn danger" title="Remove" onClick=${() => removeEffect(it.id)}><${Icon} name="x" /></button>
          </span>
        </div>
        ${spec && it.enabled !== false ? html`<div class="card-body">
          ${spec.params.map((p) => html`<${Control} key=${p.name} spec=${p}
            label=${spec.params.length === 1 ? spec.label : p.label}
            value=${it[p.name] ?? p.neutral} onChange=${(v) => setEffect(it.id, p.name, v)} />`)}
          <${Field} label="Strength" doc="Lower it to blend the effect with the picture underneath.">
            <${Slider} value=${it.mix ?? 1} min=${0} max=${1} step=${0.01} onInput=${(v) => setEffect(it.id, "mix", v)} />
          </${Field}>
        </div>` : null}
      </li>`;
    })}
  </ul>`;
}

export { Toggle, Select };
