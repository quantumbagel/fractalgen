// Form controls. `Control` renders any schema'd value (generator params, layer / effect / background keys).

import { html, useEffect, useRef, useState } from "../vendor/preact-htm.js";
import { S, isSlowLook, pickLook, showError } from "./store.js";
import { Thumb } from "./thumbs.js";
import { apiUpload, fmtNum, hexOf, paletteGradient } from "./util.js";

// ---------------------------------------------------------------- layout ----

export function Field({ label, doc, modified, onReset, children, inline = false, extra = null }) {
  return html`
    <div class=${`field${modified ? " mod" : ""}${inline ? " inline" : ""}`}>
      <div class="field-head">
        <span class="field-label" title=${doc || ""}>${label}</span>
        ${extra}
        ${modified && onReset
          ? html`<button type="button" class="link reset" title="Go back to the look's own value" onClick=${onReset}>Reset</button>`
          : null}
      </div>
      <div class="field-body">${children}</div>
      ${S.ui.help && doc ? html`<div class="doc">${doc}</div>` : null}
    </div>`;
}

export function Section({ title, children, open = true, count = null, actions = null, id = null }) {
  const [isOpen, setOpen] = useState(open);
  return html`
    <section class=${`section${isOpen ? "" : " closed"}`} id=${id}>
      <header class="section-head" onClick=${() => setOpen(!isOpen)}>
        <span class="chev">${isOpen ? "▾" : "▸"}</span>
        <h3>${title}</h3>
        ${count != null ? html`<span class="count">${count}</span>` : null}
        <span class="grow"></span>
        <span onClick=${(e) => e.stopPropagation()}>${actions}</span>
      </header>
      ${isOpen ? html`<div class="section-body">${children}</div>` : null}
    </section>`;
}

// ---------------------------------------------------------------- primitives ----

/** Slider + number box. `log`: the slider moves on a log10 scale (positive values only). */
export function Slider({ value, min, max, step, int = false, log = false, unit = "", onInput }) {
  const v = Number(value);
  let lo = Number(min), hi = Number(max);
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) { lo = (v || 0) - 1; hi = (v || 0) + 1; }
  if (v < lo) lo = v;
  if (v > hi) hi = v;
  const toS = (x) => (log ? Math.log10(Math.max(x, 1e-300)) : x);
  const fromS = (x) => (log ? Math.pow(10, x) : x);
  const sStep = log ? "any" : step || (int ? 1 : "any");
  const emitV = (x) => {
    if (!Number.isFinite(x)) return;
    onInput(int ? Math.round(x) : x);
  };
  return html`
    <div class="slider">
      <input type="range" min=${toS(lo)} max=${toS(hi)} step=${sStep} value=${toS(v)}
        onInput=${(e) => emitV(fromS(parseFloat(e.currentTarget.value)))} />
      <input type="number" class="num" step=${int ? 1 : "any"} value=${int ? Math.round(v) : fmtNum(v, 6)}
        onChange=${(e) => emitV(parseFloat(e.currentTarget.value))} />
      ${unit ? html`<span class="unit">${unit}</span>` : null}
    </div>`;
}

export function Toggle({ value, onChange, label = null }) {
  return html`
    <label class="toggle">
      <input type="checkbox" checked=${!!value} onChange=${(e) => onChange(e.currentTarget.checked)} />
      <span class="track"><span class="knob"></span></span>
      ${label ? html`<span class="toggle-label">${label}</span>` : null}
    </label>`;
}

export function Select({ value, options, onChange, title = "" }) {
  return html`
    <select value=${value} title=${title} onChange=${(e) => onChange(e.currentTarget.value)}>
      ${options.map((o) => (typeof o === "string"
        ? html`<option value=${o}>${o || "(none)"}</option>`
        : html`<option value=${o.value}>${o.label}</option>`))}
    </select>`;
}

export function Segmented({ value, options, onChange }) {
  return html`
    <div class="segmented" role="radiogroup">
      ${options.map((o) => html`
        <button type="button" role="radio" aria-checked=${o.value === value}
          class=${o.value === value ? "on" : ""} title=${o.title || ""} onClick=${() => onChange(o.value)}>
          ${o.label}
        </button>`)}
    </div>`;
}

export function ColorInput({ value, onChange }) {
  const asArray = Array.isArray(value);
  const fromHex = (h) => {
    if (!asArray) return h.replace("#", "");
    const n = parseInt(h.slice(1), 16);
    return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  };
  return html`
    <div class="color-input">
      <input type="color" value=${hexOf(value)} onInput=${(e) => onChange(fromHex(e.currentTarget.value))} />
      <input type="text" class="hex" value=${hexOf(value).slice(1)} maxlength="7"
        onChange=${(e) => { const h = hexOf(e.currentTarget.value); onChange(fromHex(h)); }} />
    </div>`;
}

/** Text that applies on Enter / blur (or as you type when `live`). */
export function TextInput({ value, onCommit, multiline = false, placeholder = "", list = null, live = false }) {
  const [draft, setDraft] = useState(value ?? "");
  useEffect(() => setDraft(value ?? ""), [value]);
  const commit = (v) => { if (v !== (value ?? "")) onCommit(v); };
  if (multiline) {
    return html`<textarea rows="2" placeholder=${placeholder} value=${draft}
      onInput=${(e) => { setDraft(e.currentTarget.value); if (live) onCommit(e.currentTarget.value); }}
      onBlur=${(e) => commit(e.currentTarget.value)}
      onKeyDown=${(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) commit(e.currentTarget.value); }} />`;
  }
  return html`<input type="text" placeholder=${placeholder} list=${list} value=${draft}
    onInput=${(e) => { setDraft(e.currentTarget.value); if (live) onCommit(e.currentTarget.value); }}
    onBlur=${(e) => commit(e.currentTarget.value)}
    onKeyDown=${(e) => { if (e.key === "Enter") commit(e.currentTarget.value); }} />`;
}

// ---------------------------------------------------------------- palettes ----

const knownPalette = (n) => S.catalog.palettes.find((p) => p.name === n);
const baseName = (v) => (typeof v === "string" && v.endsWith("_r") && knownPalette(v.slice(0, -2)) ? v.slice(0, -2) : v);

export function paletteStops(v) {
  if (v == null) return null;
  if (Array.isArray(v)) return v;
  if (typeof v === "object") return v.stops || null;
  const p = knownPalette(baseName(v));
  if (!p) return null;
  return baseName(v) !== v ? [...p.stops].reverse() : p.stops;
}

export function PaletteSwatch({ value, big = false }) {
  const stops = paletteStops(value);
  const pos = value && !Array.isArray(value) && typeof value === "object" ? value.positions : null;
  return html`<span class=${`swatch${big ? " big" : ""}`} style=${{ background: paletteGradient(stops, pos) }}></span>`;
}

/** Every palette as a clickable swatch, plus reverse and a stop editor. */
export function PaletteGrid({ value, onChange, allowNone = false }) {
  const reversed = typeof value === "string" && baseName(value) !== value;
  const current = typeof value === "string" ? baseName(value) : value == null ? null : "__custom__";
  const pick = (name) => onChange(reversed ? `${name}_r` : name);
  const reverse = () => {
    if (typeof value === "string") onChange(reversed ? baseName(value) : `${value}_r`);
    else if (Array.isArray(value)) onChange([...value].reverse());
    else if (value?.stops) {
      onChange({ ...value, stops: [...value.stops].reverse(),
                 ...(value.positions ? { positions: value.positions.map((p) => 1 - p).reverse() } : {}) });
    }
  };
  return html`
    <div class="palette-grid">
      ${allowNone ? html`<button type="button" class=${`pal ${value == null ? "on" : ""}`} onClick=${() => onChange(null)}>
        <span class="swatch none"></span><span class="pal-name">None</span></button>` : null}
      ${S.catalog.palettes.map((p) => html`
        <button type="button" class=${`pal ${current === p.name ? "on" : ""}`} title=${p.description || p.name}
          onClick=${() => pick(p.name)}>
          <span class="swatch" style=${{ background: paletteGradient(reversed && current === p.name ? [...p.stops].reverse() : p.stops) }}></span>
          <span class="pal-name">${p.name}</span>
        </button>`)}
    </div>
    <div class="row-actions">
      <button type="button" class="btn small" disabled=${value == null} onClick=${reverse}>⇄ Reverse</button>
      <${StopsEditor} value=${value} onChange=${onChange} />
    </div>`;
}

function StopsEditor({ value, onChange }) {
  const [open, setOpen] = useState(value != null && typeof value === "object");
  const stops = paletteStops(value) || ["000000", "ffffff"];
  const edit = (fn) => {
    const next = [...stops];
    fn(next);
    onChange(next);
  };
  if (!open) {
    return html`<button type="button" class="btn small" onClick=${() => setOpen(true)}>Custom colors…</button>`;
  }
  return html`
    <div class="stops">
      ${stops.map((hex, i) => html`
        <div class="stop">
          <input type="color" value=${hexOf(hex)} onInput=${(e) => edit((s) => { s[i] = e.currentTarget.value.slice(1); })} />
          <button type="button" class="icon-btn" title="Remove" disabled=${stops.length <= 2}
            onClick=${() => edit((s) => s.splice(i, 1))}>−</button>
          <button type="button" class="icon-btn" title="Add a color after this one"
            onClick=${() => edit((s) => s.splice(i + 1, 0, s[i]))}>+</button>
        </div>`)}
      <button type="button" class="btn small" onClick=${() => setOpen(false)}>Done</button>
    </div>`;
}

/** Compact palette field: swatch + name; opens the grid inline. */
export function PaletteField({ value, onChange, allowNone = false }) {
  const [open, setOpen] = useState(false);
  const name = value == null ? "None" : typeof value === "string" ? value : "Custom";
  return html`
    <div class="palette-field">
      <button type="button" class="palette-current" onClick=${() => setOpen(!open)}>
        <${PaletteSwatch} value=${value} /><span>${name}</span><span class="chev">${open ? "▴" : "▾"}</span>
      </button>
      ${open ? html`<${PaletteGrid} value=${value} onChange=${onChange} allowNone=${allowNone} />` : null}
    </div>`;
}

// ---------------------------------------------------------------- fills ----

function fillKind(v) {
  const s = String(v ?? "").toLowerCase();
  if (s === "fractal" || s === "self") return "fractal";
  if (s === "image") return "image";
  if (s === "transparent" || s === "none" || s === "alpha") return "transparent";
  if (s.startsWith("palette:")) return "palette";
  if (s.startsWith("recipe:")) return "recipe";
  return "color";
}

/** What fills a layer or the background. `mode`: "layer" | "image-layer" | "background". */
export function FillPicker({ value, onChange, mode = "layer" }) {
  const kind = fillKind(value);
  const opts = mode === "background"
    ? [{ value: "color", label: "Color" }, { value: "fractal", label: "Dimmed look", title: "The look itself, darkened (and blurred)" },
       { value: "palette", label: "Gradient" }, { value: "recipe", label: "Other look" },
       { value: "transparent", label: "Transparent", title: "PNG with transparency" }]
    : [{ value: "fractal", label: "Fractal", title: "The render shows through: a cut-out" },
       ...(mode === "image-layer" ? [{ value: "image", label: "Picture", title: "The image's own colors" }] : []),
       { value: "color", label: "Color" }, { value: "palette", label: "Gradient" }, { value: "recipe", label: "Other look" }];
  const switchTo = (k) => {
    if (k === kind) return;
    if (k === "fractal") onChange(mode === "background" ? "self" : "fractal");
    else if (k === "image") onChange("image");
    else if (k === "transparent") onChange("transparent");
    else if (k === "palette") onChange(`palette:${S.catalog.palettes[0].name}`);
    else if (k === "recipe") chooseLook();
    else onChange(mode === "background" ? "000000" : "ffffff");
  };
  const spec = String(value ?? "");
  const chooseLook = () => pickLook(mode === "background" ? "Background: choose a look" : "Fill the layer with which look?",
    (name) => onChange(`recipe:${name}`));
  return html`
    <div class="fill-picker">
      <${Segmented} value=${kind} options=${opts} onChange=${switchTo} />
      ${kind === "color" ? html`<${ColorInput} value=${spec || "000000"} onChange=${onChange} />` : null}
      ${kind === "palette" ? html`<${PaletteField} value=${spec.slice(8)} onChange=${(p) => onChange(`palette:${p}`)} />` : null}
      ${kind === "recipe" ? html`<${LookChoice} name=${spec.slice(7)} onClick=${chooseLook} />` : null}
    </div>`;
}

/** The look a fill uses: its thumbnail and name; click to pick another from the gallery. */
function LookChoice({ name, onClick }) {
  return html`<button type="button" class="look-choice" title="Choose another look" onClick=${onClick}>
    <${Thumb} recipe=${name} orient="land" key=${name} />
    <span class="look-choice-name">${name.replace(/_/g, " ")}${isSlowLook(name) ? html` <span class="chip slow">slow</span>` : null}</span>
    <span class="muted small">Change…</span>
  </button>`;
}

// ---------------------------------------------------------------- image upload ----

export function ImagePicker({ value, onChange }) {
  const input = useRef(null);
  const [busy, setBusy] = useState(false);
  const upload = async (file) => {
    if (!file) return;
    setBusy(true);
    try {
      const { path } = await apiUpload(file);
      onChange(path);
    } catch (e) {
      showError(e, "Upload failed: ");
    } finally {
      setBusy(false);
    }
  };
  return html`
    <div class="image-picker"
      onDragOver=${(e) => { e.preventDefault(); }}
      onDrop=${(e) => { e.preventDefault(); upload(e.dataTransfer.files?.[0]); }}>
      <button type="button" class="btn" disabled=${busy} onClick=${() => input.current.click()}>
        ${busy ? "Uploading…" : value ? "Replace image…" : "Choose image…"}
      </button>
      <input ref=${input} type="file" accept="image/*" hidden onChange=${(e) => upload(e.currentTarget.files?.[0])} />
      ${value ? html`<span class="file-name" title=${value}>${String(value).split("/").pop()}</span>` : html`<span class="muted">or drop one here</span>`}
    </div>`;
}

// ---------------------------------------------------------------- generic control ----

function numRange(spec, value) {
  const r = spec.range || (spec.kind === "int" ? [0, Math.max(10, Math.abs(value) * 4), 1] : [-1, 1]);
  return { min: r[0], max: r[1], step: r[2] };
}

/** One value from a schema entry: {name, label, kind, doc, range, choices, neutral|default, unit}. */
export function Control({ spec, value, onChange, modified = false, onReset = null, label = null }) {
  const kind = spec.kind || "float";
  const lbl = label || spec.label || spec.name;
  const unit = spec.unit === "px" ? "px" : spec.unit === "deg" ? "°" : spec.unit || "";
  const field = (body, inline = false) => html`<${Field} label=${lbl} doc=${spec.doc} modified=${modified}
    onReset=${onReset} inline=${inline}>${body}</${Field}>`;
  if (kind === "bool") return field(html`<${Toggle} value=${value} onChange=${onChange} />`, true);
  if (kind === "int" || kind === "float") {
    const { min, max, step } = numRange(spec, Number(value));
    return field(html`<${Slider} value=${value ?? spec.neutral ?? spec.default ?? 0} min=${min} max=${max} step=${step}
      int=${kind === "int"} unit=${unit} onInput=${onChange} />`);
  }
  if (kind === "choice") {
    const choices = spec.choices || [];
    if (choices.length <= 4 && choices.every((c) => String(c).length < 10)) {
      return field(html`<${Segmented} value=${value} options=${choices.map((c) => ({ value: c, label: c || "none" }))}
        onChange=${onChange} />`);
    }
    return field(html`<${Select} value=${value} options=${choices} onChange=${onChange} />`);
  }
  if (kind === "color") {
    const nullable = spec.default === null || spec.neutral === null;
    if (nullable) {
      return field(html`<div class="hbox"><${Toggle} value=${value != null}
        onChange=${(on) => onChange(on ? [1, 1, 1] : null)} />
        ${value != null ? html`<${ColorInput} value=${value} onChange=${onChange} />` : null}</div>`);
    }
    return field(html`<${ColorInput} value=${value ?? spec.neutral} onChange=${onChange} />`);
  }
  if (kind === "palette") {
    return field(html`<${PaletteField} value=${value} onChange=${onChange} allowNone=${spec.default === null} />`);
  }
  if (kind === "font") {
    return field(html`<${TextInput} value=${value || ""} list="fontList" placeholder="Heavy system font"
      onCommit=${onChange} />`);
  }
  if (kind === "path") return field(html`<${ImagePicker} value=${value} onChange=${onChange} />`);
  if (kind === "fill") return field(html`<${FillPicker} value=${value} onChange=${onChange} />`);
  if (kind === "complex") {
    const c = Array.isArray(value) ? value : [0, 0];
    const mag = Math.max(2, 2 * Math.hypot(c[0], c[1]));
    return field(html`<div class="vbox">
      <div class="sub"><span class="sub-label">real</span><${Slider} value=${c[0]} min=${-mag} max=${mag} step=${0.0005}
        onInput=${(x) => onChange([x, c[1]])} /></div>
      <div class="sub"><span class="sub-label">imag</span><${Slider} value=${c[1]} min=${-mag} max=${mag} step=${0.0005}
        onInput=${(x) => onChange([c[0], x])} /></div></div>`);
  }
  if (kind === "vector") {
    const arr = Array.isArray(value) ? value : [];
    const names = arr.length === 3 && /color|light|dir|cam|pos/i.test(spec.name) ? ["x", "y", "z"] : null;
    return field(html`<div class="vbox">${arr.map((x, i) => {
      const mag = Math.max(1, 3 * Math.abs(x));
      return html`<div class="sub"><span class="sub-label">${names ? names[i] : i}</span>
        <${Slider} value=${x} min=${-mag} max=${mag} onInput=${(v) => { const n = [...arr]; n[i] = v; onChange(n); }} /></div>`;
    })}</div>`);
  }
  if (kind === "numdict") {
    const d = value || {};
    return field(html`<div class="vbox">${Object.keys(d).sort().map((k) => {
      const mag = Math.max(1, 3 * Math.abs(d[k]));
      return html`<div class="sub"><span class="sub-label">${k}</span>
        <${Slider} value=${d[k]} min=${-mag} max=${mag} onInput=${(v) => onChange({ ...d, [k]: v })} /></div>`;
    })}</div>`);
  }
  if (kind === "json") return field(html`<${JsonInput} value=${value} onChange=${onChange} />`);
  if (spec.name === "text") {
    return field(html`<${TextInput} value=${value ?? ""} multiline=${true} live=${true} onCommit=${onChange} />`);
  }
  return field(html`<${TextInput} value=${value ?? ""} onCommit=${onChange} />`);
}

function JsonInput({ value, onChange }) {
  const [bad, setBad] = useState(false);
  return html`<textarea rows="3" class=${bad ? "bad" : ""} value=${JSON.stringify(value, null, 1)}
    onBlur=${(e) => {
      try { onChange(JSON.parse(e.currentTarget.value)); setBad(false); } catch (_) { setBad(true); }
    }} />`;
}
