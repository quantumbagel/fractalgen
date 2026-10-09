// Dialogs and popovers: Browse looks, Save wallpaper, Variations, Save recipe, menu, shortcuts, toasts.

import { html, useEffect, useRef, useState } from "../vendor/preact-htm.js";
import { Segmented, Slider, Toggle } from "./controls.js";
import {
  S, closeOverlay, commit, downloadExport, exportDestination, isSlowLook, setExportDestination, copyCli, currentAspect, dismissToast, emit, loadRecipe, makeVariations, orient,
  pickVariation, revealExport, saveRecipe, setAspect, setHelp, setPreviewSize, startExport,
} from "./store.js";
import { Thumb } from "./thumbs.js";
import { downloadName } from "./util.js";

export { Thumb };

export function Modal({ title, children, onClose, wide = false, tall = false, footer = null }) {
  useEffect(() => {
    const k = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return html`
    <div class="backdrop" onClick=${(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div class=${`modal${wide ? " wide" : ""}${tall ? " tall" : ""}`} role="dialog" aria-modal="true" aria-label=${title}>
        <header class="modal-head"><h2>${title}</h2>
          <button type="button" class="icon-btn close" title="Close (Esc)" onClick=${onClose}>✕</button></header>
        <div class="modal-body">${children}</div>
        ${footer ? html`<footer class="modal-foot">${footer}</footer>` : null}
      </div>
    </div>`;
}

export const pretty = (name) => {
  const [base, variant] = name.split("@");
  return base.replace(/_/g, " ") + (variant ? ` (${variant.replace(/_/g, " ")})` : "");
};

// ---------------------------------------------------------------- browse ----

export function BrowseOverlay() {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("all");
  const recipes = S.catalog.recipes;
  const cats = [...new Set(recipes.map((r) => r.category))].sort();
  const hasCustom = recipes.some((r) => r.custom);
  const ql = q.trim().toLowerCase();
  const rows = recipes.filter((r) => (cat === "all" || (cat === "mine" ? r.custom : r.category === cat)) &&
    (!ql || [r.name, r.description, r.generator, r.category, ...(r.tags || [])].join(" ").toLowerCase().includes(ql)));
  const o = orient();
  const target = S.ui.pick;
  const pick = (name) => {
    closeOverlay();
    if (target) target.onPick(name);
    else loadRecipe(name);
  };
  return html`<${Modal} title=${target?.title || "Choose a look"} wide=${true} tall=${true} onClose=${closeOverlay}>
    <div class="browse-bar">
      <input type="search" class="filter big" placeholder="Search looks, like julia or neon" value=${q} autofocus
        onInput=${(e) => setQ(e.currentTarget.value)}
        onKeyDown=${(e) => { if (e.key === "Enter" && rows[0]) pick(rows[0].name); }} />
      <div class="chips">
        <button type="button" class=${`chip-btn${cat === "all" ? " on" : ""}`} onClick=${() => setCat("all")}>All ${recipes.length}</button>
        ${hasCustom ? html`<button type="button" class=${`chip-btn${cat === "mine" ? " on" : ""}`} onClick=${() => setCat("mine")}>My looks</button>` : null}
        ${cats.map((c) => html`<button type="button" class=${`chip-btn${cat === c ? " on" : ""}`} onClick=${() => setCat(c)}>${c}</button>`)}
      </div>
    </div>
    <div class=${`gallery ${o}`}>
      ${rows.map((r) => html`<button type="button" key=${r.name} class=${`card${r.name === S.state.recipe ? " current" : ""}`}
        title=${r.description || r.name} onClick=${() => pick(r.name)}>
        <${Thumb} recipe=${r.name} orient=${o} />
        <span class="card-name">${pretty(r.name)}${isSlowLook(r.name)
          ? html` <span class="chip slow" title="This one is a simulation and takes several seconds to draw">slow</span>` : null}</span>
        <span class="card-desc">${r.description || r.generator}</span>
      </button>`)}
      ${rows.length ? null : html`<div class="empty">No looks match “${q}”.</div>`}
    </div>
  </${Modal}>`;
}

// ---------------------------------------------------------------- export ----

export function ExportDialog() {
  const asp = currentAspect();
  const choices = [];
  const seen = new Set();
  const add = (label, w, h, note = "") => {
    const k = `${w}x${h}`;
    if (seen.has(k)) return;
    seen.add(k);
    choices.push({ key: k, label, w, h, note });
  };
  add(asp.label.replace(/ \(.*/, ""), asp.size[0], asp.size[1], "matches the preview");
  for (const a of S.aspects) if (a.id !== asp.id) add(a.label.replace(/ \(.*/, ""), a.size[0], a.size[1]);
  const [key, setKey] = useState(choices[0].key);
  const [custom, setCustom] = useState({ w: asp.size[0], h: asp.size[1] });
  const [fmt, setFmt] = useState("png");
  const [ss, setSs] = useState(1);
  const [jq, setJq] = useState(95);
  const [name, setName] = useState(S.state.recipe || S.state.generator);
  const job = S.exportJob;
  const chosen = key === "custom" ? { w: custom.w, h: custom.h } : choices.find((c) => c.key === key);
  const running = job && (job.status === "running" || job.status === "starting");
  const mismatch = chosen && (chosen.h > chosen.w) !== (S.state.height > S.state.width);
  const maxSs = chosen ? Math.max(1, Math.min(3, Math.floor(16384 / Math.max(chosen.w, chosen.h)))) : 1;
  const quality = Math.min(ss, maxSs);
  const dest = exportDestination();
  const go = () => startExport({ width: Math.round(chosen.w), height: Math.round(chosen.h), format: fmt, name,
                                 supersample: quality, jpeg_quality: jq, destination: dest });
  const mp = chosen ? (chosen.w * chosen.h / 1e6).toFixed(1) : 0;
  const QUALITY = [
    { value: 1, label: "Standard", title: "Renders at exactly the size you picked." },
    { value: 2, label: "Smooth", title: "Renders at twice the size and scales it down, for smoother edges and finer detail." },
    { value: 3, label: "Finest", title: "Renders at three times the size. The cleanest result, but much slower." },
  ].filter((q) => q.value <= maxSs);

  return html`<${Modal} title="Save wallpaper" onClose=${closeOverlay} footer=${html`
      <button type="button" class="btn ghost" onClick=${() => { S.ui.overlay = "save"; emit(); }}>Save as recipe…</button>
      <button type="button" class="btn ghost" onClick=${copyCli}>Copy command</button>
      <span class="grow"></span>
      <button type="button" class="btn primary big" disabled=${running || !chosen} onClick=${go}>
        ${running ? "Rendering…" : `${dest === "download" ? "Download" : "Save"} ${chosen ? `${chosen.w}×${chosen.h}` : ""}`}</button>`}>
    <div class="size-grid">
      ${choices.map((c) => html`<button type="button" class=${`size-card${key === c.key ? " on" : ""}`} onClick=${() => setKey(c.key)}>
        <span class="size-shape" style=${(() => { const k = Math.min(56 / c.w, 40 / c.h); return { width: `${c.w * k}px`, height: `${c.h * k}px` }; })()}></span>
        <span class="size-label">${c.label}</span><span class="size-dims">${c.w}×${c.h}</span>
        ${c.note ? html`<span class="size-note">${c.note}</span>` : null}
      </button>`)}
      <button type="button" class=${`size-card${key === "custom" ? " on" : ""}`} onClick=${() => setKey("custom")}>
        <span class="size-label">Custom</span>
        ${key === "custom" ? html`<span class="custom-dims" onClick=${(e) => e.stopPropagation()}>
          <input type="number" min="16" max="16384" value=${custom.w} onInput=${(e) => setCustom({ ...custom, w: +e.currentTarget.value })} />×
          <input type="number" min="16" max="16384" value=${custom.h} onInput=${(e) => setCustom({ ...custom, h: +e.currentTarget.value })} />
        </span>` : html`<span class="size-dims">any size</span>`}
      </button>
    </div>
    ${mismatch ? html`<div class="note warn">This size is ${chosen.h > chosen.w ? "portrait" : "landscape"} but the preview isn't.
      Looks frame themselves for each shape, so <button type="button" class="link"
      onClick=${() => setAspect(chosen.h > chosen.w ? "phone" : "desktop")}>preview that shape</button> first.</div>` : null}
    <div class="form-row">
      <label>Quality <${Segmented} value=${quality} options=${QUALITY} onChange=${setSs} /></label>
      <label>Format <${Segmented} value=${fmt} options=${[{ value: "png", label: "PNG" }, { value: "jpg", label: "JPEG" }]}
        onChange=${setFmt} /></label>
      ${fmt === "jpg" ? html`<label class="jpeg-q">JPEG quality <${Slider} value=${jq} min=${60} max=${100} step=${1} int=${true}
        onInput=${setJq} /></label>` : null}
      <label>Save to <${Segmented} value=${dest} onChange=${setExportDestination} options=${[
        { value: "download", label: "Download", title: "Your browser saves it, usually to your Downloads folder." },
        { value: "folder", label: "Wallpapers folder", title: "Saved in the backgrounds/gui folder of this project." }]} /></label>
      <label class="grow">File name <input type="text" value=${name} onInput=${(e) => setName(e.currentTarget.value)} /></label>
    </div>
    <div class="muted small">${quality > 1 ? `Rendered at ${chosen.w * quality}×${chosen.h * quality} and scaled down for smoother edges. ` : ""}
      Large sizes, higher quality and simulations take a while, but you can keep editing in the meantime.</div>
    ${job ? html`<${ExportStatus} job=${job} />` : null}
  </${Modal}>`;
}

function ExportStatus({ job }) {
  if (job.status === "error") {
    return html`<div class="note error">Export failed: ${(job.error || "").split("\n").filter(Boolean).pop()}</div>`;
  }
  if (job.status !== "done") {
    const pct = Math.round(100 * (job.progress || 0));
    const stage = job.stage ? job.stage[0].toUpperCase() + job.stage.slice(1) : "Starting";
    const secs = job.seconds || 0;
    const eta = job.progress > 0.05 ? Math.max(0, secs / job.progress - secs) : null;
    return html`<div class="export-progress">
      <div class="bar"><div class="fill" style=${{ width: `${pct}%` }}></div></div>
      <div class="muted small">${stage}, ${pct}% done${eta != null ? `, about ${Math.ceil(eta)} seconds left` : ""}</div>
    </div>`;
  }
  const name = downloadName(job.path);
  return html`<div class="export-done">
    <img src=${`/api/export/${job.id}/file`} alt="" />
    <div class="grow">
      <div><b>${job.download ? "Downloaded" : "Saved"}</b> ${name} <span class="muted small">in ${(job.seconds || 0).toFixed(1)}s</span></div>
      ${job.download
        ? html`<div class="muted small">Check your browser's downloads.</div>`
        : html`<div class="muted small path" title=${job.path}>${job.path}</div>`}
      <div class="btn-row">
        ${job.download
          ? html`<button type="button" class="btn" onClick=${() => downloadExport(job)}>Download again</button>`
          : html`<button type="button" class="btn" onClick=${revealExport}>Show in ${S.catalog.platform === "mac" ? "Finder" : "folder"}</button>
                 <a class="btn" href=${`/api/export/${job.id}/file`} download=${name}>Download a copy</a>`}
      </div>
    </div>
  </div>`;
}

// ---------------------------------------------------------------- save recipe ----

export function SaveDialog() {
  const [name, setName] = useState((S.state.recipe || S.state.generator || "my_look").replace(/[^A-Za-z0-9_-]/g, "_"));
  const [overwrite, setOverwrite] = useState(false);
  const [msg, setMsg] = useState(null);
  const exists = S.catalog.recipes.some((r) => r.name === name);
  const save = async () => {
    try {
      const info = await saveRecipe(name, overwrite);
      setMsg({ ok: true, text: `Saved to ${info.path}` });
    } catch (e) {
      setMsg({ ok: false, text: e.message });
    }
  };
  return html`<${Modal} title="Save as recipe" onClose=${closeOverlay} footer=${html`
      <span class="grow"></span>
      <button type="button" class="btn" onClick=${closeOverlay}>Close</button>
      <button type="button" class="btn primary" disabled=${!name || (exists && !overwrite)} onClick=${save}>Save</button>`}>
    <p class="muted">Keeps this look as a recipe file in <code>custom/</code>. It will show up under My looks, and on the command line.</p>
    <label>Name <input type="text" value=${name} autofocus onInput=${(e) => setName(e.currentTarget.value.replace(/[^A-Za-z0-9_-]/g, "_"))}
      onKeyDown=${(e) => { if (e.key === "Enter") save(); }} /></label>
    ${exists ? html`<label class="check"><input type="checkbox" checked=${overwrite} onChange=${(e) => setOverwrite(e.currentTarget.checked)} />
      Replace the existing “${name}”</label>` : null}
    ${msg ? html`<div class=${`note ${msg.ok ? "ok" : "error"}`}>${msg.text}</div>` : null}
  </${Modal}>`;
}

// ---------------------------------------------------------------- variations ----

export function VariationsOverlay() {
  const [strength, setStrength] = useState(0.12);
  const o = orient();
  return html`<${Modal} title="Variations" wide=${true} onClose=${closeOverlay} footer=${html`
      <label class="inline-slider">Subtle <${Slider} value=${strength} min=${0.03} max=${0.4} step=${0.01} onInput=${setStrength} /> Wild</label>
      <span class="grow"></span>
      <button type="button" class="btn primary" onClick=${() => makeVariations(8, strength)}>Shuffle again</button>`}>
    <p class="muted small">Eight random takes on this look, with your framing kept as it is. Pick one to carry on from it. Undo takes you back.</p>
    <div class=${`gallery variations ${o}`}>
      ${S.variations.map((v, i) => html`<button type="button" class="card" key=${i} disabled=${!v.url} onClick=${() => pickVariation(i)}>
        <div class=${`thumb ${o}`}>${v.url ? html`<img src=${v.url} alt="" />` : v.error ? html`<span class="muted small">${v.error}</span>` : html`<span class="spinner"></span>`}</div>
      </button>`)}
    </div>
  </${Modal}>`;
}

// ---------------------------------------------------------------- menu / help ----

/** The ⋯ menu, fixed under its button (the top bar clips overflow on phones). */
export function Menu({ anchor }) {
  const close = () => { S.ui.menu = false; emit(); };
  useEffect(() => {
    const away = (e) => { if (!e.target.closest(".menu-pop, button[title='More']")) close(); };
    window.addEventListener("pointerdown", away);
    return () => window.removeEventListener("pointerdown", away);
  }, []);
  const pos = { top: `${anchor.bottom + 4}px`, right: `${Math.max(8, window.innerWidth - anchor.right)}px` };
  return html`<div class="menu-pop" style=${pos}>
    <button type="button" class="dd-item" onClick=${() => { S.ui.overlay = "save"; close(); }}>Save as recipe…</button>
    <button type="button" class="dd-item" onClick=${() => { copyCli(); close(); }}>Copy command line</button>
    <div class="dd-sep"></div>
    <div class="dd-item static">
      <span>Preview quality</span>
      <${Slider} value=${S.previewSize} min=${240} max=${2000} step=${20} int=${true} unit="px"
        onInput=${(v) => setPreviewSize(v)} />
    </div>
    <div class="dd-item static"><${Toggle} value=${S.ui.help} onChange=${setHelp} label="Show explanations under controls" /></div>
    <div class="dd-sep"></div>
    <button type="button" class="dd-item" onClick=${() => { S.ui.overlay = "keys"; close(); }}>Keyboard shortcuts</button>
  </div>`;
}

const KEYS = [
  ["⌘Z / ⇧⌘Z", "Undo / redo"], ["B", "Browse looks"], ["S", "Save wallpaper"], ["V", "Variations"],
  [". (period)", "Surprise me"], ["[ / ]", "Previous / next look"], ["R", "Reset framing"], ["\\ (hold)", "Compare with the original"],
  ["1 to 5", "Switch between the tabs"], ["Arrows", "Nudge the selected layer (Shift: more)"],
  ["Delete", "Delete the selected layer"], ["Esc", "Close dialogs / deselect"], ["?", "This list"],
  ["Drag / scroll", "Pan / zoom the picture (or move / resize a layer on the Layers tab)"],
];

export function KeysOverlay() {
  return html`<${Modal} title="Keyboard shortcuts" onClose=${closeOverlay}>
    <dl class="keys">${KEYS.map(([k, d]) => html`<dt><kbd>${k}</kbd></dt><dd>${d}</dd>`)}</dl>
  </${Modal}>`;
}

export function Toasts() {
  return html`<div class="toasts" aria-live="polite">${S.ui.toasts.map((t) => html`
    <div class=${`toast ${t.kind}`} key=${t.id}>
      <span>${t.text}</span>
      ${t.detail ? html`<details><summary>Details</summary><pre>${t.detail}</pre></details>` : null}
      <button type="button" class="icon-btn" onClick=${() => dismissToast(t.id)}>✕</button>
    </div>`)}</div>`;
}

export { commit };
