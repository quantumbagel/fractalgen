// fractalgen GUI: top bar, side panel tabs, preview, dialogs and keyboard shortcuts.

import { html, render, useEffect } from "../vendor/preact-htm.js";
import { Icon } from "./icons.js";
import {
  BrowseOverlay, ExportDialog, KeysOverlay, Menu, SaveDialog, Thumb, Toasts, VariationsOverlay,
} from "./overlays.js";
import { ColorPanel, EffectsPanel, FramePanel, LayersPanel, LookPanel } from "./panels.js";
import { Stage } from "./preview.js";
import {
  S, boot, closeOverlay, commit, emit, makeVariations, orient, redo, removeLayer, resetAll, resetView, selectLayer,
  setAspect, showError, startCompare, stepRecipe, stopCompare, surprise, undo, useStore,
} from "./store.js";
import * as store from "./store.js";
import { planeFromClient } from "./preview.js";
import { clamp, formFocused } from "./util.js";

window.fg = { ...store, planeFromClient }; // for tools/gui_check.py and the browser console

const TABS = [
  { id: "look", label: "Look" },
  { id: "frame", label: "Frame" },
  { id: "color", label: "Color" },
  { id: "layers", label: "Layers" },
  { id: "effects", label: "Effects" },
];

const PANELS = { look: LookPanel, frame: FramePanel, color: ColorPanel, layers: LayersPanel, effects: EffectsPanel };

function setTab(id) {
  S.ui.tab = id;
  if (window.innerWidth < 800) S.ui.sheet = true;
  emit();
}

function TopBar() {
  const st = S.state;
  const groups = {};
  for (const a of S.aspects) (groups[a.group] ||= []).push(a);
  const changed = S.modified.size > 0;
  return html`
    <header class="topbar">
      <div class="brand">fractalgen</div>
      <button type="button" class="look-btn" title="Browse looks (B)" onClick=${() => { S.ui.overlay = "browse"; emit(); }}>
        ${st?.recipe ? html`<${Thumb} recipe=${st.recipe} orient="land" key=${st.recipe} />` : html`<span class="thumb land"></span>`}
        <span class="look-btn-text"><span class="look-btn-name">${st?.recipe?.replace(/_/g, " ") || st?.generator || "…"}</span>
          <span class="look-btn-sub">${changed ? "edited" : "Browse looks"}</span></span>
        <span class="chev"><${Icon} name="chevron-down" /></span>
      </button>
      <button type="button" class="icon-btn big step-btn" title="Previous look ([)" onClick=${() => stepRecipe(-1)}><${Icon} name="chevron-left" size=${18} /></button>
      <button type="button" class="icon-btn big step-btn" title="Next look (])" onClick=${() => stepRecipe(1)}><${Icon} name="chevron-right" size=${18} /></button>
      <button type="button" class="icon-btn big" title="Surprise me (.)" onClick=${surprise}><${Icon} name="dice" size=${18} /></button>
      <span class="grow"></span>
      <label class="screen-pick" title="The screen you are making this for">
        <span class="lbl">Screen</span>
        <select value=${S.aspect} onChange=${(e) => setAspect(e.currentTarget.value)}>
          ${Object.entries(groups).map(([g, list]) => html`<optgroup label=${g}>
            ${list.map((a) => html`<option value=${a.id}>${a.label}</option>`)}</optgroup>`)}
        </select>
      </label>
      <div class="btn-group">
        <button type="button" class="icon-btn big" title="Undo (⌘Z)" disabled=${!S.history.undo.length} onClick=${undo}><${Icon} name="undo" size=${18} /></button>
        <button type="button" class="icon-btn big redo-btn" title="Redo (⇧⌘Z)" disabled=${!S.history.redo.length} onClick=${redo}><${Icon} name="redo" size=${18} /></button>
        <button type="button" class="btn ghost" title="Undo every change to this look" disabled=${!changed} onClick=${resetAll}>Reset</button>
      </div>
      <div class="menu-wrap">
        <button type="button" class="icon-btn big" title="More" onClick=${(e) => {
          S.ui.menu = S.ui.menu ? false : e.currentTarget.getBoundingClientRect();
          emit();
        }}><${Icon} name="more" size=${18} /></button>
      </div>
      <button type="button" class="btn primary" title="Render at full size and save (S)"
        onClick=${() => { S.ui.overlay = "export"; emit(); }}><span class="long">Save wallpaper</span><span class="short">Save</span></button>
    </header>`;
}

function Sidebar() {
  const Panel = PANELS[S.ui.tab] || LookPanel;
  const counts = { layers: S.state.layers.length, effects: S.state.post.length };
  return html`
    <aside class=${`sidebar${S.ui.sheet ? " open" : ""}`}>
      <nav class="tabs" role="tablist">
        ${TABS.map((t) => html`<button type="button" role="tab" aria-selected=${S.ui.tab === t.id}
          class=${`tab${S.ui.tab === t.id ? " on" : ""}`} onClick=${() => setTab(t.id)}>
          <span class="tab-label">${t.label}</span>
          ${counts[t.id] ? html`<span class="badge">${counts[t.id]}</span>` : null}
        </button>`)}
        <button type="button" class="tab sheet-close" aria-label="Close panel" onClick=${() => { S.ui.sheet = false; emit(); }}><${Icon} name="x" /></button>
      </nav>
      <div class="panel-scroll"><${Panel} /></div>
    </aside>`;
}

function App() {
  useStore();
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") {
        if (S.ui.overlay) closeOverlay();
        else if (S.ui.menu) { S.ui.menu = false; emit(); } else if (S.ui.selected) selectLayer(null);
        return;
      }
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "z" && !formFocused()) {
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
        return;
      }
      if (mod && e.key.toLowerCase() === "y" && !formFocused()) { e.preventDefault(); redo(); return; }
      if (mod || e.altKey || formFocused() || S.ui.overlay || !S.state) return;
      const sel = S.state.layers.find((l) => l.id === S.ui.selected);
      if (sel && e.key.startsWith("Arrow")) {
        e.preventDefault();
        const d = e.shiftKey ? 0.05 : 0.01;
        const dx = e.key === "ArrowLeft" ? -d : e.key === "ArrowRight" ? d : 0;
        const dy = e.key === "ArrowUp" ? -d : e.key === "ArrowDown" ? d : 0;
        sel.x = clamp((sel.x ?? 0.5) + dx, -0.25, 1.25);
        sel.y = clamp((sel.y ?? 0.5) + dy, -0.25, 1.25);
        commit(`layer.${sel.id}.nudge`);
        return;
      }
      if (sel && (e.key === "Delete" || e.key === "Backspace")) { e.preventDefault(); removeLayer(sel.id); return; }
      const k = e.key;
      if (k === "?") { S.ui.overlay = "keys"; emit(); } else if (k === "b") { S.ui.overlay = "browse"; emit(); } else if (k === "s") {
        S.ui.overlay = "export"; emit();
      } else if (k === "v") makeVariations();
      else if (k === ".") surprise();
      else if (k === "[") stepRecipe(-1);
      else if (k === "]") stepRecipe(1);
      else if (k === "r") resetView();
      else if (k === "\\" && !e.repeat) startCompare();
      else if (/^[1-5]$/.test(k)) setTab(TABS[Number(k) - 1].id);
    };
    const onKeyUp = (e) => { if (e.key === "\\") stopCompare(); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKeyUp);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("keyup", onKeyUp); };
  }, []);

  if (!S.catalog || !S.state) {
    return html`<div class="splash"><div class="brand big">fractalgen</div>
      <div class="muted">Warming up. The first picture takes a few seconds.</div><${Toasts} /></div>`;
  }
  const o = S.ui.overlay;
  return html`
    <div class=${`app ${orient()}${S.ui.help ? "" : " no-help"}`}>
      <${TopBar} />
      <main class="main">
        <${Sidebar} />
        <${Stage} />
      </main>
      <datalist id="fontList">${S.catalog.fonts.map((f) => html`<option value=${f} />`)}</datalist>
      ${o === "browse" ? html`<${BrowseOverlay} />` : null}
      ${o === "export" ? html`<${ExportDialog} />` : null}
      ${o === "save" ? html`<${SaveDialog} />` : null}
      ${o === "variations" ? html`<${VariationsOverlay} />` : null}
      ${o === "keys" ? html`<${KeysOverlay} />` : null}
      ${S.ui.menu ? html`<${Menu} anchor=${S.ui.menu} />` : null}
      <${Toasts} />
    </div>`;
}

render(html`<${App} />`, document.getElementById("app"));
boot().catch((e) => showError(e, "Could not start: "));
