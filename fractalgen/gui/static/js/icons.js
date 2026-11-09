// Line icons rendered on a 24-unit grid with 1.5 stroke width, styled via button text colors.

import { html } from "../vendor/preact-htm.js";

const dot = (cx, cy) => html`<circle cx=${cx} cy=${cy} r="1" fill="currentColor" stroke="none" />`;

const ICONS = {
  "chevron-down": html`<path d="M6 9l6 6 6-6" />`,
  "chevron-up": html`<path d="M6 15l6-6 6 6" />`,
  "chevron-right": html`<path d="M9 6l6 6-6 6" />`,
  "chevron-left": html`<path d="M15 6l-6 6 6 6" />`,
  x: html`<path d="M6 6l12 12M18 6L6 18" />`,
  plus: html`<path d="M12 5v14M5 12h14" />`,
  minus: html`<path d="M5 12h14" />`,
  "arrow-up": html`<path d="M12 19V5M6 11l6-6 6 6" />`,
  "arrow-down": html`<path d="M12 5v14M6 13l6 6 6-6" />`,
  undo: html`<path d="M9 14L4 9l5-5" /><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" />`,
  redo: html`<path d="M15 14l5-5-5-5" /><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13" />`,
  more: html`${dot(5, 12)}${dot(12, 12)}${dot(19, 12)}`,
  "reset-view": html`<path d="M3 12a9 9 0 1 0 2.64-6.36L3 8" /><path d="M3 3v5h5" />`,
  dice: html`<rect x="3.5" y="3.5" width="17" height="17" rx="3" />${dot(8.5, 8.5)}${dot(15.5, 8.5)}${dot(12, 12)}${dot(8.5, 15.5)}${dot(15.5, 15.5)}`,
  copy: html`<rect x="8.5" y="8.5" width="12" height="12" rx="2" /><path d="M15.5 8.5V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7.5a2 2 0 0 0 2 2h2.5" />`,
  swap: html`<path d="M8 3L4 7l4 4M4 7h16M16 21l4-4-4-4M20 17H4" />`,
};

export function Icon({ name, size = 16 }) {
  return html`<svg class="icon" width=${size} height=${size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name]}</svg>`;
}
