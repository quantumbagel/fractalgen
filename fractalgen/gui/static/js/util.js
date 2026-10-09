// Small helpers shared by every module.

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
export const deepClone = (o) => (o === undefined ? undefined : JSON.parse(JSON.stringify(o)));
export const deepEq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
export const uid = () => Math.random().toString(36).slice(2, 10);

export function fmtNum(v, digits = 4) {
  if (typeof v !== "number" || !Number.isFinite(v)) return String(v);
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return String(+v.toPrecision(digits));
}

export function finiteNum(x) {
  return typeof x === "number" && Number.isFinite(x) ? x : null;
}

export function b64UrlEncode(obj) {
  const s = JSON.stringify(obj);
  const b64 = btoa(unescape(encodeURIComponent(s)));
  return b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function b64UrlDecode(s) {
  let b64 = s.replace(/-/g, "+").replace(/_/g, "/");
  while (b64.length % 4) b64 += "=";
  return JSON.parse(decodeURIComponent(escape(atob(b64))));
}

export function getAtPath(root, path) {
  return path.split(".").reduce((o, k) => (o != null ? o[k] : undefined), root);
}

export function setAtPath(root, path, value) {
  const parts = path.split(".");
  let o = root;
  for (let i = 0; i < parts.length - 1; i++) {
    if (o[parts[i]] == null) o[parts[i]] = {};
    o = o[parts[i]];
  }
  o[parts[parts.length - 1]] = value;
}

export class ApiError extends Error {
  constructor(message, trace, status) {
    super(message);
    this.trace = trace;
    this.status = status;
  }
}

async function failure(r) {
  const ct = r.headers.get("Content-Type") || "";
  if (ct.includes("json")) {
    const j = await r.json();
    return new ApiError(j.error || r.statusText, j.trace, r.status);
  }
  return new ApiError((await r.text()) || r.statusText, null, r.status);
}

export async function apiJson(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opts);
  if (!r.ok) throw await failure(r);
  return r.json();
}

export async function apiRender(state) {
  const r = await fetch("/api/render", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state),
  });
  if (r.status === 204) return { stale: true };
  if (!r.ok) throw await failure(r);
  let boxes = null;
  try { boxes = JSON.parse(r.headers.get("X-Layer-Boxes") || "null"); } catch (_) { /* ignore */ }
  return {
    blob: await r.blob(),
    ms: Number(r.headers.get("X-Render-Ms")),
    extent: r.headers.get("X-View-Extent"),
    boxes,
    stale: false,
  };
}

export async function apiUpload(file) {
  const r = await fetch("/api/upload", {
    method: "POST", headers: { "X-Filename": encodeURIComponent(file.name) }, body: file,
  });
  if (!r.ok) throw await failure(r);
  return r.json();
}

export function paletteGradient(stops, positions = null) {
  if (!stops || !stops.length) return "transparent";
  const cols = stops.map((h, i) => {
    const c = String(h).startsWith("#") ? h : `#${h}`;
    return positions && positions.length === stops.length ? `${c} ${(positions[i] * 100).toFixed(1)}%` : c;
  }).join(", ");
  return `linear-gradient(90deg, ${cols})`;
}

export function hexOf(v) {
  if (typeof v === "string") {
    const m = /^#?([0-9a-f]{6}|[0-9a-f]{3})$/i.exec(v.trim());
    if (!m) return "#000000";
    const h = m[1].length === 3 ? m[1].split("").map((c) => c + c).join("") : m[1];
    return `#${h.toLowerCase()}`;
  }
  if (Array.isArray(v)) {
    return "#" + v.slice(0, 3).map((c) => clamp(Math.round(c * 255), 0, 255).toString(16).padStart(2, "0")).join("");
  }
  return "#ffffff";
}

export function formFocused() {
  const el = document.activeElement;
  if (!el) return false;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName) || el.isContentEditable;
}

export function downloadName(path) {
  return String(path || "").split(/[\\/]/).pop();
}
