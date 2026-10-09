// Recipe thumbnails rendered lazily by the server.

import { html, useEffect, useRef, useState } from "../vendor/preact-htm.js";

const thumbCache = new Map(); // "recipe|orient" -> object URL

/** A recipe thumbnail that waits for the server to render it (202 = not ready yet). */
export function Thumb({ recipe, orient: o }) {
  const ref = useRef(null);
  const key = `${recipe}|${o}`;
  const [url, setUrl] = useState(thumbCache.get(key) || null);
  useEffect(() => {
    if (url) return undefined;
    let alive = true, started = false;
    const load = async () => {
      started = true;
      let delay = 400;
      for (let i = 0; i < 40 && alive; i++) {
        try {
          const r = await fetch(`/api/thumb?recipe=${encodeURIComponent(recipe)}&orient=${o}`);
          if (r.status === 200) {
            const u = URL.createObjectURL(await r.blob());
            thumbCache.set(key, u);
            if (alive) setUrl(u);
            return;
          }
          if (r.status !== 202) return;
        } catch (_) { return; }
        await new Promise((res) => setTimeout(res, delay));
        delay = Math.min(delay * 1.4, 5000);
      }
    };
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting) && !started) load(); },
      { rootMargin: "200px" });
    if (ref.current) io.observe(ref.current);
    return () => { alive = false; io.disconnect(); };
  }, [key]);
  return html`<div class=${`thumb ${o}`} ref=${ref}>${url ? html`<img src=${url} alt="" />` : html`<span class="spinner"></span>`}</div>`;
}

