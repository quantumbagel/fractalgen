"""Buddhabrot / Nebulabrot: the density of escaping Mandelbrot-family orbits.

Instead of coloring c by how fast it escapes, every escaping orbit z0, z1, ...
is replayed and each visited z is counted. Running channels with different
iteration windows (long orbits, mid, short) and tinting each gives the
"Nebulabrot". The view frames the z-plane the orbits are drawn in.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..kernels import KIND
from .. import tone
from .base import generator

FORMULAS = ("mandelbrot", "burning_ship", "buffalo", "tricorn", "celtic", "multibrot3", "multibrot4")


@generator(
    "buddhabrot",
    category="escape-time",
    uses_view=True,
    default_view={"center": [-0.4, 0.0], "half_width": 2.1, "fit": "cover"},
    param_docs={
        "formula": "orbit formula (Mandelbrot family without a Julia constant)",
        "channels": "[[min_iter, max_iter], ...] escape windows; one image channel each",
        "colors": "[[r, g, b], ...] 0..1 tint per channel, added together",
        "samples": "random c per output pixel; higher = smoother, slower",
        "sample_window": "[re0, re1, im0, im1] region c is drawn from",
        "bailout": "escape radius",
        "seed": "random seed for the c samples; change to vary the grain",
        "gamma": "tone curve per channel; lower = brighter faint orbits",
        "hi": "per-channel density percentile mapped to full brightness; lower = more blown-out cores",
        "blur": "density blur, in pixels at 4K; softens grain",
    },
    ranges={
        "samples": (1, 200, 1),
        "bailout": (2.0, 16.0),
        "seed": (0, 9999, 1),
        "gamma": (0.2, 2.0),
        "hi": (90.0, 100.0),
        "blur": (0.0, 3.0),
    },
    choices={"formula": list(FORMULAS)},
)
def buddhabrot(
    ctx,
    *,
    formula: str = "mandelbrot",
    channels: Sequence[Sequence[int]] = ((200, 5000), (40, 500), (8, 60)),
    colors: Sequence[Sequence[float]] = ((1.0, 0.62, 0.28), (0.25, 0.85, 0.8), (0.42, 0.3, 1.0)),
    samples: int = 24,
    sample_window: tuple = (-2.1, 1.1, -1.5, 1.5),
    bailout: float = 2.0,
    seed: int = 1,
    gamma: float = 0.6,
    hi: float = 99.9,
    blur: float = 0.6,
):
    """Buddhabrot / Nebulabrot: density of escaping orbits, one tinted channel per iteration window."""
    from scipy.ndimage import gaussian_filter

    from .. import kernels

    if formula not in FORMULAS:
        raise ValueError(f"formula must be one of {FORMULAS}")
    if len(colors) != len(channels):
        raise ValueError("colors needs one [r, g, b] per channel")
    kind = KIND[formula]
    w, h = ctx.width, ctx.height
    aff = ctx.view.to_pixel(w, h)
    lo_it = np.array([int(c[0]) for c in channels])
    hi_it = np.array([int(c[1]) for c in channels])
    max_iter = int(hi_it.max())
    x0, x1, y0, y1 = (float(v) for v in sample_window)
    key = np.uint64(seed * 0x9E3779B1 + 0xB5297A4D)

    acc = np.zeros((len(channels), h * w))
    total = int(samples * ctx.pixels)
    batch = 1 << 20
    budget = 24_000_000  # orbit points replayed per splat pass
    for i0 in range(0, total, batch):
        n = min(batch, total - i0)
        cr, ci, esc = kernels.buddhabrot_escape(n, key, i0, x0, x1, y0, y1, max_iter, bailout * bailout, kind)
        mask = np.zeros(n, dtype=np.int64)
        for ch in range(len(channels)):
            mask |= ((esc >= lo_it[ch]) & (esc <= hi_it[ch])).astype(np.int64) << ch
        keep = mask > 0
        cr, ci, steps, mask = cr[keep], ci[keep], esc[keep], mask[keep]
        start = 0
        while start < steps.size:
            csum = np.cumsum(steps[start:])
            end = start + max(1, int(np.searchsorted(csum, budget)))
            st = np.ascontiguousarray(steps[start:end])
            offsets = np.concatenate(([0], np.cumsum(st)[:-1])).astype(np.int64)
            pix = np.empty(int(st.sum()), dtype=np.int64)
            kernels.buddhabrot_orbits(cr[start:end], ci[start:end], st, offsets, kind, *aff, w, h, pix)
            kernels.buddhabrot_splat(acc, pix, st, offsets, mask[start:end])
            start = end

    # zoom_gain: zoomed in, fewer orbit hits land per pixel; keep the anchor's exposure.
    norm = ctx.density_norm(total, samples * ctx.reference().pixels) * ctx.zoom_gain(2.0)
    sigma = ctx.ref_px(blur)
    rgb = np.zeros((h, w, 3))
    for ch, tint in enumerate(colors):
        d = acc[ch].reshape(h, w) * norm
        if sigma > 0:
            d = gaussian_filter(d, sigma=sigma)
        pos = d[d > 0]
        top = tone.stat(f"buddhabrot.{ch}", lambda: float(np.percentile(pos, hi)) if pos.size else 1.0)
        t = np.power(tone.levels(np.clip(d / max(top, 1e-12), 0.0, 1.0)), gamma)
        rgb += t[..., None] * np.asarray(tint, dtype=np.float64)
    return np.clip(rgb, 0.0, 1.0)
