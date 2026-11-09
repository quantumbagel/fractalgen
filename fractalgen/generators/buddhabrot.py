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
from .. import progress, tone
from ..fieldcache import cached as field_cached
from ..fieldcache import ctx_key as field_ctx_key
from .base import generator

FORMULAS = ("mandelbrot", "burning_ship", "buffalo", "tricorn", "celtic", "multibrot3", "multibrot4")


@generator(
    "buddhabrot",
    category="escape-time",
    uses_view=True,
    default_view={"center": [-0.4, 0.0], "half_width": 2.1, "fit": "cover"},
    param_docs={
        "formula": "Which member of the Mandelbrot family to trace.",
        "channels": "Escape-time windows, one per color channel. Points that escape slowly make the fine inner structure.",
        "colors": "The color of each channel, as RGB from 0 to 1. They add together.",
        "samples": "How many random points to trace per pixel. More gives a smoother image but takes longer.",
        "sample_window": "The region the random points are drawn from.",
        "bailout": "How far a point travels before it counts as escaped.",
        "seed": "Change it for a different pattern of grain.",
        "gamma": "Lower values bring out the faint paths.",
        "hi": "How bright the busiest areas get. Lower it to blow out the cores.",
        "blur": "Softens the grain.",
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
    """The ghostly Buddhabrot: the paths that escaping points trace, built up into a glowing cloud."""
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

    total = int(samples * ctx.pixels)
    fkey = field_ctx_key(
        ctx,
        "buddhabrot",
        formula=formula,
        channels=channels,
        samples=samples,
        sample_window=sample_window,
        bailout=bailout,
        seed=seed,
    )

    def _accumulate() -> np.ndarray:
        acc = np.zeros((len(channels), h * w))
        batch = 1 << 20
        budget = 24_000_000
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
            progress.update((i0 + n) / total)
        return acc

    acc = field_cached(fkey, _accumulate)

    # Scale by zoom_gain to preserve the anchor exposure as orbit hit density changes under zoom.
    norm = ctx.density_norm(total, samples * ctx.reference().pixels) * ctx.zoom_gain(2.0)
    sigma = ctx.ref_px(blur)
    rgb = np.zeros((h, w, 3))
    for ch, tint in enumerate(colors):
        d = np.array(acc[ch], copy=False).reshape(h, w) * norm
        if sigma > 0:
            d = gaussian_filter(d, sigma=sigma)
        pos = d[d > 0]
        top = tone.stat(f"buddhabrot.{ch}", lambda: float(np.percentile(pos, hi)) if pos.size else 1.0)
        t = np.power(tone.levels(np.clip(d / max(top, 1e-12), 0.0, 1.0)), gamma)
        rgb += t[..., None] * np.asarray(tint, dtype=np.float64)
    return np.clip(rgb, 0.0, 1.0)
