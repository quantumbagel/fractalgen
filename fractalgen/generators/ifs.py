"""Iterated function systems (chaos game)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..view import fit_window
from .base import generator
from .density import deposit, tone_map_density

# [a, b, c, d, e, f, p]: x' = a x + b y + e, y' = c x + d y + f, picked with probability p.
BARNSLEY_FERN = (
    (0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01),
    (0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85),
    (0.20, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07),
    (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07),
)


@generator(
    "ifs",
    category="attractor",
    param_docs={
        "maps": "[[a,b,c,d,e,f,p], ...] affine maps + probability; defines the exact shape",
        "palette": "color gradient for the density cloud",
        "n": "point count at 4K (scales up at larger sizes); higher = denser, slower",
        "seed": "random seed for which map is picked each step; change to vary the grain",
        "batch": "points advanced per inner loop; higher = more memory, usually faster; rarely change",
        "xmin": "left edge of the data window, in plane units",
        "xmax": "right edge of the data window, in plane units",
        "ymin": "bottom edge of the data window, in plane units",
        "ymax": "top edge of the data window, in plane units",
        "window_fill": "aspect fit: contain (whole shape) or cover (fill the frame, crop)",
        "blur": "blur of the density cloud, in output pixels; higher = softer",
        "gamma": "tone curve on density; lower = brighter, more filled-in midtones",
        "warmup_steps": "chaos-game steps skipped before drawing; higher = fewer stray dots",
    },
    ranges={
        "n": (500_000, 12_000_000, 100_000),
        "seed": (0, 9999, 1),
        "batch": (5_000, 100_000, 5_000),
        "xmin": (-5.0, 0.0),
        "xmax": (0.0, 5.0),
        "ymin": (-2.0, 2.0),
        "ymax": (5.0, 15.0),
        "blur": (0.0, 2.0),
        "gamma": (0.2, 2.0),
        "warmup_steps": (0, 50, 1),
    },
)
def ifs(
    ctx,
    *,
    maps: Sequence[Sequence[float]] = BARNSLEY_FERN,
    palette: Any = "forest",
    n: int = 6_000_000,
    seed: int = 1,
    batch: int = 40_000,
    xmin: float = -2.6,
    xmax: float = 2.9,
    ymin: float = -0.3,
    ymax: float = 10.6,
    window_fill: str = "contain",
    blur: float = 0.45,
    gamma: float = 0.62,
    warmup_steps: int = 8,
):
    """Chaos-game iterated function system from affine maps (Barnsley fern, etc.)."""
    rng = np.random.default_rng(seed)
    w, h = ctx.width, ctx.height
    def count(c) -> int:
        return int(n * max(1.0, c.pixels / (3840.0 * 2160.0)))

    total = count(ctx)
    p = batch
    xs = np.zeros(p, dtype=np.float64)
    ys = np.zeros(p, dtype=np.float64)
    steps = max(1, total // p)
    wx0, wx1, wy0, wy1 = fit_window(xmin, xmax, ymin, ymax, w, h, fill=window_fill)
    density = np.zeros((h, w), dtype=np.float64)
    hue_acc = np.zeros((h, w), dtype=np.float64)
    denom = max(steps - 1, 1)

    # Cumulative probability thresholds (maps sorted by old barnsley order in recipe).
    probs = [float(m[6]) for m in maps]
    bounds = np.cumsum(probs)

    for s in range(steps):
        u = rng.random(p)
        nx, ny = np.empty_like(xs), np.empty_like(ys)
        for i, m in enumerate(maps):
            a, b, c, d, e, f, _prob = m
            if i == 0:
                mask = u < bounds[i]
            elif i == len(maps) - 1:
                mask = u >= bounds[i - 1]
            else:
                mask = (u >= bounds[i - 1]) & (u < bounds[i])
            nx[mask] = a * xs[mask] + b * ys[mask] + e
            ny[mask] = c * xs[mask] + d * ys[mask] + f
        xs, ys = nx, ny
        if s > warmup_steps:
            deposit(density, hue_acc, xs, ys, wx0, wx1, wy0, wy1, hue_val=s / denom)

    norm = ctx.density_norm(total, count(ctx.reference()))
    return tone_map_density(density, hue_acc, palette, blur=blur, gamma=gamma, norm=norm)
