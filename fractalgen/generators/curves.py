"""Curve traces."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from ..color import log_tone
from ..palette import sample_palette
from .base import generator

_DEFAULT_RATIOS: tuple[tuple[float, float], ...] = (
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (3.0, 5.0),
    (1.0, 3.0),
    (4.0, 5.0),
)


@generator(
    "harmonograph",
    category="curves",
    param_docs={
        "n_curves": "Number of pendulum drawings laid on top of each other.",
        "samples": "Points along each curve. More gives smoother lines.",
        "seed": "Change it for different swings.",
        "palette": "Colors for the curves.",
        "t_max": "How long each pendulum swings.",
        "ratios": "The rhythm of each pendulum pair, which sets the shape of its loops.",
        "samples_per_width": "Keeps lines smooth on large pictures.",
        "pad": "Space around the drawing.",
        "density_blur": "Softens the lines.",
        "tone_lo": "Faintest part of the lines.",
        "tone_hi": "Brightest part of the lines.",
        "tone_gamma": "Lower values brighten the faint lines.",
        "hue_mix": "How much each curve gets its own color.",
        "tone_mix": "How much brightness picks the color.",
        "rgb_floor": "Keeps the faintest lines from going fully dark.",
    },
    ranges={
        "n_curves": (1, 24, 1),
        "samples": (1000, 12_000, 500),
        "seed": (0, 9999, 1),
        "t_max": (8.0, 64.0),
        "samples_per_width": (40, 150, 5),
        "pad": (0.0, 0.2),
        "density_blur": (0.0, 3.0),
        "tone_lo": (1.0, 20.0),
        "tone_hi": (90.0, 99.9),
        "tone_gamma": (0.2, 2.0),
        "hue_mix": (0.0, 1.0),
        "tone_mix": (0.0, 1.0),
        "rgb_floor": (0.0, 0.2),
    },
)
def harmonograph(
    ctx,
    *,
    n_curves: int = 12,
    samples: int = 4000,
    seed: int = 9,
    palette: Any = "twilight",
    t_max: float = 16.0 * np.pi,
    ratios: tuple[tuple[float, float], ...] = _DEFAULT_RATIOS,
    samples_per_width: int = 80,
    pad: float = 0.08,
    density_blur: float = 1.2,
    tone_lo: float = 6.0,
    tone_hi: float = 99.6,
    tone_gamma: float = 0.65,
    hue_mix: float = 0.55,
    tone_mix: float = 0.45,
    rgb_floor: float = 0.05,
):
    """Swinging pendulums drawing looping, fading curves."""
    if n_curves < 1:
        raise ValueError("n_curves must be at least 1")
    if not ratios:
        raise ValueError("ratios must not be empty")
    width, height = ctx.width, ctx.height
    ref_w, _ = ctx.ref_size
    ref_ctx = ctx.reference()
    rng = np.random.default_rng(seed)
    n_samp = max(samples, int(ref_w * samples_per_width))
    n_used = float(n_curves * n_samp)
    n_ref = float(n_curves * max(samples, int(ref_ctx.width * samples_per_width)))
    t = np.linspace(0.0, t_max, n_samp)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    hues: list[np.ndarray] = []
    for k in range(n_curves):
        fx, fy = ratios[k % len(ratios)]
        fx += rng.uniform(-0.012, 0.012)
        fy += rng.uniform(-0.012, 0.012)
        ax, ay = rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0)
        a2, a4 = rng.uniform(0.18, 0.42), rng.uniform(0.18, 0.42)
        f2 = fx + rng.choice([1.0, 2.0]) + rng.uniform(-0.01, 0.01)
        f4 = fy + rng.choice([1.0, 2.0]) + rng.uniform(-0.01, 0.01)
        p1, p2, p3, p4 = rng.uniform(0.0, 2.0 * np.pi, size=4)
        d1, d2 = rng.uniform(0.0012, 0.006), rng.uniform(0.0012, 0.006)
        d3, d4 = rng.uniform(0.0012, 0.006), rng.uniform(0.0012, 0.006)
        x = ax * np.sin(fx * t + p1) * np.exp(-d1 * t) + a2 * np.sin(f2 * t + p2) * np.exp(-d2 * t)
        y = ay * np.sin(fy * t + p3) * np.exp(-d3 * t) + a4 * np.sin(f4 * t + p4) * np.exp(-d4 * t)
        xs.append(x)
        ys.append(y)
        hues.append(np.full(n_samp, k / max(n_curves - 1, 1), dtype=np.float64))

    allx = np.concatenate(xs)
    ally = np.concatenate(ys)
    finite = np.isfinite(allx) & np.isfinite(ally)
    allx, ally = allx[finite], ally[finite]
    xmin, xmax = float(allx.min()), float(allx.max())
    ymin, ymax = float(ally.min()), float(ally.max())
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    xmin -= dx * pad
    xmax += dx * pad
    ymin -= dy * pad
    ymax += dy * pad
    aspect = width / height
    dx, dy = xmax - xmin, ymax - ymin
    if dx / dy > aspect:
        extra = dx / aspect - dy
        ymin -= extra / 2
        ymax += extra / 2
    else:
        extra = dy * aspect - dx
        xmin -= extra / 2
        xmax += extra / 2

    density = np.zeros((height, width), dtype=np.float64)
    hue = np.zeros((height, width), dtype=np.float64)
    for x, y, hv in zip(xs, ys, hues):
        ix = ((x - xmin) / (xmax - xmin) * (width - 1)).astype(np.int32)
        iy = ((ymax - y) / (ymax - ymin) * (height - 1)).astype(np.int32)
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        np.add.at(density, (iy[valid], ix[valid]), 1.0)
        np.add.at(hue, (iy[valid], ix[valid]), hv[valid])

    blur = ctx.ref_px(density_blur)
    density = gaussian_filter(density, sigma=blur)
    hue = gaussian_filter(hue, sigma=blur)
    with np.errstate(invalid="ignore", divide="ignore"):
        hmean = np.divide(hue, density, out=np.zeros_like(hue), where=density > 1e-9)
    density *= ctx.density_norm(n_used, n_ref)
    tone = log_tone(density, lo=tone_lo, hi=tone_hi)
    rgb = sample_palette(palette, np.clip(hmean * hue_mix + tone * tone_mix, 0.0, 1.0))
    rgb *= (rgb_floor + (1.0 - rgb_floor) * np.power(tone, tone_gamma))[..., None]
    return np.clip(rgb, 0.0, 1.0)
