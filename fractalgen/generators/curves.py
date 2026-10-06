"""Curve traces and circle packings."""

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
        "n_curves": "number of superposed damped pendulum traces",
        "samples": "minimum time samples per curve",
        "seed": "RNG seed for frequency and phase draws",
        "palette": "palette for hue and tone blend",
        "t_max": "time span of each trace",
        "ratios": "frequency ratio pairs (fx, fy) cycled per curve",
        "samples_per_width": "minimum samples ≈ 4K width × this for smooth traces",
        "pad": "fractional padding around bounding box",
        "density_blur": "Gaussian blur sigma on deposited density (pixels at 4K)",
        "tone_lo": "log-tone percentile low for density",
        "tone_hi": "log-tone percentile high for density",
        "tone_gamma": "gamma on log tone for RGB multiplier",
        "hue_mix": "weight of curve index hue vs tone",
        "tone_mix": "weight of tone in palette coordinate",
        "rgb_floor": "minimum RGB multiplier from tone",
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
    """Multi-pendulum harmonograph traces with log tone mapping."""
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


@generator(
    "apollonian",
    category="curves",
    param_docs={
        "depth": "maximum recursive Soddy depth (queue generations)",
        "palette": "palette from ring log tone",
        "rmin": "minimum circle radius to enqueue",
        "max_circles": "cap on total circles generated",
        "max_r": "drop circles with radius above this after scaling",
        "zoom": "multiply fitted half-extent (lower = tighter crop)",
        "ring_blur": "Gaussian blur sigma on composite ring image (pixels at 4K)",
        "tone_lo": "log-tone percentile low",
        "tone_hi": "log-tone percentile high",
        "tone_gamma": "gamma on log tone before palette",
        "rgb_floor": "minimum RGB multiplier from tone",
    },
    ranges={
        "depth": (4, 12, 1),
        "rmin": (0.005, 0.05),
        "max_circles": (200, 2000, 50),
        "max_r": (1.0, 5.0),
        "zoom": (0.5, 1.2),
        "ring_blur": (0.0, 2.0),
        "tone_lo": (1.0, 20.0),
        "tone_hi": (90.0, 99.9),
        "tone_gamma": (0.2, 2.0),
        "rgb_floor": (0.0, 0.2),
    },
)
def apollonian(
    ctx,
    *,
    depth: int = 8,
    palette: Any = "bronze",
    rmin: float = 0.018,
    max_circles: int = 900,
    max_r: float = 2.6,
    zoom: float = 0.78,
    ring_blur: float = 0.55,
    tone_lo: float = 4.0,
    tone_hi: float = 99.6,
    tone_gamma: float = 0.78,
    rgb_floor: float = 0.08,
):
    """Descartes circle packing rendered as glowing nested rings."""
    width, height = ctx.width, ctx.height
    aspect = ctx.aspect

    def soddy(c1, c2, c3, inner=True):
        k1, k2, k3 = 1.0 / c1[2], 1.0 / c2[2], 1.0 / c3[2]
        sign = 1.0 if inner else -1.0
        disc = k1 * k2 + k1 * k3 + k2 * k3
        k4 = k1 + k2 + k3 + sign * 2.0 * np.sqrt(abs(disc))
        if abs(k4) < 1e-9:
            return None
        r = 1.0 / k4
        z1, z2, z3 = c1[0] + 1j * c1[1], c2[0] + 1j * c2[1], c3[0] + 1j * c3[1]
        zdisc = k1 * k2 * z1 * z2 + k1 * k3 * z1 * z3 + k2 * k3 * z2 * z3
        z4 = (z1 * k1 + z2 * k2 + z3 * k3 + sign * 2.0 * np.sqrt(zdisc)) / k4
        return (float(z4.real), float(z4.imag), float(abs(r)))

    r0 = 1.0
    d = 2.0 * r0
    seeds = [
        (0.0, d / np.sqrt(3.0), r0),
        (-d / 2.0, -d / (2.0 * np.sqrt(3.0)), r0),
        (d / 2.0, -d / (2.0 * np.sqrt(3.0)), r0),
    ]
    outer = soddy(seeds[0], seeds[1], seeds[2], inner=False)
    inner0 = soddy(seeds[0], seeds[1], seeds[2], inner=True)
    circles = list(seeds)
    if outer is not None:
        circles.append(outer)
    if inner0 is not None:
        circles.append(inner0)

    seen = {(round(c[0], 5), round(c[1], 5), round(c[2], 5)) for c in circles}
    queue: list[tuple] = []
    if outer is not None:
        queue.extend([(seeds[0], seeds[1], outer), (seeds[0], seeds[2], outer), (seeds[1], seeds[2], outer)])
    if inner0 is not None:
        queue.extend([(seeds[0], seeds[1], inner0), (seeds[0], seeds[2], inner0), (seeds[1], seeds[2], inner0)])
    queue.append((seeds[0], seeds[1], seeds[2]))

    while queue and len(circles) < max_circles:
        a, b, c = queue.pop(0)
        for inner in (True, False):
            dcirc = soddy(a, b, c, inner=inner)
            if dcirc is None or dcirc[2] < rmin or dcirc[2] > 8.0:
                continue
            key = (round(dcirc[0], 5), round(dcirc[1], 5), round(dcirc[2], 5))
            if key in seen:
                continue
            seen.add(key)
            circles.append(dcirc)
            queue.append((a, b, dcirc))
            queue.append((a, c, dcirc))
            queue.append((b, c, dcirc))

    pack = [c for c in circles if c[2] < max_r]
    if not pack:
        pack = circles
    xs_c = np.array([c[0] for c in pack])
    ys_c = np.array([c[1] for c in pack])
    rs_c = np.array([c[2] for c in pack])
    x0, x1 = float((xs_c - rs_c).min()), float((xs_c + rs_c).max())
    y0, y1 = float((ys_c - rs_c).min()), float((ys_c + rs_c).max())
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.48 * max(x1 - x0, y1 - y0) * zoom
    fitted = [((c[0] - cx) / half, (c[1] - cy) / half, c[2] / half) for c in pack]

    img = np.zeros((height, width), dtype=np.float64)
    xs = np.linspace(-aspect, aspect, width)
    ys = np.linspace(1.0, -1.0, height)
    X, Y = np.meshgrid(xs, ys)
    dx = (xs[1] - xs[0]) if width > 1 else 0.01
    dy = (ys[0] - ys[1]) if height > 1 else 0.01

    for cx, cy, r in fitted:
        if r < 0.006:
            continue
        pad = 0.04 + 0.08 * r
        xlo, xhi = cx - r - pad, cx + r + pad
        ylo, yhi = cy - r - pad, cy + r + pad
        ix0 = max(0, int((xlo + aspect) / (2 * aspect) * (width - 1)))
        ix1 = min(width, int((xhi + aspect) / (2 * aspect) * (width - 1)) + 1)
        iy0 = max(0, int((1.0 - yhi) / 2.0 * (height - 1)))
        iy1 = min(height, int((1.0 - ylo) / 2.0 * (height - 1)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        slx = X[iy0:iy1, ix0:ix1]
        sly = Y[iy0:iy1, ix0:ix1]
        dist = np.sqrt((slx - cx) ** 2 + (sly - cy) ** 2)
        sigma = max(ctx.ref_px(1.2) * min(dx, dy), 0.012 * r)
        ring = np.exp(-((dist - r) ** 2) / (2.0 * sigma**2))
        fill = np.clip(1.0 - dist / (r + 1e-9), 0.0, 1.0) ** 1.35
        wgt = 0.55 + 0.45 * (0.08 / (r + 0.08))
        img[iy0:iy1, ix0:ix1] += wgt * (ring * 1.15 + 0.38 * fill)

    img = gaussian_filter(img, sigma=ctx.ref_px(ring_blur))
    t = log_tone(img, lo=tone_lo, hi=tone_hi)
    rgb = sample_palette(palette, np.power(t, tone_gamma))
    rgb *= (rgb_floor + (1.0 - rgb_floor) * t)[..., None]
    return np.clip(rgb, 0.0, 1.0)
