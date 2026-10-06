"""Caustics: sunlight focused by a wavy water surface onto the floor below.

A dense grid of vertical rays is bent by the surface slope and lands at
start + depth * grad(height); where rays bunch up the floor glows. The surface
is a sum of plane waves (random directions, power-law spectrum) plus optional
circular ripples from drops. ``dispersion`` gives red, green and blue slightly
different depths for rainbow-fringed caustics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .. import tone
from ..palette import sample_palette
from .base import generator


@generator(
    "caustics",
    category="fields",
    uses_view=True,
    default_view={"center": [0.0, 0.0], "half_width": 16 / 9, "fit": "cover"},
    param_docs={
        "waves": "number of plane waves making up the surface",
        "drops": "number of circular ripple sources (raindrops); 0 = none",
        "k_lo": "smallest wave number (longest swell), radians per plane unit",
        "k_hi": "largest wave number (finest ripple)",
        "spectrum": "amplitude falloff with wave number; higher = smoother, swell-dominated surface",
        "drop_k": "wave number of drop ripples",
        "focus": "depth in units of the surface's focal length; ~1 starts folding, higher = sharper, busier webs",
        "dispersion": "relative depth spread between red and blue; > 0 adds rainbow fringes",
        "seed": "random seed for wave directions, phases and drops",
        "palette": "light intensity -> color (dark water to bright caustic)",
        "rays": "rays per pixel along each axis (rays^2 per pixel); higher = smoother, slower",
        "exposure": "brightness multiplier on ray density before tone mapping",
        "gamma": "tone curve; lower = brighter dim water",
        "blur": "soften the floor, in pixels at 4K",
    },
    ranges={
        "waves": (1, 120, 1),
        "drops": (0, 24, 1),
        "k_lo": (0.5, 20.0),
        "k_hi": (2.0, 80.0),
        "spectrum": (0.0, 3.0),
        "drop_k": (2.0, 80.0),
        "focus": (0.2, 6.0),
        "dispersion": (0.0, 0.3),
        "seed": (0, 9999, 1),
        "rays": (1, 6, 1),
        "exposure": (0.1, 4.0),
        "gamma": (0.2, 2.0),
        "blur": (0.0, 4.0),
    },
)
def caustics(
    ctx,
    *,
    waves: int = 36,
    drops: int = 0,
    k_lo: float = 3.0,
    k_hi: float = 24.0,
    spectrum: float = 1.6,
    drop_k: float = 26.0,
    focus: float = 2.2,
    dispersion: float = 0.0,
    seed: int = 5,
    palette: Any = "abyss",
    rays: int = 2,
    exposure: float = 1.0,
    gamma: float = 0.8,
    blur: float = 1.2,
):
    """Pool-floor caustics: light rays bent by a wavy surface, density as brightness."""
    from .. import kernels

    rng = np.random.default_rng(seed)
    w, h = ctx.width, ctx.height
    view = ctx.view

    k = np.exp(rng.uniform(np.log(k_lo), np.log(k_hi), waves))
    ang = rng.uniform(0.0, 2 * np.pi, waves)
    amp = k ** (-spectrum) * rng.uniform(0.5, 1.0, waves)
    wave_tab = np.stack([k * np.cos(ang), k * np.sin(ang), amp, rng.uniform(0, 2 * np.pi, waves), np.zeros(waves)], axis=1)
    hw, hh = view.extents(w, h)
    c = view.center
    drop_tab = np.zeros((int(drops), 5))
    if drops:
        drop_tab[:, 0] = c.real + rng.uniform(-hw, hw, drops)
        drop_tab[:, 1] = c.imag + rng.uniform(-hh, hh, drops)
        drop_tab[:, 2] = drop_k
        drop_tab[:, 3] = rng.uniform(0.4, 1.0, drops) * drop_k ** (-spectrum)
        drop_tab[:, 4] = rng.uniform(0, 2 * np.pi, drops)

    # Depth so that `focus` = depth x rms surface curvature (1 ~ where caustics first fold).
    curv = np.sum((amp * k * k) ** 2) / 2.0 + np.sum((drop_tab[:, 3] * drop_tab[:, 2] ** 2) ** 2) / 4.0
    depth = focus / np.sqrt(max(curv, 1e-12))
    depths = np.array([depth * (1.0 + dispersion), depth, depth * (1.0 - dispersion)])

    fwd = np.array(view.to_pixel(w, h))
    inv_m = np.linalg.inv(np.array([[fwd[0], fwd[1]], [fwd[3], fwd[4]]]))
    inv = np.array([inv_m[0, 0], inv_m[0, 1], -(inv_m[0, 0] * fwd[2] + inv_m[0, 1] * fwd[5]),
                    inv_m[1, 0], inv_m[1, 1], -(inv_m[1, 0] * fwd[2] + inv_m[1, 1] * fwd[5])])

    # Rays can land up to ~depth x max slope away; start them that far outside the frame.
    max_slope = float(np.sum(amp * k) + np.sum(drop_tab[:, 3] * drop_tab[:, 2]))
    margin = int(np.ceil(depth * max_slope * np.hypot(fwd[0], fwd[1]) * (1.0 + dispersion))) + 2
    margin = min(margin, max(w, h))
    ss = int(rays)
    dens = np.zeros((3, h, w))
    band = max(1, 4_000_000 // ((w + 2 * margin) * ss * ss))
    for y0 in range(-margin, h + margin, band):
        y1 = min(h + margin, y0 + band)
        ox = np.empty((3, (y1 - y0) * ss, (w + 2 * margin) * ss))
        oy = np.empty_like(ox)
        kernels.caustic_rays(y0, y1, -margin, w + margin, ss, inv, fwd, wave_tab, drop_tab, depths, ox, oy)
        for ch in range(3):
            kernels.splat_bilinear(dens[ch], ox[ch], oy[ch], 1.0 / (ss * ss))

    # dens is now ~1 where light is undisturbed; tone each channel through the palette.
    sigma = ctx.ref_px(blur)
    rgb = np.empty((h, w, 3))
    for ch in range(3):
        d = gaussian_filter(dens[ch], sigma) if sigma > 0 else dens[ch]
        t = tone.levels(1.0 - np.exp(-np.maximum(d, 0.0) * exposure * 0.45))
        rgb[..., ch] = sample_palette(palette, np.power(t, gamma))[..., ch]
    return np.clip(rgb, 0.0, 1.0)
