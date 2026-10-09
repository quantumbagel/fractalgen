"""Phase portraits of area-preserving maps: KAM islands in a chaotic sea.

Many orbits of the map are traced, each in its own palette color. Regular
orbits stay on invariant curves (closed loops, island chains); chaotic ones
wander and fill the "sea", which is drawn separately and dimmer so the
islands stand out. The coverage of a coarse grid tells the two apart.

maps:
  standard -- Chirikov standard map  p' = p + K sin(theta), theta' = theta + p'
              on the torus [0, 2 pi)^2; ``k`` is K. Tiles periodically in the view.
  henon    -- Henon's quadratic area-preserving map
              x' = x cos a - (y - x^2) sin a,  y' = x sin a + (y - x^2) cos a; ``k`` is a (radians).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .. import progress, tone
from .base import generator
from .density import palette_lut

_MAPS = {
    # kind id, coverage box (x0, x1, y0, y1), periodic
    "standard": (0, (0.0, 2 * math.pi, 0.0, 2 * math.pi), True),
    "henon": (1, (-1.2, 1.2, -1.2, 1.2), False),
}


def _tile_range(aff, w, h, period):
    """Tile index ranges (i0, i1, j0, j1) whose copies of [0, period)^2 can hit the frame."""
    ax, bx, cx, ay, by, cy = aff
    det = ax * by - bx * ay
    corners = []
    for px, py in ((0, 0), (w, 0), (0, h), (w, h)):
        dx, dy = px - cx, py - cy
        corners.append(((by * dx - bx * dy) / det, (-ay * dx + ax * dy) / det))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (
        math.floor(min(xs) / period) - 1, math.floor(max(xs) / period),
        math.floor(min(ys) / period) - 1, math.floor(max(ys) / period),
    )


@generator(
    "symplectic",
    category="dynamics",
    uses_view=True,
    default_view={"center": [math.pi, math.pi], "half_width": 2 * math.pi * 16 / 9 / 2, "fit": "cover"},
    param_docs={
        "map": "standard (Chirikov, periodic) or henon (quadratic, island chains around the origin)",
        "k": "map strength: standard K (~0.6 calm .. 1.5 mostly chaos), henon rotation angle a in radians",
        "orbits": "number of traced orbits (initial conditions)",
        "steps": "iterations per orbit at 4K (scaled with resolution so curves stay solid)",
        "seed": "random seed for initial conditions and orbit colors",
        "palette": "orbit colors",
        "color_by": "random = each orbit its own color; start = by starting position (smooth bands)",
        "chaos_threshold": "grid coverage above which an orbit counts as chaotic sea",
        "sea_color": "[r, g, b] color of the chaotic sea",
        "sea_strength": "brightness of the chaotic sea; 0 hides it",
        "gamma": "tone curve on orbit density; lower = brighter faint curves",
        "blur": "density blur, in pixels at 4K",
    },
    ranges={
        "k": (0.0, 3.0),
        "orbits": (100, 6000, 50),
        "steps": (1000, 60000, 500),
        "seed": (0, 9999, 1),
        "chaos_threshold": (0.005, 0.5),
        "sea_strength": (0.0, 1.5),
        "gamma": (0.2, 1.5),
        "blur": (0.0, 3.0),
    },
    choices={"map": list(_MAPS), "color_by": ["random", "start"]},
)
def symplectic(
    ctx,
    *,
    map: str = "standard",
    k: float = 0.971635,
    orbits: int = 1400,
    steps: int = 20000,
    seed: int = 3,
    palette: Any = "rydberg",
    color_by: str = "random",
    chaos_threshold: float = 0.06,
    sea_color: tuple = (0.32, 0.36, 0.55),
    sea_strength: float = 0.35,
    gamma: float = 0.55,
    blur: float = 0.6,
):
    """Area-preserving map phase portrait: colored KAM islands in a dim chaotic sea."""
    from .. import kernels

    if map not in _MAPS:
        raise ValueError(f"map must be one of {sorted(_MAPS)}")
    if color_by not in ("random", "start"):
        raise ValueError("color_by must be 'random' or 'start'")
    kind, box, periodic = _MAPS[map]
    w, h = ctx.width, ctx.height
    aff = ctx.view.to_pixel(w, h)
    period = 2 * math.pi
    tiles = _tile_range(aff, w, h, period) if periodic else (0, 0, 0, 0)

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(box[0], box[1], orbits)
    y0 = rng.uniform(box[2], box[3], orbits)
    if color_by == "random":
        cpos = rng.random(orbits)
    else:
        r = np.hypot(x0 - 0.5 * (box[0] + box[1]), y0 - 0.5 * (box[2] + box[3]))
        cpos = r / max(float(r.max()), 1e-12)
    lut = palette_lut(palette)
    rgb_orbit = lut[np.clip((cpos * (len(lut) - 1)).astype(int), 0, len(lut) - 1)]

    n_steps = max(1000, int(steps * ctx.res_scale))
    lines = np.zeros((h, w, 4))
    sea = np.zeros((h, w, 4))
    cov_box = np.array(box, dtype=np.float64)
    chunk = max(1, 20_000_000 // n_steps)
    for s in range(0, orbits, chunk):
        progress.update(s / orbits)
        e = min(orbits, s + chunk)
        ox = np.empty((e - s, n_steps))
        oy = np.empty((e - s, n_steps))
        cov = kernels.symplectic_orbits(x0[s:e], y0[s:e], n_steps, kind, float(k), cov_box, 64, ox, oy)
        chaotic = cov > chaos_threshold
        for sel, buf in ((~chaotic, lines), (chaotic, sea)):
            if np.any(sel):
                kernels.splat_periodic(
                    buf, np.ascontiguousarray(ox[sel]), np.ascontiguousarray(oy[sel]),
                    np.ascontiguousarray(rgb_orbit[s:e][sel]), period, *tiles, *aff,
                )

    sigma = ctx.ref_px(blur)
    out = np.zeros((h, w, 3))
    # Hits along a curve scale with steps / curve length; res_scale keeps that fixed.
    # zoom_gain keeps the anchor view's exposure: the sea fills area, KAM curves are lines.
    for name, buf, weight, power in (("sea", sea, sea_strength, 2.0), ("lines", lines, 1.0, 1.0)):
        hits = buf[..., 0]
        col = buf[..., 1:]
        if sigma > 0:
            hits = gaussian_filter(hits, sigma)
            col = np.stack([gaussian_filter(col[..., i], sigma) for i in range(3)], axis=-1)
        pos = hits[hits > 0]
        if not pos.size:
            continue
        gain = ctx.zoom_gain(power)
        top = tone.stat(f"symplectic.{name}", lambda: float(np.log1p(np.percentile(pos, 99.5))))
        shade = np.power(tone.levels(np.clip(np.log1p(hits * gain) / top, 0.0, 1.0)), gamma)
        if buf is sea:
            layer = shade[..., None] * np.asarray(sea_color, dtype=np.float64) * weight
        else:
            layer = col / np.maximum(hits, 1e-12)[..., None] * shade[..., None]
            out *= 1.0 - shade[..., None]  # curves sit on top of the sea
        out += layer
    return np.clip(out, 0.0, 1.0)
