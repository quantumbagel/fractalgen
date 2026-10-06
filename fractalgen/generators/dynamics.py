"""Dynamical attractors and basin coloring."""

from __future__ import annotations

import numpy as np

from ..color import hsv_to_rgb
from .base import generator

_FRAMING: dict[str, dict[str, float]] = {
    "wide": {"mag_r": 0.62, "strength": 1.15, "friction": 0.17, "g": 0.35, "span": 0.95},
    "close": {"mag_r": 0.48, "strength": 1.35, "friction": 0.20, "g": 0.42, "span": 0.70},
}


@generator(
    "magnetic_pendulum",
    choices={"framing": ["wide", "close"]},
    category="dynamics",
    param_docs={
        "n_magnets": "magnets on a ring; sets how many colored basins (n-fold symmetry)",
        "steps": "physics steps from each pixel; higher = more settled basins, slower",
        "dt": "time step of the pendulum; higher = faster motion, less accurate",
        "seed": "unused (kept for recipe compatibility); safe to ignore",
        "framing": "wide = more of the plane, close = zoomed into the chaotic edges",
        "mag_r": "magnet ring radius in plane units; None uses the framing preset",
        "strength": "pull of each magnet; higher = larger basins, snappier capture",
        "friction": "velocity damping; higher = slower, more settled, less chaotic swirl",
        "g": "spring back toward the origin; higher = magnets compete with center pull",
        "span": "half-width of the starting-position grid; higher = more zoomed out",
        "softening": "added to r² in the force law; higher = gentler near magnets, smoother basins",
        "settled_dist": "max distance to a magnet to count as caught; higher = more settled pixels",
        "settled_speed": "max speed to count as caught; higher = more pixels treated as settled",
        "chaos_v_scale": "brightness mix 0..1 for uncaught pixels; lower = darker chaotic regions",
        "chaos_s_scale": "saturation mix 0..1 for uncaught pixels; lower = greyer chaos",
        "falloff_k": "edge vignette strength vs start radius; higher = darker corners",
        "falloff_r0_scale": "vignette starts at this × magnet radius; higher = more of the image stays bright",
    },
    ranges={
        "n_magnets": (2, 12, 1),
        "steps": (20, 200, 5),
        "dt": (0.01, 0.2),
        "seed": (0, 9999, 1),
        "softening": (0.01, 0.3),
        "settled_dist": (0.1, 1.0),
        "settled_speed": (0.5, 4.0),
        "chaos_v_scale": (0.05, 0.5),
        "chaos_s_scale": (0.1, 0.6),
        "falloff_k": (0.2, 3.0),
        "falloff_r0_scale": (0.3, 1.2),
    },
)
def magnetic_pendulum(
    ctx,
    *,
    n_magnets: int = 5,
    steps: int = 90,
    dt: float = 0.08,
    seed: int = 2,
    framing: str = "wide",
    mag_r: float | None = None,
    strength: float | None = None,
    friction: float | None = None,
    g: float | None = None,
    span: float | None = None,
    softening: float = 0.08,
    settled_dist: float = 0.42,
    settled_speed: float = 1.8,
    chaos_v_scale: float = 0.16,
    chaos_s_scale: float = 0.28,
    falloff_k: float = 1.15,
    falloff_r0_scale: float = 0.72,
):
    """Which magnet a damped pendulum settles on, colored as basins from every start point."""
    preset = _FRAMING.get(framing, _FRAMING["wide"])
    mag_r = preset["mag_r"] if mag_r is None else mag_r
    strength = preset["strength"] if strength is None else strength
    friction = preset["friction"] if friction is None else friction
    g = preset["g"] if g is None else g
    span = preset["span"] if span is None else span

    width, height = ctx.width, ctx.height
    aspect = width / float(height)
    mx = mag_r * np.cos(2 * np.pi * np.arange(n_magnets) / n_magnets)
    my = mag_r * np.sin(2 * np.pi * np.arange(n_magnets) / n_magnets)

    if aspect < 1.0:
        xs = np.linspace(-span, span, width)
        ys = np.linspace(span / aspect, -span / aspect, height)
    else:
        xs = np.linspace(-aspect * span, aspect * span, width)
        ys = np.linspace(span, -span, height)
    X, Y = np.meshgrid(xs, ys)
    X0, Y0 = X.copy(), Y.copy()
    vx = np.zeros_like(X)
    vy = np.zeros_like(Y)

    for _ in range(steps):
        ax = -g * X - friction * vx
        ay = -g * Y - friction * vy
        for i in range(n_magnets):
            dx = mx[i] - X
            dy = my[i] - Y
            r2 = dx * dx + dy * dy + softening
            inv = strength / (r2 * np.sqrt(r2))
            ax += dx * inv
            ay += dy * inv
        vx += ax * dt
        vy += ay * dt
        X += vx * dt
        Y += vy * dt

    dist = np.stack([(X - mx[i]) ** 2 + (Y - my[i]) ** 2 for i in range(n_magnets)], axis=0)
    winner = np.argmin(dist, axis=0)
    dmin = np.sqrt(np.min(dist, axis=0))
    h = winner.astype(np.float64) / n_magnets
    speed = np.sqrt(vx * vx + vy * vy)
    r0 = np.sqrt(X0 * X0 + Y0 * Y0)
    settled = (dmin < settled_dist) & (speed < settled_speed)
    falloff = np.exp(-falloff_k * np.maximum(r0 - mag_r * falloff_r0_scale, 0.0))
    if framing == "close":
        v = 0.20 + 0.80 * np.exp(-dmin * 2.6)
        v *= 1.0 / (1.0 + 0.70 * speed)
        s = 0.40 + 0.30 * np.exp(-dmin * 1.8)
    else:
        v = 0.18 + 0.82 * np.exp(-dmin * 3.0)
        v *= 1.0 / (1.0 + 0.85 * speed)
        s = 0.58 + 0.18 * np.exp(-dmin * 1.6)
    v = v * (0.12 + 0.88 * falloff)
    v = np.where(settled, v, v * chaos_v_scale)
    s = np.where(settled, s, s * chaos_s_scale)
    return hsv_to_rgb(h, s, np.clip(v, 0, 1))
