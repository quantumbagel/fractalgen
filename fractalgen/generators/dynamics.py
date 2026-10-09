"""Dynamical attractors and basin coloring."""

from __future__ import annotations

import numpy as np

from .. import progress
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
        "n_magnets": "Number of magnets, and so the number of colors.",
        "steps": "How long each pendulum swings. More gives cleaner regions but is slower.",
        "dt": "Simulation time step.",
        "seed": "Not used. Kept so older recipes still load.",
        "framing": "Wide shows more of the board, while close zooms into the tangled edges.",
        "mag_r": "How far the magnets sit from the center. Leave empty to use the framing's value.",
        "strength": "How hard the magnets pull.",
        "friction": "How quickly the pendulum slows down. More gives calmer regions.",
        "g": "How hard the pendulum is pulled back to the middle.",
        "span": "How much of the board is shown.",
        "softening": "Smooths the pull close to each magnet.",
        "settled_dist": "How close to a magnet the pendulum must end to count as caught.",
        "settled_speed": "How slow the pendulum must be moving to count as caught.",
        "chaos_v_scale": "Brightness of points that never settle.",
        "chaos_s_scale": "Color strength of points that never settle.",
        "falloff_k": "How much the corners darken.",
        "falloff_r0_scale": "Where the corner darkening begins.",
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
    """A pendulum swinging over magnets. Each point is colored by the magnet it ends up on."""
    if n_magnets < 1:
        raise ValueError("n_magnets must be at least 1")
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
    from .. import kernels

    X, Y, vx, vy = kernels.pendulum_kernel(
        np.ascontiguousarray(X), np.ascontiguousarray(Y), mx, my, int(steps), float(dt), float(g), float(friction),
        float(strength), float(softening), progress.units(height))

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
