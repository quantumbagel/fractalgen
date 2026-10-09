"""Fractal flames (Draves & Reckase): a chaos-game IFS whose affine maps are
followed by nonlinear "variations", colored by a running palette index and
tone-mapped from log hit density.

A recipe either lists its transforms::

    [[recipe.my_flame.params.xforms]]
    affine = [0.6, -0.4, 0.4, 0.6, 0.2, 0.0]   # x' = a x + b y + e, y' = c x + d y + f
    weight = 1.0                               # pick probability (normalized)
    color = 0.0                                # palette coordinate this map pulls toward
    variations = { swirl = 0.7, linear = 0.3 }

or leaves ``xforms`` empty and lets ``seed`` draw ``n_xforms`` random ones.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .. import progress
from ..view import View, fit_window
from .base import generator
from .density import auto_window, palette_lut, tone_map_rgb_acc

VARIATIONS = (
    "linear", "sinusoidal", "spherical", "swirl", "horseshoe", "polar", "handkerchief",
    "heart", "disc", "spiral", "hyperbolic", "diamond", "ex", "julia", "bent", "fisheye",
    "exponential", "cosine", "eyefish", "bubble", "cylinder",
)
_VAR_ID = {name: i for i, name in enumerate(VARIATIONS)}
_WARMUP = 20


def _random_xforms(rng: np.random.Generator, n: int, pool: Sequence[str]) -> list[dict[str, Any]]:
    if len(pool) == 0:
        raise ValueError("pool must not be empty")
    out = []
    for i in range(n):
        n_pick = min(int(rng.integers(1, 3)), len(pool))
        picks = rng.choice(len(pool), size=n_pick, replace=False)
        w = rng.dirichlet(np.ones(len(picks)))
        out.append({
            "affine": rng.uniform(-1.0, 1.0, 6).tolist(),
            "weight": float(rng.uniform(0.3, 1.0)),
            "color": i / max(n - 1, 1),
            "variations": {pool[int(p)]: float(wi) for p, wi in zip(picks, w)},
        })
    return out


def _pack(xforms: Sequence[dict[str, Any]]):
    n = len(xforms)
    coefs = np.zeros((n, 6))
    varw = np.zeros((n, len(VARIATIONS)))
    weight = np.zeros(n)
    color = np.zeros(n)
    for i, xf in enumerate(xforms):
        unknown = set(xf) - {"affine", "weight", "color", "variations"}
        if unknown:
            raise ValueError(f"xform {i}: unknown keys {sorted(unknown)} (affine, weight, color, variations)")
        coefs[i] = [float(v) for v in xf.get("affine", (1, 0, 0, 1, 0, 0))]
        weight[i] = float(xf.get("weight", 1.0))
        color[i] = float(xf.get("color", i / max(n - 1, 1)))
        for name, wv in dict(xf.get("variations") or {"linear": 1.0}).items():
            if name not in _VAR_ID:
                raise ValueError(f"xform {i}: unknown variation {name!r}; known: {', '.join(VARIATIONS)}")
            varw[i, _VAR_ID[name]] = float(wv)
    cum = np.cumsum(weight / weight.sum())
    return coefs, varw, cum, color


def _run(coefs, varw, cum, color, color_speed, symmetry, seed, walkers, steps, on_chunk, chunk=96):
    from .. import kernels

    key = np.uint64((int(seed) * 0x9E3779B1 + 0x632BE5AB) & 0xFFFFFFFFFFFFFFFF)
    x = np.array([kernels._rand01(key, i, 1 << 40) * 2.0 - 1.0 for i in range(walkers)])
    y = np.array([kernels._rand01(key, i, (1 << 40) + 1) * 2.0 - 1.0 for i in range(walkers)])
    c = np.full(walkers, 0.5)
    t = 0
    while t < steps:
        n = min(chunk, steps - t)
        ox = np.empty((n, walkers))
        oy = np.empty((n, walkers))
        oc = np.empty((n, walkers))
        kernels.flame_chunk(x, y, c, t, n, coefs, varw, cum, color, float(color_speed), int(symmetry), key, ox, oy, oc)
        skip = max(0, _WARMUP - t)
        if skip < n:
            on_chunk(ox[skip:].ravel(), oy[skip:].ravel(), oc[skip:].ravel())
        t += n
        progress.update(t / steps)


@generator(
    "flame",
    category="attractor",
    param_docs={
        "xforms": "The transforms that make the shape. Leave empty to have the seed invent some.",
        "seed": "Change it for a different flame.",
        "n_xforms": "How many transforms to invent when none are given.",
        "pool": "The kinds of twists invented transforms may use.",
        "symmetry": "Repeats the flame around the center this many times. 1 means no symmetry.",
        "color_speed": "How quickly colors change across the flame. Lower blends them more smoothly.",
        "palette": "Colors for the flame.",
        "quality": "Samples per pixel. More is smoother but slower.",
        "walkers": "Internal setting. Rarely needs changing.",
        "gamma": "Lower values brighten the faint wisps.",
        "vibrancy": "At 1 the bright areas keep their color. Lower values turn them toward white.",
        "blur": "Softens the grain.",
        "percentiles": "How tightly the frame crops around the flame.",
        "pad": "Space around the flame.",
        "zoom": "Zooms in on the middle. Above 1 crops in.",
        "rotation": "Turns the picture, in degrees.",
        "offset": "Moves the flame within the frame.",
    },
    ranges={
        "seed": (0, 99999, 1),
        "n_xforms": (1, 8, 1),
        "symmetry": (1, 12, 1),
        "color_speed": (0.0, 1.0),
        "quality": (5, 400, 5),
        "walkers": (2000, 200_000, 1000),
        "gamma": (0.15, 1.5),
        "vibrancy": (0.0, 1.0),
        "blur": (0.0, 3.0),
        "pad": (0.0, 0.3),
        "zoom": (0.3, 6.0),
        "rotation": (-180.0, 180.0),
    },
)
def flame(
    ctx,
    *,
    xforms: Sequence[dict[str, Any]] = (),
    seed: int = 7,
    n_xforms: int = 3,
    pool: Sequence[str] = (
        "linear", "sinusoidal", "spherical", "swirl", "horseshoe", "polar", "handkerchief",
        "heart", "disc", "spiral", "hyperbolic", "diamond", "julia", "bubble", "eyefish",
    ),
    symmetry: int = 1,
    color_speed: float = 0.5,
    palette: Any = "ember",
    quality: int = 60,
    walkers: int = 40_000,
    gamma: float = 0.45,
    vibrancy: float = 1.0,
    blur: float = 0.7,
    percentiles: tuple = (1.0, 99.0),
    pad: float = 0.08,
    zoom: float = 1.0,
    rotation: float = 0.0,
    offset: tuple = (0.0, 0.0),
):
    """A fractal flame: soft, silky shapes made by repeating a few twisting transforms."""
    from .. import kernels

    n_walk = int(walkers)
    if n_walk <= 0:
        raise ValueError("walkers must be positive")
    rng = np.random.default_rng(seed)
    xfs = list(xforms) or _random_xforms(rng, int(n_xforms), list(pool))
    coefs, varw, cum, color = _pack(xfs)
    w, h = ctx.width, ctx.height

    # Frame from a short pilot run.
    pilot: dict[str, list] = {"x": [], "y": []}
    with progress.span(0.0, 0.01):
        _run(coefs, varw, cum, color, color_speed, symmetry, seed + 1, 4000, 80,
             lambda x, y, c: (pilot["x"].append(x), pilot["y"].append(y)))
    x0, x1, y0, y1 = auto_window(pilot["x"], pilot["y"], pad=pad, percentiles=tuple(percentiles))
    x0, x1, y0, y1 = fit_window(x0, x1, y0, y1, w, h, fill="cover" if ctx.portrait else "contain")
    span_h = (y1 - y0) / max(zoom, 1e-6)
    center = complex(0.5 * (x0 + x1) + offset[0] * span_h, 0.5 * (y0 + y1) + offset[1] * span_h)
    view = View(center=center, half_width=0.5 * (x1 - x0) / max(zoom, 1e-6), rotation=rotation, fit="fill")
    aff = view.to_pixel(w, h)

    lut = palette_lut(palette)
    acc = np.zeros((h, w, 4))
    total = quality * ctx.pixels
    steps = _WARMUP + max(1, total // n_walk)
    with progress.span(0.01, 1.0):
        _run(coefs, varw, cum, color, color_speed, symmetry, seed, n_walk, steps,
             lambda x, y, c: kernels.splat_rgb(acc, x, y, c, lut, *aff))

    norm = ctx.density_norm(total, quality * ctx.reference().pixels)
    return tone_map_rgb_acc(acc, gamma=gamma, blur=ctx.ref_px(blur), norm=norm, vibrancy=vibrancy)
