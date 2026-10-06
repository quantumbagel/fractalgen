"""Reaction-diffusion and discrete simulations on a reduced grid."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from ..color import histogram_index
from ..palette import sample_palette
from .. import simcache
from .base import generator


@generator(
    "gray_scott",
    category="simulation",
    param_docs={
        "F": "feed rate",
        "k": "kill rate",
        "steps": "Gray–Scott time steps on the simulation grid",
        "seed": "RNG seed for initial v blobs",
        "palette": "palette for v-field histogram coloring",
        "sim_scale": "simulation grid as fraction of the 4K reference frame",
        "n_seeds": "number of random seed blobs",
        "Du": "diffusion coefficient for u",
        "Dv": "diffusion coefficient for v",
        "dt": "simulation time step per kernel iteration",
        "blur_sigma": "Gaussian blur sigma on v before upsample (pixels on sim grid)",
        "hist_gamma": "gamma on histogram index before palette",
        "mask_threshold": "minimum v for histogram masking",
        "substrate_t": "palette index for dark substrate mix",
        "mix_gain": "field strength mixed over substrate",
    },
    ranges={
        "F": (0.01, 0.08),
        "k": (0.04, 0.08),
        "steps": (1000, 20_000, 500),
        "seed": (0, 9999, 1),
        "sim_scale": (0.15, 0.6),
        "n_seeds": (1, 40, 1),
        "Du": (0.05, 0.3),
        "Dv": (0.02, 0.2),
        "dt": (0.5, 2.0),
        "blur_sigma": (0.0, 2.0),
        "hist_gamma": (0.2, 2.0),
        "mask_threshold": (0.0, 0.1),
        "substrate_t": (0.0, 0.2),
        "mix_gain": (0.5, 3.0),
    },
)
def gray_scott(
    ctx,
    *,
    F: float = 0.0545,
    k: float = 0.062,
    steps: int = 9000,
    seed: int = 0,
    palette: Any = "aurora",
    sim_scale: float = 0.28,
    n_seeds: int = 14,
    Du: float = 0.16,
    Dv: float = 0.08,
    dt: float = 1.0,
    blur_sigma: float = 0.6,
    hist_gamma: float = 0.9,
    mask_threshold: float = 0.02,
    substrate_t: float = 0.02,
    mix_gain: float = 1.4,
):
    """Gray-Scott reaction-diffusion, upsampled with histogram palette coloring."""
    from .. import kernels

    width, height = ctx.width, ctx.height
    sw, sh = ctx.sim_grid(sim_scale, min_side=8)

    def _simulate() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        u = np.ones((sh, sw), dtype=np.float64)
        v = np.zeros((sh, sw), dtype=np.float64)

        for _ in range(n_seeds):
            cy = rng.integers(sh // 8, 7 * sh // 8)
            cx = rng.integers(sw // 8, 7 * sw // 8)
            ry = rng.integers(4, max(5, sh // 18))
            rx = rng.integers(4, max(5, sw // 18))
            yy, xx = np.ogrid[:sh, :sw]
            blob = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
            u[blob] = 0.50
            v[blob] = 0.25
        v += 0.02 * rng.random((sh, sw))
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)

        u, v = kernels.gray_scott_kernel(u, v, float(F), float(k), int(steps), Du, Dv, dt)
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        return u, v

    key = ("gray_scott", sw, sh, F, k, steps, seed, n_seeds, Du, Dv, dt)
    u, v = simcache.cached(key, _simulate)
    v = np.array(v, copy=True)

    field = v / (v.max() + 1e-12)
    field = gaussian_filter(field, sigma=blur_sigma)
    field = zoom(field, (height / field.shape[0], width / field.shape[1]), order=3)
    field = field[:height, :width]
    t = histogram_index(field, field > mask_threshold)
    rgb = sample_palette(palette, np.power(t, hist_gamma))
    bg = np.array(sample_palette(palette, np.array([substrate_t]))[0])
    mix = np.clip(field * mix_gain, 0.0, 1.0)[..., None]
    rgb = bg * (1.0 - mix) + rgb * mix
    return np.clip(rgb, 0.0, 1.0)


@generator(
    "sandpile",
    category="simulation",
    param_docs={
        "palette": "palette indexed by pile height",
        "grid_n": "topple grid side length (None = resolution-scaled default)",
        "grains": "initial grain count at center (None = 2× grid²)",
        "max_iter": "maximum topple iterations",
        "n_ref": "reference short side for default grid_n at 4K portrait height",
        "ref_short": "reference short side in pixels for grid_n scaling",
        "tone_floor": "minimum RGB multiplier from height",
        "tone_gamma": "gamma on height before tone floor mix",
    },
    ranges={
        "max_iter": (10_000, 1_000_000, 10_000),
        "n_ref": (200, 800, 10),
        "ref_short": (1080.0, 4320.0),
        "tone_floor": (0.0, 0.4),
        "tone_gamma": (0.2, 2.0),
    },
)
def sandpile(
    ctx,
    *,
    palette: Any = "ember",
    grid_n: int | None = None,
    grains: int | None = None,
    max_iter: int = 400_000,
    n_ref: int = 420,
    ref_short: float = 2160.0,
    tone_floor: float = 0.12,
    tone_gamma: float = 0.7,
):
    """Abelian sandpile relaxation, nearest-neighbor upsampled."""
    from .. import kernels

    width, height = ctx.width, ctx.height
    ref_w, ref_h = ctx.ref_size
    n = grid_n
    if n is None:
        n = max(n_ref, int(round(min(ref_w, ref_h) * n_ref / ref_short)))
    if grains is None:
        grains = n * n * 2

    def _topple() -> np.ndarray:
        g = np.zeros((n, n), dtype=np.int32)
        g[n // 2, n // 2] = grains
        return kernels.sandpile_topple(g, int(max_iter))

    key = ("sandpile", n, grains, max_iter)
    g = np.array(simcache.cached(key, _topple), copy=True)
    field = g.astype(np.float64)
    field = zoom(field, (height / n, width / n), order=0)
    field = field[:height, :width]
    t = field / max(float(field.max()), 1.0)
    rgb = sample_palette(palette, t)
    rgb *= (tone_floor + (1.0 - tone_floor) * np.power(t, tone_gamma))[..., None]
    return np.clip(rgb, 0.0, 1.0)
