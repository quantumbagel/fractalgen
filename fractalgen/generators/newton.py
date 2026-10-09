"""Newton fractal basins in the complex plane (z^n - 1 and arbitrary polynomials)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..color import hsv_to_rgb
from .. import progress, tone
from .base import generator

_CINEMATIC_HUES: dict[int, np.ndarray] = {
    3: np.array([0.55, 0.80, 0.10]),
    4: np.array([0.55, 0.80, 0.10, 0.68]),
    5: np.array([0.55, 0.72, 0.80, 0.10, 0.16]),
    6: np.array([0.55, 0.63, 0.78, 0.90, 0.08, 0.16]),
    8: np.array([0.55, 0.62, 0.72, 0.80, 0.90, 0.08, 0.12, 0.18]),
}


def _color_newton_power(
    root_ix: np.ndarray,
    iters: np.ndarray,
    power: int,
    *,
    zr: np.ndarray | None,
    zi: np.ndarray | None,
    style: str,
    hues: np.ndarray | None,
    classic_shade_lo: float,
    classic_shade_hi: float,
    classic_shade_pow: float,
    classic_sat: float,
    classic_hue_iter: float,
    cine_shade_lo: float,
    cine_shade_hi: float,
    cine_shade_pow: float,
    cine_cos_base: float,
    cine_cos_amp: float,
    cine_cos_freq: float,
    cine_rad_lo: float,
    cine_rad_hi: float,
    cine_rad_decay: float,
    cine_sat_lo: float,
    cine_sat_hi: float,
    cine_hue_iter: float,
    cine_hue_base: float,
    cine_hue_span: float,
) -> np.ndarray:
    tmax = tone.stat("newton.tmax", lambda: max(float(iters.max()), 1.0))
    if style == "cinematic":
        if hues is not None:
            if hues.size < 1:
                raise ValueError("hues must be a non-empty list")
            h = hues[root_ix % hues.size]
        elif power in _CINEMATIC_HUES:
            h = _CINEMATIC_HUES[power][root_ix]
        else:
            h = np.mod(cine_hue_base + root_ix.astype(np.float64) / power * cine_hue_span, 1.0)
        t = 1.0 - tone.levels(1.0 - np.clip(iters / tmax, 0.0, 1.0))  # levels act on brightness
        shade = cine_shade_lo + cine_shade_hi * (1.0 - t) ** cine_shade_pow
        shade *= cine_cos_base + cine_cos_amp * np.cos(iters * cine_cos_freq)
        if zr is not None and zi is not None:
            r = np.sqrt(zr * zr + zi * zi)
            shade = shade * (cine_rad_lo + cine_rad_hi * np.exp(-cine_rad_decay * r))
        s = cine_sat_lo + cine_sat_hi * t
        h = np.mod(h + cine_hue_iter * t, 1.0)
        return hsv_to_rgb(h, s, np.clip(shade, 0.0, 1.0))

    h = root_ix.astype(np.float64) / power
    shade = classic_shade_lo + classic_shade_hi * tone.levels(1.0 - np.clip(iters / tmax, 0.0, 1.0)) ** classic_shade_pow
    s = np.full_like(h, classic_sat)
    h = np.mod(h + classic_hue_iter * (iters / tmax), 1.0)
    return hsv_to_rgb(h, s, shade)


def _newton_basins(zr0: np.ndarray, zi0: np.ndarray, power: int, max_iter: int, tol: float):
    from .. import kernels, progress

    return kernels.newton_kernel(
        np.ascontiguousarray(zr0, dtype=np.float64),
        np.ascontiguousarray(zi0, dtype=np.float64),
        int(power),
        int(max_iter),
        float(tol),
        progress.units(zr0.shape[0]),
    )


@generator(
    "newton",
    choices={"style": ["classic", "cinematic"]},
    category="newton",
    uses_view=True,
    param_docs={
        "power": "Number of roots, and so the number of colored regions.",
        "max_iter": "More shows finer detail at the borders, but takes longer.",
        "tol": "How close to a root counts as arrived.",
        "style": "Classic gives each region a flat color. Cinematic adds shading, glow and rings.",
        "hues": "Your own color for each root, from 0 to 1 around the color wheel (cinematic style).",
        "classic_shade_lo": "Lifts the darkest areas.",
        "classic_shade_hi": "Brightens the areas that settle quickly.",
        "classic_shade_pow": "How dark and sharp the borders are.",
        "classic_sat": "Color strength.",
        "classic_hue_iter": "How much the color shifts along the borders.",
        "cine_shade_lo": "Lifts the darkest areas.",
        "cine_shade_hi": "Brightens the areas that settle quickly.",
        "cine_shade_pow": "How dark and sharp the borders are.",
        "cine_cos_base": "Overall brightness of the rings.",
        "cine_cos_amp": "How strong the rings are.",
        "cine_cos_freq": "How tightly the rings are packed.",
        "cine_rad_lo": "Brightness toward the edges of the picture.",
        "cine_rad_hi": "Strength of the glow at the center.",
        "cine_rad_decay": "How quickly the center glow fades.",
        "cine_sat_lo": "Color strength in the calm areas.",
        "cine_sat_hi": "Extra color strength along the borders.",
        "cine_hue_iter": "How much the color shifts along the borders.",
        "cine_hue_base": "Color of the first root when no colors are set.",
        "cine_hue_span": "How far the colors spread around the wheel when none are set.",
    },
    ranges={
        "power": (2, 12, 1),
        "max_iter": (8, 200, 1),
        "tol": (1e-9, 1e-4),
        "classic_shade_lo": (0.0, 0.5),
        "classic_shade_hi": (0.2, 1.0),
        "classic_shade_pow": (0.1, 2.0),
        "classic_sat": (0.2, 1.0),
        "classic_hue_iter": (0.0, 0.2),
        "cine_shade_lo": (0.0, 0.4),
        "cine_shade_hi": (0.3, 1.0),
        "cine_shade_pow": (0.1, 1.5),
        "cine_cos_base": (0.4, 1.0),
        "cine_cos_amp": (0.0, 0.5),
        "cine_cos_freq": (0.2, 3.0),
        "cine_rad_lo": (0.2, 0.8),
        "cine_rad_hi": (0.2, 0.8),
        "cine_rad_decay": (0.02, 0.8),
        "cine_sat_lo": (0.2, 0.8),
        "cine_sat_hi": (0.0, 0.5),
        "cine_hue_iter": (0.0, 0.08),
        "cine_hue_base": (0.0, 1.0),
        "cine_hue_span": (0.1, 0.8),
    },
)
def newton(
    ctx,
    *,
    power: int = 3,
    max_iter: int = 48,
    tol: float = 1e-6,
    style: str = "classic",
    hues: list | None = None,
    classic_shade_lo: float = 0.28,
    classic_shade_hi: float = 0.72,
    classic_shade_pow: float = 0.65,
    classic_sat: float = 0.72,
    classic_hue_iter: float = 0.04,
    cine_shade_lo: float = 0.10,
    cine_shade_hi: float = 0.90,
    cine_shade_pow: float = 0.38,
    cine_cos_base: float = 0.78,
    cine_cos_amp: float = 0.22,
    cine_cos_freq: float = 1.05,
    cine_rad_lo: float = 0.50,
    cine_rad_hi: float = 0.50,
    cine_rad_decay: float = 0.16,
    cine_sat_lo: float = 0.44,
    cine_sat_hi: float = 0.20,
    cine_hue_iter: float = 0.018,
    cine_hue_base: float = 0.55,
    cine_hue_span: float = 0.38,
):
    """Newton's method finding the roots of z^n = 1. Each point is colored by the root it lands on."""
    power = int(power)
    if power < 2:
        raise ValueError(f"power must be >= 2, got {power}")
    if hues is not None and len(hues) < 1:
        raise ValueError("hues must be a non-empty list")
    zr, zi = ctx.plane()

    def run(a, b):
        return _newton_basins(a, b, power, max_iter, tol)

    root_ix, iters = ctx.strips(run, zr, zi)
    hue_arr = np.array(hues, dtype=np.float64) if hues is not None else None
    return _color_newton_power(
        root_ix,
        iters,
        power,
        zr=zr,
        zi=zi,
        style=style,
        hues=hue_arr,
        classic_shade_lo=classic_shade_lo,
        classic_shade_hi=classic_shade_hi,
        classic_shade_pow=classic_shade_pow,
        classic_sat=classic_sat,
        classic_hue_iter=classic_hue_iter,
        cine_shade_lo=cine_shade_lo,
        cine_shade_hi=cine_shade_hi,
        cine_shade_pow=cine_shade_pow,
        cine_cos_base=cine_cos_base,
        cine_cos_amp=cine_cos_amp,
        cine_cos_freq=cine_cos_freq,
        cine_rad_lo=cine_rad_lo,
        cine_rad_hi=cine_rad_hi,
        cine_rad_decay=cine_rad_decay,
        cine_sat_lo=cine_sat_lo,
        cine_sat_hi=cine_sat_hi,
        cine_hue_iter=cine_hue_iter,
        cine_hue_base=cine_hue_base,
        cine_hue_span=cine_hue_span,
    )


def _poly_newton(zr0: np.ndarray, zi0: np.ndarray, coeffs: np.ndarray, max_iter: int, tol: float):
    """Newton for sum coeffs[k] * z**(n-k); coeffs[0] is leading term."""
    from .. import kernels

    coeffs = np.asarray(coeffs, dtype=np.complex128)
    dcoeffs = np.asarray(np.polyder(coeffs), dtype=np.complex128)
    zr, zi, done_n = kernels.poly_newton_kernel(
        np.ascontiguousarray(zr0, dtype=np.float64), np.ascontiguousarray(zi0, dtype=np.float64),
        np.ascontiguousarray(coeffs.real), np.ascontiguousarray(coeffs.imag),
        np.ascontiguousarray(dcoeffs.real), np.ascontiguousarray(dcoeffs.imag),
        int(max_iter), float(tol), progress.units(zr0.shape[0]))
    return zr + 1j * zi, done_n


@generator(
    "newton_poly",
    category="newton",
    uses_view=True,
    param_docs={
        "coeffs": "The polynomial's coefficients, highest power first. [1, 0, -2, 2] means z³ - 2z + 2.",
        "max_iter": "More shows finer detail at the borders, but takes longer.",
        "tol": "How close to a root counts as arrived.",
        "hues": "The colors to cycle through, from 0 to 1 around the color wheel.",
        "basin_scale": "How many color sectors go around the center.",
        "cine_shade_lo": "Lifts the darkest areas.",
        "cine_shade_hi": "Brightens the areas that settle quickly.",
        "cine_shade_pow": "How dark and sharp the borders are.",
        "cine_cos_base": "Overall brightness of the rings.",
        "cine_cos_amp": "How strong the rings are.",
        "cine_cos_freq": "How tightly the rings are packed.",
        "cine_rad_lo": "Brightness toward the edges of the picture.",
        "cine_rad_hi": "Strength of the glow at the center.",
        "cine_rad_decay": "How quickly the center glow fades.",
        "cine_sat_lo": "Color strength in the calm areas.",
        "cine_sat_hi": "Extra color strength along the borders.",
    },
    ranges={
        "max_iter": (10, 200, 1),
        "tol": (1e-9, 1e-4),
        "basin_scale": (1.0, 8.0),
        "cine_shade_lo": (0.0, 0.4),
        "cine_shade_hi": (0.3, 1.0),
        "cine_shade_pow": (0.1, 1.5),
        "cine_cos_base": (0.4, 1.0),
        "cine_cos_amp": (0.0, 0.5),
        "cine_cos_freq": (0.2, 3.0),
        "cine_rad_lo": (0.2, 0.8),
        "cine_rad_hi": (0.2, 0.8),
        "cine_rad_decay": (0.02, 0.8),
        "cine_sat_lo": (0.2, 0.8),
        "cine_sat_hi": (0.0, 0.5),
    },
)
def newton_poly(
    ctx,
    *,
    coeffs: Any = (1.0, 0.0, -2.0, 2.0),
    max_iter: int = 50,
    tol: float = 1e-6,
    hues: list | None = None,
    basin_scale: float = 3.0,
    cine_shade_lo: float = 0.10,
    cine_shade_hi: float = 0.90,
    cine_shade_pow: float = 0.38,
    cine_cos_base: float = 0.78,
    cine_cos_amp: float = 0.22,
    cine_cos_freq: float = 1.15,
    cine_rad_lo: float = 0.52,
    cine_rad_hi: float = 0.48,
    cine_rad_decay: float = 0.16,
    cine_sat_lo: float = 0.42,
    cine_sat_hi: float = 0.22,
):
    """Newton's method on a polynomial of your choice. Each point is colored by the root it finds."""
    if hues is not None and len(hues) < 1:
        raise ValueError("hues must be a non-empty list")
    zr, zi = ctx.plane()
    coeff_arr = np.array([complex(c) for c in coeffs], dtype=np.complex128)
    if hues is None:
        hue_table = np.array([0.55, 0.80, 0.10], dtype=np.float64)
    else:
        hue_table = np.array(hues, dtype=np.float64)
    n_roots = len(hue_table)

    def run(a, b):
        return _poly_newton(a, b, coeff_arr, max_iter, tol)

    zf, iters = ctx.strips(run, zr, zi)
    hue = (np.angle(zf) + np.pi) / (2.0 * np.pi)
    basin = np.mod(np.round(hue * basin_scale), n_roots)
    h = hue_table[basin.astype(np.int32)]
    tmax = tone.stat("newton.tmax", lambda: max(float(iters.max()), 1.0))
    t = 1.0 - tone.levels(1.0 - np.clip(iters / tmax, 0.0, 1.0))  # levels act on brightness
    val = cine_shade_lo + cine_shade_hi * (1.0 - t) ** cine_shade_pow
    val *= cine_cos_base + cine_cos_amp * np.cos(iters * cine_cos_freq)
    r = np.sqrt(zr * zr + zi * zi)
    val *= cine_rad_lo + cine_rad_hi * np.exp(-cine_rad_decay * r)
    sat = cine_sat_lo + cine_sat_hi * t
    return hsv_to_rgb(h, sat, np.clip(val, 0.0, 1.0))
