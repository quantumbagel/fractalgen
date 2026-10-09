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
        "power": "number of roots (pie-slice basins); higher = more symmetry sectors",
        "max_iter": "root-finding steps per pixel; higher = finer fractal borders, slower",
        "tol": "stop when |f(z)| is this small; lower = tighter, may need more iterations",
        "style": "classic = flat hues per basin; cinematic = shaded glow and rings",
        "hues": "optional list of 0..1 hues, one per root (cinematic only; else a preset)",
        "classic_shade_lo": "classic minimum brightness (HSV V); higher = lifted slow pixels",
        "classic_shade_hi": "classic extra brightness for fast-converging pixels",
        "classic_shade_pow": "classic shade falloff; higher = darker, sharper basin borders",
        "classic_sat": "classic color intensity 0..1; higher = richer, lower = greyer",
        "classic_hue_iter": "classic hue drift with iteration; higher = more color shift in borders",
        "cine_shade_lo": "cinematic minimum brightness (HSV V); higher = lifted shadows",
        "cine_shade_hi": "cinematic extra brightness for fast-converging pixels",
        "cine_shade_pow": "cinematic shade falloff; higher = darker, sharper borders",
        "cine_cos_base": "cinematic ring shading baseline; higher = brighter overall",
        "cine_cos_amp": "cinematic ring contrast from iteration ripples; higher = stronger bands",
        "cine_cos_freq": "cinematic ring spacing in iterations; higher = tighter concentric bands",
        "cine_rad_lo": "cinematic brightness far from center; higher = less radial vignette",
        "cine_rad_hi": "extra brightness at the origin that fades with |z|; higher = stronger glow",
        "cine_rad_decay": "how fast the center glow falls off with |z|; higher = tighter glow",
        "cine_sat_lo": "cinematic saturation for fast pixels; higher = richer early basins",
        "cine_sat_hi": "extra saturation toward slow pixels; higher = richer fractal borders",
        "cine_hue_iter": "cinematic hue drift with iteration; higher = more color shift in borders",
        "cine_hue_base": "fallback first-root hue 0..1 when power has no preset (and hues unset)",
        "cine_hue_span": "fallback hue spread 0..1 across roots when power has no preset",
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
    """Colored basins of attraction for the nth roots of unity (z^n = 1)."""
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
        "coeffs": "polynomial coeffs highest degree first, e.g. [1, 0, -2, 2] for z³−2z+2",
        "max_iter": "root-finding steps per pixel; higher = finer fractal borders, slower",
        "tol": "stop when |f(z)| is this small; lower = tighter, may need more iterations",
        "hues": "list of 0..1 hues cycled across angle-bins (length sets how many colors)",
        "basin_scale": "angle bins before wrapping hues; higher = more color sectors around origin",
        "cine_shade_lo": "minimum brightness (HSV V); higher = lifted shadows",
        "cine_shade_hi": "extra brightness for fast-converging pixels",
        "cine_shade_pow": "shade falloff; higher = darker, sharper borders",
        "cine_cos_base": "ring shading baseline; higher = brighter overall",
        "cine_cos_amp": "ring contrast from iteration ripples; higher = stronger bands",
        "cine_cos_freq": "ring spacing in iterations; higher = tighter concentric bands",
        "cine_rad_lo": "brightness far from center; higher = less radial vignette",
        "cine_rad_hi": "extra brightness at the origin that fades with |z|; higher = stronger glow",
        "cine_rad_decay": "how fast the center glow falls off with |z|; higher = tighter glow",
        "cine_sat_lo": "saturation for fast pixels; higher = richer early basins",
        "cine_sat_hi": "extra saturation toward slow pixels; higher = richer fractal borders",
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
    """Colored basins for a custom polynomial's roots."""
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
