"""Domain coloring and Pickover popcorn field fractals.

Domain functions w = f(z) are pluggable: ``register_domain(name, fn, defaults=...)``
from Python, or from TOML::

    [domain.blossom]
    w = "(z**5 - a) / (z**2 + 0.5j)"
    params = { a = 1.0 }

then ``params = { kind = "blossom", expr_params = { a = 2.0 } }`` in a recipe. A
one-off ``expr = "..."`` param works too, without registering anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..color import histogram_index, hsv_to_rgb
from ..expr import compile_expr
from ..params import to_complex
from ..loader import section
from ..registry import Registry
from .base import generator


@dataclass(frozen=True)
class DomainFn:
    name: str
    fn: Callable[..., np.ndarray]
    description: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)


DOMAIN: Registry[DomainFn] = Registry("domain-coloring function")


def register_domain(
    name: str,
    fn: Callable[[np.ndarray], np.ndarray],
    *,
    description: str = "",
    defaults: dict[str, Any] | None = None,
    replace: bool = False,
) -> DomainFn:
    """Register ``fn(z, **params) -> w``; ``defaults`` are its tunable params."""
    return DOMAIN.register(name, DomainFn(name, fn, description, dict(defaults or {})), replace=replace)


def _rational(z: np.ndarray) -> np.ndarray:
    return (z**2 - 1.0) / (z**2 + z + 1.0)


def _sin(z: np.ndarray) -> np.ndarray:
    return np.sin(z)


def _exp_inv(z: np.ndarray) -> np.ndarray:
    return np.exp(1.0 / (z + 1e-6))


def _poly(z: np.ndarray) -> np.ndarray:
    return z**5 - z + 0.2


for _name, _fn, _desc in (
    ("rational", _rational, "(z^2-1)/(z^2+z+1)"),
    ("sin", _sin, "sin(z)"),
    ("exp_inv", _exp_inv, "exp(1/z)"),
    ("poly", _poly, "z^5 - z + 0.2"),
):
    register_domain(_name, _fn, description=_desc)


@section("domain")
def _domain_section(name: str, table: dict[str, Any], source: str, replace: bool) -> None:
    t = dict(table)
    expr = str(t.pop("w"))
    defaults = {k: to_complex(v) if isinstance(v, (list, str)) else v for k, v in dict(t.pop("params", None) or {}).items()}
    desc = str(t.pop("description", expr))
    if t:
        raise ValueError(f"unknown domain keys {sorted(t)} (put parameters in `params = {{...}}`)")
    compiled = compile_expr(expr, variables=["z"], params=sorted(defaults))
    register_domain(name, lambda z, **kw: compiled(z, **kw), description=desc, defaults=defaults, replace=replace)


def _eval_domain(
    zr: np.ndarray,
    zi: np.ndarray,
    kind: str,
    expr: str | None,
    expr_params: dict[str, Any],
) -> np.ndarray:
    z = zr + 1j * zi
    if expr:
        fn = compile_expr(expr, variables=["z"], params=sorted(expr_params))
        return fn(z, **expr_params)
    d = DOMAIN.get(kind)
    unknown = set(expr_params) - set(d.defaults)
    if unknown:
        raise ValueError(f"domain function {kind!r} has no params {sorted(unknown)} (has {sorted(d.defaults)})")
    return d.fn(z, **{**d.defaults, **expr_params})


def _infinite_magnitude_rgb(
    *,
    contour_lo: float,
    contour_hi: float,
    checker_lo: float,
    mag_scale: float,
    sat_lo: float,
    sat_hi: float,
) -> np.ndarray:
    """HSV color of |w| → ∞: mid contour, no checker oscillation, phase 0."""
    ridges = contour_lo + 0.5 * contour_hi
    v = float(np.clip(ridges * checker_lo * mag_scale, 0.0, 1.0))
    s = float(np.clip(sat_lo + sat_hi, 0.35, 1.0))
    return hsv_to_rgb(np.array([0.5]), np.array([s]), np.array([v]))[0]


def _color_domain(
    w: np.ndarray,
    *,
    contour_lo: float,
    contour_hi: float,
    contour_period: float,
    checker_scale: float,
    checker_lo: float,
    checker_hi: float,
    mag_pow: float,
    mag_scale: float,
    sat_lo: float,
    sat_hi: float,
    sat_mag_scale: float,
) -> np.ndarray:
    if not np.isfinite(float(contour_period)) or float(contour_period) <= 0.0:
        raise ValueError(f"contour_period must be > 0, got {contour_period!r}")
    finite = np.isfinite(w)
    w_work = w if np.all(finite) else np.where(finite, w, 1.0)
    mag = np.abs(w_work)
    phase = np.angle(w_work)
    h = (phase + np.pi) / (2.0 * np.pi)
    logm = np.log(mag + 1e-12)
    ridges = contour_lo + contour_hi * (0.5 + 0.5 * np.cos(2.0 * np.pi * logm / contour_period))
    checker = checker_lo + checker_hi * np.sign(np.sin(checker_scale * w_work.real) * np.sin(checker_scale * w_work.imag))
    v = np.clip(ridges * checker * (mag**mag_pow / (1.0 + mag**mag_pow) * mag_scale), 0.0, 1.0)
    s = np.clip(sat_lo + sat_hi * (1.0 - 1.0 / (1.0 + mag * sat_mag_scale)), 0.35, 1.0)
    rgb = hsv_to_rgb(h, s, v)
    fill = (~finite) | ~np.all(np.isfinite(rgb), axis=-1)
    if np.any(fill):
        inf_rgb = _infinite_magnitude_rgb(
            contour_lo=contour_lo,
            contour_hi=contour_hi,
            checker_lo=checker_lo,
            mag_scale=mag_scale,
            sat_lo=sat_lo,
            sat_hi=sat_hi,
        )
        rgb = np.array(rgb, copy=True)
        rgb[fill] = inf_rgb
    return rgb


@generator(
    "domain_color",
    choices={"kind": DOMAIN.names},
    category="domain",
    uses_view=True,
    param_docs={
        "kind": "preset complex function (rational, sin, exp_inv, poly); ignored if expr is set",
        "expr": "custom formula in z (and named params); if set, overrides kind",
        "expr_params": "{name: value} for symbols in expr, or overrides of the preset's params",
        "contour_lo": "base brightness of magnitude rings; higher = brighter overall",
        "contour_hi": "ring contrast on log|w|; higher = stronger concentric contours",
        "contour_period": "log|w| spacing of magnitude rings; lower = more tightly packed rings",
        "checker_scale": "checker frequency on Re(w), Im(w); higher = more grid squares",
        "checker_lo": "checker brightness baseline; higher = brighter grid",
        "checker_hi": "checker contrast (± this from baseline); higher = stronger squares",
        "mag_pow": "how fast brightness rolls off with |w|; lower = flatter, less contrast",
        "mag_scale": "overall brightness from |w|; higher = brighter image",
        "sat_lo": "minimum color intensity 0..1; higher = richer even near zeros",
        "sat_hi": "extra saturation toward large |w|; higher = richer near poles",
        "sat_mag_scale": "how quickly saturation rises with |w|; higher = saturates sooner",
    },
    ranges={
        "contour_lo": (0.2, 0.9),
        "contour_hi": (0.1, 0.8),
        "contour_period": (0.1, 2.0),
        "checker_scale": (1.0, 20.0),
        "checker_lo": (0.5, 1.0),
        "checker_hi": (0.0, 0.5),
        "mag_pow": (0.05, 0.8),
        "mag_scale": (0.5, 3.0),
        "sat_lo": (0.3, 0.9),
        "sat_hi": (0.1, 0.8),
        "sat_mag_scale": (0.2, 3.0),
    },
)
def domain_color(
    ctx,
    *,
    kind: str = "rational",
    expr: str | None = None,
    expr_params: dict | None = None,
    contour_lo: float = 0.55,
    contour_hi: float = 0.45,
    contour_period: float = 0.6931471805599453,
    checker_scale: float = 6.0,
    checker_lo: float = 0.88,
    checker_hi: float = 0.12,
    mag_pow: float = 0.15,
    mag_scale: float = 1.35,
    sat_lo: float = 0.55,
    sat_hi: float = 0.4,
    sat_mag_scale: float = 1.0,
):
    """Paints a complex function by hue = phase and brightness = magnitude."""
    if not np.isfinite(float(contour_period)) or float(contour_period) <= 0.0:
        raise ValueError(f"contour_period must be > 0, got {contour_period!r}")
    zr, zi = ctx.plane()
    kw = dict(expr_params or {})
    w = _eval_domain(zr, zi, kind, expr, kw)
    return _color_domain(
        w,
        contour_lo=contour_lo,
        contour_hi=contour_hi,
        contour_period=contour_period,
        checker_scale=checker_scale,
        checker_lo=checker_lo,
        checker_hi=checker_hi,
        mag_pow=mag_pow,
        mag_scale=mag_scale,
        sat_lo=sat_lo,
        sat_hi=sat_hi,
        sat_mag_scale=sat_mag_scale,
    )


@generator(
    "popcorn",
    category="domain",
    uses_view=True,
    param_docs={
        "steps": "popcorn iterations per pixel; higher = more warped, intricate waves, slower",
        "hstep": "step size of the popcorn map; higher = stronger warping (can blow up)",
        "saturation": "color intensity 0..1; higher = richer, lower = greyer",
        "val_lo": "minimum brightness after histogram; higher = lifted shadows",
        "val_hi": "brightness range from the histogram; higher = brighter highlights",
        "val_gamma": "gamma on histogram brightness; lower = brighter midtones",
    },
    ranges={
        "steps": (5, 120, 1),
        "hstep": (0.01, 0.2),
        "saturation": (0.2, 1.0),
        "val_lo": (0.0, 0.5),
        "val_hi": (0.3, 1.0),
        "val_gamma": (0.2, 2.0),
    },
)
def popcorn(
    ctx,
    *,
    steps: int = 40,
    hstep: float = 0.05,
    saturation: float = 0.62,
    val_lo: float = 0.18,
    val_hi: float = 0.82,
    val_gamma: float = 0.75,
):
    """Warped wave field from Pickover's popcorn map; hue from drift direction."""
    from .. import kernels, progress

    zr, zi = ctx.plane()

    def run(a, b):
        return kernels.popcorn_kernel(
            np.ascontiguousarray(a, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64),
            int(steps),
            float(hstep),
            progress.units(a.shape[0]),
        )

    dx, dy = ctx.strips(run, zr, zi)
    ang = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
    mag = np.sqrt(dx * dx + dy * dy)
    v = histogram_index(np.log1p(mag), np.isfinite(mag))
    v = val_lo + val_hi * np.power(v, val_gamma)
    return hsv_to_rgb(ang, np.full_like(ang, saturation), v)
