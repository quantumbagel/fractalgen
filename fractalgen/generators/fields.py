"""Vector-field particle flows and scalar interference / noise fields.

Vector fields for ``flow_field`` are pluggable. From Python::

    register_vector_field("spin", lambda px, py, k=1.0: (-py * k, px), defaults={"k": 1.0})

or from TOML (variables ``px``, ``py``; vectors are normalized for you)::

    [vectorfield.braids]
    vx = "sin(a*py) + 0.5*cos(2.0*px*py)"
    vy = "cos(a*px) - 0.5*sin(py)"
    params = { a = 2.2 }

A recipe can override a field's params with ``params = { coeffs = { a = 3.0 } }``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.ndimage import gaussian_filter

from ..color import histogram_index, log_tone
from ..expr import compile_expr
from ..loader import section
from ..palette import sample_palette
from ..registry import Registry
from .base import generator


@dataclass(frozen=True)
class VectorField:
    name: str
    step: Callable[..., Any]
    description: str = ""
    defaults: dict[str, float] = field(default_factory=dict)


VECTOR_FIELDS: Registry[VectorField] = Registry("vector field")


def register_vector_field(
    name: str,
    step: Callable[..., Any],
    *,
    description: str = "",
    defaults: dict[str, float] | None = None,
    replace: bool = False,
) -> VectorField:
    """Register ``step(px, py, **params) -> (vx, vy)`` (unit vectors).

    ``defaults`` names the field's tunable params; recipes override them via ``coeffs``.
    """
    vf = VectorField(name, step, description=description, defaults=dict(defaults or {}))
    return VECTOR_FIELDS.register(name, vf, replace=replace)


def _normalize_field(vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nrm = np.sqrt(vx * vx + vy * vy) + 1e-9
    return vx / nrm, vy / nrm


def _compile_vector_field(vx_src: str, vy_src: str, param_names: tuple[str, ...]) -> Callable[..., Any]:
    vx_fn = compile_expr(vx_src, ("px", "py"), param_names)
    vy_fn = compile_expr(vy_src, ("px", "py"), param_names)

    def step(px: np.ndarray, py: np.ndarray, **kw: Any) -> tuple[np.ndarray, np.ndarray]:
        return _normalize_field(vx_fn(px, py, **kw), vy_fn(px, py, **kw))

    return step


@section("vectorfield")
def _vectorfield_section(name: str, table: dict[str, Any], source: str, replace: bool) -> None:
    t = dict(table)
    vx = str(t.pop("vx"))
    vy = str(t.pop("vy"))
    defaults = {k: float(v) for k, v in dict(t.pop("params", None) or {}).items()}
    desc = str(t.pop("description", vx))
    if t:
        raise ValueError(f"unknown vectorfield keys {sorted(t)}")
    step = _compile_vector_field(vx, vy, tuple(defaults))
    register_vector_field(name, step, description=desc, defaults=defaults, replace=replace)


@generator(
    "flow_field",
    choices={"field": VECTOR_FIELDS.names},
    category="fields",
    param_docs={
        "field": "registered vector field name",
        "coeffs": "override the field's own params (merged over its defaults)",
        "particles": "base particle count before ctx.scale_particles",
        "particles_cap": "upper cap for scale_particles",
        "steps": "integration steps per particle",
        "seed": "RNG seed for initial positions and respawns",
        "palette": "palette for blended hue and tone",
        "wrap": "toroidal wrap at plane edges instead of respawn",
        "dt": "integration time step multiplier",
        "speed": "scalar speed applied to unit field vectors",
        "respawn_divisor": "every step cycle, respawn n_particles // this many particles",
        "respawn_period": "respawn every this many steps (when step % period == period-1)",
        "density_blur": "Gaussian blur sigma on deposited density (pixels at 4K)",
        "tone_lo": "log-tone percentile low for density",
        "tone_hi": "log-tone percentile high for density",
        "tone_gamma": "gamma on log tone before palette mix",
        "hue_mix": "weight of flow-angle hue vs tone index",
        "tone_mix": "weight of tone index in palette coordinate",
        "rgb_floor": "minimum RGB multiplier from tone",
        "rgb_ceil": "tone-scaled RGB ceiling multiplier",
    },
    ranges={
        "particles": (10_000, 300_000, 10_000),
        "particles_cap": (100_000, 1_000_000, 10_000),
        "steps": (50, 800, 10),
        "seed": (0, 9999, 1),
        "dt": (0.002, 0.05),
        "speed": (5.0, 40.0),
        "respawn_divisor": (4, 32, 1),
        "respawn_period": (8, 64, 4),
        "density_blur": (0.0, 3.0),
        "tone_lo": (1.0, 20.0),
        "tone_hi": (90.0, 99.9),
        "tone_gamma": (0.2, 2.0),
        "hue_mix": (0.0, 1.0),
        "tone_mix": (0.0, 1.0),
        "rgb_floor": (0.0, 0.3),
        "rgb_ceil": (0.5, 1.0),
    },
)
def flow_field(
    ctx,
    *,
    field: str = "silk",
    coeffs: dict | None = None,
    particles: int = 90_000,
    particles_cap: int = 520_000,
    steps: int = 280,
    seed: int = 1,
    palette: Any = "twilight",
    wrap: bool = False,
    dt: float = 0.012,
    speed: float = 18.0,
    respawn_divisor: int = 12,
    respawn_period: int = 32,
    density_blur: float = 0.8,
    tone_lo: float = 5.0,
    tone_hi: float = 99.6,
    tone_gamma: float = 0.75,
    hue_mix: float = 0.55,
    tone_mix: float = 0.7,
    rgb_floor: float = 0.08,
    rgb_ceil: float = 0.92,
):
    """Particle traces through a normalized 2D vector field."""
    vf = VECTOR_FIELDS.get(field)
    unknown = set(coeffs or {}) - set(vf.defaults)
    if unknown:
        raise ValueError(f"vector field {field!r} has no params {sorted(unknown)} (has {sorted(vf.defaults)})")
    field_kw = {**vf.defaults, **(coeffs or {})}
    width, height = ctx.width, ctx.height
    aspect = ctx.aspect
    ref_ctx = ctx.reference()
    n_particles = ctx.scale_particles(particles, particles_cap)
    n_ref = ref_ctx.scale_particles(particles, particles_cap)
    n_used = float(n_particles * steps)
    n_ref_samples = float(n_ref * steps)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-aspect, aspect, size=n_particles)
    y = rng.uniform(-1.0, 1.0, size=n_particles)
    density = np.zeros((height, width), dtype=np.float64)
    hue = np.zeros((height, width), dtype=np.float64)

    xmin, xmax = -aspect, aspect
    ymin, ymax = -1.0, 1.0
    from .. import kernels
    from ..parallel import chunked

    field_step = lambda x, y: vf.step(x, y, **field_kw)  # noqa: E731
    pixel_index = lambda x, y, vx, vy: (  # noqa: E731
        ((x - xmin) / (xmax - xmin) * (width - 1)).astype(np.int32),
        ((ymax - y) / (ymax - ymin) * (height - 1)).astype(np.int32),
        (np.arctan2(vy, vx) + np.pi) / (2 * np.pi),
    )
    for i in range(steps):
        vx, vy = chunked(field_step, x, y)
        x = x + vx * dt * speed
        y = y + vy * dt * speed
        if wrap:
            x = ((x + aspect) % (2 * aspect)) - aspect
            y = ((y + 1.0) % 2.0) - 1.0
        else:
            left = (np.abs(x) > aspect) | (np.abs(y) > 1.0)
            if np.any(left):
                n_left = int(left.sum())
                x[left] = rng.uniform(-aspect, aspect, size=n_left)
                y[left] = rng.uniform(-1.0, 1.0, size=n_left)
        ix, iy, ang = chunked(pixel_index, x, y, vx, vy)
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        kernels.scatter_add2(density, hue, iy[valid], ix[valid], ang[valid])
        if i % respawn_period == respawn_period - 1:
            n_die = n_particles // respawn_divisor
            idx = rng.integers(0, n_particles, size=n_die)
            x[idx] = rng.uniform(-aspect, aspect, size=n_die)
            y[idx] = rng.uniform(-1.0, 1.0, size=n_die)

    blur = ctx.ref_px(density_blur)
    density = gaussian_filter(density, sigma=blur)
    hue = gaussian_filter(hue, sigma=blur)
    with np.errstate(invalid="ignore"):
        hmean = np.divide(hue, density, out=np.zeros_like(hue), where=density > 1)
    density *= ctx.density_norm(n_used, n_ref_samples)
    tone = log_tone(density, lo=tone_lo, hi=tone_hi)
    tone = np.power(tone, tone_gamma)
    rgb = sample_palette(palette, np.mod(hmean * hue_mix + tone * tone_mix, 1.0))
    rgb *= (rgb_floor + rgb_ceil * tone)[..., None]
    return np.clip(rgb, 0.0, 1.0)


@generator(
    "cymatics",
    category="fields",
    param_docs={
        "n_sources": "number of interfering point sources",
        "freq": "base radial wave number",
        "seed": "RNG seed for source placement and phase",
        "palette": "base palette from amplitude histogram",
        "highlight_palette": "palette for amplitude highlight mix (old code used abyss)",
        "source_span": "uniform range for source positions as fraction of plane",
        "freq_jitter": "per-source frequency multiplier range",
        "falloff": "inverse-distance damping on each source",
        "node_sharpness": "exponential sharpness emphasizing nodal lines",
        "rgb_floor": "minimum RGB away from nodes",
        "node_mix": "how strongly nodes darken the base color",
        "highlight_mix": "blend toward highlight palette by local amplitude",
        "hist_gamma": "gamma on amplitude histogram index",
    },
    ranges={
        "n_sources": (2, 20, 1),
        "freq": (5.0, 40.0),
        "seed": (0, 9999, 1),
        "source_span": (0.3, 1.0),
        "falloff": (0.1, 1.0),
        "node_sharpness": (1.0, 10.0),
        "rgb_floor": (0.0, 0.5),
        "node_mix": (0.0, 1.0),
        "highlight_mix": (0.0, 0.8),
        "hist_gamma": (0.2, 2.0),
    },
)
def cymatics(
    ctx,
    *,
    n_sources: int = 7,
    freq: float = 18.0,
    seed: int = 4,
    palette: Any = "abyss",
    highlight_palette: Any = "abyss",
    source_span: float = 0.7,
    freq_jitter: tuple[float, float] = (0.75, 1.25),
    falloff: float = 0.35,
    node_sharpness: float = 3.5,
    rgb_floor: float = 0.25,
    node_mix: float = 0.55,
    highlight_mix: float = 0.35,
    hist_gamma: float = 0.8,
):
    """Superposed radial waves with nodal-line emphasis."""
    width, height = ctx.width, ctx.height
    rng = np.random.default_rng(seed)
    aspect = ctx.aspect
    xs = np.linspace(-aspect, aspect, width)
    ys = np.linspace(1.0, -1.0, height)
    X, Y = np.meshgrid(xs, ys)
    amp = np.zeros((height, width), dtype=np.float64)
    for _ in range(n_sources):
        sx = rng.uniform(-aspect * source_span, aspect * source_span)
        sy = rng.uniform(-source_span, source_span)
        phase = rng.uniform(0, 2 * np.pi)
        k = freq * rng.uniform(freq_jitter[0], freq_jitter[1])
        r = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2)
        amp += np.sin(k * r + phase) / (1.0 + falloff * r)
    t = histogram_index(np.abs(amp), np.ones_like(amp, dtype=bool))
    nodes = np.exp(-np.abs(amp) * node_sharpness)
    rgb = sample_palette(palette, np.power(t, hist_gamma))
    rgb = rgb * (rgb_floor + (1.0 - rgb_floor) * (1.0 - node_mix * nodes)[..., None])
    highlight = sample_palette(highlight_palette, np.clip(t * 1.2, 0, 1))
    mix = np.clip(np.abs(amp) / (np.abs(amp).max() + 1e-9), 0, 1)[..., None]
    rgb = rgb * (1 - highlight_mix * mix) + highlight * (highlight_mix * mix)
    return np.clip(rgb, 0.0, 1.0)


@generator(
    "chladni",
    category="fields",
    param_docs={
        "n": "Chladni mode n",
        "m": "Chladni mode m",
        "palette": "palette from field histogram",
        "node_sharpness": "exponential sharpness on nodal lines",
        "rgb_floor": "minimum RGB away from nodes",
        "node_mix": "how strongly nodes darken the base color",
        "highlight_gain": "additive cool highlight on nodes",
        "hist_gamma": "gamma on field histogram index",
    },
    ranges={
        "n": (1, 12, 1),
        "m": (1, 12, 1),
        "node_sharpness": (5.0, 40.0),
        "rgb_floor": (0.0, 0.5),
        "node_mix": (0.0, 1.0),
        "highlight_gain": (0.0, 0.8),
        "hist_gamma": (0.2, 2.0),
    },
)
def chladni(
    ctx,
    *,
    n: int = 5,
    m: int = 3,
    palette: Any = "abyss",
    node_sharpness: float = 18.0,
    rgb_floor: float = 0.2,
    node_mix: float = 0.7,
    highlight_gain: float = 0.35,
    hist_gamma: float = 0.85,
):
    """Chladni plate nodal pattern on the unit square."""
    width, height = ctx.width, ctx.height
    aspect = ctx.aspect
    xs = np.linspace(-aspect, aspect, width)
    ys = np.linspace(1.0, -1.0, height)
    X, Y = np.meshgrid(xs, ys)
    px = (X / aspect + 1.0) * 0.5
    py = (Y + 1.0) * 0.5
    field = np.cos(n * np.pi * px) * np.cos(m * np.pi * py) - np.cos(m * np.pi * px) * np.cos(
        n * np.pi * py
    )
    t = histogram_index(np.abs(field), np.ones_like(field, dtype=bool))
    nodes = np.exp(-np.abs(field) * node_sharpness)
    rgb = sample_palette(palette, np.power(t, hist_gamma))
    rgb = rgb * (rgb_floor + (1.0 - rgb_floor) * (1.0 - node_mix * nodes)[..., None])
    highlight = np.clip(nodes, 0, 1)[..., None] * np.array([0.85, 0.95, 1.0])
    return np.clip(rgb + highlight_gain * highlight, 0.0, 1.0)


@generator(
    "worley",
    choices={"mode": ["F1", "F2-F1"]},
    category="fields",
    param_docs={
        "n_points": "number of random feature points",
        "seed": "RNG seed for feature positions",
        "palette": "palette from distance histogram",
        "mode": "F2-F1 (cell walls) or F1 (distance to nearest point)",
        "tone_floor": "minimum RGB multiplier from field",
        "hist_gamma": "gamma on distance histogram index",
    },
    ranges={
        "n_points": (8, 120, 1),
        "seed": (0, 9999, 1),
        "tone_floor": (0.0, 0.4),
        "hist_gamma": (0.2, 2.0),
    },
)
def worley(
    ctx,
    *,
    n_points: int = 48,
    seed: int = 7,
    palette: Any = "cosmic",
    mode: str = "F2-F1",
    tone_floor: float = 0.15,
    hist_gamma: float = 0.9,
):
    """Worley (cellular) noise from scattered feature points."""
    width, height = ctx.width, ctx.height
    rng = np.random.default_rng(seed)
    aspect = ctx.aspect
    pts = np.column_stack(
        [rng.uniform(-aspect, aspect, n_points), rng.uniform(-1.0, 1.0, n_points)]
    )
    xs = np.linspace(-aspect, aspect, width)
    ys = np.linspace(1.0, -1.0, height)
    X, Y = np.meshgrid(xs, ys)
    f1 = np.full((height, width), np.inf, dtype=np.float64)
    f2 = np.full((height, width), np.inf, dtype=np.float64)
    for px, py in pts:
        d = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
        closer = d < f1
        f2 = np.where(closer, f1, np.minimum(f2, d))
        f1 = np.where(closer, d, f1)
    field = (f2 - f1) if mode == "F2-F1" else f1
    t = histogram_index(field, np.ones_like(field, dtype=bool))
    rgb = sample_palette(palette, np.power(t, hist_gamma))
    rgb *= (tone_floor + (1.0 - tone_floor) * t)[..., None]
    return np.clip(rgb, 0.0, 1.0)
