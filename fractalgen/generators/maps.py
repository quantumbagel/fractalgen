"""2D iterated-map strange attractors (Clifford, De Jong, Ikeda, …)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .. import progress
from ..fieldcache import cached as field_cached
from ..fieldcache import ctx_key as field_ctx_key
from ..expr import compile_expr
from ..loader import section
from ..registry import Registry
from .base import generator
from ..parallel import chunked
from .density import auto_window, dense_window, deposit, tone_map_density


@dataclass(frozen=True)
class MapSpec:
    name: str
    step: Callable[..., tuple[np.ndarray, np.ndarray]]
    defaults: dict[str, float]
    description: str = ""


MAPS: Registry[MapSpec] = Registry("2D map")


def register_map(
    name: str,
    step: Callable[..., tuple[np.ndarray, np.ndarray]],
    defaults: dict[str, float] | None = None,
    *,
    description: str = "",
    replace: bool = False,
) -> MapSpec:
    """Register ``step(x, y, **params) -> (nx, ny)`` with default coefficients."""
    spec = MapSpec(name, step, dict(defaults or {}), description=description)
    return MAPS.register(name, spec, replace=replace)


def _clifford(x, y, a, b, c, d):
    nx = np.sin(a * y) + c * np.cos(a * x)
    ny = np.sin(b * x) + d * np.cos(b * y)
    return nx, ny


def _dejong(x, y, a, b, c, d):
    nx = np.sin(a * y) - np.cos(b * x)
    ny = np.sin(c * x) - np.cos(d * y)
    return nx, ny


def _hopalong(x, y, a, b, c):
    nx = y - np.sign(x) * np.sqrt(np.abs(b * x - c))
    ny = a - x
    return nx, ny


def _ikeda(x, y, u):
    t = 0.4 - 6.0 / (1.0 + x * x + y * y)
    ct, st = np.cos(t), np.sin(t)
    nx = 1.0 + u * (x * ct - y * st)
    ny = u * (x * st + y * ct)
    return nx, ny


def _gumowski_mira(x, y, a, b):
    def f(z):
        return a * z + 2.0 * (1.0 - a) * z * z / (1.0 + z * z)

    nx = b * y + f(x)
    ny = -x + f(nx)
    return nx, ny


def _svensson(x, y, a, b, c, d):
    nx = d * np.sin(a * x) - np.sin(b * y)
    ny = c * np.cos(a * x) + np.cos(b * y)
    return nx, ny


def _bedhead(x, y, a, b):
    nx = np.sin(x * y / b) * y + np.cos(a * x - y)
    ny = x + np.sin(y) / b
    return nx, ny


def _fractal_dream(x, y, a, b, c, d):
    nx = np.sin(b * y) + c * np.sin(b * x)
    ny = np.sin(a * x) + d * np.sin(a * y)
    return nx, ny


for _n, _fn, _defs, _desc in (
    ("clifford", _clifford, {"a": -1.4, "b": 1.6, "c": 1.0, "d": 0.7}, "sin/cos Clifford map"),
    ("dejong", _dejong, {"a": -2.05, "b": -1.15, "c": -2.14, "d": 1.41}, "De Jong sinusoidal map"),
    ("hopalong", _hopalong, {"a": 2.0, "b": 1.0, "c": 0.5}, "Barry Martin hopalong"),
    ("ikeda", _ikeda, {"u": 0.918}, "Ikeda map"),
    ("gumowski_mira", _gumowski_mira, {"a": -0.48, "b": 0.95}, "Gumowski–Mira"),
    ("svensson", _svensson, {"a": 1.4, "b": 1.56, "c": 1.4, "d": -6.6}, "Svensson map"),
    ("bedhead", _bedhead, {"a": -0.81, "b": -0.92}, "Bedhead map"),
    ("fractal_dream", _fractal_dream, {"a": -0.9669, "b": 2.8798, "c": 0.765, "d": 0.745}, "Fractal dream map"),
):
    register_map(_n, _fn, _defs, description=_desc)


@section("map")
def _map_section(name: str, table: dict[str, Any], source: str, replace: bool) -> None:
    t = dict(table)
    x_src = t.pop("x")
    y_src = t.pop("y")
    defaults = {k: float(v) for k, v in dict(t.pop("params", {})).items()}
    pnames = list(defaults)
    x_fn = compile_expr(str(x_src), ["x", "y"], pnames)
    y_fn = compile_expr(str(y_src), ["x", "y"], pnames)

    def step(x, y, **kw):
        p = {**defaults, **kw}
        return x_fn(x, y, **p), y_fn(x, y, **p)

    desc = str(t.pop("description", f"custom map from {source}"))
    if t:
        raise ValueError(f"unknown map keys {sorted(t)}")
    register_map(name, step, defaults, description=desc, replace=replace)


def _make_step(map_name: str, coeffs: dict[str, Any] | None) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    spec = MAPS.get(map_name)
    params = {**spec.defaults, **(coeffs or {})}

    def step(x, y):
        return spec.step(x, y, **params)

    return step


def iterate_map(
    step: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    n_particles: int,
    warmup: int,
    steps: int,
    seed: int = 0,
    init: tuple[float, float] = (0.0, 0.0),
    init_sigma: float = 0.12,
    explode: float = 50.0,
    collect: bool = True,
    on_step: Callable[[np.ndarray, np.ndarray, float], None] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
    """Warm up particles, iterate, optionally stream via ``on_step``."""
    rng = np.random.default_rng(seed)
    cx, cy = init

    def _seed(n: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            rng.normal(cx, init_sigma, size=n),
            rng.normal(cy, init_sigma, size=n),
        )

    def _bad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return ~np.isfinite(x) | ~np.isfinite(y) | (np.abs(x) > explode) | (np.abs(y) > explode)

    x, y = _seed(n_particles)
    for _ in range(warmup):
        x, y = chunked(step, x, y)
        bad = chunked(_bad, x, y)
        if np.any(bad):
            x[bad], y[bad] = _seed(int(bad.sum()))
    xs, ys, hs = [], [], []
    denom = max(steps - 1, 1)
    for i in range(steps):
        x, y = chunked(step, x, y)
        progress.update((warmup + i) / (warmup + steps))
        bad = chunked(_bad, x, y)
        if np.any(bad):
            x[bad], y[bad] = _seed(int(bad.sum()))
        t = i / denom
        if on_step is not None:
            on_step(x, y, t)
        if collect:
            xs.append(x.copy())
            ys.append(y.copy())
            hs.append(np.full(n_particles, t, dtype=np.float64))
    if not collect:
        return None
    return xs, ys, hs


@generator(
    "map_attractor",
    choices={"map": MAPS.names},
    category="attractor",
    param_docs={
        "map": "Which attractor. Each has its own shape.",
        "coeffs": "Fine-tunes the shape: its folds and arms.",
        "palette": "Colors for the cloud.",
        "n": "Number of particles. More gives a denser cloud but is slower.",
        "warmup": "Early steps left undrawn so stray dots don't show.",
        "steps": "How long each particle is drawn. More gives a denser picture.",
        "seed": "Change it for a different pattern of grain.",
        "blur": "Softens the cloud.",
        "gamma": "Lower values brighten and fill in the cloud.",
        "percentiles": "How tightly the frame crops around the shape.",
        "tight": "Crop to the dense core, leaving out thin tails.",
        "init": "Where the particles start.",
        "init_sigma": "How spread out the particles start. Rarely needs changing.",
        "explode": "Particles that fly further than this start over.",
        "xlim": "Set the horizontal range shown yourself (set both ranges).",
        "ylim": "Set the vertical range shown yourself (set both ranges).",
        "pad": "Space around the shape.",
        "particle_cap": "Limits how many particles larger pictures add. Raise it only if a 4K render looks thin.",
    },
    ranges={
        "n": (10_000, 500_000, 10_000),
        "warmup": (10, 300, 10),
        "steps": (50, 800, 10),
        "seed": (0, 9999, 1),
        "blur": (0.0, 3.0),
        "gamma": (0.2, 2.0),
        "init_sigma": (0.01, 0.5),
        "explode": (10.0, 100.0),
        "pad": (0.0, 0.2),
        "particle_cap": (50_000, 1_000_000, 10_000),
    },
)
def map_attractor(
    ctx,
    *,
    map: str = "clifford",
    coeffs: dict[str, Any] | None = None,
    palette: Any = "cosmic",
    n: int = 180_000,
    warmup: int = 80,
    steps: int = 220,
    seed: int = 0,
    blur: float = 0.75,
    gamma: float = 0.72,
    percentiles: tuple[float, float] = (0.4, 99.6),
    tight: bool = False,
    init: tuple[float, float] = (0.0, 0.0),
    init_sigma: float = 0.12,
    explode: float = 50.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    pad: float = 0.06,
    particle_cap: int = 420_000,
):
    """A strange attractor: one simple rule, repeated millions of times, painting a cloud of light."""
    from ..view import fit_window

    step = _make_step(map, coeffs)
    w, h = ctx.width, ctx.height
    n_particles = ctx.scale_particles(n, particle_cap)
    fill = "cover" if ctx.portrait else "contain"

    norm = ctx.density_norm(n_particles, ctx.reference().scale_particles(n, particle_cap))
    fkey = field_ctx_key(
        ctx,
        "map_attractor",
        map=map,
        coeffs=coeffs or {},
        n=n,
        n_particles=n_particles,
        warmup=warmup,
        steps=steps,
        seed=seed,
        init=init,
        init_sigma=init_sigma,
        explode=explode,
        xlim=xlim,
        ylim=ylim,
        tight=tight,
        percentiles=percentiles,
        pad=pad,
        particle_cap=particle_cap,
        fill=fill,
    )

    def _fields() -> tuple[np.ndarray, np.ndarray]:
        if xlim is not None and ylim is not None:
            xmin, xmax = float(xlim[0]), float(xlim[1])
            ymin, ymax = float(ylim[0]), float(ylim[1])
        else:
            probe = min(n_particles, 80_000)
            xs, ys, hs = iterate_map(
                step,
                probe,
                warmup,
                steps,
                seed=seed,
                init=init,
                init_sigma=init_sigma,
                explode=explode,
            )
            if tight:
                xmin, xmax, ymin, ymax = dense_window(xs, ys)
            else:
                xmin, xmax, ymin, ymax = auto_window(xs, ys, pad=pad, percentiles=percentiles)
        xmin, xmax, ymin, ymax = fit_window(xmin, xmax, ymin, ymax, w, h, fill=fill)
        density = np.zeros((h, w), dtype=np.float64)
        hue_acc = np.zeros((h, w), dtype=np.float64)

        def _on_step(x, y, t):
            deposit(density, hue_acc, x, y, xmin, xmax, ymin, ymax, hue_val=t)

        iterate_map(
            step,
            n_particles,
            warmup,
            steps,
            seed=seed,
            init=init,
            init_sigma=init_sigma,
            explode=explode,
            collect=False,
            on_step=_on_step,
        )
        return density, hue_acc

    density, hue_acc = field_cached(fkey, _fields)
    return tone_map_density(
        np.array(density, copy=False), np.array(hue_acc, copy=False),
        palette, blur=blur, gamma=gamma, norm=norm,
    )
