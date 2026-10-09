"""3D ODE flow attractors (Lorenz, Rössler, …) projected to 2D density."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .. import progress
from ..expr import compile_expr
from ..loader import section
from ..registry import Registry
from ..view import oblique_projection
from .base import generator
from ..parallel import chunked
from ..fieldcache import cached as field_cached
from ..fieldcache import ctx_key as field_ctx_key
from .density import accumulate_density, tone_map_density


@dataclass(frozen=True)
class FlowSpec:
    name: str
    deriv: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
    defaults: dict[str, float]
    description: str = ""


FLOWS: Registry[FlowSpec] = Registry("3D flow")


def register_flow(
    name: str,
    deriv: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    defaults: dict[str, float] | None = None,
    *,
    description: str = "",
    replace: bool = False,
) -> FlowSpec:
    """Register ``deriv(x, y, z, **params) -> (dx, dy, dz)``."""
    defaults = dict(defaults or {})
    clash = set(defaults) & {"x", "y", "z"}
    if clash:
        raise ValueError(
            f"flow {name!r} params must not use state names {sorted(clash)} (reserved for x, y, z)"
        )
    spec = FlowSpec(name, deriv, defaults, description=description)
    return FLOWS.register(name, spec, replace=replace)


def _lorenz_deriv(x, y, z, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


def _rossler_deriv(x, y, z, a=0.2, b=0.2, c=5.7):
    return -y - z, x + a * y, b + z * (x - c)


def _aizawa_deriv(x, y, z, a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1):
    return (
        (z - b) * x - d * y,
        d * x + (z - b) * y,
        c + a * z - z**3 / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * x**3,
    )


def _thomas_deriv(x, y, z, b=0.19):
    return np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z


def _halvorsen_deriv(x, y, z, a=1.89):
    dx = -a * x - 4.0 * y - 4.0 * z - y * y
    dy = -a * y - 4.0 * z - 4.0 * x - z * z
    dz = -a * z - 4.0 * x - 4.0 * y - x * x
    return dx, dy, dz


for _n, _fn, _defs, _desc in (
    ("lorenz", _lorenz_deriv, {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}, "Lorenz butterfly"),
    ("rossler", _rossler_deriv, {"a": 0.2, "b": 0.2, "c": 5.7}, "Rössler attractor"),
    ("aizawa", _aizawa_deriv, {"a": 0.95, "b": 0.7, "c": 0.6, "d": 3.5, "e": 0.25, "f": 0.1}, "Aizawa attractor"),
    ("thomas", _thomas_deriv, {"b": 0.19}, "Thomas cyclically symmetric attractor"),
    ("halvorsen", _halvorsen_deriv, {"a": 1.89}, "Halvorsen attractor (ODE only)"),
):
    register_flow(_n, _fn, _defs, description=_desc)


@section("flow")
def _flow_section(name: str, table: dict[str, Any], source: str, replace: bool) -> None:
    t = dict(table)
    dx_src = t.pop("dx")
    dy_src = t.pop("dy")
    dz_src = t.pop("dz")
    defaults = {k: float(v) for k, v in dict(t.pop("params", {})).items()}
    pnames = list(defaults)
    dx_fn = compile_expr(str(dx_src), ["x", "y", "z"], pnames)
    dy_fn = compile_expr(str(dy_src), ["x", "y", "z"], pnames)
    dz_fn = compile_expr(str(dz_src), ["x", "y", "z"], pnames)

    def deriv(x, y, z, **kw):
        p = {**defaults, **kw}
        return dx_fn(x, y, z, **p), dy_fn(x, y, z, **p), dz_fn(x, y, z, **p)

    desc = str(t.pop("description", f"custom flow from {source}"))
    if t:
        raise ValueError(f"unknown flow keys {sorted(t)}")
    register_flow(name, deriv, defaults, description=desc, replace=replace)


def _rk4_step(
    deriv: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx, dy, dz = deriv(x, y, z)
    k1x, k1y, k1z = dt * dx, dt * dy, dt * dz
    dx, dy, dz = deriv(x + 0.5 * k1x, y + 0.5 * k1y, z + 0.5 * k1z)
    k2x, k2y, k2z = dt * dx, dt * dy, dt * dz
    dx, dy, dz = deriv(x + 0.5 * k2x, y + 0.5 * k2y, z + 0.5 * k2z)
    k3x, k3y, k3z = dt * dx, dt * dy, dt * dz
    dx, dy, dz = deriv(x + k3x, y + k3y, z + k3z)
    k4x, k4y, k4z = dt * dx, dt * dy, dt * dz
    x = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
    y = y + (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0
    z = z + (k1z + 2 * k2z + 2 * k3z + k4z) / 6.0
    return x, y, z


def _rossler_particles(ctx, base: int, area_exp: float, area_cap: float) -> int:
    area = max(1.0, ctx.pixels / (1280 * 720))
    n = int(base * min(area_cap, max(1.0, area**area_exp)))
    return int(n * ctx.ultra_mul())


@generator(
    "flow_attractor",
    choices={"flow": FLOWS.names, "fill": ["auto", "cover", "contain"]},
    category="attractor",
    param_docs={
        "flow": "which 3D ODE (lorenz, rossler, aizawa, …); changes the overall shape",
        "coeffs": "{name: value} merged over the flow's defaults; retunes the attractor",
        "palette": "color gradient for the trajectory density",
        "seed": "random seed for the starting cloud; change to vary the grain",
        "particle_base": "base trajectory count; higher = denser cloud, slower",
        "particle_cap": "max particles after scaling (unused for lorenz/aizawa/thomas)",
        "rossler_particles": "if true, use the Rössler-specific particle-count curve (rarely needed)",
        "rossler_base": "base count when rossler_particles is on; users rarely change this",
        "rossler_area_exp": "how fast Rössler particle count grows with resolution; rarely change",
        "rossler_area_cap": "max area multiplier for Rössler particle count; rarely change",
        "steps": "integration steps per particle; higher = longer traces, slower",
        "dt": "integration time step; higher = faster travel along the flow, can skip loops",
        "skip": "warmup steps not drawn; higher = less clutter near the start point",
        "angle_deg": "camera yaw in degrees; spins the attractor in the xy plane",
        "z_scale": "vertical stretch of depth; higher = taller, more 3D-looking shape",
        "tilt": "how much xy leans into the vertical; higher = more isometric lean",
        "blur": "blur of the density cloud, in output pixels; higher = softer",
        "gamma": "tone curve on density; lower = brighter, more filled-in midtones",
        "tight": "if true, crop to the dense core instead of a percentile window",
        "percentiles": "[lo, hi] percentile crop for auto-framing; tighter = zoom into the core",
        "pad": "extra fraction of the auto window; higher = more margin around the shape",
        "init_mean": "[x, y, z] center of the starting Gaussian cloud",
        "init_sigma": "[sx, sy, sz] starting spread (sz unused if init_z_uniform is set)",
        "init_z_uniform": "optional [z_lo, z_hi] uniform z instead of a Gaussian; None = Gaussian",
        "fill": "aspect fit: auto, contain (whole shape), or cover (fill the frame, crop)",
    },
    ranges={
        "seed": (0, 9999, 1),
        "particle_base": (500, 15_000, 100),
        "particle_cap": (2000, 50_000, 500),
        "rossler_base": (1000, 20_000, 100),
        "rossler_area_exp": (0.1, 0.8),
        "rossler_area_cap": (1.0, 6.0),
        "steps": (200, 5000, 50),
        "dt": (0.001, 0.05),
        "skip": (0, 500, 10),
        "angle_deg": (0.0, 90.0),
        "z_scale": (0.2, 1.5),
        "tilt": (0.0, 0.8),
        "blur": (0.0, 3.0),
        "gamma": (0.2, 2.0),
        "pad": (0.0, 0.2),
    },
)
def flow_attractor(
    ctx,
    *,
    flow: str = "lorenz",
    coeffs: dict[str, Any] | None = None,
    palette: Any = "ember",
    seed: int = 3,
    particle_base: int = 2500,
    particle_cap: int = 10_000,
    rossler_particles: bool = False,
    rossler_base: int = 5200,
    rossler_area_exp: float = 0.38,
    rossler_area_cap: float = 3.5,
    steps: int = 1800,
    dt: float = 0.008,
    skip: int = 80,
    angle_deg: float = 38.0,
    z_scale: float = 0.72,
    tilt: float = 0.22,
    blur: float = 0.85,
    gamma: float = 0.68,
    tight: bool = False,
    percentiles: tuple[float, float] = (0.4, 99.6),
    pad: float = 0.06,
    init_mean: tuple[float, float, float] = (0.0, 1.0, 20.0),
    init_sigma: tuple[float, float, float] = (0.4, 0.4, 2.0),
    init_z_uniform: tuple[float, float] | None = None,
    fill: str = "auto",
):
    """3D ODE attractor (Lorenz, Rössler, …) projected to a 2D density cloud."""
    spec = FLOWS.get(flow)
    unknown = set(coeffs or {}) - set(spec.defaults)
    if unknown:
        raise ValueError(
            f"flow {flow!r} has no params {sorted(unknown)} (has {sorted(spec.defaults)})"
        )
    if len(init_mean) != 3:
        raise ValueError(f"init_mean must have length 3, got {len(init_mean)}")
    if len(init_sigma) != 3:
        raise ValueError(f"init_sigma must have length 3, got {len(init_sigma)}")
    if len(percentiles) != 2:
        raise ValueError(f"percentiles must have length 2, got {len(percentiles)}")
    params = {**spec.defaults, **(coeffs or {})}
    deriv = lambda x, y, z: spec.deriv(x, y, z, **params)

    rng = np.random.default_rng(seed)

    def count(c) -> int:
        if rossler_particles:
            return _rossler_particles(c, rossler_base, rossler_area_exp, rossler_area_cap)
        if flow in ("lorenz", "aizawa", "thomas"):
            return int(particle_base * c.ultra_mul())
        return c.scale_particles(particle_base, particle_cap)

    n = count(ctx)
    portrait_fill = "cover" if ctx.portrait else "contain"
    if fill == "auto":
        fill = portrait_fill

    norm = ctx.density_norm(n, count(ctx.reference()))
    fkey = field_ctx_key(
        ctx,
        "flow_attractor",
        flow=flow,
        coeffs=params,
        seed=seed,
        particle_base=particle_base,
        particle_cap=particle_cap,
        rossler_particles=rossler_particles,
        rossler_base=rossler_base,
        rossler_area_exp=rossler_area_exp,
        rossler_area_cap=rossler_area_cap,
        steps=steps,
        dt=dt,
        skip=skip,
        angle_deg=angle_deg,
        z_scale=z_scale,
        tilt=tilt,
        tight=tight,
        percentiles=percentiles,
        pad=pad,
        init_mean=init_mean,
        init_sigma=init_sigma,
        init_z_uniform=init_z_uniform,
        fill=fill,
        n=n,
    )

    def _fields() -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(init_mean[0], init_sigma[0], size=n)
        y = rng.normal(init_mean[1], init_sigma[1], size=n)
        if init_z_uniform is not None:
            z = rng.uniform(float(init_z_uniform[0]), float(init_z_uniform[1]), size=n)
        else:
            z = rng.normal(init_mean[2], init_sigma[2], size=n)
        xs, ys, zs = [], [], []
        rk4 = lambda a, b, c: _rk4_step(deriv, a, b, c, dt)  # noqa: E731
        for i in range(steps):
            x, y, z = chunked(rk4, x, y, z)
            progress.update(i / steps)
            if i > skip:
                xs.append(x.copy())
                ys.append(y.copy())
                zs.append(z.copy())
        px: list[np.ndarray] = []
        py: list[np.ndarray] = []
        hs: list[np.ndarray] = []
        denom = max(len(xs) - 1, 1)
        for i, (xx, yy, zz) in enumerate(zip(xs, ys, zs)):
            pxi, pyi = chunked(
                lambda a, b, c: oblique_projection(a, b, c, angle_deg=angle_deg, z_scale=z_scale, tilt=tilt),
                xx, yy, zz,
            )
            px.append(pxi)
            py.append(pyi)
            hs.append(np.full(n, i / denom, dtype=np.float64))
        return accumulate_density(
            ctx.width, ctx.height, px, py, hs,
            tight=tight, percentiles=percentiles, pad=pad, fill=fill,
        )[:2]

    density, hue_acc = field_cached(fkey, _fields)
    return tone_map_density(
        np.array(density, copy=False), np.array(hue_acc, copy=False),
        palette, blur=blur, gamma=gamma, norm=norm,
    )
