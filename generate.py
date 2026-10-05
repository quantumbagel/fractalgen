#!/usr/bin/env python3
"""Generate 4K math / simulation desktop wallpapers."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from bg.attractors import (
    aizawa_flow,
    bedhead_step,
    clifford_step,
    dejong_step,
    fractal_dream_step,
    gumowski_mira_step,
    hopalong_step,
    ikeda_step,
    iterate_map,
    lorenz_flow,
    paint_attractor,
    rossler_flow,
    svensson_step,
    thomas_flow,
    tinkerbell_step,
    _auto_window,
    _dense_window,
    _deposit,
    fit_attractor_frame,
    tone_map_density,
)
from bg.color import histogram_index, hsv_to_rgb, sample_palette, save_png
from bg.cinematic import (
    ferrofluid_rosettes,
    gray_scott_morphogenesis,
    halvorsen_ribbon,
    hydrogen_orbitals,
    kerr_black_hole,
    mandelbulb_grotto,
    physarum_network,
    poincare_mirage,
)
from bg.extra import (
    apollonian_dots,
    barnsley_fern,
    chladni,
    cyclic_ca,
    harmonograph,
    sandpile,
    worley,
)
from bg.fractals import (
    color_escape,
    color_lyapunov,
    color_newton,
    complex_plane,
    domain_color,
    escape_time,
    lyapunov,
    newton_basins,
    newton_shifted,
    orbit_trap,
    popcorn_field,
)
from bg.sims import cymatics, flow_field, gray_scott, magnetic_pendulum

OUT = Path(__file__).resolve().parent / "backgrounds"
DEFAULT_W, DEFAULT_H = 3840, 2160
_REF_ASPECT = 16.0 / 9.0
_CURRENT_RECIPE: str | None = None

# Per-recipe portrait windows. half_width is the phone's *horizontal*
# half-extent when fit='fill'. Deep zooms omit an entry and get a 16:9 cover-crop.
PORTRAIT: dict[str, dict] = {
    "julia_swirl": {"center": -0.52 + 0.10j, "half_width": 0.48, "fit": "fill"},
    "julia_rabbit": {"center": 0.08 - 0.12j, "half_width": 0.70, "fit": "fill"},
    "julia_dragon": {"center": -0.18 + 0.24j, "half_width": 0.60, "fit": "fill"},
    "julia_snowflake": {"center": 0.02 + 0.18j, "half_width": 0.66, "fit": "fill"},
    "julia_san_marco": {"center": 0.0 + 0.0j, "half_width": 0.78, "fit": "fill"},
    "julia_golden": {"center": 0.16 + 0.24j, "half_width": 0.52, "fit": "fill"},
    "julia_siegel": {"center": 0.06 - 0.18j, "half_width": 0.68, "fit": "fill"},
    "julia_dendrite": {"center": 0.0 + 0.20j, "half_width": 0.70, "fit": "fill"},
    "julia_spirals": {"center": -0.10 + 0.40j, "half_width": 0.56, "fit": "fill"},
    "julia_airplane": {"center": -0.22 + 0.0j, "half_width": 0.52, "fit": "fill"},
    "julia_cauliflower": {"center": 0.58 + 0.10j, "half_width": 0.15, "fit": "fill"},
    "phoenix": {"center": 0.0 + 0.62j, "half_width": 0.50, "fit": "fill"},
    "phoenix_close": {"center": 0.0 + 0.66j, "half_width": 0.42, "fit": "fill"},
    "mandelbrot_cosmic": {"center": -0.62 + 0.0j, "half_width": 0.92, "fit": "fill"},
    "mandelbrot_orbit_trap": {"center": -0.22 + 0.0j, "half_width": 0.92, "fit": "fill"},
    "mandelbrot_orbit_cross": {"center": -0.28 + 0.0j, "half_width": 0.88, "fit": "fill"},
    "mandelbrot_antenna": {"center": -0.16 + 1.04045j, "half_width": 0.013, "fit": "fill"},
    "mandelbrot_seahorse": {"center": -0.74315 + 0.13145j, "half_width": 0.00072, "fit": "fill"},
    "mandelbrot_spiral": {"center": -0.76140 - 0.08472j, "fit": "cover"},
    "burning_ship": {"center": -0.78 - 0.58j, "half_width": 0.52, "fit": "fill"},
    "burning_ship_cathedral": {"center": -1.769 - 0.026j, "half_width": 0.020, "fit": "fill"},
    "buffalo": {"center": -0.65 - 0.52j, "half_width": 0.55, "fit": "fill"},
    "celtic": {"center": -0.88 + 0.0j, "half_width": 0.70, "fit": "fill"},
    "tricorn": {"center": 0.58 + 0.0j, "half_width": 0.82, "fit": "fill"},
    "multibrot3": {"center": -0.12 + 0.0j, "half_width": 0.70, "fit": "fill"},
    "multibrot4": {"center": 0.0 + 0.0j, "half_width": 0.60, "fit": "fill"},
    "newton_cubic_dusk": {"half_width": 0.95, "fit": "fill"},
    "newton_sextic": {"half_width": 0.90, "fit": "fill"},
    "newton_sextic_dusk": {"half_width": 0.90, "fit": "fill"},
    "newton_quartic": {"half_width": 0.88, "fit": "fill"},
    "newton_quintic": {"half_width": 0.88, "fit": "fill"},
    "newton_octic": {"half_width": 0.85, "fit": "fill"},
    "newton_shifted": {"half_width": 1.05, "fit": "fill"},
    "domain_rational": {"half_width": 1.15, "fit": "fill"},
    "domain_poly": {"half_width": 0.95, "fit": "fill"},
    "domain_sin": {"half_width": 1.65, "fit": "fill"},
    "domain_exp": {"half_width": 0.70, "fit": "fill"},
    "popcorn": {"half_width": 1.55, "fit": "fill"},
}


def _plane(w, h, center, half_width):
    fit = "fill"
    if h > w:
        p = PORTRAIT.get(_CURRENT_RECIPE or "", {})
        center = p.get("center", center)
        half_width = p.get("half_width", half_width)
        fit = p.get("fit", "cover")
    return complex_plane(w, h, center, half_width, fit=fit, ref_aspect=_REF_ASPECT)


def render_mandelbrot(w, h, center, half_width, max_iter, palette, kind="mandelbrot", **kw):
    cr, ci = _plane(w, h, center, half_width)
    n, esc = escape_time(cr, ci, max_iter, kind=kind, **kw)
    return color_escape(n, esc, palette)


def render_orbit_trap(w, h, center, half_width, max_iter, palette, trap="rings"):
    cr, ci = _plane(w, h, center, half_width)
    d = orbit_trap(cr, ci, max_iter, trap=trap)
    t = histogram_index(-np.log(d), np.ones_like(d, dtype=bool))
    rgb = sample_palette(palette, np.power(t, 0.85), cyclic=True)
    rgb *= 0.12 + 0.88 * np.power(t, 0.7)[..., None]
    return np.clip(rgb, 0.0, 1.0)


def render_newton(w, h, center, half_width, power, max_iter=48, style="classic"):
    cr, ci = _plane(w, h, center, half_width)
    roots, iters = newton_basins(cr, ci, power, max_iter=max_iter)
    return color_newton(roots, iters, power, zr=cr, zi=ci, style=style)


_REF_4K = 3840 * 2160


def _n_particles(w: int, h: int, base: int, cap: int) -> int:
    """4K keeps the old density curve. Above 4K, particle count scales with
    pixel count so hits/pixel (and zoom-in sharpness) stay constant."""
    ref = 1280 * 720
    px = w * h
    if px <= _REF_4K * 1.05:
        area = max(1.0, px / ref)
        scale = min(7.5, area**0.42)
        scaled = int(base * max(1.0, scale))
        dyn_cap = max(cap, int(cap * max(1.0, area**0.25)))
        return min(dyn_cap, scaled)
    area_4k = _REF_4K / ref
    scale_4k = min(7.5, area_4k**0.42)
    n4k = int(base * max(1.0, scale_4k))
    dyn_cap_4k = max(cap, int(cap * max(1.0, area_4k**0.25)))
    n4k = min(dyn_cap_4k, n4k)
    return min(int(n4k * px / _REF_4K), 2_500_000)


def render_lyapunov(w, h, sequence="BBBBBBAAAAAA", a_range=(3.4, 4.0), b_range=(3.4, 4.0)):
    a0, a1 = a_range
    b0, b1 = b_range
    mid_a = 0.5 * (a0 + a1)
    mid_b = 0.5 * (b0 + b1)
    db = 0.5 * (b1 - b0)
    da = db * (w / float(h))
    # Portrait: slide onto the high-contrast seam of the 16:9 window, not a
    # bland mid-blue slab.
    if h > w:
        mid_a = a0 + 0.62 * (a1 - a0)
    a = np.linspace(mid_a - da, mid_a + da, w)
    b = np.linspace(mid_b + db, mid_b - db, h)
    A, B = np.meshgrid(a, b)
    return color_lyapunov(lyapunov(A, B, sequence=sequence))


def render_domain(w, h, kind, half_width=2.4):
    cr, ci = _plane(w, h, 0j, half_width)
    return domain_color(cr, ci, kind=kind)


def render_map(
    w,
    h,
    step,
    palette,
    n=180_000,
    warmup=80,
    steps=220,
    seed=0,
    blur=0.75,
    percentiles=(0.4, 99.6),
    tight=False,
    init=(0.0, 0.0),
    init_sigma=0.12,
    explode=50.0,
    xlim=None,
    ylim=None,
):
    n = _n_particles(w, h, n, cap=420_000)
    fill = "cover" if h > w else "contain"
    # Probe a subset for the crop window, then stream the full particle count
    # into the density image. Storing every 8K step would be several GB.
    if xlim is not None and ylim is not None:
        xmin, xmax = float(xlim[0]), float(xlim[1])
        ymin, ymax = float(ylim[0]), float(ylim[1])
    else:
        probe = min(n, 80_000)
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
            xmin, xmax, ymin, ymax = _dense_window(xs, ys)
        else:
            xmin, xmax, ymin, ymax = _auto_window(xs, ys, percentiles=percentiles)
        del xs, ys, hs
    xmin, xmax, ymin, ymax = fit_attractor_frame(xmin, xmax, ymin, ymax, w, h, fill=fill)
    density = np.zeros((h, w), dtype=np.float64)
    hue_acc = np.zeros((h, w), dtype=np.float64)

    def _on_step(x, y, t):
        _deposit(density, hue_acc, x, y, xmin, xmax, ymin, ymax, hue_val=t)

    iterate_map(
        step,
        n,
        warmup,
        steps,
        seed=seed,
        init=init,
        init_sigma=init_sigma,
        explode=explode,
        collect=False,
        on_step=_on_step,
    )
    return tone_map_density(density, hue_acc, palette, blur=blur)


RECIPES: dict[str, callable] = {}


def recipe(name: str):
    def wrap(fn):
        def run(w, h):
            global _CURRENT_RECIPE
            prev = _CURRENT_RECIPE
            _CURRENT_RECIPE = name
            try:
                return fn(w, h)
            finally:
                _CURRENT_RECIPE = prev

        RECIPES[name] = run
        return fn

    return wrap


@recipe("mandelbrot_cosmic")
def _r1(w, h):
    return render_mandelbrot(w, h, -0.55 + 0j, 1.85, 420, "cosmic")


@recipe("mandelbrot_seahorse")
def _r2(w, h):
    return render_mandelbrot(w, h, -0.743643887037151 + 0.131825904205330j, 0.0018, 900, "ember")


@recipe("mandelbrot_spiral")
def _r3(w, h):
    return render_mandelbrot(w, h, -0.761574 - 0.0847596j, 0.00055, 1100, "aurora")


@recipe("mandelbrot_antenna")
def _r4(w, h):
    return render_mandelbrot(w, h, -0.16 + 1.04045j, 0.026, 700, "plasma")


@recipe("mandelbrot_orbit_trap")
def _r5(w, h):
    return render_orbit_trap(w, h, -0.45 + 0j, 1.7, 180, "neon", trap="rings")


@recipe("julia_rabbit")
def _r6(w, h):
    return render_mandelbrot(w, h, 0j, 1.55, 360, "twilight", kind="julia", julia_c=-0.123 + 0.745j)


@recipe("julia_swirl")
def _r7(w, h):
    return render_mandelbrot(w, h, 0j, 1.5, 400, "cosmic", kind="julia", julia_c=-0.7269 + 0.1889j)


@recipe("julia_san_marco")
def _r8(w, h):
    return render_mandelbrot(w, h, 0j, 1.65, 320, "abyss", kind="julia", julia_c=-0.75 + 0.0j)


@recipe("julia_dragon")
def _r9(w, h):
    return render_mandelbrot(w, h, 0.05j, 1.45, 480, "ember", kind="julia", julia_c=-0.8 + 0.156j)


@recipe("julia_snowflake")
def _r10(w, h):
    return render_mandelbrot(w, h, 0j, 1.5, 420, "aurora", kind="julia", julia_c=0.285 + 0.01j)


@recipe("burning_ship")
def _r11(w, h):
    return render_mandelbrot(w, h, -0.45 - 0.55j, 1.15, 380, "bronze", kind="burning_ship")


@recipe("burning_ship_cathedral")
def _r11b(w, h):
    return render_mandelbrot(w, h, -1.755 - 0.03j, 0.048, 700, "bronze", kind="burning_ship")


@recipe("phoenix")
def _r12(w, h):
    return render_mandelbrot(
        w, h, 0j, 1.35, 280, "twilight", kind="phoenix", julia_c=0.5667 + 0j, phoenix_k=-0.5
    )


@recipe("newton_cubic")
def _r13(w, h):
    return render_newton(w, h, 0j, 1.7, 3, style="cinematic")


@recipe("newton_sextic")
def _r14(w, h):
    return render_newton(w, h, 0j, 1.55, 6, style="cinematic")


@recipe("newton_cubic_dusk")
def _r13b(w, h):
    return render_newton(w, h, 0j, 1.7, 3, style="cinematic")


@recipe("newton_sextic_dusk")
def _r14b(w, h):
    return render_newton(w, h, 0j, 1.55, 6, style="cinematic")


@recipe("lyapunov_logistic")
def _r15(w, h):
    return render_lyapunov(w, h)


@recipe("domain_rational")
def _r16(w, h):
    return render_domain(w, h, "rational", half_width=2.2)


@recipe("domain_poly")
def _r17(w, h):
    return render_domain(w, h, "poly", half_width=1.8)


@recipe("clifford_gold")
def _r18(w, h):
    step = lambda x, y: clifford_step(x, y, -1.4, 1.6, 1.0, 0.7)
    return render_map(w, h, step, "bronze", n=220_000, steps=260, seed=1, blur=0.65)


@recipe("clifford_ice")
def _r19(w, h):
    step = lambda x, y: clifford_step(x, y, 1.7, 1.7, 0.6, 1.2)
    return render_map(w, h, step, "abyss", n=220_000, steps=260, seed=2, blur=0.7)


@recipe("dejong_ember")
def _r20(w, h):
    step = lambda x, y: dejong_step(x, y, -2.05, -1.15, -2.14, 1.41)
    return render_map(w, h, step, "ember", n=200_000, steps=240, seed=4, blur=0.6)


@recipe("ikeda")
def _r21(w, h):
    step = lambda x, y: ikeda_step(x, y, 0.918)
    return render_map(
        w,
        h,
        step,
        "neon",
        n=240_000,
        warmup=70,
        steps=300,
        seed=5,
        blur=0.5,
        tight=True,
        init=(0.55, -0.35),
        init_sigma=0.22,
        explode=8.0,
        xlim=(0.15, 1.20),
        ylim=(-1.10, 0.25),
    )


@recipe("ikeda_whirl")
def _r21b(w, h):
    step = lambda x, y: ikeda_step(x, y, 0.918)
    return render_map(
        w,
        h,
        step,
        "neon",
        n=280_000,
        warmup=80,
        steps=340,
        seed=5,
        blur=0.42,
        tight=True,
        init=(0.55, -0.35),
        init_sigma=0.18,
        explode=8.0,
        xlim=(0.22, 1.05),
        ylim=(-0.92, 0.12),
    )


@recipe("hopalong")
def _r22(w, h):
    step = lambda x, y: hopalong_step(x, y, 2.0, 1.0, 0.5)
    return render_map(w, h, step, "plasma", n=200_000, warmup=20, steps=200, seed=6, blur=0.5)


@recipe("gumowski_mira")
def _r23(w, h):
    step = lambda x, y: gumowski_mira_step(x, y, -0.48, 0.95)
    return render_map(w, h, step, "forest", n=180_000, warmup=30, steps=220, seed=7, blur=0.7)


@recipe("lorenz_butterfly")
def _r24(w, h):
    return lorenz_flow(w, h, palette="ember")


@recipe("flow_silk")
def _r25(w, h):
    n = _n_particles(w, h, 160_000, cap=520_000)
    return flow_field(w, h, n_particles=n, steps=360, palette="twilight", kind="silk", seed=11, wrap=False)


@recipe("flow_swirl")
def _r26(w, h):
    n = _n_particles(w, h, 100_000, cap=380_000)
    return flow_field(w, h, palette="cosmic", kind="swirl", n_particles=n, steps=320, seed=12, wrap=True)


@recipe("gray_scott_coral")
def _r27(w, h):
    return gray_scott(w, h, F=0.0545, k=0.062, steps=10000, seed=8, palette="aurora")


@recipe("gray_scott_worms")
def _r28(w, h):
    return gray_scott(w, h, F=0.046, k=0.063, steps=11000, seed=9, palette="abyss")


@recipe("cymatics")
def _r29(w, h):
    return cymatics(w, h, n_sources=8, freq=16.5, seed=14, palette="abyss")


@recipe("magnetic_pendulum")
def _r30(w, h):
    return magnetic_pendulum(w, h, n_magnets=5, steps=110, seed=3, framed="wide")


@recipe("phoenix_close")
def _r31(w, h):
    return render_mandelbrot(
        w, h, 0j, 1.05, 320, "twilight", kind="phoenix", julia_c=0.5667 + 0j, phoenix_k=-0.5
    )


@recipe("flow_silk_open")
def _r32(w, h):
    n = _n_particles(w, h, 120_000, cap=360_000)
    return flow_field(w, h, n_particles=n, palette="twilight", kind="silk", seed=11, wrap=False)


@recipe("flow_swirl_open")
def _r33(w, h):
    n = _n_particles(w, h, 140_000, cap=380_000)
    return flow_field(w, h, palette="cosmic", kind="swirl", n_particles=n, steps=340, seed=12, wrap=False)


@recipe("magnetic_pendulum_close")
def _r34(w, h):
    return magnetic_pendulum(w, h, n_magnets=5, steps=160, seed=3, framed="close")


@recipe("magnetic_pendulum_tri")
def _r35(w, h):
    return magnetic_pendulum(w, h, n_magnets=3, steps=130, seed=4, framed="wide")


@recipe("mandelbrot_elephant")
def _r36(w, h):
    return render_mandelbrot(w, h, 0.27334 + 0.00742j, 0.016, 820, "twilight")


@recipe("mandelbrot_minibrot")
def _r37(w, h):
    return render_mandelbrot(w, h, -1.25066 + 0.02012j, 0.0018, 800, "plasma")


@recipe("mandelbrot_tendril")
def _r38(w, h):
    return render_mandelbrot(w, h, -0.235125 + 0.827215j, 0.0004, 1000, "cosmic")


@recipe("mandelbrot_orbit_cross")
def _r39(w, h):
    return render_orbit_trap(w, h, -0.5 + 0j, 1.65, 160, "ember", trap="cross")


@recipe("tricorn")
def _r40(w, h):
    return render_mandelbrot(w, h, 0j, 2.0, 360, "abyss", kind="tricorn")


@recipe("multibrot3")
def _r41(w, h):
    return render_mandelbrot(w, h, -0.15 + 0j, 1.55, 280, "plasma", kind="multibrot3")


@recipe("multibrot4")
def _r42(w, h):
    return render_mandelbrot(w, h, 0j, 1.35, 240, "twilight", kind="multibrot4")


@recipe("celtic")
def _r43(w, h):
    return render_mandelbrot(w, h, -0.4 + 0j, 1.7, 340, "forest", kind="celtic")


@recipe("buffalo")
def _r44(w, h):
    return render_mandelbrot(w, h, -0.45 - 0.35j, 1.35, 360, "bronze", kind="buffalo")


@recipe("julia_cauliflower")
def _r45(w, h):
    return render_mandelbrot(w, h, 0.55 + 0.12j, 0.28, 420, "forest", kind="julia", julia_c=0.25 + 0j)


@recipe("julia_siegel")
def _r46(w, h):
    return render_mandelbrot(w, h, 0j, 1.5, 420, "neon", kind="julia", julia_c=-0.391 - 0.587j)


@recipe("julia_dendrite")
def _r47(w, h):
    return render_mandelbrot(w, h, 0j, 1.55, 380, "ember", kind="julia", julia_c=0.0 + 1.0j)


@recipe("julia_spirals")
def _r48(w, h):
    return render_mandelbrot(w, h, 0j, 1.45, 440, "cosmic", kind="julia", julia_c=-0.4 + 0.6j)


@recipe("julia_golden")
def _r49(w, h):
    return render_mandelbrot(w, h, 0j, 1.4, 480, "bronze", kind="julia", julia_c=0.355 + 0.355j)


@recipe("julia_airplane")
def _r50(w, h):
    return render_mandelbrot(w, h, 0j, 1.7, 340, "abyss", kind="julia", julia_c=-1.75 + 0j)


@recipe("newton_quartic")
def _r51(w, h):
    return render_newton(w, h, 0j, 1.6, 4, style="classic")


@recipe("newton_quintic")
def _r52(w, h):
    return render_newton(w, h, 0j, 1.55, 5, style="classic")


@recipe("newton_octic")
def _r53(w, h):
    return render_newton(w, h, 0j, 1.5, 8, style="cinematic")


@recipe("newton_shifted")
def _r54(w, h):
    cr, ci = _plane(w, h, 0j, 1.8)
    zf, iters = newton_shifted(cr, ci)
    # teal / violet / gold basins (not rainbow-flat argument coloring)
    hue = (np.angle(zf) + np.pi) / (2.0 * np.pi)
    basin = np.mod(np.round(hue * 3.0), 3.0)
    cinematic = np.array([0.55, 0.80, 0.10])
    h = cinematic[basin.astype(np.int32)]
    tmax = max(float(iters.max()), 1.0)
    t = np.clip(iters / tmax, 0.0, 1.0)
    val = 0.10 + 0.90 * (1.0 - t) ** 0.38
    val *= 0.78 + 0.22 * np.cos(iters * 1.15)
    r = np.sqrt(cr * cr + ci * ci)
    val *= 0.52 + 0.48 * np.exp(-0.16 * r)
    sat = 0.42 + 0.22 * t
    return hsv_to_rgb(h, sat, np.clip(val, 0.0, 1.0))


@recipe("lyapunov_ab")
def _r55(w, h):
    return render_lyapunov(w, h, sequence="AB", a_range=(2.0, 4.0), b_range=(2.0, 4.0))


@recipe("lyapunov_baba")
def _r56(w, h):
    return render_lyapunov(w, h, sequence="BABA", a_range=(3.0, 4.0), b_range=(3.0, 4.0))


@recipe("domain_sin")
def _r57(w, h):
    return render_domain(w, h, "sin", half_width=3.2)


@recipe("domain_exp")
def _r58(w, h):
    return render_domain(w, h, "exp_inv", half_width=1.4)


@recipe("popcorn")
def _r59(w, h):
    cr, ci = _plane(w, h, 0j, 3.2)
    return popcorn_field(cr, ci)


@recipe("clifford_fire")
def _r60(w, h):
    step = lambda x, y: clifford_step(x, y, -1.7, 1.8, -1.9, -0.4)
    return render_map(w, h, step, "ember", n=220_000, steps=260, seed=8, blur=0.65)


@recipe("dejong_ice")
def _r61(w, h):
    step = lambda x, y: dejong_step(x, y, 1.4, -2.3, 2.4, -2.1)
    return render_map(w, h, step, "abyss", n=200_000, steps=240, seed=9, blur=0.6)


@recipe("svensson")
def _r62(w, h):
    step = lambda x, y: svensson_step(x, y, 1.4, 1.56, 1.4, -6.6)
    return render_map(w, h, step, "plasma", n=200_000, warmup=40, steps=220, seed=10, blur=0.55)


@recipe("bedhead")
def _r63(w, h):
    step = lambda x, y: bedhead_step(x, y, -0.81, -0.92)
    return render_map(w, h, step, "twilight", n=180_000, warmup=50, steps=240, seed=11, blur=0.6)


@recipe("tinkerbell")
def _r64(w, h):
    step = lambda x, y: tinkerbell_step(x, y)
    # origin is a fixed point — seed on the strange attractor and crop tight
    return render_map(
        w,
        h,
        step,
        "neon",
        n=280_000,
        warmup=90,
        steps=340,
        seed=12,
        blur=0.55,
        tight=True,
        init=(-0.72, -0.64),
        init_sigma=0.07,
        explode=6.0,
    )


@recipe("fractal_dream")
def _r65(w, h):
    step = lambda x, y: fractal_dream_step(x, y, -0.9669, 2.8798, 0.765, 0.745)
    return render_map(w, h, step, "cosmic", n=200_000, steps=240, seed=13, blur=0.6)


@recipe("hopalong_alt")
def _r66(w, h):
    step = lambda x, y: hopalong_step(x, y, -1.1, 1.6, 0.4)
    return render_map(w, h, step, "aurora", n=200_000, warmup=20, steps=220, seed=14, blur=0.5)


@recipe("gumowski_alt")
def _r67(w, h):
    step = lambda x, y: gumowski_mira_step(x, y, 0.31, 1.0)
    return render_map(w, h, step, "bronze", n=180_000, warmup=30, steps=220, seed=15, blur=0.7)


@recipe("rossler")
def _r68(w, h):
    return rossler_flow(w, h, palette="aurora")


@recipe("aizawa")
def _r69(w, h):
    return aizawa_flow(w, h, palette="twilight")


@recipe("thomas")
def _r70(w, h):
    return thomas_flow(w, h, palette="abyss")


@recipe("gray_scott_spots")
def _r71(w, h):
    return gray_scott(w, h, F=0.035, k=0.065, steps=10000, seed=16, palette="plasma")


@recipe("gray_scott_maze")
def _r72(w, h):
    return gray_scott(w, h, F=0.029, k=0.057, steps=11000, seed=17, palette="cosmic")


@recipe("gray_scott_bubbles")
def _r73(w, h):
    return gray_scott(w, h, F=0.012, k=0.050, steps=12000, seed=18, palette="ember")


@recipe("cymatics_dense")
def _r74(w, h):
    return cymatics(w, h, n_sources=12, freq=28.0, seed=19, palette="neon")


@recipe("chladni")
def _r75(w, h):
    return chladni(w, h, n=5, m=3, palette="abyss")


@recipe("chladni_high")
def _r76(w, h):
    return chladni(w, h, n=7, m=5, palette="ember")


@recipe("worley_cells")
def _r77(w, h):
    return worley(w, h, n_points=42, seed=20, palette="cosmic", mode="F2-F1")


@recipe("worley_blobs")
def _r78(w, h):
    return worley(w, h, n_points=28, seed=21, palette="aurora", mode="F1")


@recipe("barnsley_fern")
def _r79(w, h):
    return barnsley_fern(w, h, palette="forest")


@recipe("cyclic_cellular")
def _r80(w, h):
    return cyclic_ca(w, h, n_states=14, steps=80, seed=22)


@recipe("sandpile")
def _r81(w, h):
    return sandpile(w, h, palette="ember")


@recipe("harmonograph")
def _r82(w, h):
    return harmonograph(w, h, palette="twilight")


@recipe("apollonian")
def _r83(w, h):
    return apollonian_dots(w, h, palette="bronze")


@recipe("kerr_einstein_rings")
def _r84(w, h):
    return kerr_black_hole(w, h)


@recipe("gray_scott_morphogenesis")
def _r85(w, h):
    return gray_scott_morphogenesis(w, h)


@recipe("physarum_dark_matter")
def _r86(w, h):
    return physarum_network(w, h)


@recipe("poincare_mirage")
def _r87(w, h):
    return poincare_mirage(w, h)


@recipe("halvorsen_ribbon")
def _r88(w, h):
    return halvorsen_ribbon(w, h)


@recipe("hydrogen_orbitals")
def _r89(w, h):
    return hydrogen_orbitals(w, h)


@recipe("ferrofluid_rosettes")
def _r90(w, h):
    return ferrofluid_rosettes(w, h)


@recipe("mandelbulb_grotto")
def _r91(w, h):
    return mandelbulb_grotto(w, h)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate math-based 4K desktop backgrounds.")
    p.add_argument("--list", action="store_true", help="List wallpaper names")
    p.add_argument("--only", nargs="+", help="Generate only these names")
    p.add_argument("--width", type=int, default=DEFAULT_W)
    p.add_argument("--height", type=int, default=DEFAULT_H)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--skip-existing", action="store_true", help="Skip wallpapers whose PNG already exists")
    args = p.parse_args(argv)

    if args.list:
        for name in RECIPES:
            print(name)
        return 0

    names = list(RECIPES)
    if args.only:
        unknown = [n for n in args.only if n not in RECIPES]
        if unknown:
            print(f"Unknown: {unknown}", file=sys.stderr)
            return 2
        names = args.only

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(names)} wallpaper(s) at {args.width}x{args.height} → {args.out}")
    for i, name in enumerate(names, 1):
        path = args.out / f"{name}.png"
        if args.skip_existing and path.exists():
            print(f"[{i}/{len(names)}] {name}  (skip existing)", flush=True)
            continue
        t0 = time.perf_counter()
        print(f"[{i}/{len(names)}] {name} ...", flush=True)
        rgb = RECIPES[name](args.width, args.height)
        save_png(str(path), rgb)
        dt = time.perf_counter() - t0
        print(f"    saved {path.name}  ({dt:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
