"""Cinematic wallpapers: morphogenesis, slime mold, hyperbolic glass, and
depth-of-field chaos.

Each generator runs heavy work at a reduced resolution (``sim_size``) then upsamples.
Add a recipe in ``data/recipes/cinematic.toml`` pointing at a generator name and tune
``params``; extend the family by copying a generator block and wiring a new kernel call.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from ..color import histogram_index, log_tone
from ..palette import sample_palette
from .. import progress, simcache
from ..view import angle, rotate_yaw_pitch
from .base import generator

_REF_4K_AREA = 3840.0 * 2160.0


def sim_size(ctx, scale: float, min_s: int = 160) -> tuple[int, int]:
    """Simulation grid from the 4K reference frame (not output width)."""
    rw, rh = ctx.ref_size
    sw = max(min_s, int(rw * scale))
    sh = max(max(64, min_s * rh // max(rw, 1)), int(rh * scale))
    return sw, sh


def render_size(ctx, scale: float, min_s: int = 160) -> tuple[int, int, float]:
    """Buffer for per-pixel renders (ray tracing / marching): proportional to the
    output, since each pixel is computed independently. Also returns the buffer's
    width relative to the 4K one, to convert lengths given in buffer pixels at 4K.
    """
    w, h = ctx.width, ctx.height
    sw = max(min_s, int(w * scale))
    sh = max(max(64, min_s * h // max(w, 1)), int(h * scale))
    return sw, sh, sw / float(sim_size(ctx, scale, min_s)[0])


def upsample(field: np.ndarray, width: int, height: int, order: int = 3) -> np.ndarray:
    z = (height / field.shape[0], width / field.shape[1])
    if field.ndim == 2:
        return zoom(field, z, order=order)[:height, :width]
    # Run a 2D zoom per channel in threads, since a full 3D zoom also spline-filters the
    # 3-long channel axis, which is ~20x slower for the same result.
    from ..parallel import _executor

    chans = list(_executor().map(lambda c: zoom(field[..., c], z, order=order), range(field.shape[2])))
    return np.stack(chans, axis=-1)[:height, :width]


def downsample_area(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Shrink a float image by area averaging (per channel)."""
    from PIL import Image

    chans = img[..., None] if img.ndim == 2 else img
    out = [
        np.asarray(Image.fromarray(chans[..., c].astype(np.float32), mode="F").resize((width, height), Image.BOX))
        for c in range(chans.shape[2])
    ]
    res = np.stack(out, axis=-1).astype(np.float64)
    return res[..., 0] if img.ndim == 2 else res


def _gs_seed(sh: int, sw: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = np.ones((sh, sw), dtype=np.float64)
    v = np.zeros((sh, sw), dtype=np.float64)
    for _ in range(16):
        cy = int(rng.integers(sh // 10, 9 * sh // 10))
        cx = int(rng.integers(sw // 10, 9 * sw // 10))
        ry = int(rng.integers(3, max(4, sh // 18)))
        rx = int(rng.integers(3, max(4, sw // 18)))
        yy, xx = np.ogrid[:sh, :sw]
        blob = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        u[blob] = 0.50
        v[blob] = 0.25
    v += 0.02 * rng.random((sh, sw))
    return np.clip(u, 0, 1), np.clip(v, 0, 1)


@generator(
    "gray_scott_blend",
    category="cinematic",
    uses_view=False,
    param_docs={
        "phases": "The pattern recipes, blended from left to right.",
        "steps": "How long each pattern grows.",
        "seed": "Change it for a different starting pattern.",
        "sim_scale": "Resolution of the simulation compared with the picture.",
        "sim_min": "The smallest the simulation grid is allowed to get.",
        "du": "How fast the first chemical spreads.",
        "dv": "How fast the second chemical spreads.",
        "dt": "Simulation time step.",
        "blur_sigma": "Softens each pattern.",
        "hip_sigma": "Size of the glow along the left edge.",
        "palette": "Colors for the pattern.",
        "palette_gamma": "Shifts the colors toward the dark or bright end.",
        "glow_color": "Color of the glow along the left edge.",
        "background_color": "Color underneath the pattern.",
        "field_mix": "How strongly the pattern shows over the background.",
        "glow_strength": "How bright the edge glow is.",
        "mask_threshold": "Faint parts of the pattern below this are left out.",
    },
    ranges={
        "steps": (2000, 15_000, 500),
        "seed": (0, 9999, 1),
        "sim_scale": (0.15, 0.5),
        "sim_min": (120, 400, 20),
        "du": (0.05, 0.3),
        "dv": (0.02, 0.2),
        "dt": (0.5, 2.0),
        "blur_sigma": (0.1, 2.0),
        "hip_sigma": (0.5, 5.0),
        "palette_gamma": (0.4, 1.5),
        "field_mix": (0.5, 3.0),
        "glow_strength": (0.2, 1.2),
        "mask_threshold": (0.0, 0.05),
    },
)
def gray_scott_blend(
    ctx,
    *,
    phases: tuple = ((0.029, 0.057), (0.0545, 0.062), (0.037, 0.065)),
    steps: int = 9000,
    seed: int = 21,
    sim_scale: float = 0.30,
    sim_min: int = 200,
    du: float = 0.16,
    dv: float = 0.08,
    dt: float = 1.0,
    blur_sigma: float = 0.55,
    hip_sigma: float = 2.2,
    palette: Any = "biolum",
    palette_gamma: float = 0.82,
    glow_color: tuple = (0.42, 0.18, 0.95),
    background_color: tuple = (0.010, 0.018, 0.045),
    field_mix: float = 1.65,
    glow_strength: float = 0.70,
    mask_threshold: float = 0.015,
):
    """Three reaction-diffusion patterns flowing into each other: maze, coral, then spots."""
    from .. import kernels

    if len(phases) != 3:
        raise ValueError("gray_scott_blend requires exactly three phases")
    w, h = ctx.width, ctx.height
    sw, sh = sim_size(ctx, sim_scale, min_s=sim_min)
    phase_fields = []
    for i, pair in enumerate(phases):
        f_val, k_val = float(pair[0]), float(pair[1])
        phase_seed = seed + i * 17

        def _phase(fv=f_val, kv=k_val, ps=phase_seed) -> np.ndarray:
            u, v = _gs_seed(sh, sw, ps)
            u, v = kernels.gray_scott_kernel(u, v, fv, kv, int(steps), du, dv, dt, progress.units(steps))
            field = np.clip(v, 0.0, 1.0)
            field = field / (field.max() + 1e-12)
            return gaussian_filter(field, sigma=blur_sigma)

        pkey = ("gs_blend_phase", sw, sh, f_val, k_val, steps, phase_seed, du, dv, dt, blur_sigma)
        with progress.span(i / len(phases), (i + 1) / len(phases)):
            phase_fields.append(simcache.cached(pkey, _phase))
    xs = np.linspace(0.0, 1.0, sw)
    w0 = np.clip(1.0 - xs / 0.45, 0.0, 1.0)
    w0 = w0 * w0 * (3.0 - 2.0 * w0)
    w2 = np.clip((xs - 0.55) / 0.45, 0.0, 1.0)
    w2 = w2 * w2 * (3.0 - 2.0 * w2)
    w1 = np.clip(1.0 - w0 - w2, 0.0, 1.0)
    field = phase_fields[0] * w0[None, :] + phase_fields[1] * w1[None, :] + phase_fields[2] * w2[None, :]
    hip = np.abs(field - gaussian_filter(field, sigma=hip_sigma))
    field = upsample(field, w, h)
    hip = upsample(hip, w, h)
    t = histogram_index(field, field > mask_threshold)
    rgb = sample_palette(palette, np.power(t, palette_gamma))
    xx = np.linspace(0.0, 1.0, w)
    xw = np.broadcast_to(xx, (h, w))
    violet = np.array(glow_color, dtype=np.float64)
    rgb = rgb + (hip * (1.0 - xw) * glow_strength)[..., None] * violet
    bg = np.array(background_color, dtype=np.float64)
    mix = np.clip(field * field_mix, 0.0, 1.0)[..., None]
    rgb = bg * (1.0 - mix) + rgb * mix
    return np.clip(rgb, 0.0, 1.0)


@generator(
    "physarum",
    category="cinematic",
    uses_view=False,
    param_docs={
        "seed": "Change it for a different network.",
        "sim_scale": "Resolution of the simulation compared with the picture.",
        "sim_min": "The smallest the simulation grid is allowed to get.",
        "agent_count": "Exact number of slime-mold cells. Leave empty to size it from the picture.",
        "agent_min": "Fewest cells when sized automatically.",
        "agent_max": "Most cells when sized automatically.",
        "agent_density": "Cells per simulation pixel when sized automatically.",
        "sensor_angle": "How wide each cell looks ahead, in degrees.",
        "rotate_angle": "How sharply a cell turns toward food, in degrees.",
        "sensor_distance": "How far ahead each cell looks.",
        "step": "How far a cell moves each step.",
        "deposit": "How much trail each cell leaves behind.",
        "food_nodes": "Number of food sources the network grows between.",
        "chunks": "Growth rounds. The trails fade a little between rounds.",
        "inner_steps": "Steps in each growth round.",
        "palette": "Colors for the trails.",
        "tone_gamma": "Lower values bring out the faint trails.",
        "log_lo": "Trails fainter than this stay dark.",
        "log_hi": "Trails brighter than this are full brightness.",
        "smoke_sigma": "Size of the misty haze behind the network.",
        "smoke_strength": "How strong the haze is.",
        "background_color": "Color of the dark areas and the haze.",
    },
    ranges={
        "seed": (0, 9999, 1),
        "sim_scale": (0.3, 0.85),
        "sim_min": (200, 512, 20),
        "agent_min": (50_000, 400_000, 10_000),
        "agent_max": (500_000, 2_500_000, 50_000),
        "agent_density": (0.2, 1.2),
        "sensor_angle": (5.0, 45.0),
        "rotate_angle": (10.0, 60.0),
        "sensor_distance": (2.0, 15.0),
        "step": (0.5, 2.0),
        "deposit": (0.3, 1.5),
        "food_nodes": (10, 80, 2),
        "chunks": (4, 32, 2),
        "inner_steps": (5, 40, 1),
        "tone_gamma": (0.3, 1.2),
        "log_lo": (2.0, 20.0),
        "log_hi": (90.0, 99.9),
        "smoke_sigma": (2.0, 15.0),
        "smoke_strength": (0.05, 0.5),
    },
)
def physarum(
    ctx,
    *,
    seed: int = 7,
    sim_scale: float = 0.62,
    sim_min: int = 360,
    agent_count: int | None = None,
    agent_min: int = 180_000,
    agent_max: int = 1_600_000,
    agent_density: float = 0.55,
    sensor_angle: float = 22.5,
    rotate_angle: float = 35.0,
    sensor_distance: float = 7.0,
    step: float = 1.05,
    deposit: float = 0.85,
    food_nodes: int = 36,
    chunks: int = 16,
    inner_steps: int = 20,
    palette: Any = "goldsmoke",
    tone_gamma: float = 0.62,
    log_lo: float = 8.0,
    log_hi: float = 99.7,
    smoke_sigma: float = 7.0,
    smoke_strength: float = 0.22,
    background_color: tuple = (0.035, 0.032, 0.030),
):
    """A slime mold growing a glowing network between food sources."""
    from .. import kernels

    w, h = ctx.width, ctx.height
    sw, sh = sim_size(ctx, sim_scale, min_s=sim_min)
    px = sw * sh
    n_agents = agent_count
    if n_agents is None:
        n_agents = int(min(agent_max, max(agent_min, px * agent_density)))

    def _simulate() -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, sw, n_agents)
        y = rng.uniform(0.0, sh, n_agents)
        heading = rng.uniform(0.0, 2.0 * np.pi, n_agents)
        trail = np.zeros((sh, sw), dtype=np.float64)
        food = np.zeros((sh, sw), dtype=np.float64)
        yy, xx = np.ogrid[:sh, :sw]
        for _ in range(int(food_nodes)):
            cx = rng.uniform(0.08 * sw, 0.92 * sw)
            cy = rng.uniform(0.08 * sh, 0.92 * sh)
            sig = rng.uniform(0.018, 0.045) * min(sw, sh)
            food += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig * sig))
        food *= 2.8
        sa = math.radians(float(sensor_angle))
        ra = math.radians(float(rotate_angle))
        tie_seed = int(rng.integers(1, 2**31 - 1))
        for c in range(int(chunks)):
            trail += food * 0.05
            with progress.span(c / chunks, (c + 1) / chunks):
                kernels.physarum_steps(
                    x, y, heading, trail, sa, ra, sensor_distance, step, deposit, inner_steps,
                    tie_seed + c * inner_steps, progress.units(inner_steps),
                )
            trail = gaussian_filter(trail, sigma=0.40) * 0.88
        return gaussian_filter(trail, sigma=0.28)

    pkey = (
        "physarum",
        sw,
        sh,
        seed,
        n_agents,
        food_nodes,
        chunks,
        inner_steps,
        sensor_angle,
        rotate_angle,
        sensor_distance,
        step,
        deposit,
    )
    trail = simcache.cached(pkey, _simulate)
    field = upsample(trail, w, h, order=3)
    tone = log_tone(field, lo=log_lo, hi=log_hi)
    tone = np.power(tone, tone_gamma)
    rgb = sample_palette(palette, tone)
    smoke = gaussian_filter(tone, sigma=ctx.ref_px(smoke_sigma)) * smoke_strength
    bg = np.array(background_color, dtype=np.float64)
    mist = bg + smoke[..., None] * np.array([0.10, 0.08, 0.06])
    rgb = mist + rgb * (0.15 + 1.25 * tone)[..., None]
    return np.clip(rgb, 0.0, 1.0)


@generator(
    "hyperbolic_tiling",
    category="cinematic",
    uses_view=False,
    param_docs={
        "p": "Sides on each tile.",
        "q": "Tiles meeting at each corner.",
        "max_fold": "How far toward the rim the tiling is drawn.",
        "depth_norm": "How quickly tiles darken toward the rim.",
        "sector_scale": "How many color sectors go around the disc.",
        "edge_sharpness": "How crisp the tile edges are.",
        "tile_colors": "The two tile colors.",
        "metal_color": "Color of the metal along the edges.",
        "disk_scale": "Size of the disc in the frame.",
        "rim_sigma": "How sharply the disc fades at its rim.",
        "outside_smoke": "Color outside the disc.",
        "rim_color": "Color of the glow around the rim.",
    },
    ranges={
        "p": (5, 12, 1),
        "q": (3, 8, 1),
        "max_fold": (20, 150, 5),
        "depth_norm": (4.0, 24.0),
        "sector_scale": (0.2, 1.5),
        "edge_sharpness": (10.0, 60.0),
        "disk_scale": (0.8, 1.3),
        "rim_sigma": (500.0, 8000.0),
    },
)
def hyperbolic_tiling(
    ctx,
    *,
    p: int = 7,
    q: int = 3,
    max_fold: int = 80,
    depth_norm: float = 12.0,
    sector_scale: float = 0.55,
    edge_sharpness: float = 36.0,
    tile_colors: tuple = ((0.07, 0.20, 0.24), (0.18, 0.07, 0.20)),
    metal_color: tuple = (0.82, 0.74, 0.48),
    disk_scale: float = 1.06,
    rim_sigma: float = 4000.0,
    outside_smoke: tuple = (0.035, 0.030, 0.036),
    rim_color: tuple = (0.20, 0.18, 0.14),
):
    """An endless hyperbolic tiling inside a disc, in polished stone and metal."""
    from .. import kernels

    w, h = ctx.width, ctx.height
    depth, edge, sector = kernels.poincare_kernel(h, w, int(p), int(q), int(max_fold), progress.units(h))
    outside = depth < 0
    depth = np.where(outside, 0.0, depth)
    t = np.clip(depth / depth_norm, 0.0, 1.0)
    sec = 0.5 + 0.5 * np.sin(sector * sector_scale)
    teal = np.array(tile_colors[0], dtype=np.float64)
    plum = np.array(tile_colors[1], dtype=np.float64)
    mix = sec[..., None]
    base = teal * (1.0 - mix) + plum * mix
    base = base * (0.45 + 0.55 * (1.0 - 0.65 * t))[..., None]
    line = np.exp(-np.abs(edge) * edge_sharpness)
    line = np.where(outside, 0.0, line)
    metal = np.array(metal_color, dtype=np.float64) * (0.75 + 0.25 * sec)[..., None]
    rgb = base + line[..., None] * metal * 1.15
    rgb += (line**3)[..., None] * np.array([0.25, 0.22, 0.14])
    yy, xx = np.ogrid[:h, :w]
    aspect = w / float(h)
    u = (2.0 * xx / max(w - 1, 1) - 1.0) * aspect
    v = 1.0 - 2.0 * yy / max(h - 1, 1)
    scale = disk_scale * (aspect if aspect < 1.0 else 1.0)
    rr = (u / scale) ** 2 + (v / scale) ** 2
    rim = np.exp(-((np.sqrt(rr) - 0.995) ** 2) * rim_sigma)
    smoke = np.array(outside_smoke, dtype=np.float64)
    rgb = np.where(outside[..., None], smoke + rim[..., None] * np.array(rim_color), rgb)
    return np.clip(rgb, 0.0, 1.0)


@generator(
    "bokeh_attractor",
    category="cinematic",
    uses_view=False,
    param_docs={
        "seed": "Change it for a different cloud of particles.",
        "a": "Shapes the attractor's three arms.",
        "dt": "Simulation time step.",
        "steps": "How long each particle travels.",
        "skip": "Early steps left undrawn so stray starting points don't show.",
        "particle_base": "Particle count at 4K.",
        "particle_cap": "Limits how many particles larger pictures add.",
        "yaw_deg": "Turns the camera left or right, in degrees.",
        "pitch_deg": "Tilts the camera up or down, in degrees.",
        "frame_pad": "Space around the shape.",
        "x_percentile": "How much of the shape's width fills the frame.",
        "trim_percentile": "Ignores this many outlying points when framing.",
        "focus_percentile": "Where the focus sits, from the nearest particles to the farthest.",
        "aperture_scale": "How blurry out-of-focus particles get, like a wider lens opening.",
        "coc_thin": "Very blurry particles past this are thinned out.",
        "coc_keep": "How many of those very blurry particles are kept.",
        "palette": "Colors for the background glow.",
        "palette_gamma": "Lower values brighten the background glow.",
        "tone_lo": "Dimmest part of the glow.",
        "tone_hi": "Brightest part of the glow.",
        "blur_sigma": "Softens the whole picture slightly.",
    },
    ranges={
        "seed": (0, 9999, 1),
        "a": (1.2, 2.2),
        "dt": (0.001, 0.01),
        "steps": (1000, 6000, 100),
        "skip": (50, 500, 10),
        "particle_base": (1000, 8000, 200),
        "particle_cap": (1.5, 5.0),
        "yaw_deg": (0.0, 60.0),
        "pitch_deg": (0.0, 45.0),
        "frame_pad": (0.0, 0.2),
        "focus_percentile": (10.0, 60.0),
        "aperture_scale": (2.0, 12.0),
        "coc_thin": (1.0, 10.0),
        "coc_keep": (0.05, 0.5),
        "palette_gamma": (0.4, 1.2),
        "tone_lo": (1.0, 10.0),
        "tone_hi": (90.0, 99.9),
        "blur_sigma": (0.0, 1.5),
    },
)
def bokeh_attractor(
    ctx,
    *,
    seed: int = 4,
    a: float = 1.89,
    dt: float = 0.0035,
    steps: int = 2800,
    skip: int = 240,
    particle_base: int = 3200,
    particle_cap: float = 3.2,
    yaw_deg: float = math.degrees(0.72),
    pitch_deg: float = math.degrees(0.38),
    frame_pad: float = 0.08,
    x_percentile: tuple = (0.8, 99.2),
    trim_percentile: tuple = (1.5, 98.5),
    focus_percentile: float = 32.0,
    aperture_scale: float = 6.2,
    coc_thin: float = 4.0,
    coc_keep: float = 0.22,
    palette: Any = "mercury",
    palette_gamma: float = 0.72,
    tone_lo: float = 3.0,
    tone_hi: float = 99.5,
    blur_sigma: float = 0.35,
):
    """A three-armed Halvorsen attractor, photographed with a shallow depth of field."""
    from .. import kernels

    w, h = ctx.width, ctx.height
    rw, rh = ctx.ref_size

    def _particle_count(c) -> int:
        area = max(1.0, c.pixels / _REF_4K_AREA)
        return int(particle_base * min(particle_cap, area**0.4))

    n = _particle_count(ctx.reference())

    def _splat_img() -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.normal(-1.2, 0.35, n)
        y = rng.normal(0.4, 0.35, n)
        z = rng.normal(0.2, 0.35, n)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        zs: list[np.ndarray] = []
        for i in range(int(steps)):
            progress.update(i / steps)
            dx = -a * x - 4.0 * y - 4.0 * z - y * y
            dy = -a * y - 4.0 * z - 4.0 * x - z * z
            dz = -a * z - 4.0 * x - 4.0 * y - x * x
            x = x + dx * dt
            y = y + dy * dt
            z = z + dz * dt
            if i > skip and i % 2 == 0:
                xs.append(x.copy())
                ys.append(y.copy())
                zs.append(z.copy())
        x_arr = np.concatenate(xs)
        y_arr = np.concatenate(ys)
        z_arr = np.concatenate(zs)
        yaw = angle(yaw_deg)
        pitch = angle(pitch_deg)
        x1, y1, z2 = rotate_yaw_pitch(x_arr, y_arr, z_arr, yaw, pitch)
        finite = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(z2)
        x1, y1, z2 = x1[finite], y1[finite], z2[finite]
        y_src = y_arr[finite]
        x_src = x_arr[finite]
        lo, hi = np.percentile(x1, list(trim_percentile))
        m = (x1 > lo) & (x1 < hi)
        x1, y1, z2 = x1[m], y1[m], z2[m]
        y_src, x_src = y_src[m], x_src[m]
        xmin, xmax = np.percentile(x1, list(x_percentile))
        ymin, ymax = np.percentile(y1, list(x_percentile))
        dx_, dy_ = xmax - xmin, ymax - ymin
        xmin -= dx_ * frame_pad
        xmax += dx_ * frame_pad
        ymin -= dy_ * frame_pad
        ymax += dy_ * frame_pad
        aspect = rw / float(rh)
        if (xmax - xmin) / (ymax - ymin) > aspect:
            extra = (xmax - xmin) / aspect - (ymax - ymin)
            ymin -= extra / 2
            ymax += extra / 2
        else:
            extra = (ymax - ymin) * aspect - (xmax - xmin)
            xmin -= extra / 2
            xmax += extra / 2
        px = (x1 - xmin) / (xmax - xmin) * (rw - 1)
        py = (ymax - y1) / (ymax - ymin) * (rh - 1)
        zf = float(np.percentile(z2, focus_percentile))
        aperture = float(aperture_scale)
        coc = aperture * np.abs(z2 - zf)
        keep = np.ones(px.shape[0], dtype=np.float64)
        far = coc > coc_thin
        if np.any(far):
            keep[far] = (rng.random(int(far.sum())) < coc_keep).astype(np.float64)
        near = np.clip(1.0 - np.abs(z2 - zf) / (np.std(z2) * 2.4 + 1e-6), 0.0, 1.0)
        ang = (np.arctan2(y_src, x_src) + np.pi) / (2.0 * np.pi)
        cr = 0.22 + 0.78 * near + 0.06 * ang
        cg = 0.26 + 0.68 * near
        cb = 0.50 + 0.42 * near
        wgt = 0.9 + 1.4 * near
        out = np.zeros((rh, rw, 3), dtype=np.float64)
        kernels.splat_bokeh(
            out,
            px.astype(np.float64),
            py.astype(np.float64),
            z2.astype(np.float64),
            zf,
            aperture,
            wgt.astype(np.float64),
            cr.astype(np.float64),
            cg.astype(np.float64),
            cb.astype(np.float64),
            keep,
        )
        return gaussian_filter(out, sigma=(blur_sigma, blur_sigma, 0.0))

    skey = (
        "bokeh_splat",
        rw,
        rh,
        seed,
        a,
        dt,
        steps,
        skip,
        n,
        yaw_deg,
        pitch_deg,
        frame_pad,
        x_percentile,
        trim_percentile,
        focus_percentile,
        aperture_scale,
        coc_thin,
        coc_keep,
        blur_sigma,
    )
    img = simcache.cached(skey, _splat_img)
    if w < rw:
        img = downsample_area(img, w, h)  # average the splats, don't point-sample them
    elif rw != w or rh != h:
        img = upsample(img, w, h, order=1)
    tone = log_tone(img.max(axis=2), lo=tone_lo, hi=tone_hi)
    splat_n = img / (img.max() + 1e-12)
    rgb = sample_palette(palette, np.power(tone, palette_gamma))
    rgb = 0.25 * rgb + 0.85 * splat_n
    rgb *= (0.10 + 0.95 * np.power(tone, 0.62))[..., None]
    return np.clip(rgb, 0.0, 1.0)
