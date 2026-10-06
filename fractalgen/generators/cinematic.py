"""Cinematic wallpapers: GR lensing, morphogenesis, slime mold, hyperbolic glass,
depth-of-field chaos, hydrogen orbitals, ferrofluid spikes, and raymarched 3D fractals.

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
from .. import simcache
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
    # One 2D zoom per channel, in threads: a 3D zoom also spline-filters the
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


def _assoc_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    m = abs(m)
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))
        fact = 1.0
        for _i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    pll = pmmp1
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def _hydrogen_psi(n: int, l: int, m: int, r: np.ndarray, ct: np.ndarray, phi: np.ndarray) -> np.ndarray:
    rho = 2.0 * r / n
    k = n - l - 1
    alpha = 2 * l + 1
    lag = np.ones_like(rho)
    if k >= 1:
        lag = 1.0 + alpha - rho
    if k >= 2:
        l0 = np.ones_like(rho)
        l1 = 1.0 + alpha - rho
        for i in range(2, k + 1):
            l2 = ((2 * i - 1 + alpha - rho) * l1 - (i - 1 + alpha) * l0) / i
            l0, l1 = l1, l2
        lag = l1
    radial = np.exp(-0.5 * rho) * np.power(np.maximum(rho, 0.0), l) * lag
    plm = _assoc_legendre(l, m, ct)
    ang = plm * np.exp(1j * m * phi)
    return radial * ang * (2.0 / n) ** 1.5


@generator(
    "kerr_black_hole",
    category="cinematic",
    uses_view=False,
    param_docs={
        "spin": "how fast the hole spins (0..0.998); higher = frame dragging skews the shadow and inner disk",
        "camera_distance": "camera distance from the hole, in black-hole masses (shadow radius is ~5.2); lower = bigger",
        "camera_elevation": "camera height above the disk plane, degrees; ~5 = edge-on 'Interstellar' look, 60 = from above",
        "roll": "tilts the whole picture, degrees counter-clockwise",
        "fov": "field of view across the frame's shorter side, degrees; lower = zoomed in",
        "disk_inner": "inner edge of the disk in masses; None = innermost stable orbit for this spin (min 3.2)",
        "disk_outer": "outer edge of the disk in masses; higher = wider disk",
        "temperature": "peak disk temperature in kelvin; sets the color (4000 orange, 6500 white, 12000 blue)",
        "doppler": "strength of relativistic beaming (0 = none, 1 = physical): the approaching side brightens and blues",
        "turbulence": "contrast of the gas streaks in the disk (0 = smooth, 1 = very clumpy)",
        "streaks": "how fine the streaks are across the disk radius; higher = more, thinner bands",
        "disk_opacity": "how opaque the disk gas is (0..2); lower lets the far side show through the gaps",
        "stars": "fraction of sky cells holding a star (0..1); higher = denser starfield",
        "star_size": "star radius in radians of view angle (stays the same at every resolution)",
        "galaxy": "brightness of the faint galactic band behind the hole (0 = none)",
        "exposure": "overall brightness in stops (+1 doubles)",
        "saturation": "color intensity of the disk and stars (1 = physical blackbody colors)",
        "bloom": "strength of the glow around the bright disk",
        "bloom_radius": "glow radius as a fraction of image height",
        "seed": "random seed for the disk streaks and the starfield",
        "max_steps": "ray integration steps limit; rarely change (raise if the photon ring looks broken)",
        "max_megapixels": "trace at output size up to this many megapixels, then upscale; higher = sharper, slower",
    },
    ranges={
        "spin": (0.0, 0.998),
        "camera_distance": (12.0, 80.0),
        "camera_elevation": (-89.0, 89.0),
        "roll": (-180.0, 180.0),
        "fov": (5.0, 90.0),
        "disk_outer": (6.0, 40.0),
        "temperature": (2500.0, 20000.0),
        "doppler": (0.0, 1.0),
        "turbulence": (0.0, 1.0),
        "streaks": (2.0, 40.0),
        "disk_opacity": (0.1, 2.0),
        "stars": (0.0, 0.5),
        "star_size": (0.0002, 0.004),
        "galaxy": (0.0, 0.3),
        "exposure": (-3.0, 3.0),
        "saturation": (0.0, 2.5),
        "bloom": (0.0, 1.5),
        "bloom_radius": (0.002, 0.08),
        "seed": (0, 9999, 1),
        "max_steps": (200, 2000, 50),
        "max_megapixels": (0.5, 16.0),
    },
)
def kerr_black_hole(
    ctx,
    *,
    spin: float = 0.6,
    camera_distance: float = 42.0,
    camera_elevation: float = 4.0,
    roll: float = 0.0,
    fov: float = 21.0,
    disk_inner: float | None = None,
    disk_outer: float = 18.0,
    temperature: float = 4300.0,
    doppler: float = 0.4,
    turbulence: float = 0.7,
    streaks: float = 9.0,
    disk_opacity: float = 1.6,
    stars: float = 0.07,
    star_size: float = 0.0005,
    galaxy: float = 0.05,
    exposure: float = 0.3,
    saturation: float = 1.35,
    bloom: float = 0.6,
    bloom_radius: float = 0.018,
    seed: int = 3,
    max_steps: int = 700,
    max_megapixels: float = 4.0,
):
    """Black hole with a turbulent accretion disk: lensed far side, photon ring, Doppler beaming."""
    from .. import blackhole

    w, h = ctx.width, ctx.height
    # Each pixel is traced independently, so trace at output size (sharp previews)
    # up to a pixel budget; the look does not depend on the buffer size.
    k = min(1.0, math.sqrt(max_megapixels * 1e6 / (w * h)))
    sw, sh = max(16, int(w * k)), max(16, int(h * k))
    # Spin pulls the ISCO in, but the rays follow Schwarzschild lensing (photon sphere
    # r = 3), so keep the default edge just outside it to avoid slivers inside the shadow.
    r_in = float(disk_inner) if disk_inner is not None else max(float(blackhole.isco(float(spin))), 3.2)

    def _trace() -> np.ndarray:
        img, trans, esc = blackhole.trace(
            sh, sw, float(spin), float(camera_distance), math.radians(camera_elevation), math.radians(roll),
            math.radians(fov), r_in, float(disk_outer), float(temperature), float(doppler), float(turbulence),
            float(streaks), float(disk_opacity), float(stars), float(star_size), float(galaxy), int(max_steps),
            int(seed),
        )
        pix_angle = 2.0 * math.tan(0.5 * math.radians(fov)) / min(sw, sh)
        blackhole.shade_sky(img, trans, esc, pix_angle, float(stars), float(star_size), float(galaxy), int(seed))
        return img

    key = ("kerr2", sw, sh, spin, camera_distance, camera_elevation, roll, fov, r_in, disk_outer, temperature,
           doppler, turbulence, streaks, disk_opacity, stars, star_size, galaxy, max_steps, seed)
    hdr = simcache.cached(key, _trace) * (2.0 ** exposure)
    if saturation != 1.0:
        lum = (hdr @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
        hdr = np.clip(lum + saturation * (hdr - lum), 0.0, None)
    if bloom > 0:
        lum = hdr.max(axis=2)
        hot = np.clip(lum - 0.8, 0.0, None)[..., None] * (hdr / (lum[..., None] + 1e-9))
        glow = gaussian_filter(hot, sigma=(bloom_radius * sh, bloom_radius * sh, 0))
        glow += 0.5 * gaussian_filter(hot, sigma=(4 * bloom_radius * sh, 4 * bloom_radius * sh, 0))
        hdr = hdr + bloom * glow
    # ACES filmic curve (Narkowicz fit), then display gamma
    x = hdr * 0.8
    rgb = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0) ** (1 / 2.2)
    if (sw, sh) != (w, h):
        rgb = upsample(rgb, w, h, order=3)
    return np.clip(rgb, 0.0, 1.0)


@generator(
    "gray_scott_blend",
    category="cinematic",
    uses_view=False,
    param_docs={
        "phases": "list of (F, k) Gray–Scott pairs blended left-to-right",
        "steps": "simulation steps per phase",
        "seed": "RNG seed for initial u/v blobs",
        "sim_scale": "internal sim resolution as fraction of output",
        "sim_min": "minimum sim short side in pixels",
        "du": "u diffusion coefficient",
        "dv": "v diffusion coefficient",
        "dt": "simulation time step per kernel iteration",
        "blur_sigma": "Gaussian blur on each phase field (sim pixels)",
        "hip_sigma": "large blur for edge glow extraction (sim pixels)",
        "palette": "palette for histogram-colored field",
        "palette_gamma": "gamma on histogram index",
        "glow_color": "RGB tint for left-edge violet glow",
        "background_color": "RGB substrate mixed under the field",
        "field_mix": "strength of field over background",
        "glow_strength": "amplitude of hip glow toward left edge",
        "mask_threshold": "minimum field for histogram masking",
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
    """Spatial blend of Gray–Scott phases (maze → coral → spots)."""
    from .. import kernels

    w, h = ctx.width, ctx.height
    sw, sh = sim_size(ctx, sim_scale, min_s=sim_min)
    phase_fields = []
    for i, pair in enumerate(phases):
        f_val, k_val = float(pair[0]), float(pair[1])
        phase_seed = seed + i * 17

        def _phase(fv=f_val, kv=k_val, ps=phase_seed) -> np.ndarray:
            u, v = _gs_seed(sh, sw, ps)
            u, v = kernels.gray_scott_kernel(u, v, fv, kv, int(steps), du, dv, dt)
            field = np.clip(v, 0.0, 1.0)
            field = field / (field.max() + 1e-12)
            return gaussian_filter(field, sigma=blur_sigma)

        pkey = ("gs_blend_phase", sw, sh, f_val, k_val, steps, phase_seed, du, dv, dt, blur_sigma)
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
        "seed": "RNG seed for agents and food nodes",
        "sim_scale": "simulation resolution as fraction of output",
        "sim_min": "minimum sim short side in pixels",
        "agent_count": "explicit agent count (None = density × pixels)",
        "agent_min": "lower cap when auto-counting agents",
        "agent_max": "upper cap when auto-counting agents",
        "agent_density": "agents per sim pixel when agent_count is None",
        "sensor_angle": "chemotaxis sensor cone half-angle, degrees",
        "rotate_angle": "heading adjustment when turning, degrees",
        "sensor_distance": "forward sensor offset in sim pixels",
        "step": "agent movement per inner step",
        "deposit": "trail deposit per inner step",
        "food_nodes": "number of Gaussian food blobs",
        "chunks": "outer simulation chunks (trail decay between)",
        "inner_steps": "physarum kernel steps per chunk",
        "palette": "palette for log-tone trail coloring",
        "tone_gamma": "gamma on log tone before palette",
        "log_lo": "log-tone percentile low",
        "log_hi": "log-tone percentile high",
        "smoke_sigma": "large blur sigma for background mist (pixels at 4K)",
        "smoke_strength": "mist amplitude from blurred tone",
        "background_color": "RGB base for smoke and dark regions",
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
    """Jeff Jones slime-mold transport network with golden filaments."""
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
            kernels.physarum_steps(
                x, y, heading, trail, sa, ra, sensor_distance, step, deposit, inner_steps, tie_seed + c * inner_steps
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
        "p": "hyperbolic {p,q} tessellation p (sides per tile)",
        "q": "hyperbolic {p,q} tessellation q (tiles at vertex)",
        "max_fold": "maximum Poincaré disk folding iterations",
        "depth_norm": "depth scale for tile shading",
        "sector_scale": "angular frequency of sector coloring",
        "edge_sharpness": "exponential sharpness of tile edges",
        "tile_colors": "pair of RGB tile base colors",
        "metal_color": "RGB filigree on edges",
        "disk_scale": "Poincaré disk radius scale in plane coordinates",
        "rim_sigma": "Gaussian sharpness of disk boundary vignette",
        "outside_smoke": "RGB for points outside the disk",
        "rim_color": "RGB rim highlight outside the disk",
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
    """Hyperbolic {p,q} tessellation with refractive obsidian and metal filigree."""
    from .. import kernels

    w, h = ctx.width, ctx.height
    depth, edge, sector = kernels.poincare_kernel(h, w, int(p), int(q), int(max_fold))
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
        "seed": "RNG seed for initial particles and far-disk culling",
        "a": "Halvorsen coupling parameter",
        "dt": "ODE integration time step",
        "steps": "total integration steps",
        "skip": "initial steps discarded before splat deposit",
        "particle_base": "base particle count at 4K reference",
        "particle_cap": "max area-based particle multiplier",
        "yaw_deg": "camera yaw for 3D→2D projection, degrees",
        "pitch_deg": "camera pitch for 3D→2D projection, degrees",
        "frame_pad": "fractional padding on projected bounds",
        "x_percentile": "percentile range for frame width",
        "trim_percentile": "percentile trim before framing",
        "focus_percentile": "depth percentile used as focus plane",
        "aperture_scale": "circle-of-confusion scale (f-stop proxy)",
        "coc_thin": "CoC threshold for stochastic far-disk dropout",
        "coc_keep": "probability to keep splats above coc_thin",
        "palette": "palette for log-tone background fill",
        "palette_gamma": "gamma on log tone before palette",
        "tone_lo": "log-tone percentile low",
        "tone_hi": "log-tone percentile high",
        "blur_sigma": "Gaussian blur sigma on splat buffer (pixels at 4K)",
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
    """Halvorsen 3-fold attractor with synthetic-aperture bokeh splats."""
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


@generator(
    "hydrogen_orbitals",
    category="cinematic",
    uses_view=False,
    param_docs={
        "terms": "list of (n, l, m, amplitude, phase_deg) eigenstate terms",
        "sim_scale": "march buffer as fraction of output size",
        "sim_min": "minimum march buffer short side in pixels",
        "grid_size": "3D density grid side (None = 4K reference default)",
        "extent": "physical cube half-extent for ψ grid",
        "density_lo": "lower percentile for density normalization",
        "density_hi": "upper percentile for density normalization",
        "density_gamma": "gamma on normalized density before march",
        "camera_position": "(x, y, z) camera position for ray march",
        "camera_target": "(x, y, z) look-at point",
        "tone_denom": "Reinhard denominator on march output",
        "output_mul": "linear gain before output gamma",
        "output_gamma": "final display gamma",
    },
    ranges={
        "sim_scale": (0.25, 0.6),
        "sim_min": (160, 400, 20),
        "extent": (10.0, 40.0),
        "density_lo": (70.0, 95.0),
        "density_hi": (95.0, 99.95),
        "density_gamma": (0.8, 2.5),
        "tone_denom": (0.3, 0.9),
        "output_mul": (0.8, 1.6),
        "output_gamma": (0.6, 1.2),
    },
)
def hydrogen_orbitals(
    ctx,
    *,
    terms: tuple = (
        (4, 3, 0, 0.95, 0.0),
        (3, 2, 1, 0.55, 45.0),
        (3, 2, -1, 0.55, -36.0),
        (4, 2, 0, 0.40, 0.0),
    ),
    sim_scale: float = 0.42,
    sim_min: int = 240,
    grid_size: int | None = None,
    extent: float = 22.0,
    density_lo: float = 88.0,
    density_hi: float = 99.85,
    density_gamma: float = 1.55,
    camera_position: tuple = (26.0, 14.0, 34.0),
    camera_target: tuple = (0.0, 0.8, 0.0),
    tone_denom: float = 0.55,
    output_mul: float = 1.20,
    output_gamma: float = 0.90,
):
    """Volume-render a superposition of hydrogen eigenstates."""
    from .. import kernels

    w, h = ctx.width, ctx.height
    sw, sh, _ = render_size(ctx, sim_scale, min_s=sim_min)
    ref_w, _ = ctx.ref_size
    ngrid = grid_size if grid_size is not None else (96 if ref_w < 5000 else 112)

    def _march() -> np.ndarray:
        ax = np.linspace(-extent, extent, ngrid)
        xg, yg, zg = np.meshgrid(ax, ax, ax, indexing="ij")
        r = np.sqrt(xg * xg + yg * yg + zg * zg) + 1e-9
        ct = np.clip(zg / r, -1.0, 1.0)
        phi = np.arctan2(yg, xg)
        psi = np.zeros_like(r, dtype=np.complex128)
        for term in terms:
            n, l, m, amp, phase_deg = term
            phase = math.radians(float(phase_deg))
            psi += float(amp) * np.exp(1j * phase) * _hydrogen_psi(int(n), int(l), int(m), r, ct, phi)
        dens = np.abs(psi) ** 2
        pos = dens[dens > 0]
        lo, hi = np.percentile(pos, [density_lo, density_hi]) if pos.size else (0.0, 1.0)
        dens = np.clip((dens - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        dens = np.power(dens, density_gamma)
        phase = np.cos(np.angle(psi))
        cx, cy, cz = camera_position
        tx, ty, tz = camera_target
        raw = kernels.volume_march_kernel(
            sh, sw, dens, phase, float(extent), float(cx), float(cy), float(cz), float(tx), float(ty), float(tz)
        )
        return np.clip(raw, 0.0, None)

    mkey = (
        "hydrogen",
        sw,
        sh,
        ngrid,
        extent,
        terms,
        density_lo,
        density_hi,
        density_gamma,
        camera_position,
        camera_target,
    )
    raw = simcache.cached(mkey, _march)
    raw = raw / (tone_denom + raw)
    rgb = upsample(raw, w, h, order=3)
    rgb = np.power(np.clip(rgb * output_mul, 0.0, 1.0), output_gamma)
    return rgb


@generator(
    "ferrofluid",
    category="cinematic",
    uses_view=False,
    param_docs={
        "spacing": "hexagonal spike lattice spacing",
        "h0": "base spike height scale",
        "camera_position": "(x, y, z) studio camera position",
        "camera_target": "(x, y, z) look-at point",
        "sim_scale": "internal render as fraction of output",
        "sim_min": "minimum internal short side in pixels",
        "bloom_threshold": "luminance threshold before bloom",
        "bloom_sigma": "glow radius around highlights, in render-buffer pixels at 4K; higher = softer glow",
        "bloom_strength": "RGB bloom additive strength",
        "tone_denom": "Reinhard tone-map denominator",
    },
    ranges={
        "spacing": (0.1, 0.3),
        "h0": (0.1, 0.5),
        "sim_scale": (0.3, 0.75),
        "sim_min": (200, 480, 20),
        "bloom_threshold": (0.5, 0.9),
        "bloom_sigma": (0.5, 4.0),
        "bloom_strength": (0.1, 0.7),
        "tone_denom": (0.1, 0.5),
    },
)
def ferrofluid(
    ctx,
    *,
    spacing: float = 0.185,
    h0: float = 0.28,
    camera_position: tuple = (0.0, 1.45, -2.15),
    camera_target: tuple = (0.0, 0.08, 0.20),
    sim_scale: float = 0.52,
    sim_min: int = 280,
    bloom_threshold: float = 0.75,
    bloom_sigma: float = 1.8,
    bloom_strength: float = 0.35,
    tone_denom: float = 0.25,
):
    """Rosensweig hexagonal spike field with glossy studio lighting."""
    from .. import kernels

    w, h = ctx.width, ctx.height
    sw, sh, k = render_size(ctx, sim_scale, min_s=sim_min)
    cx, cy, cz = camera_position
    tx, ty, tz = camera_target

    def _render() -> np.ndarray:
        raw = kernels.ferrofluid_kernel(
            sh, sw, float(spacing), float(h0), float(cx), float(cy), float(cz), float(tx), float(ty), float(tz)
        )
        raw = np.clip(raw, 0.0, None)
        lum = raw.max(axis=2)
        bloom = gaussian_filter(np.clip(lum - bloom_threshold, 0.0, None), sigma=bloom_sigma * k)
        return raw + bloom[..., None] * bloom_strength

    fkey = (
        "ferrofluid",
        sw,
        sh,
        spacing,
        h0,
        camera_position,
        camera_target,
        bloom_threshold,
        bloom_sigma,
        bloom_strength,
    )
    raw = simcache.cached(fkey, _render)
    rgb = upsample(raw, w, h, order=3)
    rgb = rgb / (1.0 + tone_denom * rgb)
    return np.clip(rgb, 0.0, 1.0)


@generator(
    "raymarch_fractal",
    category="cinematic",
    choices={"kind": ["mandelbulb", "quaternion_julia"]},
    param_docs={
        "kind": "mandelbulb power map or quaternion Julia set",
        "power": "Mandelbulb power (ignored for quaternion Julia)",
        "camera_position": "(x, y, z) ray origin",
        "camera_target": "(x, y, z) look-at point",
        "max_steps": "maximum ray-march steps",
        "max_iter": "fractal iteration count per distance estimate",
        "julia_constant": "five quaternion components for Julia mode",
        "sim_scale": "internal march as fraction of output",
        "sim_min": "minimum internal short side in pixels",
        "exposure": "linear exposure before gamma",
        "gamma": "output display gamma",
    },
    ranges={
        "power": (4.0, 12.0),
        "max_steps": (40, 150, 5),
        "max_iter": (4, 16, 1),
        "sim_scale": (0.3, 0.65),
        "sim_min": (200, 480, 20),
        "exposure": (1.0, 4.0),
        "gamma": (0.5, 1.2),
    },
)
def raymarch_fractal(
    ctx,
    *,
    kind: str = "mandelbulb",
    power: float = 8.0,
    camera_position: tuple = (0.95, 0.28, -1.15),
    camera_target: tuple = (0.18, 0.02, 0.12),
    max_steps: int = 90,
    max_iter: int = 9,
    julia_constant: tuple = (0.0, 0.0, 0.0, 0.0, 0.0),
    sim_scale: float = 0.46,
    sim_min: int = 280,
    exposure: float = 2.55,
    gamma: float = 0.86,
):
    """Raymarched Mandelbulb or quaternion Julia with weathered bronze shading."""
    from .. import kernels

    w, h = ctx.width, ctx.height
    sw, sh, _ = render_size(ctx, sim_scale, min_s=sim_min)
    kind_id = 0 if kind == "mandelbulb" else 1
    cx, cy, cz = camera_position
    tx, ty, tz = camera_target
    jc = np.array(julia_constant, dtype=np.float64)
    if jc.size < 5:
        jc = np.resize(jc, 5)

    def _march() -> np.ndarray:
        return np.clip(
            kernels.fractal3d_kernel(
                sh,
                sw,
                kind_id,
                float(cx),
                float(cy),
                float(cz),
                float(tx),
                float(ty),
                float(tz),
                int(max_steps),
                int(max_iter),
                float(power),
                jc,
            ),
            0.0,
            None,
        )

    rkey = (
        "raymarch_fractal",
        sw,
        sh,
        kind,
        power,
        camera_position,
        camera_target,
        max_steps,
        max_iter,
        tuple(jc.tolist()),
    )
    raw = simcache.cached(rkey, _march)
    rgb = upsample(raw, w, h, order=3)
    rgb = np.power(np.clip(rgb * exposure, 0.0, 1.0), gamma)
    return rgb
