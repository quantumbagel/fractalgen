"""Cinematic wallpapers: GR, morphogenesis, slime mold, hyperbolic, DoF chaos, QM, ferrofluid, 3D fractals."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from .color import histogram_index, log_tone, sample_palette


def _sim_hw(width: int, height: int, scale: float, min_s: int = 160) -> tuple[int, int]:
    if width >= 7000:
        scale = max(scale, min(0.72, scale * 1.35))
    elif width >= 5000:
        scale = max(scale, min(0.58, scale * 1.18))
    sw = max(min_s, int(width * scale))
    sh = max(max(64, min_s * height // max(width, 1)), int(height * scale))
    return sw, sh


def _upsample(field: np.ndarray, width: int, height: int, order: int = 3) -> np.ndarray:
    zoom_y = height / field.shape[0]
    zoom_x = width / field.shape[1]
    z = (zoom_y, zoom_x) if field.ndim == 2 else (zoom_y, zoom_x, 1.0)
    out = zoom(field, z, order=order)
    if field.ndim == 2:
        return out[:height, :width]
    return out[:height, :width]


def kerr_black_hole(width: int, height: int) -> np.ndarray:
    """Geodesic raytrace of a spinning Kerr-like hole with a Doppler accretion disk."""
    sw, sh = _sim_hw(width, height, scale=0.48, min_s=320)
    from .fast import kerr_trace_kernel

    raw = kerr_trace_kernel(sh, sw, 1.0, 0.82, 14.5, 0.18, 480, 0.055)
    raw = np.clip(raw, 0.0, None)
    lum = 0.3 * raw[..., 0] + 0.5 * raw[..., 1] + 0.2 * raw[..., 2]
    hot = np.clip(lum - 0.72, 0.0, None)
    bloom = gaussian_filter(hot, sigma=1.5)
    raw = raw + bloom[..., None] * np.array([0.22, 0.40, 0.62])
    rgb = _upsample(raw, width, height, order=3)
    rgb = rgb / (1.0 + 0.45 * rgb)
    rgb = np.power(np.clip(rgb * 1.18, 0.0, 1.0), 0.90)
    return np.clip(rgb, 0.0, 1.0)


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


def gray_scott_morphogenesis(width: int, height: int, steps: int = 9000, seed: int = 21) -> np.ndarray:
    """Maze → coral → spots, from a spatial F/k blend of patterned Gray-Scott fields."""
    sw, sh = _sim_hw(width, height, scale=0.30, min_s=200)
    from .fast import gray_scott_kernel

    phases = []
    for i, (F, k) in enumerate(((0.029, 0.057), (0.0545, 0.062), (0.037, 0.065))):
        u, v = _gs_seed(sh, sw, seed + i * 17)
        u, v = gray_scott_kernel(u, v, float(F), float(k), int(steps), 0.16, 0.08, 1.0)
        field = np.clip(v, 0.0, 1.0)
        field = field / (field.max() + 1e-12)
        phases.append(gaussian_filter(field, sigma=0.55))
    xs = np.linspace(0.0, 1.0, sw)
    # smooth hermite weights for left / center / right
    w0 = np.clip(1.0 - xs / 0.45, 0.0, 1.0)
    w0 = w0 * w0 * (3.0 - 2.0 * w0)
    w2 = np.clip((xs - 0.55) / 0.45, 0.0, 1.0)
    w2 = w2 * w2 * (3.0 - 2.0 * w2)
    w1 = np.clip(1.0 - w0 - w2, 0.0, 1.0)
    field = phases[0] * w0[None, :] + phases[1] * w1[None, :] + phases[2] * w2[None, :]
    hip = np.abs(field - gaussian_filter(field, sigma=2.2))
    field = _upsample(field, width, height)
    hip = _upsample(hip, width, height)
    t = histogram_index(field, field > 0.015)
    rgb = sample_palette("biolum", np.power(t, 0.82))
    xx = np.linspace(0.0, 1.0, width)
    Xw = np.broadcast_to(xx, (height, width))
    violet = np.array([0.42, 0.18, 0.95])
    rgb = rgb + (hip * (1.0 - Xw) * 0.70)[..., None] * violet
    bg = np.array([0.010, 0.018, 0.045])
    mix = np.clip(field * 1.65, 0.0, 1.0)[..., None]
    rgb = bg * (1.0 - mix) + rgb * mix
    return np.clip(rgb, 0.0, 1.0)


def physarum_network(width: int, height: int, seed: int = 7) -> np.ndarray:
    """Jeff Jones slime-mold transport: golden filaments / cosmic web."""
    sw, sh = _sim_hw(width, height, scale=0.62, min_s=360)
    px = sw * sh
    n_agents = int(min(1_600_000, max(180_000, px * 0.55)))
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, sw, n_agents)
    y = rng.uniform(0.0, sh, n_agents)
    heading = rng.uniform(0.0, 2.0 * np.pi, n_agents)
    trail = np.zeros((sh, sw), dtype=np.float64)
    # "galaxy" attractors so the web anastomoses around nodes
    food = np.zeros((sh, sw), dtype=np.float64)
    yy, xx = np.ogrid[:sh, :sw]
    for _ in range(36):
        cx = rng.uniform(0.08 * sw, 0.92 * sw)
        cy = rng.uniform(0.08 * sh, 0.92 * sh)
        sig = rng.uniform(0.018, 0.045) * min(sw, sh)
        food += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig * sig))
    food *= 2.8
    from .fast import physarum_steps

    sa = np.deg2rad(22.5)
    ra = np.deg2rad(35.0)
    sd = 7.0
    step = 1.05
    deposit = 0.85
    chunks = 16
    inner = 20
    rng_state = np.array([rng.integers(1, 2**31 - 1)], dtype=np.uint32)
    for _ in range(chunks):
        trail += food * 0.05
        physarum_steps(x, y, heading, trail, sa, ra, sd, step, deposit, inner, rng_state)
        trail = gaussian_filter(trail, sigma=0.40) * 0.88
    trail = gaussian_filter(trail, sigma=0.28)
    field = _upsample(trail, width, height, order=3)
    tone = log_tone(field, lo=8.0, hi=99.7)
    tone = np.power(tone, 0.62)
    rgb = sample_palette("goldsmoke", tone)
    # additive filaments over charcoal smoke
    smoke = gaussian_filter(tone, sigma=7.0) * 0.22
    bg = np.array([0.035, 0.032, 0.030])
    mist = bg + smoke[..., None] * np.array([0.10, 0.08, 0.06])
    rgb = mist + rgb * (0.15 + 1.25 * tone)[..., None]
    return np.clip(rgb, 0.0, 1.0)


def poincare_mirage(width: int, height: int) -> np.ndarray:
    """{7,3} hyperbolic tessellation as refractive obsidian / platinum glass."""
    from .fast import poincare_kernel

    depth, edge, sector = poincare_kernel(height, width, 7, 3, 80)
    outside = depth < 0
    depth = np.where(outside, 0.0, depth)
    t = np.clip(depth / 12.0, 0.0, 1.0)
    sec = 0.5 + 0.5 * np.sin(sector * 0.55)
    # dark obsidian chambers, teal / plum by tile
    teal = np.array([0.07, 0.20, 0.24])
    plum = np.array([0.18, 0.07, 0.20])
    mix = sec[..., None]
    base = teal * (1.0 - mix) + plum * mix
    base = base * (0.45 + 0.55 * (1.0 - 0.65 * t))[..., None]
    # platinum / brushed-gold filigree — |edge| is distance to the folded side
    line = np.exp(-np.abs(edge) * 36.0)
    line = np.where(outside, 0.0, line)
    metal = np.array([0.82, 0.74, 0.48]) * (0.75 + 0.25 * sec)[..., None]
    rgb = base + line[..., None] * metal * 1.15
    rgb += (line**3)[..., None] * np.array([0.25, 0.22, 0.14])
    # outside the disk: smoky charcoal + faint rim
    yy, xx = np.ogrid[:height, :width]
    aspect = width / float(height)
    u = (2.0 * xx / max(width - 1, 1) - 1.0) * aspect
    v = 1.0 - 2.0 * yy / max(height - 1, 1)
    scale = 1.06 * (aspect if aspect < 1.0 else 1.0)
    rr = (u / scale) ** 2 + (v / scale) ** 2
    rim = np.exp(-((np.sqrt(rr) - 0.995) ** 2) * 4000.0)
    smoke = np.array([0.035, 0.030, 0.036])
    rgb = np.where(outside[..., None], smoke + rim[..., None] * np.array([0.20, 0.18, 0.14]), rgb)
    return np.clip(rgb, 0.0, 1.0)


def halvorsen_ribbon(width: int, height: int, seed: int = 4) -> np.ndarray:
    """Halvorsen 3-fold attractor projected with synthetic-aperture bokeh."""
    from .fast import splat_bokeh

    area = max(1.0, width * height / (3840.0 * 2160.0))
    n = int(3200 * min(3.2, area**0.4))
    rng = np.random.default_rng(seed)
    x = rng.normal(-1.2, 0.35, n)
    y = rng.normal(0.4, 0.35, n)
    z = rng.normal(0.2, 0.35, n)
    a = 1.89
    dt = 0.0035
    steps = 2800
    skip = 240
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    for i in range(steps):
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
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    Z = np.concatenate(zs)
    # camera: rotate so the 3-fold face us, then a shallow pitch
    yaw, pitch = 0.72, 0.38
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x1 = X * cy + Z * sy
    z1 = -X * sy + Z * cy
    y1 = Y * cp - z1 * sp
    z2 = Y * sp + z1 * cp
    # frame
    finite = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(z2)
    x1, y1, z2 = x1[finite], y1[finite], z2[finite]
    # drop explosive outliers
    lo, hi = np.percentile(x1, [1.5, 98.5])
    m = (x1 > lo) & (x1 < hi)
    x1, y1, z2 = x1[m], y1[m], z2[m]
    xmin, xmax = np.percentile(x1, [0.8, 99.2])
    ymin, ymax = np.percentile(y1, [0.8, 99.2])
    pad = 0.08
    dx_, dy_ = xmax - xmin, ymax - ymin
    xmin -= dx_ * pad
    xmax += dx_ * pad
    ymin -= dy_ * pad
    ymax += dy_ * pad
    aspect = width / float(height)
    if (xmax - xmin) / (ymax - ymin) > aspect:
        extra = (xmax - xmin) / aspect - (ymax - ymin)
        ymin -= extra / 2
        ymax += extra / 2
    else:
        extra = (ymax - ymin) * aspect - (xmax - xmin)
        xmin -= extra / 2
        xmax += extra / 2
    px = (x1 - xmin) / (xmax - xmin) * (width - 1)
    py = (ymax - y1) / (ymax - ymin) * (height - 1)
    zf = float(np.percentile(z2, 32.0))
    aperture = 6.2 * (width / 3840.0)
    coc = aperture * np.abs(z2 - zf)
    # keep all sharp points; thin the deep bokeh so disks stay creamy, not solid fog
    keep = np.ones(px.shape[0], dtype=np.float64)
    far = coc > 4.0
    if np.any(far):
        keep[far] = (rng.random(int(far.sum())) < 0.22).astype(np.float64)
    # mercury: near-camera ribbons go chrome, far loops go indigo
    near = np.clip(1.0 - np.abs(z2 - zf) / (np.std(z2) * 2.4 + 1e-6), 0.0, 1.0)
    ang = (np.arctan2(Y[finite][m], X[finite][m]) + np.pi) / (2.0 * np.pi)
    cr = 0.22 + 0.78 * near + 0.06 * ang
    cg = 0.26 + 0.68 * near
    cb = 0.50 + 0.42 * near
    wgt = 0.9 + 1.4 * near
    img = np.zeros((height, width, 3), dtype=np.float64)
    splat_bokeh(
        img,
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
    img = gaussian_filter(img, sigma=(0.35, 0.35, 0.0))
    tone = log_tone(img.max(axis=2), lo=3.0, hi=99.5)
    splat_n = img / (img.max() + 1e-12)
    rgb = sample_palette("mercury", np.power(tone, 0.72))
    rgb = 0.25 * rgb + 0.85 * splat_n
    rgb *= (0.10 + 0.95 * np.power(tone, 0.62))[..., None]
    return np.clip(rgb, 0.0, 1.0)


def _assoc_legendre(l: int, m: int, x: np.ndarray) -> np.ndarray:
    """Associated Legendre P_l^{|m|}(x) for l <= 5 via recurrence."""
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
    """Complex hydrogen ψ_nlm on arrays (atomic units, Z=1)."""
    # radial: e^{-ρ/2} ρ^l L_{n-l-1}^{2l+1}(ρ), ρ = 2r/n
    rho = 2.0 * r / n
    # associated Laguerre via explicit low-n polynomials
    k = n - l - 1
    alpha = 2 * l + 1
    # L_k^{α}(ρ)
    lag = np.ones_like(rho)
    if k >= 1:
        lag = 1.0 + alpha - rho
    if k >= 2:
        L0 = np.ones_like(rho)
        L1 = 1.0 + alpha - rho
        for i in range(2, k + 1):
            L2 = ((2 * i - 1 + alpha - rho) * L1 - (i - 1 + alpha) * L0) / i
            L0, L1 = L1, L2
        lag = L1
    # crude normalization (relative amplitudes only)
    radial = np.exp(-0.5 * rho) * np.power(np.maximum(rho, 0.0), l) * lag
    # angular
    plm = _assoc_legendre(l, m, ct)
    # Condon–Shortley already in P. Phase e^{imφ}
    ang = plm * np.exp(1j * m * phi)
    # n-dependent scale so higher-n states don't vanish
    return radial * ang * (2.0 / n) ** 1.5


def hydrogen_orbitals(width: int, height: int) -> np.ndarray:
    """Volume-render a superposition of excited hydrogen eigenstates."""
    sw, sh = _sim_hw(width, height, scale=0.42, min_s=240)
    ngrid = 96 if width < 5000 else 112
    extent = 22.0
    ax = np.linspace(-extent, extent, ngrid)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z) + 1e-9
    ct = np.clip(Z / r, -1.0, 1.0)
    phi = np.arctan2(Y, X)
    # hybrid high-ℓ Rydberg mix: clover + torus + hexapole
    psi = (
        0.95 * _hydrogen_psi(4, 3, 0, r, ct, phi)
        + 0.55 * np.exp(1j * np.pi / 4.0) * _hydrogen_psi(3, 2, 1, r, ct, phi)
        + 0.55 * np.exp(-1j * np.pi / 5.0) * _hydrogen_psi(3, 2, -1, r, ct, phi)
        + 0.40 * _hydrogen_psi(4, 2, 0, r, ct, phi)
    )
    dens = np.abs(psi) ** 2
    pos = dens[dens > 0]
    lo, hi = np.percentile(pos, [88.0, 99.85]) if pos.size else (0.0, 1.0)
    dens = np.clip((dens - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    dens = np.power(dens, 1.55)
    phase = np.cos(np.angle(psi))
    from .fast import volume_march_kernel

    raw = volume_march_kernel(sh, sw, dens, phase, extent, 26.0, 14.0, 34.0, 0.0, 0.8, 0.0)
    raw = np.clip(raw, 0.0, None)
    raw = raw / (0.55 + raw)
    rgb = _upsample(raw, width, height, order=3)
    rgb = np.power(np.clip(rgb * 1.20, 0.0, 1.0), 0.90)
    return rgb


def ferrofluid_rosettes(width: int, height: int) -> np.ndarray:
    """Rosensweig hexagonal spike field with glossy studio lighting."""
    sw, sh = _sim_hw(width, height, scale=0.52, min_s=280)
    from .fast import ferrofluid_kernel

    raw = ferrofluid_kernel(sh, sw, 0.185, 0.28, 0.0, 1.45, -2.15, 0.0, 0.08, 0.20)
    raw = np.clip(raw, 0.0, None)
    # tiny bloom on specular peaks
    lum = raw.max(axis=2)
    bloom = gaussian_filter(np.clip(lum - 0.75, 0.0, None), sigma=1.8)
    raw = raw + bloom[..., None] * 0.35
    rgb = _upsample(raw, width, height, order=3)
    rgb = rgb / (1.0 + 0.25 * rgb)
    return np.clip(rgb, 0.0, 1.0)


def mandelbulb_grotto(width: int, height: int) -> np.ndarray:
    """Raymarched power-8 Mandelbulb, weathered bronze and patina."""
    sw, sh = _sim_hw(width, height, scale=0.46, min_s=280)
    from .fast import fractal3d_kernel

    jc = np.zeros(5, dtype=np.float64)
    raw = fractal3d_kernel(
        sh,
        sw,
        0,
        0.95,
        0.28,
        -1.15,
        0.18,
        0.02,
        0.12,
        90,
        9,
        8.0,
        jc,
    )
    raw = np.clip(raw, 0.0, None)
    rgb = _upsample(raw, width, height, order=3)
    rgb = np.power(np.clip(rgb * 2.55, 0.0, 1.0), 0.86)
    return rgb
