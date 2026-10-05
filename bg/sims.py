"""Reaction-diffusion, flow fields, and interference patterns."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from .color import histogram_index, hsv_to_rgb, log_tone, sample_palette


def _laplacian(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, 0)
        + np.roll(a, -1, 0)
        + np.roll(a, 1, 1)
        + np.roll(a, -1, 1)
        - 4.0 * a
    )


def gray_scott(
    width: int,
    height: int,
    F: float,
    k: float,
    steps: int = 9000,
    seed: int = 0,
    palette: str = "aurora",
    sim_scale: float = 0.28,
) -> np.ndarray:
    """Gray-Scott RD, simulated at reduced res then upsampled to wallpaper size."""
    # 4K uses 0.28 (~3.6× upsample). At 8K keep the same relative upsample
    # and the extra pixels are empty zoom. Run closer to native so 200% zoom holds.
    if width >= 7000:
        sim_scale = max(sim_scale, 0.58)
    elif width >= 5000:
        sim_scale = max(sim_scale, 0.42)
    sw, sh = max(8, int(width * sim_scale)), max(8, int(height * sim_scale))
    rng = np.random.default_rng(seed)
    u = np.ones((sh, sw), dtype=np.float64)
    v = np.zeros((sh, sw), dtype=np.float64)

    # several seed blobs
    for _ in range(14):
        cy = rng.integers(sh // 8, 7 * sh // 8)
        cx = rng.integers(sw // 8, 7 * sw // 8)
        ry = rng.integers(4, max(5, sh // 18))
        rx = rng.integers(4, max(5, sw // 18))
        yy, xx = np.ogrid[:sh, :sw]
        blob = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        u[blob] = 0.50
        v[blob] = 0.25
    v += 0.02 * rng.random((sh, sw))
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)

    Du, Dv, dt = 0.16, 0.08, 1.0
    from .fast import gray_scott_kernel

    u, v = gray_scott_kernel(u, v, float(F), float(k), int(steps), Du, Dv, dt)
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    field = v / (v.max() + 1e-12)
    field = gaussian_filter(field, sigma=0.6)
    zoom_y = height / field.shape[0]
    zoom_x = width / field.shape[1]
    field = zoom(field, (zoom_y, zoom_x), order=3)
    field = field[:height, :width]
    t = histogram_index(field, field > 0.02)
    rgb = sample_palette(palette, np.power(t, 0.9))
    # keep a dark substrate
    bg = np.array(sample_palette(palette, np.array([0.02]))[0])
    mix = np.clip(field * 1.4, 0.0, 1.0)[..., None]
    rgb = bg * (1.0 - mix) + rgb * mix
    return np.clip(rgb, 0.0, 1.0)


def flow_field(
    width: int,
    height: int,
    n_particles: int = 90000,
    steps: int = 280,
    seed: int = 1,
    palette: str = "twilight",
    kind: str = "silk",
    wrap: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    aspect = width / height
    x = rng.uniform(-aspect, aspect, size=n_particles)
    y = rng.uniform(-1.0, 1.0, size=n_particles)
    density = np.zeros((height, width), dtype=np.float64)
    hue = np.zeros((height, width), dtype=np.float64)

    def field(px, py):
        if kind == "silk":
            vx = (
                np.sin(1.6 * py + 0.4 * np.sin(2.3 * px))
                + 0.42 * np.sin(3.1 * px + 1.2 * py)
                + 0.28 * np.sin(5.1 * py - 1.7 * px)
                + 0.16 * np.cos(2.4 * px * py)
            )
            vy = (
                np.cos(1.4 * px - 0.3 * np.sin(2.0 * py))
                + 0.42 * np.cos(2.7 * py - 0.8 * px)
                + 0.28 * np.cos(4.6 * px + 1.4 * py)
                + 0.16 * np.sin(3.3 * px - 2.1 * py)
            )
        else:
            # curl-like swirls
            vx = np.sin(px * py) + np.cos(1.7 * py)
            vy = np.cos(px * px - py) * 0.7 + np.sin(1.3 * px)
        nrm = np.sqrt(vx * vx + vy * vy) + 1e-9
        return vx / nrm, vy / nrm

    dt = 0.012
    xmin, xmax = -aspect, aspect
    ymin, ymax = -1.0, 1.0
    for i in range(steps):
        vx, vy = field(x, y)
        x = x + vx * dt * 18.0
        y = y + vy * dt * 18.0
        if wrap:
            x = ((x + aspect) % (2 * aspect)) - aspect
            y = ((y + 1.0) % 2.0) - 1.0
        else:
            left = (np.abs(x) > aspect) | (np.abs(y) > 1.0)
            if np.any(left):
                n_left = int(left.sum())
                x[left] = rng.uniform(-aspect, aspect, size=n_left)
                y[left] = rng.uniform(-1.0, 1.0, size=n_left)
        ix = ((x - xmin) / (xmax - xmin) * (width - 1)).astype(np.int32)
        iy = ((ymax - y) / (ymax - ymin) * (height - 1)).astype(np.int32)
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        ang = (np.arctan2(vy, vx) + np.pi) / (2 * np.pi)
        np.add.at(density, (iy[valid], ix[valid]), 1.0)
        np.add.at(hue, (iy[valid], ix[valid]), ang[valid])
        if i % 32 == 31:
            n_die = n_particles // 12
            idx = rng.integers(0, n_particles, size=n_die)
            x[idx] = rng.uniform(-aspect, aspect, size=n_die)
            y[idx] = rng.uniform(-1.0, 1.0, size=n_die)

    density = gaussian_filter(density, sigma=0.8)
    hue = gaussian_filter(hue, sigma=0.8)
    with np.errstate(invalid="ignore"):
        hmean = np.divide(hue, density, out=np.zeros_like(hue), where=density > 1)
    tone = log_tone(density, lo=5.0, hi=99.6)
    tone = np.power(tone, 0.75)
    rgb = sample_palette(palette, np.mod(hmean * 0.55 + tone * 0.7, 1.0))
    rgb *= (0.08 + 0.92 * tone)[..., None]
    return np.clip(rgb, 0.0, 1.0)


def cymatics(
    width: int,
    height: int,
    n_sources: int = 7,
    freq: float = 18.0,
    seed: int = 4,
    palette: str = "abyss",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    aspect = width / height
    xs = np.linspace(-aspect, aspect, width)
    ys = np.linspace(1.0, -1.0, height)
    X, Y = np.meshgrid(xs, ys)
    amp = np.zeros((height, width), dtype=np.float64)
    for _ in range(n_sources):
        sx = rng.uniform(-aspect * 0.7, aspect * 0.7)
        sy = rng.uniform(-0.7, 0.7)
        phase = rng.uniform(0, 2 * np.pi)
        k = freq * rng.uniform(0.75, 1.25)
        r = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2)
        amp += np.sin(k * r + phase) / (1.0 + 0.35 * r)
    t = histogram_index(np.abs(amp), np.ones_like(amp, dtype=bool))
    # emphasize nodal lines
    nodes = np.exp(-np.abs(amp) * 3.5)
    rgb = sample_palette(palette, np.power(t, 0.8))
    rgb = rgb * (0.25 + 0.75 * (1.0 - 0.55 * nodes)[..., None])
    highlight = sample_palette("abyss", np.clip(t * 1.2, 0, 1))
    mix = np.clip(np.abs(amp) / (np.abs(amp).max() + 1e-9), 0, 1)[..., None]
    rgb = rgb * (1 - 0.35 * mix) + highlight * (0.35 * mix)
    return np.clip(rgb, 0.0, 1.0)


def magnetic_pendulum(
    width: int,
    height: int,
    n_magnets: int = 5,
    steps: int = 90,
    dt: float = 0.08,
    seed: int = 2,
    framed: str = "wide",
) -> np.ndarray:
    """Color each start position by which magnet captures the pendulum."""
    rng = np.random.default_rng(seed)
    aspect = width / float(height)
    if framed == "close":
        mag_r, strength, friction, g, span = 0.48, 1.35, 0.20, 0.42, 0.70
    else:
        mag_r, strength, friction, g, span = 0.62, 1.15, 0.17, 0.35, 0.95
    mx = mag_r * np.cos(2 * np.pi * np.arange(n_magnets) / n_magnets)
    my = mag_r * np.sin(2 * np.pi * np.arange(n_magnets) / n_magnets)

    # Landscape: ±span on Y, wider on X. Portrait: ±span on X so the
    # magnet ring fills the phone width instead of floating in a black column.
    if aspect < 1.0:
        xs = np.linspace(-span, span, width)
        ys = np.linspace(span / aspect, -span / aspect, height)
    else:
        xs = np.linspace(-aspect * span, aspect * span, width)
        ys = np.linspace(span, -span, height)
    X, Y = np.meshgrid(xs, ys)
    X0, Y0 = X.copy(), Y.copy()
    vx = np.zeros_like(X)
    vy = np.zeros_like(Y)

    for _ in range(steps):
        ax = -g * X - friction * vx
        ay = -g * Y - friction * vy
        for i in range(n_magnets):
            dx = mx[i] - X
            dy = my[i] - Y
            r2 = dx * dx + dy * dy + 0.08
            inv = strength / (r2 * np.sqrt(r2))
            ax += dx * inv
            ay += dy * inv
        vx += ax * dt
        vy += ay * dt
        X += vx * dt
        Y += vy * dt

    dist = np.stack([(X - mx[i]) ** 2 + (Y - my[i]) ** 2 for i in range(n_magnets)], axis=0)
    winner = np.argmin(dist, axis=0)
    dmin = np.sqrt(np.min(dist, axis=0))
    h = winner.astype(np.float64) / n_magnets
    speed = np.sqrt(vx * vx + vy * vy)
    r0 = np.sqrt(X0 * X0 + Y0 * Y0)
    # outer sea is unresolved chaos — keep the colored basins as the hero
    settled = (dmin < 0.42) & (speed < 1.8)
    falloff = np.exp(-1.15 * np.maximum(r0 - mag_r * 0.72, 0.0))
    if framed == "close":
        v = 0.20 + 0.80 * np.exp(-dmin * 2.6)
        v *= 1.0 / (1.0 + 0.70 * speed)
        s = 0.40 + 0.30 * np.exp(-dmin * 1.8)
    else:
        v = 0.18 + 0.82 * np.exp(-dmin * 3.0)
        v *= 1.0 / (1.0 + 0.85 * speed)
        s = 0.58 + 0.18 * np.exp(-dmin * 1.6)
    v = v * (0.12 + 0.88 * falloff)
    v = np.where(settled, v, v * 0.16)
    s = np.where(settled, s, s * 0.28)
    rgb = hsv_to_rgb(h, s, np.clip(v, 0, 1))
    return rgb
