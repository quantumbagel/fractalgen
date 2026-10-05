"""Strange attractors and trajectory density paintings."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label

from .color import log_tone, sample_palette


def _ultra_mul(width: int, height: int, cap: float = 4.0) -> float:
    """1 at 4K, 4 at 8K. Does not shrink sub-4K counts."""
    return min(cap, max(1.0, width * height / (3840.0 * 2160.0)))


def _deposit(
    density: np.ndarray,
    hue: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    weights: np.ndarray | None = None,
    hue_val: np.ndarray | float | None = None,
) -> None:
    if weights is None and (hue_val is None or np.isscalar(hue_val)):
        from .fast import deposit_kernel

        deposit_kernel(density, hue, x, y, xmin, xmax, ymin, ymax, float(hue_val or 0.0))
        return
    h, w = density.shape
    ix = ((x - xmin) / (xmax - xmin) * (w - 1)).astype(np.int32)
    iy = ((ymax - y) / (ymax - ymin) * (h - 1)).astype(np.int32)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return
    ix = ix[valid]
    iy = iy[valid]
    if weights is None:
        np.add.at(density, (iy, ix), 1.0)
    else:
        np.add.at(density, (iy, ix), weights[valid])
    if hue_val is not None:
        if np.isscalar(hue_val):
            np.add.at(hue, (iy, ix), hue_val)
        else:
            np.add.at(hue, (iy, ix), hue_val[valid])


def _auto_window(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    pad: float = 0.06,
    percentiles: tuple[float, float] = (0.4, 99.6),
) -> tuple[float, float, float, float]:
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    x0, x1 = np.percentile(x, list(percentiles))
    y0, y1 = np.percentile(y, list(percentiles))
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * pad
    x1 += dx * pad
    y0 -= dy * pad
    y1 += dy * pad
    return float(x0), float(x1), float(y0), float(y1)


def _dense_window(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    pad: float = 0.08,
    q: float = 84.0,
    mass_frac: float = 0.55,
) -> tuple[float, float, float, float]:
    """Crop to the high-mass core (e.g. Ikeda whirlpool), not a thin tail."""
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 32:
        return _auto_window(xs, ys)
    H, xe, ye = np.histogram2d(x, y, bins=260)
    total = float(H.sum())
    if total <= 0:
        return _auto_window(xs, ys)
    peak = float(H.max())
    pos = H[H > 0]
    thresh = max(peak * 0.16, float(np.percentile(pos, q)))
    binary = H >= thresh
    labeled, nlab = label(binary)
    if nlab == 0:
        return _auto_window(xs, ys)
    ix_peak, iy_peak = np.unravel_index(int(np.argmax(H)), H.shape)
    lab = int(labeled[ix_peak, iy_peak])
    if lab == 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        lab = int(np.argmax(sizes))
    mask = labeled == lab
    # If the connected blob is still huge (long spiral tail), keep only the
    # densest bins that hold most of the particle mass.
    if mask.mean() > 0.28 or H[mask].sum() / total > 0.92:
        flat = H.ravel()
        order = np.argsort(flat)[::-1]
        csum = np.cumsum(flat[order])
        keep = np.zeros_like(flat, dtype=bool)
        cutoff = int(np.searchsorted(csum, mass_frac * total) + 1)
        keep[order[: max(cutoff, 8)]] = True
        mask = keep.reshape(H.shape)
    ix, iy = np.where(mask)
    if ix.size == 0:
        return _auto_window(xs, ys)
    x0, x1 = float(xe[ix.min()]), float(xe[ix.max() + 1])
    y0, y1 = float(ye[iy.min()]), float(ye[iy.max() + 1])
    dx, dy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    return x0 - dx * pad, x1 + dx * pad, y0 - dy * pad, y1 + dy * pad


def clifford_step(x, y, a, b, c, d):
    nx = np.sin(a * y) + c * np.cos(a * x)
    ny = np.sin(b * x) + d * np.cos(b * y)
    return nx, ny


def dejong_step(x, y, a, b, c, d):
    nx = np.sin(a * y) - np.cos(b * x)
    ny = np.sin(c * x) - np.cos(d * y)
    return nx, ny


def hopalong_step(x, y, a, b, c):
    nx = y - np.sign(x) * np.sqrt(np.abs(b * x - c))
    ny = a - x
    return nx, ny


def ikeda_step(x, y, u):
    t = 0.4 - 6.0 / (1.0 + x * x + y * y)
    ct, st = np.cos(t), np.sin(t)
    nx = 1.0 + u * (x * ct - y * st)
    ny = u * (x * st + y * ct)
    return nx, ny


def gumowski_mira_step(x, y, a, b):
    def f(z):
        return a * z + 2.0 * (1.0 - a) * z * z / (1.0 + z * z)

    nx = b * y + f(x)
    ny = -x + f(nx)
    return nx, ny


def iterate_map(
    step,
    n_particles: int,
    warmup: int,
    steps: int,
    seed: int = 0,
    init: tuple[float, float] = (0.0, 0.0),
    init_sigma: float = 0.12,
    explode: float = 50.0,
    collect: bool = True,
    on_step=None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
    rng = np.random.default_rng(seed)
    cx, cy = init

    def _seed(n: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            rng.normal(cx, init_sigma, size=n),
            rng.normal(cy, init_sigma, size=n),
        )

    x, y = _seed(n_particles)
    for _ in range(warmup):
        x, y = step(x, y)
        bad = ~np.isfinite(x) | ~np.isfinite(y) | (np.abs(x) > explode) | (np.abs(y) > explode)
        if np.any(bad):
            x[bad], y[bad] = _seed(int(bad.sum()))
    xs, ys, hs = [], [], []
    denom = max(steps - 1, 1)
    for i in range(steps):
        x, y = step(x, y)
        bad = ~np.isfinite(x) | ~np.isfinite(y) | (np.abs(x) > explode) | (np.abs(y) > explode)
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


def fit_attractor_frame(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    fill: str = "auto",
) -> tuple[float, float, float, float]:
    aspect = width / float(height)
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    if fill == "auto":
        fill = "cover" if aspect < 1.0 else "contain"
    if fill == "cover":
        if dx / dy > aspect:
            new_dx = dy * aspect
            mid = 0.5 * (xmin + xmax)
            xmin, xmax = mid - new_dx / 2, mid + new_dx / 2
        else:
            new_dy = dx / aspect
            mid = 0.5 * (ymin + ymax)
            ymin, ymax = mid - new_dy / 2, mid + new_dy / 2
    else:
        if dx / dy > aspect:
            extra = dx / aspect - dy
            ymin -= extra / 2
            ymax += extra / 2
        else:
            extra = dy * aspect - dx
            xmin -= extra / 2
            xmax += extra / 2
    return xmin, xmax, ymin, ymax


def tone_map_density(
    density: np.ndarray,
    hue_acc: np.ndarray,
    palette: str,
    blur: float = 0.7,
    gamma: float = 0.72,
) -> np.ndarray:
    if blur > 0:
        density = gaussian_filter(density, sigma=blur)
        hue_acc = gaussian_filter(hue_acc, sigma=blur)
    tone = log_tone(density, lo=2.0, hi=99.7)
    tone = np.power(tone, gamma)
    rgb = sample_palette(palette, tone)
    rgb *= 0.15 + 0.85 * tone[..., None]
    return np.clip(rgb, 0.0, 1.0)


def paint_attractor(
    width: int,
    height: int,
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    hs: list[np.ndarray],
    palette: str,
    blur: float = 0.7,
    gamma: float = 0.72,
    hue_mix: float = 0.35,
    percentiles: tuple[float, float] = (0.4, 99.6),
    tight: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    pad: float = 0.06,
    fill: str = "auto",
) -> np.ndarray:
    density = np.zeros((height, width), dtype=np.float64)
    hue_acc = np.zeros((height, width), dtype=np.float64)
    if xlim is not None and ylim is not None:
        xmin, xmax = float(xlim[0]), float(xlim[1])
        ymin, ymax = float(ylim[0]), float(ylim[1])
    elif tight:
        xmin, xmax, ymin, ymax = _dense_window(xs, ys)
    else:
        xmin, xmax, ymin, ymax = _auto_window(xs, ys, pad=pad, percentiles=percentiles)
    xmin, xmax, ymin, ymax = fit_attractor_frame(xmin, xmax, ymin, ymax, width, height, fill=fill)

    for x, y, h in zip(xs, ys, hs):
        _deposit(density, hue_acc, x, y, xmin, xmax, ymin, ymax, hue_val=h)

    return tone_map_density(density, hue_acc, palette, blur=blur, gamma=gamma)


def lorenz_flow(
    width: int,
    height: int,
    n_particles: int = 2500,
    steps: int = 1800,
    dt: float = 0.008,
    seed: int = 3,
    palette: str = "ember",
) -> np.ndarray:
    n_particles = int(n_particles * _ultra_mul(width, height))
    rng = np.random.default_rng(seed)
    # butterfly-shaped cloud of ICs near the attractor
    x = rng.normal(0.0, 0.4, size=n_particles)
    y = rng.normal(1.0, 0.4, size=n_particles)
    z = rng.normal(20.0, 2.0, size=n_particles)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def deriv(x, y, z):
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    xs, ys, zs = [], [], []
    for i in range(steps):
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
        if i > 80:
            xs.append(x.copy())
            ys.append(y.copy())
            zs.append(z.copy())

    # isometric-ish projection
    ang = np.deg2rad(38)
    ca, sa = np.cos(ang), np.sin(ang)
    px = [xx * ca + yy * sa for xx, yy in zip(xs, ys)]
    py = [zz * 0.72 - 0.22 * xx * sa + 0.22 * yy * ca for xx, yy, zz in zip(xs, ys, zs)]
    hs = [np.full(n_particles, i / max(len(px) - 1, 1)) for i in range(len(px))]
    return paint_attractor(width, height, px, py, hs, palette, blur=0.85, gamma=0.68)


def svensson_step(x, y, a, b, c, d):
    nx = d * np.sin(a * x) - np.sin(b * y)
    ny = c * np.cos(a * x) + np.cos(b * y)
    return nx, ny


def bedhead_step(x, y, a, b):
    nx = np.sin(x * y / b) * y + np.cos(a * x - y)
    ny = x + np.sin(y) / b
    return nx, ny


def tinkerbell_step(x, y, a=0.9, b=-0.6013, c=2.0, d=0.5):
    nx = x * x - y * y + a * x + b * y
    ny = 2.0 * x * y + c * x + d * y
    return nx, ny


def fractal_dream_step(x, y, a, b, c, d):
    nx = np.sin(b * y) + c * np.sin(b * x)
    ny = np.sin(a * x) + d * np.sin(a * y)
    return nx, ny


def _rk4_flow(
    width: int,
    height: int,
    deriv,
    x,
    y,
    z,
    steps: int,
    dt: float,
    skip: int,
    palette: str,
    ang: float = 32.0,
    tight: bool = False,
    blur: float = 0.8,
    gamma: float = 0.7,
    percentiles: tuple[float, float] = (0.4, 99.6),
) -> np.ndarray:
    n_particles = x.shape[0]
    xs, ys, zs = [], [], []
    for i in range(steps):
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
        if i > skip:
            xs.append(x.copy())
            ys.append(y.copy())
            zs.append(z.copy())
    rad = np.deg2rad(ang)
    ca, sa = np.cos(rad), np.sin(rad)
    px = [xx * ca + yy * sa for xx, yy in zip(xs, ys)]
    py = [zz * 0.7 - 0.2 * xx * sa + 0.2 * yy * ca for xx, yy, zz in zip(xs, ys, zs)]
    hs = [np.full(n_particles, i / max(len(px) - 1, 1)) for i in range(len(px))]
    return paint_attractor(
        width,
        height,
        px,
        py,
        hs,
        palette,
        blur=blur,
        gamma=gamma,
        tight=tight,
        percentiles=percentiles,
    )


def rossler_flow(width, height, seed=4, palette="aurora"):
    rng = np.random.default_rng(seed)
    area = max(1.0, width * height / (1280 * 720))
    n = int(5200 * min(3.5, max(1.0, area**0.38)))
    n = int(n * _ultra_mul(width, height))
    # seed on the attractor band, not a tiny blob at the origin
    x = rng.normal(-2.0, 1.4, n)
    y = rng.normal(0.0, 1.2, n)
    z = rng.uniform(0.05, 4.0, n)
    a, b, c = 0.2, 0.2, 5.7

    def deriv(x, y, z):
        return -y - z, x + a * y, b + z * (x - c)

    return _rk4_flow(
        width,
        height,
        deriv,
        x,
        y,
        z,
        steps=4200,
        dt=0.016,
        skip=180,
        palette=palette,
        ang=48,
        tight=True,
        blur=1.15,
        gamma=0.52,
        percentiles=(3.0, 97.5),
    )


def aizawa_flow(width, height, seed=5, palette="twilight"):
    rng = np.random.default_rng(seed)
    n = int(1800 * _ultra_mul(width, height))
    x = rng.normal(0.1, 0.15, n)
    y = rng.normal(0.0, 0.15, n)
    z = rng.normal(0.0, 0.15, n)
    a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1

    def deriv(x, y, z):
        return (z - b) * x - d * y, d * x + (z - b) * y, c + a * z - z**3 / 3.0 - (x*x + y*y) * (1.0 + e * z) + f * z * x**3

    return _rk4_flow(width, height, deriv, x, y, z, steps=2000, dt=0.01, skip=100, palette=palette, ang=28)


def thomas_flow(width, height, seed=6, palette="abyss"):
    rng = np.random.default_rng(seed)
    n = int(2000 * _ultra_mul(width, height))
    x = rng.normal(0.0, 0.4, n)
    y = rng.normal(0.0, 0.4, n)
    z = rng.normal(0.0, 0.4, n)
    b = 0.19

    def deriv(x, y, z):
        return np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z

    return _rk4_flow(width, height, deriv, x, y, z, steps=2400, dt=0.05, skip=80, palette=palette, ang=40)
