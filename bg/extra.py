"""Additional wallpaper simulations: CA, IFS, plates, noise, traces."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from .color import histogram_index, hsv_to_rgb, log_tone, sample_palette
from .attractors import _deposit, fit_attractor_frame, tone_map_density


def chladni(
    width: int,
    height: int,
    n: int = 5,
    m: int = 3,
    palette: str = "abyss",
) -> np.ndarray:
    aspect = width / height
    xs = np.linspace(-aspect, aspect, width)
    ys = np.linspace(1.0, -1.0, height)
    X, Y = np.meshgrid(xs, ys)
    # map to unit square-ish plate
    px = (X / aspect + 1.0) * 0.5
    py = (Y + 1.0) * 0.5
    field = np.cos(n * np.pi * px) * np.cos(m * np.pi * py) - np.cos(m * np.pi * px) * np.cos(
        n * np.pi * py
    )
    t = histogram_index(np.abs(field), np.ones_like(field, dtype=bool))
    nodes = np.exp(-np.abs(field) * 18.0)
    rgb = sample_palette(palette, np.power(t, 0.85))
    rgb = rgb * (0.2 + 0.8 * (1.0 - 0.7 * nodes)[..., None])
    highlight = np.clip(nodes, 0, 1)[..., None] * np.array([0.85, 0.95, 1.0])
    return np.clip(rgb + 0.35 * highlight, 0.0, 1.0)


def worley(
    width: int,
    height: int,
    n_points: int = 48,
    seed: int = 7,
    palette: str = "cosmic",
    mode: str = "F2-F1",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    aspect = width / height
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
    rgb = sample_palette(palette, np.power(t, 0.9))
    rgb *= (0.15 + 0.85 * t)[..., None]
    return np.clip(rgb, 0.0, 1.0)


def barnsley_fern(width: int, height: int, n: int = 6_000_000, seed: int = 1, palette: str = "forest") -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(n * max(1.0, width * height / (3840.0 * 2160.0)))
    p = 40_000
    xs = np.zeros(p, dtype=np.float64)
    ys = np.zeros(p, dtype=np.float64)
    steps = max(1, n // p)
    xmin, xmax, ymin, ymax = fit_attractor_frame(-2.6, 2.9, -0.3, 10.6, width, height, fill="contain")
    density = np.zeros((height, width), dtype=np.float64)
    hue_acc = np.zeros((height, width), dtype=np.float64)
    denom = max(steps - 1, 1)
    for s in range(steps):
        u = rng.random(p)
        nx, ny = np.empty_like(xs), np.empty_like(ys)
        m1 = u < 0.01
        m2 = (u >= 0.01) & (u < 0.86)
        m3 = (u >= 0.86) & (u < 0.93)
        m4 = u >= 0.93
        nx[m1], ny[m1] = 0.0, 0.16 * ys[m1]
        nx[m2] = 0.85 * xs[m2] + 0.04 * ys[m2]
        ny[m2] = -0.04 * xs[m2] + 0.85 * ys[m2] + 1.6
        nx[m3] = 0.20 * xs[m3] - 0.26 * ys[m3]
        ny[m3] = 0.23 * xs[m3] + 0.22 * ys[m3] + 1.6
        nx[m4] = -0.15 * xs[m4] + 0.28 * ys[m4]
        ny[m4] = 0.26 * xs[m4] + 0.24 * ys[m4] + 0.44
        xs, ys = nx, ny
        if s > 8:
            _deposit(density, hue_acc, xs, ys, xmin, xmax, ymin, ymax, hue_val=s / denom)
    return tone_map_density(density, hue_acc, palette, blur=0.45, gamma=0.62)


def cyclic_ca(
    width: int,
    height: int,
    n_states: int = 14,
    steps: int = 70,
    seed: int = 3,
    sim_scale: float = 0.42,
) -> np.ndarray:
    if width >= 7000:
        sim_scale = max(sim_scale, 0.75)
    sw, sh = max(32, int(width * sim_scale)), max(32, int(height * sim_scale))
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_states, size=(sh, sw), dtype=np.int32)
    for _ in range(steps):
        nxt = (g + 1) % n_states
        neigh = (
            (np.roll(g, 1, 0) == nxt)
            | (np.roll(g, -1, 0) == nxt)
            | (np.roll(g, 1, 1) == nxt)
            | (np.roll(g, -1, 1) == nxt)
            | (np.roll(np.roll(g, 1, 0), 1, 1) == nxt)
            | (np.roll(np.roll(g, 1, 0), -1, 1) == nxt)
            | (np.roll(np.roll(g, -1, 0), 1, 1) == nxt)
            | (np.roll(np.roll(g, -1, 0), -1, 1) == nxt)
        )
        g = np.where(neigh, nxt, g)
    field = zoom(g.astype(np.float64), (height / sh, width / sw), order=0)
    field = field[:height, :width]
    h = field / n_states
    v = 0.45 + 0.55 * (0.5 + 0.5 * np.sin(field * 0.9))
    s = np.full_like(h, 0.62)
    return hsv_to_rgb(h, s, v)


def sandpile(
    width: int,
    height: int,
    n: int | None = None,
    grains: int | None = None,
    palette: str = "ember",
) -> np.ndarray:
    if n is None:
        # 420 at 4K; linear with the short side so 8K is a real 2× grid.
        n = max(420, int(round(min(width, height) * 420 / 2160)))
    if grains is None:
        grains = n * n * 2
    g = np.zeros((n, n), dtype=np.int32)
    g[n // 2, n // 2] = grains
    from .fast import sandpile_topple

    g = sandpile_topple(g, 400_000)
    field = g.astype(np.float64)
    field = zoom(field, (height / n, width / n), order=0)
    field = field[:height, :width]
    t = field / max(float(field.max()), 1.0)
    rgb = sample_palette(palette, t)
    rgb *= (0.12 + 0.88 * np.power(t, 0.7))[..., None]
    return np.clip(rgb, 0.0, 1.0)


def harmonograph(
    width: int,
    height: int,
    n_curves: int = 12,
    samples: int = 4000,
    seed: int = 9,
    palette: str = "twilight",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Dense enough that consecutive samples sit ~1px apart at 4K.
    n_samp = max(samples, int(width * 80))
    t = np.linspace(0.0, 16.0 * np.pi, n_samp)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    hues: list[np.ndarray] = []
    ratios = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (3.0, 5.0), (1.0, 3.0), (4.0, 5.0)]
    for k in range(n_curves):
        fx, fy = ratios[k % len(ratios)]
        fx += rng.uniform(-0.012, 0.012)
        fy += rng.uniform(-0.012, 0.012)
        ax, ay = rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0)
        a2, a4 = rng.uniform(0.18, 0.42), rng.uniform(0.18, 0.42)
        f2 = fx + rng.choice([1.0, 2.0]) + rng.uniform(-0.01, 0.01)
        f4 = fy + rng.choice([1.0, 2.0]) + rng.uniform(-0.01, 0.01)
        p1, p2, p3, p4 = rng.uniform(0.0, 2.0 * np.pi, size=4)
        # Slow decay so traces stay long and readable instead of collapsing to a dot.
        d1, d2 = rng.uniform(0.0012, 0.006), rng.uniform(0.0012, 0.006)
        d3, d4 = rng.uniform(0.0012, 0.006), rng.uniform(0.0012, 0.006)
        x = ax * np.sin(fx * t + p1) * np.exp(-d1 * t) + a2 * np.sin(f2 * t + p2) * np.exp(-d2 * t)
        y = ay * np.sin(fy * t + p3) * np.exp(-d3 * t) + a4 * np.sin(f4 * t + p4) * np.exp(-d4 * t)
        xs.append(x)
        ys.append(y)
        hues.append(np.full(n_samp, k / max(n_curves - 1, 1), dtype=np.float64))

    allx = np.concatenate(xs)
    ally = np.concatenate(ys)
    finite = np.isfinite(allx) & np.isfinite(ally)
    allx, ally = allx[finite], ally[finite]
    xmin, xmax = float(allx.min()), float(allx.max())
    ymin, ymax = float(ally.min()), float(ally.max())
    dx, dy = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    pad = 0.08
    xmin -= dx * pad
    xmax += dx * pad
    ymin -= dy * pad
    ymax += dy * pad
    aspect = width / height
    dx, dy = xmax - xmin, ymax - ymin
    if dx / dy > aspect:
        extra = dx / aspect - dy
        ymin -= extra / 2
        ymax += extra / 2
    else:
        extra = dy * aspect - dx
        xmin -= extra / 2
        xmax += extra / 2

    density = np.zeros((height, width), dtype=np.float64)
    hue = np.zeros((height, width), dtype=np.float64)
    for x, y, hv in zip(xs, ys, hues):
        ix = ((x - xmin) / (xmax - xmin) * (width - 1)).astype(np.int32)
        iy = ((ymax - y) / (ymax - ymin) * (height - 1)).astype(np.int32)
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        np.add.at(density, (iy[valid], ix[valid]), 1.0)
        np.add.at(hue, (iy[valid], ix[valid]), hv[valid])

    density = gaussian_filter(density, sigma=1.2)
    hue = gaussian_filter(hue, sigma=1.2)
    tone = log_tone(density, lo=6.0, hi=99.6)
    with np.errstate(invalid="ignore", divide="ignore"):
        hmean = np.divide(hue, density, out=np.zeros_like(hue), where=density > 1e-9)
    rgb = sample_palette(palette, np.clip(hmean * 0.55 + tone * 0.45, 0.0, 1.0))
    rgb *= (0.05 + 0.95 * np.power(tone, 0.65))[..., None]
    return np.clip(rgb, 0.0, 1.0)


def apollonian_dots(
    width: int,
    height: int,
    depth: int = 8,
    palette: str = "bronze",
) -> np.ndarray:
    """Descartes circle theorem packing, filled as glowing nested rings."""
    aspect = width / height

    def soddy(c1, c2, c3, inner=True):
        k1, k2, k3 = 1.0 / c1[2], 1.0 / c2[2], 1.0 / c3[2]
        sign = 1.0 if inner else -1.0
        disc = k1 * k2 + k1 * k3 + k2 * k3
        k4 = k1 + k2 + k3 + sign * 2.0 * np.sqrt(abs(disc))
        if abs(k4) < 1e-9:
            return None
        r = 1.0 / k4
        z1, z2, z3 = c1[0] + 1j * c1[1], c2[0] + 1j * c2[1], c3[0] + 1j * c3[1]
        zdisc = k1 * k2 * z1 * z2 + k1 * k3 * z1 * z3 + k2 * k3 * z2 * z3
        z4 = (z1 * k1 + z2 * k2 + z3 * k3 + sign * 2.0 * np.sqrt(zdisc)) / k4
        return (float(z4.real), float(z4.imag), float(abs(r)))

    # three mutually tangent equal circles (equilateral centers)
    r0 = 1.0
    d = 2.0 * r0
    seeds = [
        (0.0, d / np.sqrt(3.0), r0),
        (-d / 2.0, -d / (2.0 * np.sqrt(3.0)), r0),
        (d / 2.0, -d / (2.0 * np.sqrt(3.0)), r0),
    ]
    outer = soddy(seeds[0], seeds[1], seeds[2], inner=False)
    inner0 = soddy(seeds[0], seeds[1], seeds[2], inner=True)
    circles = list(seeds)
    if outer is not None:
        circles.append(outer)
    if inner0 is not None:
        circles.append(inner0)

    seen = {(round(c[0], 5), round(c[1], 5), round(c[2], 5)) for c in circles}
    queue: list[tuple] = []
    if outer is not None:
        queue.append((seeds[0], seeds[1], outer))
        queue.append((seeds[0], seeds[2], outer))
        queue.append((seeds[1], seeds[2], outer))
    if inner0 is not None:
        queue.append((seeds[0], seeds[1], inner0))
        queue.append((seeds[0], seeds[2], inner0))
        queue.append((seeds[1], seeds[2], inner0))
    queue.append((seeds[0], seeds[1], seeds[2]))

    rmin = 0.018
    max_circles = 900
    while queue and len(circles) < max_circles:
        a, b, c = queue.pop(0)
        for inner in (True, False):
            dcirc = soddy(a, b, c, inner=inner)
            if dcirc is None or dcirc[2] < rmin or dcirc[2] > 8.0:
                continue
            key = (round(dcirc[0], 5), round(dcirc[1], 5), round(dcirc[2], 5))
            if key in seen:
                continue
            seen.add(key)
            circles.append(dcirc)
            queue.append((a, b, dcirc))
            queue.append((a, c, dcirc))
            queue.append((b, c, dcirc))

    # drop the huge encompassing circle and fit the gasket to the 16:9 frame
    pack = [c for c in circles if c[2] < 2.6]
    if not pack:
        pack = circles
    xs_c = np.array([c[0] for c in pack])
    ys_c = np.array([c[1] for c in pack])
    rs_c = np.array([c[2] for c in pack])
    x0, x1 = float((xs_c - rs_c).min()), float((xs_c + rs_c).max())
    y0, y1 = float((ys_c - rs_c).min()), float((ys_c + rs_c).max())
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.48 * max(x1 - x0, y1 - y0)
    # zoom in so nested circles, not empty margin, own the frame
    half *= 0.78
    fitted = [((c[0] - cx) / half, (c[1] - cy) / half, c[2] / half) for c in pack]

    img = np.zeros((height, width), dtype=np.float64)
    xs = np.linspace(-aspect, aspect, width)
    ys = np.linspace(1.0, -1.0, height)
    X, Y = np.meshgrid(xs, ys)
    dx = (xs[1] - xs[0]) if width > 1 else 0.01
    dy = (ys[0] - ys[1]) if height > 1 else 0.01

    for cx, cy, r in fitted:
        if r < 0.006:
            continue
        pad = 0.04 + 0.08 * r
        xlo, xhi = cx - r - pad, cx + r + pad
        ylo, yhi = cy - r - pad, cy + r + pad
        ix0 = max(0, int((xlo + aspect) / (2 * aspect) * (width - 1)))
        ix1 = min(width, int((xhi + aspect) / (2 * aspect) * (width - 1)) + 1)
        iy0 = max(0, int((1.0 - yhi) / 2.0 * (height - 1)))
        iy1 = min(height, int((1.0 - ylo) / 2.0 * (height - 1)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        slx = X[iy0:iy1, ix0:ix1]
        sly = Y[iy0:iy1, ix0:ix1]
        dist = np.sqrt((slx - cx) ** 2 + (sly - cy) ** 2)
        sigma = max(1.2 * min(dx, dy), 0.012 * r)
        ring = np.exp(-((dist - r) ** 2) / (2.0 * sigma**2))
        fill = np.clip(1.0 - dist / (r + 1e-9), 0.0, 1.0) ** 1.35
        wgt = 0.55 + 0.45 * (0.08 / (r + 0.08))
        img[iy0:iy1, ix0:ix1] += wgt * (ring * 1.15 + 0.38 * fill)

    img = gaussian_filter(img, sigma=0.55)
    t = log_tone(img, lo=4.0, hi=99.6)
    rgb = sample_palette(palette, np.power(t, 0.78))
    rgb *= (0.08 + 0.92 * t)[..., None]
    return np.clip(rgb, 0.0, 1.0)
