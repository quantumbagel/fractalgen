"""Escape-time fractals, Newton basins, Lyapunov space, and domain coloring."""

from __future__ import annotations

import numpy as np

from .color import hsv_to_rgb, histogram_index, sample_palette


def complex_plane(
    width: int,
    height: int,
    center: complex,
    half_width: float,
    fit: str = "fill",
    ref_aspect: float = 16.0 / 9.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a complex mesh.

    ``half_width`` is the horizontal half-extent of the *authored* 16:9 window
    unless ``fit='fill'``, in which case it is the horizontal half-extent of
    *this* frame.

    fit:
      fill    — half_width is for this aspect (legacy)
      cover   — same zoom as the 16:9 window, crop to this aspect
      contain — show the whole 16:9 window, pad along the long axis
    """
    aspect = width / float(height)
    if fit == "cover":
        ref_hh = half_width / ref_aspect
        if aspect >= ref_aspect:
            half_height = ref_hh
            half_width = half_height * aspect
        else:
            half_height = ref_hh
            half_width = half_height * aspect
    elif fit == "contain":
        half_height = half_width / aspect
    else:
        half_height = half_width / aspect
    re = np.linspace(center.real - half_width, center.real + half_width, width, dtype=np.float64)
    im = np.linspace(center.imag + half_height, center.imag - half_height, height, dtype=np.float64)
    return np.meshgrid(re, im)


def _park(mask: np.ndarray, zr: np.ndarray, zi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zr = np.where(mask, zr, 4.0)
    zi = np.where(mask, zi, 0.0)
    return zr, zi


def escape_time(
    cr: np.ndarray,
    ci: np.ndarray,
    max_iter: int,
    kind: str = "mandelbrot",
    julia_c: complex | None = None,
    phoenix_k: float = -0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (smooth_iter, escaped_mask). Interior has smooth_iter = 0."""
    from .fast import KIND, escape_kernel

    kind_id = KIND.get(kind, 0)
    jcr = float(julia_c.real) if julia_c is not None else 0.5667
    jci = float(julia_c.imag) if julia_c is not None else 0.0
    escaped_n, esc_u8 = escape_kernel(
        np.ascontiguousarray(cr, dtype=np.float64),
        np.ascontiguousarray(ci, dtype=np.float64),
        int(max_iter),
        kind_id,
        jcr,
        jci,
        float(phoenix_k),
    )
    return escaped_n, esc_u8.astype(bool)


def color_escape(
    escaped_n: np.ndarray,
    escaped: np.ndarray,
    palette: str,
    *,
    cyclic: bool = False,
    interior: tuple[float, float, float] = (0.0, 0.0, 0.0),
    gamma: float = 0.85,
) -> np.ndarray:
    t = histogram_index(escaped_n, escaped)
    t = np.power(t, gamma)
    rgb = sample_palette(palette, t, cyclic=cyclic)
    rgb[~escaped] = np.array(interior, dtype=np.float64)
    return rgb


def orbit_trap(
    cr: np.ndarray,
    ci: np.ndarray,
    max_iter: int,
    trap: str = "rings",
) -> np.ndarray:
    """Minimum distance to a trap shape along the Mandelbrot orbit."""
    from .fast import orbit_trap_kernel

    trap_id = {"rings": 0, "cross": 1, "circle": 2}.get(trap, 3)
    return orbit_trap_kernel(
        np.ascontiguousarray(cr, dtype=np.float64),
        np.ascontiguousarray(ci, dtype=np.float64),
        int(max_iter),
        trap_id,
    )


def newton_basins(
    zr0: np.ndarray,
    zi0: np.ndarray,
    power: int,
    max_iter: int = 40,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Newton for z^n - 1 = 0. Returns (root_index, iterations)."""
    from .fast import newton_kernel

    return newton_kernel(
        np.ascontiguousarray(zr0, dtype=np.float64),
        np.ascontiguousarray(zi0, dtype=np.float64),
        int(power),
        int(max_iter),
        float(tol),
    )


def color_newton(
    root_ix: np.ndarray,
    iters: np.ndarray,
    power: int,
    zr: np.ndarray | None = None,
    zi: np.ndarray | None = None,
    style: str = "classic",
) -> np.ndarray:
    tmax = max(float(iters.max()), 1.0)
    if style == "cinematic":
        tables = {
            3: np.array([0.55, 0.80, 0.10]),
            4: np.array([0.55, 0.80, 0.10, 0.68]),
            5: np.array([0.55, 0.72, 0.80, 0.10, 0.16]),
            6: np.array([0.55, 0.63, 0.78, 0.90, 0.08, 0.16]),
            8: np.array([0.55, 0.62, 0.72, 0.80, 0.90, 0.08, 0.12, 0.18]),
        }
        if power in tables:
            h = tables[power][root_ix]
        else:
            h = np.mod(0.55 + root_ix.astype(np.float64) / power * 0.38, 1.0)
        t = np.clip(iters / tmax, 0.0, 1.0)
        shade = 0.10 + 0.90 * (1.0 - t) ** 0.38
        shade *= 0.78 + 0.22 * np.cos(iters * 1.05)
        if zr is not None and zi is not None:
            r = np.sqrt(zr * zr + zi * zi)
            shade = shade * (0.50 + 0.50 * np.exp(-0.16 * r))
        s = 0.44 + 0.20 * t
        h = np.mod(h + 0.018 * t, 1.0)
        return hsv_to_rgb(h, s, np.clip(shade, 0.0, 1.0))

    h = root_ix.astype(np.float64) / power
    shade = 0.28 + 0.72 * (1.0 - np.clip(iters / tmax, 0.0, 1.0)) ** 0.65
    s = np.full_like(h, 0.72)
    h = np.mod(h + 0.04 * (iters / tmax), 1.0)
    return hsv_to_rgb(h, s, shade)


def lyapunov(
    a: np.ndarray,
    b: np.ndarray,
    sequence: str = "BBBBBBAAAAAA",
    warmup: int = 80,
    samples: int = 220,
) -> np.ndarray:
    """Lyapunov exponent of the logistic map with a/b sequence."""
    from .fast import lyapunov_kernel

    seq = np.array([1 if ch == "B" else 0 for ch in sequence], dtype=np.uint8)
    return lyapunov_kernel(
        np.ascontiguousarray(a, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
        seq,
        int(warmup),
        int(samples),
    )


def color_lyapunov(exponents: np.ndarray) -> np.ndarray:
    """Classic gold chaotic / blue stable split."""
    from .color import sample_palette

    t = np.zeros_like(exponents)
    cha = exponents > 0
    sta = ~cha
    if np.any(sta):
        s = exponents[sta]
        t[sta] = 0.5 * histogram_index(-s, np.ones_like(s, dtype=bool))
    if np.any(cha):
        c = exponents[cha]
        t[cha] = 0.5 + 0.5 * histogram_index(c, np.ones_like(c, dtype=bool))
    rgb = sample_palette("icefire", t)
    return rgb


def domain_color(zr: np.ndarray, zi: np.ndarray, kind: str = "rational") -> np.ndarray:
    z = zr + 1j * zi
    if kind == "rational":
        w = (z**2 - 1.0) / (z**2 + z + 1.0)
    elif kind == "sin":
        w = np.sin(z)
    elif kind == "exp_inv":
        w = np.exp(1.0 / np.clip(z, 1e-6, None)) if False else np.exp(1.0 / (z + 1e-6))
    elif kind == "poly":
        w = z**5 - z + 0.2
    else:
        w = z

    mag = np.abs(w)
    phase = np.angle(w)
    h = (phase + np.pi) / (2.0 * np.pi)
    logm = np.log(mag + 1e-12)
    # modulus contours
    ridges = 0.55 + 0.45 * (0.5 + 0.5 * np.cos(2.0 * np.pi * logm / np.log(2.0)))
    # checker of real/imag for extra structure
    checker = 0.88 + 0.12 * np.sign(np.sin(6.0 * w.real) * np.sin(6.0 * w.imag))
    v = np.clip(ridges * checker * (mag**0.15 / (1.0 + mag**0.15) * 1.35), 0.0, 1.0)
    s = np.clip(0.55 + 0.4 * (1.0 - 1.0 / (1.0 + mag)), 0.35, 1.0)
    return hsv_to_rgb(h, s, v)


def newton_shifted(
    zr0: np.ndarray,
    zi0: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Newton for z^3 - 2z + 2 = 0, colored by attractor argument."""
    zr, zi = zr0.copy(), zi0.copy()
    done_n = np.zeros(zr.shape, dtype=np.float64)
    done = np.zeros(zr.shape, dtype=bool)
    for k in range(1, max_iter + 1):
        live = ~done
        if not np.any(live):
            break
        z = zr + 1j * zi
        p = z**3 - 2.0 * z + 2.0
        dp = 3.0 * z**2 - 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            z_new = z - p / dp
        z_new = np.where(np.isfinite(z_new), z_new, z)
        zr = np.where(live, z_new.real, zr)
        zi = np.where(live, z_new.imag, zi)
        newly = live & (np.abs(p) < tol)
        done[newly] = True
        done_n[newly] = k
    return zr + 1j * zi, done_n


def popcorn_field(zr: np.ndarray, zi: np.ndarray, steps: int = 40, hstep: float = 0.05) -> np.ndarray:
    """Pickover popcorn: color by orbit displacement on the plane."""
    from .fast import popcorn_kernel

    dx, dy = popcorn_kernel(
        np.ascontiguousarray(zr, dtype=np.float64),
        np.ascontiguousarray(zi, dtype=np.float64),
        int(steps),
        float(hstep),
    )
    ang = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
    mag = np.sqrt(dx * dx + dy * dy)
    v = histogram_index(np.log1p(mag), np.isfinite(mag))
    v = 0.18 + 0.82 * np.power(v, 0.75)
    return hsv_to_rgb(ang, np.full_like(ang, 0.62), v)
