"""Shared density accumulation and tone mapping for trajectory renders."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, label

from ..color import log_tone
from ..palette import sample_palette
from ..view import fit_window


def deposit(
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
    """Splat particle hits into ``density`` / ``hue`` (in-place)."""
    if weights is None and (hue_val is None or np.isscalar(hue_val)):
        from .. import kernels

        kernels.deposit_kernel(density, hue, x, y, xmin, xmax, ymin, ymax, float(hue_val or 0.0))
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


def auto_window(
    xs: Sequence[np.ndarray],
    ys: Sequence[np.ndarray],
    pad: float = 0.06,
    percentiles: tuple[float, float] = (0.4, 99.6),
) -> tuple[float, float, float, float]:
    """Percentile bounding box with padding."""
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


def dense_window(
    xs: Sequence[np.ndarray],
    ys: Sequence[np.ndarray],
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
        return auto_window(xs, ys)
    H, xe, ye = np.histogram2d(x, y, bins=260)
    total = float(H.sum())
    if total <= 0:
        return auto_window(xs, ys)
    peak = float(H.max())
    pos = H[H > 0]
    thresh = max(peak * 0.16, float(np.percentile(pos, q)))
    binary = H >= thresh
    labeled, nlab = label(binary)
    if nlab == 0:
        return auto_window(xs, ys)
    ix_peak, iy_peak = np.unravel_index(int(np.argmax(H)), H.shape)
    lab = int(labeled[ix_peak, iy_peak])
    if lab == 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        lab = int(np.argmax(sizes))
    mask = labeled == lab
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
        return auto_window(xs, ys)
    x0, x1 = float(xe[ix.min()]), float(xe[ix.max() + 1])
    y0, y1 = float(ye[iy.min()]), float(ye[iy.max() + 1])
    dx, dy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    return x0 - dx * pad, x1 + dx * pad, y0 - dy * pad, y1 + dy * pad


def tone_map_density(
    density: np.ndarray,
    hue_acc: np.ndarray,
    palette: str,
    blur: float = 0.7,
    gamma: float = 0.72,
    norm: float = 1.0,
) -> np.ndarray:
    """Blur, log-tone, palette-map, and shade a density buffer.

    ``norm`` (from ``ctx.density_norm``) rescales hit counts to the 4K
    reference so the tone curve, and with it the brightness, match at any size.
    """
    if norm != 1.0:
        density = density * norm
    if blur > 0:
        density = gaussian_filter(density, sigma=blur)
        hue_acc = gaussian_filter(hue_acc, sigma=blur)
    tone = log_tone(density, lo=2.0, hi=99.7)
    tone = np.power(tone, gamma)
    rgb = sample_palette(palette, tone)
    rgb *= 0.15 + 0.85 * tone[..., None]
    return np.clip(rgb, 0.0, 1.0)


def paint_density(
    width: int,
    height: int,
    xs: Sequence[np.ndarray],
    ys: Sequence[np.ndarray],
    hs: Sequence[np.ndarray],
    palette: str,
    blur: float = 0.7,
    gamma: float = 0.72,
    percentiles: tuple[float, float] = (0.4, 99.6),
    tight: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    pad: float = 0.06,
    fill: str = "auto",
    norm: float = 1.0,
) -> np.ndarray:
    """Deposit trajectories into a frame and tone-map (full in-memory path)."""
    density = np.zeros((height, width), dtype=np.float64)
    hue_acc = np.zeros((height, width), dtype=np.float64)
    if xlim is not None and ylim is not None:
        xmin, xmax = float(xlim[0]), float(xlim[1])
        ymin, ymax = float(ylim[0]), float(ylim[1])
    elif tight:
        xmin, xmax, ymin, ymax = dense_window(xs, ys)
    else:
        xmin, xmax, ymin, ymax = auto_window(xs, ys, pad=pad, percentiles=percentiles)
    xmin, xmax, ymin, ymax = fit_window(xmin, xmax, ymin, ymax, width, height, fill=fill)

    for x, y, h in zip(xs, ys, hs):
        deposit(density, hue_acc, x, y, xmin, xmax, ymin, ymax, hue_val=h)

    return tone_map_density(density, hue_acc, palette, blur=blur, gamma=gamma, norm=norm)
