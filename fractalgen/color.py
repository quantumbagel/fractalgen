"""Color-space helpers, tone mapping, and image output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = np.mod(h, 1.0)
    s = np.clip(s, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i6 = i % 6
    r = np.choose(i6, [v, q, p, p, t, v])
    g = np.choose(i6, [t, v, v, q, p, p])
    b = np.choose(i6, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def histogram_index(values: np.ndarray, mask: np.ndarray, bins: int = 4096) -> np.ndarray:
    """Histogram-equalize ``values[mask]`` into [0, 1]; unmasked pixels are 0."""
    out = np.zeros(values.shape, dtype=np.float64)
    if not np.any(mask):
        return out
    v = values[mask]
    vmin, vmax = np.percentile(v, 0.5), np.percentile(v, 99.5)
    if vmax <= vmin:
        out[mask] = 1.0
        return out
    scaled = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
    ix = np.clip((scaled * (bins - 1)).astype(np.int32), 0, bins - 1)
    hist = np.bincount(ix, minlength=bins).astype(np.float64)
    cdf = np.cumsum(hist)
    cdf /= cdf[-1]
    out[mask] = cdf[ix]
    return out


def log_tone(density: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> np.ndarray:
    """log1p + percentile stretch for accumulation buffers."""
    d = np.log1p(np.maximum(density, 0.0))
    finite = np.isfinite(d) & (d > 0)
    if not np.any(finite):
        return np.zeros_like(d)
    a, b = np.percentile(d[finite], [lo, hi])
    if b <= a:
        return np.clip(d / (d.max() + 1e-12), 0.0, 1.0)
    return np.clip((d - a) / (b - a), 0.0, 1.0)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def to_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(to_uint8(rgb), mode="RGB")


def save_image(path: str | Path, rgb: np.ndarray, fmt: str | None = None, jpeg_quality: int = 95) -> Path:
    """Save float RGB in [0, 1]. ``fmt`` defaults to the file extension."""
    Image.MAX_IMAGE_PIXELS = None
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or dest.suffix.lstrip(".") or "png").lower()
    img = to_image(rgb)
    # Write beside the target and rename, so a killed render never leaves a
    # truncated file that --skip-existing would then accept.
    tmp = dest.with_name(f".{dest.name}.partial")
    try:
        if fmt in ("jpg", "jpeg"):
            img.save(tmp, format="JPEG", quality=jpeg_quality, optimize=False)
        else:
            # optimize=True is RAM-heavy on 8K/16K; keep it for smaller frames only.
            optimize = img.size[0] * img.size[1] < 8_000_000
            img.save(tmp, format="PNG", optimize=optimize, compress_level=6)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest
