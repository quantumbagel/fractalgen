"""Color-space helpers, tone mapping, and image output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import tone


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


def histogram_index(
    values: np.ndarray, mask: np.ndarray, bins: int = 4096, key: str = "histogram"
) -> np.ndarray:
    """Histogram-equalize ``values[mask]`` into [0, 1]; unmasked pixels are 0.

    The equalization curve is a tone statistic (see tone.py): off the anchor
    view it comes from the anchor, so panning and zooming don't re-expose.
    """
    out = np.zeros(values.shape, dtype=np.float64)
    if not np.any(mask):
        return out
    v = values[mask]
    curve = tone.stat(key, lambda: _equalize_curve(v, bins))
    if curve is None:
        out[mask] = tone.levels(np.ones(v.shape))
        return out
    vmin, vmax, cdf = curve
    scaled = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
    ix = np.clip((scaled * (len(cdf) - 1)).astype(np.int32), 0, len(cdf) - 1)
    out[mask] = tone.levels(cdf[ix])
    return out


def _equalize_curve(v: np.ndarray, bins: int) -> tuple[float, float, np.ndarray] | None:
    vmin, vmax = np.percentile(v, 0.5), np.percentile(v, 99.5)
    if vmax <= vmin:
        return None
    scaled = np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0)
    ix = np.clip((scaled * (bins - 1)).astype(np.int32), 0, bins - 1)
    cdf = np.cumsum(np.bincount(ix, minlength=bins).astype(np.float64))
    return float(vmin), float(vmax), cdf / cdf[-1]


def log_tone(density: np.ndarray, lo: float = 1.0, hi: float = 99.5, key: str = "log_tone") -> np.ndarray:
    """log1p + percentile stretch for accumulation buffers (range is a tone statistic)."""
    d = np.log1p(np.maximum(density, 0.0))

    def stretch() -> tuple[float, float] | None:
        finite = np.isfinite(d) & (d > 0)
        if not np.any(finite):
            return None
        a, b = np.percentile(d[finite], [lo, hi])
        return (float(a), float(b)) if b > a else (0.0, float(d.max()) + 1e-12)

    ab = tone.stat(key, stretch)
    if ab is None:
        return np.zeros_like(d)
    a, b = ab
    return tone.levels(np.clip((d - a) / (b - a), 0.0, 1.0))


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
