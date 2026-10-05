"""Palettes and tone mapping for cinematic wallpapers."""

from __future__ import annotations

import numpy as np
from PIL import Image


def _hex(c: str) -> tuple[float, float, float]:
    c = c.lstrip("#")
    return (int(c[0:2], 16) / 255.0, int(c[2:4], 16) / 255.0, int(c[4:6], 16) / 255.0)


PALETTES: dict[str, np.ndarray] = {
    # deep space → violet → magenta → gold
    "cosmic": np.array(
        [
            _hex("05010a"),
            _hex("12062a"),
            _hex("2b0b5e"),
            _hex("6a1b9a"),
            _hex("c2185b"),
            _hex("ff8a3d"),
            _hex("ffe7a0"),
        ],
        dtype=np.float64,
    ),
    # black → navy → teal → ice
    "abyss": np.array(
        [
            _hex("000208"),
            _hex("031428"),
            _hex("043c5c"),
            _hex("0a7a8c"),
            _hex("4ecdc4"),
            _hex("c8f4ff"),
            _hex("f5fdff"),
        ],
        dtype=np.float64,
    ),
    # ember / magma
    "ember": np.array(
        [
            _hex("080206"),
            _hex("2a0612"),
            _hex("7a1024"),
            _hex("d43b1a"),
            _hex("ff7a18"),
            _hex("ffd166"),
            _hex("fff3d6"),
        ],
        dtype=np.float64,
    ),
    # aurora borealis
    "aurora": np.array(
        [
            _hex("02060d"),
            _hex("06221f"),
            _hex("0b4d45"),
            _hex("1b8a5a"),
            _hex("7be07a"),
            _hex("c6f4c0"),
            _hex("e8fff0"),
        ],
        dtype=np.float64,
    ),
    # plasma-like
    "plasma": np.array(
        [
            _hex("0d0887"),
            _hex("41049d"),
            _hex("6a00a8"),
            _hex("b12a90"),
            _hex("e16462"),
            _hex("fca636"),
            _hex("f0f921"),
        ],
        dtype=np.float64,
    ),
    # ice fire diverging (Lyapunov)
    "icefire": np.array(
        [
            _hex("001a33"),
            _hex("0a4c7a"),
            _hex("7ec8e3"),
            _hex("f2efe9"),
            _hex("e8a03c"),
            _hex("c23b22"),
            _hex("3a0a08"),
        ],
        dtype=np.float64,
    ),
    # neon ink
    "neon": np.array(
        [
            _hex("02010a"),
            _hex("0b1a4a"),
            _hex("123e8c"),
            _hex("00c2ff"),
            _hex("7b2cbf"),
            _hex("ff2d95"),
            _hex("fff5c0"),
        ],
        dtype=np.float64,
    ),
    # copper / bronze on black
    "bronze": np.array(
        [
            _hex("070504"),
            _hex("1c1008"),
            _hex("4a2410"),
            _hex("b85c20"),
            _hex("e8a45c"),
            _hex("f5d7a8"),
            _hex("fff8ec"),
        ],
        dtype=np.float64,
    ),
    # twilight rose
    "twilight": np.array(
        [
            _hex("0a0614"),
            _hex("1a1040"),
            _hex("3d2b8c"),
            _hex("7a4daf"),
            _hex("e07bb0"),
            _hex("f7c1c6"),
            _hex("fff0e6"),
        ],
        dtype=np.float64,
    ),
    # forest gold
    "forest": np.array(
        [
            _hex("030805"),
            _hex("0a1f14"),
            _hex("14532d"),
            _hex("3f7d3a"),
            _hex("c4a35a"),
            _hex("efe0b0"),
            _hex("fffbeb"),
        ],
        dtype=np.float64,
    ),
    # deep-sea bioluminescence
    "biolum": np.array(
        [
            _hex("02040c"),
            _hex("061428"),
            _hex("0a3a4a"),
            _hex("0e8a7a"),
            _hex("3ee0c8"),
            _hex("7b6cff"),
            _hex("d8fff4"),
        ],
        dtype=np.float64,
    ),
    # slime-mold / dark-matter gold
    "goldsmoke": np.array(
        [
            _hex("070705"),
            _hex("16110a"),
            _hex("3a2410"),
            _hex("8a5a18"),
            _hex("e0a030"),
            _hex("f6d56a"),
            _hex("fff4c8"),
        ],
        dtype=np.float64,
    ),
    # refractive obsidian / smoky glass
    "obsidian": np.array(
        [
            _hex("07060c"),
            _hex("1a1430"),
            _hex("2a3a48"),
            _hex("3d6a6e"),
            _hex("8a6a9a"),
            _hex("c8b8a0"),
            _hex("e8e4dc"),
        ],
        dtype=np.float64,
    ),
    # liquid mercury & midnight indigo
    "mercury": np.array(
        [
            _hex("05060e"),
            _hex("101428"),
            _hex("2a3358"),
            _hex("6a7388"),
            _hex("b0b8c4"),
            _hex("e4e8ee"),
            _hex("f7f9fc"),
        ],
        dtype=np.float64,
    ),
    # quantum phase: amethyst → solar amber
    "rydberg": np.array(
        [
            _hex("1a0828"),
            _hex("4a1a6a"),
            _hex("8a3a9a"),
            _hex("d47840"),
            _hex("f0b020"),
            _hex("ffe080"),
            _hex("fff6d0"),
        ],
        dtype=np.float64,
    ),
    # weathered bronze & patina
    "patina": np.array(
        [
            _hex("0a0806"),
            _hex("1c140c"),
            _hex("3a2814"),
            _hex("6a4a22"),
            _hex("2a6a4a"),
            _hex("c4a050"),
            _hex("e8d8a8"),
        ],
        dtype=np.float64,
    ),
}


def sample_palette(name: str, t: np.ndarray, *, cyclic: bool = False) -> np.ndarray:
    """Map values in [0, 1] to RGB using named color stops."""
    stops = PALETTES[name]
    t = np.asarray(t, dtype=np.float64)
    if cyclic:
        t = np.mod(t, 1.0)
    else:
        t = np.clip(t, 0.0, 1.0)
    n = len(stops) - 1
    x = t * n
    i = np.floor(x).astype(np.int32)
    i = np.clip(i, 0, n - 1)
    f = (x - i)[..., None]
    c0 = stops[i]
    c1 = stops[i + 1]
    return c0 * (1.0 - f) + c1 * f


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
    d = np.log1p(np.maximum(density, 0.0))
    finite = np.isfinite(d) & (d > 0)
    if not np.any(finite):
        return np.zeros_like(d)
    a, b = np.percentile(d[finite], [lo, hi])
    if b <= a:
        return np.clip(d / (d.max() + 1e-12), 0.0, 1.0)
    return np.clip((d - a) / (b - a), 0.0, 1.0)


def to_image(rgb: np.ndarray) -> Image.Image:
    rgb8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb8, mode="RGB")


def save_png(path: str, rgb: np.ndarray) -> None:
    to_image(rgb).save(path, format="PNG", optimize=True)
