"""Post-processing that works on any render, whatever generator made it.

Recipes (or ``--post key=value`` on the CLI) set a ``post`` table::

    post = { exposure = 0.3, saturation = 1.2, hue_shift = 20, vignette = 0.35 }

Ops run in registry order (the order below), each only when its key is set.
Add your own with::

    @fractalgen.post_op("posterize")
    def posterize(rgb, levels, ctx):
        return np.round(rgb * levels) / levels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .color import hsv_to_rgb
from .palette import get_palette
from .registry import Registry

PostFn = Callable[[np.ndarray, Any, Any], np.ndarray]


@dataclass(frozen=True)
class PostOp:
    fn: PostFn
    doc: str
    neutral: Any  # value at which the op does nothing (GUI reset / default)
    range: tuple | None = None  # slider hint
    kind: str = ""  # GUI hint: "float", "bool", "palette", "color"

    def __iter__(self):  # (fn, doc) unpacking for simple callers
        return iter((self.fn, self.doc))


POST_OPS: Registry[PostOp] = Registry("post op")


def post_op(
    name: str, doc: str = "", *, neutral: Any = None, range: tuple | None = None, kind: str = "", replace: bool = False
) -> Callable[[PostFn], PostFn]:
    def wrap(fn: PostFn) -> PostFn:
        k = kind or ("bool" if isinstance(neutral, bool) else "float" if isinstance(neutral, (int, float)) else "")
        op = PostOp(fn, doc or (fn.__doc__ or "").strip(), neutral, range, k)
        POST_OPS.register(name, op, replace=replace)
        return fn

    return wrap


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    safe = np.where(d > 0, d, 1.0)
    h = np.where(mx == r, (g - b) / safe, np.where(mx == g, 2.0 + (b - r) / safe, 4.0 + (r - g) / safe))
    h = np.where(d > 0, (h / 6.0) % 1.0, 0.0)
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0), 0.0)
    return h, s, mx


@post_op("palette", "Re-color by luminance through a palette (any palette spec)", kind="palette")
def _palette(rgb, spec, ctx):
    return get_palette(spec).sample(np.clip(_luma(rgb), 0.0, 1.0))


@post_op("palette_mix", "Blend amount for `palette` (0 = original, 1 = fully remapped)", neutral=1.0, range=(0.0, 1.0))
def _palette_mix(rgb, amount, ctx):
    return rgb  # consumed by apply_post


@post_op("invert", "Invert colors (true/false)", neutral=False)
def _invert(rgb, on, ctx):
    return 1.0 - rgb if on else rgb


@post_op("hue_shift", "Rotate hue by this many degrees", neutral=0.0, range=(-180.0, 180.0))
def _hue_shift(rgb, deg, ctx):
    h, s, v = _rgb_to_hsv(rgb)
    return hsv_to_rgb(h + float(deg) / 360.0, s, v)


@post_op("saturation", "Saturation multiplier (0 = grayscale, 1 = unchanged)", neutral=1.0, range=(0.0, 3.0))
def _saturation(rgb, amount, ctx):
    y = _luma(rgb)[..., None]
    return y + (rgb - y) * float(amount)


@post_op("exposure", "Brightness in stops (+1 doubles, -1 halves)", neutral=0.0, range=(-3.0, 3.0))
def _exposure(rgb, stops, ctx):
    return rgb * (2.0 ** float(stops))


@post_op("contrast", "Contrast around mid-gray (1 = unchanged)", neutral=1.0, range=(0.0, 3.0))
def _contrast(rgb, amount, ctx):
    return (rgb - 0.5) * float(amount) + 0.5


@post_op("gamma", "Gamma (>1 brightens shadows, <1 deepens them)", neutral=1.0, range=(0.2, 3.0))
def _gamma(rgb, g, ctx):
    return np.power(np.clip(rgb, 0.0, 1.0), 1.0 / float(g))


@post_op("tint", "Multiply by an RGB color: hex string or [r, g, b] in 0..1", kind="color")
def _tint(rgb, color, ctx):
    from .palette import hex_to_rgb

    c = hex_to_rgb(color) if isinstance(color, str) else tuple(float(x) for x in color)
    return rgb * np.asarray(c)


@post_op("vignette", "Darken corners (0 = off, ~0.3 subtle, 1 strong)", neutral=0.0, range=(0.0, 1.0))
def _vignette(rgb, amount, ctx):
    h, w = rgb.shape[:2]
    y = np.linspace(-1.0, 1.0, h)[:, None]
    x = np.linspace(-1.0, 1.0, w)[None, :]
    r2 = (x * x + y * y) / 2.0
    return rgb * (1.0 - float(amount) * r2)[..., None]


@post_op("flip_x", "Mirror left-right", neutral=False)
def _flip_x(rgb, on, ctx):
    return rgb[:, ::-1] if on else rgb


@post_op("flip_y", "Mirror top-bottom", neutral=False)
def _flip_y(rgb, on, ctx):
    return rgb[::-1] if on else rgb


def apply_post(rgb: np.ndarray, post: dict[str, Any] | None, ctx: Any = None) -> np.ndarray:
    if not post:
        return rgb
    unknown = [k for k in post if k not in POST_OPS]
    if unknown:
        raise KeyError(f"Unknown post op(s) {unknown}. Known: {POST_OPS.names()}")
    out = np.asarray(rgb, dtype=np.float64)
    for name in POST_OPS:
        if name not in post or post[name] is None or name == "palette_mix":
            continue
        fn = POST_OPS.get(name).fn
        if name == "palette":
            mix = float(post.get("palette_mix", 1.0))
            out = out * (1.0 - mix) + fn(out, post[name], ctx) * mix
        else:
            out = fn(out, post[name], ctx)
    return np.clip(out, 0.0, 1.0)
