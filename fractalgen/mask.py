"""Masks: cut any render out with text, a shape, or an image (fractal lettering).

A recipe (or ``--mask key=value`` on the CLI, or the Mask section in the GUI)
sets a ``mask`` table. It is off until one of ``text``, ``shape`` or ``image``
is set::

    mask = { text = "DRAGON", font = "Arial Black", background = "self", glow = 0.5 }
    mask = { shape = "heart", size = 0.7, outline = 6, outline_color = "ffffff" }
    mask = { image = "logo.png", background = "transparent" }   # PNG with alpha

The fractal shows through the mask; ``background`` fills the rest:

- a hex color (``"000000"``, ``"#0b0b12"``)
- ``"transparent"``: PNG with alpha (JPEG flattens onto black)
- ``"self"``: the fractal itself, dimmed by ``background_dim`` (blur it with ``background_blur``)
- ``"recipe:NAME"``: another recipe rendered at the same size (also dimmed / blurred)
- ``"palette:NAME"``: a top-to-bottom gradient through a palette

The mask runs before ``post``, so post finishes the whole frame. Lengths are
pixels at 4K, like everywhere else, so previews and exports match.
``python -m fractalgen mask`` lists every key.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import progress

SHAPES = ("", "circle", "ring", "heart", "star", "hexagon", "rounded")


@dataclass(frozen=True)
class MaskKey:
    doc: str
    neutral: Any  # the default; "neutral" matches post / tone ops for the GUI
    kind: str  # float, int, bool, string, choice, color
    range: tuple | None = None
    choices: tuple[str, ...] | None = None

    def coerce(self, name: str, value: Any) -> Any:
        try:
            if self.kind == "float":
                v = float(value)
                if not math.isfinite(v):
                    raise ValueError("must be finite")
                return v
            if self.kind == "bool":
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                return bool(value)
            if self.kind == "color" and isinstance(value, (list, tuple)):
                return [float(x) for x in value[:3]]
            v = str(value)
            if self.choices is not None and v not in self.choices:
                raise ValueError(f"must be one of {[c for c in self.choices if c]}")
            return v
        except (TypeError, ValueError) as e:
            raise ValueError(f"mask.{name}={value!r}: {e}") from None


MASK_KEYS: dict[str, MaskKey] = {
    "text": MaskKey('Text to cut the fractal into ("\\n" breaks lines)', "", "string"),
    "font": MaskKey("Font name (e.g. \"Arial Black\", \"Avenir Next Heavy\") or a .ttf/.otf/.ttc path; "
                    "empty = a heavy system font", "", "string"),
    "shape": MaskKey("Shape to cut the fractal into", "", "choice", choices=SHAPES),
    "image": MaskKey("Image file whose alpha (or brightness) is the mask; relative paths may sit in custom/",
                     "", "string"),
    "image_channel": MaskKey("Which part of `image` is the mask (auto: alpha if it has any, else brightness)",
                             "auto", "choice", choices=("auto", "alpha", "luma")),
    "size": MaskKey("Fraction of the frame the mask fills (it fits inside size x width by size x height)",
                    0.8, "float", (0.05, 1.5)),
    "x": MaskKey("Horizontal center of the mask, 0 = left edge, 1 = right edge", 0.5, "float", (0.0, 1.0)),
    "y": MaskKey("Vertical center of the mask, 0 = top edge, 1 = bottom edge", 0.5, "float", (0.0, 1.0)),
    "rotate": MaskKey("Rotate the mask counter-clockwise, degrees", 0.0, "float", (-180.0, 180.0)),
    "invert": MaskKey("Show the fractal outside the mask instead of inside", False, "bool"),
    "feather": MaskKey("Soften the mask edge, pixels at 4K", 0.0, "float", (0.0, 40.0)),
    "outline": MaskKey("Outline around the mask, pixels at 4K (0 = none)", 0.0, "float", (0.0, 40.0)),
    "outline_color": MaskKey("Outline color (hex)", "ffffff", "color"),
    "glow": MaskKey("The fractal's own colors glowing out past the mask edge (0 = off)", 0.0, "float", (0.0, 2.0)),
    "glow_radius": MaskKey("Size of the glow, pixels at 4K", 40.0, "float", (2.0, 200.0)),
    "shadow": MaskKey("Drop shadow under the mask onto the background (0 = off, 1 = black)", 0.0, "float", (0.0, 1.0)),
    "bevel": MaskKey("Raised, lit edges inside the mask (0 = flat)", 0.0, "float", (0.0, 2.0)),
    "background": MaskKey('Outside the mask: hex color, "transparent", "self" (the fractal, dimmed), '
                          '"recipe:NAME", or "palette:NAME" (vertical gradient)', "000000", "string"),
    "background_dim": MaskKey("Brightness of an image background (self / recipe), 0..1", 0.15, "float", (0.0, 1.0)),
    "background_blur": MaskKey("Blur of an image background, pixels at 4K", 0.0, "float", (0.0, 120.0)),
}


def check_mask(table: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and coerce a mask table. ``None`` values drop a key (back to its default)."""
    out: dict[str, Any] = {}
    for k, v in (table or {}).items():
        if k not in MASK_KEYS:
            raise KeyError(f"Unknown mask key {k!r}. Known: {sorted(MASK_KEYS)}")
        if v is not None:
            out[k] = MASK_KEYS[k].coerce(k, v)
    return out


def active(table: dict[str, Any] | None) -> bool:
    t = table or {}
    return bool(t.get("text") or t.get("shape") or t.get("image"))


def _get(table: dict[str, Any], key: str) -> Any:
    return table.get(key, MASK_KEYS[key].neutral)


# ---------------------------------------------------------------- helpers ----

def blur(a: np.ndarray, radius: float) -> np.ndarray:
    """Approximately Gaussian blur (sigma = radius / 2) over the first two axes.

    Large radii blur a block-averaged copy and scale it back up, so glows and
    soft backgrounds stay cheap at 8K / 16K.
    """
    a = np.asarray(a, dtype=np.float32)
    if radius < 0.5:
        return a
    from scipy.ndimage import uniform_filter, zoom

    h, w = a.shape[:2]
    k = max(1, min(int(radius / 6.0), h // 4, w // 4))
    small = a
    if k > 1:
        hh, ww = h // k, w // k
        small = a[: hh * k, : ww * k].reshape(hh, k, ww, k, *a.shape[2:]).mean(axis=(1, 3))
    sigma = radius / 2.0 / k
    # Three box passes of width b have variance 3 * (b^2 - 1) / 12.
    b = max(1, int(round(math.sqrt(4.0 * sigma * sigma + 1.0))))
    size = (b, b) + (1,) * (a.ndim - 2)
    for _ in range(3):
        small = uniform_filter(small, size=size, mode="nearest")
    if k == 1:
        return small
    factors = (h / small.shape[0], w / small.shape[1]) + (1,) * (a.ndim - 2)
    out = zoom(small, factors, order=1, mode="nearest", grid_mode=True)
    out = out[:h, :w]
    if out.shape[:2] != (h, w):  # zoom rounding: pad the last row / column
        out = np.pad(out, ((0, h - out.shape[0]), (0, w - out.shape[1])) + ((0, 0),) * (a.ndim - 2), mode="edge")
    return out


def hex_rgb(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        return np.asarray([float(x) for x in value[:3]], dtype=np.float64)
    s = str(value).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"not a hex color: {value!r}")
    return np.asarray([int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


def resolve_path(p: str) -> Path:
    """A user file: absolute, relative to the working directory, or relative to custom/."""
    from .loader import CUSTOM_DIR

    path = Path(str(p)).expanduser()
    if path.is_absolute() or path.exists():
        return path
    alt = CUSTOM_DIR / path
    return alt if alt.exists() else path


# ---------------------------------------------------------------- fonts ----

_FONT_DIRS = (
    "~/Library/Fonts", "/Library/Fonts", "/System/Library/Fonts/Supplemental", "/System/Library/Fonts",
    "~/.local/share/fonts", "~/.fonts", "/usr/local/share/fonts", "/usr/share/fonts", "C:/Windows/Fonts",
)
_FONT_EXT = (".ttf", ".otf", ".ttc")
_DEFAULT_FONTS = ("Arial Black", "Impact", "Helvetica Neue Bold", "Arial Bold", "DejaVuSans-Bold",
                  "LiberationSans-Bold", "Arial", "DejaVuSans")


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _font_files() -> dict[str, Path]:
    found: dict[str, Path] = {}
    extra = os.environ.get("FRACTALGEN_FONTS", "")
    dirs = [d for d in extra.split(os.pathsep) if d] + list(_FONT_DIRS)
    for d in dirs:
        root = Path(d).expanduser()
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() in _FONT_EXT:
                found.setdefault(_norm(f.stem), f)
    return found


@lru_cache(maxsize=64)
def find_font(name: str) -> tuple[str, int] | None:
    """(file, face index) for a font name or path; None if nothing matches.

    Names match file names ignoring case, spaces and dashes ("Arial Black",
    "arial-black"). A style after a family name picks a face inside a .ttc
    collection: "Avenir Next Heavy", "Helvetica Neue Bold".
    """
    from PIL import ImageFont

    p = Path(name).expanduser()
    if p.suffix.lower() in _FONT_EXT:
        p = resolve_path(name)
        if not p.exists():
            raise FileNotFoundError(f"mask.font: no font file {name!r}")
        return str(p), 0
    q = _norm(name)
    files = _font_files()
    if q in files:
        return str(files[q]), 0
    # "Family Style": longest family file that prefixes the query, then a face whose style matches.
    for stem in sorted((s for s in files if q.startswith(s) and s != q), key=len, reverse=True):
        path, style = files[stem], q[len(stem):]
        if path.suffix.lower() != ".ttc":
            continue
        for idx in range(64):
            try:
                face = ImageFont.truetype(str(path), 12, index=idx)
            except OSError:
                break
            if _norm(face.getname()[1] or "") == style:
                return str(path), idx
    return None


def load_font(name: str, size: float):
    from PIL import ImageFont

    size = max(1, int(round(size)))
    names = [name] if name else list(_DEFAULT_FONTS)
    for n in names:
        hit = find_font(n)
        if hit:
            return ImageFont.truetype(hit[0], size, index=hit[1])
    if name:
        raise ValueError(f"mask.font: no font named {name!r} (give a family name or a .ttf/.otf/.ttc path)")
    return ImageFont.load_default(size)


# ---------------------------------------------------------------- mask shapes ----

def _text_mask(text: str, font: str, width: int, height: int, box: tuple[float, float],
               center: tuple[float, float]) -> np.ndarray:
    from PIL import Image, ImageDraw

    text = text.replace("\\n", "\n")
    probe = 200.0
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))

    def bbox(f, s):
        return draw.multiline_textbbox((0, 0), text, font=f, align="center", spacing=0.12 * s)

    l, t, r, b = bbox(load_font(font, probe), probe)
    if r - l <= 0 or b - t <= 0:
        return np.zeros((height, width), np.float32)
    size = probe * min(box[0] / (r - l), box[1] / (b - t))
    f = load_font(font, size)
    l, t, r, b = bbox(f, size)
    canvas = Image.new("L", (width, height), 0)
    x0 = center[0] - (r - l) / 2.0 - l
    y0 = center[1] - (b - t) / 2.0 - t
    ImageDraw.Draw(canvas).multiline_text((x0, y0), text, font=f, fill=255, align="center", spacing=0.12 * size)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _image_mask(path: str, channel: str, width: int, height: int, box: tuple[float, float],
                center: tuple[float, float]) -> np.ndarray:
    from PIL import Image

    src = resolve_path(path)
    if not src.exists():
        raise FileNotFoundError(f"mask.image: no file {path!r}")
    with Image.open(src) as im:
        im.load()
        has_alpha = "A" in im.getbands() or (im.mode == "P" and "transparency" in im.info)
        rgba = im.convert("RGBA")
    alpha = rgba.getchannel("A")
    use_alpha = channel == "alpha" or (channel == "auto" and has_alpha and alpha.getextrema()[0] < 250)
    m = alpha if use_alpha else rgba.convert("L")
    k = min(box[0] / m.width, box[1] / m.height)
    nw, nh = max(1, round(m.width * k)), max(1, round(m.height * k))
    m = m.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("L", (width, height), 0)
    canvas.paste(m, (round(center[0] - nw / 2.0), round(center[1] - nh / 2.0)))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _shape_mask(shape: str, width: int, height: int, box: tuple[float, float], center: tuple[float, float],
                rotate: float) -> np.ndarray:
    """Antialiased signed-distance shapes; rotation is applied to the coordinates."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    dx, dy = xs + 0.5 - center[0], center[1] - (ys + 0.5)  # y up
    th = math.radians(rotate)
    c, s = math.cos(th), math.sin(th)
    px, py = c * dx + s * dy, -s * dx + c * dy
    if shape == "rounded":
        hx, hy = box[0] / 2.0, box[1] / 2.0
        rr = 0.18 * min(hx, hy)
        qx, qy = np.abs(px) - hx + rr, np.abs(py) - hy + rr
        d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rr
    else:
        R = min(box) / 2.0
        x, y = px / R, py / R
        if shape == "circle":
            d = np.hypot(x, y) - 1.0
        elif shape == "ring":
            d = np.abs(np.hypot(x, y) - 0.78) - 0.22
        elif shape == "heart":
            d = _sd_heart(x * 0.604, y * 0.604 + 0.552) / 0.604
        elif shape == "star":
            d = _sd_star5(x, y, 1.0, 0.45)
        elif shape == "hexagon":
            d = _sd_hexagon(x, y, 0.866)
        else:
            raise ValueError(f"mask.shape: unknown shape {shape!r}")
        d = d * R
    return np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)


def _sd_heart(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    upper = np.hypot(x - 0.25, y - 0.75) - math.sqrt(2.0) / 4.0
    t = 0.5 * np.maximum(x + y, 0.0)
    lower = np.sqrt(np.minimum(x * x + (y - 1.0) ** 2, (x - t) ** 2 + (y - t) ** 2)) * np.sign(x - y)
    return np.where(x + y > 1.0, upper, lower)


def _sd_star5(x: np.ndarray, y: np.ndarray, r: float, rf: float) -> np.ndarray:
    k1x, k1y = 0.809016994375, -0.587785252292
    k2x, k2y = -k1x, k1y
    x = np.abs(x)
    d = np.maximum(k1x * x + k1y * y, 0.0) * 2.0
    x, y = x - d * k1x, y - d * k1y
    d = np.maximum(k2x * x + k2y * y, 0.0) * 2.0
    x, y = x - d * k2x, y - d * k2y
    x = np.abs(x)
    y = y - r
    bax, bay = rf * -k1y, rf * k1x - 1.0
    h = np.clip((x * bax + y * bay) / (bax * bax + bay * bay), 0.0, r)
    return np.hypot(x - bax * h, y - bay * h) * np.sign(y * bax - x * bay)


def _sd_hexagon(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    kx, ky, kz = -0.866025404, 0.5, 0.577350269
    x, y = np.abs(x), np.abs(y)
    d = np.minimum(kx * x + ky * y, 0.0) * 2.0
    x, y = x - d * kx, y - d * ky
    x = x - np.clip(x, -kz * r, kz * r)
    y = y - r
    return np.hypot(x, y) * np.sign(y)


def _rotate(m: np.ndarray, deg: float, center: tuple[float, float]) -> np.ndarray:
    if not deg:
        return m
    from PIL import Image

    img = Image.fromarray(np.clip(m * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
    img = img.rotate(deg, resample=Image.BICUBIC, center=center)
    return np.asarray(img, dtype=np.float32) / 255.0


def build(table: dict[str, Any], width: int, height: int) -> np.ndarray:
    """The coverage mask (H, W) in 0..1, before feathering / inversion."""
    size = float(_get(table, "size"))
    box = (max(1.0, size * width), max(1.0, size * height))
    center = (float(_get(table, "x")) * width, float(_get(table, "y")) * height)
    rotate = float(_get(table, "rotate"))
    m = np.zeros((height, width), np.float32)
    if table.get("shape"):
        m = np.maximum(m, _shape_mask(table["shape"], width, height, box, center, rotate))
    if table.get("image"):
        im = _image_mask(table["image"], _get(table, "image_channel"), width, height, box, center)
        m = np.maximum(m, _rotate(im, rotate, center))
    if table.get("text"):
        tm = _text_mask(table["text"], str(_get(table, "font")), width, height, box, center)
        m = np.maximum(m, _rotate(tm, rotate, center))
    return m


# ---------------------------------------------------------------- compositing ----

_BG_DEPTH = [0]  # keeps recipe:NAME backgrounds from recursing forever


def _background(table: dict[str, Any], width: int, height: int, fractal: np.ndarray, scale: float):
    """(background (H, W, 3), transparent?)."""
    spec = str(_get(table, "background")).strip()
    low = spec.lower()
    if low in ("transparent", "none", "alpha"):
        return np.zeros((height, width, 3), np.float32), True
    if low.startswith("palette:"):
        from .palette import get_palette

        ramp = np.asarray(get_palette(spec.split(":", 1)[1].strip()).sample(np.linspace(0.0, 1.0, height)))
        return np.broadcast_to(ramp[:, None, :3], (height, width, 3)).astype(np.float32), False
    if low == "self":
        img = fractal
    elif low.startswith("recipe:"):
        from .recipe import RECIPES

        if _BG_DEPTH[0] >= 2:
            raise ValueError("mask.background: recipe backgrounds nest too deeply")
        _BG_DEPTH[0] += 1
        try:
            img = np.asarray(RECIPES.get(spec.split(":", 1)[1].strip()).render(width, height))[..., :3]
        finally:
            _BG_DEPTH[0] -= 1
    else:
        try:
            color = hex_rgb(spec)
        except ValueError:
            raise ValueError(
                f"mask.background={spec!r}: use a hex color, transparent, self, recipe:NAME or palette:NAME"
            ) from None
        return np.broadcast_to(color.astype(np.float32), (height, width, 3)).copy(), False
    img = blur(img, float(_get(table, "background_blur")) * scale)
    return np.asarray(img, dtype=np.float32) * float(_get(table, "background_dim")), False


def _outline_ring(m: np.ndarray, width_px: float) -> np.ndarray:
    """Antialiased band of ``width_px`` just outside the mask."""
    from scipy.ndimage import distance_transform_edt

    inside = m >= 0.5
    if not inside.any():
        return np.zeros_like(m)
    pad = int(math.ceil(width_px)) + 2
    ys, xs = np.nonzero(inside)
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad + 1)
    dist = distance_transform_edt(~inside[y0:y1, x0:x1])
    ring = np.zeros_like(m)
    ring[y0:y1, x0:x1] = np.clip(width_px + 0.5 - dist, 0.0, 1.0)
    return ring * (1.0 - m)


def _bevel(rgb: np.ndarray, m: np.ndarray, amount: float, scale: float) -> np.ndarray:
    radius = max(2.0, 10.0 * scale)
    h = blur(m, radius)
    gy, gx = np.gradient(h)
    k = 2.0 * radius
    nx, ny, nz = -gx * k, -gy * k, np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    lx, ly, lz = -0.5, -0.6, 0.62  # from the upper left (screen y points down)
    ln = 1.0 / math.sqrt(lx * lx + ly * ly + lz * lz)
    lit = (nx * lx + ny * ly + nz * lz) * inv * ln
    shade = 1.0 + amount * (lit / (lz * ln) - 1.0)
    spec = np.clip(lit, 0.0, 1.0) ** 24 * 0.5 * amount
    return rgb * shade[..., None] + spec[..., None]


def apply_mask(fractal: np.ndarray, table: dict[str, Any], scale: float) -> np.ndarray:
    """Composite ``fractal`` (H, W, 3) through the mask. Returns RGB, or RGBA when transparent.

    ``scale`` converts "pixels at 4K" to this frame's pixels (RenderContext.res_scale).
    """
    height, width = fractal.shape[:2]
    F = np.asarray(fractal, dtype=np.float32)
    with progress.span(0.0, 0.4, "mask"):
        m = build(table, width, height)
        feather = float(_get(table, "feather")) * scale
        if feather >= 0.5:
            m = np.clip(blur(m, feather), 0.0, 1.0)
        if _get(table, "invert"):
            m = 1.0 - m
    with progress.span(0.4, 0.7, "background"):
        bg, transparent = _background(table, width, height, F, scale)
    shadow = float(_get(table, "shadow"))
    glow = float(_get(table, "glow"))
    glow_r = float(_get(table, "glow_radius")) * scale
    alpha_parts = [m]
    if shadow > 0:
        r = 24.0 * scale
        off = max(1, int(round(0.5 * r)))
        shifted = np.zeros_like(m)
        shifted[off:, off:] = m[:-off, :-off]
        s = np.clip(shadow * blur(shifted, r), 0.0, 1.0) * (1.0 - m)
        bg = bg * (1.0 - s[..., None])
        alpha_parts.append(s)
    bevel = float(_get(table, "bevel"))
    if bevel > 0:
        F = np.clip(_bevel(F, m, bevel, scale), 0.0, None)
    out = bg * (1.0 - m[..., None]) + F * m[..., None]
    outline = float(_get(table, "outline")) * scale
    if outline > 0:
        ring = _outline_ring(m, max(outline, 0.75))
        out = out * (1.0 - ring[..., None]) + hex_rgb(_get(table, "outline_color")).astype(np.float32) * ring[..., None]
        alpha_parts.append(ring)
    if glow > 0:
        g = np.clip(glow * blur(F * m[..., None], glow_r) * (1.0 - m[..., None]), 0.0, 1.0)
        out = 1.0 - (1.0 - out) * (1.0 - g)  # screen blend
        alpha_parts.append(g.max(axis=-1))
    if not transparent:
        return np.clip(out, 0.0, 1.0).astype(np.float64)
    keep = np.ones_like(m)
    for a in alpha_parts:
        keep *= 1.0 - np.clip(a, 0.0, 1.0)
    alpha = 1.0 - keep
    # `out` was composited over black, i.e. it is premultiplied by alpha.
    rgb = np.where(alpha[..., None] > 1e-6, out / np.maximum(alpha, 1e-6)[..., None], 0.0)
    return np.concatenate([np.clip(rgb, 0.0, 1.0), alpha[..., None]], axis=-1).astype(np.float64)
