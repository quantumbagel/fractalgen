"""Layers: any number of text, shape and image layers over (or cut from) a render.

A recipe's ``layers`` is a list, drawn bottom to top::

    background = { fill = "self", dim = 0.15, blur = 24 }

    [[recipe.NAME.layers]]
    type = "text"
    text = "FRACTAL"
    glow = 0.7

    [[recipe.NAME.layers]]
    type = "shape"
    shape = "circle"
    fill = "ffffff"      # a solid white disc on top
    size = 0.1
    x = 0.9
    y = 0.1

Every layer has a ``fill``:

- ``"fractal"`` (text and shapes by default): the render shows through the
  layer, a cut-out. When any layer is a cut-out, ``background`` fills the rest
  of the frame: a hex color, ``"transparent"``, ``"self"`` (the render, dimmed
  by ``dim`` and blurred by ``blur``), ``"recipe:NAME"`` or ``"palette:NAME"``.
- a hex color, ``"palette:NAME"`` (a top-to-bottom gradient) or
  ``"recipe:NAME"`` (another look rendered at the same size): painted on top.
- ``"image"`` (image layers by default): the picture's own pixels.

Each layer also has position / size / rotation, ``opacity`` and ``blend``,
and its own outline, glow, drop shadow and bevel. Lengths are pixels at 4K.

The older single ``mask = {...}`` table still works: it becomes one layer plus
the background (see :func:`from_mask`). Plugins add layer kinds with
:func:`layer_type` and shapes with ``fractalgen.shape``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import progress
from .blend import BLEND_MODES, blend
from .mask import (
    SHAPES, _bevel, _image_mask, _outline_ring, _rotate, _shape_mask, _text_mask, blur, hex_rgb, resolve_path,
)
from .registry import Registry
from .schema import Key, check_table

# ---------------------------------------------------------------- schema ----

COMMON_KEYS: dict[str, Key] = {
    "enabled": Key("Show this layer", True, "bool"),
    "name": Key("Layer name (shown in the GUI; portrait overrides can address a layer by it)", "", "string",
                advanced=True),
    "fill": Key('What fills the layer: "fractal" (the render shows through), a hex color, "palette:NAME", '
                '"recipe:NAME", or "image" (an image layer\'s own pixels)', "fractal", "fill"),
    "x": Key("Horizontal center, 0 = left edge, 1 = right edge", 0.5, "float", (0.0, 1.0)),
    "y": Key("Vertical center, 0 = top edge, 1 = bottom edge", 0.5, "float", (0.0, 1.0)),
    "size": Key("Fraction of the frame the layer fits inside", 0.8, "float", (0.02, 1.5)),
    "rotate": Key("Rotation, degrees counter-clockwise", 0.0, "float", (-180.0, 180.0), unit="deg"),
    "opacity": Key("How opaque the layer is", 1.0, "float", (0.0, 1.0)),
    "blend": Key("How the layer mixes with what is under it", "normal", "choice", choices=lambda: BLEND_MODES.keys()),
    "invert": Key("Fill everything except the layer's shape", False, "bool"),
    "feather": Key("Soften the edge", 0.0, "float", (0.0, 60.0), unit="px"),
    "outline": Key("Outline width (0 = none)", 0.0, "float", (0.0, 40.0), unit="px"),
    "outline_color": Key("Outline color", "ffffff", "color"),
    "glow": Key("The layer's colors glowing out past its edge (0 = off)", 0.0, "float", (0.0, 2.0)),
    "glow_radius": Key("Size of the glow", 40.0, "float", (2.0, 200.0), unit="px"),
    "shadow": Key("Drop shadow under the layer (0 = off, 1 = black)", 0.0, "float", (0.0, 1.0)),
    "bevel": Key("Raised, lit edges (0 = flat)", 0.0, "float", (0.0, 2.0)),
}

BACKGROUND_KEYS: dict[str, Key] = {
    "fill": Key('Behind cut-out layers: hex color, "transparent", "self" (the render, dimmed), '
                '"recipe:NAME" or "palette:NAME"', "000000", "fill"),
    "dim": Key("Brightness of an image background (self / recipe), 0..1", 0.15, "float", (0.0, 1.0)),
    "blur": Key("Blur of an image background", 0.0, "float", (0.0, 120.0), unit="px"),
}

Geometry = tuple[int, int, tuple[float, float], tuple[float, float], float]  # w, h, box, center, rotate
CoverageFn = Callable[[dict[str, Any], Geometry], np.ndarray]


@dataclass(frozen=True)
class LayerType:
    """A kind of layer: ``coverage(layer, geometry) -> (H, W)`` in 0..1, rotation included."""

    coverage: CoverageFn
    keys: dict[str, Key]
    doc: str = ""
    label: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)  # per-type defaults for common keys (e.g. fill)
    start: dict[str, Any] = field(default_factory=dict)  # values a layer added in the GUI starts with
    # Optional ``pixels(layer, geometry) -> (rgb, alpha)``: the layer's own colors (fill = "image").
    pixels: Callable[[dict[str, Any], Geometry], tuple[np.ndarray, np.ndarray]] | None = None

    def all_keys(self) -> dict[str, Key]:
        out = dict(COMMON_KEYS)
        for k, v in self.defaults.items():
            base = out[k]
            out[k] = Key(base.doc, v, base.kind, base.range, base.choices, base.label, base.unit, base.advanced)
        out.update(self.keys)
        return out

    def schema(self, name: str) -> dict[str, Any]:
        from .labels import label_for

        return {
            "name": name,
            "label": self.label or label_for(name),
            "doc": self.doc,
            "keys": [k.schema(n) for n, k in self.all_keys().items()],
            "common": list(COMMON_KEYS),
            "start": dict(self.start),
        }


LAYER_TYPES: Registry[LayerType] = Registry("layer type")


def layer_type(name: str, keys: dict[str, Key], doc: str = "", *, label: str = "", defaults: dict[str, Any] | None = None,
               start: dict[str, Any] | None = None, pixels=None, replace: bool = False):
    """Register a layer kind; decorates its coverage function."""

    def wrap(fn: CoverageFn) -> CoverageFn:
        LAYER_TYPES.register(name, LayerType(fn, dict(keys), doc or (fn.__doc__ or "").strip(), label,
                                             dict(defaults or {}), dict(start or {}), pixels), replace=replace)
        return fn

    return wrap


# ---------------------------------------------------------------- built-in kinds ----

@layer_type("text", {
    "text": Key('The words ("\\n" breaks lines)', "", "string"),
    "font": Key('Font family (e.g. "Arial Black", "Avenir Next Heavy") or a .ttf/.otf/.ttc path; '
                "empty = a heavy system font", "", "font"),
}, "Words cut from the render or painted over it", label="Text", start={"text": "Hello", "size": 0.6})
def _text_coverage(layer, g):
    w, h, box, center, rotate = g
    m = _text_mask(str(layer.get("text", "")), str(layer.get("font", "")), w, h, box, center)
    return _rotate(m, rotate, center)


@layer_type("shape", {
    "shape": Key("Which shape", "circle", "choice", choices=lambda: SHAPES.names()),
}, "A circle, heart, star, ...", label="Shape", start={"shape": "circle", "size": 0.6})
def _shape_coverage(layer, g):
    w, h, box, center, rotate = g
    return _shape_mask(str(layer.get("shape") or "circle"), w, h, box, center, rotate)


def _image_pixels(layer, g):
    from PIL import Image

    w, h, box, center, rotate = g
    path = str(layer.get("image") or "")
    src = resolve_path(path)
    if not path or not src.exists():
        raise FileNotFoundError(f"image layer: no file {path!r}")
    with Image.open(src) as im:
        rgba = im.convert("RGBA")
    k = min(box[0] / rgba.width, box[1] / rgba.height)
    nw, nh = max(1, round(rgba.width * k)), max(1, round(rgba.height * k))
    rgba = rgba.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(rgba, (round(center[0] - nw / 2.0), round(center[1] - nh / 2.0)))
    if rotate:
        canvas = canvas.rotate(rotate, resample=Image.BICUBIC, center=center)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    a = arr[..., 3]
    rgb = np.where(a[..., None] > 1e-6, arr[..., :3], 0.0)  # PIL rotate/resize keep straight alpha
    return rgb, a


@layer_type("image", {
    "image": Key("Image file (PNG with transparency works best); relative paths may sit in custom/", "", "path"),
    "image_channel": Key("For cut-outs: which part of the image is the shape (auto: alpha if it has any, "
                         "else brightness)", "auto", "choice", choices=("auto", "alpha", "luma"), advanced=True),
}, "A picture: painted as-is, or used as a cut-out shape", label="Image", defaults={"fill": "image"},
    start={"size": 0.4}, pixels=_image_pixels)
def _image_coverage(layer, g):
    w, h, box, center, rotate = g
    path = str(layer.get("image") or "")
    if not path:
        return np.zeros((h, w), np.float32)
    m = _image_mask(path, str(layer.get("image_channel") or "auto"), w, h, box, center)
    return _rotate(m, rotate, center)


# ---------------------------------------------------------------- validation ----

def check_layer(raw: dict[str, Any], where: str = "layer") -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"{where} must be a table, got {raw!r}")
    kind = str(raw.get("type") or "")
    if not kind:
        raise ValueError(f"{where} needs a `type` ({', '.join(LAYER_TYPES.names())})")
    lt = LAYER_TYPES.get(kind)
    body = {k: v for k, v in raw.items() if k not in ("type", "id")}
    out = {"type": kind, **check_table(lt.all_keys(), body, f"{where}({kind})")}
    if "blend" in out and out["blend"] not in BLEND_MODES:
        raise ValueError(f"{where}.blend={out['blend']!r}: one of {list(BLEND_MODES)}")
    return out


def check_layers(layers: Any) -> list[dict[str, Any]]:
    if not layers:
        return []
    if isinstance(layers, dict):
        raise TypeError("layers must be a list of tables ([[recipe.NAME.layers]])")
    return [check_layer(raw, f"layers[{i}]") for i, raw in enumerate(layers)]


def check_background(table: dict[str, Any] | None) -> dict[str, Any]:
    return check_table(BACKGROUND_KEYS, table, "background")


def patch_layers(layers: list[dict[str, Any]], patch: Any) -> list[dict[str, Any]]:
    """Portrait / override edits: a list replaces; a table patches layers by ``name`` or index."""
    if patch is None:
        return layers
    if isinstance(patch, (list, tuple)):
        return check_layers(patch)
    out = [dict(l) for l in layers]
    for key, edit in patch.items():
        idx = next((i for i, l in enumerate(out) if l.get("name") == key), None)
        if idx is None:
            try:
                idx = int(key)
            except (TypeError, ValueError):
                raise KeyError(f"layers: no layer named {key!r}") from None
            if not -len(out) <= idx < len(out):
                raise IndexError(f"layers: no layer {idx} (there are {len(out)})")
        merged = {**out[idx], **(edit or {})}
        out[idx] = check_layer({k: v for k, v in merged.items() if v is not None}, f"layers[{key}]")
    return out


# ---------------------------------------------------------------- legacy mask ----

_LEGACY_BG = {"background": "fill", "background_dim": "dim", "background_blur": "blur"}
_LEGACY_SOURCES = (("shape", "shape"), ("image", "image"), ("text", "text"))  # old build() order


def from_mask(mask: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """A legacy ``mask`` table as (layers, background). Text, shape and image set together
    become one layer each with the same placement."""
    mask = dict(mask or {})
    background = {new: mask.pop(old) for old, new in _LEGACY_BG.items() if old in mask}
    shared = {k: v for k, v in mask.items() if k in COMMON_KEYS}
    layers = []
    for kind, key in _LEGACY_SOURCES:
        if not mask.get(key):
            continue
        layer = {"type": kind, key: mask[key], **shared}
        if kind == "text" and "font" in mask:
            layer["font"] = mask["font"]
        if kind == "image":
            layer["fill"] = "fractal"
            if "image_channel" in mask:
                layer["image_channel"] = mask["image_channel"]
        layers.append(layer)
    return check_layers(layers), check_background(background)


def to_mask(layers: list[dict[str, Any]], background: dict[str, Any]) -> dict[str, Any] | None:
    """The legacy ``mask`` table for (layers, background), or None if it needs the new form."""
    if not layers:
        return {} if not background else None
    if len(layers) != 1:
        return None
    layer = dict(layers[0])
    kind = layer.pop("type")
    fill = layer.pop("fill", "image" if kind == "image" else "fractal")
    if fill != "fractal" or layer.get("enabled") is False:
        return None
    layer.pop("enabled", None)
    if any(k in layer for k in ("opacity", "blend", "name")):
        return None
    out = dict(layer)
    for old, new in _LEGACY_BG.items():
        if new in background:
            out[old] = background[new]
    return out


# ---------------------------------------------------------------- rendering ----

_DEPTH = [0]  # keeps recipe:NAME fills from recursing forever


def _get(layer: dict[str, Any], key: str) -> Any:
    if key in layer:
        return layer[key]
    lt = LAYER_TYPES.get(layer["type"])
    if key in lt.defaults:
        return lt.defaults[key]
    k = lt.keys.get(key) or COMMON_KEYS[key]
    return k.neutral


def geometry(layer: dict[str, Any], width: int, height: int) -> Geometry:
    size = float(_get(layer, "size"))
    box = (max(1.0, size * width), max(1.0, size * height))
    center = (float(_get(layer, "x")) * width, float(_get(layer, "y")) * height)
    return width, height, box, center, float(_get(layer, "rotate"))


def coverage(layer: dict[str, Any], width: int, height: int, scale: float, *, raw: bool = False) -> np.ndarray:
    """The layer's (H, W) coverage in 0..1: its shape, feathered and inverted (``raw``: neither)."""
    lt = LAYER_TYPES.get(layer["type"])
    g = geometry(layer, width, height)
    fill = str(_get(layer, "fill"))
    if fill == "image" and lt.pixels is not None:
        m = lt.pixels(layer, g)[1]
    else:
        m = lt.coverage(layer, g)
    m = np.asarray(m, dtype=np.float32)
    if raw:
        return m
    feather = float(_get(layer, "feather")) * scale
    if feather >= 0.5:
        m = np.clip(blur(m, feather), 0.0, 1.0)
    if _get(layer, "invert"):
        m = 1.0 - m
    return m


def _image_source(spec: str, width: int, height: int, fractal: np.ndarray, what: str) -> np.ndarray | None:
    """RGB for "self" / "fractal" / "recipe:NAME" / "palette:NAME" / hex; None for others."""
    low = spec.strip().lower()
    if low in ("self", "fractal"):
        return fractal
    if low.startswith("palette:"):
        from .palette import get_palette

        ramp = np.asarray(get_palette(spec.split(":", 1)[1].strip()).sample(np.linspace(0.0, 1.0, height)))
        return np.broadcast_to(ramp[:, None, :3], (height, width, 3)).astype(np.float32)
    if low.startswith("recipe:"):
        from .recipe import RECIPES

        if _DEPTH[0] >= 2:
            raise ValueError(f"{what}: recipe fills nest too deeply")
        _DEPTH[0] += 1
        try:
            img = np.asarray(RECIPES.get(spec.split(":", 1)[1].strip()).render(width, height))
        finally:
            _DEPTH[0] -= 1
        return img[..., :3].astype(np.float32)
    try:
        color = hex_rgb(spec)
    except ValueError:
        raise ValueError(
            f"{what}={spec!r}: use a hex color, "
            + ("transparent, self" if what == "background" else "fractal, image")
            + ", recipe:NAME or palette:NAME"
        ) from None
    return np.broadcast_to(color.astype(np.float32), (height, width, 3)).copy()


def _background(background: dict[str, Any], width: int, height: int, fractal: np.ndarray, scale: float):
    """(background (H, W, 3), transparent?)."""
    spec = str(background.get("fill", BACKGROUND_KEYS["fill"].neutral)).strip()
    if spec.lower() in ("transparent", "none", "alpha"):
        return np.zeros((height, width, 3), np.float32), True
    img = _image_source(spec, width, height, fractal, "background")
    if spec.lower() in ("self", "fractal") or spec.lower().startswith("recipe:"):
        img = blur(img, float(background.get("blur", BACKGROUND_KEYS["blur"].neutral)) * scale)
        img = np.asarray(img, dtype=np.float32) * float(background.get("dim", BACKGROUND_KEYS["dim"].neutral))
    return np.asarray(img, dtype=np.float32), False


def is_cutout(layer: dict[str, Any]) -> bool:
    return str(_get(layer, "fill")).lower() in ("fractal", "self")


def active(layers: list[dict[str, Any]] | None) -> bool:
    return any(l.get("enabled", True) for l in layers or ())


def apply_layers(fractal: np.ndarray, layers: list[dict[str, Any]], background: dict[str, Any] | None,
                 scale: float) -> np.ndarray:
    """Draw ``layers`` (bottom to top) over / out of ``fractal`` (H, W, 3). Returns RGB, or RGBA
    when a cut-out sits on a transparent background. ``scale`` = RenderContext.res_scale."""
    shown = [l for l in layers or () if l.get("enabled", True)]
    F = np.asarray(fractal, dtype=np.float32)
    if not shown:
        return np.asarray(fractal, dtype=np.float64)
    height, width = F.shape[:2]
    background = background or {}
    transparent = False
    if any(is_cutout(l) for l in shown):
        with progress.span(0.0, 0.3, "background"):
            out, transparent = _background(background, width, height, F, scale)
        alpha = np.zeros((height, width), np.float32) if transparent else None
    else:
        out, alpha = F.copy(), None
    n = len(shown)
    for i, layer in enumerate(shown):
        with progress.span(0.3 + 0.7 * i / n, 0.3 + 0.7 * (i + 1) / n, "layers"):
            out, alpha = _draw(out, alpha, F, layer, width, height, scale)
    out = np.asarray(out, dtype=np.float64)
    if not transparent:
        return np.clip(out, 0.0, 1.0)
    a = np.clip(alpha, 0.0, 1.0).astype(np.float64)
    # `out` was composited over black, i.e. it is premultiplied by alpha.
    rgb = np.where(a[..., None] > 1e-6, out / np.maximum(a, 1e-6)[..., None], 0.0)
    return np.concatenate([np.clip(rgb, 0.0, 1.0), a[..., None]], axis=-1)


def _over(out, alpha, color, weight):
    """Lerp ``out`` toward ``color`` by ``weight`` (H, W); alpha accumulates the coverage."""
    w = weight[..., None]
    out = out * (1.0 - w) + color * w
    if alpha is not None:
        alpha = alpha * (1.0 - weight) + weight
    return out, alpha


def _draw(out, alpha, F, layer, width, height, scale):
    lt = LAYER_TYPES.get(layer["type"])
    m = coverage(layer, width, height, scale)
    opacity = float(_get(layer, "opacity"))
    fill = str(_get(layer, "fill")).strip()
    if fill.lower() in ("fractal", "self"):
        C = F
    elif fill.lower() == "image":
        if lt.pixels is None:
            raise ValueError(f"{layer['type']} layer: fill 'image' needs an image layer")
        C = lt.pixels(layer, geometry(layer, width, height))[0]
    else:
        C = _image_source(fill, width, height, F, f"layers.fill ({layer['type']})")
    shadow = float(_get(layer, "shadow")) * opacity
    if shadow > 0:
        r = 24.0 * scale
        off = max(1, int(round(0.5 * r)))
        shifted = np.zeros_like(m)
        shifted[off:, off:] = m[:-off, :-off]
        s = np.clip(shadow * blur(shifted, r), 0.0, 1.0) * (1.0 - m)
        out, alpha = _over(out, alpha, 0.0, s)
    bevel = float(_get(layer, "bevel"))
    if bevel > 0:
        C = np.clip(_bevel(C, m, bevel, scale), 0.0, None)
    mode = str(_get(layer, "blend"))
    a = m * opacity
    if mode == "normal":
        out, alpha = _over(out, alpha, C, a)
    else:
        out = blend(out, C, a, mode)
        if alpha is not None:
            alpha = alpha * (1.0 - a) + a
    outline = float(_get(layer, "outline")) * scale
    if outline > 0:
        ring = _outline_ring(m, max(outline, 0.75)) * opacity
        out, alpha = _over(out, alpha, hex_rgb(_get(layer, "outline_color")).astype(np.float32), ring)
    glow = float(_get(layer, "glow")) * opacity
    if glow > 0:
        g = np.clip(glow * blur(C * m[..., None], float(_get(layer, "glow_radius")) * scale) * (1.0 - m[..., None]),
                    0.0, 1.0)
        out = 1.0 - (1.0 - out) * (1.0 - g)  # screen blend
        if alpha is not None:
            alpha = 1.0 - (1.0 - alpha) * (1.0 - g.max(axis=-1))
    return out, alpha


def layer_boxes(layers: list[dict[str, Any]], width: int, height: int) -> list[list[float] | None]:
    """Each layer's bounding box as fractions [x0, y0, x1, y1] (None when empty), for the GUI."""
    out = []
    for layer in layers or ():
        try:
            m = coverage(layer, width, height, 1.0, raw=True)
        except Exception:
            out.append(None)
            continue
        ys, xs = np.nonzero(m > 0.05)
        if not len(xs):
            out.append(None)
        else:
            out.append([xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height])
    return out


def describe(layer: dict[str, Any]) -> str:
    """Short human label for a layer (GUI list)."""
    if layer.get("name"):
        return str(layer["name"])
    kind = layer.get("type", "")
    if kind == "text":
        t = str(layer.get("text") or "").replace("\\n", " ").replace("\n", " ")
        return f'"{t[:24]}"' if t else "Text"
    if kind == "shape":
        return str(layer.get("shape") or "circle").capitalize()
    if kind == "image":
        p = str(layer.get("image") or "")
        return p.rsplit("/", 1)[-1] or "Image"
    return kind.capitalize()


__all__ = [
    "LAYER_TYPES", "COMMON_KEYS", "BACKGROUND_KEYS", "LayerType", "layer_type", "check_layer", "check_layers",
    "check_background", "patch_layers", "from_mask", "to_mask", "apply_layers", "layer_boxes", "coverage",
    "active", "is_cutout", "describe",
]
