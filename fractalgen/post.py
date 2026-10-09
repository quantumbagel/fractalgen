"""Post-processing that works on any render, whatever generator made it.

A recipe's ``post`` is an ordered list of effects; each runs on the result of
the one before, so effects can repeat and their order matters::

    [[recipe.NAME.post]]
    op = "bloom"
    amount = 0.4

    [[recipe.NAME.post]]
    op = "vignette"
    value = 0.35
    mix = 0.8          # every effect: blend its result with its input (0..1)
    # enabled = false  # every effect: keep it in the list but skip it

The older table form still works and runs in registry order (the order below)::

    post = { exposure = 0.3, saturation = 1.2, hue_shift = 20, vignette = 0.35 }

``--post key=value`` on the CLI sets one effect's value (adding it if needed).
Single-value effects take ``value``; multi-parameter ones (bloom, grain, ...)
name their parameters. Add your own with::

    @fractalgen.post_op("posterize2", neutral=0, range=(0, 16))
    def posterize2(rgb, levels, ctx):
        return np.round(rgb * levels) / levels if levels else rgb

    @fractalgen.post_op("glow2", params={"amount": Key("How strong.", 0.5), "radius": Key("Size", 20.0, unit="px")})
    def glow2(rgb, p, ctx):  # p = {"amount": ..., "radius": ...}
        ...

Lengths with ``unit="px"`` are pixels at 4K: multiply by ``ctx.res_scale`` so
previews and exports match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .color import hsv_to_rgb
from .palette import get_palette
from .registry import Registry
from .schema import Key

PostFn = Callable[[np.ndarray, Any, Any], np.ndarray]

CATEGORIES = ("color", "light", "texture", "transform")


@dataclass(frozen=True)
class PostOp:
    fn: PostFn
    doc: str
    neutral: Any  # value at which the op does nothing (GUI reset / default)
    range: tuple | None = None  # slider hint
    kind: str = ""  # GUI hint: "float", "bool", "palette", "color", "int"
    params: dict[str, Key] | None = None  # multi-parameter effect: fn gets {name: value}
    start: Any = None  # value a newly added effect starts at (GUI "+ Add"); default: neutral
    category: str = "color"
    label: str = ""
    hidden: bool = False  # not offered in the GUI (legacy keys)

    def __iter__(self):  # (fn, doc) unpacking for simple callers
        return iter((self.fn, self.doc))

    @property
    def primary(self) -> str:
        """Parameter a bare value (``post = { bloom = 0.4 }``) sets."""
        return next(iter(self.params)) if self.params else "value"

    def schema(self, name: str) -> dict[str, Any]:
        from .labels import label_for

        if self.params:
            params = [k.schema(n) for n, k in self.params.items()]
        else:
            kind = self.kind or "float"
            params = [Key(self.doc, self.neutral, kind, self.range, label="Amount").schema("value")]
            params[0]["neutral"] = _jsonable(self.neutral)
        return {
            "name": name,
            "label": self.label or label_for(name),
            "doc": self.doc,
            "category": self.category,
            "params": params,
            "start": _jsonable(self.start_item()),
            "hidden": self.hidden,
        }

    def start_item(self) -> dict[str, Any]:
        if self.params:
            out = {n: k.neutral for n, k in self.params.items()}
            if isinstance(self.start, dict):
                out.update(self.start)
            return out
        return {"value": self.start if self.start is not None else self.neutral}


def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


POST_OPS: Registry[PostOp] = Registry("post op")


def post_op(
    name: str, doc: str = "", *, neutral: Any = None, range: tuple | None = None, kind: str = "",
    params: dict[str, Key] | None = None, start: Any = None, category: str = "color", label: str = "",
    hidden: bool = False, replace: bool = False,
) -> Callable[[PostFn], PostFn]:
    def wrap(fn: PostFn) -> PostFn:
        k = kind or ("bool" if isinstance(neutral, bool) else "int" if isinstance(neutral, int) and range and len(range) > 2
                     else "float" if isinstance(neutral, (int, float)) else "")
        op = PostOp(fn, doc or (fn.__doc__ or "").strip(), neutral, range, k, dict(params) if params else None,
                    start, category, label, hidden)
        POST_OPS.register(name, op, replace=replace)
        return fn

    return wrap


effect = post_op  # the same thing, under the name the GUI uses


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


def _scale(ctx: Any) -> float:
    return float(getattr(ctx, "res_scale", 1.0) or 1.0)


@post_op("palette", "Repaints the picture with new colors, matched by brightness.", kind="palette", start="ember", label="Recolor")
def _palette(rgb, spec, ctx):
    return get_palette(spec).sample(np.clip(_luma(rgb), 0.0, 1.0))


@post_op("palette_mix", "How much of the recolor to use (0 keeps the original).", neutral=1.0, range=(0.0, 1.0),
         hidden=True)
def _palette_mix(rgb, amount, ctx):
    return rgb  # consumed by apply_post


@post_op("invert", "Turns the colors into their opposites, like a film negative.", neutral=False, start=True)
def _invert(rgb, on, ctx):
    return 1.0 - rgb if on else rgb


@post_op("hue_shift", "Shifts every color around the color wheel.", neutral=0.0, range=(-180.0, 180.0), start=30.0)
def _hue_shift(rgb, deg, ctx):
    h, s, v = _rgb_to_hsv(rgb)
    return hsv_to_rgb(h + float(deg) / 360.0, s, v)


@post_op("saturation", "Makes colors richer, or fades them toward gray at 0.", neutral=1.0, range=(0.0, 3.0),
         start=1.3)
def _saturation(rgb, amount, ctx):
    y = _luma(rgb)[..., None]
    return y + (rgb - y) * float(amount)


@post_op("exposure", "Brightens or darkens the whole picture. Each stop doubles or halves it.", neutral=0.0, range=(-3.0, 3.0), start=0.3,
         category="light", label="Brightness")
def _exposure(rgb, stops, ctx):
    return rgb * (2.0 ** float(stops))


@post_op("contrast", "Pushes darks and lights further apart, or pulls them together below 1.", neutral=1.0, range=(0.0, 3.0), start=1.2,
         category="light")
def _contrast(rgb, amount, ctx):
    return (rgb - 0.5) * float(amount) + 0.5


@post_op("gamma", "Above 1 opens up the shadows, and below 1 deepens them.", neutral=1.0, range=(0.2, 3.0), start=1.2,
         category="light")
def _gamma(rgb, g, ctx):
    gv = float(g)
    if gv <= 0:
        raise ValueError("post gamma must be > 0")
    return np.power(np.clip(rgb, 0.0, 1.0), 1.0 / gv)


@post_op("tint", "Washes the picture with a color, like a photo filter.", kind="color", start="ffd9a0")
def _tint(rgb, color, ctx):
    from .palette import hex_to_rgb

    c = hex_to_rgb(color) if isinstance(color, str) else tuple(float(x) for x in color)
    return rgb * np.asarray(c)


@post_op("vignette", "Darkens the corners to draw the eye inward. Around 0.3 is subtle.", neutral=0.0, range=(0.0, 1.0), start=0.35,
         category="light")
def _vignette(rgb, amount, ctx):
    h, w = rgb.shape[:2]
    y = np.linspace(-1.0, 1.0, h)[:, None]
    x = np.linspace(-1.0, 1.0, w)[None, :]
    r2 = (x * x + y * y) / 2.0
    return rgb * (1.0 - float(amount) * r2)[..., None]


@post_op("flip_x", "Flips the picture left to right.", neutral=False, start=True, category="transform")
def _flip_x(rgb, on, ctx):
    return rgb[:, ::-1] if on else rgb


@post_op("flip_y", "Flips the picture upside down.", neutral=False, start=True, category="transform")
def _flip_y(rgb, on, ctx):
    return rgb[::-1] if on else rgb




# ---------------------------------------------------------------- multi-parameter effects ----

@post_op("bloom", "A soft glow spilling out of the bright areas.", category="light", params={
    "amount": Key("How strong the glow is.", 0.6, range=(0.0, 3.0)),
    "threshold": Key("Only areas brighter than this glow.", 0.55, range=(0.0, 0.95)),
    "radius": Key("How far the glow spreads.", 60.0, range=(2.0, 300.0), unit="px"),
})
def _bloom(rgb, p, ctx):
    from .mask import blur

    thr = float(p["threshold"])
    w = np.clip((_luma(rgb) - thr) / max(1e-6, 1.0 - thr), 0.0, 1.0)
    return rgb + float(p["amount"]) * blur(rgb * w[..., None], float(p["radius"]) * _scale(ctx))


@post_op("blur", "Softens the whole picture.", category="texture", params={
    "radius": Key("How soft.", 8.0, range=(0.5, 200.0), unit="px"),
})
def _blur(rgb, p, ctx):
    from .mask import blur

    return blur(rgb, float(p["radius"]) * _scale(ctx)).astype(np.float64)


@post_op("sharpen", "Makes fine detail crisper.", category="texture", params={
    "amount": Key("How strong.", 0.6, range=(0.0, 3.0)),
    "radius": Key("Size of the detail it sharpens.", 4.0, range=(0.5, 40.0), unit="px"),
})
def _sharpen(rgb, p, ctx):
    from .mask import blur

    return rgb + float(p["amount"]) * (rgb - blur(rgb, float(p["radius"]) * _scale(ctx)))


@post_op("grain", "Adds film grain.", category="texture", params={
    "amount": Key("How visible the grain is.", 0.06, range=(0.0, 0.4)),
    "size": Key("Size of each grain.", 2.0, range=(1.0, 12.0), unit="px"),
    "color": Key("Colored grain instead of gray.", False, "bool"),
    "seed": Key("Change it for a different grain pattern.", 0, "int", (0, 999, 1), advanced=True),
})
def _grain(rgb, p, ctx):
    from scipy.ndimage import zoom

    h, w = rgb.shape[:2]
    px = max(1.0, float(p["size"]) * _scale(ctx))
    gh, gw = max(2, math.ceil(h / px)), max(2, math.ceil(w / px))
    chans = 3 if p["color"] else 1
    noise = np.random.default_rng(int(p["seed"])).standard_normal((gh, gw, chans)).astype(np.float32)
    if (gh, gw) != (h, w):
        noise = zoom(noise, (h / gh, w / gw, 1), order=1, mode="nearest", grid_mode=True)[:h, :w]
        if noise.shape[:2] != (h, w):
            noise = np.pad(noise, ((0, h - noise.shape[0]), (0, w - noise.shape[1]), (0, 0)), mode="edge")
    weight = 1.0 - 0.6 * np.abs(2.0 * np.clip(_luma(rgb), 0.0, 1.0) - 1.0)  # strongest in the midtones
    return rgb + float(p["amount"]) * noise * weight[..., None]


@post_op("chromatic_aberration", "Red and blue fringes toward the edges, like an old lens.", category="texture", params={
    "amount": Key("How wide the fringes get in the corners.", 8.0, range=(0.0, 60.0), unit="px"),
})
def _chromatic_aberration(rgb, p, ctx):
    from scipy.ndimage import map_coordinates

    h, w = rgb.shape[:2]
    shift = float(p["amount"]) * _scale(ctx)
    if shift < 0.05:
        return rgb
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    k = shift / math.hypot(cx, cy)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.array(rgb, dtype=np.float64, copy=True)
    for ch, s in ((0, 1.0 - k), (2, 1.0 + k)):
        coords = [cy + (yy - cy) * s, cx + (xx - cx) * s]
        out[..., ch] = map_coordinates(rgb[..., ch], coords, order=1, mode="nearest")
    return out


@post_op("gradient", "Lays a color gradient over the picture.", category="color", params={
    "palette": Key("Colors of the gradient.", "twilight", "palette"),
    "angle": Key("Direction of the gradient. 90 runs top to bottom, 0 left to right.", 90.0, range=(-180.0, 180.0), unit="deg"),
    "blend": Key("How the gradient mixes with the picture.", "soft_light", "choice",
                 choices=lambda: __import__("fractalgen.blend", fromlist=["BLEND_MODES"]).BLEND_MODES.keys()),
})
def _gradient(rgb, p, ctx):
    from .blend import blend

    h, w = rgb.shape[:2]
    th = math.radians(float(p["angle"]))
    ys, xs = np.linspace(-0.5, 0.5, h)[:, None], np.linspace(-0.5, 0.5, w)[None, :]
    aspect = w / h
    t = xs * aspect * math.cos(th) + ys * math.sin(th)
    lo, hi = -0.5 * (aspect * abs(math.cos(th)) + abs(math.sin(th))), 0.5 * (aspect * abs(math.cos(th)) + abs(math.sin(th)))
    t = (t - lo) / max(hi - lo, 1e-9)
    grad = np.asarray(get_palette(p["palette"]).sample(np.clip(t, 0.0, 1.0)))[..., :3]
    return blend(rgb, grad, 1.0, str(p["blend"]))


@post_op("posterize", "Flattens the colors into a few bands, like a screen print.", neutral=0, range=(0, 32, 1), kind="int", start=6)
def _posterize(rgb, levels, ctx):
    n = int(levels)
    if n < 2:
        return rgb
    return np.round(np.clip(rgb, 0.0, 1.0) * (n - 1)) / (n - 1)


# ---------------------------------------------------------------- the effect list ----

_ITEM_KEYS = ("op", "enabled", "mix", "id")


def check_post(post: Any) -> list[dict[str, Any]]:
    """Normalize a ``post`` (ordered list, or the legacy table) to a validated effect list.

    Legacy tables run in registry order; ``palette_mix`` becomes the palette's ``mix``.
    """
    if not post:
        return []
    if isinstance(post, dict):
        unknown = [k for k in post if k not in POST_OPS]
        if unknown:
            raise KeyError(f"Unknown post op(s) {unknown}. Known: {POST_OPS.names()}")
        if post.get("palette_mix") is not None and post.get("palette") is None:
            raise ValueError("post.palette_mix requires post.palette")
        items = []
        for name in POST_OPS:
            v = post.get(name)
            if v is None or name == "palette_mix":
                continue
            op = POST_OPS.get(name)
            item = dict(v, op=name) if isinstance(v, dict) and op.params else {"op": name, op.primary: v}
            if name == "palette" and post.get("palette_mix") is not None:
                item["mix"] = float(post["palette_mix"])
            items.append(item)
        post = items
    if not isinstance(post, (list, tuple)):
        raise TypeError(f"post must be a list of effects (or a table), got {type(post).__name__}")
    out = []
    for i, raw in enumerate(post):
        if not isinstance(raw, dict) or "op" not in raw:
            raise ValueError(f"post[{i}] must be a table with an 'op' key, got {raw!r}")
        name = str(raw["op"])
        op = POST_OPS.get(name)
        item: dict[str, Any] = {"op": name}
        for k, v in raw.items():
            if k in ("op", "id") or v is None:
                continue
            if k == "enabled":
                item[k] = bool(v)
            elif k == "mix":
                m = float(v)
                if not math.isfinite(m):
                    raise ValueError(f"post[{i}] ({name}).mix must be finite")
                item[k] = min(1.0, max(0.0, m))
            elif op.params:
                if k == "value":
                    k = op.primary
                if k not in op.params:
                    raise KeyError(f"post[{i}] ({name}): unknown parameter {k!r}. Known: {list(op.params)}")
                item[k] = op.params[k].coerce(f"post.{name}.{k}", v)
            elif k == "value":
                item[k] = v
            else:
                raise KeyError(f"post[{i}] ({name}) takes `value` (and enabled / mix), not {k!r}")
        out.append(item)
    return out


def patch_post(items: list[dict[str, Any]], patch: dict[str, Any]) -> list[dict[str, Any]]:
    """``--post key=value`` style edits on an effect list: set the first effect of that kind
    (appending it when missing); ``None`` removes every effect of that kind."""
    out = [dict(it) for it in items]
    for name, v in check_patch(patch).items():
        if v is None:
            out = [it for it in out if it["op"] != name]
            continue
        if name == "palette_mix":
            for it in out:
                if it["op"] == "palette":
                    it["mix"] = float(v)
                    break
            continue
        new = check_post({name: v})
        hit = next((it for it in out if it["op"] == name), None)
        if hit is None:
            out.extend(new)
        else:
            mix = hit.get("mix")
            hit.clear()
            hit.update(new[0] if new else {"op": name})
            if mix is not None and "mix" not in hit:
                hit["mix"] = mix
    return out


def post_to_legacy(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The legacy ``post`` table for an effect list, or None if order / repeats / switches need the list."""
    order = list(POST_OPS)
    out: dict[str, Any] = {}
    last = -1
    for it in items:
        name = it["op"]
        if name in out or order.index(name) < last or it.get("enabled") is False:
            return None
        if "mix" in it and name != "palette":
            return None
        last = order.index(name)
        op = POST_OPS.get(name)
        params = {k: v for k, v in it.items() if k not in _ITEM_KEYS}
        out[name] = params if op.params else params.get("value", op.neutral)
        if "mix" in it:
            out["palette_mix"] = it["mix"]
    return out


def check_patch(patch: dict[str, Any]) -> dict[str, Any]:
    unknown = [k for k in patch if k not in POST_OPS]
    if unknown:
        raise KeyError(f"Unknown post op(s) {unknown}. Known: {POST_OPS.names()}")
    return dict(patch)


def apply_post(rgb: np.ndarray, post: Any, ctx: Any = None) -> np.ndarray:
    """Run an effect list (or a legacy ``post`` table) over ``rgb``; RGBA keeps its alpha."""
    items = check_post(post)
    if not items:
        return rgb
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim == 3 and rgb.shape[-1] == 4:  # transparent mask output: finish the color, keep alpha
        return np.concatenate([apply_post(rgb[..., :3], items, ctx), rgb[..., 3:]], axis=-1)
    out = rgb
    for item in items:
        if item.get("enabled") is False:
            continue
        op = POST_OPS.get(item["op"])
        if op.params:
            value = {n: item.get(n, k.neutral) for n, k in op.params.items()}
        else:
            value = item.get("value", op.neutral)
            if value is None:
                continue
        res = np.asarray(op.fn(out, value, ctx), dtype=np.float64)
        mix = item.get("mix", 1.0)
        out = res if mix >= 1.0 else out * (1.0 - mix) + res * mix
    return np.clip(out, 0.0, 1.0)
