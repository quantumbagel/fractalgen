"""Write recipes back out as TOML (``fork``, ``try --emit``, the GUI's save)."""

from __future__ import annotations

import math
import string
from pathlib import Path
from typing import Any

from .generators.base import GENERATORS
from .recipe import Recipe


def toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, complex):
        return f"[{v.real!r}, {v.imag!r}]"
    if isinstance(v, float):
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return "nan" if math.isnan(v) else repr(v)
    if isinstance(v, int):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {toml_value(x)}" for k, x in v.items() if x is not None) + " }"
    if hasattr(v, "hex_stops"):  # Palette object
        return toml_value(v.hex_stops())
    return '"' + "".join(_ESCAPES.get(ch) or (f"\\u{ord(ch):04x}" if ord(ch) < 32 or ord(ch) == 127 else ch)
                         for ch in str(v)) + '"'


# TOML basic strings: escape quotes, backslashes and control characters (newlines in mask text).
_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r", "\b": "\\b", "\f": "\\f"}


def _key(k: str) -> str:
    bare = string.ascii_letters + string.digits + "_-"
    if k and all(c in bare for c in k):
        return k
    return f'"{k}"'


def _recipe_table(name: str, *suffix: str) -> str:
    return "[" + ".".join(_key(s) for s in ("recipe", name, *suffix)) + "]"


def _array_table(name: str, *suffix: str) -> str:
    return "[" + _recipe_table(name, *suffix) + "]"


def _line(key: str, value: Any, doc: str = "") -> str:
    comment = f"  # {doc}" if doc else ""
    if value is None:
        return f"# {key} = (unset){comment}"
    return f"{_key(key)} = {toml_value(value)}{comment}"


def recipe_to_toml(src: Recipe, name: str, *, width: int = 3840, height: int = 2160) -> str:
    """A self-contained recipe with every generator parameter written out."""
    from .layers import BACKGROUND_KEYS, to_mask
    from .mask import MASK_KEYS
    from .post import POST_OPS, post_to_legacy
    from .tone import TONE_KEYS

    spec, params, view = src.resolve(width, height)
    out = [
        f"# Forked from `{src.name}`" + (f" ({src.source})" if src.source else ""),
        "# Every parameter is listed with its current value. Edit anything; delete a line",
        "# to fall back to the generator default. `fractalgen generators "
        f"{spec.name}` documents them.",
        "",
        _recipe_table(name),
        _line("generator", spec.name),
    ]
    if src.description:
        out.append(_line("description", src.description))
    if "palette" in params:
        out.append(_line("palette", params["palette"], "any palette name (+ _r to reverse) or list of hex stops"))
    if src.tone_from and spec.uses_view:
        out.append(_line("tone_from", src.tone_from, "keep this recipe's exposure when the view differs from it"))
    out.append("# tags = [\"keep\"]  # opt into preset batches (keep = 8K, vip = 16K)")

    if spec.uses_view:
        out += [
            "",
            _recipe_table(name, "view"),
            _line("center", view.center, "complex-plane center [re, im]"),
            _line("half_width", view.half_width, "horizontal half-extent; smaller = deeper zoom"),
            _line("rotation", view.rotation, "degrees counter-clockwise"),
            _line("fit", view.fit, "fill | cover | contain"),
        ]

    out += ["", _recipe_table(name, "params")]
    for pname, p in spec.params.items():
        if pname == "palette":
            continue
        out.append(_line(pname, params.get(pname), p.doc))

    legacy_post = post_to_legacy(src.post)
    if legacy_post is not None:
        out += ["", _recipe_table(name, "post"), "# Finishing effects for any generator; uncomment to use. For repeats or "
                "your own order use [[recipe.NAME.post]] tables (see `fractalgen post`)."]
        for op, pop in POST_OPS.items():
            if pop.hidden and op not in legacy_post:
                continue
            if op in legacy_post:
                out.append(_line(op, legacy_post[op], pop.doc))
            elif not pop.params:
                out.append(f"# {op} = ...  # {pop.doc}")
            else:
                out.append(f"# {op} = {{ {', '.join(f'{k} = {toml_value(v.neutral)}' for k, v in pop.params.items())} }}"
                           f"  # {pop.doc}")
    else:
        out += ["", "# Finishing effects, applied in this order."]
        for item in src.post:
            out.append(_array_table(name, "post"))
            out += [_line(k, v) for k, v in item.items()]
    out += ["", _recipe_table(name, "tone"), "# Auto-exposure adjustments; omitted keys keep the look's default."]
    for key, tk in TONE_KEYS.items():
        if key in src.tone:
            out.append(_line(key, src.tone[key], tk.doc))
        else:
            out.append(f"# {key} = {toml_value(tk.neutral)}  # {tk.doc}")

    legacy_mask = to_mask(src.layers, src.background)
    if legacy_mask is not None:
        out += ["", _recipe_table(name, "mask"), "# Cut the render out with text / a shape / an image; set text, shape "
                "or image to use. For several layers use [[recipe.NAME.layers]] tables (see `fractalgen mask`)."]
        for key, mk in MASK_KEYS.items():
            if key in legacy_mask:
                out.append(_line(key, legacy_mask[key], mk.doc))
            else:
                out.append(f"# {key} = {toml_value(mk.neutral)}  # {mk.doc}")
    else:
        out += ["", _recipe_table(name, "background"), "# Behind cut-out layers (fill = \"fractal\")."]
        for key, bk in BACKGROUND_KEYS.items():
            if key in src.background:
                out.append(_line(key, src.background[key], bk.doc))
            else:
                out.append(f"# {key} = {toml_value(bk.neutral)}  # {bk.doc}")
        out += ["", "# Layers, bottom to top."]
        for layer in src.layers:
            out.append(_array_table(name, "layers"))
            out += [_line(k, v) for k, v in layer.items()]
    portrait = {k: dict(v) if isinstance(v, dict) else v for k, v in src.portrait.items()}
    if spec.portrait_params:
        portrait["params"] = {**spec.portrait_params, **(portrait.get("params") or {})}
    out += ["", "# Overrides merged in when height > width (phones)."]
    if portrait:
        out.append(_recipe_table(name, "portrait"))
        for key, val in portrait.items():
            if isinstance(val, dict):
                val = {k: _plain(v) for k, v in val.items()}
            out.append(_line(key, val))
    else:
        out.append(f"# {_recipe_table(name, 'portrait')}\n# view = {{ half_width = 0.8 }}")

    out += [
        "",
        "# Named tweaks render as NAME@VARIANT, e.g.:",
        f"# {_recipe_table(name, 'variants', 'alt')}",
        '# palette = "aurora"',
        "# post = { hue_shift = 30 }",
        "",
    ]
    return "\n".join(out)


def image_meta(recipe: Recipe, width: int, height: int) -> dict[str, str]:
    """Text chunks embedded in rendered images: the look as a self-contained recipe.

    ``fractalgen fork IMAGE NEW`` turns them back into ``custom/NEW.toml``.
    """
    name = "".join(c if c.isalnum() or c in "_-" else "_" for c in recipe.name) or "look"
    return {
        "Software": "fractalgen",
        "fractalgen.name": recipe.name,
        "fractalgen.size": f"{width}x{height}",
        "fractalgen.recipe": recipe_to_toml(recipe, name),
    }


def recipe_from_image(path: str | Path) -> tuple[str, dict[str, Any]]:
    """(name, recipe table) embedded in an image :func:`image_meta` wrote."""
    import tomllib

    from .color import read_meta

    meta = read_meta(path)
    text = meta.get("fractalgen.recipe")
    if not text:
        raise ValueError(f"{path}: no fractalgen recipe embedded (rendered before metadata, or edited elsewhere)")
    tables = tomllib.loads(text).get("recipe") or {}
    if len(tables) != 1:
        raise ValueError(f"{path}: embedded recipe is malformed")
    ((name, table),) = tables.items()
    return name, table


def _plain(v: Any) -> Any:
    from .params import to_complex

    if isinstance(v, str) and v.replace(" ", "").endswith("j"):
        try:
            return to_complex(v)
        except ValueError:
            return v
    return v


def overrides_to_toml(name: str, generator: str, ov: dict[str, Any]) -> str:
    GENERATORS.get(generator)  # validate
    lines = [_recipe_table(name), _line("generator", generator)]
    if "palette" in ov:
        lines.append(_line("palette", ov["palette"]))
    for key in ("view", "params", "post", "tone", "mask", "layers", "background"):
        if ov.get(key):
            lines.append(f"{key} = {toml_value(ov[key])}")
    return "\n".join(lines)
