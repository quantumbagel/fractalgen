"""Write recipes back out as TOML (``fork``, ``try --emit``, the GUI's save)."""

from __future__ import annotations

import math
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
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _key(k: str) -> str:
    return k if k.replace("_", "").replace("-", "").isalnum() else f'"{k}"'


def _line(key: str, value: Any, doc: str = "") -> str:
    comment = f"  # {doc}" if doc else ""
    if value is None:
        return f"# {key} = (unset){comment}"
    return f"{_key(key)} = {toml_value(value)}{comment}"


def recipe_to_toml(src: Recipe, name: str, *, width: int = 3840, height: int = 2160) -> str:
    """A self-contained recipe with every generator parameter written out."""
    from .post import POST_OPS
    from .tone import TONE_KEYS

    spec, params, view = src.resolve(width, height)
    out = [
        f"# Forked from `{src.name}`" + (f" ({src.source})" if src.source else ""),
        "# Every parameter is listed with its current value. Edit anything; delete a line",
        "# to fall back to the generator default. `fractalgen generators "
        f"{spec.name}` documents them.",
        "",
        f"[recipe.{name}]",
        _line("generator", spec.name),
    ]
    if src.description:
        out.append(_line("description", src.description))
    if "palette" in params:
        out.append(_line("palette", params["palette"], "any palette name (+ _r to reverse) or list of hex stops"))
    out.append("# tags = [\"keep\"]  # opt into preset batches (keep = 8K, vip = 16K)")

    if spec.uses_view:
        out += [
            "",
            f"[recipe.{name}.view]",
            _line("center", view.center, "complex-plane center [re, im]"),
            _line("half_width", view.half_width, "horizontal half-extent; smaller = deeper zoom"),
            _line("rotation", view.rotation, "degrees counter-clockwise"),
            _line("fit", view.fit, "fill | cover | contain"),
        ]

    out += ["", f"[recipe.{name}.params]"]
    for pname, p in spec.params.items():
        if pname == "palette":
            continue
        out.append(_line(pname, params.get(pname), p.doc))

    out += ["", f"[recipe.{name}.post]", "# Finishing ops for any generator; uncomment to use."]
    for op, (_, doc) in POST_OPS.items():
        if op in src.post:
            out.append(_line(op, src.post[op], doc))
        else:
            out.append(f"# {op} = ...  # {doc}")

    out += ["", f"[recipe.{name}.tone]", "# Auto-exposure adjustments; omitted keys keep the look's default."]
    for key, tk in TONE_KEYS.items():
        if key in src.tone:
            out.append(_line(key, src.tone[key], tk.doc))
        else:
            out.append(f"# {key} = {toml_value(tk.neutral)}  # {tk.doc}")

    portrait = {k: dict(v) if isinstance(v, dict) else v for k, v in src.portrait.items()}
    if spec.portrait_params:
        portrait["params"] = {**spec.portrait_params, **(portrait.get("params") or {})}
    out += ["", "# Overrides merged in when height > width (phones)."]
    if portrait:
        out.append(f"[recipe.{name}.portrait]")
        for key, val in portrait.items():
            if isinstance(val, dict):
                val = {k: _plain(v) for k, v in val.items()}
            out.append(_line(key, val))
    else:
        out.append(f"# [recipe.{name}.portrait]\n# view = {{ half_width = 0.8 }}")

    out += [
        "",
        "# Named tweaks render as NAME@VARIANT, e.g.:",
        f"# [recipe.{name}.variants.alt]",
        '# palette = "aurora"',
        "# post = { hue_shift = 30 }",
        "",
    ]
    return "\n".join(out)


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
    lines = [f"[recipe.{name}]", _line("generator", generator)]
    if "palette" in ov:
        lines.append(_line("palette", ov["palette"]))
    for key in ("view", "params", "post", "tone"):
        if ov.get(key):
            lines.append(f"{key} = {toml_value(ov[key])}")
    return "\n".join(lines)
