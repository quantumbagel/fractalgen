"""fractalgen: an extensible math and fractal wallpaper generator.

Third-party extensions can be registered via:

* **pip**: declare a ``fractalgen.plugins`` entry point that invokes ``register()`` hooks
* **TOML**: add files under ``custom/*.toml`` or pass ``--load`` (palettes, recipes, formulas, etc.)
* **Python**: add files under ``custom/*.py`` or pass ``--load`` with ``@generator``, ``@post_op``, etc.

Call :func:`auto_tone` in custom generators so recipe ``tone`` settings apply. Type hints
``Annotated[..., Range(...)]``, ``Literal[...]``, and ``Doc(...)`` supply schema metadata to
the GUI when ``ranges``, ``choices``, or ``param_docs`` are omitted. Reload user-defined paths
at runtime with :func:`fractalgen.loader.reload_user_paths`.

Quick start::

    import fractalgen as fg
    fg.bootstrap()
    rgb = fg.render("julia_dragon", 1920, 1080)
    fg.save_image("dragon.png", rgb)
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

try:
    __version__ = version("fractalgen")
except PackageNotFoundError:
    __version__ = "0.1.0"

from .color import histogram_index, hsv_to_rgb, log_tone, save_image
from .context import RenderContext
from .generators.base import GENERATORS, generator
from .layers import LAYER_TYPES, apply_layers, layer_type
from .loader import bootstrap, load_path, section
from .mask import SHAPES, shape
from .palette import PALETTES, Palette, get_palette, register_palette, sample_palette
from .params import Doc, Range
from .post import POST_OPS, apply_post, effect, post_op
from .recipe import RECIPES, Recipe, register_recipe
from .schema import Key
from .view import View, angle

__all__ = [
    "__version__",
    "GENERATORS",
    "PALETTES",
    "POST_OPS",
    "LAYER_TYPES",
    "SHAPES",
    "Key",
    "RECIPES",
    "Recipe",
    "RenderContext",
    "View",
    "Doc",
    "Range",
    "angle",
    "apply_post",
    "apply_layers",
    "effect",
    "layer_type",
    "shape",
    "auto_tone",
    "bootstrap",
    "generator",
    "get_palette",
    "histogram_index",
    "hsv_to_rgb",
    "load_path",
    "log_tone",
    "post_op",
    "register_palette",
    "register_recipe",
    "render",
    "sample_palette",
    "save_image",
    "section",
    "register_map",
    "register_flow",
    "register_formula",
    "register_vector_field",
    "register_domain",
    "progress",
]


def auto_tone(
    values: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    how: str = "histogram",
    **kw: Any,
) -> np.ndarray:
    """Map raw generator values to [0, 1] for palette sampling, integrating recipe tone adjustments.

    ``how="histogram"`` (default) calls :func:`histogram_index`; ``how="log"`` calls
    :func:`log_tone` on accumulation-style buffers.
    """
    if how == "histogram":
        if mask is None:
            mask = np.ones(values.shape, dtype=bool)
        return histogram_index(values, mask, **kw)
    if how == "log":
        return log_tone(values, **kw)
    raise ValueError(f"auto_tone how must be 'histogram' or 'log', not {how!r}")


def render(recipe: str | Recipe, width: int, height: int, **overrides: Any):
    """Render a recipe by name or instance to an RGB float array.

    The ``overrides`` mapping supports ``params``, ``view``, and ``palette`` matching CLI options.
    """
    bootstrap()
    r = RECIPES.get(recipe) if isinstance(recipe, str) else recipe
    return r.render(width, height, overrides=overrides or None)


def __getattr__(name: str):
    lazy = {
        "register_map": ("fractalgen.generators.maps", "register_map"),
        "register_flow": ("fractalgen.generators.flows", "register_flow"),
        "register_formula": ("fractalgen.generators.escape", "register_formula"),
        "register_vector_field": (
            "fractalgen.generators.fields",
            "register_vector_field",
        ),
        "register_domain": ("fractalgen.generators.domain", "register_domain"),
        "progress": ("fractalgen.progress", None),
    }
    if name in lazy:
        import importlib

        mod, attr = lazy[name]
        m = importlib.import_module(mod)
        return m if attr is None else getattr(m, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
