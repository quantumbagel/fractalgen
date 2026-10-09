"""fractalgen: an extensible math / fractal wallpaper generator.

Third-party extensions:

* **pip** — declare a ``fractalgen.plugins`` entry point that calls ``register()`` hooks
* **TOML** — ``custom/*.toml`` or ``--load`` (palettes, recipes, formulas, …)
* **Python** — ``custom/*.py`` or ``--load`` with ``@generator``, ``@post_op``, etc.

Use :func:`auto_tone` in generators so recipe ``tone`` adjustments apply. Type hints
``Annotated[..., Range(...)]``, ``Literal[...]``, and ``Doc(...)`` feed the GUI when
``ranges`` / ``choices`` / ``param_docs`` are omitted. Hot-reload user files with
:func:`fractalgen.loader.reload_user_paths`.

Quick use::

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

from .color import hsv_to_rgb, histogram_index, log_tone, save_image
from .context import RenderContext
from .generators.base import GENERATORS, generator
from .loader import bootstrap, load_path, section
from .params import Doc, Range
from .post import POST_OPS, apply_post, post_op
from .palette import PALETTES, Palette, get_palette, register_palette, sample_palette
from .recipe import RECIPES, Recipe, register_recipe
from .view import View, angle

__all__ = [
    "__version__",
    "GENERATORS",
    "PALETTES",
    "POST_OPS",
    "RECIPES",
    "Recipe",
    "RenderContext",
    "View",
    "Doc",
    "Range",
    "angle",
    "apply_post",
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
    """Map raw generator values to [0, 1] for palette sampling (recipe ``tone`` hooks in).

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
    """Render a recipe (name or object) to float RGB. ``overrides`` may hold
    ``params``, ``view``, and ``palette`` like the CLI flags."""
    bootstrap()
    r = RECIPES.get(recipe) if isinstance(recipe, str) else recipe
    return r.render(width, height, overrides=overrides or None)


def __getattr__(name: str):
    lazy = {
        "register_map": ("fractalgen.generators.maps", "register_map"),
        "register_flow": ("fractalgen.generators.flows", "register_flow"),
        "register_formula": ("fractalgen.generators.escape", "register_formula"),
        "register_vector_field": ("fractalgen.generators.fields", "register_vector_field"),
        "register_domain": ("fractalgen.generators.domain", "register_domain"),
        "progress": ("fractalgen.progress", None),
    }
    if name in lazy:
        import importlib

        mod, attr = lazy[name]
        m = importlib.import_module(mod)
        return m if attr is None else getattr(m, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
