"""fractalgen: an extensible math / fractal wallpaper generator.

Extension points (all importable from here):

* ``generator``        -- decorator registering a new image algorithm
* ``register_palette`` -- add a color gradient
* ``Recipe`` / ``register_recipe`` -- a named, parameterized look
* ``register_map`` / ``register_flow`` / ``register_formula`` -- formulas for
  the map-attractor, ODE-attractor, and escape-time generators
* ``loader.section``   -- teach the TOML loader a new top-level table

Quick use::

    import fractalgen as fg
    fg.bootstrap()
    rgb = fg.render("julia_dragon", 1920, 1080)
    fg.save_image("dragon.png", rgb)
"""

from __future__ import annotations

from typing import Any

from .color import hsv_to_rgb, histogram_index, log_tone, save_image
from .context import RenderContext
from .generators.base import GENERATORS, generator
from .loader import bootstrap, load_path
from .palette import PALETTES, Palette, get_palette, register_palette, sample_palette
from .recipe import RECIPES, Recipe, register_recipe
from .view import View, angle


def render(recipe: str | Recipe, width: int, height: int, **overrides: Any):
    """Render a recipe (name or object) to float RGB. ``overrides`` may hold
    ``params``, ``view``, and ``palette`` like the CLI flags."""
    bootstrap()
    r = RECIPES.get(recipe) if isinstance(recipe, str) else recipe
    return r.render(width, height, overrides=overrides or None)


def __getattr__(name: str):
    # Formula registries live with their generators; import lazily to avoid cycles.
    lazy = {
        "register_map": ("fractalgen.generators.maps", "register_map"),
        "register_flow": ("fractalgen.generators.flows", "register_flow"),
        "register_formula": ("fractalgen.generators.escape", "register_formula"),
    }
    if name in lazy:
        import importlib

        mod, attr = lazy[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(name)
