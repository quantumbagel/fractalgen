"""Recipes: named, data-driven wallpapers.

A recipe picks a generator and fixes its parameters, palette, and framing.
Recipes are TOML tables (see ``data/recipes/*.toml``)::

    [recipe.julia_dragon]
    generator = "escape_time"
    palette = "ember"
    tags = ["keep", "vip"]
    view = { center = [0.0, 0.05], half_width = 1.45 }
    params = { formula = "julia", c = [-0.8, 0.156], max_iter = 480 }
    # Merged over the above when height > width:
    portrait.view = { center = [-0.18, 0.24], half_width = 0.60, fit = "fill" }

``extends = "other_recipe"`` inherits everything from another recipe and
deep-merges this table on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .context import RenderContext
from .generators.base import GENERATORS, GeneratorSpec
from .params import deep_merge
from .registry import Registry
from .view import View

_KEYS = {"generator", "palette", "params", "view", "portrait", "tags", "description", "extends"}


@dataclass
class Recipe:
    name: str
    generator: str
    params: dict[str, Any] = field(default_factory=dict)
    view: dict[str, Any] = field(default_factory=dict)
    portrait: dict[str, Any] = field(default_factory=dict)
    palette: Any = None
    tags: tuple[str, ...] = ()
    description: str = ""
    source: str = ""

    @classmethod
    def from_table(cls, name: str, table: dict[str, Any], source: str = "") -> "Recipe":
        unknown = set(table) - _KEYS
        if unknown:
            raise ValueError(f"Recipe {name!r}: unknown keys {sorted(unknown)} (allowed: {sorted(_KEYS)})")
        if "generator" not in table:
            raise ValueError(f"Recipe {name!r} needs a 'generator'")
        portrait = dict(table.get("portrait") or {})
        bad = set(portrait) - {"params", "view", "palette"}
        if bad:
            raise ValueError(f"Recipe {name!r}: portrait may only set params/view/palette, got {sorted(bad)}")
        return cls(
            name=name,
            generator=str(table["generator"]),
            params=dict(table.get("params") or {}),
            view=dict(table.get("view") or {}),
            portrait=portrait,
            palette=table.get("palette"),
            tags=tuple(table.get("tags") or ()),
            description=str(table.get("description") or ""),
            source=source,
        )

    @property
    def spec(self) -> GeneratorSpec:
        return GENERATORS.get(self.generator)

    def resolve(
        self, width: int, height: int, overrides: dict[str, Any] | None = None
    ) -> tuple[GeneratorSpec, dict[str, Any], View]:
        """Return (generator, bound params, view) for a frame size."""
        spec = self.spec
        portrait = height > width
        params: dict[str, Any] = {}
        if portrait:
            params.update(spec.portrait_params)
        params.update(self.params)
        palette = self.palette
        view = dict(self.view)
        if portrait:
            params.update(self.portrait.get("params") or {})
            palette = self.portrait.get("palette", palette)
            view = {**view, "fit": "cover", **(self.portrait.get("view") or {})}
        ov = overrides or {}
        params.update(ov.get("params") or {})
        view.update(ov.get("view") or {})
        palette = ov.get("palette", palette)
        if palette is not None:
            if "palette" not in spec.params:
                raise TypeError(f"Generator {spec.name!r} does not take a palette")
            params["palette"] = palette
        if view and not spec.uses_view:
            raise TypeError(f"Generator {spec.name!r} does not use a view; remove 'view' from {self.name!r}")
        return spec, spec.bind(params), View.from_dict(view)

    def render(
        self,
        width: int,
        height: int,
        *,
        overrides: dict[str, Any] | None = None,
        n_strips: int = 1,
    ) -> np.ndarray:
        spec, params, view = self.resolve(width, height, overrides)
        ctx = RenderContext(width=width, height=height, view=view, name=self.name, n_strips=n_strips)
        rgb = np.asarray(spec.fn(ctx, **params))
        if rgb.shape != (height, width, 3):
            raise ValueError(f"{self.name}: generator returned shape {rgb.shape}, expected {(height, width, 3)}")
        return rgb


RECIPES: Registry[Recipe] = Registry("recipe")
_RAW: dict[str, tuple[dict[str, Any], str]] = {}


def add_recipe_table(name: str, table: dict[str, Any], source: str = "", *, replace: bool = False) -> None:
    """Queue a raw recipe table; ``extends`` chains resolve in :func:`finalize_recipes`."""
    if name in _RAW and not replace:
        raise KeyError(f"recipe {name!r} is already defined in {_RAW[name][1]}")
    _RAW[name] = (table, source)


def _flatten(name: str, seen: tuple[str, ...] = ()) -> dict[str, Any]:
    if name in seen:
        raise ValueError(f"Recipe 'extends' cycle: {' -> '.join(seen + (name,))}")
    if name not in _RAW:
        raise KeyError(f"Recipe {seen[-1]!r} extends unknown recipe {name!r}")
    table, _ = _RAW[name]
    parent = table.get("extends")
    if not parent:
        return dict(table)
    base = _flatten(str(parent), seen + (name,))
    base.pop("tags", None)  # tags and descriptions describe one look, not its children
    base.pop("description", None)
    merged = deep_merge(base, {k: v for k, v in table.items() if k != "extends"})
    return merged


def finalize_recipes() -> None:
    for name, (_, source) in _RAW.items():
        RECIPES.register(name, Recipe.from_table(name, _flatten(name), source), replace=True)


def register_recipe(recipe: Recipe, *, replace: bool = False) -> Recipe:
    return RECIPES.register(recipe.name, recipe, replace=replace)


def select(
    names: list[str] | None = None,
    tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    generators: list[str] | None = None,
) -> list[str]:
    """Filter recipe names in registry order. Explicit ``names`` win over filters."""
    if names:
        missing = [n for n in names if n not in RECIPES]
        if missing:
            raise KeyError(f"Unknown recipe(s): {missing}")
        return list(names)
    out = []
    for name, r in RECIPES.items():
        if tags and not set(tags) & set(r.tags):
            continue
        if exclude_tags and set(exclude_tags) & set(r.tags):
            continue
        if generators and r.generator not in generators:
            continue
        out.append(name)
    return out


def source_path(recipe: Recipe) -> Path | None:
    return Path(recipe.source) if recipe.source else None
