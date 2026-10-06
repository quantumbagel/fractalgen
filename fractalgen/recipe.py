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
deep-merges this table on top. ``variants`` define small named tweaks of one
recipe in place; each becomes its own recipe ``NAME@VARIANT``::

    [recipe.julia_dragon.variants.ice]
    palette = "abyss"
    post = { saturation = 0.8 }

``post`` applies generator-independent finishing (see post.py); ``tone``
adjusts the generator's own auto-exposure (black / white point, curve; see tone.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import simcache, tone
from .context import RenderContext
from .generators.base import GENERATORS, GeneratorSpec
from .params import deep_merge
from .post import apply_post
from .registry import Registry
from .view import View

_KEYS = {"generator", "palette", "params", "view", "post", "tone", "portrait", "tags", "description", "extends", "variants"}
_PORTRAIT_KEYS = {"params", "view", "palette", "post", "tone"}


@dataclass
class Recipe:
    name: str
    generator: str
    params: dict[str, Any] = field(default_factory=dict)
    view: dict[str, Any] = field(default_factory=dict)
    portrait: dict[str, Any] = field(default_factory=dict)
    palette: Any = None
    post: dict[str, Any] = field(default_factory=dict)
    tone: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    description: str = ""
    source: str = ""
    # View dict whose tone the recipe keeps when its view is edited (merged over
    # the generator default); None = the recipe's own view without overrides.
    tone_view: dict[str, Any] | None = None

    @classmethod
    def from_table(cls, name: str, table: dict[str, Any], source: str = "") -> "Recipe":
        unknown = set(table) - _KEYS
        if unknown:
            raise ValueError(f"Recipe {name!r}: unknown keys {sorted(unknown)} (allowed: {sorted(_KEYS)})")
        if "generator" not in table:
            raise ValueError(f"Recipe {name!r} needs a 'generator'")
        portrait = dict(table.get("portrait") or {})
        bad = set(portrait) - _PORTRAIT_KEYS
        if bad:
            raise ValueError(f"Recipe {name!r}: portrait may only set {sorted(_PORTRAIT_KEYS)}, got {sorted(bad)}")
        return cls(
            name=name,
            generator=str(table["generator"]),
            params=dict(table.get("params") or {}),
            view=dict(table.get("view") or {}),
            portrait=portrait,
            palette=table.get("palette"),
            post=dict(table.get("post") or {}),
            tone=tone.check_tone(table.get("tone")),
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
        view = {**spec.default_view, **self.view}
        if portrait:
            params.update(self.portrait.get("params") or {})
            palette = self.portrait.get("palette", palette)
            if spec.uses_view:  # portrait default: same zoom as the 16:9 window, cropped
                view = {**view, "fit": "cover", **(self.portrait.get("view") or {})}
            elif self.portrait.get("view"):
                view = {**view, **self.portrait["view"]}
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

    def anchor_view(self, width: int, height: int, overrides: dict[str, Any] | None = None) -> View:
        """The view whose tone statistics frames of this recipe reuse (see tone.py)."""
        if self.tone_view is not None:
            return View.from_dict({**self.spec.default_view, **self.tone_view})
        ov = {k: v for k, v in (overrides or {}).items() if k != "view"}
        return self.resolve(width, height, ov)[2]

    def resolve_tone(self, width: int, height: int, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        out = dict(self.tone)
        if height > width:
            out.update(self.portrait.get("tone") or {})
        out.update((overrides or {}).get("tone") or {})
        return tone.check_tone(out)

    def resolve_post(self, width: int, height: int, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        post = dict(self.post)
        if height > width:
            post.update(self.portrait.get("post") or {})
        post.update((overrides or {}).get("post") or {})
        return post

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
        adjust = self.resolve_tone(width, height, overrides)
        anchor = self.anchor_view(width, height, overrides) if spec.uses_view else view
        if anchor == view or adjust.get("per_frame"):
            with tone.adjusting(adjust):
                rgb = np.asarray(spec.fn(ctx, **params))
        else:
            # Off the authored view: expose with the authored view's statistics
            # (measured on a small frame of the same aspect) so pans and zooms
            # don't re-expose the image.
            rw, rh = _tone_frame(width, height)
            ref = RenderContext(width=rw, height=rh, view=anchor, name=self.name)
            key = (spec.name, simcache.freeze(params), anchor, rw, rh)
            stats = tone.anchored(key, lambda: spec.fn(ref, **params))
            ctx.tone_view = anchor
            with tone.replaying(stats), tone.adjusting(adjust):
                rgb = np.asarray(spec.fn(ctx, **params))
        if rgb.shape != (height, width, 3):
            raise ValueError(f"{self.name}: generator returned shape {rgb.shape}, expected {(height, width, 3)}")
        return apply_post(rgb, self.resolve_post(width, height, overrides), ctx)


_TONE_PIXELS = 640 * 360


def _tone_frame(width: int, height: int) -> tuple[int, int]:
    """Frame tone statistics are measured on: this aspect, a fixed ~0.23 MP.

    Fixed, so the GUI's draft, its preview and a full export share one measurement.
    """
    s = (_TONE_PIXELS / float(width * height)) ** 0.5
    return max(16, round(width * s)), max(16, round(height * s))


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
    for key in ("tags", "description", "variants"):  # these describe one look, not its children
        base.pop(key, None)
    merged = deep_merge(base, {k: v for k, v in table.items() if k != "extends"})
    return merged


VARIANT_SEP = "@"


def finalize_recipes() -> None:
    for name, (_, source) in _RAW.items():
        table = _flatten(name)
        variants = table.pop("variants", None) or {}
        RECIPES.register(name, Recipe.from_table(name, table, source), replace=True)
        base = {k: v for k, v in table.items() if k not in ("tags", "description")}
        for vname, vtable in variants.items():
            full = f"{name}{VARIANT_SEP}{vname}"
            RECIPES.register(full, Recipe.from_table(full, deep_merge(base, vtable), source), replace=True)


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
