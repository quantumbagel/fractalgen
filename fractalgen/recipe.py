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
adjusts the generator's own auto-exposure (black / white point, curve; see tone.py);
``mask`` cuts the render out with text, a shape or an image (see mask.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import fieldcache, progress, simcache, tone
from . import mask as masking
from .context import RenderContext
from .generators.base import GENERATORS, GeneratorSpec
from .params import deep_merge
from .post import apply_post
from .registry import Registry
from .view import View

_KEYS = {
    "generator", "palette", "params", "view", "post", "tone", "mask", "portrait", "tags", "description", "extends", "variants",
    "tone_from",
}
_PORTRAIT_KEYS = {"params", "view", "palette", "post", "tone", "mask"}


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
    mask: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    description: str = ""
    source: str = ""
    # View dict whose tone the recipe keeps when its view is edited (merged over
    # the generator default); None = the recipe's own view without overrides.
    tone_view: dict[str, Any] | None = None
    # Recipe whose anchor view this one measures its tone on, so a look saved after
    # panning / zooming another recipe keeps that recipe's exposure ("" = its own view).
    tone_from: str = ""

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
        if portrait.get("mask"):
            portrait["mask"] = masking.check_mask(portrait["mask"])
        return cls(
            name=name,
            generator=str(table["generator"]),
            params=dict(table.get("params") or {}),
            view=dict(table.get("view") or {}),
            portrait=portrait,
            palette=table.get("palette"),
            post=dict(table.get("post") or {}),
            tone=tone.check_tone(table.get("tone")),
            mask=masking.check_mask(table.get("mask")),
            tone_from=str(table.get("tone_from") or ""),
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
        ov_view = ov.get("view") or {}
        params.update(ov.get("params") or {})
        view.update(ov_view)
        palette = ov.get("palette", palette)
        if palette is not None:
            if "palette" not in spec.params:
                raise TypeError(f"Generator {spec.name!r} does not take a palette")
            params["palette"] = palette
        if not spec.uses_view:
            if ov_view:
                raise TypeError(
                    f"Generator {spec.name!r} has no view framing; "
                    f"--rotate/--center/--zoom are not supported for it"
                )
            if view:
                raise TypeError(f"Generator {spec.name!r} does not use a view; remove 'view' from {self.name!r}")
        return spec, spec.bind(params), View.from_dict(view)

    def anchor_view(
        self, width: int, height: int, overrides: dict[str, Any] | None = None, _seen: tuple[str, ...] = ()
    ) -> View:
        """The view whose tone statistics frames of this recipe reuse (see tone.py)."""
        if self.tone_view is not None:
            return View.from_dict({**self.spec.default_view, **self.tone_view})
        src = RECIPES.get(self.tone_from) if self.tone_from in RECIPES else None
        if src is not None and src.generator == self.generator and self.tone_from not in _seen + (self.name,):
            return src.anchor_view(width, height, _seen=_seen + (self.name,))
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

    def with_overrides(self, overrides: dict[str, Any] | None, name: str | None = None) -> "Recipe":
        """This recipe with CLI / API ``overrides`` baked in (in both orientations)."""
        ov = overrides or {}
        if not ov:
            return self
        portrait = {k: dict(v) if isinstance(v, dict) else v for k, v in self.portrait.items()}
        for key in ("params", "view", "post", "tone", "mask"):
            if ov.get(key) and key in portrait:
                portrait[key] = {**portrait[key], **ov[key]}
        if "palette" in ov and "palette" in portrait:
            portrait["palette"] = ov["palette"]
        drop = lambda d: {k: v for k, v in d.items() if v is not None}  # noqa: E731
        return Recipe(
            name=name or self.name,
            generator=self.generator,
            params={**self.params, **(ov.get("params") or {})},
            view={**self.view, **(ov.get("view") or {})},
            portrait=portrait,
            palette=ov.get("palette", self.palette),
            post=drop({**self.post, **(ov.get("post") or {})}),
            tone=drop({**self.tone, **(ov.get("tone") or {})}),
            mask=drop({**self.mask, **(ov.get("mask") or {})}),
            tags=self.tags,
            description=self.description,
            source=self.source,
            tone_view=self.tone_view,
            # A view override keeps this recipe's exposure (anchor_view ignores it).
            tone_from=self.tone_from or (self.name if ov.get("view") else ""),
        )

    def resolve_mask(self, width: int, height: int, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        out = dict(self.mask)
        if height > width:
            out.update(self.portrait.get("mask") or {})
        out.update((overrides or {}).get("mask") or {})
        return masking.check_mask(out)

    def render(
        self,
        width: int,
        height: int,
        *,
        overrides: dict[str, Any] | None = None,
        n_strips: int = 1,
    ) -> np.ndarray:
        ov = overrides or {}
        render_ov = {k: v for k, v in ov.items() if k not in ("post", "mask")}
        spec, params, view = self.resolve(width, height, render_ov)
        ctx = RenderContext(width=width, height=height, view=view, name=self.name, n_strips=n_strips)
        adjust = self.resolve_tone(width, height, overrides)
        anchor = self.anchor_view(width, height, overrides) if spec.uses_view else view
        post = self.resolve_post(width, height, overrides)
        mask = self.resolve_mask(width, height, overrides)

        def _render_rgb() -> np.ndarray:
            if anchor == view or adjust.get("per_frame"):
                with tone.adjusting(adjust):
                    return np.asarray(spec.fn(ctx, **params))
            rw, rh = _tone_frame(width, height)
            ref = RenderContext(width=rw, height=rh, view=anchor, name=self.name)
            anchor_params = {k: v for k, v in params.items() if k != "palette"}
            key = (spec.name, simcache.freeze(anchor_params), simcache.freeze(adjust), anchor, rw, rh)
            with progress.span(0.0, 0.1, "measuring exposure"):
                stats = tone.anchored(key, lambda: spec.fn(ref, **params))
            ctx.tone_view = anchor
            with progress.span(0.1, 0.98, "rendering"), tone.replaying(stats), tone.adjusting(adjust):
                return np.asarray(spec.fn(ctx, **params))

        cache_key = fieldcache.ctx_key(
            ctx,
            "recipe_rgb",
            generator=spec.name,
            params=simcache.freeze(params),
            adjust=simcache.freeze(adjust),
            anchor=fieldcache.view_key(anchor) if spec.uses_view else None,
            per_frame=bool(adjust.get("per_frame")),
        )
        end = 0.9 if masking.active(mask) else 0.98
        with progress.span(0.0, end, "rendering"):
            rgb = fieldcache.cached(cache_key, _render_rgb)
        if rgb.shape != (height, width, 3):
            raise ValueError(f"{self.name}: generator returned shape {rgb.shape}, expected {(height, width, 3)}")
        if masking.active(mask):
            with progress.span(end, 1.0, "mask"):
                return apply_post(masking.apply_mask(rgb, mask, ctx.res_scale), post, ctx)
        with progress.span(0.98, 1.0, "finishing"):
            return apply_post(np.array(rgb, copy=True), post, ctx)


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
        if VARIANT_SEP in name:
            base, vname = name.split(VARIANT_SEP, 1)
            if base not in _RAW:
                raise KeyError(f"Recipe {seen[-1]!r} extends unknown recipe {name!r}")
            merged = _flatten(base, seen)
            variants = merged.pop("variants", None) or {}
            if vname not in variants:
                raise KeyError(f"Recipe {seen[-1]!r} extends unknown recipe {name!r}")
            return deep_merge({k: v for k, v in merged.items() if k != "variants"}, variants[vname])
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
    from .loader import LoadError

    for name in list(_RAW.keys()):
        _, source = _RAW[name]
        try:
            table = _flatten(name)
        except (KeyError, ValueError) as exc:
            del _RAW[name]
            detail = exc.args[0] if exc.args else exc
            raise LoadError(
                f"{source}: [recipe.{name}]: {detail}",
                path=source,
                section="recipe",
                entry=name,
            ) from exc
        variants = table.pop("variants", None) or {}
        # Recorded with its file so dropping that file (hot reload, failed load) drops the recipe too.
        RECIPES.register(name, Recipe.from_table(name, table, source), replace=True, source=source)
        base = {k: v for k, v in table.items() if k not in ("tags", "description")}
        for vname, vtable in variants.items():
            full = f"{name}{VARIANT_SEP}{vname}"
            RECIPES.register(full, Recipe.from_table(full, deep_merge(base, vtable), source), replace=True, source=source)


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
