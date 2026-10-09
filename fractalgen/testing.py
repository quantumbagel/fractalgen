"""Helpers for authors and CI: smoke-render and validate extensions."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from . import bootstrap
from .context import REFERENCE_PIXELS
from .generators.base import GENERATORS, GeneratorSpec, _EMPTY
from .recipe import RECIPES


def _with_small_reference(fn):
    saved = REFERENCE_PIXELS
    try:
        import fractalgen.context as ctx

        ctx.REFERENCE_PIXELS = 480 * 270
        return fn()
    finally:
        import fractalgen.context as ctx

        ctx.REFERENCE_PIXELS = saved


def _assert_rgb(rgb: np.ndarray, *, label: str) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise AssertionError(f"{label}: expected (H, W, 3) RGB, got shape {rgb.shape}")
    if not np.isfinite(rgb).all():
        raise AssertionError(f"{label}: non-finite values in output")
    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise AssertionError(f"{label}: RGB outside [0, 1]: min={rgb.min()}, max={rgb.max()}")
    if float(rgb.std()) <= 0.0:
        raise AssertionError(f"{label}: image is blank (zero variance)")
    return rgb


def smoke_generator(
    name: str,
    *,
    width: int = 48,
    height: int = 27,
    params: dict[str, Any] | None = None,
    palette: str | None = None,
    view: dict[str, Any] | None = None,
) -> np.ndarray:
    """Render a generator at thumbnail size; raises AssertionError on bad output."""
    bootstrap()

    def run():
        spec = GENERATORS.get(name)
        from .context import RenderContext
        from .view import View

        v = View(**(view or {})) if view else None
        if v is None and spec.uses_view:
            v = View(**spec.default_view) if spec.default_view else View()
        ctx = RenderContext(width, height, view=v)
        kw = spec.bind(params or {})
        if palette is not None:
            kw["palette"] = palette
        rgb = spec(ctx, **kw)
        return _assert_rgb(np.asarray(rgb), label=f"generator {name!r}")

    return _with_small_reference(run)


def smoke_recipe(
    name: str,
    *,
    width: int = 48,
    height: int = 27,
    **overrides: Any,
) -> np.ndarray:
    """Render a recipe at thumbnail size; raises AssertionError on bad output."""
    bootstrap()

    def run():
        rgb = RECIPES.get(name).render(width, height, overrides=overrides or None)
        return _assert_rgb(np.asarray(rgb), label=f"recipe {name!r}")

    return _with_small_reference(run)


def check_generator(name: str) -> list[str]:
    """Validate a generator spec; return human-readable warnings (empty if clean)."""
    bootstrap()
    warnings: list[str] = []
    try:
        spec = GENERATORS.get(name)
    except KeyError as exc:
        return [str(exc)]
    if not isinstance(spec, GeneratorSpec):
        return [f"{name!r} is not a GeneratorSpec"]
    if not spec.description:
        warnings.append("missing docstring (first line becomes the short description)")
    for pname, p in spec.params.items():
        if not p.doc:
            warnings.append(f"parameter {pname!r} has no documentation")
        if p.range is not None:
            r = p.range
            if len(r) < 2:
                warnings.append(f"parameter {pname!r}: range needs at least (lo, hi)")
            elif r[0] > r[1]:
                warnings.append(f"parameter {pname!r}: range lo > hi ({r})")
        if p.required:
            warnings.append(f"parameter {pname!r} is required (no default)")
        elif p.default is not _EMPTY:
            try:
                from .params import coerce

                coerce(p.default, p.default)
            except Exception as exc:
                warnings.append(f"parameter {pname!r}: default not coercible ({exc})")
    if not inspect.isfunction(spec.fn):
        warnings.append("implementation is not a plain function")
    return warnings
