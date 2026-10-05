"""The generator contract.

A generator is a function ``fn(ctx, *, **params) -> rgb`` that returns a float
array of shape (ctx.height, ctx.width, 3) in [0, 1]. Its keyword-only
arguments *are* its parameter schema: names, defaults, and (through the
defaults) the types recipe values are coerced to.

    @generator("my_noise", category="noise", uses_view=True)
    def my_noise(ctx, *, palette="cosmic", octaves=4, scale=1.0):
        '''One-line description shown by `fractalgen generators`.'''
        re, im = ctx.plane()
        ...
        return rgb
"""

from __future__ import annotations

import difflib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from ..params import coerce
from ..registry import Registry

_EMPTY = inspect.Parameter.empty


@dataclass(frozen=True)
class Param:
    name: str
    default: Any
    doc: str = ""

    @property
    def required(self) -> bool:
        return self.default is _EMPTY


@dataclass(frozen=True)
class GeneratorSpec:
    name: str
    fn: Callable[..., Any]
    params: dict[str, Param]
    category: str = "misc"
    description: str = ""
    # Whether the generator reads ctx.view / ctx.plane(); drives view validation.
    uses_view: bool = False
    # Param defaults applied automatically for portrait frames (recipe can override).
    portrait_params: dict[str, Any] = field(default_factory=dict)

    def bind(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate names and coerce values against the defaults."""
        unknown = [k for k in params if k not in self.params]
        if unknown:
            hints = []
            for k in unknown:
                close = difflib.get_close_matches(k, self.params, n=1)
                hints.append(f"{k!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
            raise TypeError(f"Generator {self.name!r} has no parameter(s) {', '.join(hints)}")
        out: dict[str, Any] = {}
        for name, p in self.params.items():
            if name in params:
                out[name] = coerce(params[name], None if p.required else p.default)
            elif p.required:
                raise TypeError(f"Generator {self.name!r} requires parameter {name!r}")
            else:
                out[name] = p.default
        return out

    def __call__(self, ctx, **params: Any):
        return self.fn(ctx, **self.bind(params))


GENERATORS: Registry[GeneratorSpec] = Registry("generator")


def generator(
    name: str,
    *,
    category: str = "misc",
    uses_view: bool = False,
    portrait_params: dict[str, Any] | None = None,
    param_docs: dict[str, str] | None = None,
    replace: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register ``fn`` as generator ``name``. Returns ``fn`` unchanged."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        params: dict[str, Param] = {}
        docs = param_docs or {}
        for i, (pname, p) in enumerate(sig.parameters.items()):
            if i == 0:
                continue  # ctx
            if p.kind is not inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(f"{fn.__name__}: parameter {pname!r} must be keyword-only (after `*`)")
            params[pname] = Param(pname, p.default, docs.get(pname, ""))
        doc = inspect.getdoc(fn) or ""
        spec = GeneratorSpec(
            name=name,
            fn=fn,
            params=params,
            category=category,
            description=doc.splitlines()[0] if doc else "",
            uses_view=uses_view,
            portrait_params=dict(portrait_params or {}),
        )
        GENERATORS.register(name, spec, replace=replace)
        return fn

    return wrap
