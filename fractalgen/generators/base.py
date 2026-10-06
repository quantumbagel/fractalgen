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
    # UI hints (optional): (min, max[, step]) for sliders, and allowed values.
    range: tuple | None = None
    choices: Any = None  # list, or a zero-arg callable returning one (e.g. a registry's names)

    def choice_list(self) -> list | None:
        c = self.choices() if callable(self.choices) else self.choices
        return list(c) if c is not None else None

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
    # View used when a recipe doesn't set one (only for uses_view generators).
    default_view: dict[str, Any] = field(default_factory=dict)
    # Param defaults applied automatically for portrait frames (recipe can override).
    portrait_params: dict[str, Any] = field(default_factory=dict)

    def bind(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate names and coerce values against the defaults."""
        unknown = [k for k in params if k not in self.params]
        if unknown:
            from ..post import POST_OPS

            hints = []
            for k in unknown:
                close = difflib.get_close_matches(k, self.params, n=1)
                if k in POST_OPS:
                    hints.append(f"{k!r} (that's a post op: use `post = {{ {k} = ... }}`, --post {k}=..., or post.{k})")
                else:
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
    default_view: dict[str, Any] | None = None,
    portrait_params: dict[str, Any] | None = None,
    param_docs: dict[str, str] | None = None,
    ranges: dict[str, tuple] | None = None,
    choices: dict[str, Any] | None = None,
    replace: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register ``fn`` as generator ``name``. Returns ``fn`` unchanged.

    ``ranges`` / ``choices`` are optional UI hints for the GUI and docs:
    ``ranges={"max_iter": (10, 5000)}``, ``choices={"formula": FORMULAS.names}``.
    """

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        params: dict[str, Param] = {}
        docs = param_docs or {}
        rng = ranges or {}
        chs = choices or {}
        for hint in (*docs, *rng, *chs):
            if hint not in sig.parameters:
                raise TypeError(f"{fn.__name__}: hint for unknown parameter {hint!r}")
        for i, (pname, p) in enumerate(sig.parameters.items()):
            if i == 0:
                continue  # ctx
            if p.kind is not inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(f"{fn.__name__}: parameter {pname!r} must be keyword-only (after `*`)")
            params[pname] = Param(pname, p.default, docs.get(pname, ""), rng.get(pname), chs.get(pname))
        doc = inspect.getdoc(fn) or ""
        spec = GeneratorSpec(
            name=name,
            fn=fn,
            params=params,
            category=category,
            description=doc.splitlines()[0] if doc else "",
            uses_view=uses_view,
            default_view=dict(default_view or {}),
            portrait_params=dict(portrait_params or {}),
        )
        GENERATORS.register(name, spec, replace=replace)
        return fn

    return wrap
