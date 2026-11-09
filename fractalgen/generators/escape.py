"""Escape-time fractals (Mandelbrot family, Julia sets) and orbit traps.

Formulas are pluggable. The classic ones run in a hand-tuned numba kernel;
anything else registers a numba-compiled step ``z_next = f(z, c, zp, k)``
(``zp`` = previous z, ``k`` = the ``k`` parameter), either from Python::

    @numba.njit(fastmath=True)
    def cubic(z, c, zp, k):
        return z * z * z + c
    fractalgen.register_formula("cubic", cubic, degree=3)

or from TOML, where ``z`` is an expression compiled to numba::

    [formula.sine_julia]
    z = "c * sin(z)"
    julia = true        # pixel is z0, `c` param is the constant
    degree = 1.0        # polynomial degree for smooth coloring (<= 1: plain counts)
    bailout = 50.0
"""

from __future__ import annotations

import ast
import cmath
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..color import histogram_index
from ..fieldcache import cached as field_cached
from ..fieldcache import ctx_key as field_ctx_key
from ..loader import section
from ..palette import sample_palette
from ..registry import Registry
from .base import generator


@dataclass(frozen=True)
class Formula:
    name: str
    kind_id: int | None = None  # built-in kernel branch
    step: Callable[..., Any] | None = None  # numba step for the generic kernel
    julia: bool = False  # pixel seeds z0 (True) or c (False)
    degree: float = 2.0
    bailout: float = 2.0
    description: str = ""


FORMULAS: Registry[Formula] = Registry("escape-time formula")

for _name, _kid, _julia, _deg, _desc in (
    ("mandelbrot", 0, False, 2, "z^2 + c"),
    ("julia", 1, True, 2, "z^2 + c, pixel is z0"),
    ("burning_ship", 2, False, 2, "(|Re z| + i|Im z|)^2 + c"),
    ("buffalo", 3, False, 2, "|z|-folded z^2 - z + c"),
    ("tricorn", 4, False, 2, "conj(z)^2 + c"),
    ("phoenix", 5, True, 2, "z^2 + Re(c) + k * z_prev, pixel is z0"),
    ("celtic", 6, False, 2, "|Re(z^2)| + i Im(z^2) + c"),
    ("multibrot3", 7, False, 3, "z^3 + c"),
    ("multibrot4", 8, False, 4, "z^4 + c"),
):
    FORMULAS.register(_name, Formula(_name, kind_id=_kid, julia=_julia, degree=_deg, description=_desc))


def register_formula(
    name: str,
    step: Callable[..., Any],
    *,
    julia: bool = False,
    degree: float = 2.0,
    bailout: float = 2.0,
    description: str = "",
    replace: bool = False,
) -> Formula:
    """Register a numba-jitted ``step(z, c, zp, k) -> complex``."""
    if float(degree) > 1.0 and float(bailout) <= 1.0:
        raise ValueError(
            f"bailout must be > 1 when degree > 1 (smooth coloring needs log(bailout) > 0), "
            f"got bailout={bailout!r}, degree={degree!r}"
        )
    f = Formula(name, step=step, julia=julia, degree=degree, bailout=bailout, description=description)
    return FORMULAS.register(name, f, replace=replace)


# Expression formulas compile complex-scalar arithmetic directly into Numba kernels.
_CMATH = {n: f"cmath.{n}" for n in "sin cos tan sinh cosh tanh asin acos atan exp log sqrt".split()}
_ALLOWED_NAMES = {"z", "c", "zp", "k", "abs", "conj", "re", "im", "pi", "i", "j", *(_CMATH)}


class _Rewrite(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        fn = node.func.id if isinstance(node.func, ast.Name) else None
        if fn in ("conj", "re", "im") and len(node.args) == 1:
            attr = {"conj": "conjugate", "re": "real", "im": "imag"}[fn]
            target = ast.Attribute(value=node.args[0], attr=attr, ctx=ast.Load())
            out = ast.Call(func=target, args=[], keywords=[]) if fn == "conj" else target
            if fn != "conj":  # keep the result complex so `z` stays complex128
                out = ast.Call(func=ast.Name("complex", ast.Load()), args=[out, ast.Constant(0.0)], keywords=[])
            return out
        if fn in _CMATH:
            node.func = ast.Attribute(value=ast.Name("cmath", ast.Load()), attr=fn, ctx=ast.Load())
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in ("i", "j"):
            return ast.Constant(1j)
        if node.id == "pi":
            return ast.Constant(math.pi)
        return node


def compile_formula_expr(expr: str) -> Callable[..., Any]:
    from numba import njit

    from ..expr import _check

    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) or (isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES):
            bad = getattr(node, "id", getattr(node, "attr", "?"))
            raise ValueError(f"Formula {expr!r}: unknown name {bad!r}; allowed {sorted(_ALLOWED_NAMES)}")
    _check(tree, set(_ALLOWED_NAMES), expr)
    body = ast.fix_missing_locations(_Rewrite().visit(tree)).body
    fn_def = ast.parse("def step(z, c, zp, k):\n    return 0").body[0]
    fn_def.body[0].value = body  # type: ignore[attr-defined]
    module = ast.fix_missing_locations(ast.Module(body=[fn_def], type_ignores=[]))
    ns: dict[str, Any] = {"__builtins__": {}, "cmath": cmath, "abs": abs, "complex": complex}
    exec(compile(module, f"<formula {expr!r}>", "exec"), ns)  # noqa: S102 - AST-checked
    return njit(fastmath=True)(ns["step"])


@section("formula")
def _formula_section(name: str, table: dict[str, Any], source: str, replace: bool) -> None:
    t = dict(table)
    expr = t.pop("z")
    register_formula(
        name,
        compile_formula_expr(expr),
        julia=bool(t.pop("julia", False)),
        degree=float(t.pop("degree", 2.0)),
        bailout=float(t.pop("bailout", 2.0)),
        description=str(t.pop("description", expr)),
        replace=replace,
    )
    if t:
        raise ValueError(f"unknown formula keys {sorted(t)}")


def escape_field(
    cr: np.ndarray, ci: np.ndarray, max_iter: int, formula: str, c: complex, k: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (smooth_iteration, escaped_mask). Interior pixels are 0 / False."""
    from .. import kernels, progress

    f = FORMULAS.get(formula)
    cr = np.ascontiguousarray(cr, dtype=np.float64)
    ci = np.ascontiguousarray(ci, dtype=np.float64)
    if f.kind_id is not None:
        n, esc = kernels.escape_kernel(
            cr, ci, int(max_iter), f.kind_id, float(c.real), float(c.imag), float(k), progress.units(cr.shape[0])
        )
    else:
        n, esc = kernels.escape_kernel_generic(
            cr, ci, int(max_iter), f.step, f.julia, complex(c), float(k),
            float(f.bailout) ** 2, math.log(f.degree) if f.degree > 1.0 else 0.0, progress.units(cr.shape[0]),
        )
    return n, esc.astype(bool)


@generator(
    "escape_time",
    category="escape-time",
    uses_view=True,
    param_docs={
        "formula": "Which fractal to draw.",
        "max_iter": "More shows finer detail at the edges, but takes longer.",
        "c": "The constant that shapes a Julia set. Small changes make very different shapes. Mandelbrot-style formulas ignore it.",
        "k": "An extra ingredient some formulas use, like the Phoenix fractal. Most ignore it.",
        "palette": "Colors outside the set.",
        "coloring": "Histogram spreads the colors evenly, while smooth draws repeating bands.",
        "density": "In smooth mode, how tightly the color bands repeat.",
        "gamma": "Lower values brighten the midtones.",
        "cyclic": "Wrap the colors around instead of stopping at the end. Rarely needed.",
        "interior": "Color of the set itself, as RGB from 0 to 1.",
    },
    ranges={"max_iter": (10, 5000, 10), "k": (-1.5, 1.5), "density": (0.005, 0.5), "gamma": (0.2, 3.0)},
    choices={"formula": FORMULAS.names, "coloring": ["histogram", "smooth"]},
)
def escape_time(
    ctx,
    *,
    formula: str = "mandelbrot",
    max_iter: int = 500,
    c: complex = -0.7269 + 0.1889j,
    k: float = -0.5,
    palette: Any = "cosmic",
    coloring: str = "histogram",
    density: float = 0.05,
    gamma: float = 0.85,
    cyclic: bool = False,
    interior: tuple = (0.0, 0.0, 0.0),
):
    """The Mandelbrot and Julia family. Each point is colored by how quickly it flies off to infinity."""
    if int(max_iter) < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter!r}")
    fkey = field_ctx_key(ctx, "escape_field", formula=formula, max_iter=max_iter, c=c, k=k)

    def _field() -> tuple[np.ndarray, np.ndarray]:
        cr, ci = ctx.plane()
        return ctx.strips(lambda a, b: escape_field(a, b, max_iter, formula, c, k), cr, ci)

    n, esc = field_cached(fkey, _field)
    esc = np.asarray(esc, dtype=bool)
    _suggest_detail(n, esc, int(max_iter))
    if coloring == "smooth":
        t = np.where(esc, n * density, 0.0)
        cyclic = True
    elif coloring == "histogram":
        t = histogram_index(n, esc)
    else:
        raise ValueError(f"coloring must be 'histogram' or 'smooth', got {coloring!r}")
    rgb = sample_palette(palette, np.power(t, gamma), cyclic=cyclic)
    rgb[~esc] = np.array(interior, dtype=np.float64)
    return rgb


def _suggest_detail(n: np.ndarray, esc: np.ndarray, max_iter: int) -> None:
    """Deep zooms need more iterations: when most escaping points only just escape, the
    edges of the set are being cut off, so suggest doubling (no built-in look comes close)."""
    if not esc.any():
        return
    late = float(np.mean(n[esc] > 0.5 * max_iter))
    if late > 0.15:
        from .. import hints

        hints.suggest("more_detail", "This close in, the picture needs more detail to show its shape.",
                      param="max_iter", value=max_iter * 2, label="Add detail")


TRAPS = {"rings": 0, "cross": 1, "circle": 2, "point": 3}


@generator(
    "orbit_trap",
    category="escape-time",
    uses_view=True,
    param_docs={
        "max_iter": "More gives richer detail, but takes longer.",
        "trap": "The shape the paths are measured against.",
        "palette": "Colors from touching the shape to far away.",
        "gamma": "Lower values brighten the midtones.",
        "shade_gamma": "How much points far from the shape darken.",
        "floor": "Keeps the darkest areas from going fully black.",
    },
    ranges={"max_iter": (10, 2000, 10), "gamma": (0.2, 3.0), "shade_gamma": (0.2, 3.0), "floor": (0.0, 1.0)},
    choices={"trap": list(TRAPS)},
)
def orbit_trap(
    ctx,
    *,
    max_iter: int = 180,
    trap: str = "rings",
    palette: Any = "neon",
    gamma: float = 0.85,
    shade_gamma: float = 0.7,
    floor: float = 0.12,
):
    """The Mandelbrot set colored by how close each point's path comes to a shape."""
    from .. import kernels, progress

    if trap not in TRAPS:
        raise ValueError(f"trap must be one of {sorted(TRAPS)}")
    fkey = field_ctx_key(ctx, "orbit_trap", max_iter=max_iter, trap=trap)

    def _field() -> np.ndarray:
        cr, ci = ctx.plane()

        def run(a, b):
            return kernels.orbit_trap_kernel(
                np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64),
                int(max_iter), TRAPS[trap], progress.units(a.shape[0]),
            )

        return ctx.strips(run, cr, ci)

    d = field_cached(fkey, _field)
    t = histogram_index(-np.log(d), np.ones_like(d, dtype=bool))
    rgb = sample_palette(palette, np.power(t, gamma), cyclic=True)
    rgb *= floor + (1.0 - floor) * np.power(t, shade_gamma)[..., None]
    return np.clip(rgb, 0.0, 1.0)
