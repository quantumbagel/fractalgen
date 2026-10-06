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
    f = Formula(name, step=step, julia=julia, degree=degree, bailout=bailout, description=description)
    return FORMULAS.register(name, f, replace=replace)


# Expression formulas: complex-scalar math compiled straight to numba.
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

    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) or (isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES):
            bad = getattr(node, "id", getattr(node, "attr", "?"))
            raise ValueError(f"Formula {expr!r}: unknown name {bad!r}; allowed {sorted(_ALLOWED_NAMES)}")
    body = ast.fix_missing_locations(_Rewrite().visit(tree)).body
    fn_def = ast.parse("def step(z, c, zp, k):\n    return 0").body[0]
    fn_def.body[0].value = body  # type: ignore[attr-defined]
    module = ast.fix_missing_locations(ast.Module(body=[fn_def], type_ignores=[]))
    ns: dict[str, Any] = {"cmath": cmath}
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
    from .. import kernels

    f = FORMULAS.get(formula)
    cr = np.ascontiguousarray(cr, dtype=np.float64)
    ci = np.ascontiguousarray(ci, dtype=np.float64)
    if f.kind_id is not None:
        n, esc = kernels.escape_kernel(cr, ci, int(max_iter), f.kind_id, float(c.real), float(c.imag), float(k))
    else:
        n, esc = kernels.escape_kernel_generic(
            cr, ci, int(max_iter), f.step, f.julia, complex(c), float(k),
            float(f.bailout) ** 2, math.log(f.degree) if f.degree > 1.0 else 0.0,
        )
    return n, esc.astype(bool)


@generator(
    "escape_time",
    category="escape-time",
    uses_view=True,
    param_docs={
        "formula": "which fractal to draw (mandelbrot, julia, burning_ship, … or a custom name)",
        "max_iter": "iterations per pixel; higher = more filament detail, slower",
        "c": "complex constant [re, im] for Julia formulas; ignored by Mandelbrot-style",
        "k": "extra formula weight (phoenix: mix of previous z); ignored by most formulas",
        "palette": "color gradient for the exterior (name, name_r to reverse, or hex list)",
        "coloring": "histogram = even color spread; smooth = repeating bands from iteration count",
        "density": "palette cycles per iteration in smooth mode; higher = thinner color bands",
        "gamma": "tone curve on the color index; lower = brighter midtones",
        "cyclic": "wrap the palette when the index exceeds 1 (smooth always wraps; rarely change)",
        "interior": "interior color [r, g, b] in 0..1 for points that never escape (the 'set')",
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
    """Mandelbrot / Julia family: each pixel colored by how fast its orbit escapes."""
    cr, ci = ctx.plane()
    n, esc = ctx.strips(lambda a, b: escape_field(a, b, max_iter, formula, c, k), cr, ci)
    del cr, ci
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


TRAPS = {"rings": 0, "cross": 1, "circle": 2, "point": 3}


@generator(
    "orbit_trap",
    category="escape-time",
    uses_view=True,
    param_docs={
        "max_iter": "orbit steps per pixel; higher = richer trap hits, slower",
        "trap": "shape each orbit is measured against (rings, cross, circle, or point)",
        "palette": "color gradient mapped to closeness to the trap",
        "gamma": "tone curve on trap-distance colors; lower = brighter midtones",
        "shade_gamma": "darkening from trap distance; higher = punchier contrast, darker far pixels",
        "floor": "minimum brightness mix 0..1; higher = less dark shading overall",
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
    """Mandelbrot colored by how close each orbit comes to a trap shape."""
    from .. import kernels

    if trap not in TRAPS:
        raise ValueError(f"trap must be one of {sorted(TRAPS)}")
    cr, ci = ctx.plane()

    def run(a, b):
        return kernels.orbit_trap_kernel(
            np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64),
            int(max_iter), TRAPS[trap],
        )

    d = ctx.strips(run, cr, ci)
    t = histogram_index(-np.log(d), np.ones_like(d, dtype=bool))
    rgb = sample_palette(palette, np.power(t, gamma), cyclic=True)
    rgb *= floor + (1.0 - floor) * np.power(t, shade_gamma)[..., None]
    return np.clip(rgb, 0.0, 1.0)
