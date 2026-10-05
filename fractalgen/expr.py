"""Tiny math-expression compiler so formulas can live in TOML.

Expressions are plain Python arithmetic over numpy arrays, e.g.
``"sin(a*y) + c*cos(a*x)"`` or ``"(z**2 - 1) / (z**2 + z + 1)"``.
Only names from :data:`NAMESPACE` plus the declared variables / parameters
are visible; attribute access and dunder names are rejected.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, Sequence

import numpy as np

NAMESPACE: dict[str, Any] = {
    name: getattr(np, name)
    for name in (
        "sin cos tan arcsin arccos arctan arctan2 sinh cosh tanh arcsinh arccosh arctanh "
        "exp log log2 log10 sqrt abs sign floor ceil mod hypot minimum maximum clip "
        "real imag conj angle where pi e"
    ).split()
}
NAMESPACE.update(asin=np.arcsin, acos=np.arccos, atan=np.arctan, atan2=np.arctan2, i=1j, j=1j)


def _check(tree: ast.AST, allowed: set[str], source: str) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            raise ValueError(f"Attribute access not allowed in expression: {source!r}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id not in allowed:
                raise ValueError(f"Unknown name {node.id!r} in expression: {source!r}")
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            raise ValueError(f"Unsupported syntax in expression: {source!r}")


def compile_expr(source: str, variables: Sequence[str], params: Sequence[str] = ()) -> Callable[..., Any]:
    """Compile ``source`` into ``f(*variables, **params)``."""
    tree = ast.parse(source.strip(), mode="eval")
    allowed = set(NAMESPACE) | set(variables) | set(params)
    _check(tree, allowed, source)
    code = compile(tree, f"<expr {source!r}>", "eval")
    var_names = tuple(variables)

    def fn(*args: Any, **kw: Any) -> Any:
        scope = dict(zip(var_names, args))
        scope.update(kw)
        return eval(code, {"__builtins__": {}, **NAMESPACE}, scope)  # noqa: S307 - AST-checked

    fn.__doc__ = source
    return fn
