"""Coerce recipe / CLI values (TOML, strings) into Python parameter types."""

from __future__ import annotations

import ast
import math
from typing import Any


def to_complex(value: Any) -> complex:
    """Accept ``[re, im]``, ``{re=, im=}``, numbers, or strings like ``"-0.8+0.156j"``."""
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float)):
        return complex(float(value), 0.0)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, dict) and "re" in value:
        return complex(float(value["re"]), float(value.get("im", 0.0)))
    if isinstance(value, str):
        return complex(value.replace(" ", "").replace("i", "j"))
    raise TypeError(f"Cannot interpret {value!r} as a complex number")


def parse_literal(text: str) -> Any:
    """Best-effort parse of a CLI ``key=value`` right-hand side."""
    s = text.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    try:
        return complex(s.replace("i", "j"))
    except ValueError:
        return s


def coerce(value: Any, like: Any) -> Any:
    """Coerce ``value`` to the type of the default ``like`` when that is unambiguous."""
    if value is None or like is None:
        return value
    if isinstance(like, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(like, complex):
        return to_complex(value)
    if isinstance(like, int) and not isinstance(value, bool):
        if isinstance(value, float) and not value.is_integer():
            return value
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return int(float(value))
    if isinstance(like, float) and isinstance(value, (int, str)) and not isinstance(value, bool):
        return float(value)
    if isinstance(like, tuple) and isinstance(value, list):
        return tuple(coerce(v, like[i]) if i < len(like) else v for i, v in enumerate(value))
    return value


def deg(value: float) -> float:
    """Degrees -> radians. Recipes always author angles in degrees."""
    return math.radians(float(value))


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
