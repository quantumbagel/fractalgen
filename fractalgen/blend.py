"""Blend modes for layers and effects (all inputs RGB in 0..1, numpy arrays)."""

from __future__ import annotations

from typing import Callable

import numpy as np

BlendFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _overlay(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))


def _soft_light(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (1.0 - 2.0 * b) * a * a + 2.0 * b * a


BLEND_MODES: dict[str, BlendFn] = {
    "normal": lambda a, b: b,
    "multiply": lambda a, b: a * b,
    "screen": lambda a, b: 1.0 - (1.0 - a) * (1.0 - b),
    "overlay": _overlay,
    "soft_light": _soft_light,
    "add": lambda a, b: a + b,
    "difference": lambda a, b: np.abs(a - b),
    "lighten": np.maximum,
    "darken": np.minimum,
}


def blend(base: np.ndarray, top: np.ndarray, alpha: np.ndarray | float, mode: str = "normal") -> np.ndarray:
    """``top`` over ``base`` with blend ``mode``; ``alpha`` (H, W) or scalar is the top's coverage."""
    fn = BLEND_MODES.get(mode)
    if fn is None:
        raise ValueError(f"unknown blend mode {mode!r}; one of {list(BLEND_MODES)}")
    mixed = fn(np.clip(base, 0.0, 1.0) if mode != "normal" else base, top)
    a = alpha[..., None] if isinstance(alpha, np.ndarray) and alpha.ndim == base.ndim - 1 else alpha
    return base * (1.0 - a) + mixed * a
