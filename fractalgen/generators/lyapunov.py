"""Logistic-map Lyapunov exponent fractals in (a, b) parameter space."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..color import histogram_index
from ..palette import sample_palette
from ..view import REF_ASPECT
from .base import generator


def _lyapunov_field(
    a: np.ndarray,
    b: np.ndarray,
    sequence: str,
    warmup: int,
    samples: int,
) -> np.ndarray:
    from .. import kernels

    seq = np.array([1 if ch == "B" else 0 for ch in sequence], dtype=np.uint8)
    return kernels.lyapunov_kernel(
        np.ascontiguousarray(a, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
        seq,
        int(warmup),
        int(samples),
    )


def _color_lyapunov(exponents: np.ndarray, palette: Any) -> np.ndarray:
    t = np.zeros_like(exponents)
    cha = exponents > 0
    sta = ~cha
    if np.any(sta):
        s = exponents[sta]
        t[sta] = 0.5 * histogram_index(-s, np.ones_like(s, dtype=bool), key="lyapunov.stable")
    if np.any(cha):
        c = exponents[cha]
        t[cha] = 0.5 + 0.5 * histogram_index(c, np.ones_like(c, dtype=bool), key="lyapunov.chaotic")
    return sample_palette(palette, t)


@generator(
    "lyapunov",
    category="lyapunov",
    uses_view=True,
    # The plane is (a, b) = (re, im). cover keeps b spanning 3.4..4.0 at every aspect.
    default_view={"center": [3.7, 3.7], "half_width": 0.3 * REF_ASPECT, "fit": "cover"},
    param_docs={
        "sequence": "string of A's and B's picking a vs b each step; the pattern changes the image",
        "warmup": "settling iterations before measuring; higher = cleaner, slower",
        "samples": "iterations averaged into the exponent; higher = sharper, slower",
        "palette": "first half of the gradient colors stable pixels, second half chaotic",
    },
    ranges={
        "warmup": (10, 500, 10),
        "samples": (50, 2000, 10),
    },
)
def lyapunov(
    ctx,
    *,
    sequence: str = "BBBBBBAAAAAA",
    warmup: int = 80,
    samples: int = 220,
    palette: Any = "icefire",
):
    """Stable vs chaotic bands of a two-parameter logistic map driven by an A/B sequence.

    The view's horizontal axis is the logistic r for letter A, the vertical axis r for B.
    """
    a_grid, b_grid = ctx.plane()
    exponents = _lyapunov_field(a_grid, b_grid, sequence, warmup, samples)
    return _color_lyapunov(exponents, palette)
