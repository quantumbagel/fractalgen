"""Logistic-map Lyapunov exponent fractals in (a, b) parameter space."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..color import histogram_index
from ..palette import sample_palette
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
        t[sta] = 0.5 * histogram_index(-s, np.ones_like(s, dtype=bool))
    if np.any(cha):
        c = exponents[cha]
        t[cha] = 0.5 + 0.5 * histogram_index(c, np.ones_like(c, dtype=bool))
    return sample_palette(palette, t)


@generator(
    "lyapunov",
    category="lyapunov",
    uses_view=False,
    portrait_params={"a_focus": 0.62},
    param_docs={
        "sequence": "string of A's and B's picking a vs b each step; the pattern changes the image",
        "a_range": "[lo, hi] logistic r for letter A; view is centered at a_focus along this span",
        "b_range": "[lo, hi] logistic r for letter B; sets the full vertical axis span",
        "warmup": "settling iterations before measuring; higher = cleaner, slower",
        "samples": "iterations averaged into the exponent; higher = sharper, slower",
        "palette": "first half of the gradient colors stable pixels, second half chaotic",
        "a_focus": "horizontal center as 0..1 of a_range; 0.5 = midpoint",
    },
    ranges={
        "warmup": (10, 500, 10),
        "samples": (50, 2000, 10),
        "a_focus": (0.0, 1.0),
    },
)
def lyapunov(
    ctx,
    *,
    sequence: str = "BBBBBBAAAAAA",
    a_range: tuple = (3.4, 4.0),
    b_range: tuple = (3.4, 4.0),
    warmup: int = 80,
    samples: int = 220,
    palette: Any = "icefire",
    a_focus: float = 0.5,
):
    """Stable vs chaotic bands of a two-parameter logistic map driven by an A/B sequence."""
    a0, a1 = float(a_range[0]), float(a_range[1])
    b0, b1 = float(b_range[0]), float(b_range[1])
    mid_a = a0 + a_focus * (a1 - a0)
    mid_b = 0.5 * (b0 + b1)
    db = 0.5 * (b1 - b0)
    da = db * (ctx.width / float(ctx.height))
    a = np.linspace(mid_a - da, mid_a + da, ctx.width)
    b = np.linspace(mid_b + db, mid_b - db, ctx.height)
    a_grid, b_grid = np.meshgrid(a, b)
    exponents = _lyapunov_field(a_grid, b_grid, sequence, warmup, samples)
    return _color_lyapunov(exponents, palette)
