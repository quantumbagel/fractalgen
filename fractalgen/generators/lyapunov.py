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
    from .. import kernels, progress

    if not isinstance(sequence, str) or not sequence:
        raise ValueError(f"sequence must be a non-empty string of A's and B's, got {sequence!r}")
    seq = np.empty(len(sequence), dtype=np.uint8)
    for i, ch in enumerate(sequence):
        u = ch.upper()
        if u not in ("A", "B"):
            raise ValueError(f"sequence must contain only A and B, got {ch!r} in {sequence!r}")
        seq[i] = 1 if u == "B" else 0
    return kernels.lyapunov_kernel(
        np.ascontiguousarray(a, dtype=np.float64),
        np.ascontiguousarray(b, dtype=np.float64),
        seq,
        int(warmup),
        int(samples),
        progress.units(a.shape[0]),
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
        "sequence": "A pattern of A's and B's. Each pattern draws a different picture.",
        "warmup": "Settling time before measuring. More is cleaner but slower.",
        "samples": "How long each point is measured. More is sharper but slower.",
        "palette": "The first half of the colors shows stable areas, the second half chaotic ones.",
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
    """Lyapunov space: bands where a simple population model is stable, against swirls where it turns chaotic.

    The view's horizontal axis is the logistic r for letter A, the vertical axis r for B.
    """
    a_grid, b_grid = ctx.plane()
    exponents = _lyapunov_field(a_grid, b_grid, sequence, warmup, samples)
    return _color_lyapunov(exponents, palette)
