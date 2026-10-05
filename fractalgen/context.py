"""Per-render context handed to every generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .view import View

REF_4K = 3840 * 2160
_REF_720 = 1280 * 720


@dataclass
class RenderContext:
    width: int
    height: int
    view: View = field(default_factory=View)
    name: str = ""
    # Row strips for per-pixel kernels (bounds peak memory at 8K/16K).
    n_strips: int = 1

    @property
    def portrait(self) -> bool:
        return self.height > self.width

    @property
    def aspect(self) -> float:
        return self.width / float(self.height)

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def plane(self, view: View | None = None) -> tuple[np.ndarray, np.ndarray]:
        return (view or self.view).plane(self.width, self.height)

    def ultra_mul(self, cap: float = 4.0) -> float:
        """1 at 4K, 4 at 8K. Never shrinks sub-4K counts."""
        return min(cap, max(1.0, self.pixels / float(REF_4K)))

    def scale_particles(self, base: int, cap: int) -> int:
        """Particle budget for density renders.

        Up to 4K this follows a gentle area curve. Above 4K the count grows
        with pixel count so hits/pixel (and zoom-in sharpness) stay constant.
        """
        px = self.pixels
        if px <= REF_4K * 1.05:
            area = max(1.0, px / _REF_720)
            scale = min(7.5, area**0.42)
            scaled = int(base * max(1.0, scale))
            dyn_cap = max(cap, int(cap * max(1.0, area**0.25)))
            return min(dyn_cap, scaled)
        area_4k = REF_4K / _REF_720
        scale_4k = min(7.5, area_4k**0.42)
        n4k = int(base * max(1.0, scale_4k))
        dyn_cap_4k = max(cap, int(cap * max(1.0, area_4k**0.25)))
        n4k = min(dyn_cap_4k, n4k)
        return min(int(n4k * px / REF_4K), 2_500_000)

    def strips(self, fn: Callable[..., Any], *arrays: np.ndarray, **kwargs: Any) -> Any:
        """Run per-pixel ``fn(*arrays, **kwargs)`` in horizontal row strips.

        ``arrays`` must all be (H, W[, ...]); the result (array or tuple of
        arrays) is reassembled to full height.
        """
        height = int(arrays[0].shape[0])
        bounds = _strip_bounds(height, self.n_strips)
        if len(bounds) <= 1:
            return fn(*arrays, **kwargs)
        full: Any = None
        for y0, y1 in bounds:
            piece = fn(*(a[y0:y1] for a in arrays), **kwargs)
            if full is None:
                full = _alloc(height, piece)
            _place(full, y0, y1, piece)
            del piece
        return full


def _strip_bounds(height: int, n_strips: int) -> list[tuple[int, int]]:
    n_strips = max(1, min(int(n_strips), height))
    base, rem = divmod(height, n_strips)
    bounds, y = [], 0
    for i in range(n_strips):
        span = base + (1 if i < rem else 0)
        bounds.append((y, y + span))
        y += span
    return bounds


def _alloc(height: int, piece: Any) -> Any:
    if isinstance(piece, tuple):
        return tuple(np.empty((height,) + p.shape[1:], dtype=p.dtype) for p in piece)
    return np.empty((height,) + piece.shape[1:], dtype=piece.dtype)


def _place(full: Any, y0: int, y1: int, piece: Any) -> None:
    if isinstance(piece, tuple):
        for dst, src in zip(full, piece):
            dst[y0:y1] = src
    else:
        full[y0:y1] = piece
