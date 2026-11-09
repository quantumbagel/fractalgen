"""Per-render context handed to every generator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import progress
from .view import View

REF_4K = 3840 * 2160
_REF_720 = 1280 * 720

# Pixel count of the reference frame every look is defined at (see RenderContext.res_scale).
# Lowering it makes simulations cheaper but changes their look; the tests do that for speed.
REFERENCE_PIXELS = REF_4K


@dataclass
class RenderContext:
    width: int
    height: int
    view: View = field(default_factory=View)
    name: str = ""
    # Row strips for per-pixel kernels (bounds peak memory at 8K/16K).
    n_strips: int = 1
    # View this frame's tone statistics come from (see tone.py); None = this view.
    tone_view: View | None = None

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

    # ---- resolution independence ------------------------------------------
    # Looks are defined against the 4K reference canvas, sharing 3840x2160's pixel
    # count and this frame's aspect. Pixel-sized knobs are "pixels at 4K" and
    # simulations run on a grid sized from the reference, so a 720p preview and
    # a 16K export show the same picture (only the sampling differs).

    @property
    def res_scale(self) -> float:
        """Linear size relative to the reference: 0.5 at 1080p, 1 at 4K, 2 at 8K."""
        return math.sqrt(self.pixels / REFERENCE_PIXELS)

    @property
    def ref_size(self) -> tuple[int, int]:
        """(width, height) of the 4K reference frame with this aspect."""
        s = 1.0 / self.res_scale
        return max(1, round(self.width * s)), max(1, round(self.height * s))

    def ref_px(self, px_at_4k: float) -> float:
        """A length given in 4K pixels, in this frame's pixels."""
        return px_at_4k * self.res_scale

    def reference(self) -> "RenderContext":
        """This context at the 4K reference size (same view)."""
        w, h = self.ref_size
        return RenderContext(width=w, height=h, view=self.view, name=self.name)

    def density_norm(self, n_used: float, n_ref: float) -> float:
        """Factor that puts a hit-count buffer from ``n_used`` samples on the
        scale it would have at the reference, where ``n_ref`` samples are drawn.

        Hits per pixel otherwise change with resolution, and log tone mapping
        turns that into a brighter or darker picture. 1.0 at the reference.
        """
        rw, rh = self.ref_size
        return (n_ref / float(rw * rh)) / (max(n_used, 1) / float(self.pixels))

    def zoom_gain(self, power: float = 2.0) -> float:
        """How many times more hits per pixel the tone view would collect.

        Density renders over the plane multiply their hit counts by this so a
        zoom keeps the anchor's exposure: ``power`` 2 for area-filling density,
        1 for hits along curves. 1.0 on the tone view itself.
        """
        if self.tone_view is None:
            return 1.0
        here = self.view.extents(self.width, self.height)[0]
        there = self.tone_view.extents(self.width, self.height)[0]
        return (there / max(here, 1e-300)) ** power

    def sim_grid(self, scale: float, min_side: int = 8) -> tuple[int, int]:
        """Simulation grid of ``scale`` x the reference frame (independent of output size)."""
        rw, rh = self.ref_size
        return max(min_side, int(rw * scale)), max(min_side, int(rh * scale))

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
            with progress.span(y0 / height, y1 / height):
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
