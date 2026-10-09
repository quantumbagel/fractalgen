"""Render progress: a fraction in [0, 1] that any thread can poll.

A :class:`Tracker` is active on the thread that renders (see :func:`tracking`).
Code reports into the current *span*, a slice of the whole render::

    with progress.span(0.1, 1.0, "rendering"):
        for i in range(steps):
            ...
            progress.update((i + 1) / steps)     # 0..1 of the span

Numba kernels can't call back into Python, so they report through flags
instead: ``done = progress.units(h)`` covers the current span, the kernel sets
``done[y] = 1`` as it finishes each row (or step), and ``Tracker.fraction``
counts them. Those kernels are ``nogil`` so a poller can run mid-call.

Without an active tracker everything here is a no-op (``units`` still returns
a scratch array, so kernels can always take one).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

import numpy as np

_local = threading.local()


class Tracker:
    """Progress of one render. Written by the rendering thread, read by any."""

    def __init__(self) -> None:
        self.stage = ""
        self._value = 0.0
        self._live: tuple[np.ndarray, float, float] | None = None  # (flags, lo, hi)
        self._spans: list[tuple[float, float]] = [(0.0, 1.0)]

    @property
    def fraction(self) -> float:
        v = self._value
        live = self._live
        if live is not None:
            flags, lo, hi = live
            v = max(v, lo + (hi - lo) * np.count_nonzero(flags) / flags.size)
        return min(1.0, v)

    def _set(self, value: float) -> None:
        if value > self._value:
            self._value = value

    def _fold(self) -> None:
        """Bank the live flags' progress and stop counting them."""
        if self._live is not None:
            self._set(self.fraction)
            self._live = None

    def _abs(self, f: float) -> float:
        lo, hi = self._spans[-1]
        return lo + (hi - lo) * min(1.0, max(0.0, f))


def current() -> Tracker | None:
    return getattr(_local, "tracker", None)


@contextmanager
def tracking(tracker: Tracker | None = None) -> Iterator[Tracker]:
    """Make ``tracker`` (or a new one) receive this thread's progress."""
    t = tracker or Tracker()
    prev = current()
    _local.tracker = t
    try:
        yield t
    finally:
        t._fold()
        _local.tracker = prev


@contextmanager
def span(lo: float, hi: float, stage: str | None = None) -> Iterator[None]:
    """Map progress inside the block to ``lo..hi`` of the enclosing span."""
    t = current()
    if t is None:
        yield
        return
    a, b = t._abs(lo), t._abs(hi)
    t._fold()
    t._set(a)
    t._spans.append((a, b))
    prev_stage = t.stage
    if stage:
        t.stage = stage
    try:
        yield
    finally:
        t._spans.pop()
        t._fold()
        t._set(b)
        t.stage = prev_stage


def update(f: float) -> None:
    """``f`` (0..1) of the current span is done."""
    t = current()
    if t is not None:
        t._set(t._abs(f))


def units(n: int) -> np.ndarray:
    """``n`` zeroed uint8 flags whose set fraction is the current span's progress."""
    flags = np.zeros(max(1, int(n)), dtype=np.uint8)
    t = current()
    if t is not None:
        t._fold()
        lo, hi = t._spans[-1]
        t._live = (flags, lo, hi)
    return flags
