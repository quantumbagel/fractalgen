"""Tone anchoring: keep auto-exposure fixed while the view moves.

Many generators stretch their output to the frame's own statistics (histogram
equalization, percentile ranges, peak normalization). Measured per frame, that
re-exposes the image every time the view pans or zooms: the same spot changes
brightness and color depending on what else is on screen.

Instead, a recipe's tone is measured once on its *anchor* view (the authored
view, before any GUI / CLI view edits) and reused for every other view::

    stats = tone.record(lambda: generator(anchor_ctx, **params))
    with tone.replaying(stats):
        rgb = generator(ctx, **params)

Generators opt in by routing each frame statistic through :func:`stat`::

    lo, hi = tone.stat("log_tone", lambda: np.percentile(d, [1, 99]))

Outside a session (or when a key was not recorded) ``stat`` just computes.

On top of that default, a recipe's ``tone`` table (``--tone key=value`` on the
CLI, the Tone section in the GUI) adjusts the stretch itself, before the
palette is applied::

    tone = { black = 0.05, white = 0.9, curve = 1.3 }

Generators pass their normalized 0..1 tone through :func:`levels` to honor it.
"""

from __future__ import annotations

import contextvars
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Hashable, Iterator

import numpy as np


@dataclass(frozen=True)
class ToneKey:
    doc: str
    neutral: Any
    range: tuple | None = None

    @property
    def kind(self) -> str:
        return "bool" if isinstance(self.neutral, bool) else "float"


TONE_KEYS: dict[str, ToneKey] = {
    "black": ToneKey("Black point of the auto stretch (0..1); higher = deeper shadows", 0.0, (0.0, 0.9)),
    "white": ToneKey("White point of the auto stretch (0..1); lower = brighter, blown-out highlights", 1.0, (0.1, 1.0)),
    "curve": ToneKey("Midtone curve on the stretch; >1 brightens midtones, <1 darkens them", 1.0, (0.2, 4.0)),
    "per_frame": ToneKey("Re-measure the auto stretch on every frame (pans and zooms re-expose)", False),
}


def check_tone(tone: dict[str, Any] | None) -> dict[str, Any]:
    """Validate a ``tone`` table and drop unset keys."""
    tone = {k: v for k, v in (tone or {}).items() if v is not None}
    unknown = sorted(set(tone) - set(TONE_KEYS))
    if unknown:
        raise KeyError(f"Unknown tone key(s) {unknown}. Known: {sorted(TONE_KEYS)}")
    return tone


_levels: contextvars.ContextVar[tuple[float, float, float] | None] = contextvars.ContextVar("tone_levels", default=None)


@contextmanager
def adjusting(tone: dict[str, Any] | None) -> Iterator[None]:
    """Apply a recipe's ``tone`` table to every :func:`levels` call inside."""
    tone = check_tone(tone)
    lv = (float(tone.get("black", 0.0)), float(tone.get("white", 1.0)), float(tone.get("curve", 1.0)))
    token = _levels.set(None if lv == (0.0, 1.0, 1.0) else lv)
    try:
        yield
    finally:
        _levels.reset(token)


def levels(t: np.ndarray) -> np.ndarray:
    """Black point, white point and curve on a 0..1 tone (identity by default)."""
    lv = _levels.get()
    if lv is None:
        return t
    black, white, curve = lv
    t = np.clip((t - black) / max(white - black, 1e-6), 0.0, 1.0)
    return t if curve == 1.0 else np.power(t, 1.0 / max(curve, 1e-6))


_session: contextvars.ContextVar["_Session | None"] = contextvars.ContextVar("tone_session", default=None)


class _Session:
    def __init__(self, recorded: dict[str, list[Any]] | None = None) -> None:
        self.recording = recorded is None
        self.stats: dict[str, list[Any]] = {} if recorded is None else recorded
        self.used: dict[str, int] = {}


def stat(key: str, compute: Callable[[], Any]) -> Any:
    """A frame statistic: measured on the anchor view, replayed on others.

    Calls are matched by ``key`` and call order within that key, so give
    conditional calls their own key.
    """
    s = _session.get()
    if s is None:
        return compute()
    if s.recording:
        value = compute()
        s.stats.setdefault(key, []).append(value)
        return value
    i = s.used.get(key, 0)
    s.used[key] = i + 1
    seq = s.stats.get(key, ())
    return seq[i] if i < len(seq) else compute()


def record(render: Callable[[], Any]) -> dict[str, list[Any]]:
    """Run ``render`` and return every statistic it measured."""
    s = _Session()
    token = _session.set(s)
    try:
        render()
    finally:
        _session.reset(token)
    return s.stats


@contextmanager
def replaying(stats: dict[str, list[Any]]) -> Iterator[None]:
    token = _session.set(_Session(stats))
    try:
        yield
    finally:
        _session.reset(token)


# Anchor stats are tiny; keep a few per process so pans and zooms reuse them.
_MAX_ENTRIES = 32
_cache: OrderedDict[Hashable, dict[str, list[Any]]] = OrderedDict()
_lock = Lock()


def anchored(key: Hashable, render: Callable[[], Any]) -> dict[str, list[Any]]:
    """:func:`record`, memoized on ``key`` (which must hold every input)."""
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    stats = record(render)
    with _lock:
        _cache[key] = stats
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)
    return stats


def clear() -> None:
    with _lock:
        _cache.clear()
