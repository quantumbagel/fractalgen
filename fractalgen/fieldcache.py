"""LRU cache for expensive generator fields (densities, iteration counts, …).

Palette, recipe ``tone`` black/white/curve, and ``post`` are applied after these
buffers, so GUI tweaks can reuse compute when only those change. Keys must list
every input that affects the cached arrays. Values are sealed read-only; copy
before in-place work.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Callable, Hashable

import numpy as np

from . import simcache
from .context import RenderContext
from .view import View

MAX_ENTRIES = 16
MAX_BYTES = 512 * 1024 * 1024

_cache: OrderedDict[Hashable, Any] = OrderedDict()
_lock = Lock()


def view_key(v: View) -> tuple[Any, ...]:
    return (v.center.real, v.center.imag, v.half_width, v.rotation, v.fit, v.ref_aspect)


def ctx_key(ctx: RenderContext, tag: str, **parts: Any) -> Hashable:
    tv = ctx.tone_view
    return (
        tag,
        ctx.name,
        ctx.width,
        ctx.height,
        ctx.n_strips,
        view_key(ctx.view),
        view_key(tv) if tv is not None else None,
        simcache.freeze(parts),
    )


def seal(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        out = np.array(value, copy=True)
        out.flags.writeable = False
        return out
    if isinstance(value, tuple):
        return tuple(seal(v) for v in value)
    if isinstance(value, list):
        return [seal(v) for v in value]
    return value


def _nbytes(value: Any) -> int:
    return simcache._nbytes(value)


def cached(key: Hashable, compute: Callable[[], Any]) -> Any:
    key = simcache.freeze(key)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    value = seal(compute())
    size = _nbytes(value)
    if size > MAX_BYTES // 4:
        return value
    with _lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > MAX_ENTRIES or sum(_nbytes(v) for v in _cache.values()) > MAX_BYTES:
            _cache.popitem(last=False)
    return value


def clear() -> None:
    with _lock:
        _cache.clear()
