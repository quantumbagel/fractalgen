"""Memoize expensive simulation results within a process.

Simulations run on a grid that does not depend on the output size (see
``RenderContext.sim_grid``), so the GUI's draft and full previews, palette and
post tweaks, and repeated renders can all reuse one result::

    u, v = simcache.cached(("gray_scott", sw, sh, F, k, steps, seed, Du, Dv, dt),
                           lambda: run_simulation(...))

Keys must hold every input that affects the result. Values are returned as-is;
callers must not modify them in place (copy first).
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any, Callable, Hashable

MAX_ENTRIES = 6
MAX_BYTES = 1_500_000_000  # total budget; results bigger than a quarter of it are not kept

_cache: OrderedDict[Hashable, Any] = OrderedDict()
_lock = Lock()


def freeze(value: Any) -> Hashable:
    """A hashable stand-in for nested lists / dicts / arrays-as-lists used in keys."""
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def _nbytes(value: Any) -> int:
    if isinstance(value, (tuple, list)):
        return sum(_nbytes(v) for v in value)
    return int(getattr(value, "nbytes", 0))


def cached(key: Hashable, compute: Callable[[], Any]) -> Any:
    key = freeze(key)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    value = compute()
    size = _nbytes(value)
    if size > MAX_BYTES // 4:  # e.g. a 16K export's buffers: don't pin them in memory
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
