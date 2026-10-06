"""Run elementwise numpy work on all cores.

numpy ufuncs release the GIL, so an elementwise function applied to chunks of
its input arrays in threads gives exactly the same result as one big call, just
faster. Use it for per-particle steps (maps, ODE integrators, vector fields)::

    x, y = chunked(step, x, y)

``fn`` must treat every element independently (no reductions, no RNG).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

WORKERS = max(1, os.cpu_count() or 1)
MIN_CHUNK = 32_768  # below this, thread overhead outweighs the gain

_pool: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="fractalgen")
    return _pool


def chunked(fn: Callable[..., Any], *arrays: np.ndarray) -> Any:
    """``fn(*arrays)`` computed on row chunks in parallel; returns an array or tuple of arrays."""
    n = int(arrays[0].shape[0])
    parts = min(WORKERS, n // MIN_CHUNK)
    if parts <= 1:
        return fn(*arrays)
    bounds = np.linspace(0, n, parts + 1).astype(int)
    futures = [_executor().submit(fn, *(a[i:j] for a in arrays)) for i, j in zip(bounds[:-1], bounds[1:])]
    results = [f.result() for f in futures]
    if isinstance(results[0], tuple):
        return tuple(np.concatenate([r[k] for r in results]) for k in range(len(results[0])))
    return np.concatenate(results)
