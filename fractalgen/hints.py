"""Suggestions a render can make about its own settings, shown in the GUI.

A generator that notices something off (say, a deep zoom with too few
iterations) calls :func:`suggest`; whoever is rendering collects them::

    with hints.collecting() as found:
        rgb = recipe.render(w, h)
    found  # [{"key": "more_detail", "text": "...", "param": "max_iter", "value": 1400}]

Outside :func:`collecting` suggestions are dropped.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator

_found: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("hints", default=None)


def suggest(key: str, text: str, **data: Any) -> None:
    """Offer one suggestion (later ones with the same key replace earlier ones)."""
    found = _found.get()
    if found is None:
        return
    found[:] = [h for h in found if h["key"] != key]
    found.append({"key": key, "text": text, **data})


def replay(items: list[dict[str, Any]]) -> None:
    for h in items:
        suggest(**h)


@contextmanager
def collecting() -> Iterator[list[dict[str, Any]]]:
    found: list[dict[str, Any]] = []
    token = _found.set(found)
    try:
        yield found
    finally:
        _found.reset(token)


@contextmanager
def capturing() -> Iterator[list[dict[str, Any]]]:
    """Collect suggestions made inside, and pass them on to any outer collector too."""
    with collecting() as found:
        yield found
    replay(found)
