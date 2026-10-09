"""Named registries that every extension point plugs into."""

from __future__ import annotations

import difflib
from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")

_LIST_HINTS: dict[str, str] = {
    "palette": "`fractalgen palettes`",
    "generator": "`fractalgen generators`",
    "recipe": "`fractalgen list`",
    "post op": "`fractalgen post`",
    "preset": "`fractalgen presets`",
}


def _default_source() -> str:
    from .loader import current_source

    return current_source.get() or ""


class Registry(Generic[T]):
    """A name -> item mapping with a decorator for registration.

    ``kind`` is only used in error messages ("Unknown palette 'x'").
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}
        self._sources: dict[str, str] = {}
        # Items a `replace` from another source shadowed, restored when that source is dropped.
        self._shadowed: dict[str, list[tuple[T, str]]] = {}

    def register(self, name: str, item: T, *, replace: bool = False, source: str | None = None) -> T:
        if name in self._items and not replace:
            prev = self._sources.get(name, "")
            where = f" (already in {prev})" if prev else ""
            raise KeyError(f"{self.kind} {name!r} is already registered{where}")
        src = source if source is not None else _default_source()
        if name in self._items and self._sources.get(name, "") != src:
            self._shadowed.setdefault(name, []).append((self._items[name], self._sources.get(name, "")))
        self._items[name] = item
        self._sources[name] = src
        return item

    def decorator(self, name: str | None = None, *, replace: bool = False) -> Callable[[T], T]:
        def wrap(item: T) -> T:
            self.register(name or getattr(item, "__name__"), item, replace=replace)
            return item

        return wrap

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            close = difflib.get_close_matches(name, self._items, n=3, cutoff=0.6)
            n = len(self._items)
            msg = f"Unknown {self.kind} {name!r}."
            if close:
                msg += f" Did you mean: {', '.join(close)}?"
            hint = _LIST_HINTS.get(self.kind, "`fractalgen list`")
            msg += f" ({n} known; see {hint})"
            raise KeyError(msg) from None

    def source_of(self, name: str) -> str:
        return self._sources.get(name, "")

    def unregister(self, name: str) -> None:
        self._items.pop(name, None)
        self._sources.pop(name, None)
        self._shadowed.pop(name, None)

    def drop_from(self, source: str) -> list[str]:
        """Remove every item registered from ``source``. Returns removed names.

        An item that replaced one from another source (``replace = true``)
        brings the one it shadowed back, so dropping a user override restores
        the built-in.
        """
        if not source:
            return []
        for stack in self._shadowed.values():
            stack[:] = [entry for entry in stack if entry[1] != source]
        removed = [n for n, s in self._sources.items() if s == source]
        for n in removed:
            stack = self._shadowed.get(n)
            if stack:
                self._items[n], self._sources[n] = stack.pop()
            else:
                self.unregister(n)
        return removed

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def names(self) -> list[str]:
        return list(self._items)

    def items(self):
        return self._items.items()
