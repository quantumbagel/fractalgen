"""Named registries that every extension point plugs into."""

from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> item mapping with a decorator for registration.

    ``kind`` is only used in error messages ("Unknown palette 'x'").
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T, *, replace: bool = False) -> T:
        if name in self._items and not replace:
            raise KeyError(f"{self.kind} {name!r} is already registered")
        self._items[name] = item
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
            known = ", ".join(sorted(self._items)) or "(none)"
            raise KeyError(f"Unknown {self.kind} {name!r}. Known: {known}") from None

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
