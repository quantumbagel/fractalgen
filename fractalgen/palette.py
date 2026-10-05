"""Color palettes: gradient stops that map scalars in [0, 1] to RGB.

Palettes are data. The built-ins live in ``data/palettes.toml``; more can be
added from any TOML file (``[palette.NAME] stops = [...]``) or from Python
with :func:`register_palette`.

Anywhere a palette is accepted you may pass:

* a registered name: ``"ember"``
* a reversed registered name: ``"ember_r"``
* a list of hex stops: ``["000000", "ff8800", "ffffff"]``
* a table: ``{stops = [...], positions = [0, 0.2, 1]}``
* a :class:`Palette` instance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .registry import Registry


def hex_to_rgb(c: str) -> tuple[float, float, float]:
    c = c.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(f"Bad hex color {c!r}")
    return (int(c[0:2], 16) / 255.0, int(c[2:4], 16) / 255.0, int(c[4:6], 16) / 255.0)


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    return "".join(f"{int(round(float(x) * 255)):02x}" for x in rgb)


@dataclass(frozen=True)
class Palette:
    name: str
    stops: np.ndarray  # (n, 3) float64 in [0, 1]
    positions: np.ndarray | None = None  # optional (n,) increasing in [0, 1]
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_spec(cls, name: str, spec: Any) -> "Palette":
        if isinstance(spec, Palette):
            return spec
        if isinstance(spec, dict):
            stops = spec["stops"]
            positions = spec.get("positions")
            desc = spec.get("description", "")
            tags = tuple(spec.get("tags", ()))
        else:
            stops, positions, desc, tags = spec, None, "", ()
        rgb = np.array(
            [hex_to_rgb(s) if isinstance(s, str) else tuple(float(x) for x in s) for s in stops],
            dtype=np.float64,
        )
        if rgb.ndim != 2 or rgb.shape[1] != 3 or rgb.shape[0] < 2:
            raise ValueError(f"Palette {name!r} needs at least two RGB stops")
        pos = None
        if positions is not None:
            pos = np.asarray(positions, dtype=np.float64)
            if pos.shape != (rgb.shape[0],) or np.any(np.diff(pos) < 0):
                raise ValueError(f"Palette {name!r}: positions must be increasing, one per stop")
        return cls(name=name, stops=rgb, positions=pos, description=desc, tags=tags)

    def reversed(self) -> "Palette":
        pos = None if self.positions is None else (1.0 - self.positions)[::-1].copy()
        return Palette(f"{self.name}_r", self.stops[::-1].copy(), pos, self.description, self.tags)

    def sample(self, t: np.ndarray, *, cyclic: bool = False) -> np.ndarray:
        """Map values in [0, 1] to RGB with linear interpolation between stops."""
        stops = self.stops
        t = np.asarray(t, dtype=np.float64)
        t = np.mod(t, 1.0) if cyclic else np.clip(t, 0.0, 1.0)
        if self.positions is not None:
            return np.stack([np.interp(t, self.positions, stops[:, k]) for k in range(3)], axis=-1)
        n = len(stops) - 1
        x = t * n
        i = np.clip(np.floor(x).astype(np.int32), 0, n - 1)
        f = (x - i)[..., None]
        return stops[i] * (1.0 - f) + stops[i + 1] * f

    def hex_stops(self) -> list[str]:
        return [_rgb_to_hex(s) for s in self.stops]


PALETTES: Registry[Palette] = Registry("palette")


def register_palette(name: str, spec: Any, *, replace: bool = False) -> Palette:
    pal = Palette.from_spec(name, spec)
    if pal.name != name:
        pal = Palette(name, pal.stops, pal.positions, pal.description, pal.tags)
    return PALETTES.register(name, pal, replace=replace)


def get_palette(spec: Any) -> Palette:
    """Resolve any palette spec (see module docstring) to a :class:`Palette`."""
    if isinstance(spec, Palette):
        return spec
    if isinstance(spec, str):
        if spec in PALETTES:
            return PALETTES.get(spec)
        if spec.endswith("_r") and spec[:-2] in PALETTES:
            return PALETTES.get(spec[:-2]).reversed()
        return PALETTES.get(spec)  # raises with the known list
    return Palette.from_spec("<inline>", spec)


def sample_palette(spec: Any, t: np.ndarray, *, cyclic: bool = False) -> np.ndarray:
    return get_palette(spec).sample(t, cyclic=cyclic)
