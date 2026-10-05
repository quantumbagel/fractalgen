"""Load built-ins, data files, and plugins.

Search path (later entries may add to, or with ``replace``, override earlier):

1. built-in generators (``fractalgen.generators``)
2. built-in data (``fractalgen/data/**.toml``)
3. every entry of ``$FRACTALGEN_PATH`` (``os.pathsep``-separated)
4. paths passed to :func:`bootstrap` (CLI ``--load``)

A path may be a ``.toml`` data file, a ``.py`` plugin, or a directory of them.
A data file may hold any of these top-level tables::

    [palette.NAME]   stops = [...]                        # see palette.py
    [recipe.NAME]    generator = "..." ...                # see recipe.py
    [preset.NAME]    width = ... height = ...             # see engine.py
    [map.NAME]       x = "..." y = "..." params = {...}   # 2D iterated maps
    [flow.NAME]      dx = "..." dy = "..." dz = "..."     # 3D ODE attractors
    [formula.NAME]   z = "..."                            # escape-time formulas

and set ``replace = true`` at top level to override existing names.
A plugin is any Python file; it registers things by importing ``fractalgen``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterable

DATA_DIR = Path(__file__).resolve().parent / "data"

# section name -> handler(name, table, source, replace)
SECTION_HANDLERS: dict[str, Callable[[str, dict[str, Any], str, bool], None]] = {}

_loaded: list[str] = []
_booted = False


def section(name: str):
    """Register a handler for a new top-level TOML table (extension hook)."""

    def wrap(fn):
        SECTION_HANDLERS[name] = fn
        return fn

    return wrap


def load_toml(path: Path) -> None:
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    replace = bool(doc.pop("replace", False))
    for sect, entries in doc.items():
        if sect not in SECTION_HANDLERS:
            raise ValueError(f"{path}: unknown section [{sect}] (known: {sorted(SECTION_HANDLERS)})")
        if not isinstance(entries, dict):
            raise ValueError(f"{path}: [{sect}] must be a table of named entries")
        for name, table in entries.items():
            try:
                SECTION_HANDLERS[sect](name, table, str(path), replace)
            except Exception as exc:
                raise type(exc)(f"{path}: [{sect}.{name}]: {exc}") from exc


def load_plugin(path: Path) -> None:
    mod_name = f"fractalgen_plugin_{abs(hash(str(path.resolve())))}"
    if mod_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)


def load_path(path: str | Path) -> None:
    p = Path(path).expanduser()
    if p.is_dir():
        # Plugins first so data files can reference what they register.
        for child in sorted(p.rglob("*.py")):
            load_plugin(child)
        for child in sorted(p.rglob("*.toml")):
            load_toml(child)
    elif p.suffix == ".toml":
        load_toml(p)
    elif p.suffix == ".py":
        load_plugin(p)
    elif p.exists():
        raise ValueError(f"Don't know how to load {p} (want .toml, .py, or a directory)")
    else:
        # A dotted module name, e.g. `--load mypkg.fractals`
        importlib.import_module(str(path))


def _register_core_sections() -> None:
    from . import engine
    from .palette import register_palette
    from .recipe import add_recipe_table

    SECTION_HANDLERS.setdefault("palette", lambda n, t, s, r: register_palette(n, t, replace=r))
    SECTION_HANDLERS.setdefault("recipe", lambda n, t, s, r: add_recipe_table(n, t, s, replace=r))
    SECTION_HANDLERS.setdefault("preset", lambda n, t, s, r: engine.register_preset(n, t, replace=r))


def bootstrap(extra: Iterable[str | Path] = ()) -> list[str]:
    """Idempotently load everything. Returns the extra paths in load order."""
    global _booted
    from .recipe import finalize_recipes

    if not _booted:
        _register_core_sections()
        importlib.import_module("fractalgen.generators")
        load_toml(DATA_DIR / "palettes.toml")
        load_toml(DATA_DIR / "presets.toml")
        for f in sorted((DATA_DIR / "recipes").glob("*.toml")):
            load_toml(f)
        env = os.environ.get("FRACTALGEN_PATH", "")
        for entry in filter(None, env.split(os.pathsep)):
            load_path(entry)
        _booted = True
    for p in extra:
        key = str(p)
        if key not in _loaded:
            load_path(p)
            _loaded.append(key)
    finalize_recipes()
    return list(_loaded)
