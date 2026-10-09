"""Load built-ins, entry-point plugins, data files, and user extensions.

Load order (later may override earlier when ``replace`` is set):

1. built-in generators (``fractalgen.generators``) and ``fractalgen/data/**/*.toml``
2. ``fractalgen.plugins`` entry points (``pyproject.toml`` / pip packages)
3. ``$FRACTALGEN_PATH`` (``os.pathsep``-separated)
4. project ``custom/`` if present
5. paths passed to :func:`bootstrap` (CLI ``--load``)

Python plugins use module names ``fractalgen_user.<stem>_<hash>`` and register via
``fractalgen`` decorators while :data:`current_source` points at the file path.
TOML files use the same ``source`` for registry tracking and hot reload via
:func:`reload_user_paths`.

Extension hooks: :func:`section` for new top-level tables; see ``README.md``.
"""

from __future__ import annotations

import contextvars
import hashlib
import importlib
import importlib.util
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterable

DATA_DIR = Path(__file__).resolve().parent / "data"
CUSTOM_DIR = Path(__file__).resolve().parent.parent / "custom"

SECTION_HANDLERS: dict[str, Callable[[str, dict[str, Any], str, bool], None]] = {}

# Default ``source`` for Registry.register when not passed explicitly.
current_source: contextvars.ContextVar[str] = contextvars.ContextVar("fractalgen_source", default="")

_loaded_extra: list[str] = []
_loaded_sources: list[str] = []
_reload_files: list[str] = []
_entry_points_loaded: list[str] = []
_booted = False


class LoadError(ValueError):
    """A TOML or plugin load failure with location metadata."""

    def __init__(
        self,
        message: str,
        *,
        path: str | Path | None = None,
        section: str | None = None,
        entry: str | None = None,
        line: int | None = None,
        col: int | None = None,
    ) -> None:
        self.path = str(path) if path is not None else None
        self.section = section
        self.entry = entry
        self.line = line
        self.col = col
        super().__init__(message)


def loaded_sources() -> list[str]:
    """Paths and entry-point ids loaded so far (diagnostics)."""
    return list(_loaded_sources)


def section(name: str):
    """Register a handler for a new top-level TOML table (extension hook)."""

    def wrap(fn):
        SECTION_HANDLERS[name] = fn
        return fn

    return wrap


def _note_source(key: str) -> None:
    if key not in _loaded_sources:
        _loaded_sources.append(key)


def _note_reload_file(path: Path) -> None:
    key = str(path.resolve())
    if key not in _reload_files:
        _reload_files.append(key)


def _plugin_mod_name(path: Path) -> str:
    stem = path.stem.replace(".", "_")
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
    return f"fractalgen_user.{stem}_{digest}"


def _drop_source(source: str) -> dict[str, list[str]]:
    from .engine import PRESETS
    from .generators.base import GENERATORS
    from .generators.domain import DOMAIN
    from .generators.escape import FORMULAS
    from .generators.fields import VECTOR_FIELDS
    from .generators.flows import FLOWS
    from .generators.maps import MAPS
    from .palette import PALETTES
    from .layers import LAYER_TYPES
    from .mask import SHAPES
    from .post import POST_OPS
    from .recipe import RECIPES, _RAW

    registries = {
        "generators": GENERATORS,
        "palettes": PALETTES,
        "recipes": RECIPES,
        "post_ops": POST_OPS,
        "shapes": SHAPES,
        "layer_types": LAYER_TYPES,
        "presets": PRESETS,
        "maps": MAPS,
        "flows": FLOWS,
        "formulas": FORMULAS,
        "vector_fields": VECTOR_FIELDS,
        "domain": DOMAIN,
    }
    out: dict[str, list[str]] = {}
    for label, reg in registries.items():
        out[label] = reg.drop_from(source)
    for name in list(_RAW):
        if _RAW[name][1] == source:
            del _RAW[name]
    return out


def _toml_has_known_sections(doc: dict[str, Any]) -> bool:
    return any(k in SECTION_HANDLERS for k in doc)


def _looks_like_load_path(path: str) -> bool:
    if "/" in path or "\\" in path or path.startswith("."):
        return True
    suf = Path(path).suffix
    return suf in (".toml", ".py")


def _is_builtin_data(path: Path) -> bool:
    p = path.resolve()
    data = DATA_DIR.resolve()
    if p in (data / "palettes.toml", data / "presets.toml"):
        return True
    return p.parent == data / "recipes"


def load_toml(path: Path, *, skip_unrelated: bool = False) -> None:
    path = path.resolve()
    try:
        with open(path, "rb") as f:
            doc = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        line = getattr(exc, "lineno", None)
        col = getattr(exc, "colno", None)
        raise LoadError(
            f"{path}: invalid TOML: {exc}",
            path=path,
            line=line,
            col=col,
        ) from exc
    if skip_unrelated and not _toml_has_known_sections(doc):
        return
    _note_source(str(path))
    if not _is_builtin_data(path):
        _note_reload_file(path)
    replace = bool(doc.pop("replace", False))
    token = current_source.set(str(path))
    src = str(path)
    try:
        for sect, entries in doc.items():
            if sect not in SECTION_HANDLERS:
                raise LoadError(
                    f"{path}: unknown section [{sect}] (known: {sorted(SECTION_HANDLERS)})",
                    path=path,
                    section=sect,
                )
            if not isinstance(entries, dict):
                raise LoadError(f"{path}: [{sect}] must be a table of named entries", path=path, section=sect)
            for name, table in entries.items():
                if not isinstance(table, dict):
                    raise LoadError(
                        f"{path}: [{sect}.{name}] must be a table",
                        path=path,
                        section=sect,
                        entry=name,
                    )
                entry_table = dict(table)
                entry_replace = bool(entry_table.pop("replace", replace))
                try:
                    SECTION_HANDLERS[sect](name, entry_table, str(path), entry_replace)
                except KeyError as exc:
                    if len(exc.args) == 1 and exc.args[0] and " " not in str(exc.args[0]):
                        key = exc.args[0]
                        raise LoadError(
                            f"{path}: [{sect}.{name}]: missing required key {key!r}",
                            path=path,
                            section=sect,
                            entry=name,
                        ) from exc
                    detail = exc.args[0] if exc.args else exc
                    raise LoadError(
                        f"{path}: [{sect}.{name}]: {detail}",
                        path=path,
                        section=sect,
                        entry=name,
                    ) from exc
                except LoadError:
                    raise
                except Exception as exc:
                    raise LoadError(
                        f"{path}: [{sect}.{name}]: {exc}",
                        path=path,
                        section=sect,
                        entry=name,
                    ) from exc
    except Exception:
        _drop_source(src)
        raise
    finally:
        current_source.reset(token)


def _ensure_user_package() -> None:
    if "fractalgen_user" not in sys.modules:
        import types

        pkg = types.ModuleType("fractalgen_user")
        pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["fractalgen_user"] = pkg


def load_plugin(path: Path, *, force: bool = False) -> None:
    path = path.resolve()
    mod_name = _plugin_mod_name(path)
    if not force and mod_name in sys.modules:
        return
    if force and mod_name in sys.modules:
        del sys.modules[mod_name]
    if force:
        cached = importlib.util.cache_from_source(str(path))
        try:
            Path(cached).unlink()
        except OSError:
            pass
    _ensure_user_package()
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin {path}")
    mod = importlib.util.module_from_spec(spec)
    _note_source(str(path))
    _note_reload_file(path)
    token = current_source.set(str(path))
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        _drop_source(str(path))
        sys.modules.pop(mod_name, None)
        raise ImportError(f"{path}: plugin failed: {exc}") from exc
    else:
        sys.modules[mod_name] = mod
    finally:
        current_source.reset(token)


def load_path(path: str | Path, *, force: bool = False) -> None:
    raw = str(Path(path).expanduser())
    p = Path(raw).resolve()
    if p.is_dir():
        for child in sorted(p.rglob("*.py")):
            load_plugin(child, force=force)
        for child in sorted(p.rglob("*.toml")):
            load_toml(child, skip_unrelated=True)
    elif p.suffix == ".toml":
        if force:
            _drop_source(str(p))
        load_toml(p)
    elif p.suffix == ".py":
        if force:
            _drop_source(str(p))
            mod_name = _plugin_mod_name(p)
            sys.modules.pop(mod_name, None)
        load_plugin(p, force=True if force else False)
    elif p.exists():
        raise LoadError(f"Don't know how to load {p} (want .toml, .py, or a directory)", path=p)
    elif _looks_like_load_path(raw) or p.suffix:
        raise FileNotFoundError(f"No such file or directory to load: {p}")
    else:
        try:
            importlib.import_module(raw)
        except ModuleNotFoundError:
            raise FileNotFoundError(f"No such file or directory to load: {raw}") from None


def _load_entry_points() -> None:
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return
    try:
        eps = entry_points(group="fractalgen.plugins")
    except TypeError:
        eps = entry_points().get("fractalgen.plugins", [])
    for ep in eps:
        src = f"entry-point:{ep.name}"
        try:
            token = current_source.set(src)
            try:
                obj = ep.load()
                if callable(obj):
                    obj()
            finally:
                current_source.reset(token)
            _note_source(src)
            if ep.name not in _entry_points_loaded:
                _entry_points_loaded.append(ep.name)
        except Exception as exc:
            if os.environ.get("FRACTALGEN_STRICT") == "1":
                raise
            print(f"fractalgen: plugin entry-point {ep.name!r} failed: {exc}", file=sys.stderr)


def _register_core_sections() -> None:
    from . import engine
    from .palette import register_palette
    from .recipe import add_recipe_table

    SECTION_HANDLERS.setdefault("palette", lambda n, t, s, r: register_palette(n, t, replace=r))
    SECTION_HANDLERS.setdefault("recipe", lambda n, t, s, r: add_recipe_table(n, t, s, replace=r))
    SECTION_HANDLERS.setdefault("preset", lambda n, t, s, r: engine.register_preset(n, t, replace=r))


def reload_user_paths() -> dict[str, Any]:
    """Drop and re-load user plugins, TOML, and entry points. Returns a summary dict."""
    from .recipe import finalize_recipes

    summary: dict[str, Any] = {"dropped": {}, "reloaded": [], "errors": []}
    for ep_name in list(_entry_points_loaded):
        src = f"entry-point:{ep_name}"
        summary["dropped"][src] = _drop_source(src)
    for file_path in list(_reload_files):
        summary["dropped"][file_path] = _drop_source(file_path)
        mod = Path(file_path)
        if mod.suffix == ".py":
            sys.modules.pop(_plugin_mod_name(mod), None)
    try:
        from importlib.metadata import entry_points as eps_fn
    except ImportError:
        eps_fn = None
    if eps_fn is not None:
        try:
            eps = eps_fn(group="fractalgen.plugins")
        except TypeError:
            eps = eps_fn().get("fractalgen.plugins", [])
        ep_by_name = {ep.name: ep for ep in eps}
        for ep_name in _entry_points_loaded:
            ep = ep_by_name.get(ep_name)
            if ep is None:
                continue
            src = f"entry-point:{ep_name}"
            try:
                token = current_source.set(src)
                try:
                    obj = ep.load()
                    if callable(obj):
                        obj()
                finally:
                    current_source.reset(token)
                summary["reloaded"].append(src)
            except Exception as exc:
                summary["errors"].append({"source": src, "error": str(exc)})
    for file_path in list(_reload_files):
        try:
            p = Path(file_path)
            if p.suffix == ".py":
                load_plugin(p, force=True)
            elif p.suffix == ".toml":
                load_toml(p)
            summary["reloaded"].append(file_path)
        except Exception as exc:
            summary["errors"].append({"source": file_path, "error": str(exc)})
    finalize_recipes()
    return summary


def bootstrap(extra: Iterable[str | Path] = ()) -> list[str]:
    """Idempotently load everything. Returns the extra paths in load order."""
    global _booted
    from .recipe import finalize_recipes

    if not _booted:
        _register_core_sections()
        importlib.import_module("fractalgen.generators")
        _note_source("builtin:generators")
        load_toml(DATA_DIR / "palettes.toml")
        load_toml(DATA_DIR / "presets.toml")
        for f in sorted((DATA_DIR / "recipes").glob("*.toml")):
            load_toml(f)
        _load_entry_points()
        env = os.environ.get("FRACTALGEN_PATH", "")
        for entry in filter(None, env.split(os.pathsep)):
            load_path(entry)
        if CUSTOM_DIR.is_dir():
            load_path(CUSTOM_DIR)
        _booted = True
    for p in extra:
        key = str(p)
        if key not in _loaded_extra:
            load_path(p)
            _loaded_extra.append(key)
    finalize_recipes()
    # Extra paths are replayed in worker processes; forget files deleted since they were loaded.
    _loaded_extra[:] = [k for k in _loaded_extra
                        if not _looks_like_load_path(k) or Path(k).expanduser().exists()]
    return list(_loaded_extra)
