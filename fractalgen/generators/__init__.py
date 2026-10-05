"""Built-in generators. Every module in this package is imported on load,
so dropping a new ``*.py`` file here is enough to register its generators."""

import importlib
import pkgutil

for _mod in pkgutil.iter_modules(__path__):
    if not _mod.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_mod.name}")
