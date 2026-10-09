"""Typed keys for the tables that finish a render: layers, effects, background.

Each key knows its doc, default ("neutral"), kind and UI hints, so the CLI
docs, recipe validation and the GUI controls all come from one definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

KINDS = ("float", "int", "bool", "string", "choice", "color", "palette", "font", "path", "fill")


@dataclass(frozen=True)
class Key:
    doc: str
    neutral: Any  # the default; the value at which an effect does nothing
    kind: str = "float"  # one of KINDS
    range: tuple | None = None  # slider hint (min, max[, step])
    choices: Any = None  # tuple, or a zero-arg callable returning names (a registry's live list)
    label: str = ""  # GUI label; derived from the key name when empty
    unit: str = ""  # "px" = pixels at 4K (scaled with the frame), "deg", "%"
    advanced: bool = False  # GUI shows it under "More"

    def choice_list(self) -> list[str] | None:
        c = self.choices() if callable(self.choices) else self.choices
        return list(c) if c is not None else None

    def coerce(self, where: str, value: Any) -> Any:
        try:
            if self.kind in ("float", "int"):
                if isinstance(value, bool):
                    raise ValueError("must be a number")
                v = float(value)
                if not math.isfinite(v):
                    raise ValueError("must be finite")
                return int(round(v)) if self.kind == "int" else v
            if self.kind == "bool":
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                return bool(value)
            if self.kind == "color" and isinstance(value, (list, tuple)):
                return [float(x) for x in value[:3]]
            if self.kind == "palette" and not isinstance(value, str):
                return value  # inline stops / {stops, positions}
            v = str(value)
            choices = self.choice_list()
            if self.kind == "choice" and choices is not None and v not in choices:
                raise ValueError(f"must be one of {[c for c in choices if c]}")
            return v
        except (TypeError, ValueError) as e:
            raise ValueError(f"{where}={value!r}: {e}") from None

    def schema(self, name: str) -> dict[str, Any]:
        """JSON description for the GUI."""
        from .labels import label_for

        return {
            "name": name,
            "label": self.label or label_for(name),
            "doc": self.doc,
            "kind": self.kind,
            "neutral": _jsonable(self.neutral),
            "range": list(self.range) if self.range else None,
            "choices": self.choice_list(),
            "unit": self.unit,
            "advanced": self.advanced,
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    return v


def check_table(keys: dict[str, Key], table: dict[str, Any] | None, where: str, *, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    """Validate and coerce ``table`` against ``keys``; ``None`` values drop a key."""
    out: dict[str, Any] = {}
    for k, v in (table or {}).items():
        if k in extra:
            out[k] = v
            continue
        if k not in keys:
            import difflib

            close = difflib.get_close_matches(k, list(keys), n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            raise KeyError(f"Unknown {where} key {k!r}{hint}. Known: {sorted(keys)}")
        if v is not None:
            out[k] = keys[k].coerce(f"{where}.{k}", v)
    return out
