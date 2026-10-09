"""Live-tweaking GUI: a local web app served with only the standard library.

    python -m fractalgen gui            # opens http://127.0.0.1:8765

The page (static/index.html) builds its controls from ``/api/catalog``, which
is generated from the generator / palette / post-op registries, so plugins and
TOML formulas show up automatically.

API (JSON unless noted)
-----------------------
GET  /api/catalog                     generators (+ param schema), recipes, palettes, post / tone ops, presets,
                                      and per-formula default `coeffs` for map/flow attractors
GET  /api/state?recipe=NAME&w=&h=     a recipe resolved to an editable *state*
POST /api/render   {state}            -> image/png (204 if the same client asked for a newer seq meanwhile)
                                      headers: X-Render-Ms, X-View-Extent "half_w,half_h" (view generators;
                                      for the frame size `extent_size` [w, h] if given)
POST /api/save     {name, state, overwrite}  -> writes custom/NAME.toml and registers it
POST /api/export   {state, width, height, format, name} -> {job}; full-size render in the background
GET  /api/export/JOB                  -> {status, path, error, seconds, progress, stage}
GET  /api/progress?client=ID          -> {progress, stage} of that client's running preview render
                                      (progress null when none is running)

A *state* is::

    {"recipe": "julia_dragon" | null, "generator": "escape_time",
     "palette": "ember" | ["000000", ...] | null,
     "params": {...}, "view": {"center": [re, im], "half_width": h, "rotation": deg, "fit": "fill"} | null,
     "post": {...}, "tone": {...}, "mask": {...}, "width": 640, "height": 360, "seq": 12, "client": "a1b2"}
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shlex
import string
import threading
import time
import traceback
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import progress
from ..color import save_image, to_image
from ..engine import OUT_ROOT
from ..generators.base import GENERATORS, Param
from ..loader import CUSTOM_DIR, bootstrap
from ..palette import PALETTES
from ..mask import MASK_KEYS, check_mask
from ..post import POST_OPS
from ..tone import TONE_KEYS
from ..recipe import RECIPES, Recipe

STATIC = Path(__file__).resolve().parent / "static"
_COLOR_WORDS = ("color", "colour", "tint", "bg", "background", "interior", "glow", "metal", "teal", "plum", "violet")
_MAX_BODY_BYTES = 10 * 1024 * 1024
_PREVIEW_MAX_DIM = 4096
_EXPORT_MAX_DIM = 16384
_SAVE_NAME_CHARS = frozenset(string.ascii_letters + string.digits + "_-")


# ---------------------------------------------------------------- schema ----

def jsonable(v: Any) -> Any:
    if isinstance(v, complex):
        return [v.real, v.imag]
    if isinstance(v, (tuple, list)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "hex_stops"):
        return v.hex_stops()
    return v


def param_kind(p: Param) -> str:
    d = p.default
    name = p.name.lower()
    if name == "palette" or name.endswith("_palette"):
        return "palette"
    if p.choices is not None:
        return "choice"
    if isinstance(d, bool):
        return "bool"
    if isinstance(d, int):
        return "int"
    if isinstance(d, float):
        return "float"
    if isinstance(d, complex):
        return "complex"
    if isinstance(d, str):
        return "string"
    if isinstance(d, dict) and d and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in d.values()):
        return "numdict"
    if isinstance(d, (tuple, list)) and d and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in d):
        if len(d) == 3 and all(0.0 <= float(x) <= 1.0 for x in d) and any(w in name for w in _COLOR_WORDS):
            return "color"
        if len(d) <= 6:
            return "vector"
    return "json"


def default_range(kind: str, d: Any) -> list | None:
    """Heuristic slider bounds when a generator gives no hint."""
    if kind == "int":
        hi = max(10, abs(int(d)) * 4)
        return [0 if d >= 0 else -hi, hi, 1]
    if kind == "float":
        d = float(d)
        if d == 0.0:
            return [-1.0, 1.0]
        if 0.0 < d <= 1.0:
            return [0.0, 1.0 if d <= 0.5 else 2.0]
        mag = abs(d) * 3.0
        return [0.0, mag] if d > 0 else [-mag, mag]
    return None


def param_schema(p: Param) -> dict[str, Any]:
    kind = param_kind(p)
    default = None if p.required else jsonable(p.default)
    rng = list(p.range) if p.range else default_range(kind, p.default)
    return {
        "name": p.name,
        "kind": kind,
        "default": default,
        "doc": p.doc,
        "range": rng,
        "choices": p.choice_list(),
    }


def catalog() -> dict[str, Any]:
    from ..engine import PRESETS

    gens = {}
    for name in GENERATORS:
        g = GENERATORS.get(name)
        gens[name] = {
            "category": g.category,
            "description": g.description,
            "uses_view": g.uses_view,
            "default_view": jsonable(g.default_view),
            "params": [param_schema(p) for p in g.params.values()],
        }
    from ..generators.fields import VECTOR_FIELDS
    from ..generators.flows import FLOWS
    from ..generators.maps import MAPS

    # Generators whose `coeffs` dict is keyed by a formula chosen in another param.
    coeff_sets = {
        "map_attractor": {"via": "map", "defaults": {n: dict(MAPS.get(n).defaults) for n in MAPS}},
        "flow_attractor": {"via": "flow", "defaults": {n: dict(FLOWS.get(n).defaults) for n in FLOWS}},
        "flow_field": {"via": "field", "defaults": {n: dict(VECTOR_FIELDS.get(n).defaults) for n in VECTOR_FIELDS}},
    }
    return {
        "generators": gens,
        "coefficients": coeff_sets,
        "recipes": [
            {"name": n, "generator": r.generator, "description": r.description, "tags": list(r.tags)}
            for n, r in RECIPES.items()
        ],
        "palettes": [
            {"name": n, "stops": PALETTES.get(n).hex_stops(), "description": PALETTES.get(n).description}
            for n in PALETTES
        ],
        "post_ops": [
            {"name": n, "doc": op.doc, "kind": op.kind or "float", "neutral": jsonable(op.neutral),
             "range": list(op.range) if op.range else None}
            for n, op in POST_OPS.items()
        ],
        "tone_ops": [
            {"name": n, "doc": k.doc, "kind": k.kind, "neutral": k.neutral, "range": list(k.range) if k.range else None}
            for n, k in TONE_KEYS.items()
        ],
        "mask_ops": [
            {"name": n, "doc": k.doc, "kind": k.kind, "neutral": k.neutral, "range": list(k.range) if k.range else None,
             "choices": list(k.choices) if k.choices else None}
            for n, k in MASK_KEYS.items()
        ],
        "presets": [
            {"name": n, "width": p.width, "height": p.height, "format": p.format}
            for n, p in PRESETS.items() if n == p.name
        ],
    }


def recipe_state(name: str, w: int, h: int) -> dict[str, Any]:
    r = RECIPES.get(name)
    spec, params, view = r.resolve(w, h)
    palette = params.pop("palette", None)
    return {
        "recipe": name,
        "generator": spec.name,
        "palette": jsonable(palette),
        "params": jsonable(params),
        "view": {"center": jsonable(view.center), "half_width": view.half_width,
                 "rotation": view.rotation, "fit": view.fit} if spec.uses_view else None,
        "post": jsonable(r.resolve_post(w, h)),
        "tone": jsonable(r.resolve_tone(w, h)),
        "mask": jsonable(r.resolve_mask(w, h)),
        "width": w,
        "height": h,
    }


def state_edits(state: dict[str, Any], base: Recipe) -> dict[str, Any]:
    """What the GUI changed relative to ``base`` resolved at the state's own frame size."""
    ref = recipe_state(base.name, int(state.get("width", 640)), int(state.get("height", 360)))
    edits: dict[str, Any] = {"params": {}, "view": {}, "post": {}, "tone": {}, "mask": {}}
    for k, v in (state.get("params") or {}).items():
        if v != ref["params"].get(k):
            edits["params"][k] = v
    for k, v in (state.get("view") or {}).items():
        if ref["view"] is not None and v != ref["view"].get(k):
            edits["view"][k] = v
    post, ref_post = state.get("post") or {}, ref["post"]
    for k in set(post) | set(ref_post):
        if post.get(k) != ref_post.get(k):
            edits["post"][k] = post.get(k)  # None switches a recipe's op off
    tone, ref_tone = state.get("tone") or {}, ref["tone"]
    for k in set(tone) | set(ref_tone):
        if tone.get(k) != ref_tone.get(k):
            edits["tone"][k] = tone.get(k)
    mask, ref_mask = state.get("mask") or {}, ref["mask"]
    for k in set(mask) | set(ref_mask):
        if mask.get(k) != ref_mask.get(k):
            edits["mask"][k] = mask.get(k)  # None drops a recipe's key back to its default
    if state.get("palette") != ref["palette"]:
        edits["palette"] = state.get("palette")
    return edits


def state_recipe(state: dict[str, Any], name: str = "gui") -> Recipe:
    """Turn a GUI state into a Recipe.

    A state is a recipe resolved for one frame size. When it came from a recipe
    with the same generator, only the GUI's edits are layered over that recipe
    (in its portrait overrides too), so the result still frames correctly in the
    other orientation and portrait overrides never undo an edit.
    """
    gen = GENERATORS.get(state["generator"])
    base = RECIPES.get(state["recipe"]) if state.get("recipe") in RECIPES else None
    if base is not None and base.generator == gen.name:
        w, h = int(state.get("width", 640)), int(state.get("height", 360))
        e = state_edits(state, base)
        e["params"] = {k: v for k, v in e["params"].items() if k in gen.params and k != "palette"}
        portrait = {k: dict(v) if isinstance(v, dict) else v for k, v in base.portrait.items()}
        for key in ("params", "post", "tone", "mask"):
            if e[key]:
                portrait[key] = {**(portrait.get(key) or {}), **e[key]}
        # View edits are absolute in the orientation they were made in; the other
        # orientation gets the same pan / zoom / turn relative to its own framing.
        view_edits, other_edits = e["view"], {}
        if view_edits and gen.uses_view:
            other_edits = _carry_view_edits(view_edits, recipe_state(base.name, w, h)["view"],
                                            recipe_state(base.name, h, w)["view"])
        land_edits, port_edits = (other_edits, view_edits) if h > w else (view_edits, other_edits)
        if port_edits:
            portrait["view"] = {**(portrait.get("view") or {}), **port_edits}
        palette = base.palette
        if "palette" in e:
            palette = portrait["palette"] = e["palette"]
        return Recipe(
            name=name,
            generator=gen.name,
            params={**base.params, **e["params"]},
            view={**base.view, **land_edits} if gen.uses_view else {},
            portrait=portrait,
            palette=palette if "palette" in gen.params else None,
            post={k: v for k, v in {**base.post, **e["post"]}.items() if v is not None},
            tone={k: v for k, v in {**base.tone, **e["tone"]}.items() if v is not None},
            mask={k: v for k, v in {**base.mask, **e["mask"]}.items() if v is not None},
            description=base.description,
            # Pans and zooms keep the recipe's own exposure (see tone.py): tone is
            # measured on that recipe's anchor view, in whichever orientation renders.
            tone_from=(base.tone_from or base.name) if gen.uses_view else "",
        )
    view = dict(state.get("view") or {}) if gen.uses_view else {}
    return Recipe(
        name=name,
        generator=gen.name,
        params={k: v for k, v in (state.get("params") or {}).items() if k in gen.params and k != "palette"},
        view=view,
        portrait={"view": dict(view)} if view else {},  # keep the chosen fit on phones too
        palette=state.get("palette") if "palette" in gen.params else None,
        post={k: v for k, v in (state.get("post") or {}).items() if v is not None},
        tone={k: v for k, v in (state.get("tone") or {}).items() if v is not None},
        mask=check_mask(state.get("mask")),
        tone_view={} if gen.uses_view else None,  # expose as at the generator's default view
    )


def _carry_view_edits(edits: dict[str, Any], ref: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """``edits`` made against view ``ref``, applied to ``other`` as a pan, zoom and turn."""
    out: dict[str, Any] = {}
    if "center" in edits:
        (nx, ny), (rx, ry), (ox, oy) = edits["center"], ref["center"], other["center"]
        out["center"] = [ox + nx - rx, oy + ny - ry]
    if "half_width" in edits:
        out["half_width"] = other["half_width"] * float(edits["half_width"]) / float(ref["half_width"])
    if "rotation" in edits:
        out["rotation"] = float(other["rotation"]) + float(edits["rotation"]) - float(ref["rotation"])
    if "fit" in edits:
        out["fit"] = edits["fit"]
    return out


# ---------------------------------------------------------------- CLI string ---

def _shell_quote(s: str) -> str:
    return shlex.quote(s)


def _fmt_literal(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, complex):
        im = v.imag
        if im == 0:
            return str(v.real)
        sign = "+" if im >= 0 else ""
        return f"{v.real}{sign}{im}j"
    if isinstance(v, (list, tuple)):
        if len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            return _fmt_literal(complex(v[0], v[1]))
        return json.dumps(jsonable(v), separators=(",", ":"))
    if isinstance(v, dict):
        return json.dumps(jsonable(v), separators=(",", ":"))
    return str(v)


def _fmt_palette_cli(p: Any) -> str:
    if isinstance(p, str):
        return p
    if isinstance(p, list):
        return ",".join(str(x).replace("#", "") for x in p)
    if isinstance(p, dict) and "stops" in p:
        return ",".join(str(x).replace("#", "") for x in p["stops"])
    return _fmt_literal(p)


def _diff_maps(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in set(a) | set(b):
        if a.get(k) != b.get(k):
            out[k] = a.get(k)
    return out


def _finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _parse_dims(w: Any, h: Any, *, max_dim: int, label: str = "size") -> tuple[int, int]:
    try:
        wi, hi = int(w), int(h)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} width and height must be integers") from exc
    if wi < 1 or hi < 1:
        raise ValueError(f"{label} width and height must be >= 1, got {wi}x{hi}")
    if wi > max_dim or hi > max_dim:
        raise ValueError(f"{label} too large (max {max_dim}), got {wi}x{hi}")
    return wi, hi


def _validate_gui_view(view: dict[str, Any] | None) -> None:
    if view is None:
        return
    if not isinstance(view, dict):
        raise ValueError("view must be an object")
    if "center" in view:
        c = view["center"]
        if not isinstance(c, (list, tuple)) or len(c) < 2:
            raise ValueError("view.center must be [re, im]")
        for i, x in enumerate(c[:2]):
            if not _finite_number(x):
                raise ValueError(f"view.center[{i}] must be a finite number")
    if "half_width" in view:
        hw = view["half_width"]
        if not _finite_number(hw) or float(hw) <= 0:
            raise ValueError("view.half_width must be a finite positive number")
    if "rotation" in view:
        rot = view["rotation"]
        if not _finite_number(rot):
            raise ValueError("view.rotation must be a finite number")


def _validate_render_state(state: dict[str, Any]) -> None:
    _validate_gui_view(state.get("view"))
    _parse_dims(state.get("width", 640), state.get("height", 360), max_dim=_PREVIEW_MAX_DIM, label="preview")


def _validate_export_body(body: dict[str, Any]) -> tuple[dict[str, Any], int, int, str]:
    if "state" not in body or not isinstance(body["state"], dict):
        raise ValueError("export requires a state object")
    state = body["state"]
    _validate_gui_view(state.get("view"))
    w, h = _parse_dims(body.get("width"), body.get("height"), max_dim=_EXPORT_MAX_DIM, label="export")
    fmt_raw = str(body.get("format", "png")).lower()
    if fmt_raw not in ("png", "jpg", "jpeg"):
        raise ValueError("export format must be png, jpg, or jpeg")
    fmt = "jpg" if fmt_raw in ("jpg", "jpeg") else "png"
    return state, w, h, fmt


def cli_command(state: dict[str, Any], width: int, height: int) -> str:
    """POSIX-shell command for the current GUI state (only overrides vs the recipe)."""
    w, h = int(width), int(height)
    size = f"{w}x{h}"
    parts = ["python", "-m", "fractalgen"]
    recipe_name = state.get("recipe")
    gen = state.get("generator")
    if recipe_name and recipe_name in RECIPES:
        ref = recipe_state(recipe_name, w, h)
        parts.extend(["render", _shell_quote(str(recipe_name)), "--size", _shell_quote(size)])
    else:
        if not gen:
            raise ValueError("missing generator")
        parts.extend(["try", _shell_quote(str(gen)), "--size", _shell_quote(size)])
        ref = {
            "recipe": None,
            "generator": gen,
            "palette": None,
            "params": {},
            "view": None,
            "post": {},
            "tone": {},
            "mask": {},
        }
        g = GENERATORS.get(gen)
        if g:
            for p in g.params.values():
                if p.name == "palette":
                    ref["palette"] = jsonable(p.default)
                elif not p.required:
                    ref["params"][p.name] = jsonable(p.default)
            if g.uses_view:
                ref["view"] = {
                    "center": jsonable(g.default_view.get("center", [0, 0]) if g.default_view else [0, 0]),
                    "half_width": (g.default_view or {}).get("half_width", 2),
                    "rotation": (g.default_view or {}).get("rotation", 0),
                    "fit": (g.default_view or {}).get("fit", "fill"),
                }

    if state.get("palette") != ref.get("palette") and state.get("palette") is not None:
        parts.extend(["--palette", _shell_quote(_fmt_palette_cli(state["palette"]))])
    pdiff = _diff_maps(state.get("params") or {}, ref.get("params") or {})
    for k in sorted(pdiff):
        parts.extend(["--set", _shell_quote(f"{k}={_fmt_literal(pdiff[k])}")])
    v, rv = state.get("view"), ref.get("view")
    if v and rv:
        if v.get("center") != rv.get("center") and v.get("center") is not None:
            parts.extend(["--center", _shell_quote(_fmt_literal(v["center"]))])
        if v.get("half_width") != rv.get("half_width") and v.get("half_width") is not None:
            parts.extend(["--half-width", _shell_quote(str(v["half_width"]))])
        if v.get("rotation") != rv.get("rotation") and v.get("rotation") is not None:
            parts.extend(["--rotate", _shell_quote(str(v["rotation"]))])
        if v.get("fit") != rv.get("fit") and v.get("fit") is not None:
            parts.extend(["--fit", _shell_quote(str(v["fit"]))])
    elif v and not rv:
        if v.get("center") is not None:
            parts.extend(["--center", _shell_quote(_fmt_literal(v["center"]))])
        if v.get("half_width") is not None:
            parts.extend(["--half-width", _shell_quote(str(v["half_width"]))])
        if v.get("rotation") is not None:
            parts.extend(["--rotate", _shell_quote(str(v["rotation"]))])
        if v.get("fit") is not None:
            parts.extend(["--fit", _shell_quote(str(v["fit"]))])
    for k, val in sorted(_diff_maps(state.get("post") or {}, ref.get("post") or {}).items()):
        if val is not None:
            parts.extend(["--post", _shell_quote(f"{k}={_fmt_literal(val)}")])
    for k, val in sorted(_diff_maps(state.get("tone") or {}, ref.get("tone") or {}).items()):
        if val is not None:
            parts.extend(["--tone", _shell_quote(f"{k}={_fmt_literal(val)}")])
    for k, val in sorted(_diff_maps(state.get("mask") or {}, ref.get("mask") or {}).items()):
        if val is None:  # back to the default
            val = MASK_KEYS[k].neutral if k in MASK_KEYS else None
        if val is not None:
            parts.extend(["--mask", _shell_quote(f"{k}={val if isinstance(val, str) else _fmt_literal(val)}")])
    return " ".join(parts)


# ---------------------------------------------------------------- thumbs ----

def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "fractalgen" / "thumbs"


def _thumb_dims(orient: str) -> tuple[int, int]:
    return (192, 108) if orient == "land" else (108, 192)


def thumb_fingerprint(recipe_name: str, w: int, h: int) -> str:
    st = recipe_state(recipe_name, w, h)
    payload = {
        "generator": st["generator"],
        "params": st["params"],
        "palette": st["palette"],
        "view": st["view"],
        "post": st["post"],
        "tone": st["tone"],
        "mask": st["mask"],
        "w": w,
        "h": h,
    }
    blob = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _thumb_path(recipe_name: str, orient: str) -> Path:
    w, h = _thumb_dims(orient)
    key = thumb_fingerprint(recipe_name, w, h)
    return _cache_root() / f"{key}_{w}x{h}.png"


class ThumbQueue:
    """Single background thread; yields to preview/export renders."""

    def __init__(self, renderer: Renderer, exports: Exports) -> None:
        from ..kernels import _PARALLEL

        self._renderer = renderer
        self._exports = exports
        self._parallel = _PARALLEL
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, str], int] = {}  # (recipe, orient) -> priority
        self._cv = threading.Condition(self._lock)
        self._thread = threading.Thread(target=self._run, name="fractalgen-thumbs", daemon=True)
        self._thread.start()

    def enqueue(self, recipe: str, orient: str, priority: int = 0) -> None:
        with self._cv:
            key = (recipe, orient)
            if key not in self._pending or priority < self._pending[key]:
                self._pending[key] = priority
            self._cv.notify()

    def _render_busy(self) -> bool:
        if self._renderer.lock.locked():
            return True
        if self._exports.any_running():
            return True
        return self._parallel.locked()

    def _pop(self) -> tuple[str, str] | None:
        with self._cv:
            while not self._pending:
                self._cv.wait(timeout=0.5)
                if not self._pending:
                    return None
            key = min(self._pending, key=lambda k: self._pending[k])
            del self._pending[key]
            return key

    def _run(self) -> None:
        while True:
            item = self._pop()
            if item is None:
                continue
            recipe, orient = item
            path = _thumb_path(recipe, orient)
            if path.is_file():
                continue
            while self._render_busy():
                time.sleep(0.15)
            w, h = _thumb_dims(orient)
            try:
                while self._render_busy():
                    time.sleep(0.15)
                st = recipe_state(recipe, w, h)
                rgb = state_recipe(st, recipe).render(w, h)
                path.parent.mkdir(parents=True, exist_ok=True)
                buf = io.BytesIO()
                to_image(rgb).save(buf, format="PNG", compress_level=3)
                path.write_bytes(buf.getvalue())
            except Exception:
                traceback.print_exc()

    def warmup(self, names: list[str], delay_s: float = 3.0) -> None:
        def _go() -> None:
            time.sleep(delay_s)
            for i, name in enumerate(names[:20]):
                g = RECIPES.get(name)
                cat = GENERATORS.get(g.generator).category if g else ""
                pri = 10 if cat == "simulation" else 0
                self.enqueue(name, "land", pri + i)
                self.enqueue(name, "port", pri + i)

        threading.Thread(target=_go, daemon=True).start()


# ---------------------------------------------------------------- server ----

class Renderer:
    """Serializes renders; drops requests superseded by a newer ``seq`` from the same client (page)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: dict[str, int] = {}
        self.seq_lock = threading.Lock()
        self.active: dict[str, progress.Tracker] = {}  # client -> its render in flight
        self._waiting: dict[str, int] = {}  # client -> queued preview count

    def progress(self, client: str) -> dict[str, Any]:
        tracker = self.active.get(client)
        if tracker is None:
            out: dict[str, Any] = {"progress": None, "stage": ""}
            if self._waiting.get(client, 0) > 0:
                out["waiting"] = True
            return out
        return {"progress": tracker.fraction, "stage": tracker.stage}

    def _inc_waiting(self, client: str) -> None:
        with self.seq_lock:
            self._waiting[client] = self._waiting.get(client, 0) + 1

    def _dec_waiting(self, client: str) -> None:
        with self.seq_lock:
            n = self._waiting.get(client, 0) - 1
            if n > 0:
                self._waiting[client] = n
            else:
                self._waiting.pop(client, None)

    def render_png(self, state: dict[str, Any]) -> tuple[bytes | None, float, str]:
        _validate_render_state(state)
        seq = int(state.get("seq", 0))
        client = str(state.get("client", ""))
        with self.seq_lock:
            self.latest[client] = max(self.latest.get(client, -1), seq)
        self._inc_waiting(client)
        with self.lock:
            self._dec_waiting(client)
            if seq < self.latest[client]:
                return None, 0.0, ""
            w, h = _parse_dims(state.get("width", 640), state.get("height", 360),
                               max_dim=_PREVIEW_MAX_DIM, label="preview")
            t0 = time.perf_counter()
            recipe = state_recipe(state)
            tracker = self.active[client] = progress.Tracker()
            try:
                with progress.tracking(tracker):
                    rgb = recipe.render(w, h)
            finally:
                self.active.pop(client, None)
            buf = io.BytesIO()
            to_image(rgb).save(buf, format="PNG", compress_level=1)
            extent = ""
            if recipe.spec.uses_view:
                _, _, view = recipe.resolve(w, h)
                hw, hh = view.extents(*(state.get("extent_size") or (w, h)))
                extent = f"{hw!r},{hh!r}"
            return buf.getvalue(), (time.perf_counter() - t0) * 1000.0, extent


class Exports:
    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}

    def any_running(self) -> bool:
        return any(j.get("status") == "running" for j in self.jobs.values())

    def start(self, body: dict[str, Any]) -> str:
        state, w, h, fmt = _validate_export_body(body)
        job = uuid.uuid4().hex[:10]
        self.jobs[job] = {
            "status": "running", "path": None, "error": None, "seconds": 0.0,
            "_t0": time.perf_counter(), "_tracker": progress.Tracker(),
        }
        threading.Thread(target=self._run, args=(job, state, w, h, fmt, body), daemon=True).start()
        return job

    def status(self, job: str) -> dict[str, Any] | None:
        """A job's public fields; ``seconds`` and ``progress`` (0..1) count up while it is still running."""
        info = self.jobs.get(job)
        if info is None:
            return None
        out = {k: v for k, v in info.items() if not k.startswith("_")}
        tracker = info["_tracker"]
        if out["status"] == "done":
            out["progress"] = 1.0
        else:
            out["progress"] = tracker.fraction
        out["stage"] = tracker.stage
        if out["status"] == "running":
            out["seconds"] = time.perf_counter() - info["_t0"]
        return out

    def _run(
        self,
        job: str,
        state: dict[str, Any],
        w: int,
        h: int,
        fmt: str,
        body: dict[str, Any],
    ) -> None:
        from ..engine import RENDER_SHARE, default_n_strips

        t0 = self.jobs[job]["_t0"]
        try:
            stem = "".join(c if c.isalnum() or c in "-_@" else "_" for c in (body.get("name") or state.get("recipe") or "gui"))
            path = OUT_ROOT / "gui" / f"{stem}_{w}x{h}.{fmt}"
            with progress.tracking(self.jobs[job]["_tracker"]):
                with progress.span(0.0, RENDER_SHARE):
                    recipe = state_recipe(state, state.get("recipe") or "gui")
                    rgb = recipe.render(w, h, n_strips=default_n_strips(w, h))
                with progress.span(RENDER_SHARE, 1.0, "saving"):
                    from ..tomlout import image_meta

                    save_image(path, rgb, fmt, meta=image_meta(recipe, w, h))
            # seconds before status: a poll that sees "done" must see the final time
            self.jobs[job].update(seconds=time.perf_counter() - t0, path=str(path))
            self.jobs[job]["status"] = "done"
        except Exception:
            self.jobs[job].update(seconds=time.perf_counter() - t0, error=traceback.format_exc())
            self.jobs[job]["status"] = "error"


def save_state(name: str, state: dict[str, Any], overwrite: bool = False) -> Path:
    from ..recipe import register_recipe
    from ..tomlout import recipe_to_toml

    if not name or not all(c in _SAVE_NAME_CHARS for c in name):
        raise ValueError("name may only use ASCII letters, digits, '_' and '-'")
    path = CUSTOM_DIR / f"{name}.toml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists")
    if name in RECIPES and not overwrite:
        raise FileExistsError(f"recipe {name!r} already exists")
    recipe = state_recipe(state, name)
    text = recipe_to_toml(recipe, name)  # landscape base; portrait overrides are kept separately
    if state.get("recipe"):
        text = text.replace(f"# Forked from `{name}`", f"# Made in the GUI from `{state['recipe']}`", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    recipe.source = str(path)
    register_recipe(recipe, replace=True)
    return path


def make_handler(renderer: Renderer, exports: Exports, thumbs: ThumbQueue | None = None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "fractalgen-gui"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _error(self, exc: BaseException, code: int = 400) -> None:
            self._json({"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}, code)

        def _body(self) -> dict[str, Any]:
            raw = self.headers.get("Content-Length")
            if raw is None:
                n = 0
            else:
                raw = raw.strip()
                if not raw.isdigit():
                    raise ValueError(f"invalid Content-Length: {raw!r}")
                n = int(raw)
            if n < 0:
                raise ValueError(f"invalid Content-Length: {n}")
            if n > _MAX_BODY_BYTES:
                raise ValueError(f"request body too large ({n} bytes; max {_MAX_BODY_BYTES})")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/catalog":
                    self._json(catalog())
                elif url.path == "/api/state":
                    self._json(recipe_state(q["recipe"], int(q.get("w", 640)), int(q.get("h", 360))))
                elif url.path == "/api/progress":
                    self._json(renderer.progress(q.get("client", "")))
                elif url.path == "/api/thumb":
                    recipe = q.get("recipe", "")
                    orient = q.get("orient", "land")
                    if orient not in ("land", "port") or recipe not in RECIPES:
                        self._send(404, b"not found", "text/plain")
                    else:
                        path = _thumb_path(recipe, orient)
                        if path.is_file():
                            data = path.read_bytes()
                            self._send(200, data, "image/png", {"Cache-Control": "public, max-age=86400"})
                        elif thumbs is not None:
                            g = RECIPES.get(recipe)
                            cat = GENERATORS.get(g.generator).category if g else ""
                            pri = 10 if cat == "simulation" else 0
                            thumbs.enqueue(recipe, orient, pri)
                            self._send(202, b"", "text/plain")
                        else:
                            self._send(404, b"not found", "text/plain")
                elif url.path.startswith("/api/export/"):
                    rest = url.path[len("/api/export/"):]
                    if rest.endswith("/file"):
                        job_id = rest[:-5].rstrip("/")
                        info = exports.status(job_id)
                        if not info or info.get("status") != "done" or not info.get("path"):
                            self._send(404, b"not found", "text/plain")
                        else:
                            fp = Path(info["path"])
                            if not fp.is_file():
                                self._send(404, b"not found", "text/plain")
                            else:
                                data = fp.read_bytes()
                                name = fp.name
                                self._send(
                                    200,
                                    data,
                                    "application/octet-stream",
                                    {"Content-Disposition": f'attachment; filename="{name}"'},
                                )
                    else:
                        job = exports.status(rest)
                        self._json(job or {"status": "unknown"}, 200 if job else 404)
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as exc:
                self._error(exc)

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            try:
                body = self._body()
                if url.path == "/api/render":
                    png, ms, extent = renderer.render_png(body)
                    if png is None:  # superseded: no content, and no console error in the browser
                        self._send(204, b"", "text/plain", {"X-Stale": "1"})
                    else:
                        hdr = {"X-Render-Ms": f"{ms:.0f}"}
                        if extent:  # frame half-extents in plane units (before rotation)
                            hdr["X-View-Extent"] = extent
                        self._send(200, png, "image/png", hdr)
                elif url.path == "/api/save":
                    path = save_state(body["name"], body["state"], body.get("overwrite") is True)
                    self._json({"path": str(path), "name": body["name"]})
                elif url.path == "/api/export":
                    self._json({"job": exports.start(body)})
                elif url.path == "/api/cli":
                    st = body.get("state") or body
                    w = int(body.get("width", st.get("width", 640)))
                    h = int(body.get("height", st.get("height", 360)))
                    self._json({"command": cli_command(st, w, h)})
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as exc:
                self._error(exc)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, loads: list[str] | None = None) -> None:
    bootstrap(loads or [])
    renderer = Renderer()
    exports = Exports()
    thumbs = ThumbQueue(renderer, exports)
    thumbs.warmup([r["name"] for r in catalog()["recipes"]])
    httpd: ThreadingHTTPServer | None = None
    bound = port
    for attempt in range(11):
        try:
            httpd = ThreadingHTTPServer((host, bound), make_handler(renderer, exports, thumbs))
            break
        except OSError:
            if attempt == 10:
                raise
            bound = port + attempt + 1
    url = f"http://{host}:{bound}/"
    if bound != port:
        print(f"Port {port} busy; using {bound} instead.", flush=True)
    print(f"fractalgen GUI on {url}", flush=True)
    print("Default look: julia_dragon — drag to pan, scroll to zoom, Shift+scroll or Alt+drag to rotate.", flush=True)
    print("First preview compiles kernels (a few seconds). Ctrl-C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
