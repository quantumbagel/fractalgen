"""Live-tweaking GUI: a local web app served with only the standard library.

    python -m fractalgen gui            # opens http://127.0.0.1:8765

The page (static/index.html) builds its controls from ``/api/catalog``, which
is generated from the generator / palette / post-op registries, so plugins and
TOML formulas show up automatically.

API (JSON unless noted)
-----------------------
GET  /api/catalog                     generators (+ param schema), recipes, palettes, post ops, presets,
                                      and per-formula default `coeffs` for map/flow attractors
GET  /api/state?recipe=NAME&w=&h=     a recipe resolved to an editable *state*
POST /api/render   {state}            -> image/png (204 if the same client asked for a newer seq meanwhile)
                                      headers: X-Render-Ms, X-View-Extent "half_w,half_h" (view generators;
                                      for the frame size `extent_size` [w, h] if given)
POST /api/save     {name, state, overwrite}  -> writes custom/NAME.toml and registers it
POST /api/export   {state, width, height, format, name} -> {job}; full-size render in the background
GET  /api/export/JOB                  -> {status, path, error, seconds}

A *state* is::

    {"recipe": "julia_dragon" | null, "generator": "escape_time",
     "palette": "ember" | ["000000", ...] | null,
     "params": {...}, "view": {"center": [re, im], "half_width": h, "rotation": deg, "fit": "fill"} | null,
     "post": {...}, "width": 640, "height": 360, "seq": 12, "client": "a1b2"}
"""

from __future__ import annotations

import io
import json
import math
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

from ..color import save_image, to_image
from ..engine import OUT_ROOT
from ..generators.base import GENERATORS, Param
from ..loader import CUSTOM_DIR, bootstrap
from ..palette import PALETTES
from ..post import POST_OPS
from ..recipe import RECIPES, Recipe

STATIC = Path(__file__).resolve().parent / "static"
_COLOR_WORDS = ("color", "colour", "tint", "bg", "background", "interior", "glow", "metal", "teal", "plum", "violet")


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
        "presets": [
            {"name": n, "width": p.width, "height": p.height}
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
        "width": w,
        "height": h,
    }


def state_edits(state: dict[str, Any], base: Recipe) -> dict[str, Any]:
    """What the GUI changed relative to ``base`` resolved at the state's own frame size."""
    ref = recipe_state(base.name, int(state.get("width", 640)), int(state.get("height", 360)))
    edits: dict[str, Any] = {"params": {}, "view": {}, "post": {}}
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
        e = state_edits(state, base)
        e["params"] = {k: v for k, v in e["params"].items() if k in gen.params and k != "palette"}
        portrait = {k: dict(v) if isinstance(v, dict) else v for k, v in base.portrait.items()}
        for key in ("params", "view", "post"):
            if e[key]:
                portrait[key] = {**(portrait.get(key) or {}), **e[key]}
        palette = base.palette
        if "palette" in e:
            palette = portrait["palette"] = e["palette"]
        return Recipe(
            name=name,
            generator=gen.name,
            params={**base.params, **e["params"]},
            view={**base.view, **e["view"]} if gen.uses_view else {},
            portrait=portrait,
            palette=palette if "palette" in gen.params else None,
            post={k: v for k, v in {**base.post, **e["post"]}.items() if v is not None},
            description=base.description,
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
    )


# ---------------------------------------------------------------- server ----

class Renderer:
    """Serializes renders; drops requests superseded by a newer ``seq`` from the same client (page)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: dict[str, int] = {}
        self.seq_lock = threading.Lock()

    def render_png(self, state: dict[str, Any]) -> tuple[bytes | None, float, str]:
        seq = int(state.get("seq", 0))
        client = str(state.get("client", ""))
        with self.seq_lock:
            self.latest[client] = max(self.latest.get(client, -1), seq)
        with self.lock:
            if seq < self.latest[client]:
                return None, 0.0, ""
            w = max(16, min(4096, int(state.get("width", 640))))
            h = max(16, min(4096, int(state.get("height", 360))))
            t0 = time.perf_counter()
            # Keep the state's own frame orientation; portrait overrides follow h > w.
            recipe = state_recipe(state)
            rgb = recipe.render(w, h)
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

    def start(self, body: dict[str, Any]) -> str:
        job = uuid.uuid4().hex[:10]
        self.jobs[job] = {"status": "running", "path": None, "error": None, "seconds": 0.0}
        threading.Thread(target=self._run, args=(job, body), daemon=True).start()
        return job

    def _run(self, job: str, body: dict[str, Any]) -> None:
        from ..engine import default_n_strips

        t0 = time.perf_counter()
        try:
            state = body["state"]
            w, h = int(body["width"]), int(body["height"])
            fmt = "jpg" if str(body.get("format", "png")).lower() in ("jpg", "jpeg") else "png"
            stem = "".join(c if c.isalnum() or c in "-_@" else "_" for c in (body.get("name") or state.get("recipe") or "gui"))
            path = OUT_ROOT / "gui" / f"{stem}_{w}x{h}.{fmt}"
            rgb = state_recipe(state).render(w, h, n_strips=default_n_strips(w, h))
            save_image(path, rgb, fmt)
            self.jobs[job].update(status="done", path=str(path))
        except Exception:
            self.jobs[job].update(status="error", error=traceback.format_exc())
        self.jobs[job]["seconds"] = time.perf_counter() - t0


def save_state(name: str, state: dict[str, Any], overwrite: bool = False) -> Path:
    from ..recipe import register_recipe
    from ..tomlout import recipe_to_toml

    if not name or not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError("name may only use letters, digits, '_' and '-'")
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


def make_handler(renderer: Renderer, exports: Exports):
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
            n = int(self.headers.get("Content-Length") or 0)
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
                elif url.path.startswith("/api/export/"):
                    job = exports.jobs.get(url.path.rsplit("/", 1)[-1])
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
                    path = save_state(body["name"], body["state"], bool(body.get("overwrite")))
                    self._json({"path": str(path), "name": body["name"]})
                elif url.path == "/api/export":
                    self._json({"job": exports.start(body)})
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as exc:
                self._error(exc)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, loads: list[str] | None = None) -> None:
    bootstrap(loads or [])
    httpd = ThreadingHTTPServer((host, port), make_handler(Renderer(), Exports()))
    url = f"http://{host}:{port}/"
    print(f"fractalgen GUI on {url}  (Ctrl-C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
