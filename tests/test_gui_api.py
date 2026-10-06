"""HTTP integration tests for fractalgen.gui.server (stdlib unittest)."""

from __future__ import annotations

import json
import struct
import tempfile
import threading
import time
import tomllib
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

from fractalgen.gui.server import Exports, Renderer, make_handler
from fractalgen.loader import bootstrap
from fractalgen.recipe import RECIPES

SAVE_NAME = "_gui_test_tmp"
_BLANK_GENS = ("escape_time", "map_attractor", "domain_color")


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def _parse_extent(hdr: str) -> tuple[float, float]:
    a, b = hdr.split(",", 1)
    return float(a.strip()), float(b.strip())


class GuiApiTests(unittest.TestCase):
    _seq = 0
    _base: str = ""
    _httpd: ThreadingHTTPServer | None = None
    _custom_tmp: tempfile.TemporaryDirectory[str] | None = None
    _out_tmp: tempfile.TemporaryDirectory[str] | None = None

    @classmethod
    def setUpClass(cls) -> None:
        bootstrap()
        import fractalgen.gui.server as srv

        cls._custom_tmp = tempfile.TemporaryDirectory()
        cls._out_tmp = tempfile.TemporaryDirectory()
        cls._orig_custom, cls._orig_out = srv.CUSTOM_DIR, srv.OUT_ROOT
        srv.CUSTOM_DIR = Path(cls._custom_tmp.name)
        srv.OUT_ROOT = Path(cls._out_tmp.name)
        cls._httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Renderer(), Exports()))
        host, port = cls._httpd.server_address
        cls._base = f"http://{host}:{port}"
        threading.Thread(target=cls._httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._httpd:
            cls._httpd.shutdown()
            cls._httpd.server_close()
        import fractalgen.gui.server as srv

        srv.CUSTOM_DIR, srv.OUT_ROOT = cls._orig_custom, cls._orig_out
        if SAVE_NAME in RECIPES:
            RECIPES._items.pop(SAVE_NAME, None)
        cls._custom_tmp and cls._custom_tmp.cleanup()
        cls._out_tmp and cls._out_tmp.cleanup()

    def _url(self, path: str) -> str:
        return self._base + path

    def _get(self, path: str, *, code: int = 200) -> tuple[int, dict[str, str], bytes]:
        req = request.Request(self._url(path))
        try:
            with request.urlopen(req) as r:
                self.assertEqual(r.status, code)
                return r.status, dict(r.headers), r.read()
        except error.HTTPError as e:
            with e:
                if e.code != code:
                    raise
                return e.code, dict(e.headers), e.read()

    def _post_json(self, path: str, body: dict, *, code: int = 200) -> tuple[int, dict[str, str], bytes]:
        data = json.dumps(body).encode()
        req = request.Request(self._url(path), data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req) as r:
                self.assertEqual(r.status, code)
                return r.status, dict(r.headers), r.read()
        except error.HTTPError as e:
            with e:
                if code != e.code:
                    raise
                return e.code, dict(e.headers), e.read()

    def _json(self, path: str) -> dict:
        _, _, raw = self._get(path)
        return json.loads(raw)

    def _render(self, state: dict) -> tuple[dict[str, str], bytes]:
        GuiApiTests._seq += 1
        state = {**state, "seq": GuiApiTests._seq}
        _, hdrs, png = self._post_json("/api/render", state)
        self.assertEqual(hdrs.get("Content-type") or hdrs.get("Content-Type"), "image/png")
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        return hdrs, png

    def test_index_html(self) -> None:
        _, hdrs, body = self._get("/")
        self.assertIn("text/html", hdrs.get("Content-type") or hdrs.get("Content-Type", ""))
        self.assertIn(b"<script>", body)

    def test_catalog(self) -> None:
        cat = self._json("/api/catalog")
        for key in ("generators", "recipes", "palettes", "post_ops", "presets", "coefficients"):
            self.assertIn(key, cat)
        for g in cat["generators"].values():
            for p in g["params"]:
                for k in ("name", "kind", "default", "range", "choices"):
                    self.assertIn(k, p)
        json.loads(json.dumps(cat), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

    def test_render_every_recipe_generator(self) -> None:
        cat = self._json("/api/catalog")
        first: dict[str, str] = {}
        for r in cat["recipes"]:
            first.setdefault(r["generator"], r["name"])
        for gen, recipe in first.items():
            uses_view = cat["generators"][gen]["uses_view"]
            for w, h in ((64, 36), (36, 64)):
                st = self._json(f"/api/state?recipe={recipe}&w={w}&h={h}")
                hdrs, png = self._render(st)
                self.assertEqual(_png_size(png), (w, h))
                if uses_view:
                    ext = hdrs.get("X-View-Extent") or hdrs.get("x-view-extent")
                    self.assertIsNotNone(ext)
                    a, b = _parse_extent(ext)
                    self.assertTrue(all(abs(x) < 1e300 for x in (a, b)))

    def test_blank_generator_state(self) -> None:
        cat = self._json("/api/catalog")
        for gen in _BLANK_GENS:
            if gen not in cat["generators"]:
                continue
            schema = {p["name"]: p for p in cat["generators"][gen]["params"]}
            params = {n: p["default"] for n, p in schema.items() if not p.get("required") and n != "palette"}
            pal = schema["palette"]["default"] if "palette" in schema else None
            view = {"center": [0, 0], "half_width": 2, "rotation": 0, "fit": "fill"} if cat["generators"][gen]["uses_view"] else None
            st = {"recipe": None, "generator": gen, "palette": pal, "params": params, "view": view,
                  "post": {}, "width": 64, "height": 36}
            self._render(st)

    def test_post_and_palette_forms(self) -> None:
        st = self._json("/api/state?recipe=julia_dragon&w=64&h=36")
        for post in (
            {"palette": "ember_r", "palette_mix": 0.5, "hue_shift": 30, "tint": "ffcc88", "vignette": 0.3, "flip_x": True},
            {"palette": ["000000", "ff0000", "ffffff"]},
            {"palette": {"stops": ["000000", "00ff00", "ffffff"], "positions": [0, 0.2, 1]}},
        ):
            self._render({**st, "post": post})

    def test_stale_render_204(self) -> None:
        st = self._json("/api/state?recipe=julia_dragon&w=64&h=36")
        self._post_json("/api/render", {**st, "seq": 10**9, "client": "stale-a"})
        _, headers, body = self._post_json("/api/render", {**st, "seq": 5, "client": "stale-a"}, code=204)
        self.assertEqual(headers.get("X-Stale"), "1")
        # Another page (e.g. after a reload) keeps its own sequence.
        self._render({**st, "seq": 1, "client": "stale-b"})

    def test_edits_survive_portrait_overrides(self) -> None:
        from fractalgen.gui.server import recipe_state, state_recipe

        base = RECIPES.get("julia_dragon")
        self.assertTrue(base.portrait.get("view"), "julia_dragon should have a portrait view")
        st = recipe_state("julia_dragon", 36, 64)
        st["view"]["half_width"] = 0.123
        st["params"]["max_iter"] = 77
        r = state_recipe(st)
        _, p_port, v_port = r.resolve(36, 64)
        self.assertEqual((v_port.half_width, p_port["max_iter"]), (0.123, 77))
        _, p_land, v_land = r.resolve(64, 36)
        self.assertEqual((v_land.half_width, p_land["max_iter"]), (0.123, 77))
        # Untouched portrait framing stays portrait-only.
        self.assertEqual(v_land.center, base.resolve(64, 36)[2].center)
        self.assertEqual(v_port.center, base.resolve(36, 64)[2].center)
        # Switching a recipe's post op off sticks.
        st = recipe_state("julia_dragon", 64, 36)
        st["post"] = {"vignette": 0.5}
        self.assertEqual(state_recipe(st).resolve_post(64, 36).get("vignette"), 0.5)

    def test_bad_input_400(self) -> None:
        _, _, raw = self._get("/api/state?recipe=__no_such__&w=64&h=36", code=400)
        self.assertIn("error", json.loads(raw))
        st = self._json("/api/state?recipe=julia_dragon&w=64&h=36")
        st["generator"] = "__no_generator__"
        _, _, raw = self._post_json("/api/render", {**st, "seq": 1}, code=400)
        self.assertIn("error", json.loads(raw))

    def test_save_recipe(self) -> None:
        st = self._json("/api/state?recipe=julia_dragon&w=64&h=36")
        st["params"] = {**st["params"], "max_iter": 123}
        _, _, raw = self._post_json("/api/save", {"name": SAVE_NAME, "state": st})
        info = json.loads(raw)
        path = Path(info["path"])
        self.assertTrue(str(path).startswith(str(Path(self._custom_tmp.name))))
        self.assertTrue(path.is_file())
        tbl = tomllib.loads(path.read_text())
        self.assertEqual(tbl["recipe"][SAVE_NAME]["params"]["max_iter"], 123)
        bootstrap([path])
        self.assertEqual(RECIPES.get(SAVE_NAME).params["max_iter"], 123)
        self._post_json("/api/save", {"name": SAVE_NAME, "state": st}, code=400)
        self._post_json("/api/save", {"name": SAVE_NAME, "state": st, "overwrite": True})
        _, _, err = self._post_json("/api/save", {"name": "bad name!", "state": st}, code=400)
        self.assertIn("error", json.loads(err))

    def test_export_job(self) -> None:
        st = self._json("/api/state?recipe=julia_dragon&w=64&h=36")
        _, _, raw = self._post_json("/api/export", {"state": st, "width": 80, "height": 45, "format": "jpg", "name": "t"})
        job = json.loads(raw)["job"]
        status = "running"
        for _ in range(300):
            info = self._json(f"/api/export/{job}")
            status = info["status"]
            if status != "running":
                break
            time.sleep(0.2)
        self.assertEqual(status, "done")
        p = Path(info["path"])
        self.assertTrue(p.is_file() and str(p).endswith(".jpg"))
        try:
            from PIL import Image

            with Image.open(p) as im:
                self.assertEqual(im.size, (80, 45))
        except ImportError:
            pass
        _, _, body = self._get("/api/export/nope", code=404)
        self.assertIn(b"unknown", body)


if __name__ == "__main__":
    unittest.main()
