"""Regression tests for GUI server / tomlout fixes (tiny renders only)."""

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

import numpy as np

import fractalgen as fg
from fractalgen.gui.server import Exports, Renderer, cli_command, make_handler, save_state
from fractalgen.recipe import RECIPES
from fractalgen.tomlout import recipe_to_toml


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    return w, h


class TomloutNameQuotingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fg.bootstrap()

    def _fork_parse(self, new_name: str) -> dict:
        r = RECIPES.get("julia_dragon")
        text = recipe_to_toml(r, new_name)
        data = tomllib.loads(text)
        self.assertIn(new_name, data["recipe"])
        return data["recipe"][new_name]

    def test_dotted_name(self) -> None:
        self._fork_parse("foo.bar")

    def test_at_name(self) -> None:
        self._fork_parse("weird@name")

    def test_unicode_name(self) -> None:
        self._fork_parse("café")

    def test_space_name(self) -> None:
        self._fork_parse("hello world")


class GuiFixesHttpTests(unittest.TestCase):
    _base: str = ""
    _httpd: ThreadingHTTPServer | None = None

    @classmethod
    def setUpClass(cls) -> None:
        fg.bootstrap()
        cls._renderer = Renderer()
        cls._exports = Exports()
        cls._httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(cls._renderer, cls._exports, None),
        )
        host, port = cls._httpd.server_address
        cls._base = f"http://{host}:{port}"
        threading.Thread(target=cls._httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._httpd:
            cls._httpd.shutdown()
            cls._httpd.server_close()

    def _url(self, path: str) -> str:
        return self._base + path

    def _post_json(self, path: str, body: dict, *, code: int = 200, headers: dict | None = None) -> tuple[int, bytes]:
        data = json.dumps(body).encode()
        hdrs = {"Content-Type": "application/json", **(headers or {})}
        req = request.Request(self._url(path), data=data, headers=hdrs, method="POST")
        try:
            with request.urlopen(req) as r:
                self.assertEqual(r.status, code)
                return r.status, r.read()
        except error.HTTPError as e:
            with e:
                if e.code != code:
                    raise
                return e.code, e.read()

    def _state(self) -> dict:
        from fractalgen.gui.server import recipe_state

        return recipe_state("julia_dragon", 64, 36)

    def test_content_length_negative(self) -> None:
        data = json.dumps(self._state()).encode()
        req = request.Request(
            self._url("/api/render"),
            data=data,
            headers={"Content-Type": "application/json", "Content-Length": "-1"},
            method="POST",
        )
        with self.assertRaises(error.HTTPError) as ctx:
            request.urlopen(req)
        with ctx.exception:
            self.assertEqual(ctx.exception.code, 400)

    def test_content_length_non_numeric(self) -> None:
        data = json.dumps(self._state()).encode()
        req = request.Request(
            self._url("/api/render"),
            data=data,
            headers={"Content-Type": "application/json", "Content-Length": "nope"},
            method="POST",
        )
        with self.assertRaises(error.HTTPError) as ctx:
            request.urlopen(req)
        with ctx.exception:
            self.assertEqual(ctx.exception.code, 400)

    def test_preview_honors_size(self) -> None:
        st = {**self._state(), "width": 17, "height": 13, "seq": 1, "client": "sz"}
        _, raw = self._post_json("/api/render", st, code=200)
        self.assertEqual(_png_size(raw), (17, 13))

    def test_preview_zero_size_rejected(self) -> None:
        st = {**self._state(), "width": 0, "height": 10, "seq": 3, "client": "z"}
        _, raw = self._post_json("/api/render", st, code=400)
        self.assertIn(b"preview", raw)

    def test_preview_too_large_rejected(self) -> None:
        st = {**self._state(), "width": 4097, "height": 100, "seq": 11, "client": "big"}
        _, raw = self._post_json("/api/render", st, code=400)
        self.assertIn(b"too large", raw)

    def test_preview_inf_size_rejected(self) -> None:
        st = {**self._state(), "width": float("inf"), "height": 10, "seq": 12, "client": "infw"}
        _, raw = self._post_json("/api/render", st, code=400)
        err = json.loads(raw)["error"]
        self.assertTrue(err.startswith("ValueError"), err)
        self.assertIn("integer", err)

    def test_preview_nan_center(self) -> None:
        st = self._state()
        st["view"]["center"] = [float("nan"), 0.0]
        st.update(seq=2, client="nan")
        _, raw = self._post_json("/api/render", st, code=400)
        self.assertIn(b"center", raw)
        self.assertNotIn(b"TypeError", raw)

    def test_preview_null_half_width(self) -> None:
        st = self._state()
        st["view"]["half_width"] = None
        st.update(seq=9, client="nullhw")
        _, raw = self._post_json("/api/render", st, code=400)
        self.assertIn(b"half_width", raw)
        self.assertNotIn(b"TypeError", raw)

    def test_preview_null_center(self) -> None:
        st = self._state()
        st["view"]["center"] = None
        st.update(seq=10, client="nullc")
        _, raw = self._post_json("/api/render", st, code=400)
        self.assertIn(b"center", raw)
        self.assertNotIn(b"TypeError", raw)

    def test_save_unicode_name_rejected(self) -> None:
        _, raw = self._post_json(
            "/api/save",
            {"name": "café", "state": self._state(), "overwrite": False},
            code=400,
        )
        self.assertIn(b"ASCII", raw)

    def test_cli_missing_generator(self) -> None:
        _, raw = self._post_json("/api/cli", {"state": {"recipe": None}, "width": 64, "height": 36}, code=400)
        self.assertIn(b"missing generator", raw)

    def test_cli_fit_quoted(self) -> None:
        st = self._state()
        st["view"]["fit"] = "fill; echo PWNED"
        _, raw = self._post_json("/api/cli", {"state": st, "width": 64, "height": 36})
        cmd = json.loads(raw)["command"]
        self.assertIn("'fill; echo PWNED'", cmd)

    def test_export_invalid_format(self) -> None:
        st = self._state()
        _, raw = self._post_json(
            "/api/export",
            {"state": st, "width": 32, "height": 18, "format": "webp"},
            code=400,
        )
        self.assertIn(b"format", raw)

    def test_export_missing_state(self) -> None:
        _, raw = self._post_json("/api/export", {"width": 32, "height": 18}, code=400)
        self.assertIn(b"state", raw)

    def test_export_zero_size(self) -> None:
        st = self._state()
        _, raw = self._post_json(
            "/api/export",
            {"state": st, "width": 0, "height": 45, "format": "png"},
            code=400,
        )
        self.assertIn(b"export", raw)

    def test_export_status_error_progress_not_one(self) -> None:
        from fractalgen import progress

        exports = Exports()
        tr = progress.Tracker()
        exports.jobs["err1"] = {
            "status": "error",
            "path": None,
            "error": "boom",
            "seconds": 0.1,
            "_t0": time.perf_counter(),
            "_tracker": tr,
        }
        st = exports.status("err1")
        self.assertIsNotNone(st)
        self.assertLess(st["progress"], 1.0)

    def test_overwrite_string_false(self) -> None:
        import fractalgen.gui.server as srv

        with tempfile.TemporaryDirectory() as td:
            orig = srv.CUSTOM_DIR
            srv.CUSTOM_DIR = Path(td)
            try:
                st = self._state()
                save_state("qa_ow_fix", st, overwrite=False)
                mtime1 = (srv.CUSTOM_DIR / "qa_ow_fix.toml").stat().st_mtime
                time.sleep(0.05)
                _, _ = self._post_json(
                    "/api/save",
                    {"name": "qa_ow_fix", "state": st, "overwrite": "false"},
                    code=400,
                )
                mtime2 = (srv.CUSTOM_DIR / "qa_ow_fix.toml").stat().st_mtime
                self.assertEqual(mtime1, mtime2)
            finally:
                srv.CUSTOM_DIR = orig
                RECIPES._items.pop("qa_ow_fix", None)

    def test_waiting_two_clients(self) -> None:
        """One client's wait slot must survive another client joining the queue."""
        r = Renderer()
        threads: list[threading.Thread] = []

        def go(client: str) -> None:
            r.render_png({**self._state(), "width": 16, "height": 9, "seq": 1, "client": client})

        def wait_until(client: str) -> None:
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if r.progress(client).get("waiting"):
                    return
                time.sleep(0.01)
            self.fail(f"{client} never reported waiting: {r.progress(client)}")

        r.lock.acquire()
        try:
            t_b = threading.Thread(target=go, args=("waitB",))
            t_b.start()
            threads.append(t_b)
            wait_until("waitB")
            t_c = threading.Thread(target=go, args=("waitC",))
            t_c.start()
            threads.append(t_c)
            wait_until("waitC")
            self.assertTrue(r.progress("waitB").get("waiting"), r.progress("waitB"))
        finally:
            r.lock.release()
            for t in threads:
                t.join(timeout=30)

    def test_waiting_counter_same_client(self) -> None:
        r = Renderer()
        r._inc_waiting("same")
        r._inc_waiting("same")
        r._dec_waiting("same")
        self.assertTrue(r.progress("same").get("waiting"))
        r._dec_waiting("same")
        self.assertFalse(r.progress("same").get("waiting"))


class CliCommandUnitTests(unittest.TestCase):
    def test_try_path_requires_generator(self) -> None:
        with self.assertRaises(ValueError):
            cli_command({"generator": None, "recipe": None}, 64, 36)


if __name__ == "__main__":
    unittest.main()


class GuiSaveFramingTests(unittest.TestCase):
    """A panned / zoomed look saved from the GUI renders like the live preview."""

    def _saved(self, live):
        import tomllib

        from fractalgen.recipe import Recipe
        from fractalgen.tomlout import recipe_to_toml

        return Recipe.from_table("saved", tomllib.loads(recipe_to_toml(live, "saved"))["recipe"]["saved"])

    def test_landscape_zoom_scales_phone_framing(self):
        from fractalgen.gui.server import recipe_state, state_recipe

        st = recipe_state("julia_dragon", 64, 36)
        st["view"] = {**st["view"], "center": [0.35, -0.22], "half_width": 0.4}
        r = state_recipe(st, "x")
        port = r.resolve(36, 64)[2]
        # julia_dragon frames phones at [-0.18, 0.24] / 0.6 vs [0, 0.05] / 1.45 on desktops
        self.assertAlmostEqual(port.center, complex(0.17, -0.03))
        self.assertAlmostEqual(port.half_width, 0.6 * 0.4 / 1.45)

    def test_saved_matches_live_in_both_orientations(self):
        from fractalgen.gui.server import recipe_state, state_recipe

        for w, h in ((64, 36), (36, 64)):
            st = recipe_state("julia_dragon", w, h)
            st["view"] = {**st["view"], "half_width": st["view"]["half_width"] * 0.5}
            live = state_recipe(st, "x")
            saved = self._saved(live)
            self.assertEqual(saved.tone_from, "julia_dragon")
            for size in ((64, 36), (36, 64)):
                np.testing.assert_array_equal(live.render(*size), saved.render(*size))
