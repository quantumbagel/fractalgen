"""Mask stage tests: python -m unittest tests.test_mask"""

from __future__ import annotations

import math
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import fractalgen as fg
from fractalgen import mask as masking
from fractalgen.recipe import RECIPES, Recipe

fg.bootstrap()

W, H = 96, 54


def _flat(rgb=(0.2, 0.6, 0.9)) -> np.ndarray:
    return np.broadcast_to(np.asarray(rgb, float), (H, W, 3)).copy()


class MaskBuildTests(unittest.TestCase):
    def test_off_until_switched_on(self):
        self.assertFalse(masking.active({}))
        self.assertFalse(masking.active({"size": 0.5, "glow": 1.0}))
        self.assertTrue(masking.active({"text": "A"}))
        self.assertTrue(masking.active({"shape": "circle"}))

    def test_check_coerces_and_rejects(self):
        self.assertEqual(masking.check_mask({"size": "0.5", "invert": "true", "text": 12}),
                         {"size": 0.5, "invert": True, "text": "12"})
        self.assertEqual(masking.check_mask({"size": None}), {})
        with self.assertRaisesRegex(KeyError, "Unknown mask key"):
            masking.check_mask({"bogus": 1})
        with self.assertRaisesRegex(ValueError, "must be one of"):
            masking.check_mask({"shape": "blob"})
        with self.assertRaisesRegex(ValueError, "finite"):
            masking.check_mask({"size": float("nan")})

    def test_circle_area(self):
        m = masking.build({"shape": "circle", "size": 0.8}, 200, 100)
        r = 0.8 * 100 / 2
        self.assertAlmostEqual(float(m.sum()), math.pi * r * r, delta=0.01 * math.pi * r * r)
        self.assertEqual(m[50, 100], 1.0)
        self.assertEqual(m[0, 0], 0.0)

    def test_every_shape_covers_its_center_and_not_corners(self):
        for shape in masking.SHAPES:
            if not shape or shape == "ring":
                continue
            m = masking.build({"shape": shape}, W, H)
            self.assertGreater(m[H // 2, W // 2], 0.5, shape)
            self.assertEqual(m[0, 0], 0.0, shape)
        ring = masking.build({"shape": "ring"}, W, H)
        self.assertEqual(ring[H // 2, W // 2], 0.0)

    def test_position_and_rotation(self):
        left = masking.build({"shape": "circle", "size": 0.3, "x": 0.2}, W, H)
        ys, xs = np.nonzero(left > 0.5)
        self.assertAlmostEqual(xs.mean(), 0.2 * W - 0.5, delta=1.0)
        a = masking.build({"shape": "rounded", "size": 0.5}, W, H)
        b = masking.build({"shape": "rounded", "size": 0.5, "rotate": 90}, W, H)
        self.assertGreater(np.abs(a - b).sum(), 10)

    def test_text_fills_box(self):
        m = masking.build({"text": "HI", "size": 0.8}, W, H)
        ys, xs = np.nonzero(m > 0.5)
        self.assertGreater(len(xs), 50)
        # fits inside the box and nearly touches it on the tighter side
        self.assertGreaterEqual(xs.min(), 0.1 * W - 2)
        self.assertLessEqual(xs.max(), 0.9 * W + 2)
        self.assertGreater(max(xs.max() - xs.min(), (ys.max() - ys.min()) * W / H), 0.6 * W)

    def test_multiline_text(self):
        one = masking.build({"text": "AB"}, W, H)
        two = masking.build({"text": "A\\nB"}, W, H)
        ys1, ys2 = np.nonzero(one > 0.5)[0], np.nonzero(two > 0.5)[0]
        self.assertGreater(ys2.max() - ys2.min(), ys1.max() - ys1.min())

    def test_unknown_font_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no font named"):
            masking.build({"text": "A", "font": "No Such Font Family 123"}, W, H)

    def test_image_mask_alpha_and_luma(self):
        with tempfile.TemporaryDirectory() as d:
            rgba = np.zeros((10, 10, 4), np.uint8)
            rgba[:, :5] = (255, 255, 255, 255)  # left half opaque
            Image.fromarray(rgba, "RGBA").save(Path(d) / "a.png")
            m = masking.build({"image": str(Path(d) / "a.png"), "size": 1.0}, 40, 40)
            self.assertGreater(m[20, 5], 0.9)
            self.assertLess(m[20, 35], 0.1)
            gray = np.zeros((10, 10), np.uint8)
            gray[:, 5:] = 255  # right half white
            Image.fromarray(gray, "L").save(Path(d) / "b.png")
            m = masking.build({"image": str(Path(d) / "b.png"), "size": 1.0}, 40, 40)
            self.assertLess(m[20, 5], 0.1)
            self.assertGreater(m[20, 35], 0.9)
        with self.assertRaises(FileNotFoundError):
            masking.build({"image": "definitely_missing.png"}, 40, 40)


class MaskCompositeTests(unittest.TestCase):
    def test_inside_is_fractal_outside_is_background(self):
        out = masking.apply_mask(_flat(), {"shape": "circle", "background": "ff0000"}, 1.0)
        np.testing.assert_allclose(out[H // 2, W // 2], [0.2, 0.6, 0.9], atol=1e-6)
        np.testing.assert_allclose(out[0, 0], [1.0, 0.0, 0.0], atol=1e-6)

    def test_invert_swaps(self):
        out = masking.apply_mask(_flat(), {"shape": "circle", "invert": True}, 1.0)
        np.testing.assert_allclose(out[H // 2, W // 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(out[0, 0], [0.2, 0.6, 0.9], atol=1e-6)

    def test_transparent_gives_rgba(self):
        out = masking.apply_mask(_flat(), {"shape": "circle", "background": "transparent"}, 1.0)
        self.assertEqual(out.shape, (H, W, 4))
        self.assertEqual(out[0, 0, 3], 0.0)
        self.assertEqual(out[H // 2, W // 2, 3], 1.0)
        np.testing.assert_allclose(out[H // 2, W // 2, :3], [0.2, 0.6, 0.9], atol=1e-6)

    def test_self_background_is_dimmed_fractal(self):
        out = masking.apply_mask(_flat(), {"shape": "circle", "background": "self", "background_dim": 0.5}, 1.0)
        np.testing.assert_allclose(out[0, 0], [0.1, 0.3, 0.45], atol=1e-6)

    def test_palette_background_is_vertical_gradient(self):
        out = masking.apply_mask(_flat(), {"shape": "circle", "size": 0.1, "background": "palette:ember"}, 1.0)
        self.assertGreater(out[-1, 0].sum(), out[0, 0].sum())
        np.testing.assert_allclose(out[:, 0], out[:, -1])

    def test_bad_background_message(self):
        with self.assertRaisesRegex(ValueError, "transparent, self"):
            masking.apply_mask(_flat(), {"shape": "circle", "background": "purple-ish"}, 1.0)

    def test_effects_stay_in_range_and_change_pixels(self):
        base = masking.apply_mask(_flat(), {"shape": "star"}, 1.0)
        for fx in ({"outline": 3}, {"glow": 1.5}, {"shadow": 1.0}, {"bevel": 2.0}, {"feather": 6}):
            out = masking.apply_mask(_flat(), {"shape": "star", "background": "404040", **fx}, 1.0)
            self.assertTrue(np.isfinite(out).all(), fx)
            self.assertGreaterEqual(out.min(), 0.0, fx)
            self.assertLessEqual(out.max(), 1.0 + 1e-9, fx)
            self.assertGreater(np.abs(out - base).sum(), 1.0, fx)

    def test_blur_keeps_shape_and_mean(self):
        rng = np.random.default_rng(0)
        a = rng.random((120, 90)).astype(np.float32)
        for r in (0.0, 3.0, 40.0):
            b = masking.blur(a, r)
            self.assertEqual(b.shape, a.shape)
            self.assertAlmostEqual(float(b.mean()), float(a.mean()), delta=0.02)
        self.assertEqual(masking.blur(np.zeros((5, 7, 3)), 30).shape, (5, 7, 3))


class MaskRecipeTests(unittest.TestCase):
    def test_empty_mask_is_a_noop(self):
        r = RECIPES.get("julia_dragon")
        a = r.render(64, 36)
        b = r.render(64, 36, overrides={"mask": {"size": 0.3}})
        np.testing.assert_array_equal(a, b)

    def test_override_and_post_after_mask(self):
        r = RECIPES.get("julia_dragon")
        out = r.render(64, 36, overrides={"mask": {"shape": "circle", "background": "ffffff"},
                                          "post": {"invert": True}})
        np.testing.assert_allclose(out[0, 0], 0.0, atol=1e-6)  # white background, inverted by post

    def test_transparent_render_saves_png_alpha_and_flat_jpeg(self):
        r = RECIPES.get("julia_dragon")
        rgba = r.render(64, 36, overrides={"mask": {"text": "A", "background": "transparent"},
                                           "post": {"saturation": 1.2}})
        self.assertEqual(rgba.shape, (36, 64, 4))
        with tempfile.TemporaryDirectory() as d:
            fg.save_image(Path(d) / "a.png", rgba)
            fg.save_image(Path(d) / "a.jpg", rgba)
            self.assertEqual(Image.open(Path(d) / "a.png").mode, "RGBA")
            self.assertEqual(Image.open(Path(d) / "a.jpg").mode, "RGB")

    def test_recipe_table_and_portrait(self):
        r = Recipe.from_table("m", {
            "generator": "escape_time",
            "mask": {"text": "WIDE"},
            "portrait": {"mask": {"text": "TALL", "size": "0.5"}},
        })
        self.assertEqual(r.resolve_mask(64, 36), {"text": "WIDE"})
        self.assertEqual(r.resolve_mask(36, 64), {"text": "TALL", "size": 0.5})
        self.assertEqual(r.resolve_mask(36, 64, {"mask": {"size": None}}), {"text": "TALL"})
        with self.assertRaises(KeyError):
            Recipe.from_table("bad", {"generator": "escape_time", "mask": {"nope": 1}})

    def test_builtin_masked_recipes_render(self):
        for name in ("seahorse_porthole", "snowflake_star", "dragon_lettering"):
            self.assertTrue(RECIPES.get(name).layers, name)
            self.assertIsNotNone(RECIPES.get(name).resolve_mask(64, 36), name)

    def test_fork_round_trip_keeps_mask(self):
        import tomllib

        from fractalgen.tomlout import recipe_to_toml

        src = RECIPES.get("dragon_lettering")
        text = recipe_to_toml(src, "forked")
        self.assertIn("[recipe.forked.mask]", text)
        forked = Recipe.from_table("forked", tomllib.loads(text)["recipe"]["forked"])
        self.assertEqual(forked.layers, src.layers)
        self.assertEqual(forked.background, src.background)
        self.assertEqual(forked.resolve_mask(36, 64), src.resolve_mask(36, 64))


class MaskCliTests(unittest.TestCase):
    def test_mask_flags(self):
        from fractalgen.cli import main

        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stdout", StringIO()):
            code = main(["render", "julia_dragon", "--size", "48x27", "--out", d,
                         "--mask", "text=007", "--mask", "background=000000", "--mask", "outline_color=fff"])
            self.assertEqual(code, 0)
            self.assertTrue(any(Path(d).glob("*.png")))

    def test_mask_listing_and_errors(self):
        from fractalgen.cli import main

        out = StringIO()
        with mock.patch("sys.stdout", out):
            self.assertEqual(main(["mask"]), 0)
        self.assertIn("background", out.getvalue())
        err = StringIO()
        with mock.patch("sys.stderr", err):
            self.assertEqual(main(["render", "julia_dragon", "--size", "48x27", "--mask", "shape=blob"]), 2)
        self.assertIn("must be one of", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


class MaskGuiStateTests(unittest.TestCase):
    def test_state_round_trip(self):
        from fractalgen.gui import server

        st = server.recipe_state("dragon_lettering", 64, 36)
        self.assertEqual(st["layers"][0]["text"], "FRACTAL")
        layer = {**st["layers"][0], "glow": 0.1}
        del layer["shadow"]
        st["layers"] = [layer]
        r = server.state_recipe(st, "gui_mask_test")
        self.assertEqual(r.layers[0]["glow"], 0.1)
        self.assertNotIn("shadow", r.layers[0])
        self.assertEqual(r.resolve_mask(36, 64)["text"], "FRAC\nTAL")  # portrait text survives
        self.assertEqual(r.resolve_mask(36, 64)["glow"], 0.1)  # and so does the edit
        cmd = server.cli_command(st, 64, 36)
        self.assertIn("--mask glow=0.1", cmd)
        self.assertIn("--mask shadow=0.0", cmd)
        self.assertTrue(any(k["name"] == "fill" for k in server.catalog()["background_keys"]))


if __name__ == "__main__":
    unittest.main()
