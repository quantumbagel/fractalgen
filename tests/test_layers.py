"""Layers (several text / shape / image layers), ordered effects, shapes and their round trips."""

from __future__ import annotations

import tempfile
import tomllib
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import fractalgen as fg
from fractalgen import layers as L
from fractalgen.mask import SHAPES, shape
from fractalgen.post import POST_OPS, apply_post, check_post, patch_post, post_to_legacy
from fractalgen.recipe import RECIPES, Recipe

fg.bootstrap([])


def _flat(h=60, w=100, v=0.6):
    return np.full((h, w, 3), v)


def _grad(h=60, w=100):
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([x / w, y / h, np.full((h, w), 0.5)], axis=-1)


class LayerCompositeTests(unittest.TestCase):
    def test_overlay_layers_paint_on_top_without_cutting(self):
        out = L.apply_layers(_flat(), [{"type": "shape", "shape": "circle", "fill": "ff0000", "size": 0.3}], {}, 1.0)
        self.assertEqual(out.shape, (60, 100, 3))
        np.testing.assert_allclose(out[30, 50], [1, 0, 0], atol=1e-6)  # inside: red
        np.testing.assert_allclose(out[2, 2], [0.6, 0.6, 0.6], atol=1e-6)  # outside: the render, untouched

    def test_several_cutouts_union(self):
        layers = [{"type": "shape", "shape": "circle", "size": 0.3, "x": 0.25},
                  {"type": "shape", "shape": "circle", "size": 0.3, "x": 0.75}]
        out = L.apply_layers(_flat(), layers, {"fill": "000000"}, 1.0)
        self.assertAlmostEqual(out[30, 25, 0], 0.6, places=5)
        self.assertAlmostEqual(out[30, 75, 0], 0.6, places=5)
        self.assertAlmostEqual(out[30, 50, 0], 0.0, places=5)

    def test_order_matters(self):
        red = {"type": "shape", "shape": "circle", "fill": "ff0000", "size": 0.5}
        blue = {**red, "fill": "0000ff"}
        a = L.apply_layers(_flat(), [red, blue], {}, 1.0)
        b = L.apply_layers(_flat(), [blue, red], {}, 1.0)
        self.assertGreater(a[30, 50, 2], 0.99)
        self.assertGreater(b[30, 50, 0], 0.99)

    def test_opacity_blend_and_disabled(self):
        base = {"type": "shape", "shape": "rectangle", "fill": "ffffff", "size": 1.4}
        half = L.apply_layers(_flat(), [{**base, "opacity": 0.5}], {}, 1.0)
        self.assertAlmostEqual(half[30, 50, 0], 0.8, places=5)
        mul = L.apply_layers(_flat(), [{**base, "fill": "808080", "blend": "multiply"}], {}, 1.0)
        self.assertAlmostEqual(mul[30, 50, 0], 0.6 * 128 / 255, places=4)
        off = L.apply_layers(_flat(), [{**base, "enabled": False}], {}, 1.0)
        np.testing.assert_allclose(off, _flat())

    def test_gradient_and_recipe_fills(self):
        out = L.apply_layers(_flat(), [{"type": "shape", "shape": "rectangle", "size": 1.4, "fill": "palette:ember"}], {}, 1.0)
        self.assertFalse(np.allclose(out[2], out[-3]))  # top-to-bottom gradient
        out = L.apply_layers(_flat(48, 64), [{"type": "shape", "shape": "circle", "fill": "recipe:julia_dragon"}], {}, 1.0)
        self.assertFalse(np.allclose(out[24, 32], 0.6))

    def test_transparent_background_with_cutout(self):
        out = L.apply_layers(_grad(), [{"type": "text", "text": "HI"}], {"fill": "transparent"}, 1.0)
        self.assertEqual(out.shape[-1], 4)
        self.assertEqual(out[0, 0, 3], 0.0)
        self.assertGreater(out[..., 3].max(), 0.99)

    def test_image_layer_paints_its_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "logo.png"
            im = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
            im.paste((0, 255, 0, 255), (5, 5, 15, 15))
            im.save(p)
            out = L.apply_layers(_flat(), [{"type": "image", "image": str(p), "size": 0.5}], {}, 1.0)
            np.testing.assert_allclose(out[30, 50], [0, 1, 0], atol=0.02)
            np.testing.assert_allclose(out[2, 2], [0.6, 0.6, 0.6], atol=1e-6)
            cut = L.apply_layers(_flat(), [{"type": "image", "image": str(p), "size": 0.5, "fill": "fractal"}], {}, 1.0)
            np.testing.assert_allclose(cut[30, 50], [0.6, 0.6, 0.6], atol=0.02)
            np.testing.assert_allclose(cut[2, 2], [0, 0, 0], atol=1e-6)

    def test_boxes(self):
        boxes = L.layer_boxes([{"type": "shape", "shape": "circle", "size": 0.5, "x": 0.25},
                               {"type": "text", "text": ""}], 200, 100)
        x0, y0, x1, y1 = boxes[0]
        self.assertAlmostEqual((x0 + x1) / 2, 0.25, delta=0.02)
        self.assertLess(x1 - x0, 0.3)
        self.assertIsNone(boxes[1])

    def test_validation(self):
        with self.assertRaises(ValueError):
            L.check_layer({"text": "no type"})
        with self.assertRaises(KeyError):
            L.check_layer({"type": "sprite"})
        with self.assertRaises(KeyError):
            L.check_layer({"type": "text", "shape": "circle"})  # shape is not a text key
        with self.assertRaises(ValueError):
            L.check_layer({"type": "shape", "shape": "blob"})
        self.assertEqual(L.check_layer({"type": "text", "size": "0.5", "id": "x"}), {"type": "text", "size": 0.5})

    def test_patch_by_name_and_index(self):
        layers = L.check_layers([{"type": "text", "text": "A", "name": "title"}, {"type": "shape"}])
        out = L.patch_layers(layers, {"title": {"text": "B"}, "1": {"size": 0.3}, "-1": {"x": 0.1}})
        self.assertEqual(out[0]["text"], "B")
        self.assertEqual((out[1]["size"], out[1]["x"]), (0.3, 0.1))
        with self.assertRaises(KeyError):
            L.patch_layers(layers, {"nope": {}})


class ShapeRegistryTests(unittest.TestCase):
    def test_plugin_shape(self):
        @shape("test_plus", "A plus sign", replace=True)
        def _plus(x, y):
            return np.minimum(np.maximum(np.abs(x) - 1.0, np.abs(y) - 0.3), np.maximum(np.abs(x) - 0.3, np.abs(y) - 1.0))

        try:
            m = L.coverage(L.check_layer({"type": "shape", "shape": "test_plus"}), 100, 100, 1.0)
            self.assertGreater(m[50, 50], 0.5)
            self.assertEqual(m[10, 10], 0.0)
            from fractalgen.gui.server import catalog

            shape_key = next(k for k in catalog()["layer_types"][1]["keys"] if k["name"] == "shape")
            self.assertIn("test_plus", shape_key["choices"])
        finally:
            SHAPES.unregister("test_plus")


class EffectListTests(unittest.TestCase):
    def test_legacy_table_becomes_registry_order(self):
        items = check_post({"vignette": 0.3, "palette": "ember", "palette_mix": 0.5})
        self.assertEqual(items, [{"op": "palette", "value": "ember", "mix": 0.5}, {"op": "vignette", "value": 0.3}])
        self.assertEqual(post_to_legacy(items), {"palette": "ember", "palette_mix": 0.5, "vignette": 0.3})

    def test_legacy_palette_value_forms(self):
        spec = {"stops": ["000000", "ffffff"], "positions": [0, 1]}
        self.assertEqual(check_post({"palette": spec}), [{"op": "palette", "value": spec}])

    def test_list_order_repeat_mix_and_enabled(self):
        rgb = _grad()
        a = apply_post(rgb, [{"op": "invert", "value": True}, {"op": "gamma", "value": 2.0}])
        b = apply_post(rgb, [{"op": "gamma", "value": 2.0}, {"op": "invert", "value": True}])
        self.assertFalse(np.allclose(a, b))
        twice = apply_post(rgb, [{"op": "hue_shift", "value": 60}, {"op": "hue_shift", "value": 60}])
        once = apply_post(rgb, [{"op": "hue_shift", "value": 120}])
        np.testing.assert_allclose(twice, once, atol=1e-9)
        half = apply_post(rgb, [{"op": "invert", "value": True, "mix": 0.5}])
        np.testing.assert_allclose(half, np.full_like(rgb, 0.5), atol=1e-9)
        off = apply_post(rgb, [{"op": "invert", "value": True, "enabled": False}])
        np.testing.assert_allclose(off, rgb)
        self.assertIsNone(post_to_legacy(check_post([{"op": "gamma", "value": 2.0}, {"op": "invert", "value": True}])))

    def test_every_effect_runs_at_its_start_value(self):
        ctx = fg.RenderContext(width=100, height=60)
        for name, op in POST_OPS.items():
            if op.hidden:
                continue
            with self.subTest(name):
                out = apply_post(_grad(), [{"op": name, **op.start_item()}], ctx)
                self.assertEqual(out.shape, (60, 100, 3))
                self.assertTrue(np.isfinite(out).all())

    def test_multi_param_validation(self):
        self.assertEqual(check_post({"bloom": 0.4}), [{"op": "bloom", "amount": 0.4}])
        self.assertEqual(check_post([{"op": "bloom", "value": 0.4}]), [{"op": "bloom", "amount": 0.4}])
        with self.assertRaises(KeyError):
            check_post([{"op": "bloom", "nope": 1}])
        with self.assertRaises(KeyError):
            check_post([{"op": "vignette", "amount": 1}])
        with self.assertRaises(ValueError):
            check_post([{"vignette": 0.3}])

    def test_patch(self):
        items = check_post([{"op": "vignette", "value": 0.2, "mix": 0.5}, {"op": "bloom", "amount": 0.3}])
        out = patch_post(items, {"vignette": 0.6, "grain": 0.1, "bloom": None})
        self.assertEqual(out, [{"op": "vignette", "value": 0.6, "mix": 0.5}, {"op": "grain", "amount": 0.1}])


class RecipeTests(unittest.TestCase):
    def test_layers_and_post_lists_in_toml(self):
        text = '''
[recipe.multi]
generator = "escape_time"
background = { fill = "self", dim = 0.3 }

[[recipe.multi.layers]]
type = "text"
text = "TOP"
name = "title"
y = 0.2

[[recipe.multi.layers]]
type = "shape"
shape = "heart"
fill = "ff3366"
size = 0.2

[[recipe.multi.post]]
op = "bloom"
amount = 0.3

[[recipe.multi.post]]
op = "vignette"
value = 0.4

[recipe.multi.portrait]
layers = { title = { text = "T" } }
'''
        r = Recipe.from_table("multi", tomllib.loads(text)["recipe"]["multi"])
        self.assertEqual(len(r.layers), 2)
        self.assertEqual([e["op"] for e in r.post], ["bloom", "vignette"])
        self.assertEqual(r.resolve_layers(36, 64)[0][0]["text"], "T")
        self.assertEqual(r.resolve_layers(64, 36)[0][0]["text"], "TOP")
        self.assertIsNone(r.resolve_mask(64, 36))  # needs the list form
        out = r.render(64, 36)
        self.assertEqual(out.shape, (36, 64, 3))

    def test_mask_and_layers_conflict(self):
        with self.assertRaises(ValueError):
            Recipe.from_table("x", {"generator": "escape_time", "mask": {"text": "A"}, "layers": [{"type": "text"}]})

    def test_fork_writes_list_form_and_reloads(self):
        from fractalgen.tomlout import recipe_to_toml

        src = RECIPES.get("julia_dragon").with_overrides({
            "new_layers": [{"type": "text", "text": "A"}, {"type": "shape", "shape": "star", "fill": "ffffff"}],
            "post": [{"op": "grain", "amount": 0.1}, {"op": "grain", "amount": 0.05}],
        })
        text = recipe_to_toml(src, "forked2")
        self.assertIn("[[recipe.forked2.layers]]", text)
        self.assertIn("[[recipe.forked2.post]]", text)
        back = Recipe.from_table("forked2", tomllib.loads(text)["recipe"]["forked2"])
        self.assertEqual(back.layers, src.layers)
        self.assertEqual(back.post, src.post)

    def test_legacy_mask_override_patches_layers(self):
        r = RECIPES.get("dragon_lettering")
        layers, bg = r.resolve_layers(36, 64, {"mask": {"glow": 0.1, "background": "000000"}})
        self.assertEqual((layers[0]["text"], layers[0]["glow"], bg["fill"]), ("FRAC\nTAL", 0.1, "000000"))


class CliTests(unittest.TestCase):
    def test_layer_effect_flags(self):
        from fractalgen.cli import main

        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stdout", StringIO()):
            code = main(["render", "dragon_lettering", "--size", "48x27", "--out", d, "--no-layers",
                         "--layer", 'type="text", text="HI", fill="ffffff"', "--layer", 'type="shape", shape="star"',
                         "--no-post", "--effect", 'op="bloom", amount=0.2', "--post", "vignette=0.3",
                         "--background", "fill=self"])
            self.assertEqual(code, 0)
            self.assertTrue(any(Path(d).glob("*.png")))

    def test_bad_layer_is_a_clean_error(self):
        from fractalgen.cli import main

        err = StringIO()
        with mock.patch("sys.stderr", err), mock.patch("sys.stdout", StringIO()):
            self.assertEqual(main(["render", "julia_dragon", "--size", "48x27", "--layer", 'type="text", nope=1']), 2)
        self.assertNotIn("Traceback", err.getvalue())

    def test_gui_cli_command_round_trips(self):
        from fractalgen.gui.server import cli_command, recipe_state

        st = recipe_state("julia_dragon", 64, 36)
        st["layers"] = [{"type": "text", "text": "it's \"quoted\"", "fill": "ffffff"}]
        st["post"] = [{"op": "grain", "amount": 0.1}, {"op": "vignette", "value": 0.3}]
        cmd = cli_command(st, 64, 36)
        self.assertIn("--layer", cmd)
        self.assertIn("--effect", cmd)
        import shlex

        from fractalgen.cli import main

        argv = shlex.split(cmd)[3:]
        with tempfile.TemporaryDirectory() as d, mock.patch("sys.stdout", StringIO()):
            self.assertEqual(main(argv + ["--out", d]), 0)


class ServerTests(unittest.TestCase):
    def test_upload_validates(self):
        from fractalgen.gui.server import save_upload
        from fractalgen.loader import CUSTOM_DIR

        with self.assertRaises(ValueError):
            save_upload("x.exe", b"MZ")
        with self.assertRaises(Exception):
            save_upload("x.png", b"not a png")
        import io

        buf = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        rel = save_upload("../../evil name.png", buf.getvalue())
        self.assertTrue(rel.startswith("uploads/") and "/" not in rel[len("uploads/"):])
        (CUSTOM_DIR / rel).unlink()

    def test_reveal_only_exported_files(self):
        from fractalgen.gui.server import Exports, _exported

        with self.assertRaises(ValueError):
            _exported("/etc/hosts", Exports())


if __name__ == "__main__":
    unittest.main()


class ExportQualityTests(unittest.TestCase):
    def test_supersample_validation_and_downsample(self):
        from fractalgen.color import downsample
        from fractalgen.gui.server import _validate_export_body

        st = {"generator": "escape_time"}
        _validate_export_body({"state": st, "width": 64, "height": 36, "supersample": 3, "jpeg_quality": 80})
        for bad in ({"supersample": 5}, {"supersample": "2"}, {"jpeg_quality": 10}, {"width": 9000, "supersample": 2}):
            with self.assertRaises(ValueError):
                _validate_export_body({"state": st, "width": 64, "height": 36, **bad})
        a = np.zeros((4, 4, 4))
        a[:2, :2] = [1.0, 0.0, 0.0, 1.0]  # one opaque red quadrant, the rest transparent
        out = downsample(a, 2)
        np.testing.assert_allclose(out[0, 0], [1, 0, 0, 1])
        np.testing.assert_allclose(out[1, 1], [0, 0, 0, 0])
        b = np.zeros((2, 2, 4))
        b[0, 0] = [1, 1, 1, 1]
        np.testing.assert_allclose(downsample(b, 2)[0, 0], [1, 1, 1, 0.25])  # no dark fringe
