"""Framework unit tests: python -m unittest discover tests"""

from __future__ import annotations

import math
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

import fractalgen as fg
from fractalgen.expr import compile_expr
from fractalgen.params import coerce, parse_literal, to_complex
from fractalgen.registry import Registry
from fractalgen.view import View, angle

fg.bootstrap()


class RegistryTests(unittest.TestCase):
    def test_duplicate_and_unknown(self):
        r: Registry[int] = Registry("thing")
        r.register("a", 1)
        with self.assertRaises(KeyError):
            r.register("a", 2)
        r.register("a", 3, replace=True)
        self.assertEqual(r.get("a"), 3)
        with self.assertRaisesRegex(KeyError, "Known: a"):
            r.get("b")


class ParamTests(unittest.TestCase):
    def test_complex_forms(self):
        for v in ([-0.8, 0.156], "-0.8+0.156j", "-0.8+0.156i", {"re": -0.8, "im": 0.156}):
            self.assertEqual(to_complex(v), complex(-0.8, 0.156))

    def test_coerce_by_default_type(self):
        self.assertEqual(coerce("12", 3), 12)
        self.assertIsInstance(coerce(2, 1.5), float)
        self.assertEqual(coerce([1, 2], 0j), 1 + 2j)
        self.assertIs(coerce("yes", False), True)
        self.assertEqual(parse_literal("[1, 2]"), [1, 2])
        self.assertEqual(parse_literal("-0.8+0.156j"), complex(-0.8, 0.156))
        self.assertEqual(parse_literal("ember"), "ember")


class AngleAndViewTests(unittest.TestCase):
    def test_angle_units(self):
        self.assertAlmostEqual(angle(90), math.pi / 2)
        self.assertAlmostEqual(angle("0.72rad"), 0.72)
        self.assertAlmostEqual(angle("pi/4"), math.pi / 4)
        self.assertAlmostEqual(angle("45deg"), math.pi / 4)
        with self.assertRaises(ValueError):
            angle("__import__('os')")

    def test_unrotated_plane_matches_linspace(self):
        re_, im_ = View(center=1 + 1j, half_width=2.0).plane(8, 4)
        np.testing.assert_array_equal(re_[0], np.linspace(-1.0, 3.0, 8))
        np.testing.assert_array_equal(im_[:, 0], np.linspace(2.0, 0.0, 4))

    def test_rotation_quarter_turn(self):
        v = View(center=0j, half_width=1.0, rotation=90)
        re_, im_ = v.plane(5, 5)
        # The right edge of the unrotated frame lands on the top (+imag) side.
        self.assertAlmostEqual(re_[2, 4], 0.0, places=12)
        self.assertAlmostEqual(im_[2, 4], 1.0, places=12)

    def test_cover_keeps_vertical_zoom(self):
        hw, hh = View(half_width=1.6, fit="cover").extents(900, 1600)
        self.assertAlmostEqual(hh, 1.6 / (16 / 9))
        self.assertAlmostEqual(hw, hh * 900 / 1600)


class ReferenceFrameTests(unittest.TestCase):
    def test_reference_is_4k_at_any_size(self):
        for w, h in ((720, 405), (3840, 2160), (7680, 4320)):
            ctx = fg.RenderContext(w, h)
            self.assertEqual(ctx.ref_size, (3840, 2160))
            self.assertAlmostEqual(ctx.res_scale, w / 3840)
            self.assertEqual(ctx.sim_grid(0.28), (1075, 604))
        phone = fg.RenderContext(1290, 2796)
        rw, rh = phone.ref_size
        self.assertAlmostEqual(rw * rh / (3840 * 2160), 1.0, places=2)
        self.assertGreater(rh, rw)

    def test_density_norm(self):
        ref = fg.RenderContext(3840, 2160)
        self.assertEqual(ref.density_norm(5000, 5000), 1.0)
        small = fg.RenderContext(960, 540)  # 1/16 the pixels
        # Same particle count on 1/16 the pixels = 16x the hits per pixel.
        self.assertAlmostEqual(small.density_norm(5000, 5000), 1 / 16)


class PaletteTests(unittest.TestCase):
    def test_specs(self):
        ember = fg.get_palette("ember")
        np.testing.assert_allclose(fg.get_palette("ember_r").sample(np.array([0.0]))[0], ember.stops[-1])
        inline = fg.get_palette(["000000", "ffffff"])
        np.testing.assert_allclose(inline.sample(np.array([0.5]))[0], [0.5, 0.5, 0.5])
        positioned = fg.get_palette({"stops": ["000", "fff", "fff"], "positions": [0, 0.25, 1]})
        np.testing.assert_allclose(positioned.sample(np.array([0.25]))[0], [1, 1, 1])

    def test_cyclic_wraps(self):
        pal = fg.get_palette("cosmic")
        np.testing.assert_allclose(pal.sample(np.array([1.25]), cyclic=True), pal.sample(np.array([0.25])))


class ExprTests(unittest.TestCase):
    def test_compile_and_sandbox(self):
        f = compile_expr("sin(a*y) + c*cos(a*x)", ["x", "y"], ["a", "c"])
        self.assertAlmostEqual(float(f(0.0, 1.0, a=2.0, c=1.0)), math.sin(2.0) + 1.0)
        for bad in ("x.__class__", "__import__('os')", "open('f')"):
            with self.assertRaises(ValueError):
                compile_expr(bad, ["x"])


class GeneratorTests(unittest.TestCase):
    def test_bind_rejects_typos_with_hint(self):
        spec = fg.GENERATORS.get("escape_time")
        with self.assertRaisesRegex(TypeError, "did you mean 'max_iter'"):
            spec.bind({"max_iters": 10})

    def test_custom_generator_and_recipe(self):
        @fg.generator("test_gradient", category="test", uses_view=True, replace=True)
        def test_gradient(ctx, *, palette="cosmic", power: float = 1.0):
            """Horizontal gradient."""
            re_, _ = ctx.plane()
            t = (re_ - re_.min()) / (np.ptp(re_) or 1.0)
            return fg.sample_palette(palette, t**power)

        r = fg.Recipe("grad", "test_gradient", params={"power": 2}, palette="ember")
        rgb = r.render(16, 9)
        self.assertEqual(rgb.shape, (9, 16, 3))
        np.testing.assert_allclose(rgb[0, 0], fg.get_palette("ember").stops[0])


class LoaderTests(unittest.TestCase):
    def test_toml_extension_file(self):
        src = textwrap.dedent(
            """
            [formula.test_cubic]
            z = "z**3 + c"
            degree = 3

            [palette.test_duo]
            stops = ["102030", "f0e0d0"]

            [recipe.test_base]
            generator = "escape_time"
            palette = "test_duo"
            view = { center = [0, 0], half_width = 1.5 }
            params = { formula = "test_cubic", max_iter = 40 }
            portrait.view = { half_width = 0.9 }

            [recipe.test_child]
            extends = "test_base"
            params = { max_iter = 60 }
            view = { rotation = 15 }
            """
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ext.toml"
            p.write_text(src)
            fg.bootstrap([p])
        child = fg.RECIPES.get("test_child")
        spec, params, view = child.resolve(32, 18)
        self.assertEqual(params["max_iter"], 60)
        self.assertEqual(params["formula"], "test_cubic")
        self.assertEqual(view.rotation, 15)
        _, _, pview = child.resolve(18, 32)
        self.assertEqual((pview.half_width, pview.fit), (0.9, "cover"))
        rgb = child.render(32, 18)
        self.assertTrue(np.isfinite(rgb).all())


if __name__ == "__main__":
    unittest.main()
