"""Regression tests for core loader/recipe/post/palette/registry fixes."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

import fractalgen as fg
from fractalgen.loader import LoadError, load_path, load_plugin, load_toml
from fractalgen.post import apply_post
from fractalgen.palette import hex_to_rgb, register_palette
from fractalgen.registry import Registry
from fractalgen.generators.escape import compile_formula_expr

fg.bootstrap()


class ExtendsLoadTests(unittest.TestCase):
    def test_bad_extends_loaderror_and_recovery(self):
        bad = textwrap.dedent(
            """
            [recipe.bad_child]
            extends = "no_such_parent"
            generator = "escape_time"
            palette = "ember"
            params = { formula = "mandelbrot", max_iter = 40 }
            """
        )
        good = textwrap.dedent(
            """
            [recipe.ok_recipe]
            generator = "escape_time"
            palette = "ember"
            view = { center = [0, 0], half_width = 1.5 }
            params = { formula = "mandelbrot", max_iter = 40 }
            """
        )
        with tempfile.TemporaryDirectory() as d:
            bad_p = Path(d) / "bad.toml"
            good_p = Path(d) / "good.toml"
            bad_p.write_text(bad)
            good_p.write_text(good)
            with self.assertRaises(LoadError) as ctx:
                fg.bootstrap([bad_p])
            self.assertIn("[recipe.bad_child]", str(ctx.exception))
            self.assertIn(str(bad_p.resolve()), str(ctx.exception))
            rgb = fg.render("julia_dragon", 16, 9)
            self.assertEqual(rgb.shape, (9, 16, 3))
            fg.bootstrap([good_p])
            fg.RECIPES.get("ok_recipe")

    def test_load_toml_bad_extends_does_not_poison_render(self):
        src = textwrap.dedent(
            """
            [recipe.broken_child]
            extends = "no_such_parent"
            generator = "escape_time"
            """
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "badext.toml"
            p.write_text(src)
            load_toml(p)
            with self.assertRaises(LoadError) as ctx:
                fg.render("julia_dragon", 16, 9)
            self.assertIn("[recipe.broken_child]", str(ctx.exception))
            rgb = fg.render("julia_dragon", 16, 9)
            self.assertEqual(rgb.shape, (9, 16, 3))

    def test_extends_variant(self):
        src = textwrap.dedent(
            """
            [recipe.var_base]
            generator = "escape_time"
            palette = "ember"
            view = { center = [0, 0], half_width = 1.5 }
            params = { formula = "mandelbrot", max_iter = 40 }

            [recipe.var_base.variants.ice]
            palette = "abyss"

            [recipe.var_child]
            extends = "var_base@ice"
            params = { max_iter = 50 }
            """
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "v.toml"
            p.write_text(src)
            fg.bootstrap([p])
        child = fg.RECIPES.get("var_child")
        self.assertEqual(child.palette, "abyss")
        self.assertEqual(child.params["max_iter"], 50)
        self.assertEqual(child.params["formula"], "mandelbrot")


class LoaderRollbackTests(unittest.TestCase):
    def test_failed_toml_rolls_back(self):
        src = textwrap.dedent(
            """
            [palette.ok_partial]
            stops = ["000000", "ffffff"]

            [palette.bad_partial]
            stops = ["not-a-color"]
            """
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "partial.toml"
            path.write_text(src)
            with self.assertRaises(LoadError) as ctx:
                load_toml(path)
            self.assertIn("[palette.bad_partial]", str(ctx.exception))
            self.assertNotIn("ok_partial", fg.PALETTES)
            self.assertNotIn("bad_partial", fg.PALETTES)

    def test_duplicate_registration_names_file_and_table(self):
        src = textwrap.dedent(
            """
            [palette.ember]
            stops = ["000000", "ffffff"]
            """
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "dup.toml"
            path.write_text(src)
            with self.assertRaises(LoadError) as ctx:
                load_toml(path)
            msg = str(ctx.exception)
            self.assertIn("[palette.ember]", msg)
            self.assertIn(str(path.resolve()), msg)
            self.assertIn("already registered", msg)
            self.assertIn("ember", fg.PALETTES)

    def test_failed_plugin_rolls_back(self):
        src = textwrap.dedent(
            """
            import fractalgen as fg
            fg.register_palette("plugin_partial", {"stops": ["000000", "ffffff"]})
            raise RuntimeError("boom")
            """
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "boom.py"
            path.write_text(src)
            with self.assertRaises(ImportError):
                load_plugin(path)
            self.assertNotIn("plugin_partial", fg.PALETTES)

    def test_skip_unrelated_toml_in_directory(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pyproject.toml").write_text('[project]\nname = "x"\n')
            plug = textwrap.dedent(
                """
                [palette.dir_ok]
                stops = ["000000", "ffffff"]
                """
            )
            (Path(d) / "ext.toml").write_text(plug)
            load_path(d)
            self.assertIn("dir_ok", fg.PALETTES)

    def test_explicit_unknown_section_still_errors(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "pyproject.toml"
            path.write_text('[project]\nname = "x"\n')
            with self.assertRaises(LoadError) as ctx:
                load_toml(path)
            self.assertIn("unknown section", str(ctx.exception))

    def test_missing_bare_path_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_path("definitely_not_a_fractalgen_module_xyz")
        self.assertIn("No such file", str(ctx.exception))


class PostPaletteTests(unittest.TestCase):
    def test_gamma_positive(self):
        rgb = np.ones((4, 4, 3)) * 0.5
        with self.assertRaises(ValueError) as ctx:
            apply_post(rgb, {"gamma": 0})
        self.assertIn("gamma", str(ctx.exception).lower())
        out = apply_post(rgb, {"gamma": 1.0})
        np.testing.assert_allclose(out, rgb)

    def test_palette_mix_needs_palette(self):
        rgb = np.ones((4, 4, 3)) * 0.5
        with self.assertRaises(ValueError) as ctx:
            apply_post(rgb, {"palette_mix": 0.5})
        self.assertIn("palette", str(ctx.exception))


class PaletteValidationTests(unittest.TestCase):
    def test_positions_endpoints(self):
        with self.assertRaises(ValueError):
            register_palette(
                "bad_pos",
                {"stops": ["000000", "ffffff"], "positions": [0.1, 1.0]},
                replace=True,
            )
        with self.assertRaises(ValueError):
            register_palette(
                "bad_oor",
                {"stops": ["000000", "ff0000", "ffffff"], "positions": [-1, 0.5, 2]},
                replace=True,
            )

    def test_bad_hex_message(self):
        with self.assertRaises(ValueError) as ctx:
            hex_to_rgb("gggggg")
        self.assertIn("Bad hex color 'gggggg'", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            hex_to_rgb("#ggg")
        self.assertIn("Bad hex color", str(ctx.exception))


class RegistryHintTests(unittest.TestCase):
    def test_kind_hints(self):
        cases = [
            ("palette", "fractalgen palettes"),
            ("generator", "fractalgen generators"),
            ("recipe", "fractalgen list"),
            ("post op", "fractalgen post"),
            ("preset", "fractalgen presets"),
            ("escape-time formula", "fractalgen list"),
        ]
        for kind, hint in cases:
            r: Registry[str] = Registry(kind)
            r.register("a", "x")
            with self.assertRaises(KeyError) as ctx:
                r.get("zzz")
            self.assertIn(hint, str(ctx.exception), kind)


class ViewOverrideTests(unittest.TestCase):
    def test_rotate_on_non_view_generator(self):
        r = fg.RECIPES.get("gray_scott_coral")
        with self.assertRaises(TypeError) as ctx:
            r.resolve(64, 36, {"view": {"rotation": 10}})
        self.assertIn("no view framing", str(ctx.exception))
        self.assertNotIn("remove 'view'", str(ctx.exception))

    def test_recipe_view_still_blames_recipe(self):
        r = fg.Recipe("has_view", "sandpile", view={"rotation": 10})
        with self.assertRaises(TypeError) as ctx:
            r.resolve(64, 36)
        self.assertIn("remove 'view' from 'has_view'", str(ctx.exception))


class FormulaCompileTests(unittest.TestCase):
    def test_rejects_lambda(self):
        with self.assertRaises(ValueError):
            compile_formula_expr("(lambda z: z)(z)")
        with self.assertRaises(ValueError):
            compile_formula_expr("(lambda: 1)()")

    def test_abs_and_re_still_compile(self):
        abs_fn = compile_formula_expr("abs(z)")
        self.assertAlmostEqual(abs_fn(3 + 4j, 0j, 0j, 0), 5.0)
        re_fn = compile_formula_expr("re(z)")
        self.assertAlmostEqual(re_fn(3 + 4j, 0j, 0j, 0).real, 3.0)


if __name__ == "__main__":
    unittest.main()


class RegistryOverrideTests(unittest.TestCase):
    def test_dropping_an_override_restores_the_shadowed_item(self):
        from fractalgen.registry import Registry

        r: Registry[int] = Registry("thing")
        r.register("a", 1, source="builtin")
        r.register("a", 2, replace=True, source="user.toml")
        self.assertEqual(r.get("a"), 2)
        self.assertEqual(r.drop_from("user.toml"), ["a"])
        self.assertEqual(r.get("a"), 1)
        self.assertEqual(r.source_of("a"), "builtin")
        r.register("b", 3, source="user.toml")
        r.drop_from("user.toml")
        self.assertNotIn("b", r)
