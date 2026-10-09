"""Regression tests for CLI / view / params / engine fixes (tiny renders only)."""

from __future__ import annotations

import math
import tempfile
import textwrap
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

from fractalgen.cli import _split_preset_groups, _sweep_overrides, main
from fractalgen.engine import PRESETS, _output_stem, render_batch
from fractalgen.params import coerce, parse_literal, to_complex
from fractalgen.view import View, angle


class ViewValidationTests(unittest.TestCase):
    def test_zoom_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "zoom"):
            View.from_dict({"zoom": 0})

    def test_half_width_nonpositive_raises(self):
        with self.assertRaisesRegex(ValueError, "half_width"):
            View.from_dict({"half_width": 0})

    def test_plane_1x1_samples_center(self):
        re_, im_ = View(center=0.5 + 0.25j, half_width=2.0).plane(1, 1)
        self.assertAlmostEqual(re_[0, 0], 0.5)
        self.assertAlmostEqual(im_[0, 0], 0.25)

    def test_to_pixel_1px_invertible(self):
        for w, h in ((1, 1), (1, 64), (64, 1)):
            ax, bx, _cx, ay, by, _cy = View(center=0j, half_width=1.5).to_pixel(w, h)
            self.assertTrue(all(math.isfinite(x) for x in (ax, bx, ay, by)))
            self.assertNotAlmostEqual(ax * by - bx * ay, 0.0)

    def test_plane_two_px_unchanged(self):
        re_, im_ = View(center=1 + 1j, half_width=2.0).plane(8, 4)
        np.testing.assert_array_equal(re_[0], np.linspace(-1.0, 3.0, 8))
        np.testing.assert_array_equal(im_[:, 0], np.linspace(2.0, 0.0, 4))

    def test_plane_1_wide_is_center(self):
        re_, im_ = View(center=0.25 + 0.5j, half_width=2.0).plane(1, 5)
        self.assertTrue(np.allclose(re_, 0.25))
        np.testing.assert_array_equal(im_[:, 0], np.linspace(0.5 + 10.0, 0.5 - 10.0, 5))


class ParamsLiteralTests(unittest.TestCase):
    def test_parse_inf_nan(self):
        self.assertEqual(parse_literal("inf"), float("inf"))
        self.assertTrue(math.isnan(parse_literal("nan")))

    def test_to_complex_inf(self):
        self.assertEqual(to_complex("inf"), complex(float("inf"), 0.0))

    def test_coerce_int_rounds_fraction(self):
        # The GUI's Nudge and slider math send floats for integer knobs.
        self.assertEqual(coerce(37948.59313393296, 480), 37949)
        self.assertEqual(coerce("10.4", 100), 10)
        self.assertIsInstance(coerce(10.6, 100), int)

    def test_coerce_int_rejects_inf(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            coerce(float("inf"), 3)


class AngleTests(unittest.TestCase):
    def test_two_pi(self):
        self.assertAlmostEqual(angle("2pi"), 2 * math.pi)

    def test_bad_angle_value_error(self):
        with self.assertRaises(ValueError):
            angle("pi pi")

    def test_pi_over_zero_value_error(self):
        with self.assertRaises(ValueError):
            angle("pi/0")


class CliRenderTests(unittest.TestCase):
    def setUp(self):
        import fractalgen as fg

        fg.bootstrap()

    def test_size_with_preset_uses_explicit_size_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            toml = Path(tmp) / "tiny.toml"
            toml.write_text(
                textwrap.dedent(
                    """\
                    [preset.tinyqa]
                    width = 24
                    height = 14
                    tags = []
                    """
                )
            )
            out = Path(tmp) / "out"
            err = StringIO()
            with mock.patch("sys.stderr", err):
                code = main(
                    [
                        "--load",
                        str(toml),
                        "render",
                        "julia_dragon",
                        "--preset",
                        "tinyqa",
                        "--size",
                        "16x9",
                        "--out",
                        str(out),
                        "--no-isolate",
                        "--no-lock",
                        "--jobs",
                        "1",
                    ]
                )
            self.assertEqual(code, 0)
            png = out / "julia_dragon.png"
            self.assertTrue(png.exists())
            self.assertFalse((out / "16x9").exists())
            self.assertFalse((out / "24x14").exists())
            from PIL import Image

            with Image.open(png) as im:
                self.assertEqual(im.size, (16, 9))

    def test_zoom_zero_cli_error(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(["render", "julia_dragon", "--size", "64x36", "--zoom", "0", "--no-isolate", "--no-lock", "--jobs", "1"])
        self.assertEqual(code, 2)
        self.assertIn("error:", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_unknown_preset_message(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(["render", "--preset", "nosuchpreset", "--size", "16x9", "--no-lock", "--jobs", "1", "--out", "/tmp/x"])
        self.assertEqual(code, 2)
        self.assertIn("Unknown preset", err.getvalue())

    def test_size_rejects_zero(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(["render", "julia_dragon", "--size", "0x0", "--no-lock", "--jobs", "1", "--out", "/tmp/x"])
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err.getvalue())

    def test_timeout_rejects_zero(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(["render", "julia_dragon", "--size", "16x9", "--timeout", "0", "--no-lock", "--jobs", "1", "--out", "/tmp/x"])
        self.assertEqual(code, 2)

    def test_timeout_rejects_nan(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(["render", "julia_dragon", "--size", "16x9", "--timeout", "nan", "--no-lock", "--jobs", "1", "--out", "/tmp/x"])
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err.getvalue())

    def test_zoom_half_width_exclusive(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(
                [
                    "render",
                    "julia_dragon",
                    "--size",
                    "16x9",
                    "--zoom",
                    "2",
                    "--half-width",
                    "1.5",
                    "--no-lock",
                    "--jobs",
                    "1",
                    "--out",
                    "/tmp/x",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("half-width", err.getvalue().lower())

    def test_load_after_subcommand(self):
        ext = Path(__file__).resolve().parents[1] / "examples" / "extensions.toml"
        with tempfile.TemporaryDirectory() as tmp:
            code = main(
                [
                    "render",
                    "--load",
                    str(ext),
                    "sine_glow",
                    "--size",
                    "8x8",
                    "--out",
                    tmp,
                    "--no-isolate",
                    "--no-lock",
                    "--jobs",
                    "1",
                ]
            )
        self.assertEqual(code, 0)

    def test_load_equals_after_subcommand(self):
        ext = Path(__file__).resolve().parents[1] / "examples" / "extensions.toml"
        with tempfile.TemporaryDirectory() as tmp:
            code = main(
                [
                    "render",
                    f"--load={ext}",
                    "sine_glow",
                    "--size",
                    "8x8",
                    "--out",
                    tmp,
                    "--no-isolate",
                    "--no-lock",
                    "--jobs",
                    "1",
                ]
            )
        self.assertEqual(code, 0)

    def test_sweep_unknown_dotted_key(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(
                [
                    "render",
                    "clifford_gold",
                    "--size",
                    "32x18",
                    "--sweep",
                    "coeffs.a=-1.4,0",
                    "--out",
                    "/tmp/sw",
                    "--no-isolate",
                    "--no-lock",
                    "--jobs",
                    "1",
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("coeffs", err.getvalue())

    def test_sweep_zoom_maps_to_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sw"
            code = main(
                [
                    "render",
                    "julia_dragon",
                    "--size",
                    "32x18",
                    "--sweep",
                    "zoom=1,2",
                    "--out",
                    str(out),
                    "--no-isolate",
                    "--no-lock",
                    "--jobs",
                    "1",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out / "julia_dragon@zoom=1.png").exists())
            self.assertTrue((out / "julia_dragon@zoom=2.png").exists())

    def test_set_max_iter_inf_errors(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(
                [
                    "render",
                    "julia_dragon",
                    "--size",
                    "16x9",
                    "--set",
                    "max_iter=inf",
                    "--out",
                    "/tmp/inf",
                    "--no-isolate",
                    "--no-lock",
                    "--jobs",
                    "1",
                ]
            )
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err.getvalue())

    def test_gui_port_range(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(["gui", "--port", "65536", "--no-browser"])
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err.getvalue())

    def test_try_1x1_caustics_and_symplectic(self):
        with tempfile.TemporaryDirectory() as tmp:
            for gen, extra in (
                ("caustics", ["--set", "waves=4", "--set", "rays=1"]),
                ("symplectic", ["--set", "orbits=10", "--set", "steps=50"]),
            ):
                out = Path(tmp) / f"{gen}.png"
                err = StringIO()
                with mock.patch("sys.stderr", err):
                    code = main(["try", gen, "--size", "1x1", "-o", str(out), *extra])
                self.assertEqual(code, 0, err.getvalue())
                self.assertTrue(out.exists())
                self.assertNotIn("Traceback", err.getvalue())


class PresetSplitTests(unittest.TestCase):
    def test_unknown_preset_raises(self):
        keys = set(PRESETS.names())
        with self.assertRaises(KeyError):
            _split_preset_groups([["nosuchpreset"]], keys)


class EngineStemTests(unittest.TestCase):
    def test_traversal_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            with self.assertRaises(ValueError):
                render_batch(["../evil"], 8, 4, out, isolate=False)

    def test_safe_name_writes_inside_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            # Recipe lookup would fail; we only test stem sanitization via ValueError on bad names.
            with self.assertRaises(ValueError):
                render_batch(["foo/bar"], 8, 4, out, isolate=False)

    def test_variant_stem_kept(self):
        self.assertEqual(_output_stem("sine_glow@cold"), "sine_glow@cold")


class SweepTagTests(unittest.TestCase):
    def test_illegal_chars_replaced(self):
        tags = [t for t, _ in _sweep_overrides(("formula", "foo/bar,a:b"), {})]
        self.assertEqual(tags, ["formula=foo_bar", "formula=a_b"])

    def test_params_prefix_kept(self):
        tag, ov = _sweep_overrides(("params.n", [1000]), {})[0]
        self.assertEqual(tag, "n=1000")
        self.assertEqual(ov["params"]["n"], 1000)
