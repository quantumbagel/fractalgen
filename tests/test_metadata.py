"""Recipes embedded in rendered images: python -m unittest tests.test_metadata"""

from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import fractalgen as fg
from fractalgen.cli import main
from fractalgen.color import read_meta

fg.bootstrap()

_OV = ["--center", "0.3+0.1j", "--zoom", "1.5", "--set", "max_iter=300", "--palette", "aurora",
       "--post", "saturation=1.3", "--mask", "text=HI"]


def _quiet(argv: list[str]) -> int:
    with mock.patch("sys.stdout", StringIO()), mock.patch("sys.stderr", StringIO()):
        return main(argv)


class ImageMetadataTests(unittest.TestCase):
    def test_png_and_jpeg_round_trip(self):
        rgb = np.random.default_rng(0).random((9, 16, 3))
        meta = {"fractalgen.recipe": '[recipe.x]\ngenerator = "escape_time"\n', "Software": "fractalgen"}
        with tempfile.TemporaryDirectory() as d:
            for ext in ("png", "jpg"):
                p = fg.save_image(Path(d) / f"a.{ext}", rgb, meta=meta)
                self.assertEqual(read_meta(p), meta)
            self.assertEqual(read_meta(fg.save_image(Path(d) / "plain.png", rgb)), {})

    def test_fork_from_image_reproduces_the_render(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for size, sub in (("64x36", "land"), ("36x64", "port")):
                self.assertEqual(_quiet(["render", "julia_dragon", "--size", size, "--out", str(d / sub), "--no-isolate", *_OV]), 0)
            toml = d / "forked.toml"
            self.assertEqual(_quiet(["fork", str(d / "land" / "julia_dragon.png"), "forked_img", "-o", str(toml)]), 0)
            self.assertIn('tone_from = "julia_dragon"', toml.read_text())
            for size, sub in (("64x36", "land"), ("36x64", "port")):
                out = d / f"re_{sub}"
                self.assertEqual(_quiet(["--load", str(toml), "render", "forked_img", "--size", size, "--out", str(out),
                                         "--no-isolate"]), 0)
                a = np.asarray(Image.open(d / sub / "julia_dragon.png"))
                b = np.asarray(Image.open(out / "forked_img.png"))
                np.testing.assert_array_equal(a, b)

    def test_fork_from_image_without_metadata_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = fg.save_image(Path(d) / "plain.png", np.zeros((4, 4, 3)))
            err = StringIO()
            with mock.patch("sys.stderr", err):
                self.assertEqual(main(["fork", str(p), "x", "--stdout"]), 2)
            self.assertIn("no fractalgen recipe embedded", err.getvalue())


if __name__ == "__main__":
    unittest.main()
