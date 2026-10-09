"""CLI tests (fast; tiny renders only)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from fractalgen.cli import _split_preset_groups, main


class CliErrorTests(unittest.TestCase):
    def test_unknown_recipe_suggests(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(["show", "julia_dragoon"])
        self.assertEqual(code, 2)
        self.assertIn("Did you mean", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_bad_override_value(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(["render", "julia_dragon", "--size", "64x36", "--rotate", "__bad__"])
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err.getvalue())

    def test_render_needs_size_message(self):
        err = StringIO()
        with mock.patch("sys.stderr", err):
            code = main(["render", "julia_dragon"])
        self.assertEqual(code, 2)
        self.assertIn("presets", err.getvalue())

    def test_no_subcommand_quickstart(self):
        out = StringIO()
        with mock.patch("sys.stdout", out):
            code = main([])
        self.assertEqual(code, 0)
        self.assertIn("Quick start", out.getvalue())

    def test_version(self):
        import fractalgen as fg

        out = StringIO()
        with mock.patch("sys.stdout", out):
            code = main(["--version"])
        self.assertEqual(code, 0)
        self.assertIn(fg.__version__, out.getvalue())


class CliListTests(unittest.TestCase):
    def test_names_one_per_line(self):
        out = StringIO()
        with mock.patch("sys.stdout", out):
            code = main(["list", "--names"])
        self.assertEqual(code, 0)
        lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
        self.assertGreater(len(lines), 50)
        self.assertEqual(len(lines), len(set(lines)))

    def test_search_dragon(self):
        out = StringIO()
        with mock.patch("sys.stdout", out):
            main(["list", "--search", "dragon", "--names"])
        names = out.getvalue().splitlines()
        self.assertTrue(any("dragon" in n for n in names))

    def test_json_shape(self):
        out = StringIO()
        with mock.patch("sys.stdout", out):
            main(["list", "--json", "--search", "julia_dragon"])
        rows = json.loads(out.getvalue())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "julia_dragon")
        self.assertIn("generator", rows[0])


class PresetSplitTests(unittest.TestCase):
    def test_swallows_only_known_presets(self):
        keys = {"4k", "8k", "iphone"}
        presets, extra = _split_preset_groups([["8k", "julia_dragon"]], keys)
        self.assertEqual(presets, ["8k"])
        self.assertEqual(extra, ["julia_dragon"])

    def test_multiple_presets_in_one_flag(self):
        keys = {"8k", "iphone"}
        presets, extra = _split_preset_groups([["8k", "iphone"]], keys)
        self.assertEqual(presets, ["8k", "iphone"])
        self.assertEqual(extra, [])


class CliNewTests(unittest.TestCase):
    def test_new_generator_loads_and_renders(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = StringIO()
            with mock.patch("sys.stdout", out):
                code = main(
                    [
                        "new",
                        "generator",
                        "cli_test_gen",
                        "--dir",
                        str(tmp_path),
                        "--view",
                    ]
                )
            self.assertEqual(code, 0)
            png = tmp_path / "out.png"
            err = StringIO()
            with mock.patch("sys.stderr", err), mock.patch("sys.stdout", StringIO()):
                code = main(
                    [
                        "--load",
                        str(tmp_path),
                        "render",
                        "cli_test_gen",
                        "--size",
                        "64x36",
                        "--out",
                        str(tmp_path),
                        "--no-isolate",
                        "--no-lock",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue((tmp_path / "cli_test_gen.png").exists())


class CliCompletionTests(unittest.TestCase):
    def test_completion_bash_mentions_render(self):
        out = StringIO()
        with mock.patch("sys.stdout", out):
            main(["completion", "bash"])
        self.assertIn("render", out.getvalue())
        self.assertIn("list --names", out.getvalue())


if __name__ == "__main__":
    unittest.main()
