"""Smoke-test every generator/recipe defined in examples/."""

from __future__ import annotations

import unittest
from pathlib import Path

import fractalgen as fg
from fractalgen import testing

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


class ExampleSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fg.bootstrap([EXAMPLES / "ripples_plugin.py", EXAMPLES / "annotated_plugin.py", EXAMPLES / "extensions.toml"])

    def test_ripples_and_annotated_generators(self):
        testing.smoke_generator("ripples")
        testing.smoke_generator("annotated_waves")

    def test_extension_recipes(self):
        for name in (
            "sine_glow",
            "wobble",
            "chen_wings",
            "braided_silk",
            "blossom",
            "ripple_pool",
        ):
            with self.subTest(recipe=name):
                testing.smoke_recipe(name)


if __name__ == "__main__":
    unittest.main()
