"""Smoke-render every built-in recipe at thumbnail size, landscape and portrait."""

from __future__ import annotations

import unittest

import numpy as np

import fractalgen as fg

fg.bootstrap()


def setUpModule():
    # Simulations run on a grid sized from the reference frame; a small reference
    # keeps the smoke test fast (it checks that recipes render, not their look).
    global _saved_reference
    _saved_reference = fg.context.REFERENCE_PIXELS
    fg.context.REFERENCE_PIXELS = 480 * 270


def tearDownModule():
    fg.context.REFERENCE_PIXELS = _saved_reference


class RecipeSmokeTests(unittest.TestCase):
    def test_every_recipe_renders(self):
        for name in fg.RECIPES.names():
            if not fg.RECIPES.get(name).source.startswith(str(fg.loader.DATA_DIR)):
                continue
            for w, h in ((96, 54), (54, 117)):
                with self.subTest(recipe=name, size=f"{w}x{h}"):
                    rgb = fg.RECIPES.get(name).render(w, h)
                    self.assertEqual(rgb.shape, (h, w, 3))
                    self.assertTrue(np.isfinite(rgb).all())
                    self.assertGreaterEqual(float(rgb.min()), 0.0)
                    self.assertLessEqual(float(rgb.max()), 1.0)
                    self.assertGreater(float(rgb.std()), 0.0, "blank image")


if __name__ == "__main__":
    unittest.main()
