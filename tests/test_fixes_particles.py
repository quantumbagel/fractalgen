"""Regression tests for particle / line generator edge-case fixes."""

from __future__ import annotations

import unittest

import numpy as np

import fractalgen as fg

fg.bootstrap()

_IFS_RECIPES = (
    "barnsley_fern",
    "heighway_dragon",
    "maple_leaf",
    "ifs_spiral",
)


def _gen(gen: str, w: int = 64, h: int = 36, **params) -> np.ndarray:
    return fg.GENERATORS.get(gen)(fg.RenderContext(w, h), **params)


class IfsFixes(unittest.TestCase):
    def test_extreme_batch_warmup_not_blank(self):
        rgb = _gen(
            "ifs",
            80,
            45,
            n=500_000,
            batch=100_000,
            warmup_steps=8,
        )
        self.assertTrue(np.isfinite(rgb).all())
        self.assertGreater(float(rgb.max()), 0.05)

    def test_warmup_zero_single_step_deposits(self):
        rgb = _gen("ifs", 80, 45, n=10_000, batch=40_000, warmup_steps=0)
        self.assertGreater(float(rgb.max()), 0.05)

    def test_empty_maps_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("ifs", maps=[], n=8000, batch=800, warmup_steps=0)
        self.assertIn("maps", str(ctx.exception))

    def test_single_map_assigns_every_walker(self):
        # Probability is unused when there is only one map; both must match.
        kw = dict(
            n=8000,
            batch=800,
            warmup_steps=0,
            xmin=-1.0,
            xmax=20.0,
            ymin=-1.0,
            ymax=5.0,
            blur=0.0,
        )
        trans = (1.0, 0.0, 0.0, 1.0, 1.0, 0.0)
        a = _gen("ifs", 48, 24, maps=(trans + (0.3,),), **kw)
        b = _gen("ifs", 48, 24, maps=(trans + (1.0,),), **kw)
        np.testing.assert_array_equal(a, b)
        self.assertGreater(float(a.max()), 0.05)

    def test_batch_zero_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("ifs", n=1000, batch=0)
        self.assertIn("batch", str(ctx.exception))

    def test_batch_fraction_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("ifs", n=1000, batch=0.4)
        self.assertIn("batch", str(ctx.exception))

    def test_builtin_ifs_recipes_finite_at_160x90(self):
        for name in _IFS_RECIPES:
            with self.subTest(recipe=name):
                rgb = fg.RECIPES.get(name).render(160, 90)
                self.assertEqual(rgb.shape, (90, 160, 3))
                self.assertTrue(np.isfinite(rgb).all())
                self.assertGreater(float(rgb.std()), 1e-4)


class KernelFixes(unittest.TestCase):
    def test_poincare_1x1_finite(self):
        from fractalgen import kernels

        done = np.zeros(1, dtype=np.int8)
        depth, edge, sector = kernels.poincare_kernel(1, 1, 7, 3, 4, done)
        self.assertTrue(np.isfinite(depth).all())
        self.assertTrue(np.isfinite(edge).all())
        self.assertTrue(np.isfinite(sector).all())

    def test_deposit_kernel_keeps_legacy_border_hits(self):
        from fractalgen import kernels

        d = np.zeros((10, 10))
        hue = np.zeros((10, 10))
        kernels.deposit_kernel(
            d,
            hue,
            np.array([-0.4, 9.4]),
            np.array([9.4, -0.4]),
            0.0,
            9.0,
            0.0,
            9.0,
            1.0,
        )
        # Toward-zero int() lands a sliver of out-of-window samples on the border.
        # Rejecting them first changes maple_leaf / hopalong / ikeda pixels.
        self.assertEqual(float(d.sum()), 2.0)
        self.assertEqual(float(d[0, 0]), 1.0)
        self.assertEqual(float(d[9, 9]), 1.0)


class CausticsFixes(unittest.TestCase):
    def test_k_lo_gt_k_hi_renders(self):
        rgb = _gen("caustics", 80, 45, waves=4, rays=1, k_lo=20.0, k_hi=2.0)
        self.assertTrue(np.isfinite(rgb).all())

    def test_k_lo_zero_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("caustics", k_lo=0.0, k_hi=2.0)
        self.assertIn("positive", str(ctx.exception))


class FlameFixes(unittest.TestCase):
    def test_single_variation_pool(self):
        rgb = _gen(
            "flame",
            quality=2,
            walkers=200,
            pool=["linear"],
        )
        self.assertTrue(np.isfinite(rgb).all())

    def test_empty_pool_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("flame", quality=2, walkers=200, pool=[])
        self.assertIn("pool", str(ctx.exception))

    def test_walkers_zero_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen(
                "flame",
                walkers=0,
                quality=2,
                xforms=[
                    {
                        "affine": [0.6, -0.4, 0.4, 0.6, 0.2, 0],
                        "weight": 1,
                        "color": 0,
                        "variations": {"linear": 1},
                    }
                ],
            )
        self.assertIn("walkers", str(ctx.exception))

    def test_walkers_fraction_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("flame", walkers=0.4, quality=2, n_xforms=1)
        self.assertIn("walkers", str(ctx.exception))

    def test_huge_seed_renders(self):
        rgb = _gen("flame", seed=2**40, quality=2, walkers=500, n_xforms=1)
        self.assertTrue(np.isfinite(rgb).all())


class CinematicFixes(unittest.TestCase):
    def test_gray_scott_wrong_phase_count_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen(
                "gray_scott_blend",
                32,
                18,
                phases=((0.03, 0.06),),
                steps=20,
                sim_min=32,
            )
        self.assertIn("three phases", str(ctx.exception))


class DynamicsCurvesFixes(unittest.TestCase):
    def test_n_magnets_zero_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("magnetic_pendulum", 40, 24, n_magnets=0, steps=4)
        self.assertIn("n_magnets", str(ctx.exception))

    def test_n_curves_zero_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("harmonograph", 40, 24, n_curves=0, samples=80)
        self.assertIn("n_curves", str(ctx.exception))

    def test_empty_ratios_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("harmonograph", 40, 24, ratios=(), n_curves=2)
        self.assertIn("ratios", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
