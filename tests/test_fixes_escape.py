"""Regression tests for domain / newton / lyapunov / escape-time edge-case fixes."""

from __future__ import annotations

import unittest

import numpy as np

import fractalgen as fg
from fractalgen.generators.domain import _color_domain
from fractalgen.generators.escape import compile_formula_expr, register_formula

fg.bootstrap()


def _gen(name: str, w: int = 16, h: int = 9, **params) -> np.ndarray:
    return fg.GENERATORS.get(name)(fg.RenderContext(w, h), **params)


def _color_defaults(**overrides):
    kw = dict(
        contour_lo=0.55,
        contour_hi=0.45,
        contour_period=0.6931471805599453,
        checker_scale=6.0,
        checker_lo=0.88,
        checker_hi=0.12,
        mag_pow=0.15,
        mag_scale=1.35,
        sat_lo=0.55,
        sat_hi=0.4,
        sat_mag_scale=1.0,
    )
    kw.update(overrides)
    return kw


class DomainColorFixes(unittest.TestCase):
    def test_nonfinite_w_colors_finitely(self):
        rgb = _color_domain(np.array([[np.inf + 0j]]), **_color_defaults())
        self.assertEqual(rgb.shape, (1, 1, 3))
        self.assertTrue(np.isfinite(rgb).all())

        rgb_nan = _color_domain(np.array([[np.nan + 0j]]), **_color_defaults())
        self.assertTrue(np.isfinite(rgb_nan).all())

    def test_finite_pixels_unchanged_when_neighbor_is_pole(self):
        w = np.array([[1.0 + 0.5j, np.inf + 0j]], dtype=np.complex128)
        rgb = _color_domain(w, **_color_defaults())
        only_finite = _color_domain(np.array([[1.0 + 0.5j]]), **_color_defaults())
        np.testing.assert_array_equal(rgb[:, :1], only_finite)
        self.assertTrue(np.isfinite(rgb).all())

    def test_domain_exp_odd_size_is_finite(self):
        rgb = fg.render("domain_exp", 49, 27)
        self.assertEqual(rgb.shape, (27, 49, 3))
        self.assertTrue(np.isfinite(rgb).all())

    def test_domain_sin_inv_odd_size_is_finite(self):
        rgb = fg.render("domain_sin_inv", 7, 3)
        self.assertTrue(np.isfinite(rgb).all())

    def test_contour_period_non_positive_raises(self):
        for period in (0, 0.0, -1, -0.5):
            with self.subTest(contour_period=period):
                with self.assertRaises(ValueError) as ctx:
                    _gen("domain_color", kind="sin", contour_period=period)
                self.assertIn("contour_period", str(ctx.exception))

    def test_domain_exp_description_matches_function(self):
        desc = fg.RECIPES.get("domain_exp").description
        self.assertIn("exp(1/z)", desc)
        self.assertNotIn("z*exp(z)", desc)


class NewtonFixes(unittest.TestCase):
    def test_power_less_than_two_raises(self):
        for power in (0, 1, -3):
            with self.subTest(power=power):
                with self.assertRaises(ValueError) as ctx:
                    _gen("newton", power=power, max_iter=8)
                self.assertIn("power", str(ctx.exception))

    def test_short_hues_cycle_with_modulo(self):
        a = _gen("newton", power=4, style="cinematic", hues=[0.2, 0.4], max_iter=12)
        b = _gen("newton", power=4, style="cinematic", hues=[0.2, 0.4, 0.2, 0.4], max_iter=12)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isfinite(a).all())

    def test_empty_hues_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("newton", power=5, style="cinematic", hues=[], max_iter=8)
        self.assertIn("hues", str(ctx.exception))

    def test_newton_poly_empty_hues_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("newton_poly", hues=[], max_iter=8)
        self.assertIn("hues", str(ctx.exception))

    def test_quintic_cinematic_short_hues_no_indexerror(self):
        rgb = fg.render(
            "newton_quintic", 16, 9, params={"style": "cinematic", "hues": [0.1, 0.2]}
        )
        self.assertEqual(rgb.shape, (9, 16, 3))
        self.assertTrue(np.isfinite(rgb).all())


class LyapunovFixes(unittest.TestCase):
    def test_sequence_case_insensitive(self):
        kw = dict(warmup=10, samples=20)
        ab = _gen("lyapunov", 24, 14, sequence="AB", **kw)
        lower = _gen("lyapunov", 24, 14, sequence="ab", **kw)
        mixed = _gen("lyapunov", 24, 14, sequence="Ab", **kw)
        np.testing.assert_array_equal(ab, lower)
        np.testing.assert_array_equal(ab, mixed)

    def test_empty_sequence_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _gen("lyapunov", sequence="", warmup=4, samples=4)
        self.assertIn("sequence", str(ctx.exception))

    def test_invalid_sequence_raises(self):
        for seq in ("ABC", "A C", "Ｂ"):
            with self.subTest(sequence=seq):
                with self.assertRaises(ValueError) as ctx:
                    _gen("lyapunov", sequence=seq, warmup=4, samples=4)
                self.assertIn("sequence", str(ctx.exception))


class EscapeTimeFixes(unittest.TestCase):
    def test_register_formula_rejects_degree_gt1_bailout_le1(self):
        step = compile_formula_expr("z**2 + c")
        with self.assertRaises(ValueError) as ctx:
            register_formula("bad_bail_low", step, degree=2.0, bailout=0.5, replace=True)
        msg = str(ctx.exception).lower()
        self.assertIn("bailout", msg)

        with self.assertRaises(ValueError) as ctx:
            register_formula("bad_bail_one", step, degree=3.0, bailout=1.0, replace=True)
        self.assertIn("bailout", str(ctx.exception).lower())

    def test_register_formula_allows_degree_le1_low_bailout(self):
        step = compile_formula_expr("c * sin(z)")
        f = register_formula("plain_counts_low_bail", step, degree=1.0, bailout=0.5, replace=True)
        self.assertEqual(f.bailout, 0.5)

    def test_max_iter_less_than_one_raises(self):
        for n in (0, -1, -10):
            with self.subTest(max_iter=n):
                with self.assertRaises(ValueError) as ctx:
                    _gen("escape_time", max_iter=n)
                self.assertIn("max_iter", str(ctx.exception))

    def test_max_iter_one_still_renders(self):
        rgb = _gen("escape_time", max_iter=1, formula="mandelbrot")
        self.assertEqual(rgb.shape, (9, 16, 3))
        self.assertTrue(np.isfinite(rgb).all())


if __name__ == "__main__":
    unittest.main()
