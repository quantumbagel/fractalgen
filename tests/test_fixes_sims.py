"""Regression tests for simulation / field generator edge-case fixes."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

import fractalgen as fg

fg.bootstrap()


def _render(gen: str, **params) -> np.ndarray:
    spec = fg.GENERATORS.get(gen)
    return spec(fg.RenderContext(64, 36), **params)


class FlowAttractorFixes(unittest.TestCase):
    def test_skip_ge_steps_yields_finite_frame(self):
        rgb = _render(
            "flow_attractor",
            flow="thomas",
            particle_base=8,
            steps=200,
            skip=500,
        )
        self.assertEqual(rgb.shape, (36, 64, 3))
        self.assertTrue(np.isfinite(rgb).all())

    def test_unknown_coeff_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _render(
                "flow_attractor",
                flow="thomas",
                coeffs={"beta": 0.2},
                particle_base=8,
                steps=20,
                skip=2,
            )
        self.assertIn("beta", str(ctx.exception))
        self.assertIn("has", str(ctx.exception))

    def test_short_init_mean_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _render("flow_attractor", init_mean=(0.0, 1.0))
        self.assertIn("init_mean", str(ctx.exception))

    def test_short_init_sigma_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _render("flow_attractor", init_sigma=(0.4, 0.4))
        self.assertIn("init_sigma", str(ctx.exception))

    def test_short_percentiles_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _render("flow_attractor", percentiles=(0.4,))
        self.assertIn("percentiles", str(ctx.exception))

    def test_register_flow_rejects_state_param_name(self):
        from fractalgen.generators.flows import register_flow

        with self.assertRaises(ValueError) as ctx:
            register_flow("qa_xclash", lambda x, y, z, **kw: (x, y, z), {"x": 1.0, "a": 1.0})
        self.assertIn("reserved", str(ctx.exception).lower())
        self.assertIn("x", str(ctx.exception))


class FlowSectionFixes(unittest.TestCase):
    def test_reserved_param_name_rejected(self):
        src = textwrap.dedent(
            """
            [flow.clash]
            dx = "x + a"
            dy = "y"
            dz = "z"
            params = { x = 5.0, a = 1.0 }
            """
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.toml"
            p.write_text(src)
            with self.assertRaises(ValueError) as ctx:
                fg.bootstrap([p])
            msg = str(ctx.exception).lower()
            self.assertIn("reserved", msg)
            self.assertIn("state names", msg)


class FlowFieldFixes(unittest.TestCase):
    def test_respawn_period_zero_no_crash(self):
        rgb = _render(
            "flow_field",
            particles=20,
            particles_cap=20,
            steps=8,
            respawn_period=0,
        )
        self.assertTrue(np.isfinite(rgb).all())

    def test_respawn_divisor_zero_no_crash(self):
        rgb = _render(
            "flow_field",
            particles=20,
            particles_cap=20,
            steps=8,
            respawn_period=1,
            respawn_divisor=0,
        )
        self.assertTrue(np.isfinite(rgb).all())


class VectorFieldSectionFixes(unittest.TestCase):
    def test_register_rejects_px(self):
        from fractalgen.generators.fields import register_vector_field

        with self.assertRaises(ValueError) as ctx:
            register_vector_field("qa_pxclash", lambda px, py, **kw: (px, py), defaults={"px": 1.0})
        self.assertIn("reserved", str(ctx.exception).lower())
        self.assertIn("px", str(ctx.exception))

    def test_reserved_px_rejected(self):
        src = textwrap.dedent(
            """
            [vectorfield.badpx]
            vx = "px"
            vy = "py"
            params = { px = 1.0, a = 0.5 }
            """
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.toml"
            p.write_text(src)
            with self.assertRaises(ValueError) as ctx:
                fg.bootstrap([p])
            msg = str(ctx.exception).lower()
            self.assertIn("reserved", msg)
            self.assertIn("px", msg)


class SandpileFixes(unittest.TestCase):
    def test_grid_n_zero_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _render("sandpile", grid_n=0, grains=10, max_iter=10)
        self.assertIn("grid_n", str(ctx.exception))


class WorleyFixes(unittest.TestCase):
    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _render("worley", mode="F2", n_points=8)
        self.assertIn("F2-F1", str(ctx.exception))


class CymaticsFixes(unittest.TestCase):
    def test_inverted_freq_jitter_renders(self):
        rgb = _render("cymatics", n_sources=2, freq_jitter=(1.5, 0.5))
        self.assertTrue(np.isfinite(rgb).all())


class MapAttractorFixes(unittest.TestCase):
    def test_n_zero_finite_frame(self):
        rgb = _render("map_attractor", n=0, steps=10, warmup=2)
        self.assertEqual(rgb.shape, (36, 64, 3))
        self.assertTrue(np.isfinite(rgb).all())


class AutoWindowFixes(unittest.TestCase):
    def test_empty_and_nonfinite_return_default(self):
        from fractalgen.generators.density import auto_window

        default = (-1.0, 1.0, -1.0, 1.0)
        self.assertEqual(auto_window([], []), default)
        nan = np.array([np.nan, np.inf, -np.inf])
        self.assertEqual(auto_window([nan], [nan]), default)


if __name__ == "__main__":
    unittest.main()
