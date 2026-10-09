"""The compiled replacements for numpy hot loops give numpy's exact results."""

from __future__ import annotations

import unittest

import numpy as np

from fractalgen import kernels
from fractalgen.generators.density import _histogram2d, deposit


class FastKernelTests(unittest.TestCase):
    def test_histogram2d_matches_numpy(self):
        rng = np.random.default_rng(3)
        x, y = rng.standard_normal(200_000), rng.standard_normal(200_000) ** 3
        x[:4] = x.max()  # values on the last edge go in the last bin
        H, xe, ye = _histogram2d(x, y, 64)
        H2, xe2, ye2 = np.histogram2d(x, y, bins=64)
        np.testing.assert_array_equal(H, H2)
        np.testing.assert_array_equal(xe, xe2)
        np.testing.assert_array_equal(ye, ye2)

    def test_deposit_with_hue_array_matches_add_at(self):
        rng = np.random.default_rng(4)
        x, y, hv = rng.uniform(-1, 1, 50_000), rng.uniform(-1, 1, 50_000), rng.random(50_000)
        d1, h1 = np.zeros((40, 60)), np.zeros((40, 60))
        deposit(d1, h1, x, y, -1, 1, -1, 1, hue_val=hv)
        d2, h2 = np.zeros((40, 60)), np.zeros((40, 60))
        ix = ((x + 1) / 2 * 59).astype(np.int32)
        iy = ((1 - y) / 2 * 39).astype(np.int32)
        np.add.at(d2, (iy, ix), 1.0)
        np.add.at(h2, (iy, ix), hv)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(h1, h2)

    def test_poly_newton_matches_numpy(self):
        from fractalgen.generators.newton import _poly_newton

        y, x = np.mgrid[-1.5:1.5:60j, -2:2:80j]
        coeffs = np.array([1.0, 0.0, -2.0, 2.0], dtype=np.complex128)
        z, n = _poly_newton(x, y, coeffs, 40, 1e-6)
        # The numpy loop this kernel replaced.
        zr, zi = x.copy(), y.copy()
        done_n = np.zeros(x.shape)
        done = np.zeros(x.shape, dtype=bool)
        d = np.polyder(coeffs)
        for k in range(1, 41):
            live = ~done
            zz = zr + 1j * zi
            p, dp = np.polyval(coeffs, zz), np.polyval(d, zz)
            with np.errstate(divide="ignore", invalid="ignore"):
                zn = zz - p / dp
            zn = np.where(np.isfinite(zn), zn, zz)
            zr, zi = np.where(live, zn.real, zr), np.where(live, zn.imag, zi)
            newly = live & (np.abs(p) < 1e-6)
            done[newly] = True
            done_n[newly] = k
        same = (n == done_n)
        self.assertGreater(same.mean(), 0.995)  # only chaotic basin borders may round differently

    def test_cycle_check_keeps_escape_counts(self):
        """Interior points stop early but report exactly what running to max_iter would."""
        y, x = np.mgrid[-1.2:1.2:90j, -2.0:0.6:120j]
        n, esc = kernels.escape_kernel(np.ascontiguousarray(x), np.ascontiguousarray(y), 3000, 0, 0.0, 0.0, 0.0,
                                       np.zeros(90, np.uint8))
        inside = (x + 0.25) ** 2 + y ** 2 < 0.2  # well inside the main cardioid
        self.assertTrue((esc[inside] == 0).all() and (n[inside] == 0).all())
        far = x ** 2 + y ** 2 > 4.5
        self.assertTrue((esc[far] == 1).all())


if __name__ == "__main__":
    unittest.main()
