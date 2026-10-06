"""Numba ray tracer for the ``kerr_black_hole`` generator.

Geometric units G = c = M = 1: the Schwarzschild horizon is r = 2, the photon
sphere r = 3 and the shadow's apparent radius sqrt(27) ~ 5.2.

Photons follow the exact Schwarzschild null geodesics, written in Cartesian
form as  x'' = -3 h^2 x / r^5  (h = |x × x'|, conserved), integrated with
adaptive RK4. Spin adds frame dragging (a Lense–Thirring twist of the ray about
the spin axis), moves the horizon to r+ = 1 + sqrt(1 - a^2) and lets the disk
reach the prograde ISCO. That is not a full Kerr solution, but it reproduces the
features that read on screen: the shadow, its photon ring, the lensed far side
of the disk wrapping over and under the hole, and the D-shaped asymmetry.

The disk is geometrically thin and mostly opaque, with a Novikov–Thorne
temperature profile, Doppler beaming / gravitational redshift (g factor), and
turbulent streaks from fBm noise sheared along the orbits.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=False)
def _hash(ix, iy, iz, seed):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 144269504) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@njit(fastmath=True)
def _vnoise(x, y, z, seed):
    """Smooth 3D value noise in [0, 1]."""
    ix, iy, iz = math.floor(x), math.floor(y), math.floor(z)
    fx, fy, fz = x - ix, y - iy, z - iz
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)
    ix, iy, iz = int(ix), int(iy), int(iz)
    c000 = _hash(ix, iy, iz, seed)
    c100 = _hash(ix + 1, iy, iz, seed)
    c010 = _hash(ix, iy + 1, iz, seed)
    c110 = _hash(ix + 1, iy + 1, iz, seed)
    c001 = _hash(ix, iy, iz + 1, seed)
    c101 = _hash(ix + 1, iy, iz + 1, seed)
    c011 = _hash(ix, iy + 1, iz + 1, seed)
    c111 = _hash(ix + 1, iy + 1, iz + 1, seed)
    x00 = c000 + (c100 - c000) * ux
    x10 = c010 + (c110 - c010) * ux
    x01 = c001 + (c101 - c001) * ux
    x11 = c011 + (c111 - c011) * ux
    y0 = x00 + (x10 - x00) * uy
    y1 = x01 + (x11 - x01) * uy
    return y0 + (y1 - y0) * uz


@njit(fastmath=True)
def _fbm(x, y, z, seed, octaves):
    total, amp, norm = 0.0, 0.5, 0.0
    for o in range(octaves):
        total += amp * _vnoise(x, y, z, seed + 17 * o)
        norm += amp
        x, y, z = x * 2.03, y * 2.03, z * 2.03
        amp *= 0.5
    return total / norm


@njit(fastmath=True)
def blackbody(t):
    """Approximate sRGB-ish chromaticity of a blackbody at t kelvin, max channel 1."""
    t = min(max(t, 1000.0), 40000.0) / 100.0
    if t <= 66.0:
        r = 1.0
        g = (99.4708025861 * math.log(t) - 161.1195681661) / 255.0
        b = 0.0 if t <= 19.0 else (138.5177312231 * math.log(t - 10.0) - 305.0447927307) / 255.0
    else:
        r = 329.698727446 * (t - 60.0) ** -0.1332047592 / 255.0
        g = 288.1221695283 * (t - 60.0) ** -0.0755148492 / 255.0
        b = 1.0
    return min(max(r, 0.0), 1.0), min(max(g, 0.0), 1.0), min(max(b, 0.0), 1.0)


@njit(fastmath=True)
def _accel(px, py, pz, h2):
    r2 = px * px + py * py + pz * pz
    k = -3.0 * h2 / (r2 * r2 * math.sqrt(r2))
    return k * px, k * py, k * pz


@njit(fastmath=True)
def _sky(dx, dy, dz, pix_angle, stars, star_size, galaxy, seed):
    """Dark sky: antialiased stars + a faint galactic band (direction must be unit).

    ``pix_angle`` is the patch of sky this pixel sees (after lensing), so stars in
    squeezed regions blur into their average instead of sparkling.
    """
    r = 0.0
    g = 0.0
    b = 0.0
    # Galactic band, tilted, with dusty structure.
    lat = 0.35 * dx + 0.88 * dy + 0.32 * dz
    band = math.exp(-lat * lat * 18.0)
    if band > 0.002 and galaxy > 0.0:
        dust = _fbm(dx * 6.0, dy * 6.0, dz * 6.0, seed + 5, 5)
        glow = galaxy * band * (0.25 + 1.6 * max(dust - 0.35, 0.0))
        r += glow * 0.95
        g += glow * 0.88
        b += glow * 1.0
    # Stars: one jittered star per cell of a direction grid, drawn as a gaussian
    # no narrower than a pixel and flux-normalized, so they look the same at any size.
    if stars > 0.0:
        scale = 90.0
        sigma = max(star_size, 0.7 * pix_angle)
        norm = (star_size / sigma) ** 2
        for layer in range(2):
            s = scale * (1.0 + layer * 1.7)
            qx, qy, qz = dx * s, dy * s, dz * s
            cx, cy, cz = math.floor(qx), math.floor(qy), math.floor(qz)
            ix, iy, iz = int(cx), int(cy), int(cz)
            if _hash(ix, iy, iz, seed + 101 + layer) > stars:
                continue
            sx = cx + 0.15 + 0.7 * _hash(ix, iy, iz, seed + 11 + layer)
            sy = cy + 0.15 + 0.7 * _hash(ix, iy, iz, seed + 23 + layer)
            sz = cz + 0.15 + 0.7 * _hash(ix, iy, iz, seed + 37 + layer)
            sl = math.sqrt(sx * sx + sy * sy + sz * sz)
            ex, ey, ez = dx - sx / sl, dy - sy / sl, dz - sz / sl
            d2 = ex * ex + ey * ey + ez * ez
            mag = _hash(ix, iy, iz, seed + 53 + layer)
            amp = 3.0 * mag ** 8 + 0.06 * mag
            f = amp * norm * math.exp(-d2 / (2.0 * sigma * sigma))
            if f > 1e-5:
                cr, cg, cb = blackbody(3200.0 + 9000.0 * _hash(ix, iy, iz, seed + 71 + layer))
                r += f * cr
                g += f * cg
                b += f * cb
    return r, g, b


@njit(parallel=True, fastmath=True)
def trace(h, w, spin, cam_dist, cam_elev, roll, fov, disk_in, disk_out, t_peak, doppler,
          turbulence, streaks, disk_alpha, stars, star_size, galaxy, max_steps, seed):
    """Disk light (h, w, 3), leftover transmittance (h, w) and escape direction (h, w, 3; 0 if captured)."""
    out = np.zeros((h, w, 3), dtype=np.float64)
    trans_out = np.zeros((h, w), dtype=np.float64)
    esc = np.zeros((h, w, 3), dtype=np.float64)
    a = spin
    r_plus = 1.0 + math.sqrt(max(0.0, 1.0 - a * a))
    r_capture = 2.0 * 1.01  # horizon of the metric the rays follow
    # camera on the y-z plane, looking at the origin; y is the spin axis
    se, ce = math.sin(cam_elev), math.cos(cam_elev)
    cx, cy, cz = 0.0, se * cam_dist, ce * cam_dist
    fx, fy, fz = -cx / cam_dist, -cy / cam_dist, -cz / cam_dist
    rx, ry, rz = fz, 0.0, -fx  # fwd × up(0,1,0), normalized below
    rl = math.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / rl, ry / rl, rz / rl
    ux, uy, uz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx
    cr, sr = math.cos(roll), math.sin(roll)
    rx, ry, rz, ux, uy, uz = (cr * rx + sr * ux, cr * ry + sr * uy, cr * rz + sr * uz,
                              -sr * rx + cr * ux, -sr * ry + cr * uy, -sr * rz + cr * uz)
    # fov spans the frame's shorter side, so phones get the whole hole too
    tan_half = math.tan(0.5 * fov)
    short = float(min(w, h))
    # Novikov–Thorne-like profile T ~ r^-3/4 (1 - sqrt(r_in/r))^1/4, normalized to its peak.
    r_pk = 49.0 / 36.0 * disk_in
    t_norm = (disk_in / r_pk) ** 0.75 * (1.0 - math.sqrt(disk_in / r_pk)) ** 0.25
    escape_r = cam_dist * 1.5 + disk_out
    for j in prange(h):
        for i in range(w):
            sx = (2.0 * (i + 0.5) - w) / short * tan_half
            sy = (h - 2.0 * (j + 0.5)) / short * tan_half
            vx, vy, vz = fx + sx * rx + sy * ux, fy + sx * ry + sy * uy, fz + sx * rz + sy * uz
            vl = math.sqrt(vx * vx + vy * vy + vz * vz)
            vx, vy, vz = vx / vl, vy / vl, vz / vl
            px, py, pz = cx, cy, cz
            hx, hy, hz = py * vz - pz * vy, pz * vx - px * vz, px * vy - py * vx
            h2 = hx * hx + hy * hy + hz * hz
            er = 0.0
            eg = 0.0
            eb = 0.0
            trans = 1.0
            fate = 0  # 0 = ran out of steps, 1 = horizon, 2 = escaped
            for _ in range(max_steps):
                r = math.sqrt(px * px + py * py + pz * pz)
                if r < r_capture:
                    fate = 1
                    break
                if r > escape_r and px * vx + py * vy + pz * vz > 0.0:
                    fate = 2
                    break
                # Step: fine near the photon sphere, coarse far away.
                dl = 0.06 * r * min(1.0, max(0.12, (r - r_plus) / 3.0))
                if r > 2.0 * disk_out:
                    dl = 0.25 * r
                ax1, ay1, az1 = _accel(px, py, pz, h2)
                p2x, p2y, p2z = px + 0.5 * dl * vx, py + 0.5 * dl * vy, pz + 0.5 * dl * vz
                v2x, v2y, v2z = vx + 0.5 * dl * ax1, vy + 0.5 * dl * ay1, vz + 0.5 * dl * az1
                ax2, ay2, az2 = _accel(p2x, p2y, p2z, h2)
                p3x, p3y, p3z = px + 0.5 * dl * v2x, py + 0.5 * dl * v2y, pz + 0.5 * dl * v2z
                v3x, v3y, v3z = vx + 0.5 * dl * ax2, vy + 0.5 * dl * ay2, vz + 0.5 * dl * az2
                ax3, ay3, az3 = _accel(p3x, p3y, p3z, h2)
                p4x, p4y, p4z = px + dl * v3x, py + dl * v3y, pz + dl * v3z
                v4x, v4y, v4z = vx + dl * ax3, vy + dl * ay3, vz + dl * az3
                ax4, ay4, az4 = _accel(p4x, p4y, p4z, h2)
                npx = px + dl / 6.0 * (vx + 2.0 * v2x + 2.0 * v3x + v4x)
                npy = py + dl / 6.0 * (vy + 2.0 * v2y + 2.0 * v3y + v4y)
                npz = pz + dl / 6.0 * (vz + 2.0 * v2z + 2.0 * v3z + v4z)
                vx += dl / 6.0 * (ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4)
                vy += dl / 6.0 * (ay1 + 2.0 * ay2 + 2.0 * ay3 + ay4)
                vz += dl / 6.0 * (az1 + 2.0 * az2 + 2.0 * az3 + az4)
                # frame dragging: twist the direction about the spin axis
                if a != 0.0:
                    om = 0.4 * a / (r * r * r + a * a * r) * dl  # gentle: a full twist breaks the photon ring
                    co, so = math.cos(om), math.sin(om)
                    vx, vz = co * vx + so * vz, -so * vx + co * vz
                # disk plane crossing
                if py * npy < 0.0:
                    f = py / (py - npy)
                    qx, qz = px + f * (npx - px), pz + f * (npz - pz)
                    rho = math.sqrt(qx * qx + qz * qz)
                    if disk_in < rho < disk_out:
                        x_in = disk_in / rho
                        temp = t_peak * (x_in ** 0.75 * (1.0 - math.sqrt(x_in)) ** 0.25) / t_norm
                        # Keplerian (prograde) orbital speed and the photon's direction toward the camera
                        omega = 1.0 / (rho ** 1.5 + a)
                        beta = min(omega * rho / math.sqrt(max(1.0 - 2.0 / rho, 0.05)), 0.85)
                        gam = 1.0 / math.sqrt(1.0 - beta * beta)
                        tx, tz = qz / rho, -qx / rho  # prograde about +y
                        vn = math.sqrt(vx * vx + vy * vy + vz * vz)
                        cos_t = -(tx * vx + tz * vz) / vn
                        g = math.sqrt(max(1.0 - 3.0 / (rho + 1e-9), 0.02)) * 1.0 / (gam * (1.0 - beta * cos_t))
                        g = 1.0 + doppler * (g - 1.0)
                        # turbulence: streaks along the orbits, sheared into a slow spiral
                        phi = math.atan2(qz, qx) + 1.2 * math.log(rho)
                        lr = math.log(rho)
                        n = _fbm(math.cos(phi) * 2.2, math.sin(phi) * 2.2, lr * streaks, seed, 5)
                        n2 = _fbm(math.cos(phi) * 5.0, math.sin(phi) * 5.0, lr * streaks * 2.2, seed + 3, 3)
                        dens = 1.0 - turbulence + turbulence * 2.0 * (0.75 * n + 0.25 * n2) ** 1.8
                        edge = min(1.0, (rho - disk_in) / (0.06 * disk_in) + 0.15) * min(1.0, (disk_out - rho) / (0.35 * disk_out))
                        alpha = min(1.0, disk_alpha * dens) * max(edge, 0.0)
                        cr_, cg_, cb_ = blackbody(temp * g)
                        inten = (temp / t_peak) ** 4 * g ** 4 * dens * 2.2
                        er += trans * alpha * inten * cr_
                        eg += trans * alpha * inten * cg_
                        eb += trans * alpha * inten * cb_
                        trans *= 1.0 - alpha
                        if trans < 0.01:
                            break
                px, py, pz = npx, npy, npz
            if fate == 2:
                vn = math.sqrt(vx * vx + vy * vy + vz * vz)
                esc[j, i, 0] = vx / vn
                esc[j, i, 1] = vy / vn
                esc[j, i, 2] = vz / vn
                trans_out[j, i] = trans
            out[j, i, 0] = er
            out[j, i, 1] = eg
            out[j, i, 2] = eb
    return out, trans_out, esc


@njit(parallel=True, fastmath=True)
def shade_sky(img, trans, esc, pix_angle, stars, star_size, galaxy, seed):
    """Add the sky behind the disk. Each pixel's sky footprint comes from how far
    its escape direction is from its neighbours' (ray differentials)."""
    h, w = trans.shape
    for j in prange(h):
        for i in range(w):
            t = trans[j, i]
            if t < 0.01:
                continue
            dx, dy, dz = esc[j, i, 0], esc[j, i, 1], esc[j, i, 2]
            foot = pix_angle
            for oj, oi in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                jj, ii = j + oj, i + oi
                if 0 <= jj < h and 0 <= ii < w:
                    nx, ny, nz = esc[jj, ii, 0], esc[jj, ii, 1], esc[jj, ii, 2]
                    if nx * nx + ny * ny + nz * nz > 0.5:
                        ex, ey, ez = nx - dx, ny - dy, nz - dz
                        foot = max(foot, math.sqrt(ex * ex + ey * ey + ez * ez))
            foot = min(foot, 0.2)
            r, g, b = _sky(dx, dy, dz, foot, stars, star_size, galaxy, seed)
            img[j, i, 0] += t * r
            img[j, i, 1] += t * g
            img[j, i, 2] += t * b


@njit(fastmath=True)
def isco(a):
    """Prograde innermost stable circular orbit radius for spin a (M = 1)."""
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
