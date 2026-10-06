"""Numba kernels for the heavy per-pixel / per-particle loops.

Kernels deliberately do NOT use numba's on-disk cache (``cache=True``): with
``fastmath`` a cache-loaded kernel can round differently from a fresh compile,
flipping chaotic boundary pixels, so the same recipe rendered differently
depending on cache state. Compiling costs ~1-4 s per worker process.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

KIND = {
    "mandelbrot": 0,
    "julia": 1,
    "burning_ship": 2,
    "buffalo": 3,
    "tricorn": 4,
    "phoenix": 5,
    "celtic": 6,
    "multibrot3": 7,
    "multibrot4": 8,
}


@njit(parallel=True, fastmath=True)
def escape_kernel(cr, ci, max_iter, kind, jcr, jci, pk):
    h, w = cr.shape
    escaped_n = np.zeros((h, w), dtype=np.float64)
    escaped = np.zeros((h, w), dtype=np.uint8)
    log2 = np.log(2.0)
    for y in prange(h):
        for x in range(w):
            crx = cr[y, x]
            cix = ci[y, x]
            if kind == 1:
                zr = crx
                zi = cix
                crx = jcr
                cix = jci
            elif kind == 5:
                zr = crx
                zi = cix
                crx = jcr
                cix = 0.0
            else:
                zr = 0.0
                zi = 0.0
            pr = 0.0
            pi = 0.0
            did = 0
            n_esc = 0.0
            for n in range(max_iter):
                if kind == 2 or kind == 3:
                    zr = abs(zr)
                    zi = abs(zi)
                elif kind == 4:
                    zi = -zi
                zr2 = zr * zr
                zi2 = zi * zi
                mag2 = zr2 + zi2
                if mag2 > 4.0:
                    mag = np.sqrt(mag2)
                    if mag < 1.0000001:
                        mag = 1.0000001
                    nu = n + 1.0 - np.log(np.log(mag)) / log2
                    if nu < 0.0:
                        nu = 0.0
                    if nu > max_iter:
                        nu = float(max_iter)
                    n_esc = nu
                    did = 1
                    break
                if kind == 5:
                    nr = zr2 - zi2 + crx + pk * pr
                    ni = 2.0 * zr * zi + pk * pi
                    pr = zr
                    pi = zi
                elif kind == 6:
                    nr = abs(zr2 - zi2) + crx
                    ni = 2.0 * zr * zi + cix
                elif kind == 3:
                    nr = zr2 - zi2 - zr + crx
                    ni = 2.0 * zr * zi - zi + cix
                elif kind == 7:
                    nr = zr * (zr2 - 3.0 * zi2) + crx
                    ni = zi * (3.0 * zr2 - zi2) + cix
                elif kind == 8:
                    ar = zr2 - zi2
                    ai = 2.0 * zr * zi
                    nr = ar * ar - ai * ai + crx
                    ni = 2.0 * ar * ai + cix
                else:
                    nr = zr2 - zi2 + crx
                    ni = 2.0 * zr * zi + cix
                zr = nr
                zi = ni
            escaped_n[y, x] = n_esc
            escaped[y, x] = did
    return escaped_n, escaped


@njit(parallel=True, fastmath=True)
def orbit_trap_kernel(cr, ci, max_iter, trap_id):
    h, w = cr.shape
    trap_d = np.empty((h, w), dtype=np.float64)
    for y in prange(h):
        for x in range(w):
            zr = 0.0
            zi = 0.0
            best = 1e9
            crx = cr[y, x]
            cix = ci[y, x]
            for _n in range(max_iter):
                zr2 = zr * zr
                zi2 = zi * zi
                mag2 = zr2 + zi2
                if mag2 > 4.0:
                    break
                mag = np.sqrt(mag2) if mag2 > 0.0 else 0.0
                if trap_id == 0:
                    d = abs(mag - 0.5)
                elif trap_id == 1:
                    az = abs(zr)
                    ai = abs(zi)
                    d = az if az < ai else ai
                elif trap_id == 2:
                    d = abs(np.sqrt((zr - 0.35) * (zr - 0.35) + zi * zi) - 0.7)
                else:
                    d = mag
                # z0 = 0 lies on the cross and point traps; measuring it would trap every pixel.
                if d < best and (_n > 0 or trap_id == 0 or trap_id == 2):
                    best = d
                nr = zr2 - zi2 + crx
                ni = 2.0 * zr * zi + cix
                zr = nr
                zi = ni
            trap_d[y, x] = best if best > 1e-8 else 1e-8
            if trap_d[y, x] > 10.0:
                trap_d[y, x] = 10.0
    return trap_d


@njit(parallel=True, fastmath=True)
def newton_kernel(zr0, zi0, power, max_iter, tol):
    h, w = zr0.shape
    root_ix = np.zeros((h, w), dtype=np.int32)
    done_n = np.zeros((h, w), dtype=np.float64)
    n = power
    nm1 = n - 1
    for y in prange(h):
        for x in range(w):
            zr = zr0[y, x]
            zi = zi0[y, x]
            it = 0.0
            for k in range(1, max_iter + 1):
                # z**(n-1)
                pr = 1.0
                pi = 0.0
                for _p in range(nm1):
                    nr = pr * zr - pi * zi
                    ni = pr * zi + pi * zr
                    pr, pi = nr, ni
                # p = z^n - 1 = z * z**(n-1) - 1
                pret = zr * pr - zi * pi - 1.0
                pimt = zr * pi + zi * pr
                # dp = n * z**(n-1)
                dpr = n * pr
                dpi = n * pi
                den = dpr * dpr + dpi * dpi
                if den < 1e-30:
                    break
                # z -= p/dp
                zr -= (pret * dpr + pimt * dpi) / den
                zi -= (pimt * dpr - pret * dpi) / den
                if pret * pret + pimt * pimt < tol * tol:
                    it = float(k)
                    break
            # nearest nth root of unity
            best = 1e30
            bi = 0
            for i in range(n):
                ang = 2.0 * np.pi * i / n
                rr = np.cos(ang)
                ri = np.sin(ang)
                dr = zr - rr
                di = zi - ri
                d2 = dr * dr + di * di
                if d2 < best:
                    best = d2
                    bi = i
            root_ix[y, x] = bi
            done_n[y, x] = it
    return root_ix, done_n


@njit(parallel=True, fastmath=True)
def lyapunov_kernel(a, b, seq, warmup, samples):
    h, w = a.shape
    out = np.zeros((h, w), dtype=np.float64)
    slen = seq.shape[0]
    total = warmup + samples
    for y in prange(h):
        for x in range(w):
            xv = 0.5
            acc = 0.0
            count = 0
            for i in range(total):
                r = b[y, x] if seq[i % slen] != 0 else a[y, x]
                xv = r * xv * (1.0 - xv)
                if xv < 1e-12:
                    xv = 1e-12
                if xv > 1.0 - 1e-12:
                    xv = 1.0 - 1e-12
                if i >= warmup:
                    deriv = r * (1.0 - 2.0 * xv)
                    if deriv < 0.0:
                        deriv = -deriv
                    acc += np.log(deriv + 1e-12)
                    count += 1
            out[y, x] = acc / count if count else 0.0
    return out


@njit(parallel=True, fastmath=True)
def popcorn_kernel(zr, zi, steps, hstep):
    h, w = zr.shape
    dx = np.empty((h, w), dtype=np.float64)
    dy = np.empty((h, w), dtype=np.float64)
    for y in prange(h):
        for x in range(w):
            xv = zr[y, x]
            yv = zi[y, x]
            x0 = xv
            y0 = yv
            ok = True
            for _s in range(steps):
                ty = np.tan(3.0 * yv)
                tx = np.tan(3.0 * xv)
                if not np.isfinite(ty) or not np.isfinite(tx):
                    ok = False
                    break
                nx = xv - hstep * np.sin(yv + ty)
                ny = yv - hstep * np.sin(xv + tx)
                xv, yv = nx, ny
            if ok and np.isfinite(xv) and np.isfinite(yv):
                dx[y, x] = xv - x0
                dy[y, x] = yv - y0
            else:
                dx[y, x] = 0.0
                dy[y, x] = 0.0
    return dx, dy


@njit(fastmath=True, parallel=True)
def gray_scott_kernel(u, v, F, k, steps, Du, Dv, dt):
    sh, sw = u.shape
    u2 = np.empty_like(u)
    v2 = np.empty_like(v)
    for _i in range(steps):
        for y in prange(sh):
            ym = y - 1 if y > 0 else sh - 1
            yp = y + 1 if y + 1 < sh else 0
            for x in range(sw):
                xm = x - 1 if x > 0 else sw - 1
                xp = x + 1 if x + 1 < sw else 0
                lu = u[ym, x] + u[yp, x] + u[y, xm] + u[y, xp] - 4.0 * u[y, x]
                lv = v[ym, x] + v[yp, x] + v[y, xm] + v[y, xp] - 4.0 * v[y, x]
                uvv = u[y, x] * v[y, x] * v[y, x]
                nu = u[y, x] + dt * (Du * lu - uvv + F * (1.0 - u[y, x]))
                nv = v[y, x] + dt * (Dv * lv + uvv - (F + k) * v[y, x])
                if nu < 0.0:
                    nu = 0.0
                elif nu > 1.0:
                    nu = 1.0
                if nv < 0.0:
                    nv = 0.0
                elif nv > 1.0:
                    nv = 1.0
                u2[y, x] = nu
                v2[y, x] = nv
        u, v, u2, v2 = u2, v2, u, v
    return u, v


@njit(fastmath=True)
def scatter_add2(a, b, iy, ix, wb):
    """a[iy, ix] += 1 and b[iy, ix] += wb, in input order (same sums as np.add.at, much faster)."""
    for i in range(iy.shape[0]):
        a[iy[i], ix[i]] += 1.0
        b[iy[i], ix[i]] += wb[i]


@njit(fastmath=True)
def deposit_kernel(density, hue, x, y, xmin, xmax, ymin, ymax, hue_t):
    """Serial hit-count deposit. Not parallel: many particles share pixels."""
    h, w = density.shape
    sx = (w - 1) / (xmax - xmin)
    sy = (h - 1) / (ymax - ymin)
    n = x.shape[0]
    for i in range(n):
        xi = x[i]
        yi = y[i]
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        ix = int((xi - xmin) * sx)
        iy = int((ymax - yi) * sy)
        if ix >= 0 and ix < w and iy >= 0 and iy < h:
            density[iy, ix] += 1.0
            hue[iy, ix] += hue_t


@njit(fastmath=True)
def sandpile_topple(g, max_iter):
    """In-place abelian sandpile relaxation. Grains that fall off the edge are lost."""
    n = g.shape[0]
    for _ in range(max_iter):
        changed = False
        for i in range(n):
            for j in range(n):
                v = g[i, j]
                if v >= 4:
                    add = v // 4
                    g[i, j] = v - add * 4
                    if i > 0:
                        g[i - 1, j] += add
                    if i + 1 < n:
                        g[i + 1, j] += add
                    if j > 0:
                        g[i, j - 1] += add
                    if j + 1 < n:
                        g[i, j + 1] += add
                    changed = True
        if not changed:
            break
    return g


@njit(fastmath=True, parallel=True)
def gray_scott_field_kernel(u, v, F, k, steps, Du, Dv, dt):
    """Gray-Scott with spatially varying feed / kill."""
    sh, sw = u.shape
    u2 = np.empty_like(u)
    v2 = np.empty_like(v)
    for _i in range(steps):
        for y in prange(sh):
            ym = y - 1 if y > 0 else sh - 1
            yp = y + 1 if y + 1 < sh else 0
            for x in range(sw):
                xm = x - 1 if x > 0 else sw - 1
                xp = x + 1 if x + 1 < sw else 0
                lu = u[ym, x] + u[yp, x] + u[y, xm] + u[y, xp] - 4.0 * u[y, x]
                lv = v[ym, x] + v[yp, x] + v[y, xm] + v[y, xp] - 4.0 * v[y, x]
                uvv = u[y, x] * v[y, x] * v[y, x]
                f = F[y, x]
                kk = k[y, x]
                nu = u[y, x] + dt * (Du * lu - uvv + f * (1.0 - u[y, x]))
                nv = v[y, x] + dt * (Dv * lv + uvv - (f + kk) * v[y, x])
                if nu < 0.0:
                    nu = 0.0
                elif nu > 1.0:
                    nu = 1.0
                if nv < 0.0:
                    nv = 0.0
                elif nv > 1.0:
                    nv = 1.0
                u2[y, x] = nu
                v2[y, x] = nv
        u, v, u2, v2 = u2, v2, u, v
    return u, v


@njit(fastmath=True)
def _wrap_cell(v, n):
    i = int(np.floor(v)) % n
    return i + n if i < 0 else i


@njit(parallel=True, fastmath=True)
def physarum_steps(x, y, heading, trail, sa, ra, sd, step, deposit, n_steps, seed):
    """Jeff Jones slime-mold agents (positions in pixel coordinates).

    Synchronous update so agents run on all cores: every agent senses the same
    trail snapshot and moves, then all deposit. Ties break with a per-agent,
    per-step hash, so the result does not depend on thread scheduling.
    """
    h, w = trail.shape
    n = x.shape[0]
    wf = float(w)
    hf = float(h)
    cells = np.empty(n, dtype=np.int64)
    for s in range(n_steps):
        for i in prange(n):
            hd = heading[i]
            px = x[i]
            py = y[i]
            fl = trail[_wrap_cell(py + np.sin(hd + sa) * sd, h), _wrap_cell(px + np.cos(hd + sa) * sd, w)]
            ff = trail[_wrap_cell(py + np.sin(hd) * sd, h), _wrap_cell(px + np.cos(hd) * sd, w)]
            fr = trail[_wrap_cell(py + np.sin(hd - sa) * sd, h), _wrap_cell(px + np.cos(hd - sa) * sd, w)]
            if ff >= fl and ff >= fr:
                pass
            elif fl > fr:
                hd += ra
            elif fr > fl:
                hd -= ra
            else:
                r = (np.uint32(i) * np.uint32(2654435761)) ^ (np.uint32(s + seed) * np.uint32(2246822519))
                r = (r ^ (r >> np.uint32(15))) * np.uint32(2246822519)
                hd += ra if (r >> np.uint32(16)) & np.uint32(1) else -ra
            px += np.cos(hd) * step
            py += np.sin(hd) * step
            if px < 0.0:
                px += wf
            elif px >= wf:
                px -= wf
            if py < 0.0:
                py += hf
            elif py >= hf:
                py -= hf
            x[i] = px
            y[i] = py
            heading[i] = hd
            cells[i] = _wrap_cell(py, h) * w + _wrap_cell(px, w)
        flat = trail.reshape(-1)
        for i in range(n):
            flat[cells[i]] += deposit


@njit(parallel=True, fastmath=True)
def poincare_kernel(h, w, p_gon, q_star, max_fold):
    """Fold the Poincaré disk into a {p,q} fundamental domain."""
    depth = np.zeros((h, w), dtype=np.float64)
    edge = np.zeros((h, w), dtype=np.float64)
    sector = np.zeros((h, w), dtype=np.float64)
    aspect = w / float(h)
    pip = np.pi / p_gon
    piq = np.pi / q_star
    # hyperbolic circumradius of the regular p-gon
    num = np.cos(piq) * (1.0 + np.cos(2.0 * pip))
    den = np.sin(piq) * np.sin(2.0 * pip) + 1e-12
    ch = num / den
    if ch < 1.05:
        ch = 1.05
    Rhyp = np.log(ch + np.sqrt(ch * ch - 1.0))
    rho = np.tanh(0.5 * Rhyp)
    t_c = (rho * rho + 1.0) / (2.0 * rho * np.cos(pip))
    r_c = np.sqrt(max(t_c * t_c - 1.0, 1e-8))
    # fill the short axis with the disk
    scale = 1.06 * (aspect if aspect < 1.0 else 1.0)
    for y in prange(h):
        for x in range(w):
            u = (2.0 * x / (w - 1.0) - 1.0) * aspect / scale
            v = (1.0 - 2.0 * y / (h - 1.0)) / scale
            rr = u * u + v * v
            if rr >= 0.999:
                depth[y, x] = -1.0
                continue
            zr, zi = u, v
            folds = 0.0
            sec_acc = 0.0
            last_edge = 1e9
            for _k in range(max_fold):
                ang = np.arctan2(zi, zr)
                sector_w = 2.0 * pip
                ksec = np.floor(ang / sector_w + 0.5)
                ang -= ksec * sector_w
                rad = np.sqrt(zr * zr + zi * zi)
                zr = rad * np.cos(ang)
                zi = rad * np.sin(ang)
                sec_acc += ksec
                dx = zr - t_c
                dy = zi
                dcirc = np.sqrt(dx * dx + dy * dy)
                last_edge = dcirc - r_c
                if dcirc < r_c - 1e-8:
                    inv = (r_c * r_c) / (dcirc * dcirc + 1e-14)
                    zr = t_c + dx * inv
                    zi = dy * inv
                    folds += 1.0
                else:
                    break
            depth[y, x] = folds
            edge[y, x] = last_edge
            sector[y, x] = sec_acc
    return depth, edge, sector


@njit(fastmath=True)
def splat_bokeh(img, px, py, depth, z_focus, aperture, weight, cr, cg, cb, keep):
    """Variable-CoC splat with a hint of chromatic aberration."""
    h, w = img.shape[0], img.shape[1]
    n = px.shape[0]
    for i in range(n):
        if keep[i] < 0.5:
            continue
        x = px[i]
        y = py[i]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        coc = aperture * abs(depth[i] - z_focus)
        rad = 0.55 + coc
        if rad > 14.0:
            rad = 14.0
        if rad < 1.35:
            ix = int(x + 0.5)
            iy = int(y + 0.5)
            if ix >= 0 and ix < w and iy >= 0 and iy < h:
                img[iy, ix, 0] += weight[i] * cr[i]
                img[iy, ix, 1] += weight[i] * cg[i]
                img[iy, ix, 2] += weight[i] * cb[i]
            continue
        r = int(rad + 0.999)
        wr = weight[i] / (rad * rad)
        ca = 0.35 + 0.12 * rad
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                ix = int(x + dx + 0.5)
                iy = int(y + dy + 0.5)
                ixr = int(x + dx + ca + 0.5)
                ixb = int(x + dx - ca + 0.5)
                if iy >= 0 and iy < h:
                    if ixr >= 0 and ixr < w:
                        img[iy, ixr, 0] += wr * cr[i]
                    if ix >= 0 and ix < w:
                        img[iy, ix, 1] += wr * cg[i]
                    if ixb >= 0 and ixb < w:
                        img[iy, ixb, 2] += wr * cb[i]


@njit(parallel=True, fastmath=True)
def escape_kernel_generic(cr, ci, max_iter, step, julia, jc, k, bailout2, log_degree):
    """Escape time for any registered ``step(z, c, z_prev, k) -> z`` formula."""
    h, w = cr.shape
    escaped_n = np.zeros((h, w), dtype=np.float64)
    escaped = np.zeros((h, w), dtype=np.uint8)
    for y in prange(h):
        for x in range(w):
            p = complex(cr[y, x], ci[y, x])
            if julia:
                z = p
                c = jc
            else:
                z = 0j
                c = p
            zp = 0j
            for n in range(max_iter):
                mag2 = z.real * z.real + z.imag * z.imag
                if mag2 > bailout2:
                    mag = np.sqrt(mag2)
                    if mag < 1.0000001:
                        mag = 1.0000001
                    if log_degree > 0.0:
                        nu = n + 1.0 - np.log(np.log(mag) / np.log(np.sqrt(bailout2))) / log_degree
                    else:  # no polynomial growth rate (e.g. transcendental): plain count
                        nu = n + 1.0
                    if nu < 0.0:
                        nu = 0.0
                    if nu > max_iter:
                        nu = float(max_iter)
                    escaped_n[y, x] = nu
                    escaped[y, x] = 1
                    break
                zn = step(z, c, zp, k)
                zp = z
                z = zn
    return escaped_n, escaped


# ---- shared helpers for point-splat generators ---------------------------------

@njit(fastmath=True)
def _mix64(z):
    """splitmix64 finalizer: a stateless, scheduling-independent hash."""
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(fastmath=True)
def _rand01(key, i, t):
    """Uniform [0, 1) from (key, i, t)."""
    h = _mix64(key ^ (np.uint64(i) * np.uint64(0x9E3779B97F4A7C15)) ^ (np.uint64(t) * np.uint64(0xD1B54A32D192ED03)))
    return float(h >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(fastmath=True)
def splat_rgb(acc, x, y, c, lut, ax, bx, cx, ay, by, cy):
    """Serial nearest-pixel splat of plane points into acc[H, W, 4] = (hits, r, g, b).
    ``c`` is a 0..1 palette coordinate per point, looked up in ``lut`` (N, 3)."""
    h, w = acc.shape[0], acc.shape[1]
    nl = lut.shape[0]
    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        fx = ax * xi + bx * yi + cx + 0.5
        fy = ay * xi + by * yi + cy + 0.5
        if fx < 0.0 or fy < 0.0 or fx >= w or fy >= h:
            continue
        ix = int(fx)
        iy = int(fy)
        k = int(c[i] * (nl - 1) + 0.5)
        if k < 0:
            k = 0
        elif k >= nl:
            k = nl - 1
        acc[iy, ix, 0] += 1.0
        acc[iy, ix, 1] += lut[k, 0]
        acc[iy, ix, 2] += lut[k, 1]
        acc[iy, ix, 3] += lut[k, 2]


# ---- fractal flame -------------------------------------------------------------

@njit(fastmath=True)
def _flame_variation(v, x, y, key, i, t):
    r2 = x * x + y * y + 1e-12
    r = np.sqrt(r2)
    if v == 0:  # linear
        return x, y
    if v == 1:  # sinusoidal
        return np.sin(x), np.sin(y)
    if v == 2:  # spherical
        return x / r2, y / r2
    if v == 3:  # swirl
        s, c = np.sin(r2), np.cos(r2)
        return x * s - y * c, x * c + y * s
    if v == 4:  # horseshoe
        return (x - y) * (x + y) / r, 2.0 * x * y / r
    th = np.arctan2(x, y)
    if v == 5:  # polar
        return th / np.pi, r - 1.0
    if v == 6:  # handkerchief
        return r * np.sin(th + r), r * np.cos(th - r)
    if v == 7:  # heart
        return r * np.sin(th * r), -r * np.cos(th * r)
    if v == 8:  # disc
        return th / np.pi * np.sin(np.pi * r), th / np.pi * np.cos(np.pi * r)
    if v == 9:  # spiral
        return (np.cos(th) + np.sin(r)) / r, (np.sin(th) - np.cos(r)) / r
    if v == 10:  # hyperbolic
        return np.sin(th) / r, r * np.cos(th)
    if v == 11:  # diamond
        return np.sin(th) * np.cos(r), np.cos(th) * np.sin(r)
    if v == 12:  # ex
        p0 = np.sin(th + r)
        p1 = np.cos(th - r)
        return r * (p0 * p0 * p0 + p1 * p1 * p1), r * (p0 * p0 * p0 - p1 * p1 * p1)
    if v == 13:  # julia
        om = np.pi if _rand01(key, i, t * 7 + 3) < 0.5 else 0.0
        sr = np.sqrt(r)
        return sr * np.cos(th * 0.5 + om), sr * np.sin(th * 0.5 + om)
    if v == 14:  # bent
        nx = x * 2.0 if x < 0.0 else x
        ny = y * 0.5 if y < 0.0 else y
        return nx, ny
    if v == 15:  # fisheye
        f = 2.0 / (r + 1.0)
        return f * y, f * x
    if v == 16:  # exponential
        e = np.exp(x - 1.0)
        return e * np.cos(np.pi * y), e * np.sin(np.pi * y)
    if v == 17:  # cosine
        return np.cos(np.pi * x) * np.cosh(y), -np.sin(np.pi * x) * np.sinh(y)
    if v == 18:  # eyefish
        f = 2.0 / (r + 1.0)
        return f * x, f * y
    if v == 19:  # bubble
        f = 4.0 / (r2 + 4.0)
        return f * x, f * y
    if v == 20:  # cylinder
        return np.sin(x), y
    return x, y


@njit(parallel=True, fastmath=True)
def flame_chunk(x, y, col, t0, n_steps, coefs, varw, cum, xcol, cspeed, sym, key, out_x, out_y, out_c):
    """Advance flame walkers ``n_steps`` chaos-game steps, recording every point.

    coefs (n_xf, 6) affine [a b c d e f]; varw (n_xf, n_var) variation weights;
    cum (n_xf,) cumulative pick probabilities; xcol (n_xf,) colors. With
    ``sym`` > 1, half the steps instead rotate by a multiple of 2 pi / sym.
    """
    n = x.shape[0]
    nx = coefs.shape[0]
    nv = varw.shape[1]
    for i in prange(n):
        px = x[i]
        py = y[i]
        pc = col[i]
        for s in range(n_steps):
            t = t0 + s
            u = _rand01(key, i, t * 4)
            if sym > 1 and u < 0.5:
                kk = 1 + int(_rand01(key, i, t * 4 + 1) * (sym - 1))
                a = 2.0 * np.pi * kk / sym
                ca, sa = np.cos(a), np.sin(a)
                px, py = ca * px - sa * py, sa * px + ca * py
            else:
                u2 = _rand01(key, i, t * 4 + 2)
                j = 0
                while j < nx - 1 and cum[j] <= u2:
                    j += 1
                tx = coefs[j, 0] * px + coefs[j, 1] * py + coefs[j, 4]
                ty = coefs[j, 2] * px + coefs[j, 3] * py + coefs[j, 5]
                vx = 0.0
                vy = 0.0
                for v in range(nv):
                    wv = varw[j, v]
                    if wv != 0.0:
                        ox, oy = _flame_variation(v, tx, ty, key, i, t)
                        vx += wv * ox
                        vy += wv * oy
                px = vx
                py = vy
                pc = pc + (xcol[j] - pc) * cspeed
            if not (np.isfinite(px) and np.isfinite(py)) or abs(px) > 1e8 or abs(py) > 1e8:
                px = _rand01(key, i, t * 4 + 3) * 2.0 - 1.0
                py = _rand01(key, i ^ 0x5555, t * 4 + 3) * 2.0 - 1.0
            out_x[s, i] = px
            out_y[s, i] = py
            out_c[s, i] = pc
        x[i] = px
        y[i] = py
        col[i] = pc


# ---- buddhabrot ----------------------------------------------------------------

@njit(fastmath=True)
def _bb_step(zr, zi, cr, ci, kind):
    if kind == 2 or kind == 3:  # burning ship / buffalo fold
        zr = abs(zr)
        zi = abs(zi)
    elif kind == 4:  # tricorn
        zi = -zi
    if kind == 7:
        zr2 = zr * zr
        zi2 = zi * zi
        return zr * (zr2 - 3.0 * zi2) + cr, zi * (3.0 * zr2 - zi2) + ci
    if kind == 8:
        a = zr * zr - zi * zi
        b = 2.0 * zr * zi
        return a * a - b * b + cr, 2.0 * a * b + ci
    nr = zr * zr - zi * zi
    ni = 2.0 * zr * zi
    if kind == 6:  # celtic
        nr = abs(nr)
    if kind == 3:  # buffalo: |z|^2 - |z| + c
        return nr - zr + cr, ni - zi + ci
    return nr + cr, ni + ci


@njit(parallel=True, fastmath=True)
def buddhabrot_escape(n, key, i0, x0, x1, y0, y1, max_iter, bailout2, kind):
    """Pick ``n`` random c in the window and return (cr, ci, escape iteration or -1)."""
    cr = np.empty(n)
    ci = np.empty(n)
    out = np.full(n, -1, dtype=np.int64)
    for k in prange(n):
        a = x0 + (x1 - x0) * _rand01(key, i0 + k, 0)
        b = y0 + (y1 - y0) * _rand01(key, i0 + k, 1)
        cr[k] = a
        ci[k] = b
        if kind == 0:  # skip the main cardioid and period-2 bulb: never escape
            q = (a - 0.25) * (a - 0.25) + b * b
            if q * (q + (a - 0.25)) <= 0.25 * b * b or (a + 1.0) * (a + 1.0) + b * b <= 0.0625:
                continue
        zr = 0.0
        zi = 0.0
        for it in range(max_iter):
            zr, zi = _bb_step(zr, zi, a, b, kind)
            if zr * zr + zi * zi > bailout2:
                out[k] = it + 1
                break
    return cr, ci, out


@njit(parallel=True, fastmath=True)
def buddhabrot_orbits(cr, ci, steps, offsets, kind, ax, bx, cx, ay, by, cy, w, h, pix):
    """Replay each orbit's first ``steps[k]`` points into flat pixel indices (-1 = off frame)."""
    for k in prange(cr.shape[0]):
        zr = 0.0
        zi = 0.0
        o = offsets[k]
        for s in range(steps[k]):
            zr, zi = _bb_step(zr, zi, cr[k], ci[k], kind)
            fx = ax * zr + bx * zi + cx + 0.5
            fy = ay * zr + by * zi + cy + 0.5
            if fx >= 0.0 and fy >= 0.0 and fx < w and fy < h:
                pix[o + s] = int(fy) * w + int(fx)
            else:
                pix[o + s] = -1


@njit(fastmath=True)
def buddhabrot_splat(acc, pix, steps, offsets, chan_mask):
    """acc (n_chan, H*W): add each orbit's pixels to the channels in its bitmask."""
    nch = acc.shape[0]
    for k in range(steps.shape[0]):
        m = chan_mask[k]
        o = offsets[k]
        for s in range(steps[k]):
            p = pix[o + s]
            if p < 0:
                continue
            for ch in range(nch):
                if (m >> ch) & 1:
                    acc[ch, p] += 1.0


# ---- area-preserving maps (standard map, Henon) ----------------------------------

@njit(parallel=True, fastmath=True)
def symplectic_orbits(x0, y0, steps, kind, k, cov_box, cov_n, out_x, out_y):
    """Iterate orbits of an area-preserving map; return per-orbit coverage of a
    cov_n x cov_n grid over cov_box (chaotic orbits wander, regular ones trace curves).

    kind 0: Chirikov standard map on the torus [0, 2pi)^2 (k = kick strength).
    kind 1: Henon's quadratic area-preserving map (k = rotation angle, radians).
    Escaped orbits are NaN from the escape on.
    """
    n = x0.shape[0]
    cov = np.zeros(n)
    two_pi = 2.0 * np.pi
    ck, sk = np.cos(k), np.sin(k)
    bx0, bx1, by0, by1 = cov_box[0], cov_box[1], cov_box[2], cov_box[3]
    for i in prange(n):
        seen = np.zeros(cov_n * cov_n, dtype=np.uint8)
        hits = 0
        x = x0[i]
        y = y0[i]
        alive = True
        for s in range(steps):
            if alive:
                if kind == 0:
                    y = y + k * np.sin(x)
                    y -= two_pi * np.floor(y / two_pi)
                    x = x + y
                    x -= two_pi * np.floor(x / two_pi)
                else:
                    q = y - x * x
                    x, y = x * ck - q * sk, x * sk + q * ck
                    if abs(x) > 10.0 or abs(y) > 10.0:
                        alive = False
            if alive:
                out_x[i, s] = x
                out_y[i, s] = y
                gx = int((x - bx0) / (bx1 - bx0) * cov_n)
                gy = int((y - by0) / (by1 - by0) * cov_n)
                if gx >= 0 and gy >= 0 and gx < cov_n and gy < cov_n:
                    cell = gy * cov_n + gx
                    if seen[cell] == 0:
                        seen[cell] = 1
                        hits += 1
            else:
                out_x[i, s] = np.nan
                out_y[i, s] = np.nan
        cov[i] = hits / float(cov_n * cov_n)
    return cov


@njit(fastmath=True)
def splat_periodic(acc, xs, ys, rgb, period, i0, i1, j0, j1, ax, bx, cx, ay, by, cy):
    """Splat orbits (n_orbits, steps) with one color each into acc[H, W, 4],
    repeating every ``period`` in x and y for tile offsets i0..i1, j0..j1."""
    h, w = acc.shape[0], acc.shape[1]
    for o in range(xs.shape[0]):
        r = rgb[o, 0]
        g = rgb[o, 1]
        b = rgb[o, 2]
        for s in range(xs.shape[1]):
            x = xs[o, s]
            y = ys[o, s]
            if not np.isfinite(x):
                continue
            for ti in range(i0, i1 + 1):
                for tj in range(j0, j1 + 1):
                    px = x + ti * period
                    py = y + tj * period
                    fx = ax * px + bx * py + cx + 0.5
                    fy = ay * px + by * py + cy + 0.5
                    if fx < 0.0 or fy < 0.0 or fx >= w or fy >= h:
                        continue
                    ix = int(fx)
                    iy = int(fy)
                    acc[iy, ix, 0] += 1.0
                    acc[iy, ix, 1] += r
                    acc[iy, ix, 2] += g
                    acc[iy, ix, 3] += b


# ---- caustics ------------------------------------------------------------------

@njit(parallel=True, fastmath=True)
def caustic_rays(py0, py1, w_lo, w_hi, ss, inv, fwd, waves, drops, depths, out_x, out_y):
    """Bend a ss x ss-per-pixel grid of vertical light rays through a wavy surface.

    Rows py0..py1 and columns w_lo..w_hi are pixel ranges (may extend past the
    frame). inv: pixel -> plane affine (6,), fwd: plane -> pixel affine (6,).
    waves (n, 5): kx, ky, amp, phase, unused. drops (m, 5): x, y, k, amp, phase.
    Landing = start + depth * grad(height), for each depth (color channel).
    out_x / out_y: (n_depths, rows * ss, cols * ss) in pixel coordinates.
    """
    ncols = (w_hi - w_lo) * ss
    nrows = (py1 - py0) * ss
    nd = depths.shape[0]
    for r in prange(nrows):
        fy = py0 + (r + 0.5) / ss - 0.5
        for cidx in range(ncols):
            fx = w_lo + (cidx + 0.5) / ss - 0.5
            x = inv[0] * fx + inv[1] * fy + inv[2]
            y = inv[3] * fx + inv[4] * fy + inv[5]
            gx = 0.0
            gy = 0.0
            for q in range(waves.shape[0]):
                kx = waves[q, 0]
                ky = waves[q, 1]
                d = waves[q, 2] * np.cos(kx * x + ky * y + waves[q, 3])
                gx += d * kx
                gy += d * ky
            for q in range(drops.shape[0]):
                dx = x - drops[q, 0]
                dy = y - drops[q, 1]
                rr = np.sqrt(dx * dx + dy * dy) + 1e-9
                kq = drops[q, 2]
                d = drops[q, 3] * kq * np.cos(kq * rr + drops[q, 4]) / np.sqrt(1.0 + kq * rr)
                gx += d * dx / rr
                gy += d * dy / rr
            for m in range(nd):
                lx = x + depths[m] * gx
                ly = y + depths[m] * gy
                out_x[m, r, cidx] = fwd[0] * lx + fwd[1] * ly + fwd[2]
                out_y[m, r, cidx] = fwd[3] * lx + fwd[4] * ly + fwd[5]


@njit(fastmath=True)
def splat_bilinear(dens, px, py, weight):
    """Serial bilinear splat of pixel-space points into dens (H, W)."""
    h, w = dens.shape
    flat_x = px.reshape(-1)
    flat_y = py.reshape(-1)
    for i in range(flat_x.shape[0]):
        fx = flat_x[i]
        fy = flat_y[i]
        if not (fx > -1.0 and fy > -1.0 and fx < w and fy < h):
            continue
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        tx = fx - x0
        ty = fy - y0
        for oy in range(2):
            yy = y0 + oy
            if yy < 0 or yy >= h:
                continue
            wy = ty if oy else 1.0 - ty
            for ox in range(2):
                xx = x0 + ox
                if xx < 0 or xx >= w:
                    continue
                wx = tx if ox else 1.0 - tx
                dens[yy, xx] += weight * wx * wy
