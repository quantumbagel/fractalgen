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
                if d < best:
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
