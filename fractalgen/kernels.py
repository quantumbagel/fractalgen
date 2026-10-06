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
def _mandelbulb_de(px, py, pz, power, max_iter):
    x, y, z = px, py, pz
    dr = 1.0
    r = 0.0
    trap = 1e9
    for i in range(max_iter):
        r = np.sqrt(x * x + y * y + z * z)
        if r > 6.0:
            break
        if r < trap:
            trap = r
        if r < 1e-8:
            r = 1e-8
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        zr = r**power
        dr = (r ** (power - 1.0)) * power * dr + 1.0
        theta *= power
        phi *= power
        st = np.sin(theta)
        x = zr * st * np.cos(phi) + px
        y = zr * st * np.sin(phi) + py
        z = zr * np.cos(theta) + pz
    return 0.5 * np.log(r) * r / dr, trap, float(i)


@njit(fastmath=True)
def _quat_julia_de(px, py, pz, cr, ci, cj, ck, qw, max_iter):
    zr, zi, zj, zk = px, py, pz, qw
    dr = 1.0
    r2 = 0.0
    trap = 1e9
    for i in range(max_iter):
        r2 = zr * zr + zi * zi + zj * zj + zk * zk
        if r2 > 16.0:
            break
        r = np.sqrt(r2)
        if r < trap:
            trap = r
        dr = 2.0 * r * dr + 1.0
        nr = zr * zr - zi * zi - zj * zj - zk * zk
        ni = 2.0 * zr * zi
        nj = 2.0 * zr * zj
        nk = 2.0 * zr * zk
        zr = nr + cr
        zi = ni + ci
        zj = nj + cj
        zk = nk + ck
    r = np.sqrt(r2) if r2 > 0.0 else 0.0
    return 0.5 * np.log(max(r, 1e-8)) * r / dr, trap, float(i)


@njit(parallel=True, fastmath=True)
def fractal3d_kernel(h, w, kind, camx, camy, camz, tx, ty, tz, max_steps, max_iter, power, jc):
    """Raymarch Mandelbulb (kind=0) or quaternionic Julia (kind=1)."""
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    aspect = w / float(h)
    fx, fy, fz = tx - camx, ty - camy, tz - camz
    fl = np.sqrt(fx * fx + fy * fy + fz * fz) + 1e-12
    fx /= fl
    fy /= fl
    fz /= fl
    rx = fy * 0.0 - fz * 1.0
    ry = fz * 0.0 - fx * 0.0
    rz = fx * 1.0 - fy * 0.0
    rl = np.sqrt(rx * rx + ry * ry + rz * rz)
    if rl < 1e-8:
        rx, ry, rz = 1.0, 0.0, 0.0
        rl = 1.0
    rx /= rl
    ry /= rl
    rz /= rl
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    fov = 0.42
    cr, ci, cj, ck, qw = jc[0], jc[1], jc[2], jc[3], jc[4]
    for y in prange(h):
        for x in range(w):
            u = (2.0 * x / (w - 1.0) - 1.0) * aspect * fov
            v = (1.0 - 2.0 * y / (h - 1.0)) * fov
            dx = fx + u * rx + v * ux
            dy = fy + u * ry + v * uy
            dz = fz + u * rz + v * uz
            dl = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
            dx /= dl
            dy /= dl
            dz /= dl
            t = 0.0
            hit = False
            px = py = pz = 0.0
            trap = 0.0
            for _s in range(max_steps):
                px = camx + dx * t
                py = camy + dy * t
                pz = camz + dz * t
                if kind == 0:
                    de, trap, _it = _mandelbulb_de(px, py, pz, power, max_iter)
                else:
                    de, trap, _it = _quat_julia_de(px, py, pz, cr, ci, cj, ck, qw, max_iter)
                if de < 0.0012 * (1.0 + t):
                    hit = True
                    break
                t += max(de, 0.0015)
                if t > 9.0:
                    break
            if not hit:
                # smoky void
                vig = 0.04 + 0.03 * (1.0 - 0.5 * (u * u + v * v))
                rgb[y, x, 0] = 0.035 * vig
                rgb[y, x, 1] = 0.028 * vig
                rgb[y, x, 2] = 0.022 * vig
                continue
            # normal via tetrahedron
            e = 0.0018
            if kind == 0:
                d1, _, _ = _mandelbulb_de(px + e, py - e, pz - e, power, max_iter)
                d2, _, _ = _mandelbulb_de(px - e, py - e, pz + e, power, max_iter)
                d3, _, _ = _mandelbulb_de(px - e, py + e, pz - e, power, max_iter)
                d4, _, _ = _mandelbulb_de(px + e, py + e, pz + e, power, max_iter)
            else:
                d1, _, _ = _quat_julia_de(px + e, py - e, pz - e, cr, ci, cj, ck, qw, max_iter)
                d2, _, _ = _quat_julia_de(px - e, py - e, pz + e, cr, ci, cj, ck, qw, max_iter)
                d3, _, _ = _quat_julia_de(px - e, py + e, pz - e, cr, ci, cj, ck, qw, max_iter)
                d4, _, _ = _quat_julia_de(px + e, py + e, pz + e, cr, ci, cj, ck, qw, max_iter)
            nx = (d1 + d4) - (d2 + d3)
            ny = (d3 + d4) - (d1 + d2)
            nz = (d2 + d4) - (d1 + d3)
            nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
            nx /= nl
            ny /= nl
            nz /= nl
            # key + fill + rim
            lx, ly, lz = -0.45, 0.72, -0.35
            ll = np.sqrt(lx * lx + ly * ly + lz * lz)
            lx /= ll
            ly /= ll
            lz /= ll
            ndl = max(0.0, nx * lx + ny * ly + nz * lz)
            # cheap AO
            ao = 1.0
            ao_t = 0.018
            for k in range(5):
                qx = px + nx * ao_t
                qy = py + ny * ao_t
                qz = pz + nz * ao_t
                if kind == 0:
                    de, _, _ = _mandelbulb_de(qx, qy, qz, power, max_iter)
                else:
                    de, _, _ = _quat_julia_de(qx, qy, qz, cr, ci, cj, ck, qw, max_iter)
                ao_s = de / ao_t
                if ao_s < 0.15:
                    ao_s = 0.15
                elif ao_s > 1.0:
                    ao_s = 1.0
                ao *= ao_s
                ao_t *= 1.7
            # soft shadow
            sh = 1.0
            st = 0.03
            for _k in range(12):
                qx = px + lx * st
                qy = py + ly * st
                qz = pz + lz * st
                if kind == 0:
                    de, _, _ = _mandelbulb_de(qx, qy, qz, power, max_iter)
                else:
                    de, _, _ = _quat_julia_de(qx, qy, qz, cr, ci, cj, ck, qw, max_iter)
                sh = min(sh, 6.0 * de / st)
                st += max(de, 0.02)
                if st > 3.5 or sh < 0.08:
                    break
            sh = max(0.12, sh)
            # bronze / patina from orbit trap + facing
            pat = trap * 0.55
            if pat < 0.0:
                pat = 0.0
            elif pat > 1.0:
                pat = 1.0
            gold = ndl * ao
            br = 0.10 + 0.38 * pat + 0.42 * gold
            bg_ = 0.07 + 0.18 * pat + 0.22 * gold + 0.16 * (1.0 - pat) * 0.35
            bb = 0.04 + 0.08 * pat + 0.10 * gold + 0.14 * (1.0 - ndl)
            # verdigris in occluded crevices
            cave = (1.0 - ao) * (1.0 - ndl)
            br = br * (1.0 - 0.45 * cave) + 0.10 * cave
            bg_ = bg_ * (1.0 - 0.20 * cave) + 0.28 * cave
            bb = bb * (1.0 - 0.25 * cave) + 0.18 * cave
            spec = (max(0.0, nx * 0.15 + ny * 0.75 + nz * 0.55)) ** 22.0
            amb = 0.28 + 0.72 * ao
            br = (br * (0.35 + 0.65 * ndl * sh) + 0.70 * spec) * amb
            bg_ = (bg_ * (0.35 + 0.65 * ndl * sh) + 0.52 * spec) * amb
            bb = (bb * (0.35 + 0.65 * ndl * sh) + 0.32 * spec) * amb
            rgb[y, x, 0] = br
            rgb[y, x, 1] = bg_
            rgb[y, x, 2] = bb
    return rgb


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


@njit(fastmath=True)
def ferro_height(x, z, spacing, h0):
    sx = spacing
    sz = spacing * 0.8660254037844386
    col = x / sx
    row = z / sz
    i0 = int(np.floor(col + 0.5))
    j0 = int(np.floor(row + 0.5))
    best = 0.0
    base = 0.018
    for dj in range(-2, 3):
        for di in range(-2, 3):
            i = i0 + di
            j = j0 + dj
            cx = (i + 0.5 * (j & 1)) * sx
            cz = j * sz
            dx = x - cx
            dz = z - cz
            d = np.sqrt(dx * dx + dz * dz)
            # organic jitter of peak height
            jitter = 0.82 + 0.18 * np.sin(i * 1.7 + j * 2.3) * np.cos(j * 1.1)
            u = d / (0.34 * spacing)
            tip = np.exp(-4.4 * (u**1.75)) * jitter
            if tip > best:
                best = tip
    # shallow liquid meniscus between spikes
    return base + h0 * best


@njit(parallel=True, fastmath=True)
def ferrofluid_kernel(h, w, spacing, h0, camx, camy, camz, tx, ty, tz):
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    aspect = w / float(h)
    fx, fy, fz = tx - camx, ty - camy, tz - camz
    fl = np.sqrt(fx * fx + fy * fy + fz * fz) + 1e-12
    fx /= fl
    fy /= fl
    fz /= fl
    rx = fy * 0.0 - fz * 1.0
    ry = fz * 0.0 - fx * 0.0
    rz = fx * 1.0 - fy * 0.0
    rl = np.sqrt(rx * rx + ry * ry + rz * rz) + 1e-12
    rx /= rl
    ry /= rl
    rz /= rl
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    fov = 0.38
    for y in prange(h):
        for x in range(w):
            u = (2.0 * x / (w - 1.0) - 1.0) * aspect * fov
            v = (1.0 - 2.0 * y / (h - 1.0)) * fov
            dx = fx + u * rx + v * ux
            dy = fy + u * ry + v * uy
            dz = fz + u * rz + v * uz
            dl = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
            dx /= dl
            dy /= dl
            dz /= dl
            t = 0.4
            hit = False
            px = py = pz = 0.0
            for _s in range(90):
                px = camx + dx * t
                py = camy + dy * t
                pz = camz + dz * t
                hh = ferro_height(px, pz, spacing, h0)
                dist = py - hh
                if dist < 0.002:
                    hit = True
                    py = hh
                    break
                t += max(0.012, dist * 0.85)
                if t > 8.5:
                    break
            if not hit:
                rgb[y, x, 0] = 0.012
                rgb[y, x, 1] = 0.012
                rgb[y, x, 2] = 0.014
                continue
            e = 0.012
            hx = ferro_height(px + e, pz, spacing, h0) - ferro_height(px - e, pz, spacing, h0)
            hz = ferro_height(px, pz + e, spacing, h0) - ferro_height(px, pz - e, spacing, h0)
            nx, ny, nz = -hx / (2.0 * e), 1.0, -hz / (2.0 * e)
            nl = np.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= nl
            ny /= nl
            nz /= nl
            # cool key + warm rim
            l1x, l1y, l1z = -0.55, 0.72, 0.40
            l1n = np.sqrt(l1x * l1x + l1y * l1y + l1z * l1z)
            l1x /= l1n
            l1y /= l1n
            l1z /= l1n
            l2x, l2y, l2z = 0.65, 0.25, -0.55
            l2n = np.sqrt(l2x * l2x + l2y * l2y + l2z * l2z)
            l2x /= l2n
            l2y /= l2n
            l2z /= l2n
            nd1 = max(0.0, nx * l1x + ny * l1y + nz * l1z)
            nd2 = max(0.0, nx * l2x + ny * l2y + nz * l2z)
            # half-vector spec
            hx1 = l1x - dx
            hy1 = l1y - dy
            hz1 = l1z - dz
            hn = np.sqrt(hx1 * hx1 + hy1 * hy1 + hz1 * hz1) + 1e-12
            ndh = max(0.0, (nx * hx1 + ny * hy1 + nz * hz1) / hn)
            spec1 = ndh**110.0
            spec2 = max(0.0, nx * l2x + ny * l2y + nz * l2z) ** 55.0
            ndv = max(0.0, -(nx * dx + ny * dy + nz * dz))
            fres = 0.08 + 0.92 * (1.0 - ndv) ** 4.0
            valley = 1.0 - (py / (h0 + 0.05))
            if valley < 0.0:
                valley = 0.0
            elif valley > 1.0:
                valley = 1.0
            film = 0.5 + 0.5 * np.sin(valley * 28.0 + px * 9.0 + pz * 6.5)
            oil = valley * valley * 0.70
            r = 0.008 + 0.010 * nd1 + oil * (0.10 + 0.22 * film)
            g = 0.008 + 0.012 * nd1 + oil * (0.06 + 0.14 * (1.0 - film))
            b = 0.010 + 0.018 * nd1 + oil * (0.14 + 0.20 * (1.0 - film))
            r += 0.06 * nd1 + 1.15 * spec1 * fres + 0.55 * spec2 * fres + 0.18 * nd2
            g += 0.08 * nd1 + 1.05 * spec1 * fres + 0.28 * spec2 * fres + 0.08 * nd2
            b += 0.12 * nd1 + 1.00 * spec1 * fres + 0.10 * spec2 * fres + 0.03 * nd2
            rgb[y, x, 0] = r
            rgb[y, x, 1] = g
            rgb[y, x, 2] = b
    return rgb


@njit(parallel=True, fastmath=True)
def volume_march_kernel(h, w, dens, phase, extent, camx, camy, camz, tx, ty, tz):
    """Emission-absorption march through a cubic complex-orbital volume."""
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    n = dens.shape[0]
    aspect = w / float(h)
    fx, fy, fz = tx - camx, ty - camy, tz - camz
    fl = np.sqrt(fx * fx + fy * fy + fz * fz) + 1e-12
    fx /= fl
    fy /= fl
    fz /= fl
    rx = fy * 0.0 - fz * 1.0
    ry = fz * 0.0 - fx * 0.0
    rz = fx * 1.0 - fy * 0.0
    rl = np.sqrt(rx * rx + ry * ry + rz * rz) + 1e-12
    rx /= rl
    ry /= rl
    rz /= rl
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    fov = 0.36
    n1 = n - 1.0
    for y in prange(h):
        for x in range(w):
            u = (2.0 * x / (w - 1.0) - 1.0) * aspect * fov
            v = (1.0 - 2.0 * y / (h - 1.0)) * fov
            dx = fx + u * rx + v * ux
            dy = fy + u * ry + v * uy
            dz = fz + u * rz + v * uz
            dl = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
            dx /= dl
            dy /= dl
            dz /= dl
            t = 6.0
            er = eg = eb = 0.0
            trans = 1.0
            for _s in range(160):
                px = camx + dx * t
                py = camy + dy * t
                pz = camz + dz * t
                gx = (px + extent) / (2.0 * extent) * n1
                gy = (py + extent) / (2.0 * extent) * n1
                gz = (pz + extent) / (2.0 * extent) * n1
                if gx >= 0.0 and gy >= 0.0 and gz >= 0.0 and gx < n1 and gy < n1 and gz < n1:
                    ix = int(gx)
                    iy = int(gy)
                    iz = int(gz)
                    fx0 = gx - ix
                    fy0 = gy - iy
                    fz0 = gz - iz
                    d000 = dens[ix, iy, iz]
                    d100 = dens[ix + 1, iy, iz]
                    d010 = dens[ix, iy + 1, iz]
                    d110 = dens[ix + 1, iy + 1, iz]
                    d001 = dens[ix, iy, iz + 1]
                    d101 = dens[ix + 1, iy, iz + 1]
                    d011 = dens[ix, iy + 1, iz + 1]
                    d111 = dens[ix + 1, iy + 1, iz + 1]
                    d = (
                        d000 * (1 - fx0) * (1 - fy0) * (1 - fz0)
                        + d100 * fx0 * (1 - fy0) * (1 - fz0)
                        + d010 * (1 - fx0) * fy0 * (1 - fz0)
                        + d110 * fx0 * fy0 * (1 - fz0)
                        + d001 * (1 - fx0) * (1 - fy0) * fz0
                        + d101 * fx0 * (1 - fy0) * fz0
                        + d011 * (1 - fx0) * fy0 * fz0
                        + d111 * fx0 * fy0 * fz0
                    )
                    p000 = phase[ix, iy, iz]
                    p100 = phase[ix + 1, iy, iz]
                    p010 = phase[ix, iy + 1, iz]
                    p110 = phase[ix + 1, iy + 1, iz]
                    p001 = phase[ix, iy, iz + 1]
                    p101 = phase[ix + 1, iy, iz + 1]
                    p011 = phase[ix, iy + 1, iz + 1]
                    p111 = phase[ix + 1, iy + 1, iz + 1]
                    ph = (
                        p000 * (1 - fx0) * (1 - fy0) * (1 - fz0)
                        + p100 * fx0 * (1 - fy0) * (1 - fz0)
                        + p010 * (1 - fx0) * fy0 * (1 - fz0)
                        + p110 * fx0 * fy0 * (1 - fz0)
                        + p001 * (1 - fx0) * (1 - fy0) * fz0
                        + p101 * fx0 * (1 - fy0) * fz0
                        + p011 * (1 - fx0) * fy0 * fz0
                        + p111 * fx0 * fy0 * fz0
                    )
                    if d > 0.012:
                        # cyclic amethyst → amber
                        hph = 0.5 + 0.5 * ph
                        # lerp stops: 0 amethyst, 0.5 gold, 1 amber
                        if hph < 0.5:
                            tcol = hph * 2.0
                            pr = 0.42 + 0.48 * tcol
                            pg = 0.10 + 0.42 * tcol
                            pb = 0.62 - 0.30 * tcol
                        else:
                            tcol = (hph - 0.5) * 2.0
                            pr = 0.90 + 0.10 * tcol
                            pg = 0.52 + 0.30 * tcol
                            pb = 0.32 - 0.20 * tcol
                        emit = d ** 1.35
                        er += trans * emit * pr * 4.8
                        eg += trans * emit * pg * 4.8
                        eb += trans * emit * pb * 4.8
                        trans *= np.exp(-d * 5.5)
                        if trans < 0.012:
                            break
                t += 0.48
                if t > 88.0:
                    break
            # deep void
            rgb[y, x, 0] = er + 0.012 * trans
            rgb[y, x, 1] = eg + 0.008 * trans
            rgb[y, x, 2] = eb + 0.022 * trans
    return rgb


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
