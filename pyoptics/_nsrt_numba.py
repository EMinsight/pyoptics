#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Compiled kernels of `pyoptics.nsrt`: nearest hits of rays on the
analytic primitive faces and on triangle meshes (bounding volume
hierarchy). Optional; `nsrt` falls back to numpy without numba."""

import numpy as np

try:
    import numba
    available = True
except ImportError:  # pragma: no cover
    numba = None
    available = False

# primitive kinds (see nsrt):
CONIC, CYLINDER, CONE, RECT, SPHERE = 0, 1, 2, 3, 4


def _jit(**kw):
    def deco(f):
        return numba.njit(cache=True, **kw)(f) if available else f
    return deco


@_jit()
def _roots(a, b, C):
    """Both roots of a t**2 + b t + C = 0 (NaN where none); a = 0: linear."""

    nan = np.nan
    if a == 0.0:
        if b == 0.0:
            return nan, nan
        return -C/b, nan
    D = b*b - 4.0*a*C
    if D < 0.0:
        return nan, nan
    sq = np.sqrt(D)
    q = -0.5*(b + sq) if b >= 0.0 else -0.5*(b - sq)
    t1 = q/a
    t2 = C/q if q != 0.0 else nan
    return t1, t2


@_jit()
def _prim_hit(kind, p, x, y, z, dx, dy, dz, eps):
    """Smallest t > eps at which the ray (local frame) hits the bounded
    primitive with parameters p; inf if none."""

    best = np.inf
    if kind == RECT:
        if dz != 0.0:
            t = -z/dz
            if t > eps and abs(x + t*dx) <= p[0] and abs(y + t*dy) <= p[1]:
                best = t
        return best
    if kind == CONIC:
        c, K, rmin, rmax, zs = p[0], p[1], p[2], p[3], p[4]
        z = z - zs
        k1 = 1.0 + K
        a = c*(dx*dx + dy*dy + k1*dz*dz)
        b = 2.0*(c*(x*dx + y*dy + k1*z*dz) - dz)
        C = c*(x*x + y*y + k1*z*z) - 2.0*z
        t1, t2 = _roots(a, b, C)
        for t in (t1, t2):
            if t == t and t > eps and t < best:
                qx, qy, qz = x + t*dx, y + t*dy, z + t*dz
                if k1*c*qz <= 1.0:
                    r2 = qx*qx + qy*qy
                    if r2 >= rmin*rmin and r2 <= rmax*rmax:
                        best = t
        return best
    if kind == CYLINDER:
        R, z0, z1 = p[0], p[1], p[2]
        a = dx*dx + dy*dy
        b = 2.0*(x*dx + y*dy)
        C = x*x + y*y - R*R
        t1, t2 = _roots(a, b, C)
        for t in (t1, t2):
            if t == t and t > eps and t < best:
                qz = z + t*dz
                if qz >= z0 and qz <= z1:
                    best = t
        return best
    if kind == CONE:
        r0, z0, r1, z1 = p[0], p[1], p[2], p[3]
        k = (r1 - r0)/(z1 - z0)
        u = r0 - k*z0
        w = u + k*z
        a = dx*dx + dy*dy - k*k*dz*dz
        b = 2.0*(x*dx + y*dy - k*dz*w)
        C = x*x + y*y - w*w
        t1, t2 = _roots(a, b, C)
        lo, hi = min(z0, z1), max(z0, z1)
        for t in (t1, t2):
            if t == t and t > eps and t < best:
                qz = z + t*dz
                if qz >= lo and qz <= hi and u + k*qz >= 0.0:
                    best = t
        return best
    if kind == SPHERE:
        R = p[0]
        b = 2.0*(x*dx + y*dy + z*dz)
        C = x*x + y*y + z*z - R*R
        t1, t2 = _roots(1.0, b, C)
        for t in (t1, t2):
            if t == t and t > eps and t < best:
                best = t
        return best
    return best


@_jit()
def _aabb_hit(px, py, pz, ix, iy, iz, lo, hi, tmax):
    """Whether the ray (inverse direction ix, iy, iz) meets the box within tmax."""

    t0, t1 = 0.0, tmax
    for k in range(3):
        o = px if k == 0 else (py if k == 1 else pz)
        inv = ix if k == 0 else (iy if k == 1 else iz)
        ta = (lo[k] - o)*inv
        tb = (hi[k] - o)*inv
        if ta > tb:
            ta, tb = tb, ta
        if ta != ta:  # 0*inf: ray in the slab plane
            ta = -np.inf
        if tb != tb:
            tb = np.inf
        if ta > t0:
            t0 = ta
        if tb < t1:
            t1 = tb
        if t0 > t1:
            return False
    return True


@_jit()
def _aabb_entry(px, py, pz, ix, iy, iz, lo, hi, tmax):
    """Distance at which the ray enters the box (0 if it starts inside),
    inf if it misses the box or enters beyond tmax."""

    t0, t1 = 0.0, tmax
    for k in range(3):
        o = px if k == 0 else (py if k == 1 else pz)
        inv = ix if k == 0 else (iy if k == 1 else iz)
        ta = (lo[k] - o)*inv
        tb = (hi[k] - o)*inv
        if ta > tb:
            ta, tb = tb, ta
        if ta != ta:
            ta = -np.inf
        if tb != tb:
            tb = np.inf
        if ta > t0:
            t0 = ta
        if tb < t1:
            t1 = tb
        if t0 > t1:
            return np.inf
    return t0


if available:
    @numba.njit(parallel=True, cache=True)
    def primitives_nearest(P, d, kinds, origins, R, params, lo, hi, eps):
        """Nearest hits (t, face index) of rays on the primitive faces."""

        n = P.shape[0]
        F = kinds.shape[0]
        t_out = np.full(n, np.inf)
        f_out = np.full(n, -1, dtype=np.int64)
        for i in numba.prange(n):
            px, py, pz = P[i, 0], P[i, 1], P[i, 2]
            ex, ey, ez = d[i, 0], d[i, 1], d[i, 2]
            ix = 1.0/ex if ex != 0.0 else np.inf
            iy = 1.0/ey if ey != 0.0 else np.inf
            iz = 1.0/ez if ez != 0.0 else np.inf
            best = np.inf
            bf = -1
            for f in range(F):
                if not _aabb_hit(px, py, pz, ix, iy, iz, lo[f], hi[f], best):
                    continue
                ox, oy, oz = px - origins[f, 0], py - origins[f, 1], pz - origins[f, 2]
                # local = (P - o) @ R:
                x = ox*R[f, 0, 0] + oy*R[f, 1, 0] + oz*R[f, 2, 0]
                y = ox*R[f, 0, 1] + oy*R[f, 1, 1] + oz*R[f, 2, 1]
                z = ox*R[f, 0, 2] + oy*R[f, 1, 2] + oz*R[f, 2, 2]
                dx = ex*R[f, 0, 0] + ey*R[f, 1, 0] + ez*R[f, 2, 0]
                dy = ex*R[f, 0, 1] + ey*R[f, 1, 1] + ez*R[f, 2, 1]
                dz = ex*R[f, 0, 2] + ey*R[f, 1, 2] + ez*R[f, 2, 2]
                t = _prim_hit(kinds[f], params[f], x, y, z, dx, dy, dz, eps)
                if t < best:
                    best = t
                    bf = f
            t_out[i] = best
            f_out[i] = bf
        return t_out, f_out

    @numba.njit(parallel=True, cache=True)
    def bvh_nearest(P, d, V0, E1, E2, node_lo, node_hi, node_a, node_b, node_leaf, order,
                    eps, tmax):
        """Nearest hits (t, triangle) of rays on a triangle mesh with a BVH,
        closer than tmax per ray (inf, -1 if none). Nodes: leaf -> triangles
        order[a:b]; inner -> children a, b, visited nearer box first."""

        n = P.shape[0]
        t_out = np.full(n, np.inf)
        tri_out = np.full(n, -1, dtype=np.int64)
        for i in numba.prange(n):
            px, py, pz = P[i, 0], P[i, 1], P[i, 2]
            ex, ey, ez = d[i, 0], d[i, 1], d[i, 2]
            ix = 1.0/ex if ex != 0.0 else np.inf
            iy = 1.0/ey if ey != 0.0 else np.inf
            iz = 1.0/ez if ez != 0.0 else np.inf
            stack = np.empty(128, dtype=np.int64)
            stack[0] = 0
            sp = 1
            best = tmax[i]
            bt = -1
            while sp > 0:
                sp -= 1
                nd = stack[sp]
                # re-test: best may have shrunk since the node was pushed
                if _aabb_entry(px, py, pz, ix, iy, iz, node_lo[nd], node_hi[nd], best) == np.inf:
                    continue
                if node_leaf[nd]:
                    for k in range(node_a[nd], node_b[nd]):
                        j = order[k]
                        # Moller-Trumbore:
                        e1x, e1y, e1z = E1[j, 0], E1[j, 1], E1[j, 2]
                        e2x, e2y, e2z = E2[j, 0], E2[j, 1], E2[j, 2]
                        hx = ey*e2z - ez*e2y
                        hy = ez*e2x - ex*e2z
                        hz = ex*e2y - ey*e2x
                        a = e1x*hx + e1y*hy + e1z*hz
                        if a == 0.0:
                            continue
                        f = 1.0/a
                        sx, sy, sz = px - V0[j, 0], py - V0[j, 1], pz - V0[j, 2]
                        u = f*(sx*hx + sy*hy + sz*hz)
                        if u < 0.0 or u > 1.0:
                            continue
                        qx = sy*e1z - sz*e1y
                        qy = sz*e1x - sx*e1z
                        qz = sx*e1y - sy*e1x
                        v = f*(ex*qx + ey*qy + ez*qz)
                        if v < 0.0 or u + v > 1.0:
                            continue
                        t = f*(e2x*qx + e2y*qy + e2z*qz)
                        # ties (rays through shared edges) go to the lowest
                        # triangle index, independent of the tree, as in numpy
                        if t > eps and (t < best or (t == best and 0 <= j < bt)):
                            best = t
                            bt = j
                elif sp + 2 <= stack.shape[0]:
                    c0, c1 = node_a[nd], node_b[nd]
                    t0 = _aabb_entry(px, py, pz, ix, iy, iz, node_lo[c0], node_hi[c0], best)
                    t1 = _aabb_entry(px, py, pz, ix, iy, iz, node_lo[c1], node_hi[c1], best)
                    if t0 > t1:
                        c0, c1 = c1, c0
                        t0, t1 = t1, t0
                    # push the farther child first, so the nearer one is popped next
                    if t1 < np.inf:
                        stack[sp] = c1
                        sp += 1
                    if t0 < np.inf:
                        stack[sp] = c0
                        sp += 1
            if bt >= 0:
                t_out[i] = best
                tri_out[i] = bt
        return t_out, tri_out
