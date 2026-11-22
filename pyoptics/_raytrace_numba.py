"""Compiled ray trace kernel for raytrace.System.trace() (optional, needs numba).

The kernel traces each ray through all surfaces in one compiled loop instead
of one set of NumPy array operations per surface, which removes the Python
and temporary array overhead that dominates the trace of small and medium
bundles (ray aiming traces many of them). It covers Surface (conics, even
aspheres, mirrors), IdealLens, Axicon, CoordinateBreak, ToroidalSurface,
BiconicSurface and the Q-type aspheres (QbfsSurface, QconSurface,
Q2DSurface, QconQ2DSurface) with the same formulas as their NumPy methods (the Chebyshev
series of the Q polynomials by Clenshaw's recurrence); systems with other
surface types are traced by NumPy. Newton iterations stop ray by ray
instead of when the whole bundle has converged, so iterated intersections
agree with NumPy to rounding (about 1e-12 of the lengths), not bit for bit.

``available`` tells whether numba could be imported.
"""

import numpy as np

try:
    import numba as nb
except ImportError:  # pragma: no cover - depends on the installation
    nb = None

available = nb is not None

# surface kinds:
STANDARD, BREAK, IDEAL, AXICON, TOROID, BICONIC, QBFS, QCON, Q2D, QCONQ2D = range(10)


def _chebval(coef, n, t):
    """sum_k coef[k] T_k(t) for k < n (Clenshaw)."""

    if n == 0:
        return 0.0
    b1, b2 = 0.0, 0.0
    for k in range(n - 1, 0, -1):
        b0 = 2.0*t*b1 - b2 + coef[k]
        b2, b1 = b1, b0
    return t*b1 - b2 + coef[0]


def _q_sag_slope(kind, c, K, Qc, dQc, n_Q, rn2, r2):
    """Sag and (dz/dr)/r of a Q-bfs (c: curvature of the best fit sphere)
    or Q-con (c, K: base conic) asphere; Qc, dQc: Chebyshev series in
    t = 2 x - 1 of the polynomial sum and of its derivative in t."""

    if kind == QBFS:
        root = np.sqrt(1.0 - c*c*r2)
        z = c*r2/(1.0 + root)
        s = c/root
        if n_Q > 0:
            x = r2/rn2
            t = 2.0*x - 1.0
            S = _chebval(Qc, n_Q, t)
            dS = 2.0*_chebval(dQc, n_Q - 1, t)
            z += x*(1.0 - x)*S/root
            dD = ((1.0 - 2.0*x)*S + x*(1.0 - x)*dS)/rn2/root \
                + x*(1.0 - x)*S*c*c/(2.0*root**3)
            s += 2.0*dD
        return z, s
    sq = np.sqrt(1.0 - (1.0 + K)*c*c*r2)
    z = c*r2/(1.0 + sq)
    s = c/sq
    if n_Q > 0:
        x = r2/rn2
        t = 2.0*x - 1.0
        S = _chebval(Qc, n_Q, t)
        dS = 2.0*_chebval(dQc, n_Q - 1, t)
        z += x*x*S
        s += 2.0*(2.0*x*S + x*x*dS)/rn2
    return z, s


def _q2d_departure(rn, g0, g1, g_kind, g_m, g_coef, g_dcoef, g_n, x, y):
    """Q2D departure D(x, y) and its gradient, groups g0 to g1, normalization
    radius rn."""

    wr, wi = x/rn, y/rn
    X = wr*wr + wi*wi
    t = 2.0*X - 1.0
    D, Dx, Dy = 0.0, 0.0, 0.0
    for k in range(g0, g1):
        P = _chebval(g_coef[k], g_n[k], t)
        dP = 2.0*_chebval(g_dcoef[k], g_n[k] - 1, t)
        m = g_m[k]
        if m == 0:
            fq = X*(1.0 - X)*P
            df = (1.0 - 2.0*X)*P + X*(1.0 - X)*dP
            D += fq
            Dx += df*2.0*wr/rn
            Dy += df*2.0*wi/rn
            continue
        # w**m and m w**(m - 1), w = wr + i wi
        ar, ai = 1.0, 0.0
        for _ in range(m - 1):
            ar, ai = ar*wr - ai*wi, ar*wi + ai*wr
        br, bi = m*ar, m*ai
        ar, ai = ar*wr - ai*wi, ar*wi + ai*wr
        if g_kind[k] == 0:
            Am, Ax, Ay = ar, br, -bi
        else:
            Am, Ax, Ay = ai, bi, br
        D += P*Am
        Dx += dP*2.0*wr/rn*Am + P*Ax/rn
        Dy += dP*2.0*wi/rn*Am + P*Ay/rn
    return D, Dx, Dy


def _xy_sag_grad(kind, c, K, A, n_A, cx, Kx, B, n_B, rn, g0, g1, g_kind, g_m, g_coef, g_dcoef,
                 g_n, x, y):
    """Sag z(x, y) and gradient of a toroidal, biconic or Q2D surface."""

    if kind == TOROID:
        y2 = y*y
        zy, s = _sag_slope(STANDARD, c, K, A, n_A, 0.0, y2)
        u = 1.0 - cx*zy
        S = np.sqrt(u*u - cx*cx*x*x)
        z = zy + cx*x*x/(u + S)
        if not u > 0:
            z = np.nan
        return z, cx*x/S, u*y*s/S
    if kind == BICONIC:
        x2, y2 = x*x, y*y
        px, py = (1.0 + Kx)*cx*cx, (1.0 + K)*c*c
        f = cx*x2 + c*y2
        S = np.sqrt(1.0 - px*x2 - py*y2)
        g = 1.0 + S
        z = f/g
        gx, gy = -px*x/S, -py*y/S
        zx = (2.0*cx*x*g - f*gx)/(g*g)
        zy = (2.0*c*y*g - f*gy)/(g*g)
        p = y2
        for j in range(n_A):
            z += A[j]*p*y2
            zy += 2*(j + 2)*A[j]*y*p
            p *= y2
        p = x2
        for j in range(n_B):
            z += B[j]*p*x2
            zx += 2*(j + 2)*B[j]*x*p
            p *= x2
        return z, zx, zy
    # Q2D: best fit sphere (c) plus D/root
    r2 = x*x + y*y
    root = np.sqrt(1.0 - c*c*r2)
    D, Dx, Dy = _q2d_departure(rn, g0, g1, g_kind, g_m, g_coef, g_dcoef, g_n, x, y)
    s = c/root
    kk = D*c*c/root**3
    z = c*r2/(1.0 + root) + D/root

    return z, x*s + Dx/root + kk*x, y*s + Dy/root + kk*y

def _sag_slope(kind, c, K, A, n_A, cone, r2):
    """Sag z(r2) and (dz/dr)/r of a conic with even aspheric terms or of a
    cone (slope cone)."""

    if kind == AXICON:
        return cone*np.sqrt(r2), cone/np.sqrt(max(r2, 1e-300))
    q = 1.0 - (1.0 + K)*c*c*r2
    sq = np.sqrt(q)  # NaN beyond the conic
    z = c*r2/(1.0 + sq)
    s = c/sq
    p = r2
    for j in range(n_A):
        z += A[j]*p*r2  # A_j r2**(j + 2)
        s += 2*(j + 2)*A[j]*p  # 2 (j + 2) A_j r2**(j + 1)
        p *= r2

    return z, s


def _intersect_conic(c, K, A, n_A, x, y, z, ex, ey, ez):
    a = c*(ex*ex + ey*ey + (1.0 + K)*ez*ez)
    b = 2.0*(c*(x*ex + y*ey + (1.0 + K)*z*ez) - ez)
    C = c*(x*x + y*y + (1.0 + K)*z*z) - 2.0*z
    D = b*b - 4.0*a*C
    # root on the branch through the vertex, stable for c -> 0:
    t = -2.0*C/(b + (1.0 if b >= 0 else -1.0)*np.sqrt(D))
    if (1.0 + K)*c*(z + t*ez) > 1.0:
        t = C/(a*t) if t != 0.0 else -b/a
    if n_A > 0:
        if not np.isfinite(t):
            t = -z/ez
        prev = np.inf
        for _ in range(50):
            qx, qy, qz = x + t*ex, y + t*ey, z + t*ez
            zs, s = _sag_slope(STANDARD, c, K, A, n_A, 0.0, qx*qx + qy*qy)
            step = (qz - zs)/(ez - s*(qx*ex + qy*ey))
            t = t - step
            tol = 1.0 + abs(t)
            # converged, or the steps no longer shrink at the rounding level:
            if not np.isfinite(step) or abs(step) < 1e-15*tol or \
                    (abs(step) >= prev and abs(step) < 1e-12*tol):
                break
            prev = abs(step)
    # the solution must lie on the real part of the conic:
    qx, qy = x + t*ex, y + t*ey
    zs, _ = _sag_slope(STANDARD, c, K, A, n_A, 0.0, qx*qx + qy*qy)
    if not np.isfinite(zs):
        t = np.nan

    return t


def _intersect_cone(cone, x, y, z, ex, ey, ez):
    t_plane = -z/ez
    if cone == 0.0:
        return t_plane
    k2 = cone*cone
    a = ez*ez - k2*(ex*ex + ey*ey)
    b = 2.0*(z*ez - k2*(x*ex + y*ey))
    C = z*z - k2*(x*x + y*y)
    sq = np.sqrt(b*b - 4.0*a*C)
    if abs(a) < 1e-14*(abs(b) + abs(C) + 1.0):
        r0 = r1 = -C/b
    else:
        r0, r1 = (-b - sq)/(2.0*a), (-b + sq)/(2.0*a)
    sgn = 1.0 if cone > 0 else -1.0
    best, t = np.inf, np.nan
    for r in (r0, r1):
        if np.isfinite(r) and (z + r*ez)*sgn >= -1e-12*(1.0 + abs(z)):
            dist = abs(r - t_plane)
            if dist < best:
                best, t = dist, r

    return t


def _intersect_q(kind, c, K, A0, Qc, dQc, n_Q, rn2, x, y, z, ex, ey, ez):
    """Q-bfs or Q-con: from the base sphere or conic by Newton steps on the
    full sag (as qaspheres._QSurface.intersect)."""

    t = _intersect_conic(c, K if kind == QCON else 0.0, A0, 0, x, y, z, ex, ey, ez)
    if not np.isfinite(t):
        t = -z/ez
    prev = np.inf
    for _ in range(60):
        qx, qy, qz = x + t*ex, y + t*ey, z + t*ez
        zs, s = _q_sag_slope(kind, c, K, Qc, dQc, n_Q, rn2, qx*qx + qy*qy)
        step = (qz - zs)/(ez - s*(qx*ex + qy*ey))
        t = t - step
        tol = 1.0 + abs(t)
        if not np.isfinite(step) or abs(step) < 1e-15*tol or \
                (abs(step) >= prev and abs(step) < 1e-12*tol):
            break
        prev = abs(step)
    qx, qy = x + t*ex, y + t*ey
    zs, _ = _q_sag_slope(kind, c, K, Qc, dQc, n_Q, rn2, qx*qx + qy*qy)
    if not np.isfinite(zs):
        t = np.nan

    return t


def _intersect_xy(kind, c, K, A, n_A, cx, Kx, B, n_B, rn, g0, g1, g_kind, g_m, g_coef,
                  g_dcoef, g_n, x, y, z, ex, ey, ez):
    """Toroidal and biconic surfaces (from the osculating quadric) and Q2D
    (from the best fit sphere): Newton steps on the exact sag."""

    if kind == Q2D:
        t = _intersect_conic(c, 0.0, A, 0, x, y, z, ex, ey, ez)
    else:
        cq = cx if abs(cx) >= abs(c) else c
        a = cx*ex*ex + c*ey*ey + cq*ez*ez
        b = 2.0*(cx*x*ex + c*y*ey + cq*z*ez - ez)
        C = cx*x*x + c*y*y + cq*z*z - 2.0*z
        t = -2.0*C/(b + (1.0 if b >= 0 else -1.0)*np.sqrt(b*b - 4.0*a*C))
    if not np.isfinite(t):
        t = -z/ez
    for _ in range(60):
        qx, qy, qz = x + t*ex, y + t*ey, z + t*ez
        zs, gx, gy = _xy_sag_grad(kind, c, K, A, n_A, cx, Kx, B, n_B, rn, g0, g1, g_kind, g_m,
                                  g_coef, g_dcoef, g_n, qx, qy)
        step = (qz - zs)/(ez - gx*ex - gy*ey)
        t = t - step
        if not np.isfinite(step) or abs(step) < 1e-14*(1.0 + abs(t)):
            break
    qx, qy = x + t*ex, y + t*ey
    zs, _, _ = _xy_sag_grad(kind, c, K, A, n_A, cx, Kx, B, n_B, rn, g0, g1, g_kind, g_m,
                            g_coef, g_dcoef, g_n, qx, qy)
    if not np.isfinite(zs):
        t = np.nan

    return t


def _qconq2d_sag_grad(c, K, Qc, dQc, n_Q, rn2, rf, x0, y0, g0, g1, g_kind, g_m, g_coef,
                      g_dcoef, g_n, x, y):
    """Sag and gradient of a Q-con asphere plus a Q2D departure centred on
    (x0, y0) with the normalization radius rf (qaspheres.QconQ2DSurface)."""

    z, s = _q_sag_slope(QCON, c, K, Qc, dQc, n_Q, rn2, x*x + y*y)
    gx, gy = x*s, y*s
    if g1 > g0:
        D, Dx, Dy = _q2d_departure(rf, g0, g1, g_kind, g_m, g_coef, g_dcoef, g_n, x - x0, y - y0)
        z, gx, gy = z + D, gx + Dx, gy + Dy
    return z, gx, gy


def _intersect_qconq2d(c, K, Qc, dQc, n_Q, rn2, rf, x0, y0, g0, g1, g_kind, g_m, g_coef,
                       g_dcoef, g_n, x, y, z, ex, ey, ez):
    """Q-con + Q2D: from the base conic by Newton steps on the full sag."""

    t = _intersect_conic(c, K, Qc, 0, x, y, z, ex, ey, ez)
    if not np.isfinite(t):
        t = -z/ez
    for _ in range(60):
        qx, qy, qz = x + t*ex, y + t*ey, z + t*ez
        zs, gx, gy = _qconq2d_sag_grad(c, K, Qc, dQc, n_Q, rn2, rf, x0, y0, g0, g1, g_kind, g_m,
                                       g_coef, g_dcoef, g_n, qx, qy)
        step = (qz - zs)/(ez - gx*ex - gy*ey)
        t = t - step
        if not np.isfinite(step) or abs(step) < 1e-14*(1.0 + abs(t)):
            break
    qx, qy = x + t*ex, y + t*ey
    zs, _, _ = _qconq2d_sag_grad(c, K, Qc, dQc, n_Q, rn2, rf, x0, y0, g0, g1, g_kind, g_m,
                                 g_coef, g_dcoef, g_n, qx, qy)
    if not np.isfinite(zs):
        t = np.nan

    return t


def _trace_rays(P0, d0, opl0, O, R, kind, c, K, A, n_A, cone, f, mirror, sd, n, clip,
                cx, Kx, B, n_B, rn, Qc, dQc, n_Q, g_start, g_end, g_kind, g_m, g_coef,
                g_dcoef, g_n, rf, x0, y0, out_P, out_d, out_opl, valid):
    S = len(kind)
    for r in nb.prange(P0.shape[0]):
        px, py, pz = P0[r, 0], P0[r, 1], P0[r, 2]
        dx, dy, dz = d0[r, 0], d0[r, 1], d0[r, 2]
        opl = opl0[r]
        ok = True
        out_P[0, r, 0], out_P[0, r, 1], out_P[0, r, 2] = px, py, pz
        out_d[0, r, 0], out_d[0, r, 1], out_d[0, r, 2] = dx, dy, dz
        out_opl[0, r] = opl
        for i in range(S):
            if kind[i] != BREAK and ok:
                # local coordinates: (P - o) @ R
                ux, uy, uz = px - O[i, 0], py - O[i, 1], pz - O[i, 2]
                x = ux*R[i, 0, 0] + uy*R[i, 1, 0] + uz*R[i, 2, 0]
                y = ux*R[i, 0, 1] + uy*R[i, 1, 1] + uz*R[i, 2, 1]
                z = ux*R[i, 0, 2] + uy*R[i, 1, 2] + uz*R[i, 2, 2]
                ex = dx*R[i, 0, 0] + dy*R[i, 1, 0] + dz*R[i, 2, 0]
                ey = dx*R[i, 0, 1] + dy*R[i, 1, 1] + dz*R[i, 2, 1]
                ez = dx*R[i, 0, 2] + dy*R[i, 1, 2] + dz*R[i, 2, 2]
                k = kind[i]
                if k == AXICON:
                    t = _intersect_cone(cone[i], x, y, z, ex, ey, ez)
                elif k == QBFS or k == QCON:
                    t = _intersect_q(k, c[i], K[i], A[i], Qc[i], dQc[i], n_Q[i], rn[i]*rn[i],
                                     x, y, z, ex, ey, ez)
                elif k == QCONQ2D:
                    t = _intersect_qconq2d(c[i], K[i], Qc[i], dQc[i], n_Q[i], rn[i]*rn[i], rf[i],
                                           x0[i], y0[i], g_start[i], g_end[i], g_kind, g_m,
                                           g_coef, g_dcoef, g_n, x, y, z, ex, ey, ez)
                elif k == TOROID or k == BICONIC or k == Q2D:
                    t = _intersect_xy(k, c[i], K[i], A[i], n_A[i], cx[i], Kx[i], B[i], n_B[i],
                                      rn[i], g_start[i], g_end[i], g_kind, g_m, g_coef,
                                      g_dcoef, g_n, x, y, z, ex, ey, ez)
                else:
                    t = _intersect_conic(c[i], K[i], A[i], n_A[i], x, y, z, ex, ey, ez)
                qx, qy, qz = x + t*ex, y + t*ey, z + t*ez
                opl += n[i]*t
                r2 = qx*qx + qy*qy
                if clip and sd[i] >= 0 and not np.sqrt(r2) <= sd[i]*(1 + 1e-9):
                    ok = False
                if kind[i] == IDEAL:
                    # towards the focal point of the bundle with direction e
                    fi = f[i]
                    aez = abs(ez)
                    vx, vy, vz = fi*ex/aez - qx, fi*ey/aez - qy, fi*ez/aez - qz
                    dist = np.sqrt(vx*vx + vy*vy + vz*vz)
                    sgn = 1.0 if fi > 0 else (-1.0 if fi < 0 else 0.0)
                    opl += n[i]*(sgn*(abs(fi)/aez - dist) - (qx*ex + qy*ey + qz*ez))
                    ex, ey, ez = sgn*vx/dist, sgn*vy/dist, sgn*vz/dist
                else:
                    if k == QCONQ2D:
                        _, gx, gy = _qconq2d_sag_grad(c[i], K[i], Qc[i], dQc[i], n_Q[i],
                                                      rn[i]*rn[i], rf[i], x0[i], y0[i],
                                                      g_start[i], g_end[i], g_kind, g_m, g_coef,
                                                      g_dcoef, g_n, qx, qy)
                        Nx, Ny = -gx, -gy
                    elif k == TOROID or k == BICONIC or k == Q2D:
                        _, gx, gy = _xy_sag_grad(k, c[i], K[i], A[i], n_A[i], cx[i], Kx[i], B[i],
                                                 n_B[i], rn[i], g_start[i], g_end[i], g_kind,
                                                 g_m, g_coef, g_dcoef, g_n, qx, qy)
                        Nx, Ny = -gx, -gy
                    else:
                        if k == QBFS or k == QCON:
                            _, s = _q_sag_slope(k, c[i], K[i], Qc[i], dQc[i], n_Q[i],
                                                rn[i]*rn[i], r2)
                        else:
                            _, s = _sag_slope(k, c[i], K[i], A[i], n_A[i], cone[i], r2)
                        Nx, Ny = -qx*s, -qy*s
                    nn = np.sqrt(Nx*Nx + Ny*Ny + 1.0)
                    Nx, Ny, Nz = Nx/nn, Ny/nn, 1.0/nn
                    cos_i = ex*Nx + ey*Ny + ez*Nz
                    if mirror[i]:
                        ex, ey, ez = ex - 2*cos_i*Nx, ey - 2*cos_i*Ny, ez - 2*cos_i*Nz
                    else:
                        # orient N along the ray:
                        if cos_i < 0:
                            Nx, Ny, Nz = -Nx, -Ny, -Nz
                        elif not cos_i > 0:
                            Nx, Ny, Nz = 0.0*Nx, 0.0*Ny, 0.0*Nz  # np.sign(0) = 0, NaN stays NaN
                        cos_i = abs(cos_i)
                        mu = n[i]/n[i + 1]
                        cos_t = np.sqrt(1 - mu*mu*(1 - cos_i*cos_i))  # NaN: total reflection
                        g = cos_t - mu*cos_i
                        ex, ey, ez = mu*ex + g*Nx, mu*ey + g*Ny, mu*ez + g*Nz
                # back to global coordinates: o + Q @ R.T
                px = O[i, 0] + R[i, 0, 0]*qx + R[i, 0, 1]*qy + R[i, 0, 2]*qz
                py = O[i, 1] + R[i, 1, 0]*qx + R[i, 1, 1]*qy + R[i, 1, 2]*qz
                pz = O[i, 2] + R[i, 2, 0]*qx + R[i, 2, 1]*qy + R[i, 2, 2]*qz
                dx = R[i, 0, 0]*ex + R[i, 0, 1]*ey + R[i, 0, 2]*ez
                dy = R[i, 1, 0]*ex + R[i, 1, 1]*ey + R[i, 1, 2]*ez
                dz = R[i, 2, 0]*ex + R[i, 2, 1]*ey + R[i, 2, 2]*ez
                if not (np.isfinite(t) and np.isfinite(dx) and np.isfinite(dy)
                        and np.isfinite(dz)):
                    ok = False
            if ok:
                out_P[i + 1, r, 0], out_P[i + 1, r, 1], out_P[i + 1, r, 2] = px, py, pz
                out_d[i + 1, r, 0], out_d[i + 1, r, 1], out_d[i + 1, r, 2] = dx, dy, dz
                out_opl[i + 1, r] = opl
            else:
                for j in range(3):
                    out_P[i + 1, r, j] = np.nan
                    out_d[i + 1, r, j] = np.nan
                out_opl[i + 1, r] = np.nan
        valid[r] = ok


if available:
    _options = dict(cache=True, error_model="numpy")
    _sag_slope = nb.njit(**_options)(_sag_slope)
    _intersect_conic = nb.njit(**_options)(_intersect_conic)
    _intersect_cone = nb.njit(**_options)(_intersect_cone)
    _chebval = nb.njit(**_options)(_chebval)
    _q_sag_slope = nb.njit(**_options)(_q_sag_slope)
    _q2d_departure = nb.njit(**_options)(_q2d_departure)
    _xy_sag_grad = nb.njit(**_options)(_xy_sag_grad)
    _qconq2d_sag_grad = nb.njit(**_options)(_qconq2d_sag_grad)
    _intersect_qconq2d = nb.njit(**_options)(_intersect_qconq2d)
    _intersect_q = nb.njit(**_options)(_intersect_q)
    _intersect_xy = nb.njit(**_options)(_intersect_xy)
    _kernel = nb.njit(parallel=True, **_options)(_trace_rays)


def surface_kind(surface, types):
    """Kernel code of a surface, or None if the kernel does not cover it.

    types : dict {kind: class} (raytrace._kernel_types())."""

    t = type(surface)
    for kind, cls in types.items():
        if t is cls:
            return kind

    return None


def trace(P, d, opl, frames, surfaces, kinds, n, clip):
    """Trace rays through the surfaces.

    Parameters
    ----------
    P, d : (N, 3) arrays
        Start points and unit directions in global coordinates.
    opl : (N,) array
    frames : list of (origin, R)
        Frames of the surfaces.
    surfaces : list of Surface
    kinds : list of int
        From surface_kind().
    n : array
        Indices in front of the first surface and behind each surface.
    clip : bool

    Returns
    -------
    points, directions : (S + 1, N, 3) arrays
    opls : (S + 1, N) array
    valid : (N,) bool array
        NaN and False for rays that missed a surface, were totally
        reflected or (clip) vignetted, from that surface on.
    """

    S, N = len(surfaces), len(P)
    O = np.zeros((S, 3))
    R = np.zeros((S, 3, 3))
    c, K, cone, f = np.zeros(S), np.zeros(S), np.zeros(S), np.zeros(S)
    n_A = np.zeros(S, dtype=np.int64)
    mirror = np.zeros(S, dtype=np.bool_)
    sd = np.full(S, -1.0)
    for i, (s, kind, (o, Ri)) in enumerate(zip(surfaces, kinds, frames)):
        O[i], R[i] = o, Ri
        n_A[i] = len(s.aspheric)
        if kind == BREAK:
            continue
        c[i], K[i], mirror[i] = s.c, s.conic, s.mirror
        if s.semi_diameter is not None:
            sd[i] = s.semi_diameter
        if kind == IDEAL:
            f[i] = s.f
        elif kind == AXICON:
            cone[i] = s.slope
    A = np.zeros((S, max(1, int(n_A.max(initial=0)))))
    for i, s in enumerate(surfaces):
        A[i, :n_A[i]] = s.aspheric
    cx, Kx, rn = np.zeros(S), np.zeros(S), np.ones(S)
    rf, x0, y0 = np.ones(S), np.zeros(S), np.zeros(S)
    B_list, Q_list, groups = [()]*S, [np.zeros(0)]*S, []
    g_start, g_end = np.zeros(S, dtype=np.int64), np.zeros(S, dtype=np.int64)
    for i, (s, kind) in enumerate(zip(surfaces, kinds)):
        if kind in (TOROID, BICONIC):
            cx[i] = 0.0 if np.isinf(s.R_x) else 1.0/s.R_x
        if kind == BICONIC:
            Kx[i], B_list[i] = s.conic_x, tuple(s.aspheric_x)
        if kind in (QBFS, QCON, Q2D, QCONQ2D):
            from . import qaspheres
            rn[i] = s.norm_radius
        if kind == QBFS:
            c[i] = s.c_bfs  # the base sphere; s.c is the vertex curvature
            if s.q:
                Q_list[i] = qaspheres.qbfs_series(s.q)
        elif kind in (QCON, QCONQ2D) and s.q:
            Q_list[i] = qaspheres.qcon_series(s.q)
        if kind == QCONQ2D:
            rf[i], (x0[i], y0[i]) = s.freeform_radius, s.centre
        if kind == Q2D:
            c[i], K[i] = s.c_bfs, 0.0
        if kind in (Q2D, QCONQ2D):
            g_start[i] = len(groups)
            for (part, m), coef in s._groups().items():
                groups.append((0 if part == "cos" else 1, m, np.asarray(coef, float)))
            g_end[i] = len(groups)
    n_B = np.array([len(b) for b in B_list], dtype=np.int64)
    B = np.zeros((S, max(1, int(n_B.max(initial=0)))))
    for i, b in enumerate(B_list):
        B[i, :len(b)] = b
    n_Q = np.array([len(q) for q in Q_list], dtype=np.int64)
    Qc = np.zeros((S, max(1, int(n_Q.max(initial=0)))))
    dQc = np.zeros_like(Qc)
    for i, q in enumerate(Q_list):
        Qc[i, :len(q)] = q
        if len(q) > 1:
            dQc[i, :len(q) - 1] = np.polynomial.chebyshev.chebder(q)
    G = max(1, len(groups))
    g_kind, g_m, g_n = (np.zeros(G, dtype=np.int64), np.zeros(G, dtype=np.int64),
                        np.zeros(G, dtype=np.int64))
    width = max([1] + [len(g[2]) for g in groups])
    g_coef, g_dcoef = np.zeros((G, width)), np.zeros((G, width))
    for k, (part, m, coef) in enumerate(groups):
        g_kind[k], g_m[k], g_n[k] = part, m, len(coef)
        g_coef[k, :len(coef)] = coef
        if len(coef) > 1:
            g_dcoef[k, :len(coef) - 1] = np.polynomial.chebyshev.chebder(coef)
    out_P = np.empty((S + 1, N, 3))
    out_d = np.empty((S + 1, N, 3))
    out_opl = np.empty((S + 1, N))
    valid = np.empty(N, dtype=np.bool_)
    _kernel(np.ascontiguousarray(P, dtype=float), np.ascontiguousarray(d, dtype=float),
           np.ascontiguousarray(opl, dtype=float), O, R, np.asarray(kinds, dtype=np.int64), c, K, A,
           n_A, cone, f, mirror, sd, np.asarray(n, dtype=float), bool(clip), cx, Kx, B, n_B, rn,
           Qc, dQc, n_Q, g_start, g_end, g_kind, g_m, g_coef, g_dcoef, g_n, rf, x0, y0, out_P, out_d,
           out_opl, valid)

    return out_P, out_d, out_opl, valid
