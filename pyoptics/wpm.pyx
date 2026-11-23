#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Wave propagation method (WPM) of Brenner and Singer.

The WPM propagates a scalar field through a medium with a spatially varying
refractive index n(x, y, z) in thin slices delta_z. In each slice, the field
is decomposed into plane waves, and every plane wave is propagated at every
location (x, y) with the local k_z(x, y) = sqrt((n(x, y) k_0)^2 - k_x^2 - k_y^2)::

    U(x, y, z + delta_z) = sum_{k_x, k_y} A(k_x, k_y) exp(i (k_x x + k_y y))
                                          exp(i k_z(x, y) delta_z).

K.-H. Brenner and W. Singer, Appl. Opt. 32, 4984 (1993).

Two implementations of the same sum are available:

- "direct": the double sum for every pixel, O(N_x^2 N_y^2) per step,
  parallelized with OpenMP.
- "levels": for each distinct index value n_q in the slice, the whole
  spectrum is propagated with the homogeneous transfer function of n_q by
  FFT; each pixel takes the result of its own index value. This is the same
  sum, rearranged, and costs O(Q N_x N_y log(N_x N_y)) for Q distinct index
  values. Media made of a few materials (lenses, waveguides, gratings) are
  fast with it. This variant is known as the "fast WPM".

The default method="auto" chooses the cheaper one.

Convention: exp(i(k.r - omega*t)), i.e. propagation by delta_z multiplies a
plane wave by exp(+i*k_z*delta_z). Absorbing media have n + i*kappa with
kappa > 0. Evanescent waves decay (Im(k_z) >= 0).

Sampling: the plane wave decomposition is periodic in the window, and only
plane waves up to abs(k_x), abs(k_y) <= pi/dx are represented. To cover all
propagating waves in the densest medium, use dx <= wl/(2 n_max), see
max_sampling_distance(). Like all FFT propagators, the WPM needs a guard band
(or an absorbing index n + i*kappa at the window edges) so that light leaving
the window does not re-enter on the other side.

Fresnel coefficients: the plain WPM passes every plane wave through an index
step unchanged, i.e. without reflection losses. With fresnel="s", "p" or a
Jones vector, every plane wave that enters a pixel whose index n_2 differs
from the index n_1 of the same pixel in the previous slice is multiplied by
its Fresnel amplitude transmission::

    t_s = 2 q_1/(q_1 + q_2),   t_p = 2 n_1 n_2 q_1/(n_2^2 q_1 + n_1^2 q_2),

with q_j = k_z(n_j) of that plane wave in both media. In the sliced medium
every index step lies between two slices, i.e. it is an interface
perpendicular to z, for which these are the exact coefficients of each plane
wave. A tilted surface is a staircase of such steps, which a beam along z
meets at normal incidence: the loss is then about the normal-incidence loss,
and the dependence on the tilt and the polarization is mostly lost (see
notebook 05, wave propagation method, for checks against exact solutions).
The field
then stands for the electric field: the power flux of a plane wave is
proportional to Re(k_z) |E|^2, so the intensity in a medium n is about
n |E|^2. The reflected light is dropped. With "levels", t depends only on
the plane wave and the pair (n_1, n_2), which costs one more FFT per pair of
indices; the direct sum multiplies each term by its t.

Limitations: the WPM is a scalar, one-way method; reflected light is lost and
multiple reflections are missing.

References
----------
K.-H. Brenner and W. Singer, "Light propagation through microlenses: a new
simulation method", Appl. Opt. 32, 4984-4988 (1993).

S. Schmidt, T. Tiess, S. Schröter, R. Hambach, M. Jäger, H. Bartelt, A.
Tünnermann and H. Gross, "Wave-optical modeling beyond the
thin-element-approximation", Opt. Express 24, 30188 (2016).

M. Born and E. Wolf, Principles of Optics, 7th ed., Cambridge University
Press (1999), section 1.5 (Fresnel formulae).
"""


cdef extern from "complex.h":
    double complex cexp(double complex) nogil
    double complex csqrt(double complex) nogil
    double cimag(double complex) nogil
    double creal(double complex) nogil
    double cabs(double complex) nogil
    double complex conj(double complex) nogil


cimport cython
import numpy as np

from numpy.polynomial.legendre import leggauss
from cython.parallel import prange

from .utils import frequencies, wavenumber, k_z
from .fft import FT, inv_FT


# contributions damped by more than exp(-_MAX_DECAY) are skipped in the
# direct sum (exp(-46) ~ 1E-20):
cdef double _MAX_DECAY = 46.0

# polarization models of the Fresnel correction:
cdef int _FRESNEL_S = 1, _FRESNEL_P = 2, _FRESNEL_JONES = 3


def averaging_z_sampling(func, x, y, z1, z2, N):
    """Average func(x, y, z) between z1 and z2 by computing the integral

        1/(z2 - z1) int_{z1}^{z2} func(x, y, z) dz

    with an N point Gauss-Legendre quadrature.

    Parameters
    ----------
    func : callable
        func(x, y, z), returns an array.
    x, y : arrays
        Passed on to func.
    z1, z2 : number
        Integration limits.
    N : int
        Number of quadrature nodes. N = 1 is the midpoint rule.

    Returns
    -------
    avg : array
        func averaged over z, same shape as func(x, y, z).
    """

    z_support_unshifted, weights = leggauss(N)
    # shift and scale [-1, +1] to [z1, z2]:
    z_shifted = (z2 - z1)/2.*z_support_unshifted + (z1 + z2)/2.

    func_sampled = np.array([func(x, y, z_val) for z_val in z_shifted])

    # the weights sum up to 2 (the length of [-1, +1]):
    return np.tensordot(weights, func_sampled, axes=1)/2.


def max_sampling_distance(wl, n_max):
    """Return the largest sampling distance dx for which the plane wave
    decomposition contains all propagating waves in a medium with the
    refractive index n_max: dx = wl/(2 n_max).

    Parameters
    ----------
    wl : number
        Vacuum wavelength.
    n_max : number
        Largest (real part of the) refractive index in the problem.

    Returns
    -------
    dx : number
    """

    return wl/(2.0*np.real(n_max))


def _sample_index(n_func, X, Y, z, delta_z, n_sampling):
    """Sample n_func in the slice [z, z + delta_z]."""

    if isinstance(n_sampling, str):
        if n_sampling == "mid":
            n = n_func(X, Y, z + delta_z/2.)
        elif n_sampling == "start":
            n = n_func(X, Y, z)
        else:
            raise ValueError(f"Unknown n_sampling '{n_sampling}', use 'mid', 'start' or an int.")
    else:
        n = averaging_z_sampling(n_func, X, Y, z, z + delta_z, int(n_sampling))

    n = np.asarray(n, dtype=np.complex128)
    if n.shape != X.shape:
        n = np.broadcast_to(n, X.shape)

    return n


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _direct_sum(double complex[:, ::1] n_sampled,
                      double complex[:, ::1] spectrum,
                      double complex[:, ::1] phase_x,  # exp(i k_x x), [j, m]
                      double complex[:, ::1] phase_y,  # exp(i k_y y), [i, l]
                      double[::1] k_x, double[::1] k_y,
                      double k0, double delta_z,
                      double complex[:, ::1] out) noexcept nogil:
    """out[i, j] = sum_{l, m} spectrum[l, m] exp(i(k_x x_j + k_y y_i))
                   exp(i k_z(n[i, j]) delta_z)"""

    cdef Py_ssize_t i, j, l, m
    cdef double complex kz, k_sq, acc, acc_y
    cdef double complex imag_unit = 1j

    for i in prange(n_sampled.shape[0], schedule="dynamic"):
        for j in range(n_sampled.shape[1]):
            k_sq = (n_sampled[i, j]*k0)**2
            acc = 0.0
            for l in range(k_y.shape[0]):
                acc_y = 0.0
                for m in range(k_x.shape[0]):
                    kz = csqrt(k_sq - k_x[m]**2 - k_y[l]**2)
                    # branch with Im(k_z) >= 0: evanescent waves decay (csqrt
                    # returns Im < 0 for arguments on the negative zero side
                    # of the branch cut):
                    if cimag(kz) < 0.0:
                        kz = -kz
                    if cimag(kz)*delta_z > _MAX_DECAY:
                        continue
                    acc_y = acc_y + spectrum[l, m]*phase_x[j, m]*cexp(imag_unit*kz*delta_z)
                acc = acc + acc_y*phase_y[i, l]
            out[i, j] = acc


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double complex _sqrt_up(double complex arg) noexcept nogil:
    """Square root on the branch Im >= 0 (and Re >= 0 on the real axis)."""

    cdef double complex r = csqrt(arg)
    if cimag(r) < 0.0 or (cimag(r) == 0.0 and creal(r) < 0.0):
        r = -r
    return r


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double complex _fresnel_t(double kx, double ky, double complex q1,
                               double complex q2, double complex n1,
                               double complex n2, int mode,
                               double complex ex, double complex ey) noexcept nogil:
    """Amplitude transmission of the plane wave (kx, ky, q1) in the medium n1
    into the plane wave (kx, ky, q2) in n2 at an interface perpendicular to
    z. Same formulas as _fresnel_t_z()."""

    cdef double complex ts, tp, es, epi, ept
    cdef double k_perp, sx, sy, k1, k2, pix, piy, ptx, pty, den

    if cabs(q1 + q2) == 0.0:
        return 1.0
    ts = 2.0*q1/(q1 + q2)
    if mode == _FRESNEL_S:
        return ts
    tp = 2.0*n1*n2*q1/(n2*n2*q1 + n1*n1*q2)
    if mode == _FRESNEL_P:
        return tp

    # Jones vector (ex, ey): s = (ky, -kx)/k_perp, p = s x k/|k|
    k_perp = (kx*kx + ky*ky)**0.5
    if k_perp == 0.0:
        return ts   # normal incidence: t_s = t_p
    sx = ky/k_perp
    sy = -kx/k_perp
    k1 = (k_perp*k_perp + creal(q1)*creal(q1))**0.5
    k2 = (k_perp*k_perp + creal(q2)*creal(q2))**0.5
    if k2 == 0.0:
        return ts
    pix = sy*creal(q1)/k1
    piy = -sx*creal(q1)/k1
    ptx = sy*creal(q2)/k2
    pty = -sx*creal(q2)/k2
    es = ex*sx + ey*sy
    epi = ex*pix + ey*piy
    ept = conj(ex)*ptx + conj(ey)*pty
    den = creal(es*conj(es) + epi*conj(epi))
    if den <= 1E-24:
        return ts
    return (ts*es*conj(es) + tp*epi*ept)/den


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fresnel_sum(double complex[:, ::1] n_in,
                       double complex[:, ::1] n_out,
                       double complex[:, ::1] spectrum,
                       double complex[:, ::1] phase_x,  # exp(i k_x x), [j, m]
                       double complex[:, ::1] phase_y,  # exp(i k_y y), [i, l]
                       double[::1] k_x, double[::1] k_y,
                       double k0, double delta_z, int mode,
                       double complex ex, double complex ey,
                       double complex[:, ::1] out) noexcept nogil:
    """The direct sum with every plane wave multiplied by its Fresnel
    transmission from n_in[i, j] into n_out[i, j] (t = 1 where they agree):

    out[i, j] = sum_{l, m} spectrum[l, m] t_lm exp(i(k_x x_j + k_y y_i))
                exp(i k_z(n_out[i, j]) delta_z)"""

    cdef Py_ssize_t i, j, l, m
    cdef double complex q1, q2, k1_sq, k2_sq, arg1, acc, acc_y, t
    cdef double complex imag_unit = 1j
    cdef bint step

    for i in prange(n_out.shape[0], schedule="dynamic"):
        for j in range(n_out.shape[1]):
            k1_sq = (n_in[i, j]*k0)**2
            k2_sq = (n_out[i, j]*k0)**2
            step = (n_in[i, j] != n_out[i, j])
            acc = 0.0
            for l in range(k_y.shape[0]):
                acc_y = 0.0
                for m in range(k_x.shape[0]):
                    q2 = _sqrt_up(k2_sq - k_x[m]**2 - k_y[l]**2)
                    if cimag(q2)*delta_z > _MAX_DECAY:
                        continue
                    t = 1.0
                    arg1 = k1_sq - k_x[m]**2 - k_y[l]**2
                    # (evanescent waves of the incident medium pass unchanged)
                    if step and cimag(arg1) == 0.0 and creal(arg1) > 0.0:
                        q1 = _sqrt_up(arg1)
                        t = _fresnel_t(k_x[m], k_y[l], q1, q2, n_in[i, j],
                                       n_out[i, j], mode, ex, ey)
                    acc_y = acc_y + spectrum[l, m]*phase_x[j, m]*t*cexp(imag_unit*q2*delta_z)
                acc = acc + acc_y*phase_y[i, l]
            out[i, j] = acc


def _cost_direct(Ny, Nx):
    # rough run time in units of 100 ns (from timings with 4 threads): one
    # term of the direct sum, a complex sqrt and exp, takes about 150 ns
    return 1.5*float(Nx*Ny)**2


def _cost_levels(Ny, Nx, num_levels):
    # per level: an FFT pair plus a constant Python overhead of about 80 us
    N = Nx*Ny
    return num_levels*(800.0 + 0.15*N*np.log2(N + 1))


def _fresnel_mode(fresnel):
    """Kernel mode and Jones vector for the fresnel argument."""

    if fresnel is None:
        return 0, 0j, 0j
    if isinstance(fresnel, str):
        key = fresnel.lower()
        if key in ("s", "te"):
            return _FRESNEL_S, 0j, 1+0j
        if key in ("p", "tm"):
            return _FRESNEL_P, 1+0j, 0j
        if key == "x":
            return _FRESNEL_JONES, 1+0j, 0j
        if key == "y":
            return _FRESNEL_JONES, 0j, 1+0j
        raise ValueError(f"Unknown fresnel '{fresnel}', use None, 's', 'p', 'x', 'y' or a Jones vector.")
    e = np.asarray(fresnel, dtype=np.complex128).ravel()
    if e.shape != (2,) or not np.any(e):
        raise ValueError("A Jones vector for fresnel needs two components (e_x, e_y), not both zero.")
    e = e/np.sqrt(np.sum(np.abs(e)**2))
    return _FRESNEL_JONES, complex(e[0]), complex(e[1])


def _fresnel_t_z(K_X, K_Y, n1, n2, k0, mode, e_x, e_y):
    """Fresnel amplitude transmission of the plane waves (K_X, K_Y) from n1
    into n2 at an interface perpendicular to z (the vectorized counterpart of
    _fresnel_t() with the normal (0, 0, 1))."""

    k_perp2 = K_X**2 + K_Y**2
    arg1 = (n1*k0)**2 - k_perp2
    q1 = k_z(n1*k0, K_X, K_Y)
    q2 = k_z(n2*k0, K_X, K_Y)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_s = 2*q1/(q1 + q2)
        t_p = 2*n1*n2*q1/(n2**2*q1 + n1**2*q2)
        if mode == _FRESNEL_S:
            t = t_s
        elif mode == _FRESNEL_P:
            t = t_p
        else:
            # s = k x z/|k x z| = (k_y, -k_x, 0)/k_perp; p = s x k/|k|
            k_perp = np.sqrt(k_perp2)
            s_x, s_y = K_Y/k_perp, -K_X/k_perp
            q1r, q2r = q1.real, q2.real
            k1 = np.sqrt(k_perp2 + q1r**2)
            k2 = np.sqrt(k_perp2 + q2r**2)
            # p = s x k/|k| has the transverse part s_y k_z, -s_x k_z
            pi_x, pi_y = s_y*q1r/k1, -s_x*q1r/k1
            pt_x, pt_y = s_y*q2r/k2, -s_x*q2r/k2
            es = e_x*s_x + e_y*s_y
            epi = e_x*pi_x + e_y*pi_y
            ept = np.conj(e_x)*pt_x + np.conj(e_y)*pt_y
            den = np.abs(es)**2 + np.abs(epi)**2
            t = (t_s*np.abs(es)**2 + t_p*epi*ept)/den
            t = np.where((k_perp == 0) | (den <= 1E-24) | (k2 == 0), t_s, t)
    # evanescent waves of the incident medium and degenerate cases pass unchanged:
    ok = (np.abs(np.imag(arg1)) == 0) & (np.real(arg1) > 0) & (np.abs(q1 + q2) > 0)
    return np.where(ok, t, 1.0)


def _wpm_step(n_func, field_z, x, y, z, delta_z, wl, n_sampling, method,
              fresnel, n_in):
    """One WPM step; returns the field at z + delta_z and the sampled index."""

    field_z = np.asarray(field_z)
    one_dim = (field_z.ndim == 1)
    x = np.asarray(x, dtype=np.float64)

    if one_dim:
        if field_z.shape != x.shape:
            raise ValueError("A one-dimensional field needs the shape of x.")
        X = x
        Y = np.zeros_like(x)
        field = field_z.reshape(1, -1)
        y = np.zeros(1)
        k_y = np.zeros(1)
    else:
        y = np.asarray(y, dtype=np.float64)
        if field_z.shape != (len(y), len(x)):
            raise ValueError(f"The field must have the shape (len(y), len(x)) = {(len(y), len(x))}, "
                             f"got {field_z.shape}.")
        X, Y = np.meshgrid(x, y)
        field = field_z
        k_y = frequencies(y, wavenumbers=True, normal_order=True)
    k_x = frequencies(x, wavenumbers=True, normal_order=True)

    Ny, Nx = field.shape
    k0 = wavenumber(wl)

    if method not in ("auto", "direct", "levels"):
        raise ValueError(f"Unknown method '{method}', use 'auto', 'direct' or 'levels'.")
    mode, e_x, e_y = _fresnel_mode(fresnel)

    n_sampled = _sample_index(n_func, X, Y, z, delta_z, n_sampling).reshape(Ny, Nx)

    # Fresnel correction: the pixels where the index changes at the plane z
    if mode:
        if n_in is None:
            # the previous slice, of the same thickness:
            n_in = _sample_index(n_func, X, Y, z - delta_z, delta_z, n_sampling)
        n_in = np.asarray(n_in, dtype=np.complex128)
        if n_in.ndim == 0:
            n_in = np.full((Ny, Nx), n_in)
        else:
            n_in = n_in.reshape(Ny, Nx)
        pix_i, pix_j = np.nonzero(n_in != n_sampled)
        if not len(pix_i):
            mode = 0

    # plane wave amplitudes A(k_x, k_y) for exp(+i(k_x x + k_y y)), with x and
    # y relative to the grid points at index N//2 (the origin of FT()):
    spectrum = np.ascontiguousarray(FT(field.astype(np.complex128))/(Nx*Ny))

    if method == "auto":
        levels = np.unique(n_sampled)
        num_ffts = len(levels)
        if mode:
            num_ffts += len(np.unique(n_in[pix_i, pix_j] + 1j*np.pi*n_sampled[pix_i, pix_j]))
        if _cost_levels(Ny, Nx, num_ffts) < _cost_direct(Ny, Nx):
            method = "levels"
        else:
            method = "direct"
    elif method == "levels":
        levels = np.unique(n_sampled)

    if method == "levels":
        out = np.empty((Ny, Nx), dtype=np.complex128)
        K_X, K_Y = np.meshgrid(k_x, k_y)
        for n_q in levels:
            KZ = k_z(n_q*k0, K_X, K_Y)
            field_q = inv_FT(spectrum*np.exp(1j*KZ*delta_z))*(Nx*Ny)
            mask = (n_sampled == n_q)
            out[mask] = field_q[mask]
        if mode:
            # t depends only on the plane wave and the pair (n_in, n_out):
            # one more FFT per pair
            pairs = np.stack([n_in[pix_i, pix_j], n_sampled[pix_i, pix_j]], axis=-1)
            pairs, which = np.unique(pairs, axis=0, return_inverse=True)
            which = which.ravel()
            for k_pair, (n_1, n_2) in enumerate(pairs):
                t = _fresnel_t_z(K_X, K_Y, n_1, n_2, k0, mode, e_x, e_y)
                KZ = k_z(n_2*k0, K_X, K_Y)
                field_q = inv_FT(spectrum*t*np.exp(1j*KZ*delta_z))*(Nx*Ny)
                sel = (which == k_pair)
                out[pix_i[sel], pix_j[sel]] = field_q[pix_i[sel], pix_j[sel]]
    else:
        x_rel = x - x[Nx//2]
        y_rel = y - y[Ny//2]
        phase_x = np.ascontiguousarray(np.exp(1j*np.outer(x_rel, k_x)))
        phase_y = np.ascontiguousarray(np.exp(1j*np.outer(y_rel, k_y)))
        out = np.zeros((Ny, Nx), dtype=np.complex128)
        if mode:
            _fresnel_sum(np.ascontiguousarray(n_in), np.ascontiguousarray(n_sampled),
                         spectrum, phase_x, phase_y,
                         np.ascontiguousarray(k_x), np.ascontiguousarray(k_y),
                         k0, delta_z, mode, e_x, e_y, out)
        else:
            _direct_sum(np.ascontiguousarray(n_sampled), spectrum, phase_x, phase_y,
                        np.ascontiguousarray(k_x), np.ascontiguousarray(k_y),
                        k0, delta_z, out)

    if one_dim:
        out = out[0]
        n_sampled = n_sampled[0]

    return out, n_sampled


def wpm_propagate(n_func, field_z, x, y, z, delta_z, wl, n_sampling="mid",
                  method="auto", fresnel=None, n_in=None):
    """Propagate a field through one slice with the WPM.

    Parameters
    ----------
    n_func : callable
        Refractive index n = n_func(X, Y, z) with gridded X, Y (as from
        np.meshgrid(x, y)). May be complex (n + i*kappa, kappa > 0 absorbs).
        n_func() is called once per slice (more often for n_sampling=int)
        with the whole grid, so that it can be vectorized.
    field_z : array
        Sampled field at z, shape (len(y), len(x)). A one-dimensional field
        of shape (len(x),) describes a problem invariant in y (pass y=None);
        n_func then receives one-dimensional X and Y = 0.
    x, y : arrays
        Equidistant coordinate arrays. y is ignored for one-dimensional
        fields.
    z : number
        Current z position.
    delta_z : number
        Distance to propagate.
    wl : number
        Vacuum wavelength.
    n_sampling : "mid", "start" or int, optional
        Where in the slice [z, z + delta_z] the index is sampled: in the
        middle (default), at the start, or averaged with an n_sampling point
        Gauss-Legendre quadrature.
    method : "auto", "direct" or "levels", optional
        Implementation of the WPM sum, see the module docstring. Both give the
        same result; "auto" picks the faster one.
    fresnel : None, "s", "p", "x", "y" or (e_x, e_y), optional
        Fresnel transmission at index steps, see the module docstring. None
        (default) is the plain WPM. "s" ("TE") and "p" ("TM") use t_s or t_p
        for every plane wave. For fields invariant in y, "s" describes E
        along y. A Jones vector (e_x, e_y), or "x" and "y" for linear
        polarization, projects each plane wave onto s and p and the
        transmitted field back onto the Jones vector.
    n_in : number or array, optional
        Index of the medium the field is in before the slice, used with
        fresnel. Default: n_func sampled in the slice [z - delta_z, z].

    Returns
    -------
    field_propagated : array
        Field at z + delta_z, same shape as field_z.

    Notes
    -----
    The plane wave decomposition uses the centered DFT of pyoptics.fft, i.e.
    phases refer to the grid point x[N//2], y[M//2]. Any equidistant grid
    works, centered or not.
    """

    out, _ = _wpm_step(n_func, field_z, x, y, z, delta_z, wl, n_sampling,
                       method, fresnel, n_in)
    return out


def wpm_propagate_z(n_func, field, x, y, z_vals, wl, n_sampling="mid",
                    method="auto", fresnel=None, n_in=None):
    """Propagate a field through the slices between consecutive z_vals and
    return the field at all z_vals.

    Parameters
    ----------
    n_func : callable
        n_func(X, Y, z), see wpm_propagate().
    field : array
        Field at z_vals[0], shape (len(y), len(x)), or (len(x),) for a problem
        invariant in y (pass y=None).
    x, y : arrays
        Coordinate arrays.
    z_vals : array
        Increasing (or decreasing) z positions. Need not be equidistant.
    wl : number
        Vacuum wavelength.
    n_sampling, method, fresnel : optional
        See wpm_propagate().
    n_in : number or array, optional
        With fresnel: index of the medium the field is in at z_vals[0], see
        wpm_propagate(). After that, each slice uses the index of the
        previous one.

    Returns
    -------
    fields : array
        Fields at z_vals, shape (len(z_vals),) + field.shape. fields[0] is the
        input field.
    """

    z_vals = np.asarray(z_vals, dtype=np.float64)
    field = np.asarray(field, dtype=np.complex128)

    fields = np.empty((len(z_vals),) + field.shape, dtype=np.complex128)
    fields[0] = field
    n_prev = n_in
    for k in range(1, len(z_vals)):
        z, dz = z_vals[k-1], z_vals[k] - z_vals[k-1]
        fields[k], n_sampled = _wpm_step(n_func, fields[k-1], x, y, z, dz, wl,
                                         n_sampling, method, fresnel, n_prev)
        n_prev = n_sampled

    return fields
