#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Ince-Gaussian beams: the paraxial modes in elliptic coordinates.

The paraxial wave equation separates in Cartesian coordinates
(Hermite-Gaussian modes), in circular cylinder coordinates
(Laguerre-Gaussian modes) and in elliptic cylinder coordinates
(Ince-Gaussian modes, Bandres and Gutierrez-Vega 2004). The elliptic
coordinates (xi, eta) are ::

    x = f(z)*cosh(xi)*cos(eta),   y = f(z)*sinh(xi)*sin(eta),

with xi >= 0, 0 <= eta < 2*pi and the foci at x = +-f(z) on the x axis. The
semi-focal distance grows with the beam, f(z) = f0*w(z)/w0, and the
ellipticity parameter eps = 2*f0**2/w0**2 does not change on propagation.
The even and odd modes are ::

    IG^e_pm = C_p^m(i*xi, eps)*C_p^m(eta, eps)*G(x, y, z)*exp(-i*p*psi(z)),
    IG^o_pm = S_p^m(i*xi, eps)*S_p^m(eta, eps)*G(x, y, z)*exp(-i*p*psi(z)),

where G is the fundamental Gaussian beam (with the phase exp(i*k*z), the
curvature exp(i*k*r**2/(2*R)) and the Gouy phase exp(-i*psi)) and
psi(z) = arctan(z/z_R). The order p >= 0 and the index m have the same
parity, 0 <= m <= p for even and 1 <= m <= p for odd modes. The field is a
polynomial of degree p in x/w and y/w times the Gaussian, i.e. a finite
superposition of HG (or LG) modes of the same order N = p. All of them share
the Gouy phase (p + 1)*psi, which is why IG modes keep their shape in free
space and in every rotationally symmetric ABCD system.

Ince polynomials
----------------
C_p^m and S_p^m are the periodic solutions of the Ince equation ::

    C'' + eps*sin(2*eta)*C' + (a - p*eps*cos(2*eta))*C = 0

that are finite Fourier series. With the operator
L = d**2/deta**2 + eps*sin(2*eta)*d/deta - p*eps*cos(2*eta) ::

    L cos(n*eta) = -n**2*cos(n*eta) + eps/2*((n - p)*cos((n + 2)*eta)
                                              - (n + p)*cos((n - 2)*eta)),

and the same for sin(n*eta). The coefficient of the n = p + 2 term vanishes,
so the cosines (sines) up to n = p with the parity of p span an invariant
space. The eigenvalues -a of L on this space, sorted by a, give the
polynomials for m = 0, 2, 4, ... (or 1, 3, ...): each one tends to cos(m*eta)
(sin(m*eta)) for eps -> 0. We normalize the Fourier coefficient of
cos(m*eta) (sin(m*eta)) to be positive.

For eps -> 0, IG^e_pm and IG^o_pm tend to LG modes with
exp(+-i*m*phi) combined into cos(m*phi) and sin(m*phi) and p_LG = (p - m)/2.
For eps -> infinity, IG^e_pm tends to HG_{m, p-m} and IG^o_pm to
HG_{m-1, p-m+1} (indices in x and y). The helical modes
IG^e_pm +- i*IG^o_pm carry m vortices of charge +-1 that sit on the line
between the foci.

The modes are normalized to unit power. Since they are Gaussian times a
polynomial of degree p, Gauss-Hermite quadrature with p + 2 nodes per axis
integrates the power and the overlaps with HG and LG modes exactly;
`hg_coefficients()` and `lg_coefficients()` use this.

The phase conventions follow `pyoptics.beams` (exp(-i*omega*t), a beam in +z
carries exp(+i*k*z)); the odd modes use sinh(n*xi) for S(i*xi) (i.e. without
the factor i), so that IG^o is real at the waist.

References
----------
M. A. Bandres and J. C. Gutierrez-Vega, "Ince-Gaussian beams", Opt. Lett.
29, 144-146 (2004).

M. A. Bandres and J. C. Gutierrez-Vega, "Ince-Gaussian modes of the
paraxial wave equation and stable resonators", J. Opt. Soc. Am. A 21,
873-880 (2004).

U. T. Schwarz, M. A. Bandres and J. C. Gutierrez-Vega, "Observation of
Ince-Gaussian modes in stable resonators", Opt. Lett. 29, 1870-1872 (2004).

E. L. Ince, "A linear differential equation with periodic coefficients",
Proc. London Math. Soc. 23, 56-74 (1925).
"""

from collections import namedtuple
from math import pi, sqrt

import numpy as np

from .modes import hg_mode, lg_mode
from .utils import ensure_meshgrid, wavenumber


InceSeries = namedtuple("InceSeries", ["a", "n", "coefficients"])
InceSeries.__doc__ = """Fourier series of an Ince polynomial.

Attributes
----------
a : float
    Eigenvalue of the Ince equation.
n : array
    Harmonics: C = sum_j coefficients[j]*cos(n[j]*eta) (even) or
    S = sum_j coefficients[j]*sin(n[j]*eta) (odd).
coefficients : array
    Fourier coefficients, the one with n = m positive.
"""


def _check_indices(p, m, parity):
    if parity not in ("even", "odd"):
        raise ValueError("parity must be 'even' or 'odd'.")
    if p < 0 or m < 0 or m > p or (p - m) % 2:
        raise ValueError("Need 0 <= m <= p with p - m even.")
    if parity == "odd" and m == 0:
        raise ValueError("Odd modes need m >= 1.")


def ince_indices(max_order, parity="even"):
    """All (p, m) with p <= max_order for the given parity, by order."""

    m_min = 0 if parity == "even" else 1

    return [(p, m) for p in range(max_order + 1) for m in range(p % 2, p + 1, 2) if m >= m_min]


def ince_series(p, m, eps, parity="even"):
    """Fourier series of the Ince polynomial C_p^m (even) or S_p^m (odd).

    Parameters
    ----------
    p, m : int
        Order and index (same parity, m <= p; m >= 1 for odd).
    eps : number
        Ellipticity parameter (>= 0).
    parity : {"even", "odd"}

    Returns
    -------
    InceSeries
        Eigenvalue a, harmonics n and coefficients.
    """

    _check_indices(p, m, parity)
    n_min = (p % 2) if parity == "even" else (2 - p % 2)
    n = np.arange(n_min, p + 1, 2)
    index = {nn: j for j, nn in enumerate(n)}
    # sign of the folded term for negative harmonics: cos(-k) = cos(k),
    # sin(-k) = -sin(k), sin(0) = 0:
    fold = 1.0 if parity == "even" else -1.0

    L = np.zeros((len(n), len(n)))
    for j, nn in enumerate(n):
        L[j, j] += -nn**2
        if nn + 2 in index:
            L[index[nn + 2], j] += eps/2*(nn - p)
        k = nn - 2
        if k in index:
            L[index[k], j] += -eps/2*(nn + p)
        elif -k in index and k != 0:
            L[index[-k], j] += -fold*eps/2*(nn + p)

    values, vectors = np.linalg.eig(L)
    a = -values.real
    order = np.argsort(a)
    j = list(n).index(m)
    vec = vectors[:, order[j]].real
    vec = vec/np.linalg.norm(vec)
    if vec[j] < 0:
        vec = -vec

    return InceSeries(a[order[j]], n, vec)


def ince_polynomial(p, m, eps, eta, parity="even", series=None):
    """Ince polynomial C_p^m(eta, eps) or S_p^m(eta, eps).

    Parameters
    ----------
    p, m : int
    eps : number
    eta : array
        Angle.
    parity : {"even", "odd"}
    series : InceSeries, optional
        Precomputed `ince_series()`.

    Returns
    -------
    values : array
    """

    s = ince_series(p, m, eps, parity) if series is None else series
    eta = np.asarray(eta)
    trig = np.cos if parity == "even" else np.sin
    terms = [c*trig(nn*eta) for c, nn in zip(s.coefficients, s.n)]

    return np.sum(terms, axis=0)


def _radial(s, xi, parity):
    """C_p^m(i*xi) = sum c*cosh(n*xi); S_p^m(i*xi)/i = sum c*sinh(n*xi)."""

    trig = np.cosh if parity == "even" else np.sinh

    return np.sum([c*trig(nn*xi) for c, nn in zip(s.coefficients, s.n)], axis=0)


def elliptic_coordinates(x, y, f):
    """Elliptic coordinates (xi, eta) with the foci at (+-f, 0).

    x + i*y = f*cosh(xi + i*eta), xi >= 0, eta in [0, 2*pi).
    """

    zeta = np.arccosh((np.asarray(x) + 1j*np.asarray(y))/f)
    xi, eta = zeta.real, zeta.imag
    # the principal branch has xi >= 0; make sign(sin(eta)) = sign(y):
    flip = xi < 0
    xi = np.where(flip, -xi, xi)
    eta = np.where(flip, -eta, eta)

    return xi, np.mod(eta, 2*pi)


def _shape(p, m, eps, X, Y, parity, series=None):
    """Polynomial part C(i*xi)*C(eta) at the waist in units of w0 = 1."""

    s = ince_series(p, m, eps, parity) if series is None else series
    f0 = sqrt(eps/2)
    xi, eta = elliptic_coordinates(X, Y, f0)

    return _radial(s, xi, parity)*ince_polynomial(p, m, eps, eta, parity, s)


def _quadrature(n_nodes):
    """Gauss-Hermite nodes and weights for integrals over the plane in units of
    w0 = 1 with the Gaussian exp(-2*r**2) split off."""

    t, w = np.polynomial.hermite.hermgauss(n_nodes)
    u = t/sqrt(2)  # exp(-2*u**2) = exp(-t**2)
    U, V = np.meshgrid(u, u)
    W = np.outer(w, w)/2

    return U, V, W


def _norm(p, m, eps, parity, s):
    U, V, W = _quadrature(p + 2)

    return sqrt(np.sum(W*_shape(p, m, eps, U, V, parity, s)**2))


def ig_mode(p, m, eps, x, y, w0, wavelength, z=0.0, parity="even", x0=0.0, y0=0.0):
    """Ince-Gaussian mode IG^e_pm or IG^o_pm, normalized to unit power.

    Parameters
    ----------
    p, m : int
        Order (p = N, the order of the HG and LG modes it is made of) and
        index (same parity as p, m <= p; m >= 1 for odd modes).
    eps : number
        Ellipticity parameter, eps > 0. The foci lie at
        x = +-sqrt(eps/2)*w(z).
    x, y : arrays
        Linear or meshed coordinates.
    w0 : number
        Waist radius of the underlying Gaussian beam.
    wavelength : number
        Wavelength in the medium.
    z : number, optional
        Distance from the waist.
    parity : {"even", "odd"}
    x0, y0 : numbers, optional
        Centre of the mode.

    Returns
    -------
    field : 2d array
        Mode field including exp(i*k*z), the wavefront curvature and the
        Gouy phase exp(-i*(p + 1)*psi(z)).
    """

    _check_indices(p, m, parity)
    if eps <= 0:
        raise ValueError("eps must be > 0; use the LG modes for eps = 0.")
    X, Y = ensure_meshgrid(x, y)
    X, Y = X - x0, Y - y0
    k = wavenumber(wavelength)
    z_r = pi*w0**2/wavelength
    w = w0*sqrt(1 + (z/z_r)**2)
    psi = np.arctan2(z, z_r)
    s = ince_series(p, m, eps, parity)
    shape = _shape(p, m, eps, X/w, Y/w, parity, s)/_norm(p, m, eps, parity, s)
    r2 = X**2 + Y**2
    curvature = 0.0 if z == 0 else k*r2*z/(2*(z**2 + z_r**2))

    return shape/w*np.exp(-r2/w**2)*np.exp(1j*(k*z + curvature - (p + 1)*psi))


def helical_ig_mode(p, m, eps, x, y, w0, wavelength, z=0.0, sign=+1, x0=0.0, y0=0.0):
    """Helical Ince-Gaussian mode (IG^e_pm + sign*i*IG^o_pm)/sqrt(2).

    It carries m vortices of charge sign*1 on the interfocal line (for
    eps -> 0 they merge into one vortex of charge sign*m, the LG mode).

    Parameters are those of `ig_mode()`, with m >= 1.
    """

    even = ig_mode(p, m, eps, x, y, w0, wavelength, z, "even", x0, y0)
    odd = ig_mode(p, m, eps, x, y, w0, wavelength, z, "odd", x0, y0)

    return (even + sign*1j*odd)/sqrt(2)


def hg_coefficients(p, m, eps, parity="even"):
    """Exact expansion of an IG mode in HG modes of order p.

    IG_pm = sum_l c_l*HG_{l, p-l} (both from `modes.hg_mode()` with the same
    waist and z).

    Returns
    -------
    coefficients : dict
        {(l, p - l): c} with real c (sum of c**2 is 1).
    """

    s = ince_series(p, m, eps, parity)
    U, V, W = _quadrature(p + 2)
    ig = _shape(p, m, eps, U, V, parity, s)/_norm(p, m, eps, parity, s)
    gauss = np.exp(-(U**2 + V**2))
    out = {}
    for l in range(p, -1, -1):
        hg = hg_mode(l, p - l, U, V, 1.0, 1.0).real
        out[(l, p - l)] = np.sum(W*ig*hg/gauss)

    return out


def lg_coefficients(p, m, eps, parity="even"):
    """Exact expansion of an IG mode in LG modes of order p.

    IG_pm = sum c_(q, l)*LG_(q, l) with 2q + ``|l|`` = p (both from
    `modes.lg_mode()` with the same waist and z).

    Returns
    -------
    coefficients : dict
        {(q, l): c} with complex c (sum of ``|c|**2`` is 1).
    """

    s = ince_series(p, m, eps, parity)
    U, V, W = _quadrature(p + 2)
    ig = _shape(p, m, eps, U, V, parity, s)/_norm(p, m, eps, parity, s)
    gauss = np.exp(-(U**2 + V**2))
    out = {}
    for l in range(-p, p + 1):
        if (p - abs(l)) % 2:
            continue
        q = (p - abs(l))//2
        lg = lg_mode(q, l, U, V, 1.0, 1.0)
        out[(q, l)] = np.sum(W*ig*np.conj(lg)/gauss)

    return out


def mode_count(max_order):
    """Number of even plus odd IG modes up to max_order (as for HG and LG)."""

    return len(ince_indices(max_order, "even")) + len(ince_indices(max_order, "odd"))


__all__ = ["InceSeries", "ince_indices", "ince_series", "ince_polynomial", "elliptic_coordinates",
           "ig_mode", "helical_ig_mode", "hg_coefficients", "lg_coefficients", "mode_count"]
