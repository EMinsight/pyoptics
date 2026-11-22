#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Transverse modes of optical resonators by field tracing (Fox and Li).

A transverse mode of a resonator is a field that reproduces itself after a
round trip, up to a complex factor gamma: ::

    K E = gamma E,

with K the round-trip operator (free space, mirrors, apertures, lenses,
gain or loss elements). |gamma|**2 is the power left after one round trip
(the loss per round trip is 1 - |gamma|**2) and the phase of gamma beyond
the plane wave phase 2 k L is the Gouy phase shift that splits the
frequencies of the transverse modes. The round trip is a
`fieldtrace.Chain`; this module finds its eigenmodes:

- `fox_li()`: the original method of Fox and Li (1961): start with any
  field and go round until it stops changing (power iteration). The field
  converges to the mode with the lowest loss, at a rate set by the ratio of
  the two largest |gamma|.
- `eigenmodes()`: several modes at once with the Arnoldi method
  (scipy.sparse.linalg.eigs) applied to the round-trip operator, which is
  a linear map on the samples of the field in the reference plane.
- `round_trip()`: the chain of a two-mirror resonator (unfolded: the
  mirrors are thin lenses f = R/2, apertures at the mirrors, optional
  elements between them), its reference plane just behind mirror 1.

Reference results:

- `confocal_strip_eigenvalues()`: the one-transit eigenvalues of the
  confocal resonator with strip mirrors, the prolate spheroidal problem of
  Slepian and Pollak (computed by quadrature).
- `gaussian_mode()`, `gouy_round_trip()`: the Gaussian beam mode and the
  round-trip Gouy phase of a stable resonator (from abcd).
- `unstable_magnification()`, `geometric_loss()`: the round-trip
  magnification of an unstable resonator and its geometric output coupling
  1 - 1/M**d (d = 1 strips, d = 2 circular mirrors), around which the
  diffraction losses oscillate with the equivalent Fresnel number
  (`equivalent_fresnel_number()`).

Sign conventions: mirror radii R > 0 for concave mirrors (focusing, f =
R/2), as in abcd.g_parameters(): g = 1 - L/R. Fields vary as
exp(i(k z - omega t)), so a mode with the round-trip phase phi (beyond
2 k L) resonates at nu_q = FSR (q - phi/(2 pi)).

References
----------
A. G. Fox and T. Li, "Resonant modes in a maser interferometer", Bell
Syst. Tech. J. 40, 453 (1961).
G. D. Boyd and J. P. Gordon, "Confocal multimode resonator for millimeter
through optical wavelength masers", Bell Syst. Tech. J. 40, 489 (1961).
A. E. Siegman, "Lasers", University Science Books (1986), chapters 19-23.
D. Slepian and H. O. Pollak, "Prolate spheroidal wave functions, Fourier
analysis and uncertainty I", Bell Syst. Tech. J. 40, 43 (1961).
"""

import warnings
from collections import namedtuple

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

from . import fieldtrace as _ft
from .source_modes import field_at
from .utils import wavenumber


__all__ = ["ResonatorMode", "round_trip", "fox_li", "eigenmodes", "confocal_strip_eigenvalues",
           "gaussian_mode", "gouy_round_trip", "unstable_magnification", "geometric_loss",
           "equivalent_fresnel_number", "fresnel_number", "frequency_offsets", "GridPropagate"]


class ResonatorMode(namedtuple("ResonatorMode", ["gamma", "field", "phase"])):
    """A transverse mode.

    Attributes
    ----------
    gamma : complex
        Round-trip eigenvalue (the plane wave phase of the round trip
        removed when it is known).
    field : Field
        The mode in the reference plane (unit power).
    phase : float
        arg(gamma), the round-trip phase beyond 2 k L (rad).
    """

    @property
    def loss(self):
        """Power loss per round trip, 1 - |gamma|**2."""
        return 1 - abs(self.gamma)**2

    def __repr__(self):
        return "ResonatorMode(loss={:.4g}, phase={:.4f} rad)".format(self.loss, self.phase)


def round_trip(L, R1=np.inf, R2=np.inf, a1=None, a2=None, n=1.0, elements=(),
               elements_back=None, method="grid", shape="circle", pad=2):
    """Round-trip chain of a two-mirror resonator, unfolded.

    The reference plane lies just behind mirror 1 (after its aperture and
    its reflection). The round trip is ::

        Propagate(L) [elements] Aperture(a2) Mirror(R2)
        Propagate(L) [elements_back] Aperture(a1) Mirror(R1)

    where a mirror is a thin lens of focal length R/2 (R > 0 concave) and
    the elements are split in the middle of the cavity if given as
    `elements` only (traversed on the way out and again, mirrored, on the
    way back).

    Parameters
    ----------
    L : number
        Mirror spacing.
    R1, R2 : number, optional
        Mirror radii (inf: plane).
    a1, a2 : number, optional
        Mirror aperture radii (circle) or half widths (strip; also for
        one-dimensional fields); None: unlimited.
    n : number, optional
        Index of the medium (the fields must be in it).
    elements : sequence of Step, optional
        Steps (e.g. a ThinLens for a thermal lens, an Aperture) placed in
        the middle of the cavity: the free space is split in two halves,
        and the elements act on the way out and on the way back.
    elements_back : sequence of Step, optional
        Different steps for the way back (default: the same).
    method : str, optional
        "grid" (default): GridPropagate, the angular spectrum on the
        fixed grid of the reference plane, the right model for mirrors
        with hard edges; or a fieldtrace operator ("spw", "fresnel", ...;
        a fixed operator keeps the round trip a linear map), for soft
        apertures or none.
    pad : int, optional
        Zero padding of GridPropagate.
    shape : str, optional
        "circle" or "square" (a square aperture of half width a).

    Returns
    -------
    Chain
        With the attribute `optical_length` (2 n L) for the plane wave
        phase of the round trip.
    """
    def aperture(a, label):
        if a is None:
            return []
        return [_ft.Aperture(mask=_EdgeMask(a, shape), label=label)]

    def mirror(R, label):
        if np.isinf(R):
            return []
        s = _ft.ThinLens(R/2)
        s.label = "{} R={:g}".format(label, R)
        return [s]

    elements = list(elements)
    back = list(elements if elements_back is None else elements_back)

    def prop(d):
        if method == "grid":
            return GridPropagate(d, pad=pad)
        return _ft.Propagate(d, method=method)

    def leg(steps):
        if not steps:
            return [prop(L)]
        return [prop(L/2)] + steps + [prop(L/2)]

    steps = (leg(elements) + aperture(a2, "mirror 2 aperture") + mirror(R2, "mirror 2")
             + leg(back) + aperture(a1, "mirror 1 aperture") + mirror(R1, "mirror 1"))
    chain = _ft.Chain(steps)
    chain.optical_length = 2*n*L
    return chain


class _EdgeMask(object):
    """Hard-edged circular or square aperture whose edge samples transmit
    the share of the sample (a linear ramp over one sample spacing). A
    binary mask moves the edge by up to half a sample, which shifts the
    losses of a resonator noticeably; the ramp places it at a to second
    order."""

    def __init__(self, a, shape="circle"):
        if shape not in ("circle", "square"):
            raise ValueError("shape must be 'circle' or 'square'")
        self.a, self.shape = float(a), shape

    def __call__(self, X, Y=None):
        X = np.asarray(X, dtype=float)
        if Y is None:
            d = np.abs(X)
            h = abs(X[1] - X[0])
        else:
            Y = np.asarray(Y, dtype=float)
            h = abs(X[0, 1] - X[0, 0])
            d = np.hypot(X, Y) if self.shape == "circle" else np.maximum(np.abs(X), np.abs(Y))
        return np.clip((self.a - d)/h + 0.5, 0.0, 1.0)


def _plane_wave_phase(chain, wavelength):
    length = getattr(chain, "optical_length", None)
    if length is None:
        return 1.0
    return np.exp(1j*wavenumber(wavelength)*length)


def _grid_points(x, y):
    if y is None:
        return np.asarray(x, dtype=float), None
    return np.meshgrid(x, y)


class _Operator(object):
    """The round trip as a linear map on samples a of the amplitude in the
    reference plane, E = a*exp(i alpha_ref r**2)."""

    def __init__(self, chain, x, y, wavelength, n, alpha_ref, polarization):
        self.chain = chain if isinstance(chain, _ft.Chain) else _ft.Chain(chain)
        self.x, self.y = np.asarray(x, dtype=float), (None if y is None else np.asarray(y, float))
        self.wavelength, self.n = wavelength, n
        self.alpha = alpha_ref
        self.jones = _ft._normalize_jones(polarization)
        self.X, self.Y = _grid_points(self.x, self.y)
        self.shape = self.X.shape
        self.phase0 = np.conj(_plane_wave_phase(self.chain, wavelength))
        self.count = 0

    def field(self, a):
        a = np.asarray(a).reshape(self.shape)
        alpha = (self.alpha[0], 0.0 if self.y is None else self.alpha[1])
        return _ft.Field(a, self.x, self.y, self.wavelength, self.n, alpha=alpha, jones=self.jones)

    def apply(self, a):
        self.count += 1
        out = self.chain.run(self.field(a), polarization="scalar").field
        if isinstance(out, _ft.FieldSum):
            out = out.combined()
        Ex, Ey = field_at(out, self.X, self.Y)
        E = Ex*np.conj(self.jones[0]) + Ey*np.conj(self.jones[1])
        q = self.alpha[0]*self.X**2 + (0.0 if self.Y is None else self.alpha[1]*self.Y**2)
        return (E*np.exp(-1j*q)*self.phase0).ravel()


def _start_field(x, y, wavelength, n, polarization):
    """A broad, slightly asymmetric start field that contains every mode."""
    X, Y = _grid_points(x, y)
    w = 0.3*(x[-1] - x[0])
    r2 = X**2 + (0.0 if Y is None else Y**2)
    A = np.exp(-r2/w**2)*(1 + 0.3*X/w + (0.0 if Y is None else 0.2*Y/w))
    return _ft.Field(A, x, y, wavelength, n, jones=_ft._normalize_jones(polarization))


def _reference_alpha(chain, start, passes=3):
    """Analytic curvature of the field in the reference plane after a few
    round trips (keeps the sampled amplitude smooth)."""
    f = start
    for _ in range(passes):
        out = chain.run(f, polarization="scalar").field
        if isinstance(out, _ft.FieldSum):
            out = out.combined()
        P = out.power()
        if not P > 0:
            break
        f = out.resample(start.x, None if start.one_d else start._y)
        f = f._with(f._A/np.sqrt(max(f.power(), 1E-300)))
    return (out.alpha[0], out.alpha[1])


def _mode(op, a, gamma):
    f = op.field(a)
    P = f.power()
    f = f._with(f._A/np.sqrt(P))
    # fix the global phase: real and positive at the maximum
    A = f._A[..., 0]
    i = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    f = f._with(f._A*np.exp(-1j*np.angle(A[i])))
    return ResonatorMode(complex(gamma), f, float(np.angle(gamma)))


def fox_li(chain, x, y=None, wavelength=None, n=1.0, n_iter=300, start=None, tol=1E-10,
           alpha_ref=None, polarization="x", history=True):
    """Fox-Li iteration: trace a field round and round the resonator.

    Parameters
    ----------
    chain : Chain or sequence of steps
        The round trip (see round_trip()).
    x, y : arrays
        Grid of the reference plane (y None: strip resonator, one
        dimension).
    wavelength : number
    n : number, optional
    n_iter : int, optional
        Maximum number of round trips.
    start : Field or array, optional
        Start field (default: a broad, asymmetric Gaussian).
    tol : float, optional
        Stop when the field changes by less than tol (relative L2 norm)
        and the eigenvalue estimate by less than tol.
    alpha_ref : pair, optional
        Analytic curvature of the reference plane (default: taken from the
        field after three round trips).
    polarization : optional
    history : bool, optional
        Also return the loss after every round trip.

    Returns
    -------
    mode : ResonatorMode
    losses : array (if history)
        1 - |gamma_i|**2 after round trip i, gamma_i = <E_i, K E_i>/<E_i, E_i>.
    """
    chain = chain if isinstance(chain, _ft.Chain) else _ft.Chain(chain)
    if start is None:
        start = _start_field(x, y, wavelength, n, polarization)
    elif not isinstance(start, _ft.Field):
        start = _ft.Field(start, x, y, wavelength, n, jones=_ft._normalize_jones(polarization))
    if alpha_ref is None:
        alpha_ref = _reference_alpha(chain, start)
    op = _Operator(chain, x, y, wavelength, n, alpha_ref, polarization)
    X, Y = op.X, op.Y
    Ex, Ey = field_at(start, X, Y)
    q = alpha_ref[0]*X**2 + (0.0 if Y is None else alpha_ref[1]*Y**2)
    a = ((Ex*np.conj(op.jones[0]) + Ey*np.conj(op.jones[1]))*np.exp(-1j*q)).ravel()
    a = a/np.linalg.norm(a)
    losses = []
    gamma_old = None
    for _ in range(n_iter):
        b = op.apply(a)
        gamma = np.vdot(a, b)
        losses.append(1 - abs(gamma)**2)
        nb = np.linalg.norm(b)
        if not nb > 0:
            raise RuntimeError("the field left the resonator completely")
        b_n = b/nb
        change = np.linalg.norm(b_n*np.exp(-1j*np.angle(np.vdot(a, b_n))) - a)
        a = b_n
        if gamma_old is not None and change < tol and abs(gamma - gamma_old) < tol:
            break
        gamma_old = gamma
    mode = _mode(op, a, gamma)
    return (mode, np.array(losses)) if history else mode


def eigenmodes(chain, x, y=None, wavelength=None, n=1.0, n_modes=4, alpha_ref=None, start=None,
               solver="auto", dense_limit=2048, tol=1E-9, polarization="x", ncv=None,
               maxiter=None):
    """The lowest-loss transverse modes of a resonator (Arnoldi method).

    The round trip is a linear map on the field samples in the reference
    plane. Its eigenvalues of largest modulus belong to the modes with the
    smallest round-trip losses. For small grids (one-dimensional strip
    resonators) the matrix is built column by column (one round trip per
    sample) and diagonalized; for large ones scipy.sparse.linalg.eigs
    (Arnoldi) finds the leading eigenvalues from repeated round trips.

    Parameters
    ----------
    chain : Chain or sequence of steps
        The round trip, with fixed free-space operators (see round_trip()).
    x, y : arrays
        Grid of the reference plane (y None: one dimension).
    wavelength : number
    n : number, optional
    n_modes : int, optional
    alpha_ref : pair, optional
        Analytic curvature of the reference plane (default: from three
        Fox-Li round trips).
    start : Field, optional
        Start field for the curvature estimate.
    solver : str, optional
        "auto" (dense up to dense_limit samples, else Arnoldi), "dense" or
        "arnoldi".
    dense_limit : int, optional
    tol : float, optional
        Relative accuracy of the eigenvalues (Arnoldi).
    polarization : optional
    ncv, maxiter : optional
        Passed to eigs.

    Returns
    -------
    list of ResonatorMode
        Sorted by increasing loss.
    """
    chain = chain if isinstance(chain, _ft.Chain) else _ft.Chain(chain)
    if start is None:
        start = _start_field(x, y, wavelength, n, polarization)
    if alpha_ref is None:
        alpha_ref = _reference_alpha(chain, start)
    op = _Operator(chain, x, y, wavelength, n, alpha_ref, polarization)
    N = int(np.prod(op.shape))
    k = min(n_modes, N - 2)
    if solver == "auto":
        solver = "dense" if N <= dense_limit else "arnoldi"
    if solver == "dense":
        # the matrix of the round trip, column by column
        M = np.empty((N, N), dtype=complex)
        e = np.zeros(N, dtype=complex)
        for j in range(N):
            e[j] = 1.0
            M[:, j] = op.apply(e)
            e[j] = 0.0
        vals, vecs = np.linalg.eig(M)
    elif solver == "arnoldi":
        A = LinearOperator((N, N), matvec=op.apply, dtype=complex)
        Ex, Ey = field_at(start, op.X, op.Y)
        v0 = (Ex*np.conj(op.jones[0]) + Ey*np.conj(op.jones[1])).ravel()
        if ncv is None:
            ncv = int(min(N, max(4*k, 2*k + 1, 24)))
        try:
            vals, vecs = eigs(A, k=k, which="LM", v0=v0, tol=tol, ncv=ncv, maxiter=maxiter)
        except ArpackNoConvergence as err:
            if len(err.eigenvalues) == 0:
                raise
            warnings.warn("eigenmodes: only {} of {} modes converged".format(
                len(err.eigenvalues), k), RuntimeWarning)
            vals, vecs = err.eigenvalues, err.eigenvectors
    else:
        raise ValueError("solver must be 'auto', 'dense' or 'arnoldi'")
    order = np.argsort(-np.abs(vals))
    modes = [_mode(op, vecs[:, i], vals[i]) for i in order[:k]]
    return modes


def frequency_offsets(modes, L, n=1.0):
    """Frequency offsets nu - nu_q of the modes relative to the axial
    plane wave resonance, -phase/(2 pi) times the free spectral range
    c/(2 n L) (in units of c: multiply by the speed of light).

    Parameters
    ----------
    modes : sequence of ResonatorMode
    L : number
    n : number, optional

    Returns
    -------
    array
        Offsets in units of the free spectral range, wrapped to [0, 1).
    """
    phases = np.array([m.phase for m in modes])
    return np.mod(-phases/(2*np.pi), 1.0)


# --------------------------------------------------------------------------
# analytic references
# --------------------------------------------------------------------------

def fresnel_number(a, L, wavelength, n=1.0):
    """Fresnel number N = a**2/(lambda L) of a mirror of radius a."""
    return n*a**2/(wavelength*L)


def confocal_strip_eigenvalues(N, n_modes=4, n_quad=None):
    """One-transit eigenvalues of the symmetric confocal resonator with
    strip mirrors of half width a (Fresnel number N = a**2/(lambda L)).

    The transit operator (Boyd and Gordon 1961) is the finite Fourier
    transform ::

        sigma u(x) = sqrt(c/(2 pi)) e^(-i pi/4) int_-1^1 exp(i c x x') u(x') dx',   c = 2 pi N,

    whose eigenfunctions are the prolate spheroidal functions and whose
    eigenvalues |sigma_n|**2 are the energy concentrations of Slepian and
    Pollak. Computed here by Gauss-Legendre quadrature.

    Parameters
    ----------
    N : number
        Fresnel number.
    n_modes : int, optional
    n_quad : int, optional
        Quadrature points (default: enough for c).

    Returns
    -------
    sigma : complex array
        Eigenvalues sorted by decreasing modulus; the round-trip loss of
        mode j is 1 - |sigma_j|**4.
    """
    c = 2*np.pi*N
    if n_quad is None:
        n_quad = int(max(64, 4*c + 40))
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    s = np.sqrt(wg)
    K = np.sqrt(c/(2*np.pi))*np.exp(-1j*np.pi/4)*np.exp(1j*c*np.outer(xg, xg))*s[:, None]*s[None, :]
    vals = np.linalg.eigvals(K)
    vals = vals[np.argsort(-np.abs(vals))]
    return vals[:n_modes]


def gaussian_mode(L, R1, R2, wavelength, n=1.0):
    """Gaussian (TEM00) mode of a stable two-mirror resonator.

    Returns
    -------
    w1, w2 : float
        Beam radii (1/e**2) on mirror 1 and mirror 2.
    w0 : float
        Waist radius.
    z1 : float
        Position of the waist measured from mirror 1 towards mirror 2.
    """
    g1, g2 = 1 - L/R1, 1 - L/R2
    G = g1*g2
    if not 0 < G < 1:
        raise ValueError("the resonator is not stable (g1*g2 = {:.4g})".format(G))
    lam = wavelength/n
    w1 = np.sqrt(lam*L/np.pi*np.sqrt(g2/(g1*(1 - G))))
    w2 = np.sqrt(lam*L/np.pi*np.sqrt(g1/(g2*(1 - G))))
    w0 = np.sqrt(lam*L/np.pi*np.sqrt(G*(1 - G)/(g1 + g2 - 2*G)**2))
    z1 = g2*(1 - g1)*L/(g1 + g2 - 2*G)
    return w1, w2, w0, z1


def gouy_round_trip(L, R1, R2, l=0, m=0, circular_p=None):
    """Round-trip phase of the Hermite-Gaussian mode TEM_lm of a stable
    resonator beyond 2 k L: -2 (l + m + 1) arccos(+-sqrt(g1 g2)) (the sign
    of the root that of g1; one-dimensional modes: (l + 1/2)), so that the
    resonance is offset by (l + m + 1) arccos(...)/pi free spectral ranges.

    Parameters
    ----------
    L, R1, R2 : number
    l, m : int, optional
        Mode orders (m None: one-dimensional strip mode, factor l + 1/2).
    circular_p : int, optional
        Radial order p of a Laguerre-Gaussian mode LG_pl instead (factor
        2p + |l| + 1).

    Returns
    -------
    float
        The round-trip phase (rad), wrapped to (-pi, pi].
    """
    g1, g2 = 1 - L/R1, 1 - L/R2
    root = np.sqrt(g1*g2)*(1 if g1 >= 0 else -1)
    theta = np.arccos(root)
    if circular_p is not None:
        order = 2*circular_p + abs(l) + 1
    elif m is None:
        order = l + 0.5
    else:
        order = l + m + 1
    return float(np.angle(np.exp(-2j*order*theta)))


def unstable_magnification(L, R1, R2):
    """Round-trip magnification of an unstable resonator (|M| > 1),
    M = 2 g1 g2 - 1 +- 2 sqrt(g1 g2 (g1 g2 - 1)) (positive branch
    g1 g2 > 1: M > 1; negative branch g1 g2 < 0: M < -1)."""
    g1, g2 = 1 - L/R1, 1 - L/R2
    G = g1*g2
    if 0 <= G <= 1:
        raise ValueError("the resonator is stable (g1*g2 = {:.4g})".format(G))
    r = 2*np.sqrt(G*(G - 1))
    M1, M2 = 2*G - 1 + r, 2*G - 1 - r
    return M1 if abs(M1) > abs(M2) else M2


def geometric_loss(M, dims=2):
    """Geometric output coupling of an unstable resonator per round trip,
    1 - 1/|M|**dims (dims = 1: strip, 2: circular mirrors)."""
    return 1 - 1/abs(M)**dims


def equivalent_fresnel_number(a, L, M, wavelength):
    """Equivalent Fresnel number N_eq = (M - 1)/2 * a**2/(L lambda) of a
    positive branch confocal unstable resonator whose small (output
    coupling) mirror has the radius or half width a (Siegman, section
    22.3). The diffraction losses oscillate about the geometric ones with
    the period 1 in N_eq."""
    return (abs(M) - 1)*a**2/(2*L*wavelength)


class GridPropagate(_ft.Step):
    """Free-space propagation on the field's own grid (angular spectrum,
    rigorous transfer function, zero padding by `pad`).

    The fieldtrace operators choose their output window for the beam,
    which is what a single pass needs. A resonator needs the opposite: its
    mirrors clip the field to a fixed window, hard aperture edges diffract
    into wide angles that the mirrors throw away anyway, and the round trip
    must be the same linear map every time. This operator keeps the grid
    (the classic Fox-Li discretization); the light that leaves the window
    is lost: the zero padding and the band limit of Matsushima and
    Shimobaba (2009) keep it from wrapping around.

    Parameters
    ----------
    z : number
        Distance (in the field's medium).
    pad : int, optional
        Zero padding factor of the FFT.
    """

    def __init__(self, z, pad=2):
        self.z = float(z)
        self.pad = int(pad)
        self.label = "grid propagate {:g}".format(self.z)

    def apply(self, field, ctx):
        notes = []
        x = field.x
        one_d = field.one_d
        y = None if one_d else field._y
        X, Y = _grid_points(x, y)
        Ex, Ey = field_at(field, X, Y)
        # the analytic phase is sampled on the grid: check for aliasing
        q = field.alpha[0]*X**2 + field.beta[0]*X
        if not one_d:
            q = q + field.alpha[1]*Y**2 + field.beta[1]*Y
        steps = [np.max(np.abs(np.diff(q, axis=-1)))]
        if not one_d:
            steps.append(np.max(np.abs(np.diff(q, axis=0))))
        if max(steps) > np.pi:
            notes.append("phase aliased on the grid (step {:.2f} rad)".format(max(steps)))
        k = wavenumber(field.wavelength, field.n)
        comps = [Ex, Ey] if field.is_vector else [Ex*np.conj(field.jones[0])
                                                  + Ey*np.conj(field.jones[1])]
        outs = [_asm(c, x, y, self.z, k, self.pad) for c in comps]
        A = np.stack(outs, axis=-1) if field.is_vector else outs[0]
        out = _ft.Field(A, x, y, field.wavelength, field.n,
                        jones=None if field.is_vector else field.jones, z=field.z + self.z)
        return out, {"method": "grid ASM", "notes": notes}


def _band_limit(kx, z, k, width):
    """Matsushima-Shimobaba band limit: plane waves steeper than the
    padded window allows would wrap around; they would leave the window
    (and the mirrors) anyway, so they are dropped."""
    f_lim = k/(2*np.pi)/np.sqrt((2*abs(z)/width)**2 + 1)
    return np.abs(kx) <= 2*np.pi*f_lim


def _asm(E, x, y, z, k, pad):
    """Band-limited angular spectrum propagation of samples E on a fixed
    grid (zero padded by pad)."""
    dx = _ft._spacing(x)
    Nx = len(x)
    Px = pad*Nx
    kx = 2*np.pi*np.fft.fftfreq(Px, dx)
    if y is None:
        F = np.fft.fft(E, Px)
        kz = np.sqrt((k**2 - kx**2).astype(complex))
        H = np.exp(1j*kz*z)*_band_limit(kx, z, k, Px*dx)
        return np.fft.ifft(F*H)[:Nx]
    dy = _ft._spacing(y)
    Ny = len(y)
    Py = pad*Ny
    ky = 2*np.pi*np.fft.fftfreq(Py, dy)
    F = np.fft.fft2(E, (Py, Px))
    kz = np.sqrt((k**2 - kx[None, :]**2 - ky[:, None]**2).astype(complex))
    H = (np.exp(1j*kz*z)*_band_limit(kx, z, k, Px*dx)[None, :]
         *_band_limit(ky, z, k, Py*dy)[:, None])
    return np.fft.ifft2(F*H)[:Ny, :Nx]
