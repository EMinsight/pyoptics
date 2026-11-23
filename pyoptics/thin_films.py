#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Thin film reflection and transmission coefficients.

Convention: fields vary as exp(i(k.r - omega*t)), as in Zangwill. Absorbing
layers therefore have the complex refractive index n + i*k with k > 0, and a
layer of thickness t adds the phase +k_0*(n + i*k)*cos(theta)*t. McLeod's
characteristic matrices (exp(+i*omega*t), n - i*k) enter as their complex
conjugates.

The amplitude coefficients are ratios of the *tangential* components of the
electric field (characteristic matrix method, Macleod, Yeh). For s
polarization this is the full field. For p polarization, r_p has the
opposite sign of the full-field convention of pyoptics.interfaces (Hecht,
Born & Wolf): here r_p = r_s at normal incidence, there r_p = -r_s. The
transmission t_p is the ratio of the tangential fields, i.e. the full-field
t_p of pyoptics.interfaces times cos(theta_substrate)/cos(theta_0). The
reflectances and transmittances agree.

References
----------
A. Zangwill, "Modern Electrodynamics", Cambridge University Press (2013).

G. Brooker, "Modern Classical Optics", Oxford University Press (2003).

H. A. Macleod, "Thin-Film Optical Filters", 3rd ed., Institute of Physics
Publishing (2001).

P. Yeh, "Optical Waves in Layered Media", Wiley (1988).

M. Born and E. Wolf, "Principles of Optics", 7th ed., Cambridge University
Press (1999).
"""

# Literature:
# Zangwill, Modern Electrodynamics, CUP, 2013
# Brooker, Modern Classical Optics, OUP, 2002
# McLeod, Thin Film Optical Filters, Institute of Physics Publishing, 2001

from copy import copy

import numpy as np
from numpy.lib.scimath import sqrt as complex_sqrt
from numpy import sin

from .utils import Z_0 as Z_vac, wavenumber, freq_grid


def _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k):
    """Extend n and k arrays by embedding medium and substrate values.
    """

    n_extended = np.asarray([embedding_n] + list(n) + [substrate_n])
    k_extended = np.asarray([embedding_k] + list(k) + [substrate_k])

    return n_extended, k_extended


def _impedances(complex_n):
    """Return impedances for stack defined by (complex) refractive indices.
    """

    return Z_vac/complex_n


def _cos_theta(theta_0, complex_n):
    """Compute cosines of complex propagation angles in all media of a given
    stack.

    Parameters
    ----------
    theta_0 : double
        Angle in embedding medium.
    complex_n : array
        Refractive indices for whole stack.

    Returns
    -------
    angles : array
        Array cos(theta_i).
    """

    # cos(theta) = sqrt(1 - sin(theta)); sin(theta) is found from Snell's law:
    cos_theta = complex_sqrt(1.0-(complex_n[0]/complex_n*sin(theta_0))**2)

    # NOTE: n_i sin(theta_i) = n_j sin(theta_j) for any j, i! thus, we can use
    # any layer (here: the embedding medium) to compute angles in any other
    # layer.

    return cos_theta


def _tilted_Z(Z_in, cos_theta, pol):
    """Compute 'tilted impedances' for a stack.

    Parameters
    ----------
    Z_in, cos_theta : arrays
        (Complex) impedances of layer stack and cosines of propagation angles.
    pol : string
        One of {'s','TE'; 'p', 'TM'}.

    Returns
    -------
    Z_tilted : array
        Polarization dependent tilted impedances.
    """

    # NOTE: source is McLeod's Thin Film Optical Filters, 3rd edition. McLeod
    # uses admittances, whereas our code uses impedances (cmp. Zangwil).
    # Admittance is the reciprocal value of impedance.
    # Using admittance y: H = Y*E
    # Using impedance: E = Z*H = 1/Y*H

    Z = copy(Z_in)
    # compute 'tilted' impedance from McLeod's tilted admittances:
    if pol in ("s", "TE"):
        Z /= cos_theta  # McLeod 3rd ed, eq. (8.5)
    elif pol in ("p", "TM"):
        Z *= cos_theta  # McLeod 3rd ed, eq. (8.6)
    else:
        raise ValueError("'pol' must be one of 's'/'TE', 'p'/'TM'")

    return Z


def r_t_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
               substrate_k, pol, wavelength, phi0=None):
    """Reflection and transmission coefficients for a thin film stack.

    Parameters
    ----------
    angle : double
        Incident angle in embedding medium.
    n , k : arrays
        Real and imaginary parts of the refractive indices of the thin film
        stack, n + i*k (k > 0 for absorption).
    t : arrays
        Thicknesses of the layers.
    embedding_n, embedding_k, substrate_n, substrate_k : double
        Real and imaginary parts of the embedding medium's and the substrate's
        refractive index. Both the embedding medium and the substrate are taken
        be infinitely extend.
    pol : string
        "s" or "p" polarization. Also accepted: "TE" or "TM".

    Returns
    -------
    r, t : double
        Reflection and transmission amplitude of the stack.
    """

    # TODO: make checks more verbose and raise sensible exceptions
    assert len(n) >= 1
    assert len(k) >= 1
    assert len(t) == len(n) == len(k)

    n, k = _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k)  # extend n, k arrays by embedding medium and substrate
    r, t = _r_t(np.sin(angle), n + 1j*k, t, pol, wavelength, phi0)

    return r[()], t[()]


def _r_t(sin_theta_0, complex_n, t, pol, wavelength, phi0=None):
    """Reflection and transmission coefficients of a stack for an array of
    incidence angles given by sin(theta_0) in the embedding medium (may be
    complex or > 1 for evanescent incidence).

    complex_n holds the refractive indices of embedding medium, layers and
    substrate; t the thicknesses of the layers only (may be empty). r and t
    have the shape of sin_theta_0.
    """

    sin_theta_0 = np.asarray(sin_theta_0)
    Z = _impedances(complex_n)  # complex impedance for all media

    # cos(theta_j) in all media from Snell's law n_j sin(theta_j) = n_0 sin(theta_0);
    # first axis: media, remaining axes: those of sin_theta_0
    n_col = complex_n.reshape((-1,) + (1,)*sin_theta_0.ndim)
    cos_theta = complex_sqrt(1.0 - (complex_n[0]*sin_theta_0/n_col)**2)
    # grazing incidence (cos(theta) = 0) makes the tilted impedances singular;
    # a tiny cos(theta) gives the limiting values instead:
    cos_theta = np.where(cos_theta == 0, 1E-12, cos_theta)
    Z = _tilted_Z(Z.reshape(n_col.shape)*np.ones_like(cos_theta), cos_theta, pol)  # McLeod's 'tilted admittances'

    if phi0 is None:  # if no additional phases are given, assume zero additional phases
        phi0 = np.zeros(len(complex_n))
    else:
        # additional zeros for substrate and embedding medium; the entries may
        # be arrays that broadcast with sin_theta_0 (ensembles of phases):
        phi0 = [0.] + list(phi0) + [0.]

    # multiply Zangwill's transfer matrices of all layers, element by element
    # for all angles. Layers with evanescent waves or absorption make the
    # entries grow like exp(|Im(phi_j)|); to avoid overflows, each matrix is
    # divided by that factor and the product is kept normalized, with the
    # logarithm of the total scale in log_scale (r does not depend on it):
    A, B = np.ones_like(cos_theta[0]), np.zeros_like(cos_theta[0])
    C, D = np.zeros_like(cos_theta[0]), np.ones_like(cos_theta[0])
    log_scale = np.zeros(np.shape(cos_theta[0]))
    for j in range(1, len(complex_n) - 1):
        phi_j = t[j-1]*wavenumber(wavelength, complex_n[j])*cos_theta[j] + phi0[j]  # 'phase' accumulated in layer j

        g = np.abs(np.imag(phi_j))
        exp_plus, exp_minus = np.exp(1j*phi_j - g), np.exp(-1j*phi_j - g)
        cos_j, sin_j = (exp_plus + exp_minus)/2, (exp_plus - exp_minus)/2j  # cos(phi_j), sin(phi_j) times exp(-g)

        a, b = cos_j, -1j*Z[j]*sin_j
        c, d = -1j/Z[j]*sin_j, cos_j
        A, B, C, D = A*a + B*c, A*b + B*d, C*a + D*c, C*b + D*d

        norm = np.maximum.reduce([np.abs(A), np.abs(B*np.abs(Z[0])**-1), np.abs(C*np.abs(Z[0])), np.abs(D)])
        A, B, C, D = A/norm, B/norm, C/norm, D/norm
        log_scale += g + np.log(norm)

    # solve Zangwill's eq. (17.92) for r and t:
    denominator = A*Z[-1] + B + Z[0]*(C*Z[-1] + D)
    r = (A*Z[-1] + B - Z[0]*(C*Z[-1] + D))/denominator
    t = 2*Z[-1]/denominator*np.exp(-log_scale)

    return r, t


def stack_transfer_functions(x, y, n, k, t, embedding_n, embedding_k, substrate_n,
                             substrate_k, pol, wavelength, normal_order=True):
    """Reflection and transmission of a layer stack as transfer functions for
    the angular spectrum of a field.

    A field in a plane z = const in the embedding medium is a superposition of
    plane waves exp(i*(k_x*x + k_y*y + k_z*z)), whose amplitudes FT() yields
    (see pyoptics.fft). Each plane wave hits the stack (layers parallel to the
    x-y plane, light incident from the embedding medium towards +z) under the
    angle given by sin(theta_0) = sqrt(k_x**2 + k_y**2)/(k_0*n_0) and is
    reflected and transmitted with r(theta_0) and t(theta_0) of
    r_t_coeffs(). Thus ::

        E_reflected = inv_FT(r*FT(E_incident)),
        E_transmitted = inv_FT(t*FT(E_incident)),

    with E_incident and E_reflected taken at the top of the stack (the
    interface embedding medium/first layer) and E_transmitted at its bottom
    (last layer/substrate). Propagate them with the propagators to other
    planes. Note that reflected plane waves run towards -z: with the
    convention exp(i(k.r - omega*t)), propagating the reflected field by a
    distance d away from the stack means using exp(+i*k_z*d) as for the
    incident field, i.e. the usual propagators applied to E_reflected.

    This is a scalar model: the field is one tangential component of the
    electric field, and all plane waves are treated as either s (TE) or p
    (TM) polarized. This is exact for fields that only vary along x (k_y = 0)
    with the electric field along y (pol="TE") or the magnetic field along y
    (pol="TM"), and a good approximation for small angular spectra.
    Plane waves with sin(theta_0) > 1 are evanescent in the embedding medium;
    they are included with the complex angle that the formulas give (e.g.
    frustrated total internal reflection).

    Parameters
    ----------
    x, y : arrays
        Linear coordinates of the field's grid.
    n, k, t : arrays
        Refractive indices n + i*k and thicknesses of the layers (top to
        bottom). Empty arrays describe a single interface.
    embedding_n, embedding_k, substrate_n, substrate_k : numbers
        Refractive index of the embedding medium (incidence side) and the
        substrate.
    pol : string
        "s"/"TE" or "p"/"TM".
    wavelength : number
        Vacuum wavelength.
    normal_order : boolean, optional
        If True (default), the arrays are centered as FT() expects; if False,
        they are in the standard order of numpy.fft.fft2().

    Returns
    -------
    r, t : arrays
        Complex transfer functions of shape (Ny, Nx), ratios of the
        tangential electric field components as in r_t_coeffs().
    """

    assert len(t) == len(n) == len(k)

    K_X, K_Y = freq_grid(x, y, wavenumbers=True, normal_order=normal_order)
    complex_n = (np.asarray([embedding_n] + list(n) + [substrate_n])
                 + 1j*np.asarray([embedding_k] + list(k) + [substrate_k]))
    # r and t only depend on |k_perp|: evaluate the stack once per distinct
    # value (about 8 times fewer than grid points) and map back to the grid
    k_perp_sq, grid_index = np.unique(K_X**2 + K_Y**2, return_inverse=True)
    sin_theta_0 = np.sqrt(k_perp_sq)/wavenumber(wavelength, complex_n[0])
    r, t = _r_t(sin_theta_0, complex_n, t, pol, wavelength)

    return r[grid_index].reshape(K_X.shape), t[grid_index].reshape(K_X.shape)


def _power_factor(complex_n, cos_theta, pol):
    """Re(1/Z_tilted): the z-component of the Poynting vector per |E_tan|**2."""

    return np.real(1/_tilted_Z(_impedances(complex_n)*np.ones_like(cos_theta), cos_theta, pol))


def _R_T_incoherent(sin_theta_0, complex_n, t, pol, wavelength, thick):
    """Exact R and T of a stack whose layers with the indices in `thick`
    (counted from 0 for the first layer) are incoherent.

    The stack is cut at the thick layers into coherent sub-stacks. Each of
    them is described by the power reflectances and transmittances for
    incidence from either side, |r|**2 and |t|**2 of _r_t(), the thick layers
    by their single pass power transmission tau = |exp(i*phi)|**2. Combining
    them from the substrate side upwards, with the multiple reflections in a
    thick layer summed as a geometric series in intensity, gives (Katsidis
    and Papadopoulos)

        R = R_ab + |t_ab t_ba|**2 tau**2 R_b/(1 - R_ba tau**2 R_b),
        T = |t_ab|**2 tau T_b/(1 - R_ba tau**2 R_b),

    with R_b, T_b the values of the part of the stack below the thick
    layer. This is the limit of the phase average of R_T_A_coeffs() for
    infinitely many phases. T is |t|**2; the caller applies the ratio of the
    power factors of substrate and embedding medium.
    """

    sin_theta_0 = np.asarray(sin_theta_0)
    k_0 = wavenumber(wavelength, 1.0)
    media = [0] + [j + 1 for j in thick] + [len(complex_n) - 1]  # indices into complex_n

    def sub_stack(a, b, reverse):
        n_sub, t_sub = complex_n[a:b+1], list(t[a:b-1])
        start = b if reverse else a
        sin_start = complex_n[0]*sin_theta_0/complex_n[start]
        if reverse:
            n_sub, t_sub = n_sub[::-1], t_sub[::-1]
        return _r_t(sin_start, n_sub, t_sub, pol, wavelength)

    # start with the lowest coherent sub-stack (last thick layer/substrate):
    r, t_amp = sub_stack(media[-2], media[-1], False)
    R_b, T_b = np.abs(r)**2, np.abs(t_amp)**2
    for a, b in zip(media[-3::-1], media[-2:0:-1]):  # sub-stacks a -> b above thick layer b
        r_ab, t_ab = sub_stack(a, b, False)
        r_ba, t_ba = sub_stack(a, b, True)
        cos_b = complex_sqrt(1.0 - (complex_n[0]*sin_theta_0/complex_n[b])**2)
        phi_b = k_0*complex_n[b]*cos_b*t[b-1]
        tau = np.exp(-2*np.abs(np.imag(phi_b)))
        denominator = 1 - np.abs(r_ba)**2*tau**2*R_b
        R_b, T_b = (np.abs(r_ab)**2 + np.abs(t_ab*t_ba)**2*tau**2*R_b/denominator,
                    np.abs(t_ab)**2*tau*T_b/denominator)

    return R_b, T_b


def _phase_weights(N_phases, sigmas):
    """Weights of an N_phases**len(sigmas) grid of phase offsets 2*pi*j/N_phases
    in the thick layers for the average over a Gaussian spectrum.

    The offsets of all thick layers are driven by the same wavenumber offset,
    delta_phi_i = sigma_i*xi with xi standard normal. Folded to [0, 2*pi), the
    joint distribution has the Fourier coefficients exp(-(m.sigma)**2/2) for
    the integer harmonics m; the trapezoidal rule on the grid is then exact
    for intensities without harmonics beyond N_phases/2. sigmas = inf for
    all layers gives uniform weights (incoherent average).
    """

    m = np.meshgrid(*[np.fft.fftfreq(N_phases, 1.0/N_phases)]*len(sigmas), indexing="ij")
    m_sigma = sum(m_i*s_i for m_i, s_i in zip(m, sigmas) if np.isfinite(s_i))
    coefficients = np.exp(-m_sigma**2/2)
    for m_i, s_i in zip(m, sigmas):
        if not np.isfinite(s_i):
            coefficients = coefficients*(m_i == 0)

    return np.real(np.fft.ifftn(coefficients))


def R_T_A_coeffs(angle, n, k, t, embedding_n, embedding_k, substrate_n,
                 substrate_k, pol, wavelength, incoherent=None, N_phases=None,
                 bandwidth=None):
    """Same as r_t_coeffs, but for intensity reflectance and transmittance.
    Also return 'absorbtance' such that R + T + A = 1.

    Parameters
    ----------
    angle : double or array
        Incident angle in embedding medium.
    n , k : arrays
        Real and imaginary parts of the refractive indices of the thin film stack.
    t : arrays
        Thicknesses of the layers.
    embedding_n, embedding_k, substrate_n, substrate_k : double
        Real and imaginary parts of the embedding medium's and the substrate's
        refractive index. Both the embedding medium and the substrate are taken
        be infinitely extend.
    pol : string
        "s" or "p" polarization. Also accepted: "TE" or "TM".
    wavelength : number
        Wavelength of incident light.
    incoherent : array, boolean, optional
        Element i is True if the i-th layer is thick (e.g. a glass plate or
        lens between two coatings): its multiple reflections add up in
        intensity (bandwidth=None) or partially coherently (finite
        bandwidth). Defaults to None which is interpreted as full coherence
        in all layers. A coated plate in air is the stack
        [coating 1, plate, coating 2] with the substrate air and the plate
        flagged; the substrate itself cannot be flagged.
    N_phases : int, optional
        Without bandwidth: None (default) gives the exact incoherent sum;
        a number averages over N_phases phase offsets per thick layer
        instead (the older method; its error falls off like
        (R_1 R_2 tau**2)**(N_phases/2) with the reflectances R_1, R_2 on both
        sides of a thick layer and its single pass transmission tau). With
        bandwidth: the number of phase offsets per thick layer (default 64).
    bandwidth : number, optional
        Full width at half maximum of a Gaussian spectrum centred on
        `wavelength`, same unit. Only the thick layers see it: the phase
        k_0*n*d*cos(theta) of each spreads by the relative bandwidth, the
        coherent layers are evaluated at the centre wavelength. The fringes
        of a thick layer fade once the bandwidth exceeds the fringe spacing
        wavelength**2/(2*n*d*cos(theta)); None means the incoherent limit.
        The spread uses the phase index n; for dispersive layers replace the
        thickness by n_g/n times it when estimating, or pass a bandwidth
        scaled by n_g/n (n_g the group index).

    Returns
    -------
    R, T, A : numbers or arrays
        With the shape of angle.

    Notes
    -----
    A = 0 can only be expected for losless media under the condition that the
    embeddig medium is equal to the substrate.

    The exact incoherent sum follows C. C. Katsidis and D. I. Papadopoulos,
    "Analysis of the optical response of multilayers with coherent and
    incoherent layers", Appl. Opt. 41, 3978-3987 (2002); the phase averaging
    for partial coherence is in the spirit of M. C. Troparevsky et al.,
    "Transfer-matrix formalism for the calculation of optical response in
    multilayer systems: from coherent to incoherent interference",
    Opt. Express 18, 24715-24721 (2010).
    """

    if len(t) != len(n) or len(k) != len(n):
        raise ValueError("n, k and t must have the same length")
    thick = []
    if incoherent is not None:
        if len(incoherent) != len(n):
            raise ValueError("incoherent needs one entry per layer ({}), got {}"
                             .format(len(n), len(incoherent)))
        thick = [j for j, flag in enumerate(incoherent) if flag]
    if bandwidth is not None and not thick:
        raise ValueError("bandwidth needs at least one layer flagged in incoherent")
    if bandwidth is not None and bandwidth < 0:
        raise ValueError("bandwidth must not be negative")

    n_ext, k_ext = _n_k_arrays(n, k, embedding_n, embedding_k, substrate_n, substrate_k)
    complex_n = n_ext + 1j*k_ext
    t = np.asarray(t, dtype=float)
    angle = np.asarray(angle, dtype=float)
    sin_theta_0 = np.sin(angle)

    if not thick:  # fully coherent
        r, t_amp = _r_t(sin_theta_0, complex_n, t, pol, wavelength)
        R, T = np.abs(r)**2, np.abs(t_amp)**2
    elif bandwidth is None and N_phases is None:  # exact incoherent sum
        R, T = _R_T_incoherent(sin_theta_0, complex_n, t, pol, wavelength, thick)
    else:  # weighted average over phase offsets in the thick layers
        N = 64 if N_phases is None else int(N_phases)
        if bandwidth is None:
            sigma_rel = np.inf
        else:  # standard deviation of the relative wavenumber spread
            sigma_rel = bandwidth/(2*np.sqrt(2*np.log(2)))/wavelength
        R, T = [], []
        for s in np.atleast_1d(sin_theta_0).ravel():
            cos = complex_sqrt(1.0 - (complex_n[0]*s/complex_n)**2)
            phases = wavenumber(wavelength, 1.0)*np.real(complex_n*cos)[1:-1]*t
            sigmas = [phases[j]*sigma_rel if np.isfinite(sigma_rel) else np.inf for j in thick]
            weights = _phase_weights(N, sigmas).ravel()
            offsets = np.meshgrid(*[2*np.pi*np.arange(N)/N]*len(thick), indexing="ij")
            phi0 = [0.0]*len(n)
            for j, offset in zip(thick, offsets):
                phi0[j] = offset.ravel()
            r, t_amp = _r_t(np.full(N**len(thick), s), complex_n, t, pol, wavelength, phi0)
            R.append(np.sum(weights*np.abs(r)**2))
            T.append(np.sum(weights*np.abs(t_amp)**2))
        R = np.reshape(R, angle.shape)
        T = np.reshape(T, angle.shape)

    # r and t are ratios of the tangential field components. the z-component
    # of the Poynting vector is proportional to |E_tan|**2*Re(1/Z_tilted),
    # which holds for both s and p polarization:
    cos_0 = complex_sqrt(1.0 - sin_theta_0**2 + 0j)
    cos_s = complex_sqrt(1.0 - (complex_n[0]*sin_theta_0/complex_n[-1])**2)
    T = T*_power_factor(complex_n[-1], cos_s, pol)/_power_factor(complex_n[0], cos_0, pol)
    A = 1.0 - R - T

    return R[()], T[()], A[()]
