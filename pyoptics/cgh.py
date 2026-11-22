#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Computer-generated holograms (CGHs) for null tests of aspheres.

An asphere is measured in an interferometer by comparing the wave it
reflects with a reference wave. Without help, a concave asphere in the
spherical test wave of a Fizeau transmission sphere (TS) returns hundreds
of fringes, too dense for the camera, and rays that come back on other
paths than they went (retrace error). A CGH between the TS and the asphere
diffracts the test wave into the shape of the asphere: every ray of the
measuring order meets the asphere at normal incidence and returns on
itself, the interferogram of a perfect asphere is a null, and what is
left is the surface form deviation (SFD) of the asphere, twice over
(double pass).

This module designs and analyses such holograms with the sequential ray
tracer: the hologram is a `raytrace.ZernikePhaseSurface` (a linear
carrier plus a Zernike phase), the test arm an unfolded double pass
(`raytrace.double_pass()`).

Design
------
- `null_phase()`: the phase that turns the incoming wave (traced through
  the TS and the CGH substrate) into the wave that travels along the
  normals of the test surface. The optical path from the CGH to the test
  surface along the normals and the incoming optical path give the phase
  exactly (Fermat), which is fitted with Zernikes over the footprint; no
  optimization is needed. The carrier is chosen beforehand (a tilt carrier
  with the test surface placed on the tilted axis, or none).
- `focus_phase()` and `retro_phase()`: alignment zones. A ring that
  focuses the light onto a point of the test surface (the vertex, a cat's
  eye that fixes the spacing; or projected fiducials), and a ring that
  reflects the incoming wave back on itself in a chosen order (Littrow),
  which aligns the CGH to the interferometer.
- `binary_mask()`: the binary pattern written by the e-beam or laser
  writer.

Analysis
--------
- `order_paths()`: every combination of orders of the forward and return
  passes, traced, with the angles at which they leave towards the
  interferometer and the fraction that passes its stop. The measuring
  path (+1, +1) passes; all others must be blocked (or separated) by the
  carrier.
- `InterferometerStop`: the pinhole in the focus of the interferometer's
  collimator as an angular filter on the collimated return beam, a plane
  surface for `raytrace.System` (and so for `fieldtrace.Trace`, which turns
  the surviving ghost paths into ghost fringes).
- `binary_orders()`: the complex amplitudes of the orders of a binary
  phase grating in the scalar (thin element) model, and
  `binary_orders_rcwa()` the same rigorously (`litho.rcwa`): the efficiency
  budget, and how etch depth and duty cycle errors change the phase of
  the measuring order.
- `distortion_phase()`: the wavefront error of a pattern placement error
  (writer distortion), -m*grad(phi).epsilon per pass.

Reconstruction of the SFD
-------------------------
- `remove_alignment()`: least squares fit of piston and the phase
  sensitivities to the alignment degrees of freedom of the test surface
  (computed by finite differences of traced phase maps), which removes
  the misalignment in the null (tilt, power, coma, and the higher terms
  that a tilted asphere adds).
- `surface_map()`: the remaining phase as a surface error on the test
  surface: phase/(2 k n) along the normal (double pass, minus sign: a bump
  that rises towards the light shortens the path), at the points where the
  rays met the surface (the mapping from the camera to the surface is
  strongly distorted by the null), optionally converted into a sag error.
- `zernike_fit()`: Zernike fit of scattered data (the SFD on the surface
  coordinates, or any phase on a non-square grid).

Sign conventions are those of `pyoptics.raytrace` and
`pyoptics.interferometry`: phase = +k*OPL, a longer test path gives a
larger phase. Lengths in the unit of the system (mm).

References
----------
J. C. Wyant and V. P. Bennett, "Using computer generated holograms to test
aspheric wavefronts", Appl. Opt. 11, 2833-2839 (1972).

J. H. Burge, "Applications of computer-generated holograms for
interferometric measurement of large aspheric optics", Proc. SPIE 2576,
258-269 (1995).

Y.-C. Chang and J. H. Burge, "Error analysis for CGH optical testing",
Proc. SPIE 3782, 358-366 (1999).

P. Zhou and J. H. Burge, "Fabrication error analysis and experimental
demonstration for computer-generated holograms", Appl. Opt. 46, 657-663
(2007).

D. Malacara (ed.), "Optical Shop Testing", 3rd ed., Wiley (2007), chapter
15 (null tests using compensators and CGHs).
"""

from collections import namedtuple

import numpy as np
from numpy import pi

from .bases import zernike_indices, zernike_nm
from .raytrace import Surface, ZernikePhaseSurface


__all__ = ["NullDesign", "OrderPath", "InterferometerStop", "zernike_basis", "zernike_fit",
           "incoming_path", "null_phase", "focus_phase", "retro_phase", "binary_mask",
           "order_paths", "binary_orders", "binary_orders_rcwa", "distortion_phase",
           "remove_alignment", "surface_map"]


# --- Zernike fits of scattered data ---

def zernike_basis(x, y, norm_radius, terms, ordering="noll", centre=(0.0, 0.0)):
    """Matrix of Zernike polynomials at scattered points.

    Parameters
    ----------
    x, y : 1d arrays
        Points.
    norm_radius : number
        Radius of the unit disk.
    terms : int or sequence of int
        Indices j of the polynomials (an int J: 1...J).
    ordering : {"noll", "standard", "fringe"}, optional
        See `bases.zernike_indices()`.
    centre : (x0, y0), optional
        Centre of the disk.

    Returns
    -------
    M : (len(x), len(terms)) array
    js : list of int
    """

    js = list(range(1, terms + 1)) if np.isscalar(terms) else [int(j) for j in terms]
    u = (np.asarray(x, dtype=float) - centre[0])/norm_radius
    v = (np.asarray(y, dtype=float) - centre[1])/norm_radius
    M = np.empty((u.size, len(js)))
    for k, j in enumerate(js):
        n, m, scale = zernike_indices(j, ordering)
        M[:, k] = scale*zernike_nm(n, m, u.ravel(), v.ravel())

    return M, js


def zernike_fit(x, y, values, norm_radius=None, terms=37, ordering="fringe", centre=(0.0, 0.0),
                mask=None):
    """Least squares Zernike fit of scattered (or masked gridded) data.

    Parameters
    ----------
    x, y, values : arrays of equal shape
        Coordinates and data; NaNs are left out.
    norm_radius : number, optional
        Radius of the unit disk (default: the largest distance of a valid
        point from the centre).
    terms : int or sequence of int, optional
        Indices j (an int J: 1...J).
    ordering : {"fringe", "noll", "standard"}, optional
    centre : (x0, y0), optional
    mask : bool array, optional
        Points to use.

    Returns
    -------
    coefficients : dict
        {j: c_j}.
    residual : array
        values minus the fit, NaN where not fitted (shape of values).
    norm_radius : number
    """

    x, y, values = (np.asarray(a, dtype=float) for a in np.broadcast_arrays(x, y, values))
    ok = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    if norm_radius is None:
        norm_radius = float(np.max(np.hypot(x[ok] - centre[0], y[ok] - centre[1])))
    M, js = zernike_basis(x[ok], y[ok], norm_radius, terms, ordering, centre)
    c = np.linalg.lstsq(M, values[ok], rcond=None)[0]
    residual = np.full(values.shape, np.nan)
    residual[ok] = values[ok] - M @ c

    return dict(zip(js, c)), residual, norm_radius


# --- design ---

NullDesign = namedtuple("NullDesign", ["surface", "x", "y", "phase", "test_points",
                                       "fit_residual", "norm_radius"])
NullDesign.__doc__ = """Result of `null_phase()` (and of the alignment zone designs).

surface : raytrace.ZernikePhaseSurface
    Copy of the hologram with the fitted phase.
x, y : 1d arrays
    Design points on the hologram (its local frame).
phase : 1d array
    The exact phase there (radians, with the carrier).
test_points : (N, 3) array or None
    The points of the test surface (its local frame) that the design rays
    from (x, y) hit, i.e. the mapping hologram -> test surface.
fit_residual : number
    RMS of the Zernike fit residual in waves (of the phase, i.e. of the
    wavefront of one pass in the design order).
norm_radius : number
    Normalization radius of the Zernike fit.
"""


def _grid_disk(N):
    p = np.linspace(-1.0, 1.0, N)
    PX, PY = np.meshgrid(p, p)
    inside = PX**2 + PY**2 <= 1.0

    return PX[inside], PY[inside]


def incoming_path(system, index, N=81, terms=231, field=0.0):
    """Optical path of the incoming wave on a surface, as a smooth function.

    Rays over the entrance pupil are traced up to surface `index` (a
    diffractive surface there in the order 0, i.e. only refracting), and
    their optical path is fitted with Noll Zernikes over the footprint in
    the local frame of the surface.

    Parameters
    ----------
    system : raytrace.System
    index : int
        Surface index.
    N : int, optional
        Rays across the pupil.
    terms : int, optional
        Number of Noll Zernikes of the fit.
    field : number or pair, optional

    Returns
    -------
    opl : callable
        opl(x, y) in the local frame (length units).
    footprint : (x, y)
        The traced points.
    rms : number
        RMS fit residual in length units.
    """

    sysm = system.copy()
    s = sysm.surfaces[index]
    if hasattr(s, "with_order"):
        sysm.surfaces[index] = s.with_order(0)
    px, py = _grid_disk(N)
    b = sysm.trace(sysm.rays(field, px, py), clip=False, last=index)
    ok = b.valid
    o, R = sysm.frames()[0][index]
    Q = (b.points[index + 1, ok] - o) @ R
    opl = b.opl[index + 1, ok]
    rq = float(np.max(np.hypot(Q[:, 0], Q[:, 1])))
    M, js = zernike_basis(Q[:, 0], Q[:, 1], rq, terms)
    c = np.linalg.lstsq(M, opl, rcond=None)[0]
    rms = float(np.std(M @ c - opl))

    def path(x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return (zernike_basis(x.ravel(), y.ravel(), rq, js)[0] @ c).reshape(x.shape)

    return path, (Q[:, 0], Q[:, 1]), rms


def _fit_phase(cgh, x, y, phase, terms, ordering, centre, norm_radius):
    """Fit phase minus carrier with Zernikes, return a configured copy."""

    rest = phase - 2*pi*(cgh.carrier[0]*x + cgh.carrier[1]*y)
    if norm_radius is None:
        norm_radius = float(np.max(np.hypot(x - centre[0], y - centre[1])))
    js = list(range(2, terms + 1)) if np.isscalar(terms) else [int(j) for j in terms if j != 1]
    M, _ = zernike_basis(x, y, norm_radius, [1] + js, ordering, centre)
    c = np.linalg.lstsq(M, rest, rcond=None)[0]
    res = rest - M @ c
    out = ZernikePhaseSurface.__new__(ZernikePhaseSurface)
    out.__dict__.update(cgh.__dict__)
    out.ordering = "noll" if ordering.lower() == "standard" else ordering.lower()
    out.norm_radius = float(norm_radius)
    out.centre = (float(centre[0]), float(centre[1]))
    out.set_zernike(dict(zip(js, c[1:])), replace=True)

    return out, float(np.std(res)/(2*pi)), norm_radius


def _phase_for_path(cgh, dopl, wavelength):
    """Phase whose optical path m*lambda*phi/(2 pi) (sign flipped on a
    mirror) equals dopl."""

    return dopl*2*pi/(cgh._sign*cgh.order*wavelength)


def _sag_xy(surface, x, y):
    if hasattr(surface, "sag_xy"):
        return surface.sag_xy(x, y)

    return surface.sag(x**2 + y**2)


def null_phase(system, cgh_index, test_index, N=81, terms=231, ordering="noll",
               centre=(0.0, 0.0), norm_radius=None, field=0.0, aperture=None):
    """Null hologram for a test surface, from the normals of the surface.

    The measuring wave must meet the test surface at normal incidence. So
    it travels from the hologram along the normals of the test surface,
    and its optical path from a hologram point C to the surface is
    L(C) = n*|Q - C|, Q the foot of the normal through C. The rays of
    the incoming wave reach C with the optical path OPL_in(C) (traced through
    the system in front, `incoming_path()`). The hologram adds the path
    m*lambda*phi/(2*pi), so the total path to the test surface is the same
    for all rays if ::

        m*lambda*phi(C)/(2*pi) = -(OPL_in(C) + L(C)) + const.

    That is exact; the normals are cast from an N x N grid over the clear
    aperture (semi_diameter) of the test surface back to the hologram, and
    phi minus the carrier is fitted with Zernikes. Since the normals fix
    the wave, no optimization is involved; the result is checked by tracing
    the double pass (the residual of the fit is reported in waves).

    Parameters
    ----------
    system : raytrace.System
        The forward pass: everything in front of the hologram, the
        hologram (a `raytrace.ZernikePhaseSurface` with the carrier and
        the order of the design; the test surface placed on the axis of
        that order, e.g. decentred and tilted for a tilt carrier) and the
        test surface, with nothing in between.
    cgh_index, test_index : int
        Indices of the hologram and of the test surface.
    N : int, optional
        Normals across the aperture of the test surface (and rays across
        the pupil for the incoming wave).
    terms : int or sequence of int, optional
        Zernike indices of the fit (an int J: 2...J; piston is dropped).
        A steep asphere needs high orders: 231 Noll terms reach the radial
        order 20.
    ordering : {"noll", "standard", "fringe"}, optional
    centre : (x0, y0), optional
        Centre of the Zernikes on the hologram.
    norm_radius : number, optional
        Default: the radius of the footprint of the normals.
    field : number or pair, optional
        Field of the incoming wave.
    aperture : number, optional
        Radius on the test surface to design for (default: its semi
        diameter).

    Returns
    -------
    NullDesign
    """

    cgh = system.surfaces[cgh_index]
    if not isinstance(cgh, ZernikePhaseSurface):
        raise TypeError("The hologram must be a ZernikePhaseSurface.")
    path_in, _, _ = incoming_path(system, cgh_index, N=N, field=field)
    frames, _ = system.frames()
    o, R = frames[cgh_index]
    test = system.surfaces[test_index]
    ot, Rt = frames[test_index]
    a = test.semi_diameter if aperture is None else aperture
    if a is None:
        raise ValueError("The test surface needs a semi_diameter (or give aperture).")
    u, v = _grid_disk(N)
    u, v = a*u, a*v
    Pt = np.stack([u, v, _sag_xy(test, u, v)], axis=-1)
    Ng = test.normal(Pt) @ Rt.T
    Pg = ot + Pt @ Rt.T
    toward = np.sign(np.sum(Ng*(o - Pg), axis=-1))
    d = Ng*toward[:, None]
    Pl, dl = (Pg - o) @ R, d @ R
    t = cgh.intersect(Pl, dl)
    C = Pl + t[:, None]*dl
    L = np.real(system.n[cgh_index + 1])*t
    phi = _phase_for_path(cgh, -(L + path_in(C[:, 0], C[:, 1])), system.wavelength)
    ok = np.isfinite(phi)
    x, y, phi = C[ok, 0], C[ok, 1], phi[ok]
    surf, res, rn = _fit_phase(cgh, x, y, phi, terms, ordering, centre, norm_radius)

    return NullDesign(surf, x, y, phi, Pt[ok], res, rn)


def _ring_points(cgh, r_in, r_out, N, centre):
    s = np.linspace(-r_out, r_out, N)
    X, Y = np.meshgrid(s, s)
    r = np.hypot(X, Y)
    m = (r >= r_in) & (r <= r_out)
    x, y = X[m] + centre[0], Y[m] + centre[1]

    return x, y, np.zeros_like(x)


def focus_phase(system, cgh_index, point, r_in, r_out, N=81, terms=136, ordering="noll",
                centre=(0.0, 0.0), field=0.0):
    """Alignment zone that focuses the incoming wave onto a point.

    m*lambda*phi/(2*pi) = -(OPL_in + n*|P - C|): the order m converges to
    the point P (global coordinates), e.g. the vertex of the test surface.
    Reflected there like in a cat's eye, the light returns through the
    opposite side of the ring and leaves as the reversed incoming wave
    (reciprocity), giving a null whose defocus measures the spacing of
    the test surface from the hologram. Placed on a plane (a reticle or a
    fiducial), the focus marks a position on the test surface.

    Parameters
    ----------
    system : raytrace.System
        Everything in front of the hologram, and the hologram (a
        `raytrace.ZernikePhaseSurface` with the carrier and order of the
        zone).
    cgh_index : int
    point : (x, y, z)
        Focus in global coordinates.
    r_in, r_out : numbers
        Inner and outer radius of the zone on the hologram.
    N : int, optional
        Design points across the zone.
    terms : int or sequence, optional
    ordering : str, optional
    centre : (x0, y0), optional
        Centre of the zone (and of the Zernikes).
    field : number or pair, optional

    Returns
    -------
    NullDesign
        test_points is None.
    """

    cgh = system.surfaces[cgh_index]
    path_in, _, _ = incoming_path(system, cgh_index, N=N, field=field)
    o, R = system.frames()[0][cgh_index]
    x, y, _ = _ring_points(cgh, r_in, r_out, N, centre)
    C = np.stack([x, y, _sag_xy(cgh, x, y)], axis=-1)
    Cg = o + C @ R.T
    L = np.real(system.n[cgh_index + 1])*np.linalg.norm(np.asarray(point, dtype=float) - Cg, axis=-1)
    phi = _phase_for_path(cgh, -(L + path_in(x, y)), system.wavelength)
    surf, res, rn = _fit_phase(cgh, x, y, phi, terms, ordering, centre, r_out)

    return NullDesign(surf, x, y, phi, None, res, rn)


def retro_phase(system, cgh_index, r_in, r_out, N=81, terms=66, ordering="noll",
                centre=(0.0, 0.0), field=0.0):
    """Alignment zone that reflects the incoming wave back on itself (Littrow).

    The hologram must be a mirror (mirror=True): in the reflected order m
    the zone adds the path -2*OPL_in, so every ray returns on itself
    and the zone gives a null fringe pattern when the hologram is aligned
    to the incoming wave. Its tilt fringes measure the tilt of the
    hologram, its power fringes the spacing to the focus of the TS, and
    its coma the decentre.

    Parameters
    ----------
    system : raytrace.System
        Everything in front of the hologram and the hologram (a reflecting
        `raytrace.ZernikePhaseSurface`, any carrier).
    cgh_index : int
    r_in, r_out : numbers
        Inner and outer radius of the zone.
    N, terms, ordering, centre, field : optional
        As for `focus_phase()`.

    Returns
    -------
    NullDesign
    """

    cgh = system.surfaces[cgh_index]
    if not cgh.mirror:
        raise ValueError("A Littrow zone works in reflection: give the hologram mirror=True.")
    path_in, _, _ = incoming_path(system, cgh_index, N=N, field=field)
    x, y, _ = _ring_points(cgh, r_in, r_out, N, centre)
    phi = _phase_for_path(cgh, -2*path_in(x, y), system.wavelength)
    surf, res, rn = _fit_phase(cgh, x, y, phi, terms, ordering, centre, r_out)

    return NullDesign(surf, x, y, phi, None, res, rn)


def binary_mask(phase, duty=0.5):
    """Binary pattern of a phase function: the lines of a binary hologram.

    Parameters
    ----------
    phase : array
        Phase in radians (not wrapped).
    duty : number, optional
        Duty cycle (fraction of each period that is etched or written).

    Returns
    -------
    mask : bool array
        True on the lines, centred on phase = 0 mod 2 pi.
    """

    f = np.mod(np.asarray(phase)/(2*pi) + duty/2, 1.0)

    return f < duty


# --- orders and stray light ---

class InterferometerStop(Surface):
    """Pinhole in the focus of the interferometer's collimator, as an angular
    filter on the collimated return beam.

    The interferometer focuses the returning light through a small stop
    before the camera. A ray that leaves the test arm with the direction d
    lands at f*(d_x/d_z, d_y/d_z) in the focal plane of the collimator (focal
    length f); it passes if that point lies inside the pinhole of radius a,
    i.e. if its slope is below a/f. This surface is a plane without power
    that removes the other rays (as `trace()` removes vignetted rays), so
    the stop can be placed anywhere in the collimated return beam of a
    `raytrace.System`, and `fieldtrace.Trace` then builds the camera field
    only from the light that passes. Diffraction at the pinhole is not
    modelled.

    Parameters
    ----------
    max_slope : number
        a/f, the largest transverse slope (tangent of the angle with the
        local z axis) that passes.
    thickness : number, optional
        Distance to the next surface (or the image plane).
    **kwargs
        Passed to `raytrace.Surface` (e.g. label, semi_diameter).
    """

    has_seidel = False

    def __init__(self, max_slope, thickness=0.0, **kwargs):
        super().__init__(np.inf, thickness, **kwargs)
        self.max_slope = float(max_slope)

    def __repr__(self):
        return "InterferometerStop(max_slope={}, thickness={})".format(self.max_slope, self.thickness)

    def deflect(self, Q, d, n1, n2):
        """Pass the rays within max_slope, NaN for the others.

        Parameters
        ----------
        Q : (N, 3) array
            Intersection points.
        d : (N, 3) array
            Unit directions.
        n1, n2 : numbers
            Refractive indices (equal).

        Returns
        -------
        d2 : array
            Directions, NaN for blocked rays.
        dopl : array
            Zeros.
        """

        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.hypot(d[:, 0], d[:, 1])/np.abs(d[:, 2])
        d2 = np.where((slope <= self.max_slope)[:, None], d, np.nan)

        return d2, np.zeros(len(d))


OrderPath = namedtuple("OrderPath", ["orders", "slopes", "fraction", "passing", "px", "py"])
OrderPath.__doc__ = """One path of `order_paths()`.

orders : tuple of int
    Diffraction orders at the hologram surfaces, in the order of the trace.
slopes : (N, 2) array
    Exit slopes (d_x/d_z, d_y/d_z) of the rays in the local frame of the
    image plane (times the focal length of the collimator: the points in the
    plane of the interferometer's stop); NaN for lost rays.
fraction : number
    Fraction of the rays launched that pass the stop (and every aperture).
passing : bool array
    Which rays pass.
px, py : arrays
    Normalized pupil coordinates of the rays.
"""


def order_paths(system, indices, orders=(-2, -1, 0, 1, 2), max_slope=None, N=41, field=0.0,
                clip=True):
    """Trace every combination of diffraction orders through a hologram
    that is passed several times (an unfolded double pass).

    Parameters
    ----------
    system : raytrace.System
        The unfolded test arm.
    indices : sequence of int
        Indices of the hologram surfaces in the trace (forward and return
        pass of a double pass: two indices).
    orders : sequence of int or sequence of sequences, optional
        Orders to combine at every index (one sequence for all, or one per
        index).
    max_slope : number, optional
        Acceptance a/f of the interferometer's stop (see
        `InterferometerStop`); None: every ray that reaches the image plane
        passes.
    N : int, optional
        Rays across the pupil.
    field : number or pair, optional
    clip : bool, optional
        Vignette rays at the semi diameters.

    Returns
    -------
    list of OrderPath
        One per combination, in the order of itertools.product.
    """

    import itertools

    indices = [int(i) for i in indices]
    if np.ndim(orders[0]) == 0:
        orders = [tuple(orders)]*len(indices)
    px, py = _grid_disk(N)
    _, R_img = system.frames()[1]
    rays = system.rays(field, px, py)  # the object side does not depend on the orders
    out = []
    for combo in itertools.product(*orders):
        sysm = system.copy()
        for i, m in zip(indices, combo):
            sysm.surfaces[i] = sysm.surfaces[i].with_order(m)
        b = sysm.trace(rays, clip=clip)
        d = b.directions[-1] @ R_img
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = d[:, :2]/np.abs(d[:, 2:3])
        ok = b.valid & np.all(np.isfinite(slopes), axis=1)
        if max_slope is not None:
            ok &= np.hypot(slopes[:, 0], slopes[:, 1]) <= max_slope
        slopes[~b.valid] = np.nan
        out.append(OrderPath(tuple(combo), slopes, float(np.mean(ok)), ok, px, py))

    return out


def binary_orders(orders, duty=0.5, depth=pi):
    """Complex amplitudes of the orders of a binary phase grating (scalar,
    thin element model).

    A grating with the phase step `depth` over the fraction `duty` of each
    period transmits ::

        t_0 = 1 - D + D*exp(i*depth),
        t_m = (exp(i*depth) - 1)*sin(pi*m*D)/(pi*m)*exp(-i*pi*m*D)   (m != 0)

    for lines starting at the origin; centred on it, as `binary_mask()`
    draws them, the last factor drops out. The efficiencies are ``abs(t_m)**2``:
    for D = 1/2 and depth = pi, 40.5 % in each first order, no zeroth
    and no even orders. The phase of every order m != 0 is
    depth/2 + pi/2 (+ pi where the sine changes sign), so a depth error that
    varies over the hologram writes half its phase error into the measuring
    order, while the duty cycle changes only the amplitude (in the scalar
    model; see `binary_orders_rcwa()`).

    Parameters
    ----------
    orders : int or sequence of int
    duty : number, optional
        Duty cycle D.
    depth : number, optional
        Phase step in radians, 2*pi*(n - 1)*h/lambda for an etch depth h
        in a substrate of index n (in transmission), 4*pi*h/lambda for a
        reflective (chrome on glass or etched) pattern in air.

    Returns
    -------
    t : complex array
        Amplitudes of the orders.
    """

    m = np.atleast_1d(np.asarray(orders, dtype=float))
    step = np.exp(1j*depth) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(m == 0, 1 + duty*step, step*np.sin(pi*m*duty)/(pi*np.where(m == 0, 1, m)))

    return t if np.ndim(orders) else t[0]


def binary_orders_rcwa(period, duty, depth, wavelength, n_substrate, orders=(-1, 0, 1),
                       polarization="TE", n_orders=25, n_out=1.0):
    """Complex amplitudes of the transmitted orders of a binary grating
    etched into a substrate, rigorously (`litho.rcwa`).

    Unlike the thin element model, it includes the Fresnel losses and
    the coupling of the duty cycle into the phase of the orders at small
    periods (a few wavelengths), where the scalar model fails.

    Light comes from the substrate (index n_substrate) at normal incidence,
    passes the etched lines (substrate material in a fraction `duty` of each
    period, of height `depth`) and leaves into the medium n_out. The
    amplitudes are normalized so that their squared moduli are the
    efficiencies (power per order). The lines are centred on x = 0, so the
    phases compare with `binary_orders()`.

    Parameters
    ----------
    period : number
        Grating period (length unit of the wavelength).
    duty : number
        Fraction of the period that keeps the substrate (the ridge).
    depth : number
        Etch depth.
    wavelength : number
    n_substrate : number
    orders : sequence of int, optional
    polarization : {"TE", "TM"}, optional
        TE: E along the lines (y).
    n_orders : int, optional
        Orders -n_orders...n_orders of the calculation.
    n_out : number, optional

    Returns
    -------
    t : complex array
        Amplitudes of the requested orders (sqrt of the efficiency times the
        phase of the tangential field), with the phase referred to the
        plane of the etched bottom (the convention of `binary_orders()`, in
        which the light between the ridges has the phase 0 and the ridges
        the phase `depth`).
    """

    from .litho.rcwa import Layer, rcwa

    layer = Layer(depth, n_out, segments=[(0.0, duty*period, n_substrate)])
    res = rcwa(period, [layer], wavelength, n_in=n_substrate, n_out=n_out, n_orders=n_orders)
    jones = (0.0, 1.0) if polarization.upper() == "TE" else (1.0, 0.0)
    _, T = res.efficiencies(jones)
    J = res.t[0]
    comp = 1 if polarization.upper() == "TE" else 0
    amp = J[:, comp, comp]
    out = []
    for m in np.atleast_1d(orders):
        i = int(np.flatnonzero(res.m == m)[0])
        out.append(np.sqrt(T[0, i])*np.exp(1j*np.angle(amp[i])))
    # rcwa() refers the transmitted field to the bottom of the layer; refer it
    # to the plane where the substrate ends, like the thin element model (the
    # light between the ridges then has the phase 0)
    return np.array(out)*np.exp(-2j*pi*n_out*depth/wavelength)


def distortion_phase(surface, x, y, shift, order=None, passes=1):
    """Phase error of a pattern placement error (writer distortion).

    A hologram whose lines are displaced by epsilon(x, y) has the phase
    phi(r - epsilon) ~ phi - grad(phi).epsilon, so the diffracted wave
    of the order m gains ::

        delta_phi = -m*grad(phi).epsilon   per pass.

    The error is largest where the lines are densest (it is the fraction
    of a period by which the lines move), and only the component across
    the lines counts.

    Parameters
    ----------
    surface : raytrace.ZernikePhaseSurface
    x, y : arrays
        Points of the hologram.
    shift : (ex, ey) or callable
        Displacement of the pattern (arrays broadcast with x, y), or a
        function (x, y) -> (ex, ey).
    order : int, optional
        Default: the order of the surface.
    passes : int, optional
        Number of passes in that order (2 for a double pass).

    Returns
    -------
    delta_phi : array
        Phase error in radians.
    """

    m = surface.order if order is None else order
    ex, ey = shift(x, y) if callable(shift) else shift
    gx, gy = surface.phase_gradient(x, y)

    return -passes*m*(gx*ex + gy*ey)


# --- reconstruction ---

def remove_alignment(phase, sensitivities, mask=None, piston=True):
    """Fit and remove the misalignment terms of a null test.

    phase ~ c_0 + sum_k a_k S_k, with the sensitivities S_k = d(phase)/d(p_k)
    to the alignment degrees of freedom p_k of the test surface (its
    decentres, tilts and axial position, computed by tracing the test arm
    with small displacements); the residual is the measurement of the
    surface. Unlike a fit of tilt, power and coma Zernikes, the
    sensitivities carry the exact (non-null) signatures of the asphere,
    including the distortion of the mapping.

    Parameters
    ----------
    phase : array
        Measured (unwrapped) phase.
    sensitivities : sequence of arrays
        Phase per unit displacement of each degree of freedom, on the grid
        of phase (NaN outside).
    mask : bool array, optional
        Valid pixels (default: where everything is finite).
    piston : bool, optional
        Fit a constant as well.

    Returns
    -------
    p : array
        Fitted displacements (units of the sensitivities' steps).
    residual : array
        phase minus the fit (NaN outside the mask).
    """

    phase = np.asarray(phase, dtype=float)
    S = [np.asarray(s, dtype=float) for s in sensitivities]
    ok = np.isfinite(phase)
    for s in S:
        ok &= np.isfinite(s)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    cols = [s[ok] for s in S]
    b = phase[ok]
    if piston:
        # centred columns: a large constant in a sensitivity (an unbalanced
        # piston of the finite differences) must not cost precision
        cols = [c - np.mean(c) for c in cols]
        b = b - np.mean(b)
    A = np.column_stack(cols) if cols else np.zeros((ok.sum(), 0))
    scale = np.linalg.norm(A, axis=0) if cols else np.ones(0)
    scale[scale == 0] = 1.0
    c = np.linalg.lstsq(A/scale, b, rcond=None)[0]/scale
    residual = np.full(phase.shape, np.nan)
    residual[ok] = b - A @ c

    return c, residual


SurfaceMap = namedtuple("SurfaceMap", ["u", "v", "height", "sag", "mask"])
SurfaceMap.__doc__ = """Result of `surface_map()`.

u, v : arrays
    Coordinates on the test surface (its local frame) of every camera pixel.
height : array
    Surface error along the normal (positive: material added, the surface
    rises towards the incoming light).
sag : array
    The same as an error of the sag z(u, v) along the local +z axis of
    the surface (height divided by the z component of the normal that
    points towards the light): for a mirror lit from -z, as in a
    `raytrace.ZernikeSurface` departure, a positive sag error is a dip.
mask : bool array
"""


def surface_map(phase, ray_map, system, test_index, mask=None, passes=2):
    """Surface error of the test surface from the residual phase of a null test.

    At normal incidence, a surface error h along the normal shortens the
    path by 2*n*h (double pass), so ::

        h = -phase*lambda/(2*pi*passes*n),

    attributed to the point where the ray of that camera pixel met the
    test surface (`System.ray_map(..., surface=test_index)`): the null
    maps the camera onto the surface with a strong distortion, so Zernikes
    of the surface must be fitted in these coordinates (`zernike_fit()`),
    not in the camera's. Valid to first order in h (no retrace error), for
    a reflecting test surface.

    Parameters
    ----------
    phase : array
        Residual phase on the camera (after `remove_alignment()`).
    ray_map : raytrace.RayMap
        Ray map of the test arm with surface=test_index.
    system : raytrace.System
        The test arm the map was made with.
    test_index : int
        Index of the test surface (forward pass).
    mask : bool array, optional
    passes : int, optional
        Number of reflections at the test surface (2 for a double pass
        in a Fizeau or Twyman-Green).

    Returns
    -------
    SurfaceMap
    """

    o, R = system.frames()[0][test_index]
    P = (ray_map.points - o) @ R
    D = ray_map.directions @ R  # directions after the reflection
    test = system.surfaces[test_index]
    n = np.real(system.n[test_index])
    ok = (np.isfinite(phase) & ray_map.mask & np.all(np.isfinite(P), axis=-1)
          & np.all(np.isfinite(D), axis=-1))
    if mask is not None:
        ok &= mask
    h = np.where(ok, -np.asarray(phase)*system.wavelength/(2*pi*passes*n), np.nan)
    u, v = np.where(ok, P[..., 0], np.nan), np.where(ok, P[..., 1], np.nan)
    # normal towards the incoming light (the side the reflected rays leave to)
    Nrm = np.full(P.shape, np.nan)
    N_ok = test.normal(P[ok])
    Nrm[ok] = N_ok*np.sign(np.sum(N_ok*D[ok], axis=-1))[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        sag = h/Nrm[..., 2]

    return SurfaceMap(u, v, h, sag, ok)
