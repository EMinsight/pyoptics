#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Gridded sag and energy maps in the ray trace (`raytrace.SurfaceMap`,
`raytrace.MapSurface`, ``energy_map``) and the polishing error models of
`pyoptics.polishing`."""

import numpy as np
import pytest

from pyoptics import lens_optimization as lo
from pyoptics import off_axis as oa
from pyoptics import polishing as po
from pyoptics import raytrace as rt


def grid(a=13.0, n=131):
    x = np.linspace(-a, a, n)
    return x, np.meshgrid(x, x)


def asphere():
    return rt.hyperbolic_lens(12.5, 25.0, n=1.5, wavelength=633e-6)


def with_map(system, i, sag_map, **kwargs):
    s = system.surfaces[i]
    S = system.copy()
    S.surfaces[i] = rt.MapSurface(s.R, s.thickness, sag_map=sag_map, n_after=s.n_after,
                                  conic=s.conic, semi_diameter=s.semi_diameter, **kwargs)
    return S


@pytest.mark.parametrize("kind", ["linear", "cubic", "quintic"])
def test_map_interpolates_and_differentiates(kind):
    x = np.linspace(-2, 3, 41)
    y = np.linspace(-1, 2, 31)
    f = lambda X, Y: 0.3*X**3 - X*Y**2 + 0.5*Y   # cubic: reproduced by the splines
    m = rt.SurfaceMap.from_function(f, x, y, kind=kind)
    X, Y = np.meshgrid(x, y)
    assert np.allclose(m(X, Y), f(X, Y), atol=1e-12)
    rng = np.random.default_rng(1)
    xq, yq = rng.uniform(-2, 3, 200), rng.uniform(-1, 2, 200)
    gx, gy = m.gradient(xq, yq)
    h = 1e-6
    assert np.allclose(gx, (m(xq + h, yq) - m(xq - h, yq))/(2*h), atol=1e-6)
    assert np.allclose(gy, (m(xq, yq + h) - m(xq, yq - h))/(2*h), atol=1e-6)
    if kind != "linear":
        assert np.allclose(m(xq, yq), f(xq, yq), atol=1e-10)
        assert np.allclose(gx, 0.9*xq**2 - yq**2, atol=1e-9)
        hxx, hxy, hyy = m.hessian(xq, yq)
        assert np.allclose(hxx, 1.8*xq, atol=1e-8)
        assert np.allclose(hxy, -2*yq, atol=1e-8)
        assert np.allclose(hyy, -2*xq, atol=1e-8)


def test_map_outside_nan_fill_and_scale():
    x = np.linspace(0, 1, 11)
    V = np.outer(x, x)
    V[0, 0] = np.nan  # filled with the nearest valid sample
    m = rt.SurfaceMap(x, x, V, outside="nan")
    assert np.isfinite(m(0.0, 0.0))
    assert np.isnan(m(1.5, 0.5)) and np.all(np.isnan(m.gradient(1.5, 0.5)))
    c = rt.SurfaceMap(x, x, np.outer(x, x), outside="clamp")
    assert np.isclose(c(2.0, 0.5), c(1.0, 0.5))
    assert c.gradient(2.0, 0.5)[0] == 0 and np.isclose(c.gradient(2.0, 0.5)[1], 1.0)
    k = rt.SurfaceMap(x, x, np.outer(x, x), outside=1.0)
    assert k(-1.0, 0.0) == 1.0
    assert np.isclose(c.scaled(3.0)(0.5, 0.5), 0.75)
    assert c == rt.SurfaceMap(x, x, np.outer(x, x)) and c != c.scaled(2.0)
    n = rt.SurfaceMap(x, x, np.outer(x, x), kind="nearest")
    assert n(0.52, 0.48) == np.outer(x, x)[5, 5] and n.gradient(0.52, 0.48)[0] == 0
    with pytest.raises(ValueError):
        rt.SurfaceMap(x, x, np.ones((3, 3)))


def test_zero_map_leaves_the_trace_unchanged():
    S = asphere()
    x, _ = grid()
    S2 = with_map(S, 1, rt.SurfaceMap(x, x, np.zeros((len(x), len(x)))))
    px = np.linspace(-1, 1, 9)
    a, b = S.trace_pupil(0.0, px, 0*px), S2.trace_pupil(0.0, px, 0*px)
    assert np.allclose(a.points, b.points, atol=1e-12)
    assert np.allclose(a.opl, b.opl, atol=1e-12)


def test_map_matches_zernike_surface():
    """A map sampled from a Zernike departure traces like the ZernikeSurface."""

    S = asphere()
    s = S.surfaces[1]
    coeffs = {5: 2e-4, 7: -1e-4, 9: 1.5e-4}
    zs = rt.ZernikeSurface(s.R, s.thickness, zernike=coeffs, norm_radius=12.0, n_after=1.0,
                           conic=s.conic)
    x, (X, Y) = grid(13.0, 261)
    m = rt.SurfaceMap(x, x, zs.departure(X, Y))
    SZ, SM = S.copy(), with_map(S, 1, m)
    SZ.surfaces[1] = zs
    WZ, WM = SZ.opd(0.0, N=33).W, SM.opd(0.0, N=33).W
    ok = np.isfinite(WZ)
    assert np.nanmax(np.abs(WZ)) > 0.1
    assert np.max(np.abs(WZ[ok] - WM[ok])) < 1e-6
    # and on top of Zernike terms of its own
    SB = with_map(S, 1, m.scaled(0.5), zernike=coeffs, norm_radius=12.0)
    SZ.surfaces[1].set_zernike({j: 1.5*c for j, c in coeffs.items()})
    assert np.nanmax(np.abs(SB.opd(0.0, N=33).W - SZ.opd(0.0, N=33).W)) < 1e-6


def test_tilt_map_deflects_like_a_tilted_mirror():
    x = np.linspace(-5, 5, 21)
    a = 1e-3
    m = rt.SurfaceMap.from_function(lambda X, Y: a*Y, x, x)
    S = rt.System([rt.MapSurface(np.inf, -50.0, sag_map=m, mirror=True, semi_diameter=4.0)], 0,
                  633e-6)
    b = S.trace_pupil(0.0, 0.3, 0.2)
    d = b.directions[1, 0]
    assert np.isclose(np.arctan2(d[1], -d[2]), 2*np.arctan(a), atol=1e-12)


def test_ripple_opd_and_psf_satellites():
    """A sinusoidal sag ripple of amplitude a on the exit face adds an OPD
    ripple of about (n - 1) a, and its first diffraction order in the PSF
    sits at f wavelength/period with the relative power (pi (n-1) a/wavelength)**2."""

    S = rt.hyperbolic_lens(5.0, 100.0, n=1.5, wavelength=633e-6)  # slow: near normal incidence
    a, period = 20e-6, 1.0
    x, (X, Y) = grid(5.5, 221)
    SM = with_map(S, 1, rt.SurfaceMap(x, x, a*np.cos(2*np.pi*X/period)))
    rms = SM.rms_opd(0.0, N=129)*S.wavelength
    assert abs(rms/(0.5*a/np.sqrt(2)) - 1) < 0.03
    L = 0.8
    xi = np.linspace(-L/2, L/2, 512, endpoint=False)
    I = np.abs(SM.psf(xi, xi, 0.0, N=161))**2
    I0 = np.abs(S.psf(xi, xi, 0.0, N=161))**2
    row = I[256]
    x_sat = 100.0*633e-6/period
    j = np.argmin(np.abs(xi - x_sat))
    k = j - 4 + np.argmax(row[j - 4:j + 5])
    assert abs(xi[k] - x_sat) < 2*(xi[1] - xi[0])
    # minus the Airy rings of the unperturbed lens under the satellite
    sat = (I - I0)[256 - 8:256 + 9, k - 8:k + 9].sum()
    ratio = sat/I0[256 - 8:256 + 9, 248:265].sum()
    beta = np.pi*0.5*a/633e-6
    assert abs(ratio/beta**2 - 1) < 0.1


def test_energy_maps():
    S = asphere()
    x, _ = grid()
    gauss = rt.SurfaceMap.from_function(lambda X, Y: np.exp(-(X**2 + Y**2)/8.0**2), x, x)
    half = rt.SurfaceMap(x, x, np.full((len(x), len(x)), 0.5), kind="nearest")
    S.surfaces[0].energy_map = gauss
    S.surfaces[1].energy_map = half
    px = np.array([0.0, 0.5, 1.0])
    b = S.trace_pupil(0.0, px, 0*px)
    P = S.ray_power(b)
    r = 12.5*px
    assert P.shape == (4, 3)
    assert np.allclose(P[1], np.exp(-r**2/64), rtol=1e-4)
    assert np.allclose(P[-1], 0.5*np.exp(-r**2/64), rtol=1e-4)
    # polarization trace: bare surface transmittance times the maps
    bare = asphere().trace_pupil_polarization(0.0, px, 0*px).transmittance()
    pol = S.trace_pupil_polarization(0.0, px, 0*px).transmittance()
    assert np.allclose(pol, bare*P[-1], rtol=1e-4)
    # pupil apodization
    pup = S.pupil(0.0, N=33)
    assert np.isclose(pup.mask(0.0, 0.0), np.sqrt(0.5), rtol=1e-3)
    assert pup.mask(0.0, 0.0) > pup.mask(0.0, 0.9) > 0
    assert pup.mask(0.0, 1.2) == 0
    # the geometry is not affected
    b0 = asphere().trace_pupil(0.0, px, 0*px)
    assert np.allclose(b.points, b0.points)


def test_map_surface_derivatives():
    """The differential ray trace covers map surfaces (shape of the base,
    frames) with the spline Hessian of the map."""

    S = rt.hyperbolic_lens(10.0, 40.0, n=1.5, wavelength=633e-6)
    S.field_angle = np.radians(1.0)
    x, (X, Y) = grid(11.0, 89)
    m = rt.SurfaceMap(x, x, po.tool_path_ripple(X, Y, 2.5, 2.0, 2e-4, path="spiral")
                      + po.turned_down_edge(X, Y, 7.0, 10.0, 5e-4))
    SM = with_map(S, 1, m)
    variables = [lo.Curvature(1), lo.Conic(1), lo.Thickness(0), lo.Thickness(1),
                 lo.Tilt(1, 0), lo.Decenter(1, 1)]
    dls = lo.DampedLeastSquares(SM, variables, [])
    lin = dls.linearization()
    assert not np.any(lin.unsupported)
    field = np.radians(1.0)
    px, py = oa.pupil_rings((0.5, 1.0), 6)
    rd = lin.trace(field, px, py)

    def f(system):
        b = system.trace(system.rays(field, px, py), clip=False)
        I, D = system._image_hits(b)
        return np.concatenate([I.ravel(), D.ravel(), b.opl[-1]])

    V = len(rd.dI)
    J = np.concatenate([rd.dI.reshape(V, -1), rd.dD.reshape(V, -1), rd.dopl], axis=1).T
    Jfd = lin.finite_difference(f)
    scale = np.maximum(np.max(np.abs(Jfd), axis=0), 1e-300)
    err = np.max(np.abs(J - Jfd), axis=0)/scale
    assert np.all(err < 1e-5), err


def test_ripple_model():
    p, w, D = 1.0, 1.2, 1e-3
    x = np.linspace(0, 4*p, 400, endpoint=False)
    dz = po.tool_path_ripple(x, 0*x, p, w, D)
    # direct sum of Gaussian tracks of area D*p (mean removal D)
    s = w*po.FWHM_TO_SIGMA
    direct = sum(D*p/(np.sqrt(2*np.pi)*s)*np.exp(-(x - k*p)**2/(2*s**2)) for k in range(-30, 35))
    assert np.allclose(dz, -(direct - direct.mean()), atol=1e-12)
    assert np.isclose(np.ptp(dz)/2, po.ripple_amplitude(p, w, D), rtol=1e-3)
    # spiral: one period per turn along a radius, continuous across phi = pi
    r = np.linspace(3, 5, 7)
    assert np.allclose(po.tool_path_ripple(r, 0*r, p, w, D, path="spiral"),
                       po.tool_path_ripple(r + p, 0*r, p, w, D, path="spiral"))
    assert np.allclose(po.tool_path_ripple(-4.0, 1e-9, p, w, D, path="spiral"),
                       po.tool_path_ripple(-4.0, -1e-9, p, w, D, path="spiral"), atol=1e-12)


def test_turned_down_edge():
    r = np.array([0.0, 5.0, 7.5, 10.0])
    dz = po.turned_down_edge(r, 0*r, 5.0, 10.0, 1e-3, power=2)
    assert np.allclose(dz, [0, 0, -0.25e-3, -1e-3])


def test_random_surface_psd_and_band_rms():
    x = np.arange(256)*0.1
    A, B = 1e-8, 2.5
    psd = lambda f: po.psd_power_law(f, A, B, f_min=0.2, f_max=3.0)
    Z = po.random_surface(x, x, psd, seed=3)
    # expected variance: integral of the PSD over the band
    f = np.linspace(0.2, 3.0, 20001)
    var = np.trapezoid(psd(f)*2*np.pi*f, f)
    assert abs(np.std(Z)/np.sqrt(var) - 1) < 0.1
    assert np.isclose(po.band_rms(x, x, Z), np.std(Z), rtol=1e-9)
    assert po.band_rms(x, x, Z, 3.5) < 1e-3*np.std(Z)
    fc, P = po.radial_psd(x, x, Z, window=False)
    sel = (fc > 0.4) & (fc < 2.5)
    slope = np.polyfit(np.log(fc[sel]), np.log(P[sel]), 1)[0]
    assert abs(slope + B) < 0.3
    assert np.isclose(np.std(po.random_surface(x, x, psd, rms=1e-5, seed=1)), 1e-5)


def test_remove_zernikes_and_defects_and_tis():
    x, (X, Y) = grid(1.0, 101)
    from pyoptics.bases import fringe_zernike
    Z = 3*fringe_zernike(5, X, Y) + 0.01*np.cos(40*X)
    res, c = po.remove_zernikes(X, Y, Z, 1.0, n_terms=16)
    assert np.isclose(c[4], 3.0, atol=0.01)
    assert np.nanstd(res) < 0.0075
    T = po.defect_map(X, Y, digs=[(0.5, 0.5, 0.2)], scratches=[(-1, -0.5, 1, -0.5, 0.05)])
    assert T[np.argmin(abs(x - 0.5)), np.argmin(abs(x - 0.5))] == 0
    assert T[np.argmin(abs(x + 0.5)), 50] == 0 and T[50, 50] == 1
    # TIS: reflection (4 pi sigma/wavelength)**2, refraction (2 pi sigma (n - 1)/wavelength)**2
    assert np.isclose(1 - po.specular_fraction(1e-6, 0.5e-3), (4*np.pi*1e-6/0.5e-3)**2, rtol=1e-3)
    assert np.isclose(1 - po.specular_fraction(1e-6, 0.5e-3, 1.0, 1.5),
                      (2*np.pi*0.5e-6/0.5e-3)**2, rtol=1e-3)
