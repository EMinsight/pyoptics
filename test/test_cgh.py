#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests for raytrace.ZernikePhaseSurface and the CGH null test tools of
pyoptics.cgh (notebook 503, CGH null testing of aspheres)."""

import numpy as np
import pytest
from numpy import pi

from pyoptics import cgh
from pyoptics.raytrace import (GratingSurface, Surface, System, ZernikePhaseSurface,
                               ZernikeSurface, double_pass)

WL = 632.8e-6
L = 30.0      # CGH to the vertex of the test sphere
R_T = -50.0   # concave test sphere, its centre 20 mm behind the CGH


def _forward(hologram, test=None, carrier_tilt=True, dz=0.0, decenter=(0.0, 0.0)):
    """Collimated beam (radius 5) -> CGH (stop) -> concave sphere on the axis of the order."""
    theta = np.arcsin(WL*hologram.carrier[1]*hologram.order) if carrier_tilt else 0.0
    h = ZernikePhaseSurface.__new__(ZernikePhaseSurface)
    h.__dict__.update(hologram.__dict__)
    h.thickness = (L + dz)*np.cos(theta)
    kw = dict(mirror=True, semi_diameter=11.0, decenter=(decenter[0], (L + dz)*np.sin(theta) + decenter[1]),
              tilt=(-theta, 0.0, 0.0))
    t = Surface(R_T, **kw) if test is None else ZernikeSurface(R_T, zernike=test, norm_radius=11.0, **kw)
    return [h, t]


def _system(surfaces):
    return System(surfaces, 0, WL, ray_aiming=False)


def _test_arm(hologram, **kw):
    return System(double_pass(_forward(hologram, **kw), image_distance=-5.0), 0, WL, ray_aiming=False)


@pytest.fixture(scope="module")
def design():
    blank = ZernikePhaseSurface(carrier=(0.0, 20.0), semi_diameter=6.0)
    return cgh.null_phase(_system(_forward(blank)), 0, 1, N=41, terms=66)


def _opl_spread(system, N=21):
    px, py = cgh._grid_disk(N)
    b = system.trace(system.rays(0.0, px, py))
    ok = b.valid
    return np.ptp(b.opl[-1, ok]), b, ok


# ZernikePhaseSurface

def test_carrier_is_a_grating():
    """A pure carrier diffracts like a GratingSurface of period 1/f."""
    f = 100.0
    d = np.array([[0.0, np.sin(0.1), np.cos(0.1)], [0.0, 0.0, 1.0]])
    Q = np.zeros((2, 3))
    phs = ZernikePhaseSurface(carrier=(0.0, f), n_after=1.5)
    grat = GratingSurface(period=1/f, n_after=1.5)
    for order in (-2, -1, 1, 2):
        a, _ = phs.with_order(order).deflect(Q, d, 1.0, 1.5, WL)
        b, _ = grat.with_order(order).deflect(Q, d, 1.0, 1.5, WL)
        np.testing.assert_allclose(a, b, atol=1e-14)
        # grating equation
        np.testing.assert_allclose(1.5*a[:, 1], d[:, 1] + order*WL*f, atol=1e-14)


def test_phase_gradient_matches_finite_differences():
    s = ZernikePhaseSurface(carrier=(3.0, -2.0), zernike={4: 5.0, 7: -2.0, 12: 1.5, 37: 0.7},
                            norm_radius=4.0, centre=(0.3, -0.2))
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-3, 3, 20), rng.uniform(-3, 3, 20)
    gx, gy = s.phase_gradient(x, y)
    h = 1e-6
    np.testing.assert_allclose(gx, (s.phase_at(x + h, y) - s.phase_at(x - h, y))/(2*h), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gy, (s.phase_at(x, y + h) - s.phase_at(x, y - h))/(2*h), rtol=1e-6, atol=1e-6)
    period = s.local_period(x, y)
    np.testing.assert_allclose(period, 2*pi/np.hypot(gx, gy))


def test_path_of_the_phase():
    """The optical path gains m*lambda*phi/(2 pi) (sign flipped on a mirror)."""
    s = ZernikePhaseSurface(zernike={5: 3.0}, norm_radius=2.0, order=2)
    Q = np.array([[1.0, 0.5, 0.0]])
    d = np.array([[0.0, 0.0, 1.0]])
    _, dopl = s.deflect(Q, d, 1.0, 1.0, WL)
    np.testing.assert_allclose(dopl, 2*WL*s.phase_at(1.0, 0.5)/(2*pi))
    m = ZernikePhaseSurface(zernike={5: 3.0}, norm_radius=2.0, order=2, mirror=True)
    _, dopl = m.deflect(Q, d, 1.0, 1.0, WL)
    np.testing.assert_allclose(dopl, -2*WL*s.phase_at(1.0, 0.5)/(2*pi))


# design

def test_null_phase_gives_a_null(design):
    assert design.fit_residual < 1e-4
    spread, b, ok = _opl_spread(_test_arm(design.surface))
    assert ok.mean() > 0.5
    assert spread/WL < 1e-3
    # every ray meets the sphere at normal incidence: it returns on itself
    np.testing.assert_allclose(b.directions[-1, ok], np.tile([0.0, 0.0, -1.0], (ok.sum(), 1)), atol=1e-6)


def test_null_phase_maps_onto_the_test_surface(design):
    """test_points lie on the sphere, within its aperture."""
    P = design.test_points
    assert np.all(np.hypot(P[:, 0], P[:, 1]) <= 11.0 + 1e-9)
    s = Surface(R_T)
    np.testing.assert_allclose(P[:, 2], s.sag(P[:, 0]**2 + P[:, 1]**2), atol=1e-12)


def test_null_phase_without_carrier_is_a_lens():
    """On axis the null for a sphere is the spherical wave from its centre:
    a diverging lens of focal length -(|R| - L), no aberration beyond it."""
    blank = ZernikePhaseSurface(semi_diameter=6.0)
    d = cgh.null_phase(_system(_forward(blank, carrier_tilt=False)), 0, 1, N=41, terms=66)
    r = np.hypot(d.x, d.y)
    f = abs(R_T) - L
    expected = -2*pi/WL*(np.sqrt(r**2 + f**2) - f)
    phase = d.phase - d.phase[np.argmin(r)]
    np.testing.assert_allclose(phase, -expected + (expected - phase)[np.argmin(r)], atol=1e-6)


def test_null_phase_needs_a_phase_surface():
    with pytest.raises(TypeError):
        cgh.null_phase(_system([Surface(np.inf, L, semi_diameter=6.0), Surface(R_T, mirror=True,
                                                                           semi_diameter=11.0)]), 0, 1)


def test_focus_phase_focuses():
    blank = ZernikePhaseSurface(carrier=(0.0, 10.0), semi_diameter=6.0)
    point = np.array([0.5, 2.0, 40.0])
    sysm = _system([blank, Surface(np.inf, 0.0)])
    z = cgh.focus_phase(sysm, 0, point, 2.0, 5.0, N=41, terms=45)
    assert z.fit_residual < 1e-5
    surfaces = [z.surface, Surface(np.inf, 0.0)]
    surfaces[0].thickness = point[2]
    s2 = System(surfaces, 0, WL, ray_aiming=False, image_decenter=(0.0, 0.0))
    px, py = cgh._grid_disk(21)
    b = s2.trace(s2.rays(0.0, px, py))
    r = np.hypot(px, py)*5.0
    sel = b.valid & (r > 2.2) & (r < 4.8)
    hits = b.points[-1, sel]
    np.testing.assert_allclose(hits[:, 0], point[0], atol=1e-6)
    np.testing.assert_allclose(hits[:, 1], point[1], atol=1e-6)


def test_retro_phase_retroreflects():
    blank = ZernikePhaseSurface(mirror=True, order=3, semi_diameter=6.0)
    sysm = System([blank], 0, WL, ray_aiming=False, field_angle=0.0)
    z = cgh.retro_phase(sysm, 0, 1.0, 5.0, N=31, terms=10)
    assert z.fit_residual < 1e-6
    with pytest.raises(ValueError):
        cgh.retro_phase(System([ZernikePhaseSurface(semi_diameter=6.0)], 0, WL), 0, 1.0, 5.0)
    # a plane wave at normal incidence needs no phase beyond a constant
    gx, gy = z.surface.phase_gradient(z.x, z.y)
    assert np.max(np.hypot(gx, gy)) < 1e-6


def test_binary_mask_duty():
    x = np.linspace(0, 10, 100001)
    m = cgh.binary_mask(2*pi*x, duty=0.3)
    assert abs(m.mean() - 0.3) < 1e-3
    assert m[0]  # centred on phase 0


# orders

def test_binary_orders_scalar():
    t = cgh.binary_orders(range(-41, 42), 0.5, pi)
    eta = np.abs(t)**2
    m = np.arange(-41, 42)
    np.testing.assert_allclose(eta[m == 1], 4/pi**2)
    assert eta[m == 0] < 1e-30 and eta[m == 2] < 1e-30
    assert abs(eta.sum() - 1) < 0.01  # Parseval (the tail beyond 41 is small)
    # phase of the first orders: depth/2 + pi/2
    t1 = cgh.binary_orders(1, 0.4, 0.8*pi)
    np.testing.assert_allclose(np.angle(t1), 0.4*pi + pi/2)
    assert np.isscalar(t1) or np.ndim(t1) == 0


def test_binary_orders_rcwa_against_scalar():
    n = 1.457
    h = WL/(2*(n - 1))
    t = cgh.binary_orders_rcwa(40*WL, 0.5, h, WL, n, orders=(-1, 0, 1))
    s = cgh.binary_orders([-1, 0, 1], 0.5, pi)
    # wide lines: the scalar model holds, up to the Fresnel losses
    np.testing.assert_allclose(np.abs(t[[0, 2]])**2, np.abs(s[[0, 2]])**2, rtol=0.06)
    assert np.abs(t[1])**2 < 0.01
    assert abs(np.angle(t[2]*np.conj(s[2]))) < 0.05
    # the phase of +1 follows half the depth
    t2 = cgh.binary_orders_rcwa(40*WL, 0.5, 1.1*h, WL, n, orders=(1,))
    dphi = np.angle(t2[0]*np.conj(t[2]))
    assert abs(dphi - 0.05*pi) < 0.02


def test_order_paths_and_stop(design):
    T = _test_arm(design.surface)
    paths = cgh.order_paths(T, (0, 2), orders=(0, 1, 2), N=21, max_slope=1e-3)
    by = {p.orders: p for p in paths}
    assert len(paths) == 9
    null = by[(1, 1)]
    assert null.fraction > 0.5
    assert np.nanmax(np.abs(null.slopes)) < 1e-6
    for orders in ((0, 1), (1, 0), (1, 2), (2, 1), (0, 0), (2, 2)):
        assert by[orders].fraction == 0.0
    # the stop as a surface removes the same rays
    stop = cgh.InterferometerStop(1e-3, 0.0)
    d = np.array([[0.0, 0.0005, 1.0], [0.0, 0.002, 1.0]])
    d /= np.linalg.norm(d, axis=1)[:, None]
    d2, dopl = stop.deflect(np.zeros((2, 3)), d, 1.0, 1.0)
    assert np.all(np.isfinite(d2[0])) and np.all(np.isnan(d2[1]))
    np.testing.assert_allclose(dopl, 0.0)


def test_distortion_phase():
    s = ZernikePhaseSurface(carrier=(0.0, 50.0))
    x = np.linspace(-1, 1, 5)
    dphi = cgh.distortion_phase(s, x, 0*x, (0.0, 1e-4), passes=2)
    np.testing.assert_allclose(dphi, -2*2*pi*50.0*1e-4)
    # along the lines: no error
    np.testing.assert_allclose(cgh.distortion_phase(s, x, 0*x, (1e-4, 0.0)), 0.0)
    np.testing.assert_allclose(cgh.distortion_phase(s, x, 0*x, lambda x, y: (0*x, 0*x + 1e-4)),
                               -2*pi*50.0*1e-4)


# reconstruction

def test_zernike_fit_recovers_coefficients():
    rng = np.random.default_rng(1)
    r, t = 3.0*np.sqrt(rng.uniform(0, 1, 2000)), rng.uniform(0, 2*pi, 2000)
    x, y = r*np.cos(t) + 0.5, r*np.sin(t)
    true = {4: 1.0, 5: -0.5, 9: 0.25, 16: 0.1}
    M, js = cgh.zernike_basis(x, y, 3.0, 16, "fringe", centre=(0.5, 0.0))
    v = M @ np.array([true.get(j, 0.0) for j in js])
    c, res, rn = cgh.zernike_fit(x, y, v, norm_radius=3.0, terms=16, centre=(0.5, 0.0))
    for j in js:
        assert abs(c[j] - true.get(j, 0.0)) < 1e-10
    assert np.nanmax(np.abs(res)) < 1e-10
    _, _, rn = cgh.zernike_fit(x, y, v, terms=4, centre=(0.5, 0.0))
    assert rn <= 3.0


def test_remove_alignment():
    rng = np.random.default_rng(2)
    S = [rng.normal(size=(20, 20)) for _ in range(3)]
    h = 0.01*rng.normal(size=(20, 20))
    phase = 2.0 + 0.3*S[0] - 1.2*S[1] + 0.7*S[2]
    p, res = cgh.remove_alignment(phase, S)
    np.testing.assert_allclose(p, [0.3, -1.2, 0.7], atol=1e-12)
    np.testing.assert_allclose(res, 0.0, atol=1e-12)
    mask = np.ones((20, 20), bool)
    mask[:3] = False
    p, res = cgh.remove_alignment(phase + h, S, mask=mask)
    assert np.all(np.isnan(res[:3]))
    np.testing.assert_allclose(p, [0.3, -1.2, 0.7], atol=0.01)


def test_surface_map_recovers_a_form_error(design):
    x = np.linspace(-5.5, 5.5, 121)
    ref = System(double_pass([Surface(np.inf, 0.0, semi_diameter=6.0),
                              Surface(np.inf, mirror=True, semi_diameter=6.0)], image_distance=-5.0),
                 0, WL, ray_aiming=False)
    m_ref = ref.ray_map(x, x, field=0.0, N=101)
    sfd = {5: 20e-6, 9: -10e-6}  # sag, mm
    T = _test_arm(design.surface, test=sfd)
    m = T.ray_map(x, x, field=0.0, N=101, surface=1)
    phase = 2*pi*(m.opl - m_ref.opl)/WL
    ok = m.mask & m_ref.mask
    _, res = cgh.remove_alignment(phase, [], mask=ok)
    sm = cgh.surface_map(res, m, T, 1)
    c, _, _ = cgh.zernike_fit(sm.u, sm.v, sm.sag, norm_radius=11.0, terms=16, mask=sm.mask)
    assert abs(c[5] - sfd[5]) < 0.02*sfd[5]
    assert abs(c[9] - sfd[9]) < 0.05*abs(sfd[9])
    # height along the normal is smaller than the sag error only by the cosine
    assert np.nanmax(np.abs(sm.height)) <= np.nanmax(np.abs(sm.sag)) + 1e-15


def test_remove_alignment_with_a_large_piston_in_the_sensitivities():
    """Finite differences of phases with a large common path leave a huge
    constant in the sensitivities; with piston fitted it must not matter."""
    rng = np.random.default_rng(4)
    S = [rng.normal(size=(30, 30)) for _ in range(3)]
    phase = 0.3*S[0] - 1.2*S[1] + 0.7*S[2] + 0.01*rng.normal(size=(30, 30))
    S_big = [s + 1e11*(k + 1) for k, s in enumerate(S)]
    p, res = cgh.remove_alignment(phase, S_big)
    np.testing.assert_allclose(p, [0.3, -1.2, 0.7], atol=0.01)
    assert np.nanstd(res) < 0.02
