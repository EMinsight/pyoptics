import numpy as np
import pytest

from pyoptics import fieldtrace as ft
from pyoptics import resonators as rs
from pyoptics import abcd


WL = 1e-3
L = 100.0


def quadrature_round_trip(a, R1, R2, n=300):
    """Round-trip eigenvalues of a strip resonator from the Fresnel kernel
    on Gauss-Legendre nodes (independent of fieldtrace)."""
    k = 2*np.pi/WL
    xg, wg = np.polynomial.legendre.leggauss(n)
    xg, wg = xg*a, wg*a
    K = np.sqrt(1/(1j*WL*L))*np.exp(1j*k*(xg[:, None] - xg[None, :])**2/(2*L))
    s = np.sqrt(wg)
    T = s[:, None]*K*s[None, :]
    m1 = np.exp(-1j*k*xg**2/R1) if np.isfinite(R1) else np.ones(n)
    m2 = np.exp(-1j*k*xg**2/R2) if np.isfinite(R2) else np.ones(n)
    v = np.linalg.eigvals(np.diag(m1) @ T @ np.diag(m2) @ T)
    return v[np.argsort(-np.abs(v))]


def test_plane_strip_matches_quadrature():
    a = np.sqrt(6.25*WL*L)
    x = np.linspace(-1.6*a, 1.6*a, 512)
    modes = rs.eigenmodes(rs.round_trip(L, a1=a, a2=a), x, None, WL, n_modes=3)
    ref = quadrature_round_trip(a, np.inf, np.inf)[:3]
    np.testing.assert_allclose([m.loss for m in modes], 1 - np.abs(ref)**2, rtol=3e-3)
    np.testing.assert_allclose([m.phase for m in modes], np.angle(ref), atol=3e-3)


def test_confocal_strip_matches_prolate_eigenvalues():
    a = np.sqrt(WL*L)                                   # N = 1
    x = np.linspace(-1.6*a, 1.6*a, 512)
    modes = rs.eigenmodes(rs.round_trip(L, L, L, a, a), x, None, WL, n_modes=3)
    sigma = rs.confocal_strip_eigenvalues(1.0, 3)
    np.testing.assert_allclose([m.loss for m in modes], 1 - np.abs(sigma)**4, rtol=2e-2)
    # Gouy phase of the confocal strip modes: -(l + 1/2) pi per round trip
    for l, m in enumerate(modes):
        assert np.cos(m.phase - rs.gouy_round_trip(L, L, L, l, None)) == pytest.approx(1, abs=1e-5)


def test_confocal_reference_against_quadrature():
    sigma = rs.confocal_strip_eigenvalues(1.0, 4)
    ref = quadrature_round_trip(np.sqrt(WL*L), L, L, n=200)[:4]
    np.testing.assert_allclose(np.abs(sigma)**4, np.abs(ref)**2, rtol=1e-8)


def test_fox_li_converges_to_lowest_loss_mode():
    a = np.sqrt(6.25*WL*L)
    x = np.linspace(-1.6*a, 1.6*a, 512)
    ch = rs.round_trip(L, a1=a, a2=a)
    mode, losses = rs.fox_li(ch, x, None, WL, n_iter=600, tol=1e-9)
    ref = rs.eigenmodes(ch, x, None, WL, n_modes=1)[0]
    assert mode.loss == pytest.approx(ref.loss, rel=1e-4)
    assert losses[-1] == pytest.approx(ref.loss, rel=1e-4)
    assert mode.field.overlap(ref.field) == pytest.approx(1.0, abs=1e-6)


def test_stable_resonator_gaussian_mode_and_gouy():
    R = 200.0
    w1, w2, w0, z1 = rs.gaussian_mode(L, R, R, WL)
    a = 2.3*w1                     # losses well above the numerical floor: fast Arnoldi
    x = np.linspace(-1.3*a, 1.3*a, 64)
    modes = rs.eigenmodes(rs.round_trip(L, R, R, a, a), x, x, WL, n_modes=3)
    f = modes[0].field
    # the deflated second Arnoldi run finds the partner of the degenerate TEM01/TEM10 pair
    assert modes[1].loss == pytest.approx(modes[2].loss, rel=1e-3)
    u, v = (m.field.scalar_amplitude.ravel() for m in modes[1:3])
    assert abs(np.vdot(u, v))/(np.linalg.norm(u)*np.linalg.norm(v)) < 1e-6
    k = 2*np.pi/WL
    ref = ft.Field(np.exp(-(x[None, :]**2 + x[:, None]**2)/w1**2), x, x, WL,
                   alpha=(-k/(2*R), -k/(2*R)))                  # converging behind mirror 1
    assert f.overlap(ref) == pytest.approx(1.0, abs=2e-3)
    assert modes[0].phase == pytest.approx(rs.gouy_round_trip(L, R, R, 0, 0), abs=1e-3)
    assert modes[1].phase == pytest.approx(rs.gouy_round_trip(L, R, R, 1, 0), abs=1e-3)
    # abcd agrees on the mode radius
    M = abcd.compose(abcd.free_space(L), abcd.mirror(R), abcd.free_space(L), abcd.mirror(R))
    q = abcd.eigenmode(M)
    assert abcd.beam_radius(q, WL) == pytest.approx(w1, rel=1e-6)


def test_gaussian_mode_formulas():
    # half-symmetric resonator: waist on the plane mirror
    w1, w2, w0, z1 = rs.gaussian_mode(L, np.inf, 300.0, WL)
    assert z1 == pytest.approx(0.0, abs=1e-12)
    assert w0 == pytest.approx(w1)
    with pytest.raises(ValueError):
        rs.gaussian_mode(L, 40.0, 40.0, WL)


def test_unstable_resonator_losses_near_geometric():
    M = 2.0
    R1, R2 = -2*L/(M - 1), 2*L*M/(M - 1)        # small convex mirror first
    assert rs.unstable_magnification(L, R1, R2) == pytest.approx(M)
    a = 0.8
    x = np.linspace(-6*a, 6*a, 1024)
    mode = rs.eigenmodes(rs.round_trip(L, R1, R2, a, None), x, None, WL, n_modes=1)[0]
    assert abs(mode.loss - rs.geometric_loss(M, 1)) < 0.1
    assert rs.geometric_loss(M, 2) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        rs.unstable_magnification(L, 200.0, 200.0)


def test_intracavity_element_and_frequency_offsets():
    R = 200.0
    a = 0.45                                       # 2.5 mode radii: TEM0 loses least
    x = np.linspace(-0.6, 0.6, 256)
    base = rs.eigenmodes(rs.round_trip(L, R, R, a, a), x, None, WL, n_modes=2)
    lensed = rs.eigenmodes(rs.round_trip(L, R, R, a, a, elements=[ft.ThinLens(500.0)]), x,
                           None, WL, n_modes=2)
    # a positive thermal lens shrinks the mode on the mirrors
    def width(m):
        I = np.abs(m.field.scalar_amplitude)**2
        return np.sqrt(np.sum(I*x**2)/np.sum(I))
    assert width(lensed[0]) < width(base[0])
    off = rs.frequency_offsets(base, L)
    assert off[0] == pytest.approx(-rs.gouy_round_trip(L, R, R, 0, None)/(2*np.pi), abs=1e-4)
    assert -0.5 <= off.min() and off.max() < 0.5


def test_grid_propagate_matches_fieldtrace_for_smooth_field():
    x = np.linspace(-3, 3, 512)
    f = ft.gaussian(0.4, WL, x)
    a = ft.Chain([rs.GridPropagate(200.0)]).run(f).field
    b = ft.Chain([ft.Propagate(200.0, method="spw")]).run(f).field
    assert a.power() == pytest.approx(1.0*f.power(), rel=1e-9)
    assert a.overlap(b) == pytest.approx(1.0, abs=1e-7)
    assert a.gamma == 0 and np.angle(a.E()[0][256]) == pytest.approx(
        np.angle(b.E()[0][np.argmin(np.abs(b.x))]), abs=1e-3)


def test_fresnel_number():
    assert rs.fresnel_number(1.0, 100.0, 1e-3) == pytest.approx(10.0)
    assert rs.equivalent_fresnel_number(1.0, 100.0, 3.0, 1e-3) == pytest.approx(10.0)
