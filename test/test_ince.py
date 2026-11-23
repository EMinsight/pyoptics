#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Test the Ince polynomials and Ince-Gaussian modes: the Ince equation,
normalization, orthogonality, the LG and HG limits, propagation and the ABCD
law.
"""

import numpy as np
import pytest

from pyoptics import abcd, ince, modes
from pyoptics.propagators import fresnel_TF
from pyoptics.utils import grid2d
from pyoptics.vortices import topological_charge


X_GRID = grid2d(-6, 6, -6, 6, 256)


def _overlap(u, v, dA):
    return np.sum(u*np.conj(v))*dA


@pytest.mark.parametrize("p, m, parity", [(4, 2, "even"), (5, 3, "odd"), (6, 0, "even"), (3, 1, "odd")])
@pytest.mark.parametrize("eps", [0.5, 3.0])
def test_ince_equation(p, m, eps, parity):
    s = ince.ince_series(p, m, eps, parity)
    eta = np.linspace(0, 2*np.pi, 2001)
    h = eta[1] - eta[0]
    C = ince.ince_polynomial(p, m, eps, eta, parity, s)
    dC = np.gradient(C, h)
    d2C = np.gradient(dC, h)
    residual = d2C + eps*np.sin(2*eta)*dC + (s.a - p*eps*np.cos(2*eta))*C
    assert np.max(np.abs(residual[5:-5])) < 1E-3*np.max(np.abs(C))*max(p, 1)**2


def test_eigenvalues_tend_to_m_squared():
    for p, m, parity in [(6, 4, "even"), (7, 3, "odd")]:
        assert ince.ince_series(p, m, 1E-6, parity).a == pytest.approx(m**2, abs=1E-4)


@pytest.mark.parametrize("eps", [0.1, 2.0, 50.0])
def test_orthonormal(eps):
    x, y, _, _ = X_GRID
    dA = (x[1] - x[0])**2
    fields = [ince.ig_mode(p, m, eps, x, y, 1.0, 1.0, parity=par)
              for par in ("even", "odd") for p, m in ince.ince_indices(5, par)]
    G = np.array([[_overlap(u, v, dA) for v in fields] for u in fields])
    assert np.allclose(G, np.eye(len(fields)), atol=1E-6)


def test_mode_count_matches_hg():
    for N in range(8):
        assert ince.mode_count(N) == len(modes.hg_indices(N))


@pytest.mark.parametrize("p, m, parity", [(4, 2, "even"), (5, 3, "odd"), (6, 6, "even")])
def test_limits(p, m, parity):
    x, y, _, _ = X_GRID
    dA = (x[1] - x[0])**2
    u = ince.ig_mode(p, m, 1E-4, x, y, 1.0, 1.0, parity=parity)
    lg_p, lg_m = modes.lg_mode((p - m)//2, m, x, y, 1, 1), modes.lg_mode((p - m)//2, -m, x, y, 1, 1)
    ref = lg_p + lg_m if parity == "even" else lg_p - lg_m
    ref /= np.sqrt(np.sum(np.abs(ref)**2)*dA)
    assert abs(_overlap(u, ref, dA)) == pytest.approx(1.0, abs=1E-4)

    u = ince.ig_mode(p, m, 1E4, x, y, 1.0, 1.0, parity=parity)
    l = m if parity == "even" else m - 1
    ref = modes.hg_mode(l, p - l, x, y, 1.0, 1.0)
    assert abs(_overlap(u, ref, dA)) == pytest.approx(1.0, abs=1E-3)


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_coefficients_synthesize_mode(parity):
    p, m, eps = 6, 2, 2.5
    x, y, _, _ = X_GRID
    z = 0.7*np.pi  # 0.7 z_R
    u = ince.ig_mode(p, m, eps, x, y, 1.0, 1.0, z=z, parity=parity)
    hg = sum(c*modes.hg_mode(l, n, x, y, 1.0, 1.0, z=z) for (l, n), c in ince.hg_coefficients(p, m, eps, parity).items())
    lg = sum(c*modes.lg_mode(q, l, x, y, 1.0, 1.0, z=z) for (q, l), c in ince.lg_coefficients(p, m, eps, parity).items())
    scale = np.max(np.abs(u))
    assert np.max(np.abs(hg - u)) < 1E-10*scale
    assert np.max(np.abs(lg - u)) < 1E-10*scale


def test_free_space_propagation():
    p, m, eps, wl, w0 = 5, 3, 2.0, 0.5, 20.0
    x, y, _, _ = grid2d(-150, 150, -150, 150, 256)
    z = 1.3*np.pi*w0**2/wl
    u0 = ince.ig_mode(p, m, eps, x, y, w0, wl, parity="odd")
    u, _, _ = fresnel_TF(u0, x, y, z, wl)
    expected = ince.ig_mode(p, m, eps, x, y, w0, wl, z=z, parity="odd")
    # fresnel_TF keeps exp(i*k*z) like the analytic mode:
    assert np.max(np.abs(u - expected)) < 1E-5*np.max(np.abs(expected))


def test_abcd_law():
    """Through a lens and free space, the mode is the IG mode of the
    transformed q, with the same eps and the Gouy phase (p + 1)*psi."""
    p, m, eps, wl, w0 = 4, 2, 3.0, 0.5, 20.0
    x, y, _, _ = grid2d(-200, 200, -200, 200, 256)
    M = abcd.compose(abcd.thin_lens(2000.0), abcd.free_space(1500.0))
    u0 = ince.ig_mode(p, m, eps, x, y, w0, wl)
    u = abcd.collins(u0, x, y, M, wl)
    q_out = abcd.transform_q(M, abcd.q_from_waist(w0, 0.0, wl))
    w0_out, z_waist = abcd.waist(q_out, wl)
    expected = ince.ig_mode(p, m, eps, x, y, w0_out, wl, z=-z_waist)
    expected *= np.exp(-1j*2*np.pi/wl*(-z_waist))  # collins() drops exp(i*k*z)
    psi_sys = abcd.gouy_phase(M, abcd.q_from_waist(w0, 0.0, wl))
    psi_mode = np.arctan2(-z_waist, np.imag(q_out))
    expected *= np.exp(-1j*(p + 1)*(psi_sys - psi_mode))
    assert np.max(np.abs(u - expected)) < 1E-6*np.max(np.abs(expected))


def test_helical_charges():
    p, m, eps = 6, 4, 1.0
    x, y, _, _ = X_GRID
    u = ince.helical_ig_mode(p, m, eps, x, y, 1.0, 1.0)
    assert topological_charge(u, x, y, radius=1.0) == pytest.approx(m, abs=1E-6)
    u = ince.helical_ig_mode(p, m, eps, x, y, 1.0, 1.0, sign=-1)
    assert topological_charge(u, x, y, radius=1.0) == pytest.approx(-m, abs=1E-6)


def test_elliptic_coordinates_roundtrip():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(2, 200))
    f = 0.7
    xi, eta = ince.elliptic_coordinates(x, y, f)
    assert np.all(xi >= 0)
    assert np.allclose(f*np.cosh(xi)*np.cos(eta), x)
    assert np.allclose(f*np.sinh(xi)*np.sin(eta), y)


def test_bad_indices():
    with pytest.raises(ValueError):
        ince.ince_series(4, 1, 1.0)
    with pytest.raises(ValueError):
        ince.ince_series(4, 0, 1.0, "odd")
