"""Tests of pyoptics.prisms: image orientation, tunnel lengths, TIR phases
and the roof phase error (notebook 172, binoculars and erecting prisms)."""

import numpy as np
import pytest

from pyoptics import prisms as pr
from pyoptics.coatings import Coating

RAY = ((0.01, 0.01, -5.0), (0.0, 0.0, 1.0))


@pytest.fixture(scope="module")
def sp():
    return pr.schmidt_pechan(1.0)


def test_porro_prism_is_a_mirror_image():
    O, events = pr.orientation([pr.porro_prism(1.0)], ray=((0.01, -0.55, -5.0), (0.0, 0.0, 1.0)))
    assert np.linalg.det(O) == pytest.approx(1.0)       # two reflections
    assert np.allclose(O, np.diag([1.0, -1.0, -1.0]), atol=1e-12)
    assert sum(e[2] == "TIR" for e in events) == 2


@pytest.mark.parametrize("make", [lambda: pr.porro_system(1.0), lambda: pr.schmidt_pechan(1.0)])
def test_erecting_systems_turn_by_180_degrees(make):
    prisms = make()
    O, _ = pr.orientation(prisms, ray=RAY)
    assert np.allclose(O, np.diag([-1.0, -1.0, 1.0]), atol=1e-9)
    tr = pr.trace(prisms, [RAY[0]], [RAY[1]])
    assert tr.valid[0] and np.allclose(tr.directions[0], [0, 0, 1], atol=1e-9)


def test_schmidt_pechan_is_coaxial(sp):
    # (off the roof ridge at x = 0)
    tr = pr.trace(sp, [[0.05, 0.2, -1.0]], [[0.0, 0.0, 1.0]])
    # 180 degree rotation about the z axis: (x, y) -> (-x, -y)
    assert tr.points[0][:2] == pytest.approx([-0.05, -0.2], abs=1e-9)


def test_pechan_without_roof_mirrors_the_image():
    O, events = pr.orientation(pr.schmidt_pechan(1.0, roof=False), ray=RAY)
    assert np.linalg.det(O) == pytest.approx(-1.0)
    assert sum(e[2] in ("TIR", "mirror") for e in events) == 5


def test_glass_paths():
    assert pr.glass_path([pr.porro_prism(1.0)], ray=((0.01, -0.55, -5.0), (0, 0, 1.0))) \
        == pytest.approx(2.2, rel=1e-6)
    assert pr.glass_path(pr.porro_system(1.0), ray=RAY) == pytest.approx(4.4, rel=1e-6)


def test_tunnel_equals_3d_glass_path(sp):
    sections = [p.section for p in sp]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    L = pr.draw_tunnel(ax, sp, (-1.0, 0.0), beam=1.0)
    plt.close(fig)
    assert L == pytest.approx(pr.glass_path(sp), rel=1e-3)
    assert len(sections) == 2


def test_tir_retardance_formula():
    # textbook: tan(delta/2) = cos(theta) sqrt(sin^2 theta - 1/n^2) / sin^2
    # theta; the field ratio convention of thin_films flips the sign of r_p,
    # which adds pi
    n, theta = 1.5688, np.radians(49.2)
    s2 = np.sin(theta)**2
    expected = 2*np.arctan(np.cos(theta)*np.sqrt(s2 - 1/n**2)/s2)
    assert abs(pr.tir_retardance(n, theta)[0]) == pytest.approx(np.pi - expected, rel=1e-6)


def test_roof_phase_error_and_coating(sp):
    x, y, J = pr.jones_pupil(sp, 1.0, n=24)
    _, retard = pr.roof_error(J, x)
    assert np.degrees(abs(retard)) > 100
    assert pr.polarized_psf(J, pad=4).max() < 0.5

    xp, yp, Jp = pr.jones_pupil(pr.porro_system(1.0), 1.0, n=24, z_start=-5.0)
    assert abs(pr.roof_error(Jp, xp)[1]) < 1e-6
    assert pr.polarized_psf(Jp, pad=4).max() == pytest.approx(1.0, abs=0.01)

    coating, _ = pr.phase_coating()
    xc, yc, Jc = pr.jones_pupil(pr.schmidt_pechan(1.0, roof_coating=coating), 1.0, n=24)
    assert pr.polarized_psf(Jc, pad=4).max() > 0.95


def test_mirror_coating_costs_light():
    al = Coating((), substrate=0.96 + 6.69j)
    T = []
    for coat in (None, al):
        tr = pr.trace(pr.schmidt_pechan(1.0, mirror_coating=coat), [RAY[0]], [RAY[1]])
        P = tr.P[0]
        T.append(0.5*(np.sum(np.abs(P[:, 0])**2) + np.sum(np.abs(P[:, 1])**2)))
    assert T[1] < T[0] - 0.05
