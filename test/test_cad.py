"""Tests of the CAD (STEP) export in pyoptics.cad (need build123d)."""

import numpy as np
import pytest

build123d = pytest.importorskip("build123d")

from pyoptics import cad
from pyoptics import raytrace as rt


def _sag(R, r):
    return R - np.sign(R)*np.sqrt(R**2 - r**2)


def doublet():
    surfaces = [rt.Surface(np.inf, 5.0, semi_diameter=11.0),
                rt.Surface(62.8, 4.0, n_after=1.5168, semi_diameter=11.5),
                rt.Surface(-45.7, 2.5, n_after=1.6727, semi_diameter=11.5),
                rt.Surface(-128.2, 95.0, n_after=1.0, semi_diameter=11.5)]
    return rt.System(surfaces, 0, 550e-6, field_angle=np.radians(2.0)).at_paraxial_focus()


def test_doublet_volumes_lands_and_bevels(tmp_path):
    rc, rm, b = 11.5, 12.7, 0.3
    model = cad.system_model(doublet(), n_rays=5, mechanics=cad.Mechanics(margin=1.2, bevel=b))
    assert [s.label for s in model.lenses] == ["L1a", "L1b"]
    # front slab: lens volume inside the clear aperture, a flat land out to
    # the rim and a 45 degree bevel on the air side only:
    r = np.linspace(0, rc, 20001)
    thick = 4.0 + _sag(-45.7, r) - _sag(62.8, r)
    V = np.trapezoid(2*np.pi*r*thick, r) + np.pi*(rm**2 - rc**2)*thick[-1] \
        - 2*np.pi*(rm - b/3)*b**2/2
    assert model.lenses[0].volume == pytest.approx(V, rel=1e-6)
    # the optical faces are exact spheres, bevels cones, the rim a cylinder:
    kinds = {f.geom_type.name for f in model.lenses[0].faces()}
    assert kinds == {"SPHERE", "PLANE", "CONE", "CYLINDER"}
    assert all(s.is_valid for s in model.lenses)
    # the slabs touch along the cemented face and do not overlap:
    common = model.lenses[0] & model.lenses[1]
    assert common is None or common.volume == pytest.approx(0.0, abs=1e-6)
    assert model.lenses[0].distance_to(model.lenses[1]) < 1e-9
    assert len(model.rays) == 2*5
    path = tmp_path/"doublet.step"
    cad.export_step(model, path)
    back = build123d.import_step(str(path))
    assert len(back.solids()) == 2


def test_rays_end_on_the_traced_image_points():
    system = doublet()
    model = cad.system_model(system, fields=[0.0, 0.03], n_rays=3, image_plane=False)
    end = np.array(tuple(model.rays[-1].vertices().sort_by(build123d.Axis.Z)[-1]))
    b = system.trace_pupil(0.03, 0.0, 1.0)
    assert np.allclose(end, b.points[-1, 0], atol=1e-9)


def test_asphere_profile_follows_the_sag():
    s1 = rt.Surface(30.0, 6.0, n_after=1.5, conic=-0.8, aspheric=(-2e-6, 1e-9), semi_diameter=10)
    s2 = rt.Surface(-80.0, 50.0, n_after=1.0, semi_diameter=10)
    system = rt.System([s1, s2], 0, 550e-6)
    model = cad.system_model(system, rays=None, image_plane=False,
                             mechanics=cad.Mechanics(margin=0.0), tolerance=1e-5)
    lens = model.lenses[0]
    for r in (2.0, 5.0, 9.5):
        p = build123d.Vector(r, 0.0, float(s1.sag(np.array([r**2]))[0]))
        assert lens.distance_to(p) < 2e-5


def test_freeform_plate_and_tilted_window():
    zern = rt.ZernikeSurface(np.inf, 3.0, zernike={5: 0.05, 10: 0.03}, norm_radius=8.0,
                             n_after=1.5, semi_diameter=8.0)
    surfaces = [rt.Surface(np.inf, 2.0, semi_diameter=8.0), zern,
                rt.Surface(np.inf, 10.0, n_after=1.0, semi_diameter=8.0),
                rt.Surface(np.inf, 2.0, n_after=1.5, semi_diameter=7.0, tilt=(0.1, 0, 0)),
                rt.Surface(np.inf, 20.0, n_after=1.0, semi_diameter=7.0, tilt=(0.1, 0, 0))]
    system = rt.System(surfaces, 0, 550e-6)
    model = cad.system_model(system, rays=None,
                             mechanics={0: cad.Mechanics(margin=1.0, land="extend"),
                                        1: cad.Mechanics(margin=1.0)})
    plate, window = model.lenses
    # astigmatism and trefoil average out over the disk:
    assert plate.volume == pytest.approx(np.pi*9**2*3.0, rel=1e-4)
    # the window is a cylinder along its own (tilted) axis:
    assert window.volume == pytest.approx(np.pi*8**2*2.0*np.cos(0.1), rel=1e-6)


def test_mirror_substrate_and_too_thin_edge():
    system = rt.System([rt.Surface(np.inf, 10.0, semi_diameter=5.0),
                        rt.Surface(-100.0, -50.0, mirror=True, semi_diameter=6.0)], 0, 550e-6)
    model = cad.system_model(system, rays=None,
                             mechanics=cad.Mechanics(margin=1.0, mirror_thickness=3.0))
    (m,) = model.lenses
    assert m.label == "M1"
    # flat back 3 mm behind the rim; the concave face is thinner in the middle:
    r = np.linspace(0, 6.0, 20001)
    dish = np.trapezoid(2*np.pi*r*(_sag(-100.0, r) - _sag(-100.0, 6.0)), r)
    assert m.volume == pytest.approx(np.pi*7**2*3.0 - dish, rel=1e-6)
    thin = cad.Mechanics(semi_diameter=12.0, margin=0.0)
    lens = rt.System([rt.Surface(20.0, 2.0, n_after=1.5, semi_diameter=11.0),
                      rt.Surface(-20.0, 30.0, n_after=1.0, semi_diameter=11.0)], 0, 550e-6)
    with pytest.raises(ValueError, match="Edge thickness"):
        cad.system_model(lens, rays=None, mechanics=thin)
