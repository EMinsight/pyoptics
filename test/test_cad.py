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


def test_flat_land_and_bevels_on_a_freeform_plate():
    zern = rt.ZernikeSurface(np.inf, 3.0, zernike={5: 0.05, 10: 0.03}, norm_radius=8.0,
                             n_after=1.5, semi_diameter=8.0)
    system = rt.System([rt.Surface(np.inf, 2.0, semi_diameter=8.0), zern,
                        rt.Surface(np.inf, 10.0, n_after=1.0, semi_diameter=8.0)], 0, 550e-6)
    rm, b = 9.5, 0.3
    model = cad.system_model(system, rays=None, mechanics=cad.Mechanics(margin=1.5, bevel=b))
    (plate,) = model.lenses
    assert plate.is_valid
    # the land lies at the rim point of the freeform farthest towards air:
    phi = np.linspace(0, 2*np.pi, 181)
    z_land = np.min(zern.sag_xy(8*np.cos(phi), 8*np.sin(phi)))
    r = np.linspace(0, 8, 801)
    p = np.linspace(0, 2*np.pi, 801, endpoint=False)
    Rg, Pg = np.meshgrid(r, p)
    inner = np.trapezoid(np.mean(3 - zern.sag_xy(Rg*np.cos(Pg), Rg*np.sin(Pg)), axis=0)*2*np.pi*r,
                         r)
    V = inner + np.pi*(rm**2 - 64)*(3 - z_land) - 2*2*np.pi*(rm - b/3)*b**2/2
    assert plate.volume == pytest.approx(V, rel=1e-6)
    assert "BSPLINE" in {f.geom_type.name for f in plate.faces()}


def test_fresnel_lens_follows_the_facets():
    from pyoptics.fresnel import FresnelSurface

    fs = FresnelSurface(-30.0, n_zones=10, radius=10.0, n_after=1.0, semi_diameter=10.0,
                        thickness=60.0)
    system = rt.System([rt.Surface(np.inf, 2.0, n_after=1.5, semi_diameter=10.0), fs], 0, 550e-6)
    model = cad.system_model(system, rays=None, mechanics=cad.Mechanics(margin=1.0, bevel=0.2))
    r = np.linspace(0, 10, 400001)
    z = fs.sag(r**2)
    V = np.trapezoid(2*np.pi*r*(2 + z), r) + np.pi*(121 - 100)*(2 + z[-1]) \
        - 2*2*np.pi*(11 - 0.2/3)*0.2**2/2
    assert model.lenses[0].volume == pytest.approx(V, rel=1e-5)


def test_freeform_fold_mirror():
    surfaces = [rt.Surface(np.inf, 20.0, semi_diameter=5.0), rt.CoordinateBreak(tilt=(np.pi/4, 0, 0)),
                rt.ZernikeSurface(np.inf, 0.0, zernike={5: 0.02}, norm_radius=8.0, mirror=True,
                                  semi_diameter=8.0),
                rt.CoordinateBreak(tilt=(np.pi/4, 0, 0), thickness=-30.0)]
    system = rt.System(surfaces, 0, 550e-6)
    model = cad.system_model(system, rays=None,
                             mechanics=cad.Mechanics(margin=2.0, mirror_thickness=4.0, bevel=0.5))
    (m,) = model.lenses
    assert m.label == "M1" and m.is_valid
    # about a 4 mm thick disk of radius 10 with one bevelled edge:
    assert m.volume == pytest.approx(np.pi*100*4.0, rel=0.02)
    # the substrate lies behind the mirror (seen from the incoming light, +z):
    frame = system.frames()[0][2]
    centre = (np.array(tuple(m.center())) - frame[0]) @ frame[1]
    assert centre[2] > 1.0


def test_mount_parts_touch_but_do_not_overlap():
    S = [rt.Surface(21.48, 2.0, n_after=1.6204, semi_diameter=6.5),
         rt.Surface(-124.1, 5.26, n_after=1.0, semi_diameter=6.5),
         rt.Surface(-19.1, 1.25, n_after=1.6034, semi_diameter=4.6),
         rt.Surface(22.0, 1.5, n_after=1.0, semi_diameter=4.6),
         rt.Surface(np.inf, 4.0, semi_diameter=4.2),
         rt.Surface(328.9, 2.25, n_after=1.6204, semi_diameter=5.4),
         rt.Surface(-16.7, 43.0, n_after=1.0, semi_diameter=5.4)]
    system = rt.System(S, 4, 550e-6, field_angle=np.radians(10)).at_paraxial_focus()
    model = cad.system_model(system, n_rays=5, mount=cad.Mount(),
                             mechanics=cad.Mechanics(margin=1.5, bevel=0.2))
    labels = [p.label for p in model.mount]
    assert labels[0] == "barrel" and "ring_front" in labels and "ring_back" in labels
    assert "stop" in labels and any(x.startswith("spacer_1") for x in labels)
    for part in model.mount:
        assert part.is_valid
        for lens in model.lenses:
            common = part & lens
            assert common is None or common.volume < 1e-6
    # every lens is held by rings or spacers on both sides:
    holders = [p for p in model.mount if p.label not in ("barrel", "stop")]
    for lens in model.lenses:
        z = lens.center().Z
        touching = [p for p in holders if p.distance_to(lens) < 1e-6]
        assert any(p.center().Z < z for p in touching) and any(p.center().Z > z for p in touching)
    # the spacers stay out of the traced beam:
    spacers = [p for p in model.mount if p.label not in ("barrel", "stop")]  # rim rays graze the stop
    for ray in model.rays:
        for part in spacers:
            assert part.distance_to(ray) > 0.05


def test_plot_section():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = cad.system_model(doublet(), n_rays=3, mechanics=cad.Mechanics(margin=1.0, bevel=0.2))
    ax = cad.plot_section(model, plt.figure().gca())
    assert len(ax.patches) == 2
    plt.close("all")
