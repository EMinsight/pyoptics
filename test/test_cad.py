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
    model = cad.system_model(doublet(), n_rays=5,
                             mechanics=cad.Mechanics(margin=1.2, bevel=b, land="flat"))
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
    # the sphere goes on to the rim, the flat back lies 3 mm behind it; the
    # concave face is thinner in the middle:
    r = np.linspace(0, 7.0, 20001)
    dish = np.trapezoid(2*np.pi*r*(_sag(-100.0, r) - _sag(-100.0, 7.0)), r)
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
    model = cad.system_model(system, rays=None,
                             mechanics=cad.Mechanics(margin=1.5, bevel=b, land="flat"))
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
                             mechanics=cad.Mechanics(margin=1.0, bevel=0.1))
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


def test_plot_3d_and_gltf(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = cad.system_model(doublet(), n_rays=3, mechanics=cad.Mechanics(margin=1.0, bevel=0.2))
    ax = cad.plot_3d(model)
    assert len(ax.collections) == 2
    plt.close("all")
    path = tmp_path/"doublet.glb"
    cad.export_gltf(model, path)
    assert path.read_bytes()[:4] == b"glTF"


def test_spheres_continue_to_the_rim():
    """The default land: the spheres go on to the rim (one SPHERE face per
    side, no flat land), bevels at 45 degrees from the continued surface."""

    rc, rm, b = 11.5, 12.7, 0.3
    model = cad.system_model(doublet(), n_rays=3, mechanics=cad.Mechanics(margin=1.2))
    lens = model.lenses[0]
    r = np.linspace(0, rm, 20001)
    thick = 4.0 + _sag(-45.7, r) - _sag(62.8, r)
    assert lens.volume == pytest.approx(np.trapezoid(2*np.pi*r*thick, r), rel=1e-6)
    assert {f.geom_type.name for f in lens.faces()} == {"SPHERE", "CYLINDER"}
    assert len([f for f in lens.faces() if f.geom_type.name == "SPHERE"]) == 2
    # with a bevel the front sphere ends at rm - b, the chamfer runs at 45
    # degrees from there to the rim:
    bev = cad.system_model(doublet(), n_rays=3, mechanics=cad.Mechanics(margin=1.2, bevel=b))
    assert bev.lenses[0].is_valid
    zb = _sag(62.8, rm - b)
    cut = np.trapezoid(2*np.pi*r*np.clip((zb + (r - rm + b)) - _sag(62.8, r), 0, None)
                       * (r >= rm - b), r)
    assert bev.lenses[0].volume == pytest.approx(lens.volume - cut, rel=1e-6)


def test_aspheres_continue_as_their_best_fit_sphere():
    s1 = rt.Surface(30.0, 6.0, n_after=1.5, conic=-0.8, aspheric=(-2e-6, 1e-9), semi_diameter=10)
    s2 = rt.Surface(-80.0, 50.0, n_after=1.0, semi_diameter=10)
    system = rt.System([s1, s2], 0, 550e-6)
    model = cad.system_model(system, rays=None, image_plane=False,
                             mechanics=cad.Mechanics(margin=2.0), tolerance=1e-5)
    lens = model.lenses[0]
    z_rim = float(s1.sag(np.array([100.0]))[0])
    R_bfs = (100 + z_rim**2)/(2*z_rim)
    for r in (5.0, 9.5, 10.5, 11.5):  # asphere inside, best fit sphere outside
        z = float(s1.sag(np.array([r**2]))[0]) if r < 10 else _sag(R_bfs, r)
        assert lens.distance_to(build123d.Vector(0.0, r, z)) < 2e-5
    kinds = [f.geom_type.name for f in lens.faces()]
    assert kinds.count("SPHERE") == 2 and "REVOLUTION" in kinds  # B-spline revolved


def test_toric_lens_continues_as_a_torus():
    tor = rt.ToroidalSurface(40.0, 4.0, R_x=25.0, n_after=1.5, semi_diameter=8.0)
    system = rt.System([tor, rt.Surface(-60.0, 40.0, n_after=1.0, semi_diameter=8.0)], 0,
                       550e-6)
    model = cad.system_model(system, rays=None, image_plane=False,
                             mechanics=cad.Mechanics(margin=1.0, bevel=0.2))
    (lens,) = model.lenses
    assert lens.is_valid
    # points of the torus inside the clear aperture and beyond it, up to the bevel:
    for x, y in ((0.0, 0.0), (5.0, 3.0), (0.0, 8.5), (8.5, 0.0), (6.0, -6.0)):
        z = float(tor.sag_xy(np.array([x]), np.array([y]))[0])
        assert lens.distance_to(build123d.Vector(x, y, z)) < 2e-3
    r = np.linspace(0, 9.0, 1201)
    p = np.linspace(0, 2*np.pi, 721, endpoint=False)
    Rg, Pg = np.meshgrid(r, p)
    thick = 4.0 + _sag(-60.0, Rg) - tor.sag_xy(Rg*np.cos(Pg), Rg*np.sin(Pg))
    V = np.trapezoid(np.mean(thick, axis=0)*2*np.pi*r, r)
    V -= 2*2*np.pi*(9.0 - 0.2/3)*0.2**2/2  # the two bevels
    assert lens.volume == pytest.approx(V, rel=2e-4)


# --- off-axis mirrors with outlines ---------------------------------------------

def _inside(solid, points, eps=1e-6):
    return sum(solid.is_inside(build123d.Vector(*p), eps) for p in points)


def test_off_axis_parabola_circle_on_the_normal():
    from pyoptics import off_axis as oa

    f, h, D, margin, t = 500.0, 150.0, 100.0, 4.0, 20.0
    oap = oa.off_axis_parabola(f, h, D)
    model = cad.system_model(oap, fields=[0.0], n_rays=5, ray_start=300.0, image_plane=False,
                             mechanics=cad.Mechanics(outline="circle", margin=margin,
                                                     mirror_thickness=t))
    (m,) = model.lenses
    assert m.label == "M1" and m.is_valid
    # the side wall is a cylinder about the mirror normal at the centre of
    # the footprint, of the footprint radius plus the margin:
    (wall,) = [fc for fc in m.faces() if fc.geom_type.name == "CYLINDER"]
    normal = np.array([0.0, h/(2*f), 1.0])
    normal /= np.linalg.norm(normal)
    axis = np.array(tuple(wall.axis_of_rotation.direction))
    assert abs(axis @ normal) == pytest.approx(1.0, abs=1e-6)
    # the smallest circle around the footprint, seen along that axis, plus
    # the margin (the footprint is stretched in y by the tilt of the face
    # and its curvature, to about 50.6 mm):
    a0 = np.array(tuple(wall.axis_of_rotation.position))
    o, R = oap.frames()[0][1]
    fp = o + cad._footprint(oap, 1, [0.0]) @ R.T
    d = fp - a0 - np.outer((fp - a0) @ axis, axis)
    assert np.max(np.linalg.norm(d, axis=1)) \
        == pytest.approx(wall.radius - margin, abs=1e-4)
    assert D/2 < wall.radius - margin < 1.02*D/2
    # at least mirror_thickness thick, the back flat and perpendicular to the normal:
    (back,) = [fc for fc in m.faces() if fc.geom_type.name == "PLANE"]
    assert abs(np.array(tuple(back.normal_at())) @ normal) == pytest.approx(1.0, abs=1e-9)
    # a disk of thickness t behind the deepest point plus the concave cap
    # (sag about r^2/2R = 1.5 mm at the rim for R about 1 m):
    A = np.pi*wall.radius**2
    assert A*t < m.volume < A*(t + 1.5)
    # the traced hits lie on the mirror face, the rays stay out of the glass:
    b = oap.trace_pupil(0.0, *np.random.default_rng(1).uniform(-0.7, 0.7, (2, 20)))
    hits = b.points[2][np.all(np.isfinite(b.points[2]), axis=1)]
    assert max(m.distance_to(build123d.Vector(*p)) for p in hits) < 1e-3
    assert _inside(m, hits - 0.5*b.directions[1][:len(hits)]) == 0
    assert _inside(m, hits + 0.5*b.directions[2][:len(hits)]) == 0


def test_tma_hull_outlines_clear_the_beams(tmp_path):
    from pyoptics import off_axis as oa

    tma = oa.cook_tma(efl=600.0, f_number=4.0, field_angle=np.radians(4.0), aperture_offset=-2.0)
    F = oa.rectangular_field((0.0, np.radians(3.0)), np.radians([1.5, 0.5]), (3, 3))
    margin, t = 3.0, 20.0
    model = cad.system_model(tma, fields=[tuple(f) for f in F[[1, 4, 7]]], n_rays=5,
                             footprint_fields=[tuple(f) for f in F],
                             mechanics=cad.Mechanics(outline="hull", margin=margin,
                                                     mirror_thickness=t))
    assert [m.label for m in model.lenses] == ["M1", "M2", "M3"]
    assert all(m.is_valid and len(m.faces()) == 3 for m in model.lenses)
    # no ray runs through a substrate:
    for ray in model.rays:
        for e in ray.edges():
            p, q = np.array(tuple(e.start_point())), np.array(tuple(e.end_point()))
            pts = p + np.linspace(0.02, 0.98, 5)[:, None]*(q - p)
            assert all(_inside(m, pts) == 0 for m in model.lenses)
    # every footprint (also of the fields not drawn) lies on the mirror face
    # at least margin from the side wall:
    fp = oa.footprints(tma, F)
    frames = tma.frames()[0]
    for m, i in zip(model.lenses, (0, 2, 3)):
        o, R = frames[i]
        hits = o + np.c_[fp[i].hull, [tma.surfaces[i].sag(np.array([x @ x]))[0]
                                      for x in fp[i].hull]] @ R.T
        assert max(m.distance_to(build123d.Vector(*p)) for p in hits) < 1e-3
    path = tmp_path/"tma.step"
    cad.export_step(model, path)
    assert len(build123d.import_step(str(path)).solids()) == 3


def test_polygon_outline_along_the_parent_axis():
    from pyoptics import off_axis as oa

    oap = oa.off_axis_parabola(500.0, 150.0, 100.0)
    poly = np.array([[-60.0, 90.0], [60.0, 90.0], [60.0, 210.0], [-60.0, 210.0]])
    model = cad.system_model(oap, fields=[0.0], rays=None, image_plane=False,
                             mechanics=cad.Mechanics(outline=poly, axis="parent",
                                                     mirror_thickness=10.0))
    (m,) = model.lenses
    box = m.bounding_box()
    assert (box.min.X, box.max.X) == pytest.approx((-60.0, 60.0), abs=1e-6)
    assert (box.min.Y, box.max.Y) == pytest.approx((90.0, 210.0), abs=1e-6)
    # four plane side walls parallel to the axis, a flat back perpendicular to it:
    planes = [fc for fc in m.faces() if fc.geom_type.name == "PLANE"]
    assert len(planes) == 5
    z = [abs(np.array(tuple(fc.normal_at()))[2]) for fc in planes]
    assert sorted(np.round(z, 9)) == [0, 0, 0, 0, 1]


def test_outline_errors():
    from pyoptics import off_axis as oa

    oap = oa.off_axis_parabola(500.0, 150.0, 100.0)
    with pytest.raises(NotImplementedError):
        cad.system_model(oap, fields=[0.0], rays=None,
                         mechanics=cad.Mechanics(outline="circle", land="flat"))
    with pytest.raises(NotImplementedError):
        cad.system_model(doublet(), rays=None, mechanics=cad.Mechanics(outline="circle"))
    with pytest.raises(ValueError):
        cad.system_model(oap, fields=[0.0], rays=None,
                         mechanics=cad.Mechanics(outline="ellipse"))
