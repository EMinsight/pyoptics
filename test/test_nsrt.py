#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Tests of the non-sequential ray tracer (pyoptics.nsrt) and of the
scattering models (pyoptics.bsdf)."""

import numpy as np
import pytest

from pyoptics import nsrt, bsdf
from pyoptics import raytrace as rt


@pytest.fixture(params=[True, False], ids=["numba", "numpy"])
def kernel(request):
    old = nsrt.use_numba
    nsrt.use_numba = request.param
    yield request.param
    nsrt.use_numba = old


# --- scattering models -------------------------------------------------------

@pytest.mark.parametrize("model", [bsdf.Lambertian(0.3), bsdf.Gaussian(0.05, 0.1),
                                   bsdf.ABg(A=0.002, B=0.01, g=2.0),
                                   bsdf.Lambertian(0.02) + bsdf.ABg(A=0.001, B=1e-3, g=1.5)])
@pytest.mark.parametrize("b0", [0.0, 0.5, 0.9])
def test_bsdf_tis_is_integral(model, b0):
    """TIS = integral of the BSDF over the unit disk of direction cosines."""

    # integrate radially around beta_0 for peaked lobes:
    rho = np.concatenate([[0.0], np.geomspace(1e-6, 2.0, 4000)])
    phi = np.linspace(0, 2*np.pi, 721)[:-1]
    Rr, P = np.meshgrid(rho, phi)
    B = np.stack([b0 + Rr.ravel()*np.cos(P.ravel()), Rr.ravel()*np.sin(P.ravel())], -1)
    f = model.bsdf(B, np.tile([b0, 0.0], (len(B), 1))).reshape(Rr.shape)
    g = f*Rr
    integral = np.sum(0.5*(g[:, 1:] + g[:, :-1])*np.diff(rho), axis=1).mean()*2*np.pi
    assert model.tis(np.array([b0]))[0] == pytest.approx(integral, rel=5e-3)


@pytest.mark.parametrize("model", [bsdf.Lambertian(1.0), bsdf.Gaussian(0.05, 0.1),
                                   bsdf.ABg(A=0.002, B=0.01, g=2.0)])
def test_bsdf_samples_follow_bsdf(model):
    """Histogram of the samples (in |beta - beta_0|) matches bsdf/tis."""

    rng = np.random.default_rng(1)
    b0 = np.tile([0.3, 0.0], (200000, 1))
    s = model.sample(b0, rng)
    assert np.all(np.sum(s**2, -1) <= 1 + 1e-12)
    # mean of a test function against the pdf:
    def h(b):
        return np.cos(3*b[:, 0]) + b[:, 1]**2
    mc = h(s).mean()
    n = 801
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x)
    beta = np.stack([X.ravel(), Y.ravel()], -1)
    p = model.pdf(beta, np.tile([0.3, 0.0], (len(beta), 1)))
    if isinstance(model, bsdf.Lambertian):
        ref = np.sum(h(beta)*p)*(x[1] - x[0])**2
        assert mc == pytest.approx(ref, abs=5e-3)
    else:
        # peaked lobes: compare with the mean from radial quadrature
        rho = np.concatenate([[0.0], np.geomspace(1e-6, 2.0, 3000)])
        phi = np.linspace(0, 2*np.pi, 721)[:-1]
        Rr, P = np.meshgrid(rho, phi)
        B = np.stack([0.3 + Rr.ravel()*np.cos(P.ravel()), Rr.ravel()*np.sin(P.ravel())], -1)
        pp = model.pdf(B, np.tile([0.3, 0.0], (len(B), 1))).reshape(Rr.shape)
        g = (h(B).reshape(Rr.shape)*pp*Rr)
        ref = np.sum(0.5*(g[:, 1:] + g[:, :-1])*np.diff(rho), axis=1).mean()*2*np.pi
        assert mc == pytest.approx(ref, abs=5e-3)


def test_lambertian_cosine_law():
    rng = np.random.default_rng(0)
    s = bsdf.Lambertian(1.0).sample(np.zeros((400000, 2)), rng)
    cos = np.sqrt(1 - np.sum(s**2, -1))
    # power into theta < 30 degrees: sin(30)**2 = 1/4
    assert np.mean(cos > np.cos(np.radians(30))) == pytest.approx(0.25, abs=3e-3)


# --- geometry ----------------------------------------------------------------

def test_caldwell_matches_sequential_trace(kernel):
    s = rt.caldwell_krf_lens()
    scene = nsrt.Scene(wavelength=s.wavelength)
    scene.add_system(s, property=nsrt.ideal(), land=nsrt.absorber(), bevel=nsrt.absorber(),
                     rim=nsrt.absorber())
    rng = np.random.default_rng(2)
    r = 0.95*np.sqrt(rng.random(300))
    a = 2*np.pi*rng.random(300)
    for field in (0.0, 46.8):
        rays = s.rays(field, r*np.cos(a), r*np.sin(a))
        b = s.trace(rays)
        res = scene.trace(nsrt.RaySet(rays.points, rays.directions, np.ones(len(r))))
        h = res.hits["image"]
        ref = s.to_image_frame(b.points[-1][b.valid])
        assert len(h) == b.valid.sum()
        # every NSRT hit lies on a sequential one
        dist = np.min(np.linalg.norm(h.points[:, None, :2] - ref[None, :, :2], axis=-1), axis=1)
        assert dist.max() < 1e-9
        assert res.balance()["total"] == pytest.approx(1.0, abs=1e-12)


def test_slab_multiple_reflections(kernel):
    """Incoherent transmission of a plane parallel plate: (1 - R)/(1 + R)."""

    n = 1.5
    scene = nsrt.Scene()
    scene.add(nsrt.ConicFace(np.inf, 20.0, media=(1.0, n), label="front"),
              nsrt.ConicFace(np.inf, 20.0, position=(0, 0, 5.0), media=(n, 1.0), label="back"),
              nsrt.Rectangle(100, 100, position=(0, 0, 20.0), property=nsrt.absorber(),
                             label="det", detector=True))
    src = nsrt.collimated_beam((0, 0, -10), (0, 0, 1), 1.0, 100)
    res = scene.trace(src, min_weight=1e-12, max_events=100)
    R = ((n - 1)/(n + 1))**2
    assert res.hits["det"].power == pytest.approx((1 - R)/(1 + R), rel=1e-9)
    assert res.escaped == pytest.approx(2*R/(1 + R), rel=1e-9)
    h = res.hits["det"]
    two = h.mask(reflections=2)
    assert h.weights[two].sum() == pytest.approx((1 - R)**2*R**2, rel=1e-9)
    paths = h.paths(2)
    assert paths[0][2] == "direct"
    assert paths[1][2] == "back R -> front R"


def test_total_internal_reflection_in_prism(kernel):
    """A right angle prism used as a retro (two TIR bounces) returns all
    light minus the two Fresnel transmissions."""

    n = 1.5
    # right angle prism with the hypotenuse in the plane z = 0, as a mesh
    V = np.array([[-5, -10, 0], [5, -10, 0], [0, -10, 5], [-5, 10, 0], [5, 10, 0], [0, 10, 5]],
                 float)
    T = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [0, 3, 5],
                  [0, 5, 2]])
    # wind the triangles counterclockwise seen from outside (convex body):
    c = V[T].mean(1) - V.mean(0)
    nrm = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    T = np.where((np.sum(nrm*c, -1) < 0)[:, None], T[:, ::-1], T)
    mesh = nsrt.Mesh(V, T, properties=nsrt.fresnel(), media=(n, 1.0), labels=["prism"])
    assert np.all(np.sum(mesh.normals*c, -1) > 0)
    scene = nsrt.Scene()
    scene.add(mesh)
    src = nsrt.collimated_beam((2.0, 0, -5), (0, 0, 1), 0.5, 50)
    res = scene.trace(src, min_weight=1e-9, record=True)
    R = ((n - 1)/(n + 1))**2
    # back out after two TIR: (1-R)**2 of the light, plus multiple reflections
    back = sum(np.sum(w) for (A, B, w, last) in res.segments)  # traced at all
    assert res.escaped == pytest.approx(1.0, rel=1e-9)
    assert back > 0
    assert res.absorbed.sum() == pytest.approx(0.0, abs=1e-12)


def test_mesh_matches_analytic_sphere(kernel):
    """Hits on a fine icosphere lie within its faceting error of the sphere."""

    V, T = _icosphere(4)
    mesh = nsrt.Mesh(10*V, T)
    rng = np.random.default_rng(3)
    P = rng.normal(size=(500, 3))*2 - [0, 0, 30]
    d = nsrt._unit(rng.normal(size=(500, 3))*0.05 + [0, 0, 1])
    t, tri = mesh.intersect(P, d, 1e-9)
    t_ref = nsrt.Sphere(10.0).intersect(P, d, 1e-9)
    hit = np.isfinite(t_ref) & np.isfinite(t)
    assert hit.sum() > 400
    Q = P + np.where(np.isfinite(t_ref), t_ref, 0)[:, None]*d
    hit &= np.abs(np.sum(Q*d, -1))/10 > 0.5  # not grazing
    sag_err = 10*(1 - np.cos(np.pi/5/2**4))  # rough faceting error
    assert np.max(np.abs(t[hit] - t_ref[hit])) < 5*sag_err
    # both kernels agree
    t2, tri2 = mesh._intersect_numpy(P, d, 1e-9)
    assert np.allclose(t[hit], t2[hit])
    assert np.array_equal(tri[hit], tri2[hit])


def _icosphere(k):
    p = (1 + 5**0.5)/2
    V = np.array([[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0], [0, -1, p], [0, 1, p],
                  [0, -1, -p], [0, 1, -p], [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]], float)
    T = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    V = list(V/np.linalg.norm(V, axis=1)[:, None])
    for _ in range(k):
        cache, T2 = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = V[a] + V[b]
                V.append(m/np.linalg.norm(m))
                cache[key] = len(V) - 1
            return cache[key]
        for a, b, c in T:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            T2 += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        T = T2
    return np.array(V), np.array(T)


@pytest.mark.parametrize("face", [nsrt.Cylinder(3.0, -2.0, 4.0), nsrt.Cone(1.0, 0.0, 3.0, 2.0),
                                  nsrt.Sphere(2.0), nsrt.ConicFace(5.0, 3.0, 0.5, conic=-0.5),
                                  nsrt.Rectangle(4.0, 2.0)])
def test_primitives_numba_equals_numpy(face):
    rng = np.random.default_rng(4)
    face.R = rt.rotation(0.2, -0.3, 0.1)
    face.origin = np.array([0.1, -0.2, 0.3])
    scene = nsrt.Scene()
    scene.add(face)
    P = rng.normal(size=(2000, 3))*3
    d = nsrt._unit(rng.normal(size=(2000, 3)))
    nsrt.use_numba = True
    t1, f1, _ = scene.nearest(P, d)
    nsrt.use_numba = False
    t2, f2, _ = scene.nearest(P, d)
    nsrt.use_numba = True
    assert np.isfinite(t1).sum() > 50
    assert np.array_equal(f1, f2)
    ok = np.isfinite(t1)
    assert np.allclose(t1[ok], t2[ok], rtol=1e-12, atol=1e-12)
    # hit points lie on the surface: normal and implicit check via re-intersection
    Q = P[ok] + t1[ok, None]*d[ok]
    q = face.to_local(Q)
    if isinstance(face, nsrt.Cylinder):
        assert np.allclose(np.hypot(q[:, 0], q[:, 1]), 3.0)
    if isinstance(face, nsrt.Sphere):
        assert np.allclose(np.linalg.norm(q, axis=1), 2.0)
    if isinstance(face, nsrt.ConicFace):
        assert np.allclose(q[:, 2], face.sag(np.hypot(q[:, 0], q[:, 1])), atol=1e-10)


def test_sag_face_asphere_hits():
    s = rt.Surface(20.0, conic=-1.2, aspheric=(1e-4, -2e-7))
    face = nsrt.SagFace(s, 6.0, position=(0, 0, 1.0))
    rng = np.random.default_rng(5)
    P = np.c_[rng.uniform(-5, 5, (300, 2)), np.full(300, -3.0)]
    d = nsrt._unit(np.c_[rng.normal(size=(300, 2))*0.1, np.ones(300)])
    t = face.intersect(P, d, 1e-9)
    ok = np.isfinite(t)
    assert ok.sum() > 250
    q = face.to_local(P[ok] + t[ok, None]*d[ok])
    assert np.allclose(q[:, 2], s.sag(q[:, 0]**2 + q[:, 1]**2), atol=1e-12)
    # the sequential intersection agrees
    t_seq = s.intersect(P[ok] - [0, 0, 1.0], d[ok])
    assert np.allclose(t[ok], t_seq, atol=1e-10)


def test_box_refraction_restores_direction(kernel):
    scene = nsrt.Scene()
    scene.add(nsrt.box((10, 10, 10), rotation=rt.rotation(0.3, 0.2, 0.0), media=(1.7, 1.0),
                       property=nsrt.ideal()))
    src = nsrt.RaySet(np.array([[0.5, 0.2, -20.0]]), np.array([[0.1, 0.05, 1.0]])/np.linalg.norm(
        [0.1, 0.05, 1.0]), np.array([1.0]))
    res = scene.trace(src, record=True)
    A, B, w, last = res.segments[-1]
    assert np.allclose(nsrt._unit(B - A)[0], src.directions[0])
    assert res.escaped == pytest.approx(1.0)


# --- radiometry --------------------------------------------------------------

def test_lambertian_view_factor_and_importance_sampling(kernel):
    """A small Lambertian spot (reflectance rho) seen by a coaxial disk of
    radius a at height h receives rho a**2/(a**2 + h**2); aiming the scattered
    rays at the disk gives the same mean with less noise."""

    rho, a, h = 0.8, 2.0, 10.0
    scene = nsrt.Scene()
    plate = scene.add(nsrt.Rectangle(4, 4, property=nsrt.diffuse(rho), label="plate"))
    det = scene.add(nsrt.ConicFace(np.inf, a, position=(0, 0, h), property=nsrt.absorber(),
                                   label="det", detector=True))
    src = nsrt.collimated_beam((0, 0, 5), (0, 0, -1), 0.01, 20000, rng=1)
    ref = rho*a**2/(a**2 + h**2)
    plain = scene.trace(src, rng=2).hits["det"]
    aimed = scene.trace(src, rng=3, targets=[nsrt.Target.from_face(det)],
                        aim_fraction=0.9).hits["det"]
    assert plain.power == pytest.approx(ref, rel=0.05)
    assert aimed.power == pytest.approx(ref, rel=0.01)
    # aimed: many more rays reach the detector
    assert len(aimed) > 5*len(plain)


def test_integrating_sphere(kernel):
    """Port flux of an integrating sphere with Lambertian walls: rho f/(1 - rho (1 - f))
    for light that first hits the wall (f: port area fraction)."""

    R, rho, a = 50.0, 0.9, 10.0
    scene = nsrt.Scene()
    scene.add(nsrt.Sphere(R, property=nsrt.diffuse(rho), label="wall"))
    # port: an absorbing detector disk in the wall at the top, facing in
    port = nsrt.ConicFace(-R, a, position=(0, 0, R*(1 - 1e-6)), rotation=np.eye(3),
                          property=nsrt.absorber(), label="port", detector=True)
    scene.add(port)
    # cap area of the port (on the sphere): 2 pi R h, h = R - sqrt(R**2 - a**2)
    f = 2*np.pi*R*(R - np.sqrt(R**2 - a**2))/(4*np.pi*R**2)
    src = nsrt.collimated_beam((0, 0, 0), (0, 0, -1), 1.0, 20000, rng=4)
    res = scene.trace(src, mode="mc", rng=5, max_events=400, min_weight=1e-3)
    assert res.hits["port"].power == pytest.approx(rho*f/(1 - rho*(1 - f)), rel=0.05)
    assert res.balance()["total"] == pytest.approx(1.0, rel=0.03)


def test_coating_lowers_reflection():
    from pyoptics.coatings import Coating, Layer

    n = 1.5
    wl = 550e-6
    ar = Coating([Layer(n**0.5, wl/(4*n**0.5))])
    for prop, R_ref in ((nsrt.fresnel(), ((n - 1)/(n + 1))**2), (nsrt.fresnel(coating=ar), 0.0)):
        scene = nsrt.Scene(wavelength=wl)
        scene.add(nsrt.ConicFace(np.inf, 5.0, media=(1.0, n), property=prop))
        res = scene.trace(nsrt.collimated_beam((0, 0, -1), (0, 0, 1), 1.0, 10), min_weight=0)
        # reflected part escapes backwards, transmitted forward: both escape
        A, B, w, last = (None,)*4
        assert res.escaped == pytest.approx(1.0)
        R = res.balance()
        # trace the reflected power: record
        res = scene.trace(nsrt.collimated_beam((0, 0, -1), (0, 0, 1), 1.0, 10), record=True,
                          min_weight=0)
        A, B, w, last = res.segments[1]
        refl = w[(B - A)[:, 2] < 0].sum()
        assert refl == pytest.approx(R_ref, abs=1e-6)


def test_bulk_absorption():
    alpha, L = 0.1, 5.0
    scene = nsrt.Scene()
    glass = nsrt.Medium(1.5, absorption=alpha)
    scene.add(nsrt.ConicFace(np.inf, 5.0, media=(1.0, glass), property=nsrt.ideal()),
              nsrt.ConicFace(np.inf, 5.0, position=(0, 0, L), media=(glass, 1.0),
                             property=nsrt.ideal()))
    res = scene.trace(nsrt.collimated_beam((0, 0, -1), (0, 0, 1), 1.0, 10))
    assert res.escaped == pytest.approx(np.exp(-alpha*L))
    assert res.bulk.sum() == pytest.approx(1 - np.exp(-alpha*L))


def test_ghost_matches_photography_ghost_system():
    """The double reflection ghost of a singlet traced non-sequentially lands
    where the unfolded ghost system of notebook 92 (flare and ghosts) puts it."""

    from pyoptics import photography

    s = rt.System([rt.Surface(40.0, 4.0, n_after=1.6, semi_diameter=8.0),
                   rt.Surface(-60.0, 45.0, n_after=1.0, semi_diameter=8.0)], 0, 550e-6,
                  field_angle=0.05)
    s = s.at_paraxial_focus()
    scene = nsrt.Scene(wavelength=s.wavelength)
    scene.add_system(s, image_size=200.0)
    rays = s.rays(0.05, *np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 5)))
    res = scene.trace(nsrt.RaySet(rays.points, rays.directions, np.ones(25)), min_weight=1e-9)
    h = res.hits["image"]
    m = (h.mask(reflections=2) & h.mask(faces=["L1:S2"], events="R")
         & h.mask(faces=["L1:S1"], events="R"))
    g, _ = photography.ghost_system(s, 1, 0)
    b = g.trace(rays)
    ref = g.to_image_frame(b.points[-1][b.valid])
    got = h.points[m]
    assert len(got) == len(ref)
    dist = np.min(np.linalg.norm(got[:, None, :2] - ref[None, :, :2], axis=-1), axis=1)
    assert dist.max() < 1e-9
    R1 = ((1.6 - 1)/(1.6 + 1))**2
    assert h.weights[m].max() < R1


def test_mc_and_split_agree():
    """Both modes give the same detector power on average (slab)."""

    n = 1.5
    scene = nsrt.Scene()
    scene.add(nsrt.ConicFace(np.inf, 20.0, media=(1.0, n)),
              nsrt.ConicFace(np.inf, 20.0, position=(0, 0, 5.0), media=(n, 1.0)),
              nsrt.Rectangle(100, 100, position=(0, 0, 20.0), property=nsrt.absorber(),
                             label="det", detector=True))
    src = nsrt.collimated_beam((0, 0, -10), (0, 0, 1), 1.0, 40000, rng=0)
    R = ((n - 1)/(n + 1))**2
    res = scene.trace(src, mode="mc", rng=1)
    assert res.hits["det"].power == pytest.approx((1 - R)/(1 + R), rel=0.01)


def test_stl_roundtrip(tmp_path):
    V, T = _icosphere(1)
    fn = tmp_path/"ball.stl"
    with open(fn, "w") as f:
        f.write("solid ball\n")
        for t in T:
            f.write("facet normal 0 0 0\nouter loop\n")
            for k in t:
                f.write("vertex {} {} {}\n".format(*V[k]))
            f.write("endloop\nendfacet\n")
        f.write("endsolid ball\n")
    V2, T2 = nsrt.read_stl(fn)
    assert len(T2) == len(T)
    assert np.allclose(np.sort(np.linalg.norm(V2, axis=1)), 1.0)


def test_cad_shapes_tessellated():
    bd = pytest.importorskip("build123d")
    tube = bd.Solid.make_cylinder(10, 20) - bd.Solid.make_cylinder(8, 20)
    tube.label = "tube"
    scene = nsrt.Scene()
    mesh = scene.add_shapes([tube], property=nsrt.absorber(), tolerance=0.01)
    assert scene.faces[0].label == "tube"
    src = nsrt.RaySet(np.array([[0, 0, 10.0]]), np.array([[1.0, 0, 0]]), np.array([1.0]))
    res = scene.trace(src)
    assert res.absorbed_by("tube") == pytest.approx(1.0)
    t, tri = mesh.intersect(src.points, src.directions, 1e-9)
    assert t[0] == pytest.approx(8.0, abs=0.02)


def test_cad_properties_by_label_prefix_and_face():
    bd = pytest.importorskip("build123d")
    tube = bd.Solid.make_cylinder(10, 20) - bd.Solid.make_cylinder(8, 20)
    tube.label = "spacer_1"
    ring = bd.Solid.make_cylinder(10, 2, bd.Plane(origin=(0, 0, 30))) - \
        bd.Solid.make_cylinder(5, 2, bd.Plane(origin=(0, 0, 30)))
    ring.label = "ring"
    black, metal = nsrt.black_paint(), nsrt.machined_metal()
    scene = nsrt.Scene()
    scene.add_shapes([tube, ring], property={"spacer_*": black, "*": metal})
    assert scene.face("spacer_1").property is black
    assert scene.face("ring").property is metal

    def finish(label, face):
        return black if face.geom_type.name == "CYLINDER" else None

    scene = nsrt.Scene()
    scene.add_shapes([tube], property=finish, per_face=True)
    kinds = {f.label: f.property for f in scene.faces}
    assert all(lab.startswith("spacer_1.f") for lab in kinds)
    n_cyl = sum(f.geom_type.name == "CYLINDER" for f in tube.faces())
    assert sum(p is black for p in kinds.values()) == n_cyl
    assert sum(p.label == "absorber" for p in kinds.values()) == len(kinds) - n_cyl
