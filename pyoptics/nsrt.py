#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Non-sequential Monte Carlo ray tracing.

In a sequential trace (`pyoptics.raytrace`) every ray meets the surfaces of
a system in a fixed order. Here rays find their own way through a `Scene`
of faces: they are reflected, refracted, scattered and absorbed wherever
they hit, split into a reflected and a transmitted part at every glass
surface, and can bounce around inside a lens barrel. This is what ghost
images, stray light, illumination optics and light pipes need.

Scene
-----
A `Scene` is a list of faces in one global frame (lengths in its `units`,
mm by default). Every face is oriented: it has a normal, the medium on the
side the normal points away from (``media[0]``, "minus side") and the
medium on the side it points to (``media[1]``, "plus side"). The media are
attached to the faces, not to volumes, so a cemented surface simply has
glass on both sides and nested parts need no bookkeeping. Faces of solids
have their normals pointing out of the solid (``media = (inside,
outside)``); the faces of a lens from a `raytrace.System` have their
normals along the axis, like the surfaces of the sequential trace.

Face types:

- analytic: `ConicFace` (spheres, conics and planes, as disks or annuli,
  the surfaces of `raytrace` without aspheric terms), `SagFace` (any
  `raytrace` surface with a sag: aspheres, toroids, Zernike and Q-type
  freeforms), `Rectangle`, `Cylinder`, `Cone`, `Sphere`, and `box()` for
  six rectangles,
- `Mesh`: triangle meshes (with a bounding volume hierarchy), e.g. from
  CAD solids: `Scene.add_shapes()` tessellates build123d shapes,
  `Scene.add_step()` reads STEP files, `Scene.add_stl()` STL files.
  Triangles are flat, so the optical surfaces of lenses should stay
  analytic; meshes are meant for mounts, baffles, housings and prisms.

`Scene.add_system()` adds the lenses of a coaxial `raytrace.System` as
analytic faces: the optical surfaces up to their clear apertures, the
lands beyond them, the bevels and the cylindrical rims, with the
dimensions of `pyoptics.cad.Mechanics`, so that the traced lens is the one
exported to STEP. With a `cad.Mount` it also adds the barrel, spacers,
rings and the stop plate from `cad.system_model()`, tessellated.

Properties of CAD solids
------------------------
`Scene.add_shapes()` (build123d solids), `Scene.add_step()` (STEP files)
and the ``mount_property`` of `Scene.add_system()` assign properties by
label: each solid is one face of the scene with the solid's label, or with
``per_face=True`` each of its B-rep faces ("barrel.f0", "barrel.f1", ...).
Give one property for all, a dict by label (exact labels, prefixes
ending in "*", "*" for the rest) or a function of the label and the
build123d solid or face ::

    scene.add_step("mount.step", property={"barrel": nsrt.black_anodized(),
                                           "spacer_*": nsrt.machined_metal(),
                                           "*": nsrt.absorber()})

    def finish(label, face):       # with per_face=True
        if face.geom_type.name in ("CYLINDER", "CONE"):
            return nsrt.black_paint()  # bores and cones blackened
        return nsrt.machined_metal()   # end faces left bare
    scene.add_shapes(model.mount, property=finish, per_face=True)

Transparent solids get ``media=(inside, outside)`` in the same way.

Face properties
---------------
A `FaceProperty` decides what happens at a hit: the specular reflectance
R and transmittance T (None: Fresnel's equations from the two indices, or
a thin film `coatings.Coating`; numbers or functions of the angle), plus
scattering models (`pyoptics.bsdf`) for the reflected (``brdf``) and
transmitted (``btdf``) side, whose total integrated scatter adds to the
loss of the specular parts (for Fresnel interfaces it is taken out of
them). The rest is absorbed. Polarization is neglected: Fresnel and
coating coefficients are averaged over s and p. Bulk absorption follows
from the absorption coefficient of a `Medium`. Presets: `fresnel()`,
`ideal()`, `mirror()`, `absorber()`, `diffuse()`, `ground_glass()`,
`black_anodized()`, `black_paint()`, `machined_metal()` (invented,
plausible values).

Tracing
-------
`Scene.trace()` follows all rays together, one interaction per step:

- ``mode="split"``: every ray splits into its reflected, transmitted and
  scattered children with the weights of the channels. This follows weak
  reflections (ghosts) deterministically. Children below ``min_weight``
  (relative to the mean source ray) go through Russian roulette: they
  survive with probability weight/threshold at the threshold weight, so
  the result stays unbiased.
- ``mode="mc"``: every ray picks one channel at random (implicit capture:
  the absorbed part is booked and taken off the weight), plus Russian
  roulette. Fewer rays per step, more noise.

Scattered rays are drawn from the BSDF, or, with ``targets`` (disks, e.g.
the next lens aperture or the detector), aimed at them with probability
``aim_fraction``; both strategies are combined with multiple importance
sampling (the one-sample balance heuristic), so the weights stay unbiased
while far more scattered rays reach where they matter.

Every ray keeps the history of its interactions, (face, event) with the
events transmitted (T), reflected (R), scattered in reflection (S) and in
transmission (s). Detector faces (``detector=True``) record their hits with
that history: `Hits.paths()` ranks the stray light paths by power,
`Hits.mask()` filters (e.g. ghosts: exactly two specular reflections),
`Hits.irradiance()` bins the power. `TraceResult` books the absorbed power
per face, the escaped and the lost power.

Sources: `point_source()`, `collimated_beam()`, `lambertian_source()` and
`system_source()` (an object point of a `raytrace.System` filling its
entrance pupil), or any `RaySet`. `plot_scene()` and `plot_rays()` draw
sections with matplotlib.

The optional numba (``pip install pyoptics[fast]``) compiles the hit search
on primitives and meshes; build123d (``pyoptics[cad]``) is needed only for
CAD import and mounts. Notebook 166 (non-sequential ray tracing)
introduces all of it, notebook 167 (stray light in Caldwell's lens)
analyses ghosts and mount scatter of a lithography lens.

References
----------
J. E. Harvey, "Light-scattering characteristics of optical surfaces",
PhD thesis, University of Arizona (1976).

E. C. Fest, "Stray Light Analysis and Control", SPIE Press (2013).

R. P. Breault, "Control of stray light", in Handbook of Optics, vol. I,
ch. 38, McGraw-Hill (1995).

E. Veach and L. J. Guibas, "Optimally combining sampling techniques for Monte
Carlo rendering", SIGGRAPH 95, 419-428 (1995).

T. Möller and B. Trumbore, "Fast, minimum storage ray-triangle
intersection", J. Graphics Tools 2, 21-28 (1997).
"""

from collections import namedtuple

import numpy as np

from . import raytrace as rt
from . import bsdf as _bsdf
from . import _nsrt_numba
from .materials import Material, abbe_dispersion

__all__ = ["Medium", "AIR", "FaceProperty", "fresnel", "ideal", "mirror", "absorber", "diffuse",
           "ground_glass", "black_anodized", "black_paint", "machined_metal", "Face",
           "ConicFace", "SagFace", "Rectangle", "Cylinder", "Cone", "Sphere", "box", "Mesh",
           "Scene", "RaySet", "point_source", "collimated_beam", "lambertian_source",
           "system_source", "Target", "Hits", "TraceResult", "plot_scene", "plot_rays",
           "read_stl", "EVENTS"]

use_numba = True

#: event letters of the history codes: transmitted, reflected, scattered in
#: reflection, scattered in transmission
EVENTS = "TRSs"
_T, _R, _SR, _ST = 0, 1, 2, 3


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _unit(v):
    v = np.asarray(v, float)
    return v/np.linalg.norm(v, axis=-1, keepdims=True)


def _frame_to(axis):
    """Rotation whose third column is the unit vector axis."""

    z = _unit(axis)
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _unit(np.cross(a, z))
    return np.stack([x, np.cross(z, x), z], axis=-1)


# --- media and face properties ---------------------------------------------

class Medium(object):
    """Optical medium.

    Parameters
    ----------
    n : number, callable, (n_d, V_d) or materials.Material, optional
        Refractive index: a number, a function n(wavelength_nm), a pair for
        `materials.abbe_dispersion()` or a Material (its geometric index).
    absorption : number, optional
        Absorption coefficient (1/length in the units of the scene): the
        power falls as exp(-absorption*path length).
    label : str, optional
    """

    def __init__(self, n=1.0, absorption=0.0, label=None):
        self.n, self.absorption, self.label = n, float(absorption), label

    def __repr__(self):
        return "Medium({!r}, absorption={})".format(self.n, self.absorption)

    def index(self, wavelength_nm):
        """Refractive index at the vacuum wavelength (nm)."""

        n = self.n
        if isinstance(n, Medium):
            return n.index(wavelength_nm)
        if isinstance(n, Material):
            return float(n.n_geometric(wavelength_nm))
        if callable(n):
            return float(n(wavelength_nm))
        if isinstance(n, (tuple, list)):
            return float(abbe_dispersion(wavelength_nm, *n))
        return float(n)


AIR = Medium(1.0, label="air")


def _as_medium(m):
    return m if isinstance(m, Medium) or m is None else Medium(m)


def _fresnel_unpolarized(cos_i, n1, n2):
    """Unpolarized Fresnel reflectance and transmittance (cos_i >= 0)."""

    sin2_t = (n1/n2)**2*(1 - cos_i**2)
    tir = sin2_t >= 1.0
    cos_t = np.sqrt(np.clip(1 - sin2_t, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = (n1*cos_i - n2*cos_t)/(n1*cos_i + n2*cos_t)
        rp = (n2*cos_i - n1*cos_t)/(n2*cos_i + n1*cos_t)
    R = np.where(tir, 1.0, 0.5*(rs**2 + rp**2))
    R = np.where(np.isfinite(R), R, 1.0)
    return R, 1.0 - R


class FaceProperty(object):
    """What happens to a ray that hits a face (see the module docstring).

    Parameters
    ----------
    R, T : None, number, callable or coatings.Coating, optional
        Specular reflectance and transmittance. None: from Fresnel's
        equations (unpolarized) for both (or from ``coating``). A number,
        a function f(cos_i, wavelength_nm) of the cosine of the angle of
        incidence, or for R a Coating on a substrate (a mirror coating). If
        only one of R and T is given, the other is 0.
    coating : coatings.Coating, optional
        Thin film stack on a refracting face (R and T None), layers listed
        from the minus side of the face (``media[0]``) as in
        `pyoptics.coatings`; rays from the plus side see it reversed.
    brdf, btdf : bsdf.BSDF, optional
        Scattering in reflection and in transmission (absolute: their TIS is
        the scattered fraction of the incident power).
    label : str, optional
    """

    def __init__(self, R=None, T=None, coating=None, brdf=None, btdf=None, label=None):
        if (R is None) != (T is None):
            R = 0.0 if R is None else R
            T = 0.0 if T is None else T
        self.R, self.T, self.coating = R, T, coating
        self.brdf, self.btdf, self.label = brdf, btdf, label
        self._tables = {}

    def __repr__(self):
        if self.label:
            return "FaceProperty({!r})".format(self.label)
        return ("FaceProperty(R={!r}, T={!r}, coating={!r}, brdf={!r}, btdf={!r})"
                .format(self.R, self.T, self.coating, self.brdf, self.btdf))

    @property
    def fresnel(self):
        return self.R is None

    def _coating_table(self, coating, n1, n2, wavelength, wavelength_nm, n=721):
        key = (id(coating), n1, n2, wavelength)
        if key not in self._tables:
            th = np.linspace(0.0, 0.5*np.pi*(1 - 1e-6), n)
            with np.errstate(invalid="ignore", divide="ignore"):
                p = coating.power(th, n1, n2, wavelength=wavelength,
                                  wavelength_nm=wavelength_nm)
            R = 0.5*(np.real(p["R_s"]) + np.real(p["R_p"]))
            T = 0.5*(np.real(p["T_s"]) + np.real(p["T_p"]))
            if n2 is not None:
                tir = (n1*np.sin(th)/n2) >= 1
                T = np.where(tir, 0.0, T)
                R = np.where(tir, 1.0 - 0.0*R, R) if not np.iscomplexobj(n2) else R
            T = np.clip(np.nan_to_num(T), 0.0, 1.0)
            R = np.clip(np.nan_to_num(R), 0.0, 1.0)
            self._tables[key] = (np.cos(th)[::-1], R[::-1], T[::-1])
        return self._tables[key]

    @staticmethod
    def _value(v, cos_i, wavelength_nm):
        if callable(v):
            return np.broadcast_to(np.asarray(v(cos_i, wavelength_nm), float), cos_i.shape)
        return np.full(cos_i.shape, float(v))

    def evaluate(self, cos_i, n1, n2, from_plus, wavelength, wavelength_nm):
        """Specular reflectance and transmittance for rays with the angle
        cosines cos_i (>= 0) from the medium n1 into n2.

        Parameters
        ----------
        cos_i : (N,) array
        n1, n2 : numbers
        from_plus : (N,) bool array
            Rays that arrive from the plus side of the face.
        wavelength, wavelength_nm : numbers
            In the units of the scene and in nm.

        Returns
        -------
        R, T : (N,) arrays
        """

        if self.R is None:
            if self.coating is None:
                return _fresnel_unpolarized(cos_i, n1, n2)
            R = np.empty(cos_i.shape)
            T = np.empty(cos_i.shape)
            for side in (False, True):
                m = from_plus == side
                if not np.any(m):
                    continue
                c = self.coating.reversed() if side else self.coating
                if side:
                    key_c = getattr(self, "_rev", None)
                    if key_c is None:
                        self._rev = key_c = self.coating.reversed()
                    c = key_c
                cg, Rg, Tg = self._coating_table(c, n1, n2, wavelength, wavelength_nm)
                R[m] = np.interp(cos_i[m], cg, Rg)
                T[m] = np.interp(cos_i[m], cg, Tg)
            return R, T
        if hasattr(self.R, "layers"):  # a mirror coating on its substrate
            cg, Rg, _ = self._coating_table(self.R, n1, None, wavelength, wavelength_nm)
            R = np.interp(cos_i, cg, Rg)
        else:
            R = self._value(self.R, cos_i, wavelength_nm)
        return R, self._value(self.T, cos_i, wavelength_nm)


def fresnel(coating=None, brdf=None, btdf=None):
    """Polished refracting surface: Fresnel reflection and refraction
    (unpolarized), optionally through a thin film coating and with surface
    scatter (e.g. `bsdf.ABg` for polishing residues)."""

    return FaceProperty(coating=coating, brdf=brdf, btdf=btdf, label="fresnel")


def ideal():
    """Perfect transmission without reflection (T = 1), e.g. to compare
    with a sequential trace or for transparent detectors."""

    return FaceProperty(R=0.0, T=1.0, label="ideal")


def mirror(R=1.0, brdf=None):
    """Specular mirror of reflectance R (a number, a function f(cos_i,
    wavelength_nm) or a coatings.Coating on a substrate)."""

    return FaceProperty(R=R, T=0.0, brdf=brdf, label="mirror")


def absorber():
    """Black face: everything is absorbed."""

    return FaceProperty(R=0.0, T=0.0, label="absorber")


def diffuse(rho):
    """Lambertian reflector of reflectance rho (white paint, Spectralon)."""

    return FaceProperty(R=0.0, T=0.0, brdf=_bsdf.Lambertian(rho), label="diffuse")


def ground_glass(rho=0.04, tau=0.9):
    """Ground (fine-ground, unpolished) glass surface, e.g. a lens rim:
    Lambertian scatter rho back and tau forward, the rest absorbed in the
    damaged layer (an invented, plausible model)."""

    return FaceProperty(R=0.0, T=0.0, brdf=_bsdf.Lambertian(rho), btdf=_bsdf.Lambertian(tau),
                        label="ground glass")


def black_anodized():
    """Black anodized aluminium (invented, plausible values): about 2 %
    diffuse reflectance plus a broad ABg lobe, 0.5 % specular, about 4 %
    total hemispherical reflectance at normal incidence."""

    return FaceProperty(R=0.005, T=0.0,
                        brdf=_bsdf.Lambertian(0.02) + _bsdf.ABg(A=0.002, B=0.01, g=2.0),
                        label="black anodized")


def black_paint():
    """Flat black paint (invented, plausible values): about 3 % diffuse,
    no specular part."""

    return FaceProperty(R=0.0, T=0.0, brdf=_bsdf.Lambertian(0.03), label="black paint")


def machined_metal():
    """Bare machined (turned) aluminium (invented, plausible values): 60 %
    specular reflectance plus a lobe from the tool marks with about 20 %
    TIS and a few per cent diffuse."""

    return FaceProperty(R=0.6, T=0.0,
                        brdf=_bsdf.ABg(A=0.012, B=0.004, g=2.0) + _bsdf.Lambertian(0.03),
                        label="machined metal")


# --- faces -------------------------------------------------------------------

def _roots_np(a, b, C):
    """Both roots of a t**2 + b t + C = 0 (NaN where none), arrays."""

    a, b, C = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float),
                                  np.asarray(C, float))
    with np.errstate(invalid="ignore", divide="ignore"):
        D = b*b - 4*a*C
        sq = np.sqrt(np.where(D >= 0, D, np.nan))
        q = -0.5*(b + np.where(b >= 0, 1.0, -1.0)*sq)
        lin = a == 0
        t1 = np.where(lin, -C/b, q/a)
        t2 = np.where(lin, np.nan, C/q)
    return t1, t2


def _pick(t1, t2, ok1, ok2, eps):
    """Smallest valid root > eps, inf where none."""

    t1 = np.where(ok1 & (t1 > eps), t1, np.inf)
    t2 = np.where(ok2 & (t2 > eps), t2, np.inf)
    return np.minimum(t1, t2)


class Face(object):
    """Base class of the analytic faces: a surface in a local frame.

    Parameters
    ----------
    position : (3,) array, optional
        Origin of the local frame in the scene.
    rotation : (3, 3) array, optional
        Columns: the local x, y and z axes in the scene (e.g.
        `raytrace.rotation()`); default: identity.
    property : FaceProperty, optional
        Default: `fresnel()`.
    media : (Medium, Medium), optional
        Media on the minus and plus side of the face (numbers are indices);
        None: the background medium of the scene on both sides.
    label : str, optional
    detector : bool, optional
        Record the hits on this face (`TraceResult.hits`).
    """

    kind = None  # primitive kind of the compiled kernel

    def __init__(self, position=(0.0, 0.0, 0.0), rotation=None, property=None, media=None,
                 label=None, detector=False):
        self.origin = np.array(position, dtype=float)
        self.R = np.eye(3) if rotation is None else np.array(rotation, dtype=float)
        self.property = fresnel() if property is None else property
        self.media = None if media is None else tuple(_as_medium(m) for m in media)
        self.label = label
        self.detector = detector

    def __repr__(self):
        return "{}({!r})".format(type(self).__name__, self.label)

    @property
    def params(self):
        return np.zeros(6)

    def to_local(self, P, d=None):
        """Points (and directions) in the local frame."""

        if d is None:
            return (P - self.origin) @ self.R
        return (P - self.origin) @ self.R, d @ self.R

    def to_global(self, q):
        return self.origin + q @ self.R.T

    def intersect(self, P, d, eps):
        """Distances t > eps to the nearest hits of the rays (P, d) in the
        scene frame; inf for rays that miss."""

        q, dl = self.to_local(P, d)
        return self._intersect_local(q, dl, eps)

    def normal(self, Q, sub=None):
        """Unit normals at the hit points Q (scene frame)."""

        return self._normal_local(self.to_local(Q)) @ self.R.T

    def local_box(self):
        raise NotImplementedError

    def bounds(self):
        """Axis aligned bounding box (lo, hi) in the scene frame."""

        lo, hi = self.local_box()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        g = self.to_global(corners)
        pad = 1e-9*(1 + np.max(np.abs(g)))
        return g.min(0) - pad, g.max(0) + pad

    rotational = True  # outline points (0, r, z) are a profile r, z

    def outline(self, cut=0, value=0.0, n=121):
        """Section of the face with the plane where the scene coordinate
        number ``cut`` equals ``value`` (for `plot_scene()`): list of (M, 3)
        polylines. Faces of revolution are drawn by their profile if their
        axis lies in that plane (else not at all), other faces by their
        section with their local x = 0 plane."""

        prof = self._outline_local(n)
        if not self.rotational:
            return [self.to_global(c) for c in prof]
        nc = np.zeros(3)
        nc[cut] = 1.0
        ez = self.R[:, 2]
        if abs(ez @ nc) > 1e-9 or abs(self.origin[cut] - value) > 1e-9:
            return []
        u = np.cross(nc, ez)
        u /= np.linalg.norm(u)
        return [self.origin + c[:, 1:2]*u + c[:, 2:3]*ez for c in prof]

    def _outline_local(self, n):
        return []


def _sag_range(sag, r0, r1, n=257):
    r = np.linspace(r0, r1, n)
    z = sag(r)
    return np.nanmin(z), np.nanmax(z)


class ConicFace(Face):
    """Disk or annulus of a sphere, conic or plane (no aspheric terms),
    sag z = c r**2/(1 + sqrt(1 - (1 + K) c**2 r**2)) + z_shift in its local
    frame, normal +z.

    Parameters
    ----------
    R : number, optional
        Vertex radius of curvature (np.inf: plane).
    r_max : number
        Outer radius.
    r_min : number, optional
        Inner radius (annuli).
    conic : number, optional
        Conic constant K.
    z_shift : number, optional
        Axial shift of the vertex in the local frame.
    **kwargs
        See `Face`.
    """

    kind = _nsrt_numba.CONIC

    def __init__(self, R=np.inf, r_max=1.0, r_min=0.0, conic=0.0, z_shift=0.0, **kwargs):
        super().__init__(**kwargs)
        self.c = 0.0 if np.isinf(R) else 1.0/R
        self.K, self.r_min, self.r_max, self.z_shift = float(conic), float(r_min), \
            float(r_max), float(z_shift)

    @property
    def params(self):
        return np.array([self.c, self.K, self.r_min, self.r_max, self.z_shift, 0.0])

    def sag(self, r):
        c, K = self.c, self.K
        r2 = np.asarray(r, float)**2
        with np.errstate(invalid="ignore"):
            return c*r2/(1 + np.sqrt(1 - (1 + K)*c**2*r2)) + self.z_shift

    def _intersect_local(self, q, d, eps):
        c, K = self.c, self.K
        x, y, z = q[:, 0], q[:, 1], q[:, 2] - self.z_shift
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        k1 = 1 + K
        a = c*(dx**2 + dy**2 + k1*dz**2)
        b = 2*(c*(x*dx + y*dy + k1*z*dz) - dz)
        C = c*(x**2 + y**2 + k1*z**2) - 2*z
        ts = _roots_np(a, b, C)
        oks = []
        for t in ts:
            with np.errstate(invalid="ignore"):
                qx, qy, qz = x + t*dx, y + t*dy, z + t*dz
                r2 = qx**2 + qy**2
                oks.append((k1*c*qz <= 1) & (r2 >= self.r_min**2) & (r2 <= self.r_max**2))
        return _pick(ts[0], ts[1], oks[0], oks[1], eps)

    def _normal_local(self, q):
        c, K = self.c, self.K
        r2 = q[:, 0]**2 + q[:, 1]**2
        with np.errstate(invalid="ignore"):
            s = c/np.sqrt(np.clip(1 - (1 + K)*c**2*r2, 1e-300, None))
        N = np.stack([-q[:, 0]*s, -q[:, 1]*s, np.ones(len(q))], -1)
        return N/np.linalg.norm(N, axis=-1)[:, None]

    def local_box(self):
        r1 = self.r_max
        k1 = 1 + self.K
        if self.c != 0 and k1 > 0:
            r1 = min(r1, 1/(abs(self.c)*np.sqrt(k1)))
        z0, z1 = _sag_range(self.sag, self.r_min, r1)
        return np.array([-self.r_max, -self.r_max, z0]), np.array([self.r_max, self.r_max, z1])

    def _outline_local(self, n):
        out = []
        for sgn in (-1, 1):
            r = np.linspace(self.r_min, self.r_max, n)
            z = self.sag(r)
            ok = np.isfinite(z)
            out.append(np.stack([0*r[ok], sgn*r[ok], z[ok]], -1))
        if self.r_min == 0:
            return [np.concatenate([out[0][::-1], out[1][1:]])]
        return out


def _sag_function(surface):
    if hasattr(surface, "sag_xy"):
        return lambda x, y: np.asarray(surface.sag_xy(x, y), float)
    return lambda x, y: np.asarray(surface.sag(x**2 + y**2), float)


class SagFace(Face):
    """Any surface of `pyoptics.raytrace` with a sag (aspheres, toroids,
    biconics, Zernike and Q-type freeforms, ...) inside the annulus r_min
    <= r <= r_max of its local frame. The hits are bracketed on samples
    along the ray and refined by bisection (numpy, slower than
    `ConicFace`; a ray that crosses the surface twice between two samples
    can be missed).

    Parameters
    ----------
    surface : raytrace.Surface
    r_max, r_min : numbers
    z_shift : number, optional
    n_samples : int, optional
        Samples along the part of the ray inside the bounding box.
    **kwargs
        See `Face`.
    """

    def __init__(self, surface, r_max, r_min=0.0, z_shift=0.0, n_samples=33, **kwargs):
        super().__init__(**kwargs)
        self.rotational = not hasattr(surface, "sag_xy")
        self.surface, self.r_max, self.r_min = surface, float(r_max), float(r_min)
        self.z_shift, self.n_samples = float(z_shift), n_samples
        self._sag = _sag_function(surface)

    def sag(self, r):
        r = np.asarray(r, float)
        return self._sag(r, 0*r) + self.z_shift

    def local_box(self):
        phi = np.linspace(0, 2*np.pi, 73)
        r = np.linspace(self.r_min, self.r_max, 65)
        Rr, P = np.meshgrid(r, phi)
        with np.errstate(invalid="ignore"):
            z = self._sag(Rr*np.cos(P), Rr*np.sin(P)) + self.z_shift
        z0, z1 = np.nanmin(z), np.nanmax(z)
        pad = 0.02*(z1 - z0) + 1e-9*(1 + self.r_max)
        return (np.array([-self.r_max, -self.r_max, z0 - pad]),
                np.array([self.r_max, self.r_max, z1 + pad]))

    def _G(self, q, d, t):
        p = q + t[..., None]*d
        with np.errstate(invalid="ignore"):
            return p[..., 2] - self._sag(p[..., 0], p[..., 1]) - self.z_shift

    def _intersect_local(self, q, d, eps):
        lo, hi = self.local_box()
        n = len(q)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1/d
            ta = (lo - q)*inv
            tb = (hi - q)*inv
        t0 = np.nanmax(np.minimum(ta, tb), axis=1)
        t1 = np.nanmin(np.maximum(ta, tb), axis=1)
        t0 = np.maximum(t0, eps)
        out = np.full(n, np.inf)
        m = np.nonzero(t1 > t0)[0]
        if not len(m):
            return out
        s = np.linspace(0, 1, self.n_samples)
        T = t0[m, None] + (t1[m] - t0[m])[:, None]*s[None, :]
        G = self._G(q[m, None, :], d[m, None, :], T)
        ch = (G[:, :-1]*G[:, 1:] <= 0) & np.isfinite(G[:, :-1]) & np.isfinite(G[:, 1:])
        has = np.any(ch, axis=1)
        j = np.argmax(ch, axis=1)
        rows = np.nonzero(has)[0]
        if not len(rows):
            return out
        a = T[rows, j[rows]]
        b = T[rows, j[rows] + 1]
        ga = G[rows, j[rows]]
        qq, dd = q[m[rows]], d[m[rows]]
        for _ in range(64):
            c = 0.5*(a + b)
            gc = self._G(qq, dd, c)
            left = (ga*gc <= 0)
            b = np.where(left, c, b)
            a = np.where(left, a, c)
            ga = np.where(left, ga, gc)
        t = 0.5*(a + b)
        p = qq + t[:, None]*dd
        r2 = p[:, 0]**2 + p[:, 1]**2
        ok = (r2 >= self.r_min**2) & (r2 <= self.r_max**2) & (t > eps)
        out[m[rows[ok]]] = t[ok]
        return out

    def _normal_local(self, q):
        return self.surface.normal(q - [0.0, 0.0, self.z_shift])

    def _outline_local(self, n):
        r = np.linspace(-self.r_max, self.r_max, 2*n)
        r = r[np.abs(r) >= self.r_min]
        z = self._sag(0*r, r) + self.z_shift
        ok = np.isfinite(z)
        return [np.stack([0*r[ok], r[ok], z[ok]], -1)]


class Rectangle(Face):
    """Rectangle abs(x) <= width/2, abs(y) <= height/2 in the local plane z = 0,
    normal +z.

    Parameters
    ----------
    width, height : numbers
    **kwargs
        See `Face`.
    """

    kind = _nsrt_numba.RECT

    def __init__(self, width, height=None, **kwargs):
        super().__init__(**kwargs)
        self.a = 0.5*float(width)
        self.b = 0.5*float(width if height is None else height)

    @property
    def params(self):
        return np.array([self.a, self.b, 0, 0, 0, 0.0])

    def _intersect_local(self, q, d, eps):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -q[:, 2]/d[:, 2]
            ok = (np.abs(q[:, 0] + t*d[:, 0]) <= self.a) & (np.abs(q[:, 1] + t*d[:, 1]) <= self.b)
        return np.where(ok & (t > eps), t, np.inf)

    def _normal_local(self, q):
        return np.tile([0.0, 0.0, 1.0], (len(q), 1))

    def local_box(self):
        return np.array([-self.a, -self.b, 0.0]), np.array([self.a, self.b, 0.0])

    def outline(self, cut=0, value=0.0, n=121):
        c = self.to_global(np.array([[-self.a, -self.b, 0], [self.a, -self.b, 0],
                                     [self.a, self.b, 0], [-self.a, self.b, 0.0]]))
        s = c[:, cut] - value
        pts = []
        for i in range(4):
            j = (i + 1) % 4
            if s[i] == 0:
                pts.append(c[i])
            elif s[i]*s[j] < 0:
                pts.append(c[i] + s[i]/(s[i] - s[j])*(c[j] - c[i]))
        return [np.array(pts[:2])] if len(pts) >= 2 else []


class Cylinder(Face):
    """Lateral surface of the cylinder x**2 + y**2 = radius**2, z0 <= z <= z1,
    normal pointing away from the axis.

    Parameters
    ----------
    radius, z0, z1 : numbers
    **kwargs
        See `Face`; media: (inside, outside).
    """

    kind = _nsrt_numba.CYLINDER

    def __init__(self, radius, z0, z1, **kwargs):
        super().__init__(**kwargs)
        self.radius, self.z0, self.z1 = float(radius), float(min(z0, z1)), float(max(z0, z1))

    @property
    def params(self):
        return np.array([self.radius, self.z0, self.z1, 0, 0, 0.0])

    def _intersect_local(self, q, d, eps):
        a = d[:, 0]**2 + d[:, 1]**2
        b = 2*(q[:, 0]*d[:, 0] + q[:, 1]*d[:, 1])
        C = q[:, 0]**2 + q[:, 1]**2 - self.radius**2
        ts = _roots_np(a, b, C)
        with np.errstate(invalid="ignore"):
            oks = [((q[:, 2] + t*d[:, 2]) >= self.z0) & ((q[:, 2] + t*d[:, 2]) <= self.z1)
                   for t in ts]
        return _pick(ts[0], ts[1], oks[0], oks[1], eps)

    def _normal_local(self, q):
        N = np.stack([q[:, 0], q[:, 1], 0*q[:, 0]], -1)
        return N/np.linalg.norm(N, axis=-1)[:, None]

    def local_box(self):
        r = self.radius
        return np.array([-r, -r, self.z0]), np.array([r, r, self.z1])

    def _outline_local(self, n):
        return [np.array([[0, s*self.radius, self.z0], [0, s*self.radius, self.z1]])
                for s in (-1.0, 1.0)]


class Cone(Face):
    """Lateral surface of a cone frustum from radius r0 at z0 to r1 at z1
    (z0 != z1), normal pointing away from the axis.

    Parameters
    ----------
    r0, z0, r1, z1 : numbers
    **kwargs
        See `Face`; media: (inside, outside).
    """

    kind = _nsrt_numba.CONE

    def __init__(self, r0, z0, r1, z1, **kwargs):
        super().__init__(**kwargs)
        if z0 == z1:
            raise ValueError("A cone needs z0 != z1 (use an annulus, ConicFace).")
        self.r0, self.z0, self.r1, self.z1 = float(r0), float(z0), float(r1), float(z1)

    @property
    def params(self):
        return np.array([self.r0, self.z0, self.r1, self.z1, 0, 0.0])

    def _intersect_local(self, q, d, eps):
        k = (self.r1 - self.r0)/(self.z1 - self.z0)
        u = self.r0 - k*self.z0
        w = u + k*q[:, 2]
        a = d[:, 0]**2 + d[:, 1]**2 - k**2*d[:, 2]**2
        b = 2*(q[:, 0]*d[:, 0] + q[:, 1]*d[:, 1] - k*d[:, 2]*w)
        C = q[:, 0]**2 + q[:, 1]**2 - w**2
        ts = _roots_np(a, b, C)
        lo, hi = min(self.z0, self.z1), max(self.z0, self.z1)
        oks = []
        for t in ts:
            with np.errstate(invalid="ignore"):
                z = q[:, 2] + t*d[:, 2]
                oks.append((z >= lo) & (z <= hi) & (u + k*z >= 0))
        return _pick(ts[0], ts[1], oks[0], oks[1], eps)

    def _normal_local(self, q):
        k = (self.r1 - self.r0)/(self.z1 - self.z0)
        w = self.r0 - k*self.z0 + k*q[:, 2]
        N = np.stack([q[:, 0], q[:, 1], -k*w], -1)
        return N/np.linalg.norm(N, axis=-1)[:, None]

    def local_box(self):
        r = max(self.r0, self.r1)
        return (np.array([-r, -r, min(self.z0, self.z1)]),
                np.array([r, r, max(self.z0, self.z1)]))

    def _outline_local(self, n):
        return [np.array([[0, s*self.r0, self.z0], [0, s*self.r1, self.z1]]) for s in (-1.0, 1.0)]


class Sphere(Face):
    """Full sphere of the given radius about the local origin, normal
    outwards (ball lenses, integrating spheres).

    Parameters
    ----------
    radius : number
    **kwargs
        See `Face`; media: (inside, outside).
    """

    kind = _nsrt_numba.SPHERE

    def __init__(self, radius, **kwargs):
        super().__init__(**kwargs)
        self.radius = float(radius)

    @property
    def params(self):
        return np.array([self.radius, 0, 0, 0, 0, 0.0])

    def _intersect_local(self, q, d, eps):
        b = 2*np.sum(q*d, -1)
        C = np.sum(q*q, -1) - self.radius**2
        ts = _roots_np(np.ones(len(q)), b, C)
        ok = np.ones(len(q), bool)
        return _pick(ts[0], ts[1], ok, ok, eps)

    def _normal_local(self, q):
        return q/np.linalg.norm(q, axis=-1)[:, None]

    def local_box(self):
        r = self.radius
        return np.array([-r, -r, -r]), np.array([r, r, r])

    def _outline_local(self, n):
        a = np.linspace(0, 2*np.pi, 2*n)
        return [np.stack([0*a, self.radius*np.cos(a), self.radius*np.sin(a)], -1)]


def box(size, position=(0.0, 0.0, 0.0), rotation=None, property=None, media=None,
        label="box", detector=False):
    """Six rectangles of a cuboid, normals outwards.

    Parameters
    ----------
    size : (3,) sequence
        Edge lengths along the local x, y and z axes.
    position : (3,) array, optional
        Centre.
    rotation, property, detector : optional
        See `Face`.
    media : (inside, outside), optional
    label : str, optional
        The faces are labelled label + ".+x", ".-x", ...

    Returns
    -------
    list of Rectangle
    """

    half = 0.5*np.asarray(size, float)
    Rb = np.eye(3) if rotation is None else np.asarray(rotation, float)
    o = np.asarray(position, float)
    out = []
    for axis in range(3):
        ax_x, ax_y = (axis + 1) % 3, (axis + 2) % 3
        for sgn in (1.0, -1.0):
            z = np.zeros(3)
            z[axis] = sgn
            x = np.zeros(3)
            x[ax_x] = 1.0
            Rl = np.stack([x, np.cross(z, x), z], -1)
            out.append(Rectangle(2*half[ax_x], 2*half[ax_y], position=o + Rb @ (z*half[axis]),
                                 rotation=Rb @ Rl, property=property, media=media,
                                 detector=detector,
                                 label="{}.{}{}".format(label, "+" if sgn > 0 else "-",
                                                        "xyz"[axis])))
    return out


# --- meshes ------------------------------------------------------------------

def _build_bvh(lo_t, hi_t, leaf_size=4):
    """Bounding volume hierarchy over boxes (lo_t, hi_t) of triangles:
    median splits along the longest axis of the centroids."""

    n = len(lo_t)
    cen = 0.5*(lo_t + hi_t)
    order = np.arange(n)
    node_lo, node_hi, node_a, node_b, leaf = [], [], [], [], []
    stack = [(0, n, -1, 0)]  # start, end, parent, which child
    while stack:
        s, e, parent, which = stack.pop()
        idx = order[s:e]
        k = len(node_lo)
        node_lo.append(lo_t[idx].min(0))
        node_hi.append(hi_t[idx].max(0))
        node_a.append(0)
        node_b.append(0)
        leaf.append(False)
        if parent >= 0:
            if which == 0:
                node_a[parent] = k
            else:
                node_b[parent] = k
        if e - s <= leaf_size:
            node_a[k], node_b[k], leaf[k] = s, e, True
            continue
        c = cen[idx]
        axis = int(np.argmax(c.max(0) - c.min(0)))
        sub = np.argsort(c[:, axis], kind="stable")
        order[s:e] = idx[sub]
        m = s + (e - s)//2
        stack.append((m, e, k, 1))
        stack.append((s, m, k, 0))
    return (np.array(node_lo), np.array(node_hi), np.array(node_a, dtype=np.int64),
            np.array(node_b, dtype=np.int64), np.array(leaf, dtype=np.bool_), order)


class MeshGroup(object):
    """The part of a `Mesh` that acts as one face (one property, one pair
    of media, one label); the triangles of a CAD solid, say."""

    def __init__(self, mesh, index, label, property, media, detector):
        self.mesh, self.index, self.label = mesh, index, label
        self.property = property if property is not None else absorber()
        self.media = None if media is None else tuple(_as_medium(m) for m in media)
        self.detector = detector

    def __repr__(self):
        return "MeshGroup({!r})".format(self.label)

    def normal(self, Q, tri):
        return self.mesh.normals[tri]


class Mesh(object):
    """Triangle mesh in the scene frame, made of one or more groups (each a
    face of the scene). Triangles should be wound counterclockwise seen
    from outside (normals outwards, as build123d tessellates solids), so
    that media = (inside, outside).

    Parameters
    ----------
    vertices : (V, 3) array
    triangles : (T, 3) int array
    groups : (T,) int array, optional
        Group index of every triangle (default: all 0).
    labels : sequence of str, optional
    properties : FaceProperty or list per group, optional
        Default `absorber()`.
    media : (inside, outside) tuple or list of them per group, optional
    detector : bool or list per group, optional
    """

    def __init__(self, vertices, triangles, groups=None, labels=None, properties=None,
                 media=None, detector=None):
        V = np.asarray(vertices, float)
        T = np.asarray(triangles, np.int64)
        self.vertices, self.triangles = V, T
        self.groups = np.zeros(len(T), np.int64) if groups is None else np.asarray(groups,
                                                                                  np.int64)
        n_g = int(self.groups.max()) + 1 if len(T) else 1
        labels = labels or ["mesh{}".format(k) for k in range(n_g)]
        properties = properties if isinstance(properties, (list, tuple)) else [properties]*n_g
        media = media if isinstance(media, list) else [media]*n_g
        detector = detector if isinstance(detector, (list, tuple)) else [bool(detector)]*n_g
        self.faces = [MeshGroup(self, k, labels[k], properties[k], media[k], detector[k])
                      for k in range(n_g)]
        self.V0 = V[T[:, 0]]
        self.E1 = V[T[:, 1]] - self.V0
        self.E2 = V[T[:, 2]] - self.V0
        N = np.cross(self.E1, self.E2)
        self.normals = N/np.maximum(np.linalg.norm(N, axis=-1), 1e-300)[:, None]
        tri = V[T]
        self.bvh = _build_bvh(tri.min(1), tri.max(1))

    def __repr__(self):
        return "Mesh({} triangles, {})".format(len(self.triangles),
                                              [f.label for f in self.faces])

    def bounds(self):
        return self.vertices.min(0), self.vertices.max(0)

    def intersect(self, P, d, eps):
        """Nearest hits (t, triangle index), inf and -1 for misses."""

        if use_numba and _nsrt_numba.available:
            lo, hi, a, b, leaf, order = self.bvh
            return _nsrt_numba.bvh_nearest(np.ascontiguousarray(P), np.ascontiguousarray(d),
                                           self.V0, self.E1, self.E2, lo, hi, a, b, leaf,
                                           order, eps)
        return self._intersect_numpy(P, d, eps)

    def _intersect_numpy(self, P, d, eps, chunk=256):
        n = len(P)
        best = np.full(n, np.inf)
        bt = np.full(n, -1, np.int64)
        for s in range(0, len(self.V0), chunk):
            V0, E1, E2 = self.V0[s:s + chunk], self.E1[s:s + chunk], self.E2[s:s + chunk]
            h = np.cross(d[:, None, :], E2[None, :, :])
            a = np.sum(E1[None]*h, -1)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = 1/a
                S = P[:, None, :] - V0[None]
                u = f*np.sum(S*h, -1)
                Qv = np.cross(S, E1[None])
                v = f*np.sum(d[:, None, :]*Qv, -1)
                t = f*np.sum(E2[None]*Qv, -1)
            ok = (a != 0) & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > eps)
            t = np.where(ok, t, np.inf)
            j = np.argmin(t, axis=1)
            tj = t[np.arange(n), j]
            better = tj < best
            best[better] = tj[better]
            bt[better] = s + j[better]
        return best, bt

    def section(self, axis=0, value=0.0):
        """Segments (K, 2, 3) where the plane coordinate[axis] = value cuts
        the triangles."""

        tri = self.vertices[self.triangles]
        s = tri[:, :, axis] - value
        segs = []
        for (i, j) in ((0, 1), (1, 2), (2, 0)):
            a, b = s[:, i], s[:, j]
            cross = (a*b < 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                f = a/(a - b)
                p = tri[:, i] + f[:, None]*(tri[:, j] - tri[:, i])
            segs.append(np.where(cross[:, None], p, np.nan))
        pts = np.stack(segs, 1)  # (T, 3 edges, 3)
        out = []
        for k in range(len(pts)):
            q = pts[k][np.all(np.isfinite(pts[k]), axis=-1)]
            if len(q) == 2:
                out.append(q)
        return np.array(out).reshape(-1, 2, 3)


def read_stl(filename):
    """Vertices (V, 3) and triangles (T, 3) of an STL file (binary or ASCII).

    Parameters
    ----------
    filename : str

    Returns
    -------
    vertices, triangles : arrays
    """

    with open(filename, "rb") as f:
        data = f.read()
    tri = None
    if len(data) >= 84:
        n = int(np.frombuffer(data[80:84], "<u4")[0])
        if 84 + 50*n == len(data):
            rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                           ("a", "<u2")]), count=n)
            tri = rec["v"].astype(float)
    if tri is None:
        pts = [list(map(float, line.split()[1:4])) for line in data.decode().splitlines()
               if line.strip().startswith("vertex")]
        tri = np.array(pts).reshape(-1, 3, 3)
    V, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    return V, inv.reshape(-1, 3)


# --- rays, sources, targets --------------------------------------------------

RaySet = namedtuple("RaySet", ["points", "directions", "weights"])
RaySet.__doc__ = """Rays to trace: start points and unit directions (N, 3) in
the scene frame and their powers (N,)."""


def _concat(sets):
    return RaySet(*(np.concatenate([getattr(s, f) for s in sets]) for f in RaySet._fields))


def _orient(d, axis):
    return d @ _frame_to(axis).T


def point_source(position, n_rays, power=1.0, axis=(0.0, 0.0, 1.0), half_angle=np.pi,
                 rng=None):
    """Isotropic point source (uniform per solid angle) into a cone.

    Parameters
    ----------
    position : (3,) array
    n_rays : int
    power : number, optional
        Power into the cone (shared equally by the rays).
    axis : (3,) array, optional
    half_angle : number, optional
        Cone half angle (radians); pi: the full sphere.
    rng : None, int or numpy.random.Generator, optional

    Returns
    -------
    RaySet
    """

    rng = _rng(rng)
    u = rng.random((n_rays, 2))
    cos_t = 1 - u[:, 0]*(1 - np.cos(half_angle))
    sin_t = np.sqrt(np.clip(1 - cos_t**2, 0, None))
    phi = 2*np.pi*u[:, 1]
    d = np.stack([sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t], -1)
    return RaySet(np.tile(np.asarray(position, float), (n_rays, 1)), _orient(d, axis),
                  np.full(n_rays, power/n_rays))


def collimated_beam(center, direction, radius, n_rays, power=1.0, shape="disk", rng=None):
    """Collimated beam with a uniform disk (or square of side 2 radius)
    cross section.

    Parameters
    ----------
    center, direction : (3,) arrays
    radius : number
    n_rays : int
    power : number, optional
    shape : {"disk", "square"}, optional
    rng : None, int or numpy.random.Generator, optional

    Returns
    -------
    RaySet
    """

    rng = _rng(rng)
    if shape == "disk":
        r = radius*np.sqrt(rng.random(n_rays))
        phi = 2*np.pi*rng.random(n_rays)
        xy = np.stack([r*np.cos(phi), r*np.sin(phi)], -1)
    else:
        xy = radius*(2*rng.random((n_rays, 2)) - 1)
    F = _frame_to(direction)
    P = np.asarray(center, float) + xy[:, :1]*F[:, 0] + xy[:, 1:]*F[:, 1]
    return RaySet(P, np.tile(F[:, 2], (n_rays, 1)), np.full(n_rays, power/n_rays))


def lambertian_source(center, normal, radius, n_rays, power=1.0, m=1.0, rng=None):
    """Disk emitter with the intensity cos(theta)**m (m = 1: Lambertian),
    e.g. an LED die.

    Parameters
    ----------
    center, normal : (3,) arrays
    radius : number
        Radius of the emitting disk (0: a point).
    n_rays : int
    power : number, optional
    m : number, optional
    rng : None, int or numpy.random.Generator, optional

    Returns
    -------
    RaySet
    """

    rng = _rng(rng)
    u = rng.random((n_rays, 2))
    cos_t = (1 - u[:, 0])**(1/(m + 1.0))
    sin_t = np.sqrt(np.clip(1 - cos_t**2, 0, None))
    phi = 2*np.pi*u[:, 1]
    d = np.stack([sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t], -1)
    r = radius*np.sqrt(rng.random(n_rays))
    a = 2*np.pi*rng.random(n_rays)
    F = _frame_to(normal)
    P = np.asarray(center, float) + (r*np.cos(a))[:, None]*F[:, 0] + (r*np.sin(a))[:, None]*F[:, 1]
    return RaySet(P, d @ F.T, np.full(n_rays, power/n_rays))


def system_source(system, field=None, n_rays=1000, power=1.0, fill=1.0, rng=None,
                  aim=False):
    """Rays from an object point of a `raytrace.System` through random
    points of its entrance pupil (uniform over the disk of normalized
    radius fill; fill > 1 overfills the pupil, e.g. for stray light from
    an illumination of larger NA).

    Parameters
    ----------
    system : raytrace.System
    field : number or pair, optional
        As in `System.rays()` (default: the maximum field).
    n_rays : int
    power : number, optional
    fill : number, optional
    rng : None, int or numpy.random.Generator, optional
    aim : bool, optional
        Real ray aiming at the stop (slower, only for fill <= 1).

    Returns
    -------
    RaySet
    """

    rng = _rng(rng)
    r = fill*np.sqrt(rng.random(n_rays))
    phi = 2*np.pi*rng.random(n_rays)
    rays = system.rays(field, r*np.cos(phi), r*np.sin(phi), aim=aim)
    return RaySet(np.asarray(rays.points, float), np.asarray(rays.directions, float),
                  np.full(n_rays, power/n_rays))


class Target(object):
    """Disk that scattered rays are aimed at (importance sampling).

    Parameters
    ----------
    center, normal : (3,) arrays
    radius : number
    """

    def __init__(self, center, normal, radius):
        self.center = np.asarray(center, float)
        self.normal = _unit(normal)
        self.radius = float(radius)
        F = _frame_to(self.normal)
        self.e1, self.e2 = F[:, 0], F[:, 1]

    def __repr__(self):
        return "Target({}, {}, {})".format(self.center, self.normal, self.radius)

    @classmethod
    def from_face(cls, face):
        """The disk of a ConicFace, SagFace or Rectangle (its outer radius,
        or half the diagonal), at its vertex plane."""

        if isinstance(face, Rectangle):
            r = np.hypot(face.a, face.b)
            z = 0.0
        else:
            r = face.r_max
            z = float(face.sag(np.array([face.r_max]))[0]) if hasattr(face, "sag") else 0.0
            z = 0.5*(z + face.z_shift) if np.isfinite(z) else face.z_shift
        return cls(face.to_global(np.array([[0.0, 0.0, z]]))[0], face.R[:, 2], r)

    def hit(self, Q, d):
        """Whether the rays (Q, d) cross the disk, and the solid angle
        density dist**2/(area abs(cos)) at the crossing."""

        with np.errstate(divide="ignore", invalid="ignore"):
            cos = d @ self.normal
            t = ((self.center - Q) @ self.normal)/cos
            X = Q + t[:, None]*d
            r2 = np.sum((X - self.center)**2, -1)
            ok = (t > 0) & (r2 <= self.radius**2)
            dens = t**2/(np.pi*self.radius**2*np.abs(cos))
        return ok, np.where(ok, dens, 0.0)

    def sample(self, Q, rng):
        n = len(Q)
        r = self.radius*np.sqrt(rng.random(n))
        a = 2*np.pi*rng.random(n)
        X = self.center + (r*np.cos(a))[:, None]*self.e1 + (r*np.sin(a))[:, None]*self.e2
        return _unit(X - Q)


# --- results -----------------------------------------------------------------

class Hits(object):
    """Rays recorded on a detector face.

    Attributes
    ----------
    face : face of the scene
    points : (N, 3) array
        Hit points in the local frame of the face (scene frame for meshes).
    directions : (N, 3) array
        Directions of the arriving rays (scene frame).
    weights : (N,) array
        Powers of the rays.
    history : (N, max_events) int array
        Codes 4*face + event of the interactions before the hit (-1 unused).
    n_events : (N,) array
    """

    def __init__(self, scene, face, points, directions, weights, history, n_events):
        self.scene, self.face = scene, face
        self.points, self.directions, self.weights = points, directions, weights
        self.history, self.n_events = history, n_events

    def __repr__(self):
        return "Hits({!r}, {} rays, power {:.4g})".format(self.face.label, len(self.weights),
                                                         self.power)

    def __len__(self):
        return len(self.weights)

    @property
    def power(self):
        return float(np.sum(self.weights))

    def _events(self):
        h = self.history
        return np.where(h >= 0, h % 4, -1), np.where(h >= 0, h//4, -1)

    def count(self, events="R"):
        """Number of events of the given kinds ("T", "R", "S", "s" or a
        combination such as "RS") along every ray."""

        ev, _ = self._events()
        codes = [EVENTS.index(e) for e in events]
        return np.sum(np.isin(ev, codes), axis=1)

    def mask(self, reflections=None, scatters=None, faces=None, exclude_faces=None,
             events="RSs"):
        """Rays with a given history.

        Parameters
        ----------
        reflections : int or (min, max), optional
            Number of specular reflections.
        scatters : int or (min, max), optional
            Number of scatter events (in reflection or transmission).
        faces : sequence of str, optional
            Rays with at least one event of the kinds ``events`` on one of
            these faces (labels, or label prefixes ending in "*").
        exclude_faces : sequence of str, optional
            Rays without such events on these faces.
        events : str, optional

        Returns
        -------
        (N,) bool array
        """

        m = np.ones(len(self), bool)

        def within(x, v):
            lo, hi = (v, v) if np.isscalar(v) else v
            return (x >= lo) & (x <= hi)

        if reflections is not None:
            m &= within(self.count("R"), reflections)
        if scatters is not None:
            m &= within(self.count("Ss"), scatters)
        if faces is not None or exclude_faces is not None:
            ev, fc = self._events()
            codes = [EVENTS.index(e) for e in events]
            relevant = np.isin(ev, codes)

            def on(names):
                ids = self.scene.face_ids(names)
                return np.any(relevant & np.isin(fc, ids), axis=1)

            if faces is not None:
                m &= on(faces)
            if exclude_faces is not None:
                m &= ~on(exclude_faces)
        return m

    def select(self, mask):
        """The hits of the rays in the mask."""

        return Hits(self.scene, self.face, self.points[mask], self.directions[mask],
                    self.weights[mask], self.history[mask], self.n_events[mask])

    def signatures(self, events="RSs"):
        """Per ray, the tuple of its events of the given kinds (default: the
        reflections and scatters, not the refractions) as history codes."""

        ev, _ = self._events()
        codes = [EVENTS.index(e) for e in events]
        keep = np.isin(ev, codes)
        return [tuple(h[k]) for h, k in zip(self.history, keep)]

    def paths(self, top=10, events="RSs"):
        """The strongest paths: rays grouped by their sequence of
        reflections and scatters, ranked by power.

        Parameters
        ----------
        top : int, optional
        events : str, optional

        Returns
        -------
        list of (power, number of rays, description)
        """

        sig = self.signatures(events)
        powers = {}
        counts = {}
        for s, w in zip(sig, self.weights):
            powers[s] = powers.get(s, 0.0) + w
            counts[s] = counts.get(s, 0) + 1
        ranked = sorted(powers, key=lambda s: -powers[s])[:top]
        return [(powers[s], counts[s], self.scene.describe(s)) for s in ranked]

    def irradiance(self, bins=64, ranges=None, mask=None, axes=(0, 1)):
        """Power per area binned over the face (local coordinates).

        Parameters
        ----------
        bins : int or (int, int), optional
        ranges : ((x0, x1), (y0, y1)), optional
            Default: the extent of the face (rectangles) or of the hits.
        mask : (N,) bool array, optional
        axes : (int, int), optional
            Coordinates binned (default local x and y).

        Returns
        -------
        E : (nx, ny) array
            Irradiance (power per area), E[i, j] at x-bin i, y-bin j.
        x_edges, y_edges : arrays
        """

        p = self.points if mask is None else self.points[mask]
        w = self.weights if mask is None else self.weights[mask]
        if ranges is None:
            if isinstance(self.face, Rectangle):
                ranges = ((-self.face.a, self.face.a), (-self.face.b, self.face.b))
            elif len(p):
                ranges = tuple((p[:, a].min(), p[:, a].max()) for a in axes)
            else:
                ranges = ((-1, 1), (-1, 1))
        H, xe, ye = np.histogram2d(p[:, axes[0]], p[:, axes[1]], bins=bins, range=ranges,
                                   weights=w)
        area = np.outer(np.diff(xe), np.diff(ye))
        return H/area, xe, ye


class TraceResult(object):
    """Result of `Scene.trace()`.

    Attributes
    ----------
    source_power : number
    absorbed : (F,) array
        Power absorbed on every face of the scene (and in the medium in
        front of it, booked as ``bulk``).
    bulk : (F,) array
        Power absorbed in the media on the way to every face.
    escaped : number
        Power of the rays that left the scene.
    lost : number
        Power of the rays stopped after max_events interactions,
        max_reflections, max_scatters or max_generations steps.
    hits : dict
        `Hits` per detector label.
    segments : list
        With record=True: per step, (start, end, weight, face of the start)
        of the traced segments.
    """

    def __init__(self, scene):
        self.scene = scene
        F = len(scene.faces)
        self.absorbed = np.zeros(F)
        self.bulk = np.zeros(F)
        self.escaped = 0.0
        self.lost = 0.0
        self.source_power = 0.0
        self.roulette = 0.0
        self.hits = {}
        self.segments = []
        self.generations = 0
        self.n_rays_traced = 0

    def __repr__(self):
        return ("TraceResult(source {:.4g}, absorbed {:.4g}, escaped {:.4g}, lost {:.4g}, "
                "{} detectors)".format(self.source_power, self.absorbed.sum() + self.bulk.sum(),
                                       self.escaped, self.lost, len(self.hits)))

    def absorbed_by(self, names):
        """Power absorbed on the faces with these labels (or label prefixes
        ending in "*")."""

        return float(np.sum(self.absorbed[self.scene.face_ids(names)]))

    def balance(self):
        """Where the power went: absorbed on faces, in bulk, escaped, lost,
        and the total relative to the source (1 up to roulette noise)."""

        a, b = self.absorbed.sum(), self.bulk.sum()
        return {"absorbed": a, "bulk": b, "escaped": self.escaped, "lost": self.lost,
                "total": (a + b + self.escaped + self.lost)/self.source_power}

    def summary(self, top=10):
        """Faces ranked by absorbed power (list of (label, power))."""

        k = np.argsort(-self.absorbed)[:top]
        return [(self.scene.faces[i].label, float(self.absorbed[i])) for i in k]


# --- scene -------------------------------------------------------------------

class Scene(object):
    """Faces and meshes in one frame, traced non-sequentially.

    Parameters
    ----------
    wavelength : number
        Vacuum wavelength in the units of the scene.
    units : str, optional
        Length unit (keys of `raytrace.NM_PER_UNIT`).
    background : Medium or number, optional
        Medium of faces without media (default air, n = 1).
    eps : number, optional
        Smallest distance to the next hit (avoids hitting the face a ray
        starts on again).
    """

    def __init__(self, wavelength=550e-6, units="mm", background=AIR, eps=1e-7):
        self.wavelength, self.units = wavelength, units
        self.background = _as_medium(background)
        self.eps = eps
        self.faces = []      # every face: analytic faces and mesh groups
        self.primitives = []  # face ids of analytic primitives
        self.generic = []     # face ids of other analytic faces (SagFace)
        self.meshes = []      # (mesh, first face id)
        self._packed = None

    def __repr__(self):
        return "Scene({} faces, {} meshes)".format(len(self.faces), len(self.meshes))

    @property
    def wavelength_nm(self):
        return self.wavelength*rt.NM_PER_UNIT[self.units]

    # building:
    def add(self, *items):
        """Add faces, lists of faces or meshes; returns them."""

        for item in items:
            if isinstance(item, (list, tuple)):
                self.add(*item)
            elif isinstance(item, Mesh):
                self.meshes.append((item, len(self.faces)))
                self.faces.extend(item.faces)
            elif isinstance(item, Face):
                k = len(self.faces)
                self.faces.append(item)
                (self.primitives if item.kind is not None else self.generic).append(k)
            else:
                raise TypeError("Cannot add {!r}.".format(item))
        self._packed = None
        return items[0] if len(items) == 1 else items

    def face_ids(self, names):
        """Indices of the faces with these labels (or label prefixes ending
        in "*")."""

        names = [names] if isinstance(names, str) else list(names)
        out = []
        for k, f in enumerate(self.faces):
            lab = f.label or ""
            for nm in names:
                if (nm.endswith("*") and lab.startswith(nm[:-1])) or lab == nm:
                    out.append(k)
                    break
        return np.array(out, dtype=np.int64)

    def face(self, label):
        """The face with this label."""

        ids = self.face_ids(label)
        if not len(ids):
            raise KeyError(label)
        return self.faces[ids[0]]

    def describe(self, codes):
        """Readable form of history codes, e.g. "L3:S5 R -> barrel S"."""

        if not len(codes):
            return "direct"
        return " -> ".join("{} {}".format(self.faces[c//4].label, EVENTS[c % 4]) for c in codes)

    def bounds(self):
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for k in self.primitives + self.generic:
            a, b = self.faces[k].bounds()
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
        for mesh, _ in self.meshes:
            a, b = mesh.bounds()
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
        return lo, hi

    def add_system(self, system, mechanics=None, coatings=None, property=None, land=None,
                   bevel=None, rim=None, image=True, image_size=None, mount=None,
                   mount_property=None, mount_per_face=False, fields=None, label="L",
                   tolerance=0.002):
        """Add the lenses of a coaxial `raytrace.System` as analytic faces.

        Every element (`System.elements()`) gets its optical surfaces up to
        the clear apertures (labels "L3:S5" for surface index 4 in lens 3,
        surfaces counted from 1), the lands from there to the bevels or the
        rim ("L3:S5 land"), 45 degree bevels ("L3:S5 bevel") and the
        cylindrical rims ("L3 rim"), with the mechanical data of
        `cad.Mechanics` as in `cad.system_model()`. Normals point along +z,
        the media are those of the system. Clear apertures: the semi
        diameters, or for surfaces without one the footprints of the
        traced fields.

        Parameters
        ----------
        system : raytrace.System
            Coaxial, rotationally symmetric surfaces, no mirrors.
        mechanics : cad.Mechanics or dict, optional
            Per element index or one for all.
        coatings : dict {surface index: coatings.Coating}, optional
            Coatings of the optical surfaces and their lands (e.g. from
            `photography.ar_coatings()`).
        property, land, bevel, rim : FaceProperty, optional
            Optical surfaces (default `fresnel()` with the coating), lands
            (default: as the optical surface), bevels and rims (default
            `ground_glass()`).
        image : bool, optional
            Add the image plane as a square absorbing detector "image".
        image_size : number, optional
            Its side (default: 2.4 times the largest traced image height).
        mount : cad.Mount, optional
            Add barrel, rings, spacers and stop plate of
            `cad.system_model()` as a mesh (needs build123d).
        mount_property : FaceProperty, dict or callable, optional
            Properties of the mount parts by label ("barrel", "ring_front",
            "ring_back", "spacer_1", ..., "stop"), as in `add_shapes()`.
            Default `black_anodized()`.
        mount_per_face : bool, optional
            One face of the scene per B-rep face of the mount parts
            ("spacer_3.f2", ...), see `add_shapes()`.
        fields : sequence, optional
            Fields of the traces for the clear apertures (as in `cad`).
        label : str, optional
            Prefix of the element labels.
        tolerance : number, optional
            Tessellation tolerance of the mount (mm).

        Returns
        -------
        list
            The added faces.
        """

        from . import cad

        if system.tilted or any(s.mirror for s in system.surfaces):
            raise NotImplementedError("add_system() takes coaxial lens systems without mirrors.")
        if mechanics is None or isinstance(mechanics, cad.Mechanics):
            default = mechanics or cad.Mechanics()
            mechanics = {}
        else:
            default = cad.Mechanics()
        n = system.n
        frames, (o_img, R_img) = system.frames()
        surfaces = system.surfaces
        coatings = coatings or {}
        traced = cad._trace(system, fields, 7, "y", None)
        r_hit = np.zeros(len(surfaces))
        for b in traced:
            for i in range(len(surfaces)):
                o, R = frames[i]
                q = (b.points[i + 1] - o) @ R
                r = np.hypot(q[:, 0], q[:, 1])
                if np.any(np.isfinite(r)):
                    r_hit[i] = max(r_hit[i], np.nanmax(r))

        def clear(i):
            s = surfaces[i]
            return s.semi_diameter if s.semi_diameter is not None else r_hit[i]

        # media behind every surface:
        media = [Medium(system.n_object)]
        for s in surfaces:
            if s.material is not None:
                media.append(Medium(s.material))
            elif s.n_after is not None:
                media.append(Medium(s.n_after))
            else:
                media.append(media[-1])
        gap = [system._is_gap(x) for x in n]
        added = []

        def face_for(s, r0, r1, zs, frame, prop, lab, md):
            o, R = frame
            plain = type(s) in (rt.Surface,) and not s.aspheric
            kw = dict(position=o, rotation=R, property=prop, media=md, label=lab)
            if plain:
                return ConicFace(s.R, r1, r0, s.conic, zs, **kw)
            return SagFace(s, r1, r0, zs, **kw)

        for k, (first, last) in enumerate(system.elements()):
            if first == last:
                continue
            mech = mechanics.get(k, default)
            idx = list(range(first, last + 1))
            r_clear = max(clear(i) for i in idx)
            rm = mech.semi_diameter if mech.semi_diameter is not None else r_clear + mech.margin
            b1, b2 = mech.bevels()
            name = "{}{}".format(label, k + 1)
            edge_z = {}  # surface index -> local z of the end of its land/bevel at the rim
            for i in idx:
                s = surfaces[i]
                if not cad._is_rotational(s):
                    raise NotImplementedError("add_system() needs rotationally symmetric "
                                              "surfaces ({}).".format(s))
                md = (media[i], media[i + 1])
                coat = coatings.get(i)
                p_opt = property if property is not None else fresnel(coating=coat)
                p_land = land if land is not None else p_opt
                p_bev = bevel if bevel is not None else ground_glass()
                rc = clear(i)
                lab = "{}:S{}".format(name, i + 1)
                f = face_for(s, 0.0, rc, 0.0, frames[i], p_opt, lab, md)
                added.append(f)
                # bevel only where the surface faces air (gap in front or behind):
                faces_air = gap[i] or gap[i + 1]
                b = (b1 if gap[i] else b2) if faces_air else 0.0
                g = 1.0 if gap[i] else -1.0  # glass in +z (front surface) or -z
                cont, dz = cad._continuation(s, rc, mech.land, rm)
                e = max(rm - b, rc)
                if e > rc + 1e-12:
                    added.append(face_for(cont, rc, e, dz, frames[i], p_land, lab + " land", md))
                z_e = float(cont.sag(np.array([e**2]))[0]) + dz
                if b > 0 and faces_air:
                    o, R = frames[i]
                    added.append(Cone(e, z_e, rm, z_e + g*b, position=o, rotation=R,
                                      property=p_bev, media=(media[i + 1] if g > 0 else media[i],
                                                             media[i] if g > 0 else media[i + 1]),
                                      label=lab + " bevel"))
                    z_e = z_e + g*b
                edge_z[i] = z_e
            # rims of the glass slabs:
            slabs = [i for i in range(first, last) if not gap[i + 1]]
            for j, i in enumerate(slabs):
                o, R = frames[i]
                t = surfaces[i].thickness
                z0, z1 = edge_z[i], t + edge_z[i + 1]
                if abs(z1 - z0) > 1e-12:
                    added.append(Cylinder(rm, z0, z1, position=o, rotation=R,
                                          property=rim if rim is not None else ground_glass(),
                                          media=(media[i + 1], self.background),
                                          label=name + " rim" + ("abcdefgh"[j] if len(slabs) > 1
                                                                 else "")))
        if image:
            if image_size is None:
                h = max([np.nanmax(np.abs(b.points[-1][:, :2] - o_img[:2])) for b in traced]
                        + [1e-3])
                image_size = 2.4*h
            added.append(Rectangle(image_size, image_size, position=o_img, rotation=R_img,
                                   property=absorber(), label="image", detector=True))
        self.add(added)
        if mount is not None:
            model = cad.system_model(system, fields=fields, mechanics=mechanics or default,
                                     rays=None, image_plane=False, mount=mount)
            scale = rt.NM_PER_UNIT[system.units]/rt.NM_PER_UNIT["mm"]
            mesh = self.add_shapes(model.mount, property=mount_property or black_anodized(),
                                   per_face=mount_per_face, tolerance=tolerance,
                                   scale=1/scale)
            added.append(mesh)
        return added

    def add_shapes(self, shapes, property=None, media=None, labels=None, detector=False,
                   per_face=False, tolerance=0.01, angular_tolerance=0.1, scale=1.0):
        """Tessellate build123d shapes (CAD solids) into one `Mesh`.

        Every solid becomes one face of the scene, labelled with the solid's
        ``label`` (or ``labels``); with ``per_face=True`` every B-rep face
        of every solid becomes a face of its own, labelled
        "<solid>.f<k>" (k: the index in ``solid.faces()``).

        Properties, media and detectors are assigned by label:

        - one value for all (``property=nsrt.black_anodized()``),
        - a dict {label: value}: an exact label, a prefix ending in "*"
          ("spacer_*", "barrel.*"; the longest matching prefix wins) or
          "*" as the default,
        - a function f(label, shape) -> value, called with the label and
          the build123d solid (or, with per_face, the build123d Face, so
          that e.g. ``shape.geom_type``, ``shape.center()`` or
          ``shape.normal_at()`` can pick the cylindrical bore of a spacer).
          None from the function means the default.

        Parameters
        ----------
        shapes : sequence of build123d shapes
        property : FaceProperty, dict or callable, optional
            Default `absorber()`.
        media : (inside, outside), dict or callable, optional
            For transparent solids (normals point out of the solids, as
            build123d tessellates them). Default: the background medium.
        labels : sequence of str, optional
            Default: the shapes' labels, else "part<k>".
        detector : bool, dict or callable, optional
        per_face : bool, optional
        tolerance, angular_tolerance : numbers, optional
            Tessellation tolerances (mm, radians).
        scale : number, optional
            Factor from the shape units (mm) to the units of the scene.

        Returns
        -------
        Mesh
        """

        names = list(labels) if labels is not None else \
            [getattr(s, "label", None) or "part{}".format(k) for k, s in enumerate(shapes)]
        parts = []  # (label, build123d shape) per group
        for s, name in zip(shapes, names):
            if per_face:
                parts += [("{}.f{}".format(name, j), f) for j, f in enumerate(s.faces())]
            else:
                parts.append((name, s))
        V, T, G = [], [], []
        off = 0
        for k, (_, s) in enumerate(parts):
            v, t = s.tessellate(tolerance, angular_tolerance)
            v = np.array([tuple(x) for x in v], float).reshape(-1, 3)*scale
            t = np.asarray(t, np.int64).reshape(-1, 3)
            V.append(v)
            T.append(t + off)
            G.append(np.full(len(t), k))
            off += len(v)
        group_labels = [p[0] for p in parts]
        mesh = Mesh(np.concatenate(V), np.concatenate(T), np.concatenate(G), labels=group_labels,
                    properties=[_assign(property, lab, s, absorber()) for lab, s in parts],
                    media=[_assign(media, lab, s, None) for lab, s in parts],
                    detector=[bool(_assign(detector, lab, s, False)) for lab, s in parts])
        self.add(mesh)
        return mesh

    def add_step(self, filename, **kwargs):
        """Read the solids of a STEP file (build123d) and add them as a
        mesh, see `add_shapes()`.

        Parameters
        ----------
        filename : str
        **kwargs
            For `add_shapes()`.

        Returns
        -------
        Mesh
        """

        import build123d as bd

        shape = bd.import_step(filename)
        solids = list(shape.solids()) if hasattr(shape, "solids") else [shape]
        return self.add_shapes(solids, **kwargs)

    def add_stl(self, filename, scale=1.0, position=(0.0, 0.0, 0.0), rotation=None, **kwargs):
        """Add the triangles of an STL file as one mesh.

        Parameters
        ----------
        filename : str
        scale : number, optional
        position, rotation : optional
            Placement of the file's frame in the scene.
        **kwargs
            labels, properties, media, detector for `Mesh`.

        Returns
        -------
        Mesh
        """

        V, T = read_stl(filename)
        R = np.eye(3) if rotation is None else np.asarray(rotation, float)
        mesh = Mesh(np.asarray(position, float) + (V*scale) @ R.T, T, **kwargs)
        self.add(mesh)
        return mesh

    # tracing:
    def _pack(self):
        if self._packed is None:
            ids = np.array(self.primitives, dtype=np.int64)
            fs = [self.faces[k] for k in self.primitives]
            if fs:
                kinds = np.array([f.kind for f in fs], dtype=np.int64)
                origins = np.array([f.origin for f in fs])
                R = np.array([f.R for f in fs])
                params = np.array([f.params for f in fs])
                b = [f.bounds() for f in fs]
                lo = np.array([x[0] for x in b])
                hi = np.array([x[1] for x in b])
            else:
                kinds = np.zeros(0, np.int64)
                origins, params = np.zeros((0, 3)), np.zeros((0, 6))
                R = np.zeros((0, 3, 3))
                lo = hi = np.zeros((0, 3))
            self._packed = ids, kinds, origins, R, params, lo, hi
        return self._packed

    def nearest(self, P, d):
        """Nearest hits of rays: distances (inf: none), face ids (-1) and
        triangle indices (-1 for analytic faces)."""

        n = len(P)
        t = np.full(n, np.inf)
        fid = np.full(n, -1, np.int64)
        tri = np.full(n, -1, np.int64)
        ids, kinds, origins, R, params, lo, hi = self._pack()
        if len(ids):
            if use_numba and _nsrt_numba.available:
                tp, fp = _nsrt_numba.primitives_nearest(np.ascontiguousarray(P),
                                                       np.ascontiguousarray(d), kinds, origins,
                                                       R, params, lo, hi, self.eps)
                hit = fp >= 0
                t[hit] = tp[hit]
                fid[hit] = ids[fp[hit]]
            else:
                for j, k in enumerate(ids):
                    m = _aabb_mask(P, d, lo[j], hi[j], t)
                    if not np.any(m):
                        continue
                    tk = self.faces[k].intersect(P[m], d[m], self.eps)
                    better = tk < t[m]
                    sel = np.nonzero(m)[0][better]
                    t[sel] = tk[better]
                    fid[sel] = k
        for k in self.generic:
            f = self.faces[k]
            a, b = f.bounds()
            m = _aabb_mask(P, d, a, b, t)
            if not np.any(m):
                continue
            tk = f.intersect(P[m], d[m], self.eps)
            better = tk < t[m]
            sel = np.nonzero(m)[0][better]
            t[sel] = tk[better]
            fid[sel] = k
        for mesh, f0 in self.meshes:
            tk, tr = mesh.intersect(P, d, self.eps)
            better = tk < t
            t[better] = tk[better]
            fid[better] = f0 + mesh.groups[tr[better]]
            tri[better] = tr[better]
        return t, fid, tri

    def _media_tables(self):
        wl = self.wavelength_nm
        bg = self.background
        n_neg, n_pos, a_neg, a_pos = [], [], [], []
        for f in self.faces:
            m0, m1 = f.media if f.media is not None else (bg, bg)
            m0 = m0 or bg
            m1 = m1 or bg
            n_neg.append(m0.index(wl))
            n_pos.append(m1.index(wl))
            a_neg.append(m0.absorption)
            a_pos.append(m1.absorption)
        return np.array(n_neg), np.array(n_pos), np.array(a_neg), np.array(a_pos)

    def trace(self, rays, mode="split", min_weight=1e-4, max_events=64, max_generations=None,
              max_reflections=None, max_scatters=None, n_scatter=1, targets=None, aim_fraction=0.5, record=False, record_limit=20000,
              escape_length=None, rng=None):
        """Trace rays non-sequentially through the scene.

        Parameters
        ----------
        rays : RaySet or sequence of RaySet
        mode : {"split", "mc"}, optional
            Split every ray into all its channels, or pick one at random
            (see the module docstring).
        min_weight : number, optional
            Russian roulette threshold relative to the mean source ray weight.
        max_events : int, optional
            Interactions recorded per ray; rays with more are stopped (lost).
        max_generations : int, optional
            Steps of the trace (default 4 max_events).
        max_reflections, max_scatters : int, optional
            Stop rays after more specular reflections (e.g. 2 for a ghost
            analysis of the double reflections only) or scatter events;
            their power is booked as lost.
        n_scatter : int, optional
            Scattered children per scatter event in split mode.
        targets : sequence of Target, optional
            Disks that scattered rays are aimed at.
        aim_fraction : number, optional
            Probability of aiming a scattered ray (if a target lies in front).
        record : bool, optional
            Keep the traced segments (for `plot_rays()`).
        record_limit : int, optional
            Segments kept per step.
        escape_length : number, optional
            Length of the drawn segments of escaping rays.
        rng : None, int or numpy.random.Generator, optional

        Returns
        -------
        TraceResult
        """

        rng = _rng(rng)
        if not isinstance(rays, RaySet):
            rays = _concat(list(rays))
        P = np.array(rays.points, float).reshape(-1, 3)
        d = _unit(np.array(rays.directions, float).reshape(-1, 3))
        w = np.array(rays.weights, float).reshape(-1)
        res = TraceResult(self)
        res.source_power = float(w.sum())
        threshold = min_weight*(w.mean() if len(w) else 1.0)
        n_neg, n_pos, a_neg, a_pos = self._media_tables()
        wl, wl_nm = self.wavelength, self.wavelength_nm
        # history: a log of (parent entry, code) per interaction; rays carry
        # their last entry (-1: none) and the number of their events
        log_parent, log_code = [], []
        n_log = 0
        node = np.full(len(P), -1, np.int64)
        nev = np.zeros(len(P), np.int64)
        nref = np.zeros(len(P), np.int64)
        nsca = np.zeros(len(P), np.int64)
        det = {}
        targets = list(targets or [])
        if escape_length is None:
            lo, hi = self.bounds()
            escape_length = 0.2*np.max(hi - lo) if np.all(np.isfinite(hi - lo)) else 1.0
        max_generations = 4*max_events if max_generations is None else max_generations
        last = np.full(len(P), -1, np.int64)  # code of the last event (for record)
        for gen in range(max_generations):
            if not len(P):
                break
            res.generations = gen + 1
            res.n_rays_traced += len(P)
            t, fid, tri = self.nearest(P, d)
            out = fid < 0
            res.escaped += w[out].sum()
            if record:
                self._record(res, P, d, t, w, last, out, escape_length, record_limit)
            keep = np.nonzero(~out)[0]
            if not len(keep):
                P = P[:0]
                break
            P, d, w, t, fid, tri, node, nev, nref, nsca = (
                P[keep], d[keep], w[keep], t[keep], fid[keep], tri[keep], node[keep], nev[keep],
                nref[keep], nsca[keep])
            Q = P + t[:, None]*d
            # rays grouped by face:
            order = np.argsort(fid, kind="stable")
            uniq, start = np.unique(fid[order], return_index=True)
            groups = [(int(k), order[a:b]) for k, a, b in
                      zip(uniq, start, list(start[1:]) + [len(order)])]
            N = np.empty_like(Q)
            for k, m in groups:
                f = self.faces[k]
                N[m] = f.normal(Q[m], tri[m]) if isinstance(f, MeshGroup) else f.normal(Q[m])
            cos = np.sum(d*N, -1)
            from_plus = cos < 0
            n1 = np.where(from_plus, n_pos[fid], n_neg[fid])
            n2 = np.where(from_plus, n_neg[fid], n_pos[fid])
            alpha = np.where(from_plus, a_pos[fid], a_neg[fid])
            if np.any(alpha > 0):
                w_new = w*np.exp(-alpha*t)
                np.add.at(res.bulk, fid, w - w_new)
                w = w_new
            # channels and directions:
            ci = np.abs(cos)
            Nin = -np.sign(cos)[:, None]*N  # normal on the side of the incoming ray
            d_r = d + 2*ci[:, None]*Nin
            mu = n1/n2
            sin2_t = mu**2*(1 - ci**2)
            cos_t = np.sqrt(np.clip(1 - sin2_t, 0, None))
            d_t = mu[:, None]*d + (mu*ci - cos_t)[:, None]*Nin
            tir = sin2_t >= 1
            R = np.zeros(len(P))
            T = np.zeros(len(P))
            Sr = np.zeros(len(P))
            St = np.zeros(len(P))
            for k, m in groups:
                f = self.faces[k]
                if f.detector:
                    loc = Q[m] if isinstance(f, MeshGroup) else f.to_local(Q[m])
                    det.setdefault(k, []).append((loc, d[m], w[m], node[m], nev[m]))
                prop = f.property
                Rk, Tk = np.empty(len(m)), np.empty(len(m))
                fp = from_plus[m]
                for side in (False, True):
                    s = fp == side
                    if np.any(s):
                        ms = m[s]
                        Rk[s], Tk[s] = prop.evaluate(ci[ms], n1[ms[0]], n2[ms[0]], fp[s], wl, wl_nm)
                tm = tir[m]
                if np.any(tm):
                    if not prop.fresnel:
                        Rk = np.where(tm, Rk + Tk, Rk)
                    Tk = np.where(tm, 0.0, Tk)
                sr = prop.brdf.tis(np.sqrt(np.clip(1 - ci[m]**2, 0, 1))) if prop.brdf else 0.0
                st = prop.btdf.tis(np.sqrt(np.clip(sin2_t[m], 0, 1))) if prop.btdf else 0.0
                if prop.btdf:
                    st = np.where(tm, 0.0, st)
                if prop.fresnel:
                    sr = np.minimum(sr, Rk)
                    st = np.minimum(st, Tk)
                    Rk, Tk = Rk - sr, Tk - st
                R[m], T[m], Sr[m], St[m] = Rk, Tk, sr, st
            tot = R + T + Sr + St
            over = tot > 1
            if np.any(over):
                R[over], T[over], Sr[over], St[over] = (x[over]/tot[over] for x in (R, T, Sr, St))
                tot = np.minimum(tot, 1.0)
            np.add.at(res.absorbed, fid, w*(1 - tot))
            # children:
            kids = []
            if mode == "split":
                for ev, X, D in ((_R, R, d_r), (_T, T, d_t)):
                    m = np.nonzero(X > 0)[0]
                    kids.append((m, ev, D[m], w[m]*X[m]))
                for ev, S, attr, side in ((_SR, Sr, "brdf", 1.0), (_ST, St, "btdf", -1.0)):
                    m = np.nonzero(S > 0)[0]
                    if not len(m):
                        continue
                    for _ in range(n_scatter):
                        D, fac = self._scatter(m, fid, Q, d_r if side > 0 else d_t,
                                               side*Nin, attr, targets, aim_fraction, rng)
                        kids.append((m, ev, D, w[m]*fac/n_scatter))
            elif mode == "mc":
                live = tot > 0
                with np.errstate(invalid="ignore", divide="ignore"):
                    cum = np.cumsum(np.stack([T, R, Sr, St]), axis=0)/tot  # event order
                u = rng.random(len(P))
                ch = np.sum(u[None, :] >= cum, axis=0)
                for ev, D in ((_R, d_r), (_T, d_t)):
                    m = np.nonzero(live & (ch == ev))[0]
                    kids.append((m, ev, D[m], w[m]*tot[m]))
                for ev, S, attr, side in ((_SR, Sr, "brdf", 1.0), (_ST, St, "btdf", -1.0)):
                    m = np.nonzero(live & (ch == ev))[0]
                    if not len(m):
                        continue
                    D, fac = self._scatter(m, fid, Q, d_r if side > 0 else d_t, side*Nin,
                                           attr, targets, aim_fraction, rng)
                    kids.append((m, ev, D, w[m]*tot[m]*fac/S[m]))
            else:
                raise ValueError("mode must be 'split' or 'mc'.")
            idx = np.concatenate([k[0] for k in kids])
            ev = np.concatenate([np.full(len(k[0]), k[1], np.int64) for k in kids])
            Dn = np.concatenate([k[2] for k in kids])
            Wn = np.concatenate([k[3] for k in kids])
            ok = (Wn > 0) & np.all(np.isfinite(Dn), axis=-1)
            # Russian roulette:
            low = ok & (Wn < threshold)
            if np.any(low):
                survive = rng.random(int(low.sum())) < Wn[low]/threshold
                Wn = Wn.copy()
                Wn[low] = np.where(survive, threshold, 0.0)
                ok &= Wn > 0
            nr = nref[idx] + (ev == _R)
            ns = nsca[idx] + (ev >= _SR)
            full = ok & ((nev[idx] >= max_events)
                         | ((nr > max_reflections) if max_reflections is not None else False)
                         | ((ns > max_scatters) if max_scatters is not None else False))
            res.lost += Wn[full].sum()
            ok &= ~full
            idx, ev, Dn, Wn, nr, ns = idx[ok], ev[ok], Dn[ok], Wn[ok], nr[ok], ns[ok]
            code = 4*fid[idx] + ev
            log_parent.append(node[idx])
            log_code.append(code)
            node = n_log + np.arange(len(idx))
            n_log += len(idx)
            P, d, w, nev, last, nref, nsca = Q[idx], Dn, Wn, nev[idx] + 1, code, nr, ns
        if len(P):
            res.lost += w.sum()
        parent = np.concatenate(log_parent) if log_parent else np.zeros(0, np.int64)
        codes = np.concatenate(log_code).astype(np.int32) if log_code else np.zeros(0, np.int32)
        for k, parts in det.items():
            f = self.faces[k]
            loc, dd, ww, nd, ne = (np.concatenate([p[j] for p in parts]) for j in range(5))
            res.hits[f.label] = Hits(self, f, loc, dd, ww, _histories(nd, ne, parent, codes),
                                     ne)
        for k, f in enumerate(self.faces):
            if f.detector and f.label not in res.hits:
                res.hits[f.label] = Hits(self, f, np.zeros((0, 3)), np.zeros((0, 3)),
                                         np.zeros(0), np.zeros((0, 1), np.int32),
                                         np.zeros(0, np.int64))
        return res

    def _scatter(self, m, fid, Q, D0, Nside, attr, targets, f_aim, rng):
        """Scattered directions and weight factors bsdf/pdf of the rays m
        (scattered into the side of the normals Nside, around the specular
        directions D0)."""

        out_d = np.empty((len(m), 3))
        fac = np.empty(len(m))
        fm = fid[m]
        for k in np.unique(fm):
            sel = np.nonzero(fm == k)[0]
            model = getattr(self.faces[k].property, attr)
            Ns = Nside[m[sel]]
            e1, e2 = _bsdf.tangent_basis(Ns)
            beta0 = np.stack([np.sum(D0[m[sel]]*e1, -1), np.sum(D0[m[sel]]*e2, -1)], -1)
            Qs = Q[m[sel]]
            beta = model.sample(beta0, rng)
            D = _bsdf.beta_to_direction(beta, Ns, e1, e2)
            if targets:
                vis = np.stack([np.sum((tg.center - Qs)*Ns, -1) > 0 for tg in targets], -1)
                n_vis = vis.sum(-1)
                fa = np.where(n_vis > 0, f_aim, 0.0)
                aim = rng.random(len(sel)) < fa
                if np.any(aim):
                    # pick one visible target per aimed ray:
                    r = rng.random(len(sel))
                    cum = np.cumsum(vis, -1)
                    pick = np.argmax(cum > (r*n_vis)[:, None], axis=-1)
                    for j, tg in enumerate(targets):
                        a = np.nonzero(aim & (pick == j))[0]
                        if len(a):
                            D[a] = tg.sample(Qs[a], rng)
                    beta = np.stack([np.sum(D*e1, -1), np.sum(D*e2, -1)], -1)
                cos_s = np.sum(D*Ns, -1)
                p_aim = np.zeros(len(sel))
                for j, tg in enumerate(targets):
                    ok, dens = tg.hit(Qs, D)
                    p_aim += np.where(vis[:, j] & ok, dens, 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    p_aim = np.where(n_vis > 0, p_aim/np.maximum(n_vis, 1), 0.0)/cos_s
                    p_mix = fa*p_aim + (1 - fa)*model.pdf(beta, beta0)
                    f = model.bsdf(beta, beta0)/p_mix
                f = np.where((cos_s > 0) & np.isfinite(f), f, 0.0)
            else:
                f = model.tis(np.linalg.norm(beta0, axis=-1))
            out_d[sel] = D
            fac[sel] = f
        return out_d, fac

    def _record(self, res, P, d, t, w, last, out, escape_length, limit):
        tt = np.where(out, escape_length, t)
        k = np.arange(len(P))
        if len(k) > limit:
            k = np.sort(_rng(0).choice(len(P), limit, replace=False))
        res.segments.append((P[k], P[k] + tt[k, None]*d[k], w[k], last[k]))


def _histories(node, n_events, parent, codes):
    """Event codes (N, max n_events) of rays from their last log entries."""

    width = max(int(n_events.max()) if len(n_events) else 0, 1)
    out = np.full((len(node), width), -1, np.int32)
    cur = node.copy()
    for j in range(width):
        live = cur >= 0
        if not np.any(live):
            break
        pos = n_events[live] - 1 - j
        out[np.nonzero(live)[0], pos] = codes[cur[live]]
        cur[live] = parent[cur[live]]
    return out


def _assign(x, label, shape, default):
    """Value for a label: x itself, x[label] (exact, longest "prefix*" or
    "*") or x(label, shape)."""

    if isinstance(x, dict):
        if label in x:
            return x[label]
        best = None
        for key in x:
            if key.endswith("*") and label.startswith(key[:-1]):
                if best is None or len(key) > len(best):
                    best = key
        return x[best] if best is not None else default
    if callable(x) and not isinstance(x, (FaceProperty, Medium)):
        v = x(label, shape)
        return default if v is None else v
    return default if x is None else x


def _aabb_mask(P, d, lo, hi, tmax):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1/d
        ta = (lo - P)*inv
        tb = (hi - P)*inv
        t0 = np.nanmax(np.minimum(ta, tb), axis=1)
        t1 = np.nanmin(np.maximum(ta, tb), axis=1)
    return (t1 >= np.maximum(t0, 0)) & (t0 <= tmax)


# --- plotting ----------------------------------------------------------------

_PLANES = {"yz": (2, 1, 0), "xz": (2, 0, 1), "xy": (0, 1, 2)}


def plot_scene(scene, ax=None, plane="yz", offset=0.0, color="0.3", lw=0.8, mesh_color="0.15",
               mesh_alpha=0.6):
    """Draw the section of the scene with a coordinate plane: rectangles and meshes by their exact cut, faces of revolution by
    their profile if their axis lies in the plane (coaxial lenses), other
    analytic faces by their section with their local x = 0 plane.

    Parameters
    ----------
    scene : Scene
    ax : matplotlib Axes, optional
    plane : {"yz", "xz", "xy"}, optional
        Horizontal and vertical axes of the plot: z and y, z and x, x and y.
    offset : number, optional
        Position of the cut plane along the third axis (x, y or z).
    color, lw, mesh_color, mesh_alpha : optional

    Returns
    -------
    ax
    """

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ax = ax or plt.gca()
    h, v, cut = _PLANES[plane]
    lines = []
    for k in scene.primitives + scene.generic:
        for c in scene.faces[k].outline(cut, offset):
            lines.append(c[:, [h, v]])
    ax.add_collection(LineCollection(lines, colors=color, linewidths=lw))
    for mesh, _ in scene.meshes:
        segs = mesh.section(cut, offset)
        if len(segs):
            ax.add_collection(LineCollection(segs[:, :, [h, v]], colors=mesh_color,
                                             linewidths=lw, alpha=mesh_alpha))
    ax.autoscale_view()
    ax.set_aspect("equal")
    return ax


def plot_rays(result, ax=None, plane="yz", max_rays=3000, color=None, cmap="viridis",
              alpha=0.5, lw=0.4, log_weight=True, rng=0):
    """Draw recorded ray segments (trace with record=True), coloured by the
    number of reflections and scatters of the rays (or a fixed colour), with
    the opacity scaled by the weight.

    Parameters
    ----------
    result : TraceResult
    ax : matplotlib Axes, optional
    plane : {"yz", "xz", "xy"}, optional
    max_rays : int, optional
        Segments drawn per step.
    color : colour, optional
    cmap : str, optional
    alpha, lw : numbers, optional
    log_weight : bool, optional
        Opacity ~ log of the weight (else linear).
    rng : optional

    Returns
    -------
    ax
    """

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ax = ax or plt.gca()
    h, v, _ = _PLANES[plane]
    cm = plt.get_cmap(cmap)
    w_ref = max([s[2].max() for s in result.segments if len(s[2])] + [1e-300])
    g = _rng(rng)
    for gen, (A, B, w, last) in enumerate(result.segments):
        if not len(w):
            continue
        k = np.arange(len(w))
        if len(k) > max_rays:
            k = np.sort(g.choice(len(w), max_rays, replace=False))
        segs = np.stack([A[k][:, [h, v]], B[k][:, [h, v]]], 1)
        rel = w[k]/w_ref
        if log_weight:
            a = np.clip(1 + np.log10(np.maximum(rel, 1e-300))/4, 0.05, 1.0)
        else:
            a = np.clip(rel, 0.02, 1.0)
        if color is None:
            kind = np.where(last >= 0, last % 4, 0)[k]
            base = np.array([cm(0.15), cm(0.55), cm(0.9), cm(0.9)])[kind]
        else:
            from matplotlib.colors import to_rgba
            base = np.tile(to_rgba(color), (len(k), 1))
        base = base.copy()
        base[:, 3] = alpha*a
        ax.add_collection(LineCollection(segs, colors=base, linewidths=lw))
    ax.autoscale_view()
    ax.set_aspect("equal")
    return ax
