#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Export of `raytrace.System` lens systems to CAD (STEP) with build123d.

The lenses, mirror substrates and the image plane of a sequential system
become B-rep solids and faces, the traced rays become polylines (wires) or
thin tubes, and everything is written to one STEP file with names and
colours, ready to be placed in a mount design ::

    from pyoptics import cad
    model = cad.system_model(system, fields=[0.0, 0.02], n_rays=7,
                             mechanics={0: cad.Mechanics(margin=2.0, bevel=0.3)})
    cad.export_step(model, "doublet.step")

Needs the optional dependency build123d (``pip install pyoptics[cad]``),
which brings OpenCascade (OCP); it is imported when a function is called.

Geometry
--------
Every run of surfaces with glass behind the first one becomes one solid
per glass slab: a cemented doublet is two solids that share the cemented
face. The solids are built in the local frame of their first surface and
placed with `System.frames()`, so decentred, tilted and folded systems come
out as traced. Lengths are converted to millimetres.

- Coaxial slabs of rotationally symmetric surfaces (spheres, conics, even
  aspheres, Q-type aspheres, axicons) are revolved half sections. Planes
  are lines and spheres circular arcs in the section, so they are exact in
  the STEP file; other profiles are B-spline interpolations of the sag,
  refined until they deviate less than ``tolerance`` from it.
- Surfaces with a ``sag_xy(x, y)`` (`raytrace.ZernikeSurface`,
  `qaspheres.Q2DSurface`, `freeform.TabulatedSurface`) and slabs whose
  surfaces are decentred or tilted against each other are cut from a
  cylinder of the mechanical diameter by B-spline (or revolved) surface
  faces.

Mechanical diameter, lands and bevels
-------------------------------------
`Mechanics` sets, per element (the index into `System.elements()`), the
mechanical semi-diameter (by default the largest clear semi-diameter of
the element plus ``margin``), what happens between the clear aperture and
the rim (``land="flat"``: a flat annulus perpendicular to the axis at the
height of the clear aperture rim, for the seat of a mount; ``"extend"``:
the optical surface goes on to the rim) and 45 degree protective bevels of
face width ``bevel`` on the rim edges that face air. Cemented interfaces
keep sharp edges. A slab whose edge would get thinner than
``min_edge_thickness`` raises a ValueError.

Mirrors get a substrate of thickness ``Mechanics.mirror_thickness`` with a
flat back. Ideal lenses, coordinate breaks and dummy surfaces have no
geometry.
"""

from dataclasses import dataclass, field as _field
from typing import Optional

import numpy as np

from . import raytrace as rt


@dataclass
class Mechanics:
    """Mechanical data of one element (a lens, a cemented group or a mirror).

    Parameters
    ----------
    semi_diameter : number, optional
        Mechanical (rim) semi-diameter in system units. None: the largest
        clear semi-diameter of the element plus margin.
    margin : number, optional
        Added to the clear semi-diameter if semi_diameter is None.
    land : {"flat", "extend"}, optional
        Between clear aperture and rim: a flat annulus perpendicular to the
        axis (default), or the continued optical surface.
    bevel : number or (front, back), optional
        Face width of the 45 degree bevels on the rim edges of the surfaces
        that face air (0: sharp edges).
    mirror_thickness : number, optional
        Substrate thickness of a mirror, from the rim of the mirror face to
        its flat back.
    min_edge_thickness : number, optional
        Smallest allowed axial edge thickness of a slab (after the bevels).
    """

    semi_diameter: Optional[float] = None
    margin: float = 1.0
    land: str = "flat"
    bevel: object = 0.0
    mirror_thickness: float = 5.0
    min_edge_thickness: float = 0.1

    def bevels(self):
        b = self.bevel
        return (float(b), float(b)) if np.isscalar(b) else (float(b[0]), float(b[1]))


@dataclass
class Model:
    """CAD model of a system: build123d shapes in millimetres.

    Attributes
    ----------
    lenses : list of build123d.Solid
        Glass slabs and mirror substrates, labelled "L1", "L2a", "M1", ...
    rays : list of build123d shapes
        One wire (or a compound of tubes) per traced ray, labelled by field.
    image : build123d.Face or None
        Image plane.
    """

    lenses: list = _field(default_factory=list)
    rays: list = _field(default_factory=list)
    image: object = None

    def compound(self):
        """All parts as one labelled build123d.Compound (an assembly tree)."""

        import build123d as bd

        children = []
        if self.lenses:
            children.append(bd.Compound(children=list(self.lenses), label="lenses"))
        if self.rays:
            children.append(bd.Compound(children=list(self.rays), label="rays"))
        if self.image is not None:
            children.append(self.image)

        return bd.Compound(children=children, label="system")


# --- sag helpers -------------------------------------------------------------

def _is_freeform(s):
    return hasattr(s, "sag_xy")


def _is_rotational(s):
    return not _is_freeform(s) and not isinstance(s, (rt.IdealLens, rt.CoordinateBreak))


def _sphere_or_plane(s):
    """'plane', 'sphere' or None (general profile)."""

    if type(s) not in (rt.Surface, rt.GratingSurface, rt.DiffractiveSurface):
        return None
    if s.aspheric or s.conic:
        return "plane" if np.isinf(s.R) and not s.aspheric else None
    return "plane" if np.isinf(s.R) else "sphere"


def _profile_edges(s, r0, r1, z_offset, scale, tolerance):
    """Edges of the half section z(r) of a rotationally symmetric surface
    from r0 to r1 (system units), in mm; z_offset in system units."""

    import build123d as bd

    def pt(r):
        return bd.Vector(r*scale, 0.0, (float(s.sag(np.array([r**2]))[0]) + z_offset)*scale)

    if r1 <= r0:
        return []
    kind = _sphere_or_plane(s)
    if kind == "plane":
        return [bd.Edge.make_line(pt(r0), pt(r1))]
    if kind == "sphere":
        return [bd.Edge.make_three_point_arc(pt(r0), pt(0.5*(r0 + r1)), pt(r1))]
    # B-spline through the sag, refined until it follows the sag within the
    # tolerance halfway between the nodes:
    for n in (9, 17, 33, 65, 129, 257):
        r = np.linspace(r0, r1, n)
        pts = [pt(x) for x in r]
        tangents = None
        if r0 == 0.0:  # zero slope on the axis
            tangents = [bd.Vector(1, 0, 0), pts[-1] - pts[-2]]
        edge = bd.Edge.make_spline(pts, tangents=tangents) if tangents is None else \
            bd.Edge.make_spline(pts, tangents=[tangents[0], _tangent(s, r1, z_offset, scale)])
        r_mid = 0.5*(r[1:] + r[:-1])
        err = max(edge.distance_to(pt(x)) for x in r_mid)
        if err < tolerance:
            return [edge]
    raise ValueError("Could not fit the profile of {} within {} mm (error {:.2g} mm)."
                     .format(s, tolerance, err))


def _tangent(s, r, z_offset, scale):
    import build123d as bd

    dr = 1e-6*max(r, 1.0)
    dz = (s.sag(np.array([(r + dr)**2]))[0] - s.sag(np.array([(r - dr)**2]))[0])/(2*dr)
    return bd.Vector(1.0, 0.0, float(dz))


def _rim_sag(s, r):
    return float(s.sag(np.array([r**2]))[0])


# --- solids ------------------------------------------------------------------

def _location(origin, R, scale):
    import build123d as bd

    plane = bd.Plane(origin=tuple(origin*scale), x_dir=tuple(R[:, 0]), z_dir=tuple(R[:, 2]))
    return bd.Location(plane)


def _revolved_slab(s1, s2, t, r1, r2, rm, land, bevels, scale, tolerance, min_edge):
    """Solid of revolution between s1 (vertex at z = 0) and s2 (vertex at
    z = t), clear semi-diameters r1, r2 and rim semi-diameter rm, in the
    local frame of s1 (mm)."""

    import build123d as bd

    sgn = 1.0 if t > 0 else -1.0  # glass lies in +z (sgn = 1) of s1
    b1, b2 = bevels
    if land == "extend":
        r1 = r2 = rm
    z1 = _rim_sag(s1, r1)
    z2 = t + _rim_sag(s2, r2)
    if not (np.isfinite(z1) and np.isfinite(z2)):
        raise ValueError("Surface sag undefined at the clear aperture or rim.")
    edge = sgn*(z2 - z1) - b1 - b2
    if edge < min_edge:
        raise ValueError("Edge thickness {:.3g} below {} at semi-diameter {} between {} and {}."
                         .format(edge, min_edge, rm, s1, s2))

    def P(r, z):
        return bd.Vector(r*scale, 0.0, z*scale)

    edges = _profile_edges(s1, 0.0, r1, 0.0, scale, tolerance)
    # front land, bevel and rim:
    pts = [P(r1, z1), P(rm - b1, z1), P(rm, z1 + sgn*b1), P(rm, z2 - sgn*b2), P(rm - b2, z2),
           P(r2, z2)]
    for a, b in zip(pts[:-1], pts[1:]):
        if (a - b).length > 1e-9:
            edges.append(bd.Edge.make_line(a, b))
    back = _profile_edges(s2, 0.0, r2, t, scale, tolerance)
    edges += [e.reversed() for e in back[::-1]]
    edges.append(bd.Edge.make_line(P(0.0, t), P(0.0, 0.0)))
    face = bd.Face(bd.Wire(edges))

    return bd.Solid.revolve(face, 360.0, bd.Axis.Z)


def _surface_face(s, r_clear, rm, land, scale, n_grid=41):
    """Face of a surface in its local frame (mm), large enough to cut a
    cylinder of semi-diameter rm."""

    import build123d as bd

    ext = 1.1*rm
    if _is_freeform(s):
        # B-spline surface through the sag on a square grid; "flat" land is
        # not available for freeforms yet, the sag is extended to the rim.
        x = np.linspace(-ext, ext, n_grid)
        X, Y = np.meshgrid(x, x, indexing="ij")
        Z = s.sag_xy(X, Y)
        if not np.all(np.isfinite(Z)):
            raise ValueError("sag_xy of {} undefined up to the rim ({}).".format(s, ext))
        pts = [[bd.Vector(X[i, j]*scale, Y[i, j]*scale, Z[i, j]*scale) for j in range(n_grid)]
               for i in range(n_grid)]
        return bd.Face.make_surface_from_array_of_points(pts, tol=1e-6, min_deg=3, max_deg=5)
    # rotationally symmetric: revolve the section (with land) beyond the rim
    r_end = rm if land == "extend" else r_clear
    edges = _profile_edges(s, 0.0, r_end, 0.0, scale, 1e-5)
    z = _rim_sag(s, r_end)
    edges.append(bd.Edge.make_line(bd.Vector(r_end*scale, 0, z*scale),
                                   bd.Vector(ext*scale, 0, z*scale)))
    return bd.Face.revolve(bd.Wire(edges), 360.0, bd.Axis.Z) if hasattr(bd.Face, "revolve") \
        else bd.Shell.revolve(bd.Wire(edges), 360.0, bd.Axis.Z)


def _cut_slab(system, i, rm, r1, r2, land, scale):
    """Slab between surfaces i and i + 1, cut from a cylinder of radius rm
    along the axis of surface i (global coordinates, mm)."""

    import build123d as bd

    frames = system.frames()[0]
    s1, s2 = system.surfaces[i], system.surfaces[i + 1]
    o1, R1 = frames[i]
    o2, R2 = frames[i + 1]
    loc1, loc2 = _location(o1, R1, scale), _location(o2, R2, scale)
    f1 = loc1*_surface_face(s1, r1, rm, land, scale)
    f2 = loc2*_surface_face(s2, r2, rm, land, scale)
    t = s1.thickness
    depth = (abs(t) + 2*rm)*scale
    cyl = bd.Solid.make_cylinder(rm*scale, 2*depth,
                                 bd.Plane(origin=(0, 0, -depth)))
    cyl = loc1*cyl
    mid = bd.Vector(*((o1 + R1 @ [0, 0, 0.5*t])*scale))
    piece = cyl
    for f in (f1, f2):
        parts = piece.split(f, keep=bd.Keep.BOTH)
        parts = parts if isinstance(parts, (tuple, list)) else [parts]
        solids = [so for part in parts if part is not None for so in part.solids()]
        # keep the part that holds the middle of the slab on the axis:
        piece = min(solids, key=lambda p: p.distance_to(mid))
    return piece


def system_model(system, fields=None, n_rays=7, pupil="y", mechanics=None, rays="wire",
                 ray_radius=0.02, ray_start=None, image_plane=True, tolerance=1e-4):
    """Build the CAD model of a system.

    Parameters
    ----------
    system : raytrace.System
    fields : sequence, optional
        Fields of the ray fans (default: 0 and the maximum field).
    n_rays : int, optional
        Rays per fan.
    pupil : {"y", "x", "xy", "ring"} or (px, py) arrays, optional
        Pupil sampling: a meridional ("y") or sagittal ("x") fan, both
        ("xy"), a ring at the rim plus the chief ray ("ring"), or explicit
        normalized pupil coordinates.
    mechanics : dict or Mechanics, optional
        Mechanics per element index of `System.elements()`, or one for all.
    rays : {"wire", "tube", None}, optional
        Rays as polylines, as tubes of radius ray_radius (mm), or none.
    ray_radius : number, optional
        Tube radius in mm.
    ray_start : number, optional
        For an object at infinity: the rays start this far (system units)
        in front of the first vertex (default: 20 % of the track length).
    image_plane : bool, optional
        Add the image plane as a square face.
    tolerance : number, optional
        Largest deviation (mm) of B-spline profiles from the sag.

    Returns
    -------
    Model
    """

    import build123d as bd

    scale = rt.NM_PER_UNIT[system.units]/rt.NM_PER_UNIT["mm"]
    if mechanics is None or isinstance(mechanics, Mechanics):
        default = mechanics or Mechanics()
        mechanics = {}
    else:
        default = Mechanics()
    n = system.n
    sigma = system.sigma
    frames, (o_img, R_img) = system.frames()
    surfaces = system.surfaces
    model = Model()

    # ray traces first: unlimited surfaces get their size from the rays
    traced = _trace(system, fields, n_rays, pupil, ray_start)
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

    for k, (first, last) in enumerate(system.elements()):
        mech = mechanics.get(k, default)
        idx = range(first, last + 1)
        r_clear = max(clear(i) for i in idx)
        rm = mech.semi_diameter if mech.semi_diameter is not None else r_clear + mech.margin
        if rm < r_clear:
            raise ValueError("Mechanical semi-diameter {} of element {} below its clear "
                             "aperture {}.".format(rm, k, r_clear))
        name = "L{}".format(k + 1)
        if first == last and surfaces[first].mirror:
            solid = _mirror(system, first, clear(first), rm, mech, scale, tolerance,
                            sigma[first])
            solid.label, solid.color = "M{}".format(k + 1), bd.Color(0.75, 0.75, 0.8)
            model.lenses.append(solid)
            continue
        if first == last:  # ideal lens: no geometry
            continue
        slabs = [i for i in range(first, last) if not system._is_gap(n[i + 1])]
        for j, i in enumerate(slabs):
            s1, s2 = surfaces[i], surfaces[i + 1]
            b1, b2 = mech.bevels()
            # bevels only on rim edges facing air:
            b1 = b1 if system._is_gap(n[i]) else 0.0
            b2 = b2 if system._is_gap(n[i + 2]) else 0.0
            coaxial = not (s1.displaced or s2.displaced) and \
                not isinstance(s2, rt.CoordinateBreak)
            if coaxial and _is_rotational(s1) and _is_rotational(s2):
                solid = _revolved_slab(s1, s2, s1.thickness, clear(i), clear(i + 1), rm,
                                       mech.land, (b1, b2), scale, tolerance,
                                       mech.min_edge_thickness)
                solid = _location(*frames[i], scale)*solid
            else:
                if b1 or b2:
                    raise NotImplementedError("Bevels on freeform or displaced slabs.")
                solid = _cut_slab(system, i, rm, clear(i), clear(i + 1), mech.land, scale)
            solid.label = name + ("abcdefgh"[j] if len(slabs) > 1 else "")
            solid.color = bd.Color(0.55, 0.75, 0.95, 0.5)
            model.lenses.append(solid)

    if rays:
        palette = [(0.1, 0.3, 0.9), (0.1, 0.65, 0.2), (0.85, 0.15, 0.1), (0.8, 0.5, 0.0),
                   (0.5, 0.2, 0.7)]
        for j, b in enumerate(traced):
            color = bd.Color(*palette[j % len(palette)])
            for m in range(b.points.shape[1]):
                path = _ray_path(b.points[:, m])*scale
                if len(path) < 2:
                    continue
                shape = _ray_shape(path, rays, ray_radius)
                shape.label = "field{}_ray{}".format(j, m)
                shape.color = color
                model.rays.append(shape)

    if image_plane:
        h = 1.2*max([np.nanmax(np.abs(b.points[-1][:, :2] - o_img[:2])) for b in traced]
                    + [1e-3])
        face = bd.Face.make_rect(2*h*scale, 2*h*scale)
        face = _location(o_img, R_img, scale)*face
        face.label, face.color = "image", bd.Color(0.6, 0.6, 0.6, 0.6)
        model.image = face

    return model


def _mirror(system, i, r_clear, rm, mech, scale, tolerance, sigma_in):
    """Mirror substrate: the mirror face, a land to the rim and a flat back
    mirror_thickness behind the rim (seen from the incoming light)."""

    import build123d as bd

    s = system.surfaces[i]
    if not _is_rotational(s):
        raise NotImplementedError("Freeform mirrors.")
    back = rt.Surface(np.inf)
    z_rim = _rim_sag(s, rm if mech.land == "extend" else r_clear)
    t = z_rim + sigma_in*mech.mirror_thickness
    b1, _ = mech.bevels()
    solid = _revolved_slab(s, back, t, r_clear, rm, rm, mech.land, (b1, 0.0), scale, tolerance,
                           mech.min_edge_thickness)
    return _location(*system.frames()[0][i], scale)*solid


# --- rays --------------------------------------------------------------------

def _trace(system, fields, n_rays, pupil, ray_start):
    if fields is None:
        fields = [0.0, system.max_field] if system.max_field else [0.0]
    p = np.linspace(-1, 1, n_rays)
    if isinstance(pupil, str):
        if pupil == "y":
            px, py = 0*p, p
        elif pupil == "x":
            px, py = p, 0*p
        elif pupil == "xy":
            px, py = np.r_[0*p, p], np.r_[p, 0*p]
        elif pupil == "ring":
            a = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
            px, py = np.r_[0.0, np.cos(a)], np.r_[0.0, np.sin(a)]
        else:
            raise ValueError("Unknown pupil sampling {!r}.".format(pupil))
    else:
        px, py = pupil
    out = []
    for f in fields:
        b = system.trace_pupil(f, px, py)
        if system.infinite_object:
            pts = b.points.copy()
            z0 = -(ray_start if ray_start is not None else
                   0.2*max(abs(system.image_z), 1.0))
            P, d = pts[0], b.directions[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                pts[0] = P + ((z0 - P[:, 2])/d[:, 2])[:, None]*d
            b = b._replace(points=pts)
        out.append(b)
    return out


def _ray_path(points):
    """Finite points of one ray without repeats (coordinate breaks); stops
    at the first lost point."""

    ok = np.all(np.isfinite(points), axis=1)
    if not ok[0]:
        return np.zeros((0, 3))
    end = len(ok) if ok.all() else int(np.argmin(ok))
    pts = points[:end]
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-12]
    return pts[keep]


def _ray_shape(path, style, radius):
    import build123d as bd

    if style == "wire":
        return bd.Wire.make_polygon([bd.Vector(*p) for p in path], close=False)
    if style == "tube":
        tubes = []
        for a, b in zip(path[:-1], path[1:]):
            d = b - a
            L = np.linalg.norm(d)
            tubes.append(bd.Solid.make_cylinder(radius, L, bd.Plane(origin=tuple(a),
                                                                    z_dir=tuple(d/L))))
        return bd.Compound(children=tubes)
    raise ValueError("rays must be 'wire', 'tube' or None.")


def export_step(model, filename):
    """Write a Model (or any build123d shape) to a STEP file in mm."""

    import build123d as bd

    shape = model.compound() if isinstance(model, Model) else model
    return bd.export_step(shape, str(filename))
