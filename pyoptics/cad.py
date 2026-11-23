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
  refined until they deviate less than ``tolerance`` from it. Surfaces with
  a ``section()`` (`fresnel.FresnelSurface`) get a B-spline per facet and
  straight risers.
- Surfaces with a ``sag_xy(x, y)`` (`raytrace.ToroidalSurface`,
  `raytrace.BiconicSurface`, `raytrace.ZernikeSurface`,
  `qaspheres.Q2DSurface`, `freeform.TabulatedSurface`) and slabs whose
  surfaces are decentred or tilted against each other are cut from a
  cylinder of the mechanical diameter along the axis of their first
  surface by B-spline (or revolved) surface faces.

Mechanical diameter, lands and bevels
-------------------------------------
`Mechanics` sets, per element (the index into `System.elements()`), the
mechanical semi-diameter (by default the largest clear semi-diameter of
the element plus ``margin``), what happens between the clear aperture and
the rim, and 45 degree protective bevels of face width ``bevel`` on the
rim edges that face air. Between clear aperture and rim the surface is
by default continued in a way that is polished together with the optical
surface (``land="continue"``): a sphere goes on as the same sphere, a
toroid as the same toroid, an asphere as its best fit sphere (through the
vertex and the rim of the clear aperture, where it joins the asphere), a
freeform as itself; a flat annulus next to a curved surface cannot be
polished with it. Only a surface that cannot reach the rim (a sphere of
smaller radius than the rim, a steep face of a thick meniscus) and Fresnel
surfaces get a flat land. ``land="flat"`` gives the flat annulus perpendicular to
the axis at the height of the clear aperture rim (on a freeform at the
rim point farthest towards air, joined by a short step), e.g. for molded
or diamond turned parts, ``"extend"`` the optical surface itself up to the
rim, aspheres included. Cemented interfaces keep sharp edges. A slab
whose edge would get thinner than ``min_edge_thickness`` raises a
ValueError.

Mirrors (also freeform ones) get a substrate of thickness
``Mechanics.mirror_thickness`` with a flat back. Ideal lenses, coordinate
breaks and dummy surfaces have no geometry.

Off-axis mirrors
----------------
The mirrors of an unobscured system are off-axis segments of a coaxial
parent (an off-axis paraboloid, the three mirrors of a field-biased TMA,
the mirrors of an EUV projection lens with a ring field): only a part of
the parent surface away from its vertex carries light. With
``Mechanics(outline=...)`` such a mirror becomes a substrate cut to the
used area instead of a disk centred on the parent axis. The used area is
the footprint of the beams (rays on rings in the pupil at every field of
``footprint_fields``), and the outline around it is

- ``"circle"``: a circle around the footprint, grown by ``margin``,
- ``"rectangle"``: the bounding box of the footprint, grown by ``margin``,
- ``"hull"``: the convex hull of the footprint, grown by ``margin`` (with
  round corners), e.g. for the kidney shaped footprints of a ring field,
- a (K, 2) array: a polygon in the local x-y plane of the surface (system
  units), taken as it is.

The substrate is a prism over the outline along its mechanical axis, by
default the normal of the mirror at the centre of the footprint
(``axis="normal"``, the side walls perpendicular to the mirror as on a
real off-axis blank; ``"parent"``: parallel to the parent axis), front cut
by the mirror surface and back by a plane perpendicular to the axis
``mirror_thickness`` behind the deepest point of the mirror face, so that
the substrate is at least that thick. The margin is the surface itself
continued, the parent conic, asphere or freeform, which is what is
polished or diamond turned together with the used area of an off-axis
segment (a best fit sphere about the parent axis, as on the rim of a
coaxial lens, has nothing to do with a segment far from the axis).
Spherical and plane mirrors stay exact; other faces are cubic B-spline
height fields over the outline, refined until they deviate less than
``tolerance`` from the surface. An outline that reaches beyond where the
surface is defined (e.g. beyond the freeform radius of a
`qaspheres.QconQ2DSurface`) raises a ValueError. Flat lands and bevels are
not available with outlines.

Mounts
------
`Mount` adds a barrel, retaining rings, spacers and a stop plate to a
coaxial system of lenses, cut to the lens solids so that they sit on the
lands and bevels, with a bore that runs conically from the clear aperture
of one lens to the next and so stays outside the beam. `plot_section()`
draws a section of the model with matplotlib.
Notebook 415 (CAD export of lens systems) shows all of it.
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
    land : {"continue", "flat", "extend"}, optional
        Between clear aperture and rim: the surface continued so that it
        can be polished in one piece with the optical surface (default:
        spheres, planes, toroids, biconics and freeforms go on as
        themselves, aspheres as their best fit sphere through the vertex
        and the rim of the clear aperture, Fresnel surfaces flat), a flat
        annulus perpendicular to the axis ("flat"), or the optical surface
        itself, aspheres included ("extend").
    bevel : number or (front, back), optional
        Face width of the 45 degree bevels on the rim edges of the surfaces
        that face air (0: sharp edges).
    mirror_thickness : number, optional
        Substrate thickness of a mirror, from the rim of the mirror face to
        its flat back.
    min_edge_thickness : number, optional
        Smallest allowed axial edge thickness of a slab (after the bevels).
    outline : {None, "circle", "rectangle", "hull"} or (K, 2) array, optional
        Mirrors only: cut the substrate to an outline around the footprint
        of the beams, grown by margin (see "Off-axis mirrors" in the module
        docstring), or to a polygon in the local x-y plane of the surface.
        None: a disk of semi_diameter centred on the axis of the surface.
    axis : {"normal", "parent"}, optional
        Mechanical axis of a substrate with an outline: the normal of the
        mirror at the centre of the footprint, or the axis of the surface
        (of the parent).
    """

    semi_diameter: Optional[float] = None
    margin: float = 1.0
    land: str = "continue"
    bevel: object = 0.0
    mirror_thickness: float = 5.0
    min_edge_thickness: float = 0.1
    outline: object = None
    axis: str = "normal"

    def bevels(self):
        b = self.bevel
        return (float(b), float(b)) if np.isscalar(b) else (float(b[0]), float(b[1]))


@dataclass
class Mount:
    """A simple lens barrel for coaxial lens systems.

    A tube whose bore is the largest rim diameter plus a radial clearance,
    a retaining ring in front of the first and behind the last element,
    spacers between the elements and an aperture plate at a stop that lies
    in air. Rings and spacers fill the bore from spacer_bore to the barrel
    wall and are cut by the lenses, so they sit on the lands (or on the
    optical surface outside the clear aperture for land="extend") and on
    the bevels. Meant as a starting point for the mechanical design, not as
    a toleranced mount.

    Parameters
    ----------
    radial_clearance : number, optional
        Bore radius minus the largest rim semi-diameter.
    wall : number, optional
        Wall thickness of the barrel.
    ring_length : number, optional
        Axial length of the retaining rings beyond the first and last
        element.
    spacer_clearance : number, optional
        Inner radius of rings and spacers minus the larger clear
        semi-diameter of the elements they touch.
    stop_thickness : number, optional
        Thickness of the aperture plate at a stop in air (0: none).
    """

    radial_clearance: float = 0.05
    wall: float = 2.0
    ring_length: float = 2.0
    spacer_clearance: float = 0.3
    stop_thickness: float = 0.3


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
    mount : list of build123d shapes
        Barrel, retaining rings, spacers and stop plate (see `Mount`).
    """

    lenses: list = _field(default_factory=list)
    rays: list = _field(default_factory=list)
    image: object = None
    mount: list = _field(default_factory=list)

    def compound(self):
        """All parts as one labelled build123d.Compound (an assembly tree)."""

        import build123d as bd

        children = []
        if self.lenses:
            children.append(bd.Compound(children=list(self.lenses), label="lenses"))
        if self.mount:
            children.append(bd.Compound(children=list(self.mount), label="mount"))
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
    if hasattr(s, "section") and getattr(s, "model", None) != "ideal" and r0 == 0.0:
        return _section_edges(s, r1, z_offset, scale)
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


def _section_edges(s, r1, z_offset, scale, n_per_zone=16):
    """Edges of a stepped profile from its section() (Fresnel facets and
    risers): a B-spline per facet, lines for the risers."""

    import build123d as bd

    y, z = s.section(r1, n_per_zone)
    keep = y >= 0
    r, z = y[keep], z[keep]
    if r[0] > 0:
        r, z = np.r_[0.0, r], np.r_[z[0], z]
    step = np.r_[True, np.hypot(np.diff(r), np.diff(z)) > 1e-12]
    r, z = r[step], z[step]
    pts = [bd.Vector(a*scale, 0.0, (b + z_offset)*scale) for a, b in zip(r, z)]
    edges, run = [], [pts[0]]
    for k in range(1, len(pts)):
        if r[k] > r[k - 1] + 1e-12:  # still on the facet
            run.append(pts[k])
            continue
        edges.append(_run_edge(run))
        edges.append(bd.Edge.make_line(pts[k - 1], pts[k]))  # riser
        run = [pts[k]]
    edges.append(_run_edge(run))
    # the draft part of the risers joins a riser top to the next facet:
    out = []
    for e in edges:
        if e is None:
            continue
        if out and (out[-1].end_point() - e.start_point()).length > 1e-9:
            out.append(bd.Edge.make_line(out[-1].end_point(), e.start_point()))
        out.append(e)
    return out


def _run_edge(run):
    import build123d as bd

    if len(run) < 2:
        return None
    if len(run) == 2:
        return bd.Edge.make_line(*run)
    return bd.Edge.make_spline(run)


def _tangent(s, r, z_offset, scale):
    import build123d as bd

    dr = 1e-6*max(r, 1.0)
    dz = (s.sag(np.array([(r + dr)**2]))[0] - s.sag(np.array([(r - dr)**2]))[0])/(2*dr)
    return bd.Vector(1.0, 0.0, float(dz))


def _rim_sag(s, r):
    return float(s.sag(np.array([r**2]))[0])


def _continuation(s, r_clear, land, rm=None):
    """What carries a rotationally symmetric surface s on beyond its clear
    semi-diameter r_clear: (surface, dz), whose sag plus dz continues it.

    - land "continue": a sphere or plane goes on as itself; an asphere,
      conic, Q-type asphere or axicon as its best fit sphere, the sphere
      through the vertex and the rim of the clear aperture (so the
      continuation joins the optical surface there); a Fresnel surface
      (stepped section), or a sphere that does not reach the rim
      semi-diameter rm, as the plane at its rim height,
    - "flat": the plane perpendicular to the axis at the rim height,
    - "extend": the surface itself.
    """

    if land == "extend":
        return s, 0.0
    z = _rim_sag(s, r_clear)
    if land == "flat" or hasattr(s, "section"):
        return rt.Surface(np.inf), z
    if land != "continue":
        raise ValueError("land must be 'continue', 'flat' or 'extend', not {!r}.".format(land))
    if _sphere_or_plane(s) is not None:
        cont = s
    elif z == 0.0:
        cont = rt.Surface(np.inf)
    else:
        cont = rt.Surface((r_clear**2 + z**2)/(2*z))
    if rm is not None and not np.isfinite(_rim_sag(cont, rm)):
        # a sphere smaller than the rim (a steep face with a small clear
        # aperture on a large element) cannot go on: a flat land
        return rt.Surface(np.inf), z
    return cont, 0.0


def _surface_edges(s, cont, dz, r_clear, r_end, z_offset, scale, tolerance):
    """Section edges of s from the axis to r_clear and of its continuation
    (cont, dz) from r_clear to r_end."""

    if cont is s:
        return _profile_edges(s, 0.0, r_end, z_offset, scale, tolerance)
    return (_profile_edges(s, 0.0, r_clear, z_offset, scale, tolerance)
            + _profile_edges(cont, r_clear, r_end, z_offset + dz, scale, tolerance))


def _outer_rim_sags(s, r_clear, rm, land, n=181):
    """Sag (local frame) around the rim of semi-diameter rm of the surface
    as continued beyond r_clear (for flat lands on freeforms: at r_clear)."""

    if _is_freeform(s):
        return _rim_sags(s, r_clear if land == "flat" else rm, n)
    cont, dz = _continuation(s, r_clear, land, rm)
    return np.full(n, _rim_sag(cont, rm) + dz)


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
    (c1, dz1), (c2, dz2) = _continuation(s1, r1, land, rm), _continuation(s2, r2, land, rm)
    # the continued surfaces end where the bevels begin:
    e1, e2 = rm - b1, rm - b2
    if e1 < r1 - 1e-12 or e2 < r2 - 1e-12:
        raise ValueError("Bevels ({}, {}) reach into the clear aperture ({}, {}) at the rim "
                         "semi-diameter {}.".format(b1, b2, r1, r2, rm))
    e1, e2 = max(e1, r1), max(e2, r2)
    z1 = _rim_sag(c1, e1) + dz1
    z2 = t + _rim_sag(c2, e2) + dz2
    if not (np.isfinite(z1) and np.isfinite(z2)):
        raise ValueError("Surface sag undefined at the clear aperture or rim.")
    edge = sgn*(z2 - z1) - b1 - b2
    if edge < min_edge:
        raise ValueError("Edge thickness {:.3g} below {} at semi-diameter {} between {} and {}."
                         .format(edge, min_edge, rm, s1, s2))

    def P(r, z):
        return bd.Vector(r*scale, 0.0, z*scale)

    edges = _surface_edges(s1, c1, dz1, r1, e1, 0.0, scale, tolerance)
    # front bevel, rim, back bevel:
    pts = [P(e1, z1), P(rm, z1 + sgn*b1), P(rm, z2 - sgn*b2), P(e2, z2)]
    for a, b in zip(pts[:-1], pts[1:]):
        if (a - b).length > 1e-9:
            edges.append(bd.Edge.make_line(a, b))
    back = _surface_edges(s2, c2, dz2, r2, e2, t, scale, tolerance)
    edges += [e.reversed() for e in back[::-1]]
    edges.append(bd.Edge.make_line(P(0.0, t), P(0.0, 0.0)))
    face = bd.Face(bd.Wire(edges))

    return bd.Solid.revolve(face, 360.0, bd.Axis.Z)


def _rim_sags(s, r, n=181):
    """Sag on the circle of radius r (local frame), shape (n,)."""

    if _is_freeform(s):
        phi = np.linspace(0, 2*np.pi, n)
        return np.asarray(s.sag_xy(r*np.cos(phi), r*np.sin(phi)), float)
    return np.full(n, _rim_sag(s, r))


def _surface_face(s, rm, scale, tolerance, r_clear=None, land="flat", n_grid=41):
    """Face of a surface in its local frame (mm), continued to beyond the
    rim semi-diameter rm, to split solids with (a rotationally symmetric
    surface as `_continuation()` from r_clear on, except for flat lands,
    which are cut separately)."""

    import build123d as bd

    ext = 1.15*rm
    if _is_freeform(s):
        # B-spline surface through the sag on a square grid; where the sag
        # is undefined outside the rim it is continued radially:
        x = np.linspace(-ext, ext, n_grid)
        X, Y = np.meshgrid(x, x, indexing="ij")
        with np.errstate(invalid="ignore"):
            Z = np.asarray(s.sag_xy(X, Y), float)
        bad = ~np.isfinite(Z)
        if np.any(bad):
            f = np.minimum(1.0, 1.02*rm/np.maximum(np.hypot(X, Y), 1e-300))
            Z = np.where(bad, s.sag_xy(X*f, Y*f), Z)
        if not np.all(np.isfinite(Z)):
            raise ValueError("sag_xy of {} undefined up to the rim ({}).".format(s, rm))
        pts = [[bd.Vector(X[i, j]*scale, Y[i, j]*scale, Z[i, j]*scale) for j in range(n_grid)]
               for i in range(n_grid)]
        return bd.Face.make_surface_from_array_of_points(pts, tol=1e-6, min_deg=3, max_deg=5)
    # rotationally symmetric: revolve the section, continued beyond the
    # clear aperture and flat beyond the rim (or where the sag ends)
    if land == "flat" or r_clear is None:
        cont, dz, r_clear, r_end = s, 0.0, rm, rm
    else:
        cont, dz = _continuation(s, r_clear, land, rm)
        r_end = ext if np.isfinite(_rim_sag(cont, ext)) else rm
    edges = _surface_edges(s, cont, dz, r_clear, r_end, 0.0, scale, tolerance)
    z = _rim_sag(cont, r_end) + dz
    if r_end < ext:
        edges.append(bd.Edge.make_line(bd.Vector(r_end*scale, 0, z*scale),
                                       bd.Vector(ext*scale, 0, z*scale)))
    return bd.Face.revolve(bd.Wire(edges), 360.0, bd.Axis.Z) if hasattr(bd.Face, "revolve") \
        else bd.Shell.revolve(bd.Wire(edges), 360.0, bd.Axis.Z)


def _pick(parts, ref):
    """The solid of a split that contains (is closest to) the point ref."""

    parts = parts if isinstance(parts, (tuple, list)) else [parts]
    parts = [q for p in parts for q in (p if isinstance(p, (tuple, list)) else [p])]
    solids = [so for part in parts if part is not None for so in part.solids()]
    return min(solids, key=lambda p: p.distance_to(ref))


def _one_solid(shape):
    import build123d as bd

    solids = shape.solids() if hasattr(shape, "solids") else [shape]
    if len(solids) != 1:
        raise ValueError("Boolean operation gave {} solids.".format(len(solids)))
    return bd.Solid(solids[0].clean().wrapped) if hasattr(solids[0], "clean") else solids[0]


def _side(s, frame, axis, r_clear, rm, land, z_land, ref_in, ref_out, depth, scale,
          tolerance):
    """The part of the cylinder of semi-diameter rm about the axis frame on
    the glass side of surface s (continued beyond r_clear, or with a flat
    land from r_clear to rm at the local height z_land), global, mm. ref_in, ref_out: points in the
    glass inside the clear aperture and in the land annulus."""

    import build123d as bd

    loc = _location(*frame, scale)
    aloc = _location(*axis, scale)
    face = loc*_surface_face(s, rm, scale, tolerance, r_clear, land)
    plane = bd.Plane(origin=(0, 0, -depth*scale))
    if land != "flat" or r_clear >= rm:
        cyl = aloc*bd.Solid.make_cylinder(rm*scale, 2*depth*scale, plane)
        return _pick(cyl.split(face, keep=bd.Keep.BOTH), ref_in)
    inner = aloc*bd.Solid.make_cylinder(r_clear*scale, 2*depth*scale, plane)
    ring = aloc*(bd.Solid.make_cylinder(rm*scale, 2*depth*scale, plane)
                 - bd.Solid.make_cylinder(r_clear*scale, 2*depth*scale, plane))
    ring = _one_solid(ring)
    land_face = loc*bd.Face.make_rect(4*rm*scale, 4*rm*scale,
                                      bd.Plane(origin=(0, 0, z_land*scale)))
    inner = _pick(inner.split(face, keep=bd.Keep.BOTH), ref_in)
    ring = _pick(ring.split(land_face, keep=bd.Keep.BOTH), ref_out)
    return _one_solid(inner + ring)


def _cut_slab(s1, f1, s2, f2, g, r1, r2, rm, mech, bevels, scale, tolerance):
    """Slab between surface s1 (frame f1, the axis of the slab) and s2
    (frame f2), glass in direction g = +-1 of the local z of s1; cut from a
    cylinder of semi-diameter rm. Global coordinates, mm."""

    import build123d as bd

    (o1, R1), (o2, _) = f1, f2
    land = mech.land
    t = float(R1[:, 2] @ (o2 - o1))  # axial distance of the vertices
    rim1 = _outer_rim_sags(s1, r1, rm, land)
    rim2 = _outer_rim_sags(s2, r2, rm, land)
    # flat lands flush with the rim point that lies farthest towards air:
    z1 = np.min(g*rim1)*g
    z2 = np.max(g*rim2)*g
    b1, b2 = bevels
    edge = g*(t + z2 - z1) if land == "flat" else np.min(g*(t + rim2 - rim1))
    edge -= b1 + b2
    if edge < mech.min_edge_thickness:
        raise ValueError("Edge thickness {:.3g} below {} at semi-diameter {} between {} and {}."
                         .format(edge, mech.min_edge_thickness, rm, s1, s2))
    depth = abs(t) + 2*rm + mech.mirror_thickness

    def glob(r, z):
        return bd.Vector(*((o1 + R1 @ [r, 0.0, z])*scale))

    # points in the glass on the axis and in the middle of the land annulus:
    ref_in = glob(0.0, 0.5*t)
    ref_out = glob(0.5*(min(r1, r2) + rm), 0.5*(z1 + t + z2))
    side1 = _side(s1, f1, f1, r1, rm, land, z1, ref_in, ref_out, depth, scale, tolerance)
    side2 = _side(s2, f2, f1, r2, rm, land, z2, ref_in, ref_out, depth, scale, tolerance)
    solid = _one_solid(side1 & side2)
    if b1 or b2:
        solid = _bevel_rim(solid, f1, rm, 0.5*t, g, b1, b2, scale)
    return solid


def _bevel_rim(solid, axis, rm, z_mid, g, b1, b2, scale):
    """45 degree chamfers of face width b1 (front, towards -g) and b2 (back)
    on the edges of the rim cylinder of semi-diameter rm."""

    o, R = axis
    def radial(p):
        q = (np.array(tuple(p))/scale - o) @ R
        return np.hypot(q[0], q[1])

    rim = [f for f in solid.faces() if f.geom_type.name == "CYLINDER"
           and abs(radial(f.position_at(0.5, 0.5)) - rm) < 1e-6*max(1.0, rm)]
    front, back = [], []
    for f in rim:
        for e in f.edges():
            if e.geom_type.name == "LINE":  # seam
                continue
            c = (np.array(tuple(e.center()))/scale - o) @ R
            (front if g*(c[2] - z_mid) < 0 else back).append(e)
    for b, edges in ((b1, front), (b2, back)):
        if b and edges:
            solid = solid.chamfer(b*scale, None, edges)
    return solid


def system_model(system, fields=None, n_rays=7, pupil="y", mechanics=None, rays="wire",
                 ray_radius=0.02, ray_start=None, image_plane=True, mount=None,
                 tolerance=1e-4, footprint_fields=None):
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
    mount : Mount, optional
        Add a barrel with retaining rings, spacers and a stop plate
        (coaxial systems of lenses only).
    tolerance : number, optional
        Largest deviation (mm) of B-spline profiles from the sag.
    footprint_fields : sequence, optional
        Fields (numbers or (x, y) pairs) whose beams make up the footprints
        of mirrors with a `Mechanics.outline`, e.g. the corners and edges
        of a rectangular or ring field (default: fields).

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

    envelope = traced + (_trace(system, fields, 24, "ring", ray_start) if mount is not None else [])
    parts = []  # (solids, rim and clear semi-diameters) per element, for the mount
    for k, (first, last) in enumerate(system.elements()):
        mech = mechanics.get(k, default)
        if mech.outline is not None:
            if not (first == last and surfaces[first].mirror):
                raise NotImplementedError("Outlines for mirrors only (element {}).".format(k))
            fp = footprint_fields if footprint_fields is not None else _fields(system, fields)
            solid = _outline_mirror(system, first, mech, fp, scale, tolerance, sigma[first])
            solid.label, solid.color = "M{}".format(k + 1), bd.Color(0.75, 0.75, 0.8)
            model.lenses.append(solid)
            parts.append(None)
            continue
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
            parts.append(None)
            continue
        if first == last:  # ideal lens: no geometry
            continue
        # beam radius in the vertex planes of the first and last surface
        # (before and behind the element), for the bore of the mount:
        # behind a concave surface the beam spreads beyond the clear
        # aperture before it reaches the vertex plane
        parts.append(([], rm, r_clear, _beam_radius(envelope, first, system.vertex_z[first], True),
                      _beam_radius(envelope, last, system.vertex_z[last], False)))
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
                if isinstance(s2, rt.CoordinateBreak):
                    raise NotImplementedError("Coordinate breaks inside glass.")
                solid = _cut_slab(s1, frames[i], s2, frames[i + 1], sigma[i + 1], clear(i),
                                  clear(i + 1), rm, mech, (b1, b2), scale, tolerance)
            solid.label = name + ("abcdefgh"[j] if len(slabs) > 1 else "")
            solid.color = bd.Color(0.55, 0.75, 0.95, 0.5)
            model.lenses.append(solid)
            parts[-1][0].append(solid)

    if mount is not None:
        if system.tilted or any(p is None for p in parts):
            raise NotImplementedError("Mounts for tilted or folded systems or mirrors.")
        stop = surfaces[system.stop_index]
        stop_z = system.vertex_z[system.stop_index]
        in_air = system._is_gap(n[system.stop_index]) and system._is_gap(n[system.stop_index + 1])
        stop_plate = (stop_z, clear(system.stop_index)) if in_air and stop.semi_diameter else None
        model.mount = _mount(parts, mount, stop_plate, scale)

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
    """Mirror substrate: the mirror face, continued (or a land) to the rim and a flat back
    mirror_thickness behind the rim (seen from the incoming light)."""

    s = system.surfaces[i]
    frame = system.frames()[0][i]
    back = rt.Surface(np.inf)
    rim = _outer_rim_sags(s, r_clear, rm, mech.land)
    # flat back mirror_thickness behind the rim point farthest behind:
    t = np.max(sigma_in*rim)*sigma_in + sigma_in*mech.mirror_thickness
    b1, _ = mech.bevels()
    if _is_rotational(s):
        solid = _revolved_slab(s, back, t, r_clear, rm, rm, mech.land, (b1, 0.0), scale,
                               tolerance, mech.min_edge_thickness)
        return _location(*frame, scale)*solid
    o, R = frame
    back_frame = (o + R @ [0.0, 0.0, t], R)
    return _cut_slab(s, frame, back, back_frame, sigma_in, r_clear, rm, rm, mech, (b1, 0.0),
                     scale, tolerance)


# --- off-axis mirrors ---------------------------------------------------------

def _sag2(s, x, y):
    """Sag of s at the local points (x, y), freeform or not."""

    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    z = s.sag_xy(x.ravel(), y.ravel()) if _is_freeform(s) else s.sag(x.ravel()**2 + y.ravel()**2)
    return np.asarray(z, float).reshape(x.shape)


def _normal(s, x, y):
    """Unit normal of s at (x, y) (local frame), pointing towards +z."""

    h = 1e-6*max(1.0, np.hypot(x, y))
    gx = (_sag2(s, x + h, y) - _sag2(s, x - h, y))/(2*h)
    gy = (_sag2(s, x, y + h) - _sag2(s, x, y - h))/(2*h)
    n = np.array([-float(gx), -float(gy), 1.0])
    return n/np.linalg.norm(n)


def _lift(s, origin, R, ab, iterations=30):
    """Points of the surface s (local frame of s) on the lines through
    origin + R @ (a, b, 0) along the third column of R (Newton)."""

    ab = np.atleast_2d(ab)
    base = origin + ab[:, :1]*R[:, 0] + ab[:, 1:2]*R[:, 1]
    n = R[:, 2]
    t = np.zeros(len(ab))
    for _ in range(iterations):
        P = base + t[:, None]*n
        f = P[:, 2] - _sag2(s, P[:, 0], P[:, 1])
        h = 1e-7*max(1.0, float(np.max(np.abs(base))))
        Q = P + h*n
        df = (Q[:, 2] - _sag2(s, Q[:, 0], Q[:, 1]) - f)/h
        step = f/df
        t -= step
        if np.all(np.abs(step) < 1e-12*max(1.0, float(np.max(np.abs(base))))):
            break
    if not np.all(np.isfinite(t)):
        raise ValueError("Surface {} undefined over the outline.".format(s))
    return base + t[:, None]*n


def _footprint(system, i, fields, rings=(0.5, 1.0), spokes=48):
    """Hits (N, 3) of the beams of all fields on surface i, local frame."""

    a = np.linspace(0, 2*np.pi, spokes, endpoint=False)
    px = np.r_[0.0, np.concatenate([r*np.cos(a) for r in rings])]
    py = np.r_[0.0, np.concatenate([r*np.sin(a) for r in rings])]
    o, R = system.frames()[0][i]
    hits = []
    for f in fields:
        f = tuple(f) if np.ndim(f) else f
        P = system.trace_pupil(f, px, py).points[i + 1]
        hits.append((P - o) @ R)
    hits = np.concatenate(hits)
    hits = hits[np.all(np.isfinite(hits), axis=1)]
    if len(hits) < 3:
        raise ValueError("No footprint on surface {}: no rays of the fields reach it."
                         .format(i))
    return hits


def _outline_2d(outline, ab, margin, n_arc=96):
    """The outline in the mechanical plane: ("circle", centre, radius),
    ("rectangle", lo, hi) or ("polygon", (K, 2) points, margin)."""

    if isinstance(outline, str):
        if outline == "circle":
            # the smallest circle around the footprint
            from scipy.optimize import minimize
            from .off_axis import _hull

            h = _hull(ab)
            c = minimize(lambda c: np.max(np.hypot(*(h - c).T)), h.mean(axis=0),
                         method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12}).x
            return "circle", c, float(np.max(np.hypot(*(h - c).T))) + margin
        if outline == "rectangle":
            return "rectangle", ab.min(axis=0) - margin, ab.max(axis=0) + margin
        if outline == "hull":
            from .off_axis import _hull
            return "polygon", _hull(ab), margin
        raise ValueError("outline must be 'circle', 'rectangle', 'hull' or an array, not {!r}."
                         .format(outline))
    return "polygon", ab, 0.0


def _outline_points(kind, a, b, n=96):
    """Points on the outline, counterclockwise (closely spaced on curves
    and along the edges of rectangles)."""

    if kind == "circle":
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        return a + b*np.stack([np.cos(t), np.sin(t)], axis=1)
    if kind == "rectangle":
        corners = np.array([a, [b[0], a[1]], b, [a[0], b[1]]])
        pts = corners
    else:
        pts, margin = a, b
        if margin:
            from .off_axis import _hull

            t = np.linspace(0, 2*np.pi, n, endpoint=False)
            ring = margin*np.stack([np.cos(t), np.sin(t)], axis=1)
            return _hull((pts[:, None, :] + ring[None]).reshape(-1, 2))
    s = np.linspace(0, 1, n//4, endpoint=False)[:, None]
    return np.concatenate([p + s*(q - p) for p, q in zip(pts, np.roll(pts, -1, axis=0))])


def _outline_face(kind, a, b, scale):
    """Planar face of the outline in the x-y plane (mm)."""

    import build123d as bd

    if kind == "circle":
        edge = bd.Edge.make_circle(b*scale, bd.Plane(origin=(a[0]*scale, a[1]*scale, 0.0)))
        return bd.Face(bd.Wire([edge]))
    if kind == "rectangle":
        c, w = (a + b)/2, b - a
        return bd.Face.make_rect(w[0]*scale, w[1]*scale,
                                 bd.Plane(origin=(c[0]*scale, c[1]*scale, 0.0)))
    pts, margin = a, b
    if not margin:
        return bd.Face(bd.Wire.make_polygon([bd.Vector(x*scale, y*scale, 0.0) for x, y in pts],
                                            close=True))
    # a grown hull: one periodic B-spline through the boundary of the hull
    # with disks of radius margin on its vertices, evenly resampled (one
    # side face instead of a face per edge and corner arc)
    from .off_axis import _hull

    ring = _outline_points(kind, pts, margin)
    closed = np.vstack([ring, ring[:1]])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
    t = np.linspace(0, s[-1], 97)[:-1]
    xy = np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], axis=1)
    edge = bd.Edge.make_spline([bd.Vector(x*scale, y*scale, 0.0) for x, y in xy], periodic=True)
    return bd.Face(bd.Wire([edge]))


def _patch_face(s, P0, Rm, rim, scale, tolerance, grids=(17, 33, 65, 129)):
    """B-spline face of the surface s over the outline rim (K, 2) in the
    mechanical frame (origin P0 and axes Rm in the local frame of s): a
    cubic B-spline height field that interpolates the heights of s along
    the mechanical axis on a grid over the bounding box of the outline,
    grown by 5 %, refined until it deviates less than tolerance (mm) from
    the surface halfway between the grid points inside the outline.
    Outside the outline grown by 10 % the height is continued along the
    radius from the centre of the outline with the slope there, so that a
    polynomial sag that grows fast beyond its normalization radius (Q
    polynomials, Zernikes) does not bend the face where the substrate does
    not need it. Mechanical frame, mm."""

    import build123d as bd
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    try:
        from OCP.collections import Array2_double as Array2
    except ImportError:  # older OCP
        from OCP.TColStd import TColStd_Array2OfReal as Array2

    lo, hi = rim.min(axis=0), rim.max(axis=0)
    c, d = (lo + hi)/2, 0.05*(hi - lo).max()
    db = rim - c
    phi_b, rho_b = np.arctan2(db[:, 1], db[:, 0]), np.hypot(db[:, 0], db[:, 1])
    k = np.argsort(phi_b)

    def grid(a, b):
        """Points (u from c) and the radius of the grown outline over |u|."""
        A, B = (v.ravel() for v in np.meshgrid(a, b, indexing="ij"))
        u = np.stack([A - c[0], B - c[1]], axis=1)
        lim = 1.1*np.interp(np.arctan2(u[:, 1], u[:, 0]), phi_b[k], rho_b[k], period=2*np.pi)
        return u, lim/np.maximum(np.hypot(u[:, 0], u[:, 1]), 1e-300)

    def height(u, g):
        return (_lift(s, P0, Rm, c + u*g[:, None]) - P0) @ Rm[:, 2]

    for n in grids:
        a = np.linspace(lo[0] - d, hi[0] + d, n)
        b = np.linspace(lo[1] - d, hi[1] + d, n)
        u, f = grid(a, b)
        f = np.minimum(f, 1.0)
        h = height(u, f)
        out = f < 1.0
        if np.any(out):
            # continued along the radius with the slope at the limit (C1)
            dh = (h[out] - height(u[out], 0.98*f[out]))/(0.02*f[out])
            h[out] += dh*(1.0 - f[out])
        h = h.reshape(n, n)*scale
        Z = Array2(1, n, 1, n)
        for i in range(n):
            for j in range(n):
                Z.SetValue(i + 1, j + 1, float(h[i, j]))
        fit = GeomAPI_PointsToBSplineSurface()
        fit.Interpolate(Z, a[0]*scale, (a[1] - a[0])*scale, b[0]*scale, (b[1] - b[0])*scale)
        surface = fit.Surface()
        # check halfway between the grid points, inside the outline:
        um, fm = grid((a[:-1] + a[1:])/2, (b[:-1] + b[1:])/2)
        um = um[fm >= 1.1]  # inside the outline itself
        if len(um) == 0:
            break
        x, y = ((c + um)*scale).T
        err = np.abs([surface.Value(xi, yi).Z() for xi, yi in zip(x, y)]
                     - height(um, np.ones(len(um)))*scale)
        if np.max(err) < tolerance:
            break

    return bd.Face(BRepBuilderAPI_MakeFace(surface, 1e-7).Face())


def _outline_mirror(system, i, mech, fields, scale, tolerance, sigma_in):
    """Substrate of mirror i cut to an outline around its footprint (or to
    a given polygon): a prism along the mechanical axis between the mirror
    face and a flat back (global, mm)."""

    s = system.surfaces[i]
    if mech.land == "flat":
        raise NotImplementedError("Flat lands on mirrors with an outline.")
    if any(mech.bevels()):
        raise NotImplementedError("Bevels on mirrors with an outline.")
    if mech.axis not in ("normal", "parent"):
        raise ValueError("axis must be 'normal' or 'parent', not {!r}.".format(mech.axis))
    o, R = system.frames()[0][i]
    if isinstance(mech.outline, str):
        xy = _footprint(system, i, fields)[:, :2]
    else:
        xy = np.asarray(mech.outline, float)
    # mechanical frame: origin on the mirror at the centre of the used area
    c = (xy.min(axis=0) + xy.max(axis=0))/2
    P0 = np.array([c[0], c[1], float(_sag2(s, c[0], c[1]))])
    n = _normal(s, *c) if mech.axis == "normal" else np.array([0.0, 0.0, 1.0])
    v = np.array([0.0, 1.0, 0.0]) - n[1]*n
    v /= np.linalg.norm(v)
    Rm = np.stack([np.cross(v, n), v, n], axis=1)
    # used area in the mechanical plane (along the mechanical axis onto it):
    S = np.stack([xy[:, 0], xy[:, 1], _sag2(s, xy[:, 0], xy[:, 1])], axis=1)
    ab = ((S - P0) @ Rm)[:, :2]
    kind, a, b = _outline_2d(mech.outline, ab, mech.margin)
    rim = _outline_points(kind, a, b)
    # the mirror face over the outline, heights along the axis:
    face_pts = np.concatenate([_lift(s, P0, Rm, rim), S])
    h = (face_pts - P0) @ Rm[:, 2]
    g = sigma_in  # the substrate lies in direction g of the axis
    h_back = g*(np.max(g*h) + mech.mirror_thickness)
    pad = 0.05*(np.ptp(rim, axis=0).max() + abs(h_back)) + 1e-3
    h_front = g*(np.min(g*h) - pad)
    # the surface face must cover the prism:
    corners = np.concatenate([P0 + rim[:, :1]*Rm[:, 0] + rim[:, 1:]*Rm[:, 1] + z*Rm[:, 2]
                              for z in (h_front, h_back)])

    import build123d as bd

    r_max = 1.05*float(np.max(np.hypot(corners[:, 0], corners[:, 1])))
    mloc = _location(o + R @ P0, R @ Rm, scale)
    if _sphere_or_plane(s) and np.isfinite(_rim_sag(s, r_max)):
        # exact: a revolved arc or line
        mirror_face = _location(o, R, scale)*_surface_face(s, r_max, scale, tolerance,
                                                           land="flat")
    else:
        mirror_face = mloc*_patch_face(s, P0, Rm, rim, scale, tolerance)
    base = bd.Pos(0, 0, h_front*scale)*_outline_face(kind, a, b, scale)
    prism = mloc*bd.Solid.extrude(base, bd.Vector(0, 0, (h_back - h_front)*scale))
    ref = bd.Vector(*((o + R @ (P0 + (h_back - 0.25*g*mech.mirror_thickness)*Rm[:, 2]))
                      *scale))
    return _one_solid(_pick(prism.split(mirror_face, keep=bd.Keep.BOTH), ref))


def _mount(parts, mount, stop_plate, scale):
    """Barrel, retaining rings, spacers and stop plate (global, mm) for
    coaxial elements; parts: (solids, rim, clear) per element in order."""

    import build123d as bd

    metal = bd.Color(0.35, 0.35, 0.38)
    rb = max(p[1] for p in parts) + mount.radial_clearance
    boxes = [bd.Compound(children=list(p[0])).bounding_box() for p in parts]
    zmin = [b.min.Z for b in boxes]
    zmax = [b.max.Z for b in boxes]
    # rings and spacers meet in the middle of each rim, the part of the
    # element outside the inner radius of the spacers:
    centres = []
    for p, a, b in zip(parts, zmin, zmax):
        r_in = min(p[2] + mount.spacer_clearance, 0.5*(p[2] + p[1]))*scale
        tube = bd.Solid.make_cylinder(p[1]*scale + 1, b - a + 2, bd.Plane(origin=(0, 0, a - 1))) \
            - bd.Solid.make_cylinder(r_in, b - a + 2, bd.Plane(origin=(0, 0, a - 1)))
        rim = bd.Compound(children=[x for solid in p[0] for x in (solid & tube).solids()])
        box = rim.bounding_box()
        centres.append(0.5*(box.min.Z + box.max.Z))
    L = mount.ring_length*scale
    z_front, z_back = zmin[0] - L, zmax[-1] + L
    R = rb*scale

    def ring(r_in, z0, z1):
        return bd.Solid.make_cylinder(R, z1 - z0, bd.Plane(origin=(0, 0, z0))) - \
            bd.Solid.make_cylinder(r_in*scale, z1 - z0, bd.Plane(origin=(0, 0, z0)))

    out = []
    barrel = bd.Solid.make_cylinder(R + mount.wall*scale, z_back - z_front,
                                    bd.Plane(origin=(0, 0, z_front))) - \
        bd.Solid.make_cylinder(R, z_back - z_front, bd.Plane(origin=(0, 0, z_front)))
    barrel.label, barrel.color = "barrel", metal
    out.append(barrel)
    plate = None
    if stop_plate is not None and mount.stop_thickness > 0:
        z, r = stop_plate
        z *= scale
        T = mount.stop_thickness*scale
        plate = ring(r, z, z + T)
        plate.label, plate.color = "stop", bd.Color(0.1, 0.1, 0.1)
    def bore_ring(profile):
        """Solid of revolution between the inner radii profile [(z, r_in)]
        (mm, z increasing) and the bore."""
        pts = [bd.Vector(r, 0, z) for z, r in profile]
        pts += [bd.Vector(R, 0, profile[-1][0]), bd.Vector(R, 0, profile[0][0])]
        face = bd.Face(bd.Wire.make_polygon(pts, close=True))
        return bd.Solid.revolve(face, 360.0, bd.Axis.Z)

    # inner radii in front of and behind every element: the clear aperture
    # or the beam, whichever is larger, plus the clearance
    inner_in = [(max(p[2], p[3]) + mount.spacer_clearance)*scale for p in parts]
    inner_out = [(max(p[2], p[4]) + mount.spacer_clearance)*scale for p in parts]
    bounds = [z_front] + centres + [z_back]
    for k in range(len(bounds) - 1):
        near = [parts[j] for j in (k - 1, k) if 0 <= j < len(parts)]
        z0, z1 = bounds[k], bounds[k + 1]
        if k == 0:
            profile = [(z0, inner_in[0]), (z1, inner_in[0])]
        elif k == len(bounds) - 2:
            profile = [(z0, inner_out[-1]), (z1, inner_out[-1])]
        else:
            # the bore follows the clear apertures: constant up to the back
            # of element k - 1 and from the front of element k, conical
            # between them
            a, b = inner_out[k - 1], inner_in[k]
            za, zb = max(zmax[k - 1], z0), min(zmin[k], z1)
            # (elements that overlap in z, e.g. nested menisci: a
            # cylindrical bore outside both clear apertures, a cone would
            # cut through the glass of the larger one)
            profile = [(z0, a), (za, a), (zb, b), (z1, b)] if zb > za else \
                [(z0, max(a, b)), (z1, max(a, b))]
        piece = bore_ring(profile)
        for p in near:
            for solid in p[0]:
                piece = piece - solid
        if plate is not None and z0 <= plate.bounding_box().center().Z < z1:
            piece = piece - plate
        solids = piece.solids() if hasattr(piece, "solids") else [piece]
        solids = [x for x in solids if x.volume > 1e-9]
        if not solids:
            continue
        name = "ring_front" if k == 0 else "ring_back" if k == len(bounds) - 2 else \
            "spacer_{}".format(k)
        for j, x in enumerate(solids):
            x.label = name + ("abcdefgh"[j] if len(solids) > 1 else "")
            x.color = metal
            out.append(x)
    if plate is not None:
        out.append(plate)
    return out


# --- rays --------------------------------------------------------------------

def _beam_radius(bundles, i, z, before):
    """Largest ray radius in the plane z on the ray segments that reach
    surface i (before=True) or leave it (system units; coaxial)."""

    r = 0.0
    for b in bundles:
        A, B = (b.points[i], b.points[i + 1]) if before else (b.points[i + 1], b.points[i + 2])
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.clip((z - A[:, 2])/(B[:, 2] - A[:, 2]), 0.0, 1.0)
            Q = A + np.where(np.isfinite(t), t, 0.0)[:, None]*(B - A)
        rq = np.hypot(Q[:, 0], Q[:, 1])
        if np.any(np.isfinite(rq)):
            r = max(r, float(np.nanmax(rq)))
    return r


def _fields(system, fields):
    if fields is None:
        return [0.0, system.max_field] if system.max_field else [0.0]
    return fields


def _trace(system, fields, n_rays, pupil, ray_start):
    fields = _fields(system, fields)
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
            # in front of the first surface also where its sag (an off-axis
            # segment) lies in front of the vertex:
            first = pts[1][:, 2]
            if np.any(np.isfinite(first)):
                z0 = min(z0, float(np.nanmin(first)) + z0)
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


def _section_curves(shape, a, n=60):
    """Points (n, 3) along each edge of the section of shape with the plane
    x = 0 (a = 1) or y = 0 (a = 0)."""

    import build123d as bd
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

    normal = gp_Dir(1, 0, 0) if a == 1 else gp_Dir(0, 1, 0)
    section = BRepAlgoAPI_Section(shape.wrapped, gp_Pln(gp_Pnt(0, 0, 0), normal), False)
    section.Approximation(True)
    section.Build()
    if not section.IsDone():
        return []
    edges = bd.Compound(section.Shape()).edges()
    return [np.array([tuple(e.position_at(t)) for t in np.linspace(0, 1, n)]) for e in edges]


def _section_loops(shape, a):
    """Closed polygons of the section: the edge curves chained end to end."""

    curves = _section_curves(shape, a)
    loops = []
    while curves:
        loop = [curves.pop(0)]
        while curves:
            end, start = loop[-1][-1], loop[0][0]
            d = [min(np.linalg.norm(c[0] - end), np.linalg.norm(c[-1] - end)) for c in curves]
            k = int(np.argmin(d))
            if d[k] > np.linalg.norm(start - end) + 1e-9:
                break  # back at the start: the loop is closed
            c = curves.pop(k)
            loop.append(c if np.linalg.norm(c[0] - end) <= np.linalg.norm(c[-1] - end)
                        else c[::-1])
        loops.append(np.concatenate(loop))
    return loops


def plot_section(model, ax=None, plane="yz", rays=True, lw=0.5):
    """Draw the section of a Model through the axis with matplotlib: lenses
    and mount parts filled, the rays projected onto the plane.

    Parameters
    ----------
    model : Model
    ax : matplotlib axis, optional
    plane : {"yz", "xz"}, optional
        Section plane; z is drawn to the right.
    rays : bool, optional
        Draw the rays, projected onto the plane.
    lw : number, optional
        Line width of the rays.

    Returns
    -------
    ax : matplotlib axis
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    if ax is None:
        ax = plt.gca()
    a = 1 if plane == "yz" else 0
    def fill(shape, color, alpha):
        for loop in _section_loops(shape, a):
            ax.add_patch(Polygon(loop[:, [2, a]], closed=True, fc=color, ec="k", lw=0.6,
                                 alpha=alpha))

    for part in model.mount:
        fill(part, (0.12, 0.12, 0.12) if part.label == "stop" else (0.62, 0.62, 0.65), 1.0)
    for lens in model.lenses:
        fill(lens, tuple(lens.color)[:3] if lens.color else (0.55, 0.75, 0.95), 0.85)
    if rays:
        for ray in model.rays:
            color = tuple(ray.color)[:3]
            for e in ray.edges():
                p, q = np.array(tuple(e.start_point())), np.array(tuple(e.end_point()))
                ax.plot([p[2], q[2]], [p[a], q[a]], color=color, lw=lw)
    if model.image is not None:
        for pts in _section_curves(model.image, a):
            ax.plot(pts[:, 2], pts[:, a], color="0.5", ls="--", lw=1)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel(plane[0] + " [mm]")

    return ax


def plot_3d(model, ax=None, elev=20.0, azim=-60.0, rays=True, alpha=0.3, tolerance=0.05,
            lw=0.5, axes=False, zoom=1.0):
    """Draw a Model in 3D with matplotlib: the tessellated lens solids
    (translucent, with their edges), the mount parts and the rays. The
    optical axis z is drawn horizontally.

    Parameters
    ----------
    model : Model
    ax : matplotlib 3D axis, optional
        Default: a new figure.
    elev, azim : numbers, optional
        View angles in degrees (azim -90: looking along -y onto the y-z
        section).
    rays : bool, optional
        Draw the rays.
    alpha : number, optional
        Opacity of the lenses.
    tolerance : number, optional
        Tessellation tolerance in mm.
    lw : number, optional
        Line width of the rays.
    axes : bool, optional
        Draw the axes, panes and grid.
    zoom : number, optional
        Zoom of the axes box.

    Returns
    -------
    ax : matplotlib 3D axis
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        ax = plt.figure(figsize=(10, 6)).add_subplot(projection="3d")
    order = [2, 0, 1]  # draw z, x, y on the axes x, y, z
    points = []

    def solid(shape, color, a):
        verts, tris = shape.tessellate(tolerance, 0.2)
        v = np.array([tuple(p) for p in verts])[:, order]
        points.append(v)
        ax.add_collection3d(Poly3DCollection(v[np.array(tris)], facecolor=color, alpha=a,
                                             edgecolor="none"))
        for e in shape.edges():
            if e.geom_type.name == "LINE" and len(shape.edges()) > 12:
                continue  # seams
            pts = np.array([tuple(e.position_at(t)) for t in np.linspace(0, 1, 48)])[:, order]
            ax.plot(*pts.T, color="0.25", lw=0.4, alpha=0.8)

    for part in model.mount:
        solid(part, (0.62, 0.62, 0.65), 0.6)
    for lens in model.lenses:
        solid(lens, tuple(lens.color)[:3] if lens.color else (0.55, 0.75, 0.95), alpha)
    if rays:
        for ray in model.rays:
            color = tuple(ray.color)[:3]
            for e in ray.edges():
                p = np.array([tuple(e.start_point()), tuple(e.end_point())])[:, order]
                points.append(p)
                ax.plot(*p.T, color=color, lw=lw)
    if points:
        P = np.vstack(points)
        lo, hi = np.nanmin(P, axis=0), np.nanmax(P, axis=0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-6)), zoom=zoom)
    ax.view_init(elev, azim)
    ax.set_proj_type("ortho")
    if axes:
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("x [mm]")
        ax.set_zlabel("y [mm]")
    else:
        ax.set_axis_off()

    return ax


def export_gltf(model, filename, tolerance=0.05):
    """Write a Model (or any build123d shape) as binary glTF (.glb) for
    viewers and web pages: tessellated with the given tolerance (mm), with
    names and colours. glTF is y-up and in metres; the file holds the
    model's z-up millimetre coordinates converted to that."""

    import build123d as bd

    shape = model.compound() if isinstance(model, Model) else model
    if isinstance(model, Model) and model.rays:  # wires do not tessellate
        shape = bd.Compound(children=[c for c in shape.children if c.label != "rays"],
                            label=shape.label)
    return bd.export_gltf(shape, str(filename), binary=True, linear_deflection=tolerance,
                          angular_deflection=0.2)


def export_step(model, filename):
    """Write a Model (or any build123d shape) to a STEP file in mm."""

    import build123d as bd

    shape = model.compound() if isinstance(model, Model) else model
    return bd.export_step(shape, str(filename))
