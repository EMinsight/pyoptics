#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Image erecting prisms: Porro and Schmidt-Pechan prisms traced in 3D with
polarization, their tunnel diagrams, the image orientation and the roof
phase error (notebook 172, binoculars and erecting prisms).

A `Prism` is a convex block of glass bounded by plane `Face` objects. Rays
are traced non-sequentially through a list of prisms separated by air:
each ray enters a prism through the face it meets first, is reflected
inside by total internal reflection (TIR) or by a mirror coating until it
meets a face below the critical angle, and leaves through it. Every event
multiplies the 3 x 3 polarization ray tracing matrix of the ray (Chipman's
P, as in `raytrace.System.trace_polarization()` of notebook 20
(polarization ray tracing)): Fresnel or thin film (`coatings.Coating`) transmission at the
entrance and exit faces, TIR or the coated reflection inside. The phase of
TIR depends on the polarization, ::

    tan(phi_s/2) = -sqrt(n^2 sin^2(theta) - 1)/(n cos(theta)),
    tan(phi_p/2) = -n sqrt(n^2 sin^2(theta) - 1)/cos(theta)

(up to the sign convention of `thin_films`), so a sequence of reflections
is a retarder.

Prisms in a section
-------------------
Most prisms are prisms in the geometric sense: a polygon in the y-z plane
(the principal section, light travelling in +z) extruded along x
(`Prism.from_section()`). A *roof* replaces one face by two faces at 90 degrees
to each other whose ridge lies in the principal section; in the section it
acts like the plane face it replaces, across the section it adds a
reflection that reverses x. The rays at x > 0 meet the two roof faces in the
opposite order to the rays at x < 0: in each half the s and p directions of
the second reflection differ, the TIR phases differ, and the two halves of
the pupil leave with different polarization states and phases (the roof
phase error). A phase coating on the roof faces (`phase_coating()`)
removes the difference.

Unfolding (tunnel diagrams)
---------------------------
A reflection inside a prism can be unfolded by mirroring the rest of the
prism at the reflecting face: the ray then runs straight through a chain of
mirrored copies of the section, and the prism acts like a plane parallel
block of the same glass path (`unfold()`). The tunnel diagram shows the
size a prism needs for a given beam, and the block is what a sequential
ray trace of the telescope needs (`raytrace`: a plane parallel plate of
thickness `glass_path`): it adds spherical aberration and colour that the
objective has to correct, and shifts the focus by L (1 - 1/n).

Prisms here (dimensions for a beam of diameter D, in the units of D):

- `porro_prism()`: a right angle prism used through its hypotenuse, two
  TIR reflections, a glass path of twice the hypotenuse half width.
- `porro_system()`: two Porro prisms with crossed sections (Porro prism
  of the first kind, the classic binocular): four reflections, the image
  turned by 180 degrees, the axis displaced sideways.
- `schmidt_pechan()`: a half-penta (Bauernfeind) prism with angles 22.5,
  45 and 112.5 degrees, and a Schmidt roof prism (45, 67.5, 67.5 degrees)
  behind a small air gap: six reflections (one coated, at 22.5 degrees
  incidence, five by TIR including the two roof faces), input and output
  on one axis, the image turned by 180 degrees.

Functions:

- trace(): rays through a list of prisms with polarization matrices.
- orientation(): the 3 x 3 matrix of the reflections, the image
  orientation and handedness.
- section_trace(), unfold(): the 2D path in the principal section and the
  tunnel diagram.
- glass_path(): the length of the axial ray in glass.
- jones_pupil(): the Jones matrices across a collimated beam; roof_error():
  the polarization and phase difference between the two pupil halves.
- polarized_psf(): the far field (PSF) of a beam with a Jones pupil.
- phase_coating(): a dielectric stack for the roof faces that removes the
  phase difference.

References
----------
R. A. Chipman, W.-S. T. Lam and G. Young, "Polarized Light and Optical
Systems", CRC Press (2018), chapters 9 and 12.

W. J. Smith, "Modern Optical Engineering", 4th ed., McGraw-Hill (2008),
chapter 8 (prisms and tunnel diagrams).
"""

import copy
from collections import namedtuple

import numpy as np

from .coatings import Coating
from .materials import complex_index


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v/np.linalg.norm(v, axis=-1, keepdims=True)


class Face:
    """A plane face of a prism.

    Parameters
    ----------
    point : (3,) array
        A point of the plane (x, y, z).
    normal : (3,) array
        Outward normal.
    mirror : bool, optional
        Coated reflecting face (metal or dielectric mirror): every ray that
        meets it from inside is reflected through `coating`, which then
        needs a substrate.
    coating : coatings.Coating, optional
        Thin film stack, layers listed from the glass outwards: the AR
        coating of a refracting face (used for transmission and, with air
        as the medium behind it, for TIR), a phase coating on a roof face,
        or the stack of a mirror with its substrate (metal).
    label : str, optional
    """

    def __init__(self, point, normal, mirror=False, coating=None, label=None):
        self.point = np.asarray(point, dtype=float)
        self.normal = _unit(normal)
        self.mirror = mirror
        self.coating = coating
        self.label = label

    def __repr__(self):
        return "Face({}, normal={}, mirror={}, label={!r})".format(
            np.round(self.point, 4), np.round(self.normal, 4), self.mirror, self.label)

    def moved(self, R=np.eye(3), offset=np.zeros(3)):
        """The face rotated by R about the origin, then shifted."""

        other = copy.copy(self)
        other.point = R @ self.point + offset
        other.normal = R @ self.normal

        return other


class Prism:
    """A convex block of glass bounded by plane faces.

    Parameters
    ----------
    faces : sequence of Face
    material : number, callable or (n_d, V_d)
        Index of the glass (a `glasses.Glass` works).
    label : str, optional
    section : (K, 2) array, optional
        The principal section (z, y) for drawings and unfolding.
    section_faces : sequence, optional
        Labels of the section edges (edge k from vertex k to k + 1).
    """

    def __init__(self, faces, material, label=None, section=None, section_faces=None):
        self.faces = list(faces)
        self.material = material
        self.label = label
        self.section = None if section is None else np.asarray(section, dtype=float)
        self.section_faces = section_faces

    def __repr__(self):
        return "Prism({!r}, {} faces)".format(self.label, len(self.faces))

    def index(self, wavelength_nm):
        return complex_index(self.material, wavelength_nm).real

    @classmethod
    def from_section(cls, vertices, width, material, labels=None, mirrors=(), coatings=None,
                     roof=None, roof_coating=None, label=None):
        """Prism extruded along x from a polygon in the y-z plane.

        Parameters
        ----------
        vertices : (K, 2) array
            Corners (z, y) of the convex section, counterclockwise in the
            (z, y) plane.
        width : number
            Full extent along x (the side faces at x = +-width/2).
        material : glass
        labels : sequence of str, optional
            Labels of the edges (edge k joins vertex k and k + 1).
        mirrors : sequence of int, optional
            Edges with a mirror coating.
        coatings : dict, optional
            {edge: Coating}.
        roof : int, optional
            Edge replaced by a 90 degree roof whose ridge is the edge.
        roof_coating : coatings.Coating, optional
            Coating of both roof faces (a phase coating).
        label : str, optional
        """

        V = np.asarray(vertices, dtype=float)
        K = len(V)
        labels = ["face {}".format(k) for k in range(K)] if labels is None else list(labels)
        coatings = {} if coatings is None else coatings
        # orientation: outward normal of a counterclockwise polygon in (z, y)
        area = 0.5*np.sum(V[:, 0]*np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1)*V[:, 1])
        sign = 1.0 if area > 0 else -1.0
        faces = []
        for k in range(K):
            a, b = V[k], V[(k + 1) % K]
            dz, dy = b - a
            n_zy = sign*np.array([dy, -dz])  # right normal of a ccw edge points out
            normal = np.array([0.0, n_zy[1], n_zy[0]])
            normal = normal/np.linalg.norm(normal)
            point = np.array([0.0, a[1], a[0]])
            if roof is not None and k == roof:
                c = np.sqrt(0.5)
                for side, x in ((+1, "+"), (-1, "-")):
                    faces.append(Face(point, c*normal + side*c*np.array([1.0, 0.0, 0.0]),
                                      coating=roof_coating, label=labels[k] + " (roof " + x + ")"))
            else:
                faces.append(Face(point, normal, mirror=k in mirrors, coating=coatings.get(k),
                                  label=labels[k]))
        for side in (+1, -1):
            faces.append(Face([side*width/2, 0.0, 0.0], [side, 0.0, 0.0], label="side"))

        prism = cls(faces, material, label, section=V, section_faces=labels)
        prism.section_mirrors = set(mirrors) | ({roof} if roof is not None else set())

        return prism

    def moved(self, R=np.eye(3), offset=np.zeros(3)):
        """The prism rotated by R about the origin, then shifted (3D); the
        section is kept for drawings in its own frame."""

        other = copy.copy(self)
        other.faces = [f.moved(R, offset) for f in self.faces]

        return other


# --- polarization matrices ------------------------------------------------------

def _sp_basis(k1, k2, normal):
    s = np.cross(k1, normal)
    small = np.linalg.norm(s, axis=-1) < 1e-12
    if np.any(small):
        alt = np.cross(k1[small], np.array([0.0, 1.0, 0.0]))
        bad = np.linalg.norm(alt, axis=-1) < 1e-9
        alt[bad] = np.cross(k1[small][bad], np.array([1.0, 0.0, 0.0]))
        s[small] = alt
    s = _unit(s)

    return s, np.cross(k1, s), np.cross(k2, s)


def _amplitudes(coating, cos_i, n1, n2, wavelength, wavelength_nm):
    coating = Coating() if coating is None else coating
    if coating.layers or n2 is not None:
        return coating.amplitudes(cos_i, n1, n2, wavelength, wavelength_nm)

    return coating.amplitudes(cos_i, n1, None, wavelength, wavelength_nm)


def _refraction(k1, k2, normal, n1, n2, coating, wavelength, wavelength_nm):
    cos_i = np.abs(np.sum(k1*normal, axis=-1))
    cos_t = np.abs(np.sum(k2*normal, axis=-1))
    _, _, t_s, t_p = _amplitudes(coating, cos_i, n1, n2, wavelength, wavelength_nm)
    t_p = t_p*cos_i/cos_t
    power = np.sqrt(n2*cos_t/(n1*cos_i))
    s, p1, p2 = _sp_basis(k1, k2, normal)

    return power[:, None, None]*(t_s[:, None, None]*s[:, :, None]*s[:, None, :]
                                 + t_p[:, None, None]*p2[:, :, None]*p1[:, None, :])


def _reflection(k1, k2, normal, n1, n2, coating, wavelength, wavelength_nm):
    """Reflection inside the glass (n1) at a face with the medium n2 behind
    it (None: the substrate of the coating, a mirror)."""

    cos_i = np.abs(np.sum(k1*normal, axis=-1))
    r_s, r_p = _amplitudes(coating, cos_i, n1, n2, wavelength, wavelength_nm)[:2]
    s, p1, p2 = _sp_basis(k1, k2, normal)

    return (r_s[:, None, None]*s[:, :, None]*s[:, None, :]
            - r_p[:, None, None]*p2[:, :, None]*p1[:, None, :])


# --- tracing ----------------------------------------------------------------------

PrismTrace = namedtuple("PrismTrace", ["points", "directions", "P", "opl", "glass", "valid",
                                       "events", "path"])
PrismTrace.__doc__ = """Result of trace(): final points and directions (N, 3)
after the last prism, polarization matrices P (N, 3, 3) (E_out = P E_in),
optical path opl and geometric path in glass (N,), valid rays, the events
of each ray (list of (prism label, face label, "in", "out", "TIR" or
"mirror")) and the polyline of each ray (list of (M, 3) arrays)."""


def trace(prisms, points, directions, wavelength=550e-6, units="mm", n_air=1.0, max_events=16,
          keep_paths=False):
    """Trace rays through a sequence of prisms in air.

    Parameters
    ----------
    prisms : sequence of Prism
        In the order the light meets them.
    points, directions : (N, 3) arrays
        Start points (in front of the first prism) and directions.
    wavelength : number, optional
        Vacuum wavelength in the length unit.
    units : str, optional
        Length unit ("mm", "um", "m", "nm") for the dispersion of the glass.
    n_air : number, optional
    max_events : int, optional
        Maximum number of reflections inside one prism.
    keep_paths : bool, optional
        Keep the events and polylines of every ray (slower).

    Returns
    -------
    PrismTrace
    """

    from .raytrace import NM_PER_UNIT

    wl_nm = wavelength*NM_PER_UNIT[units]
    P = np.array(points, dtype=float).reshape(-1, 3)
    d = _unit(np.array(directions, dtype=float).reshape(-1, 3))
    N = len(P)
    M = np.tile(np.eye(3, dtype=complex), (N, 1, 1))
    opl = np.zeros(N)
    glass = np.zeros(N)
    valid = np.ones(N, dtype=bool)
    events = [[] for _ in range(N)] if keep_paths else None
    path = [[p.copy()] for p in P] if keep_paths else None

    def log(idx, prism, face, what, at):
        if keep_paths:
            for j, q in zip(np.flatnonzero(idx), at):
                events[j].append((prism.label, face.label, what))
                path[j].append(q.copy())

    for prism in prisms:
        n = prism.index(wl_nm)
        F = prism.faces
        pts = np.array([f.point for f in F])
        nrm = np.array([f.normal for f in F])
        # entry
        dn = d @ nrm.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.einsum("fk,fk->f", pts, nrm)[None, :] - P @ nrm.T
            t = t/dn
        t_in = np.where(dn < 0, t, -np.inf).max(axis=1)
        t_out = np.where(dn > 0, t, np.inf).min(axis=1)
        k_in = np.where(dn < 0, t, -np.inf).argmax(axis=1)
        hit = valid & (t_in < t_out - 1e-12) & (t_in > -1e-9)
        valid &= hit
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            break
        Q = P[idx] + t_in[idx, None]*d[idx]
        opl[idx] += n_air*t_in[idx]
        normal = nrm[k_in[idx]]
        k1 = d[idx]
        cos_i = -np.sum(k1*normal, axis=1)
        k2 = (n_air*k1 + (n_air*cos_i - np.sqrt(n**2 - n_air**2*(1 - cos_i**2)))[:, None]*normal)/n
        Pm = np.zeros((len(idx), 3, 3), dtype=complex)
        for f_i in np.unique(k_in[idx]):
            sel = k_in[idx] == f_i
            Pm[sel] = _refraction(k1[sel], k2[sel], normal[sel], n_air, n, F[f_i].coating,
                                  wavelength, wl_nm)
        M[idx] = Pm @ M[idx]
        P[idx], d[idx] = Q, _unit(k2)
        if keep_paths:
            for j, q, fi in zip(idx, Q, k_in[idx]):
                events[j].append((prism.label, F[fi].label, "in"))
                path[j].append(q.copy())
        inside = idx.copy()
        for _ in range(max_events + 1):
            if len(inside) == 0:
                break
            dn = d[inside] @ nrm.T
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (np.einsum("fk,fk->f", pts, nrm)[None, :] - P[inside] @ nrm.T)/dn
            t = np.where((dn > 1e-14) & (t > 1e-9), t, np.inf)
            k_f = t.argmin(axis=1)
            t_f = t[np.arange(len(inside)), k_f]
            lost = ~np.isfinite(t_f)
            valid[inside[lost]] = False
            inside, k_f, t_f = inside[~lost], k_f[~lost], t_f[~lost]
            Q = P[inside] + t_f[:, None]*d[inside]
            opl[inside] += n*t_f
            glass[inside] += t_f
            k1 = d[inside]
            normal = nrm[k_f]
            cos_i = np.sum(k1*normal, axis=1)
            sin2_t = (n/n_air)**2*(1 - cos_i**2)
            is_mirror = np.array([F[k].mirror for k in k_f], dtype=bool)
            tir = (sin2_t >= 1) | is_mirror
            # reflected
            r_idx = np.flatnonzero(tir)
            if len(r_idx):
                kk = k1[r_idx]
                nn = normal[r_idx]
                k2 = kk - 2*np.sum(kk*nn, axis=1)[:, None]*nn
                Pm = np.zeros((len(r_idx), 3, 3), dtype=complex)
                for f_i in np.unique(k_f[r_idx]):
                    sel = k_f[r_idx] == f_i
                    face = F[f_i]
                    if face.mirror:
                        Pm[sel] = _reflection(kk[sel], k2[sel], nn[sel], n, None, face.coating,
                                              wavelength, wl_nm)
                    else:
                        Pm[sel] = _reflection(kk[sel], k2[sel], nn[sel], n, n_air, face.coating,
                                              wavelength, wl_nm)
                j = inside[r_idx]
                M[j] = Pm @ M[j]
                P[j], d[j] = Q[r_idx], _unit(k2)
                if keep_paths:
                    for jj, q, fi in zip(j, Q[r_idx], k_f[r_idx]):
                        events[jj].append((prism.label, F[fi].label,
                                           "mirror" if F[fi].mirror else "TIR"))
                        path[jj].append(q.copy())
            # refracted out
            o_idx = np.flatnonzero(~tir)
            if len(o_idx):
                kk = k1[o_idx]
                nn = normal[o_idx]
                ci = np.sum(kk*nn, axis=1)
                k2 = (n*kk + (np.sqrt(n_air**2 - n**2*(1 - ci**2)) - n*ci)[:, None]*nn)/n_air
                Pm = np.zeros((len(o_idx), 3, 3), dtype=complex)
                for f_i in np.unique(k_f[o_idx]):
                    sel = k_f[o_idx] == f_i
                    c = F[f_i].coating
                    Pm[sel] = _refraction(kk[sel], k2[sel], nn[sel], n, n_air,
                                          None if c is None else c.reversed(), wavelength, wl_nm)
                j = inside[o_idx]
                M[j] = Pm @ M[j]
                P[j], d[j] = Q[o_idx], _unit(k2)
                if keep_paths:
                    for jj, q, fi in zip(j, Q[o_idx], k_f[o_idx]):
                        events[jj].append((prism.label, F[fi].label, "out"))
                        path[jj].append(q.copy())
            inside = inside[r_idx]
        else:
            valid[inside] = False
        if len(inside):
            valid[inside] = False

    if keep_paths:
        path = [np.array(p) for p in path]

    return PrismTrace(P, d, M, opl, glass, valid, events, path)


def orientation(prisms, ray=((1e-3, 1e-3, -10.0), (0.0, 0.0, 1.0)), wavelength=550e-6):
    """Matrix of the reflections met by a ray (default: near the axis, at
    x > 0 to choose one roof half), O = H_k ... H_1 with the Householder
    matrices H = I - 2 n n^T of the reflecting faces.

    A direction v in front of the prisms leaves as O v. With input and
    output along z, the upper left 2 x 2 block maps the transverse
    directions (the image orientation): diag(-1, -1) turns the image by
    180 degrees (erects the inverted image of a Keplerian telescope),
    det(O) = -1 means a mirror image.

    Returns
    -------
    O : (3, 3) array
    events : list
    """

    tr = trace(prisms, [ray[0]], [ray[1]], wavelength, keep_paths=True)
    O = np.eye(3)
    normals = []
    for prism in prisms:
        for f in prism.faces:
            normals.append((prism.label, f.label, f.normal))
    lookup = {(a, b): c for a, b, c in normals}
    for p_label, f_label, what in tr.events[0]:
        if what in ("TIR", "mirror"):
            nn = lookup[(p_label, f_label)]
            O = (np.eye(3) - 2*np.outer(nn, nn)) @ O

    return O, tr.events[0]


def glass_path(prisms, wavelength=550e-6, ray=((1e-3, 1e-3, -10.0), (0.0, 0.0, 1.0))):
    """Geometric path length of a near-axial ray in glass (the thickness of
    the equivalent plane parallel block)."""

    tr = trace(prisms, [ray[0]], [ray[1]], wavelength)

    return tr.glass[0]


# --- 2D: principal section and unfolding ------------------------------------------------

def _reflect2(v, n):
    return v - 2*np.dot(v, n)*n


def _edges(V):
    area = 0.5*np.sum(V[:, 0]*np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1)*V[:, 1])
    sign = 1.0 if area > 0 else -1.0
    out = []
    for k in range(len(V)):
        a, b = V[k], V[(k + 1) % len(V)]
        t_ = b - a
        nn = sign*np.array([t_[1], -t_[0]])
        out.append((a, nn/np.linalg.norm(nn)))

    return out


def section_trace(sections, start, direction, n=1.5688, max_events=12, mirrors=(),
                  return_events=False):
    """2D trace of a ray through polygons (z, y) (the principal sections of
    prisms; a roof counts as the plane face it replaces), with TIR or exit
    decided by the critical angle, and refraction at the faces.

    Parameters
    ----------
    sections : sequence of (K, 2) arrays or Prism
    start, direction : (2,) arrays
        (z, y) start point (in air, in front of the first polygon) and
        direction.
    n : number, optional
        Index of the glass.
    max_events : int, optional
    mirrors : sequence of (polygon index, edge index), optional
        Coated edges (and roofs) that reflect at any angle (taken from the
        prisms if Prism objects are given).
    return_events : bool, optional
        Also return the event of every point.

    Returns
    -------
    points : (M, 2) array
        The polyline, ending a little behind the last polygon.
    reflections : list of (polygon index, edge index)
    events : list of (kind, polygon index, edge index), optional
        kind "start", "in", "reflect", "out" or "end" for every point.
    """

    polys = [s.section if isinstance(s, Prism) else np.asarray(s, float) for s in sections]
    mirrors = set(mirrors) | {(i, e) for i, s in enumerate(sections) if isinstance(s, Prism)
                              for e in getattr(s, "section_mirrors", ())}
    p = np.asarray(start, float)
    d = np.asarray(direction, float)
    d = d/np.linalg.norm(d)
    pts = [p.copy()]
    events = [("start", None, None)]
    refl = []
    for ip, V in enumerate(polys):
        edges = _edges(V)
        t_in, k_in, t_out = -np.inf, None, np.inf
        for k, (a, nn) in enumerate(edges):
            dn = np.dot(d, nn)
            if abs(dn) < 1e-14:
                continue
            t = np.dot(a - p, nn)/dn
            if dn < 0 and t > t_in:
                t_in, k_in = t, k
            if dn > 0 and t < t_out:
                t_out = t
        if not (t_in < t_out and t_in >= -1e-9):
            continue
        p = p + t_in*d
        pts.append(p.copy())
        events.append(("in", ip, k_in))
        nn = edges[k_in][1]
        ci = -np.dot(d, nn)
        d = (d + (ci - np.sqrt(n**2 - (1 - ci**2)))*nn)/n
        d = d/np.linalg.norm(d)
        for _ in range(max_events):
            t_out, kb = np.inf, None
            for k, (a, nn) in enumerate(edges):
                dn = np.dot(d, nn)
                if dn > 1e-14:
                    t = np.dot(a - p, nn)/dn
                    if 1e-12 < t < t_out:
                        t_out, kb = t, k
            p = p + t_out*d
            pts.append(p.copy())
            nn = edges[kb][1]
            ci = np.dot(d, nn)
            if n**2*(1 - ci**2) >= 1 or (ip, kb) in mirrors:
                d = _reflect2(d, nn)
                refl.append((ip, kb))
                events.append(("reflect", ip, kb))
            else:
                d = n*d + (np.sqrt(1 - n**2*(1 - ci**2)) - n*ci)*nn
                d = d/np.linalg.norm(d)
                events.append(("out", ip, kb))
                break
    size = np.ptp(np.concatenate(polys), axis=0).max()
    pts.append(p + 0.3*size*d)
    events.append(("end", None, None))
    if return_events:
        return np.array(pts), refl, events

    return np.array(pts), refl


def unfold(sections, start, direction, n=1.5688, mirrors=()):
    """Tunnel diagram: unfold the reflections of the 2D path.

    Every reflection mirrors the rest of the path, and the polygon it
    happens in, at the reflecting edge. The copies of the polygons, laid
    along the straight unfolded ray, are the tunnel diagram: a beam passes
    the prism if it fits through the tunnel.

    Parameters
    ----------
    sections, start, direction, n, mirrors
        See `section_trace()`.

    Returns
    -------
    copies : list of (polygon index, (K, 2) array)
        The polygon of every segment of the path in glass, mirrored into
        the unfolded frame.
    line : (M, 2) array
        The unfolded (straight) ray.
    """

    polys = [s.section if isinstance(s, Prism) else np.asarray(s, float) for s in sections]
    pts, refl, events = section_trace(sections, start, direction, n, mirrors=mirrors,
                                      return_events=True)
    A, b = np.eye(2), np.zeros(2)
    copies, line = [], []
    for q, (kind, ip, e) in zip(pts, events):
        line.append(A @ q + b)
        if kind == "in":
            copies.append((ip, (A @ polys[ip].T).T + b))
        elif kind == "reflect":
            a, nn = _edges(polys[ip])[e]
            Hm = np.eye(2) - 2*np.outer(nn, nn)
            A, b = A @ Hm, A @ (a - Hm @ a) + b
            copies.append((ip, (A @ polys[ip].T).T + b))

    return copies, np.array(line)


def draw_section(ax, sections, start=None, direction=(1.0, 0.0), beam=None, n=1.5688,
                 colors=("#cfe3f3", "#e3ecf5"), edge="#4a6f8a", ray_color="#d6604d",
                 mirror_color="0.25", labels=True):
    """Draw the principal sections of prisms (z horizontal, y vertical)
    with the axial ray, and the edge rays of a beam of diameter `beam`.
    Mirror coated edges are drawn thick, roofs dashed."""

    import matplotlib.patches as mpatches

    prisms = [s for s in sections if isinstance(s, Prism)]
    for i, s in enumerate(sections):
        V = s.section if isinstance(s, Prism) else np.asarray(s, float)
        ax.add_patch(mpatches.Polygon(V, closed=True, facecolor=colors[i % len(colors)],
                                      edgecolor=edge, lw=1.0, zorder=1))
        if isinstance(s, Prism):
            for e in getattr(s, "section_mirrors", ()):
                a, b = V[e], V[(e + 1) % len(V)]
                roof = any("roof" in (f.label or "") for f in s.faces) and \
                    "roof" in s.section_faces[e].lower()
                ax.plot([a[0], b[0]], [a[1], b[1]], color=mirror_color, lw=1.2 if roof else 3,
                        ls="--" if roof else "-", zorder=2)
    if start is not None:
        offs = [0.0] if beam is None else [-beam/2, 0.0, beam/2]
        d = np.asarray(direction, float)
        perp = np.array([-d[1], d[0]])/np.linalg.norm(d)
        for o in offs:
            pts, _ = section_trace(sections, np.asarray(start) + o*perp, d, n)
            ax.plot(pts[:, 0], pts[:, 1], color=ray_color, lw=1.4 if o == 0 else 0.8,
                    alpha=1.0 if o == 0 else 0.7, zorder=3)
    ax.set_aspect("equal")


def draw_tunnel(ax, sections, start, direction=(1.0, 0.0), beam=None, n=1.5688,
                colors=("#cfe3f3", "#e3ecf5"), edge="#4a6f8a", ray_color="#d6604d"):
    """Draw the tunnel diagram of `unfold()`: the mirrored sections and the
    straight axial ray (and the edge rays of a beam of diameter `beam`).

    Returns
    -------
    length : number
        Length of the axial ray in glass (the unfolded glass path).
    """

    import matplotlib.patches as mpatches

    copies, line = unfold(sections, start, direction, n)
    for k, (ip, V) in enumerate(copies):
        ax.add_patch(mpatches.Polygon(V, closed=True, facecolor=colors[ip % len(colors)],
                                      edgecolor=edge, lw=0.8, alpha=0.85, zorder=1))
    ax.plot(line[:, 0], line[:, 1], color=ray_color, lw=1.4, zorder=3)
    d = line[-1] - line[0]
    d = d/np.linalg.norm(d)
    perp = np.array([-d[1], d[0]])
    if beam is not None:
        for o in (-beam/2, beam/2):
            ax.plot(line[[0, -1], 0] + o*perp[0], line[[0, -1], 1] + o*perp[1], color=ray_color,
                    lw=0.8, alpha=0.7, zorder=3)
    ax.set_aspect("equal")
    pts, _, events = section_trace(sections, start, direction, n, return_events=True)
    length = 0.0
    inside = False
    for q0, q1, (kind, _, _) in zip(pts[:-1], pts[1:], events[:-1]):
        if kind in ("in", "reflect"):
            length += np.linalg.norm(q1 - q0)

    return length


def _is_reflection(p0, p1, p2):
    d1 = p1 - p0
    d2 = p2 - p1
    return np.dot(d1, d2) < np.linalg.norm(d1)*np.linalg.norm(d2)*(1 - 1e-9)


# --- the prisms ------------------------------------------------------------------

def _bak4():
    from . import glasses
    return glasses.get("N-BAK4")


def porro_prism(D=1.0, material=None, margin=0.1, label="Porro"):
    """Right angle prism used as a Porro prism: the beam enters through one
    half of the hypotenuse (in +z), is reflected by TIR on both legs and
    leaves through the other half (in -z), displaced by half the
    hypotenuse in y.

    Parameters
    ----------
    D : number, optional
        Beam diameter; the hypotenuse is 2 D (1 + margin) long and the
        prism D (1 + margin) wide.
    material : glass, optional
        Default N-BAK4.
    margin : number, optional
    label : str, optional

    Returns
    -------
    Prism
        Hypotenuse in the plane z = 0, the right angle at
        (z, y) = (H, 0), H = D (1 + margin).
    """

    material = _bak4() if material is None else material
    H = D*(1 + margin)
    V = [(0.0, -H), (H, 0.0), (0.0, H)]
    # edges: 0 leg (lower), 1 leg (upper), 2 hypotenuse
    return Prism.from_section(V, 2*H*0 + D*(1 + margin), material,
                              labels=["lower leg", "upper leg", "hypotenuse"], label=label)


def porro_system(D=1.0, material=None, margin=0.1, gap=None):
    """Porro prism system of the first kind: two Porro prisms with their
    sections at 90 degrees.

    The first prism (section in the y-z plane) turns the beam back and
    shifts it by H = D (1 + margin) in +y; the second, rotated by 90 degrees
    about z (section in the x-z plane), turns it forward again and shifts
    it by H in -x. The output beam is parallel to the input, along +z,
    displaced by (-H, +H) and turned by 180 degrees about its axis. The
    input axis is x = y = 0; the hypotenuse of the first prism is at z = 0,
    that of the second at z = -gap.

    Returns
    -------
    list of Prism
    """

    material = _bak4() if material is None else material
    H = D*(1 + margin)
    gap = 0.2*H if gap is None else gap
    # prism 1: hypotenuse at z = 0, facing -z, centred at y = 0, the input
    # beam enters the lower half (y = -H/2) and leaves the upper half
    p1 = porro_prism(D, material, margin, label="Porro 1")
    # it is shifted so that the input axis (x = 0, y = 0) hits y = -H/2:
    p1 = p1.moved(offset=np.array([0.0, H/2, 0.0]))
    # the beam leaves at y = H, travelling in -z. Prism 2: rotated by 180
    # degrees about y (faces +z, hypotenuse towards +z... facing the beam)
    # and by 90 degrees about z (its section in the x-z plane)
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    p2 = porro_prism(D, material, margin, label="Porro 2").moved(Ry @ Rz)
    # its hypotenuse now lies at z = 0 facing +z; move it in front of
    # prism 1: hypotenuse at z = -gap, centred so that the beam at
    # (x, y) = (0, H) enters its first half
    p2 = p2.moved(offset=np.array([-H/2, H, -gap]))

    return [p1, p2]


def schmidt_pechan(D=1.0, material=None, margin=0.1, gap=0.02, roof=True, mirror_coating=None,
                   roof_coating=None, ar_coating=None, top=1.35, bottom=1.35, length=4.25,
                   width=1.3):
    """Schmidt-Pechan prism: a half-penta (Bauernfeind) prism and a Schmidt
    roof prism, input and output on the z axis.

    Section (z, y): the beam enters the Bauernfeind prism through its face
    A (z = 0), is reflected on the face B (22.5 degrees to A, incidence
    22.5 degrees, below the critical angle: mirror coated), then by TIR on
    A (45 degrees) and leaves through C (45 degrees to A) at normal
    incidence, deviated by 45 degrees. Across an air gap it enters the
    Schmidt prism through E (parallel to C), is reflected by TIR on X (45
    degrees), on the roof R (22.5 degrees in the section, 49.2 degrees on
    each roof face) and on E (45 degrees), and leaves through X at normal
    incidence along +z.

    Parameters
    ----------
    D : number, optional
        Beam diameter.
    material : glass, optional
        Default N-BAK4.
    margin : number, optional
        Clear aperture margin.
    gap : number, optional
        Air gap between the prisms (in units of D).
    roof : bool, optional
        False: a plain plane face instead of the roof (a Pechan prism,
        which leaves a mirror image). Its incidence there is only 22.5
        degrees, so the face gets the mirror coating of B.
    mirror_coating : coatings.Coating, optional
        Coating of face B with its substrate (default: a perfect mirror,
        r = -1 for both polarizations).
    roof_coating : coatings.Coating, optional
        Coating of the roof faces (layers from the glass outwards; TIR
        towards air behind it), e.g. from `phase_coating()`.
    ar_coating : coatings.Coating, optional
        Coating of the entrance and exit faces (layers from air into the
        glass). None: bare glass.
    top, bottom : numbers, optional
        Distances of the upper (with B) and lower (with C) corner of the
        face A from the axis, in units of D (1 + margin).
    length : number, optional
        Length of the faces E and X of the Schmidt prism from their common
        corner, in units of D (1 + margin).
    width : number, optional
        Extent of the prisms across the section (x), in units of D (1 +
        margin).

    Returns
    -------
    list of Prism
    """

    material = _bak4() if material is None else material
    a = D*(1 + margin)
    c22, s22 = np.cos(np.radians(22.5)), np.sin(np.radians(22.5))
    c45 = np.sqrt(0.5)
    # Bauernfeind prism: A on z = 0 from y = yA0 (bottom, vertex with C) to
    # y = yA1 (top, vertex with B). B from the top vertex in the direction
    # (sin 22.5, -cos 22.5); C from the bottom vertex in (cos 45, cos 45).
    # Sizes: the beam (|y| <= a/2) enters A, meets B, returns to A below
    # the entrance zone and leaves through C.
    yA1 = top*a
    # the beam centre hits B at z_B(0) = yA1*tan(22.5) (B at y = 0)
    # a generous lower vertex:
    yA0 = -bottom*a
    top = np.array([0.0, yA1])
    bottom = np.array([0.0, yA0])
    dB = np.array([s22, -c22])
    dC = np.array([c45, c45])
    # intersection of B and C
    Mx = np.array([dB, -dC]).T
    s, _ = np.linalg.solve(Mx, bottom - top)
    third = top + s*dB
    sec1 = np.array([bottom, third, top])  # (z, y): edges C, B, A
    labels1 = ["C (exit)", "B (mirror)", "A (entrance, TIR)"]
    coat1 = {}
    if ar_coating is not None:
        coat1 = {0: ar_coating, 2: ar_coating}
    mirror_coat = mirror_coating
    p1 = Prism.from_section(sec1, a*width, material, labels=labels1, mirrors=(1,),
                            coatings={**coat1, **({1: mirror_coat} if mirror_coat else {})},
                            label="Bauernfeind")
    if mirror_coat is None:
        # perfect mirror: a coating that reflects everything with r = -1
        p1.faces[1].coating = _PerfectMirror()
    # path in the Bauernfeind prism, to place the Schmidt prism
    # Schmidt prism: E parallel to C at the distance gap; X vertical (z =
    # const) facing +z, apex angle 45 degrees between E and X, roof R the
    # base. Outward normals: E (-c45, c45), X (1, 0), R (-sin 22.5... )
    nE = np.array([-c45, c45])
    # line C: through `bottom` with direction dC; E is shifted by gap along -nE
    pE = bottom - gap*D*nE
    # the beam leaves C travelling in (c45, -c45); the Schmidt prism is
    # placed (shift s along E) such that the output is on the z axis.
    # Build it with the apex at the origin, then position it.
    nX = np.array([1.0, 0.0])
    # apex at the intersection of E and X; E direction (along the line)
    tE = np.array([c45, c45])
    L = length*a  # length of E and X from the apex
    apex = np.zeros(2)
    # the triangle: apex, the point L along -tE (down-left along E) and the
    # point L along X downwards (-y): E goes from apex in direction -tE,
    # X from apex in direction -y
    e_end = apex - L*tE
    x_end = apex + L*np.array([0.0, -1.0])
    sec2 = np.array([apex, e_end, x_end])  # edges: E, R (base), X
    labels2 = ["E (entrance, TIR)", "R (roof)", "X (TIR, exit)"]
    # place: E must lie on the line through pE with direction tE; choose the
    # shift along E (u) and solve for the output on y = 0 numerically
    n_d = complex_index(material, 587.56).real

    def placed(u):
        off = pE + u*tE
        return sec2 + off

    def out_y(u):
        pts, refl = section_trace([sec1, placed(u)], (-1.0, 0.0), (1.0, 0.0), n=n_d,
                                  mirrors=[(0, 1), (1, 1)])
        return pts[-2][1]

    from scipy.optimize import brentq
    us = np.linspace(0.0, 8*a, 161)
    vals = []
    for u in us:
        try:
            vals.append(out_y(u))
        except Exception:
            vals.append(np.nan)
    vals = np.array(vals)
    ok = np.isfinite(vals)
    sign_change = np.flatnonzero(ok[:-1] & ok[1:] & (np.sign(vals[:-1]) != np.sign(vals[1:])))
    u0 = brentq(out_y, us[sign_change[0]], us[sign_change[0] + 1])
    sec2 = placed(u0)
    coat2 = {}
    if ar_coating is not None:
        coat2 = {0: ar_coating, 2: ar_coating}
    mirrors2 = ()
    if not roof:
        coat2[1] = mirror_coat if mirror_coat else _PerfectMirror()
        mirrors2 = (1,)
    p2 = Prism.from_section(sec2, a*width, material, labels=labels2, coatings=coat2,
                            mirrors=mirrors2, roof=1 if roof else None,
                            roof_coating=roof_coating, label="Schmidt")

    return [p1, p2]


class _PerfectMirror(Coating):
    """A mirror with r_s = r_p = -1 (field ratio convention of thin_films:
    an ideal metal)."""

    def __init__(self):
        super().__init__((), substrate=1.0, label="perfect mirror")

    def amplitudes(self, cos_i, n1, n2, wavelength, wavelength_nm=None, x=None, y=None):
        one = -np.ones(np.shape(cos_i), dtype=complex)
        return one, one, 0*one, 0*one


# --- pupils -----------------------------------------------------------------------

def jones_pupil(prisms, D, wavelength=550e-6, n=33, z_start=-1.0, center=(0.0, 0.0)):
    """Jones matrices of a collimated beam along +z through the prisms
    (output along +z), on an n x n grid across the beam diameter D.

    Returns
    -------
    x, y : 1d arrays
        Grid coordinates.
    J : (n, n, 2, 2) complex array
        E_out(x, y) = J E_in in the x, y basis, NaN outside the beam or for
        lost rays; the common optical path is removed, its variation across
        the beam is included as a phase.
    """

    x = np.linspace(-D/2, D/2, n) + center[0]
    y = np.linspace(-D/2, D/2, n) + center[1]
    X, Y = np.meshgrid(x, y)
    m = (X - center[0])**2 + (Y - center[1])**2 <= (D/2)**2
    P = np.column_stack([X[m], Y[m], np.full(m.sum(), z_start)])
    d = np.tile([0.0, 0.0, 1.0], (m.sum(), 1))
    tr = trace(prisms, P, d, wavelength)
    # optical path to a common plane normal to z behind the prisms
    z_end = np.nanmax(tr.points[:, 2]) + 1.0
    extra = (z_end - tr.points[:, 2])/tr.directions[:, 2]
    opl = tr.opl + extra
    opl = opl - np.nanmean(opl[tr.valid])
    k0 = 2*np.pi/wavelength
    Jm = tr.P[:, :2, :2]*np.exp(1j*k0*opl)[:, None, None]
    Jm[~tr.valid] = np.nan
    J = np.full((n, n, 2, 2), np.nan, dtype=complex)
    J[m] = Jm

    return x, y, J


def roof_error(J, x):
    """Difference between the two halves of a Jones pupil (x > 0 and
    x < 0), as the eigen-decomposition of J_left^-1 J_right: the phase
    difference (radians) of the two eigenvalues' mean, and the retardance
    between the halves (the phase difference of the eigenvalues).

    Parameters
    ----------
    J : (n, n, 2, 2) array
        From `jones_pupil()`.
    x : 1d array

    Returns
    -------
    phase, retardance : numbers (radians)
    """

    Jr = np.nanmean(J[:, x > 0].reshape(-1, 2, 2), axis=0)
    Jl = np.nanmean(J[:, x < 0].reshape(-1, 2, 2), axis=0)
    T = np.linalg.solve(Jl, Jr)
    ev = np.linalg.eigvals(T)
    ph = np.angle(ev)

    return float(np.angle(np.exp(0.5j*(ph[0] + ph[1])))), \
        float(np.abs(np.angle(np.exp(1j*(ph[0] - ph[1])))))


def polarized_psf(J, E_in=None, pad=8):
    """Far field intensity of a beam whose pupil carries the Jones matrices
    J (from `jones_pupil()`), for an input polarization E_in, or
    unpolarized light (E_in None: the mean of x and y input).

    Returns
    -------
    I : 2d array
        Intensity on the grid of the zero padded FFT (pad times finer than
        the diffraction limit), normalized to the peak of a pupil with the
        same transmitted power, a uniform polarization and phase: I.max() is
        the Strehl ratio.
    """

    n = J.shape[0]
    N = pad*n
    mask = np.isfinite(J[..., 0, 0])
    inputs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])] if E_in is None else [np.asarray(E_in)]
    I, ref = 0.0, 0.0
    for E in inputs:
        E = E/np.linalg.norm(E)
        F = np.where(mask[..., None], np.nan_to_num(J) @ E, 0.0)
        for c in range(2):
            A = np.zeros((N, N), dtype=complex)
            A[:n, :n] = F[..., c]
            I = I + np.abs(np.fft.fftshift(np.fft.fft2(A)))**2
        # peak of a pupil with the same transmitted power and no
        # polarization or phase effects
        ref = ref + np.sum(np.abs(F)**2)*mask.sum()

    return I/ref


def phase_coating(material=None, angle=None, wavelengths=(450e-6, 550e-6, 650e-6), n_layers=4,
                  n_high=2.35, n_low=1.46, start=None, prisms=None, seed=0):
    """Phase correcting coating for the roof faces: an alternating stack of
    high and low index layers (from the glass outwards, TIR towards air)
    whose thicknesses make the reflection phases of s and p equal at the
    angle of incidence on the roof faces (the retardance of each roof
    reflection zero), which makes the two halves of the pupil alike.

    Parameters
    ----------
    material : glass, optional
        Default N-BAK4.
    angle : number, optional
        Angle of incidence on the roof faces in radians; default the
        49.2 degrees of the Schmidt-Pechan prism (arccos(cos 22.5 /
        sqrt 2)).
    wavelengths : sequence, optional
        Wavelengths (mm) over which the retardance is minimized.
    n_layers : int, optional
    n_high, n_low : numbers, optional
        Indices of the layer materials (TiO2-like and SiO2-like).
    start : sequence, optional
        Start thicknesses (mm).
    prisms : optional
        Unused (reserved).
    seed : int, optional

    Returns
    -------
    coating : coatings.Coating
    retardance : array
        Remaining retardance (radians) at the wavelengths.
    """

    from scipy.optimize import least_squares

    material = _bak4() if material is None else material
    angle = np.arccos(np.cos(np.radians(22.5))*np.sqrt(0.5)) if angle is None else angle
    wls = np.asarray(wavelengths, float)
    idx = [n_high if k % 2 == 0 else n_low for k in range(n_layers)]

    def retard(t):
        c = Coating([(ni, ti) for ni, ti in zip(idx, t)], substrate=1.0)
        out = []
        for w in wls:
            n = complex_index(material, w*1e6).real
            r_s, r_p = c.amplitudes(np.array([np.cos(angle)]), n, 1.0, w, w*1e6)[:2]
            out.append(np.angle(r_p[0]/r_s[0]))
        return np.array(out)

    lam0 = wls[len(wls)//2]
    rng = np.random.default_rng(seed)
    best = None
    starts = [start] if start is not None else \
        [np.array([lam0/(4*ni)*f for ni, f in zip(idx, rng.uniform(0.3, 1.7, n_layers))])
         for _ in range(12)]
    for t0 in starts:
        res = least_squares(lambda t: np.angle(np.exp(1j*retard(t))), t0,
                            bounds=(np.full(n_layers, 1e-6), np.full(n_layers, lam0)))
        if best is None or res.cost < best.cost:
            best = res
    t = best.x
    coating = Coating([(ni, ti) for ni, ti in zip(idx, t)], substrate=1.0, label="phase coating")

    return coating, np.angle(np.exp(1j*retard(t)))


def tir_retardance(n, angle):
    """Retardance phi_p - phi_s (radians) of a bare TIR reflection inside a
    glass of index n at the angle of incidence (radians), in the field
    ratio convention of `thin_films`, where r_p carries the opposite sign
    of the Born and Wolf convention: the retardance is pi minus the
    textbook delta with tan(delta/2) = cos(theta) sqrt(sin^2 theta - 1/n^2)
    / sin^2 theta, pi at the critical angle and at grazing incidence."""

    c = Coating((), substrate=1.0)
    angle = np.atleast_1d(angle)
    r_s, r_p = c.amplitudes(np.cos(angle), n, 1.0, 550e-6)[:2]

    return np.angle(r_p/r_s)
