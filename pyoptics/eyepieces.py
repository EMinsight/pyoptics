#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Eyepieces: Huygens, Ramsden, Kellner, Plössl and Erfle designs traced
backwards from the eye pupil, with their eye relief, apparent field, field
curvature in diopters, lateral colour and the spherical aberration of the
pupil (notebook 171, eyepieces).

An eyepiece is used with a collimated beam on the eye side: every point of
the field stop (the image plane of the objective) becomes a parallel beam
that enters the eye's pupil. Eyepieces are therefore designed and analysed
*backwards*: the eye pupil is the aperture stop, the object is at infinity
on the eye side, the field angle is the apparent field angle, and the image
plane is the field stop. In this reversed system

- the focal length is the focal length f of the eyepiece; the magnifying
  power of a telescope is M = f_objective/f, its exit pupil D/M,
- the distance from the eye pupil to the first lens vertex is the eye
  relief; with the objective at the distance L behind the field stop, the
  eye pupil sits where the eyepiece images the objective
  (`Eyepiece.eye_relief()`), for L -> infinity at the eye-side focal
  point of the eyepiece,
- the chief ray height in the focal plane is the radius of the field stop
  for an apparent field angle (`apparent_field()`); the distortion of an
  eyepiece is the departure of this height from f tan(theta),
- transverse ray errors in the focal plane, divided by f, are the angular
  errors that the eye sees.

Lengths are in mm. The prescriptions (`huygens()`, `ramsden()`,
`kellner()`, `ploessl()`, `erfle()`) are for f = 25 mm and scale with
`focal_length`; their surfaces are listed from the eye to the field stop.
Huygens and Ramsden eyepieces follow the classical thin lens rules (Huygens:
plano-convex lenses of focal lengths 3a and a, a distance 2a apart; Ramsden:
two equal plano-convex lenses at two thirds of their focal length) with
lenses of real thickness, the spacing adjusted to bring the first order
lateral colour (Huygens) or the field stop (Ramsden) to the intended place.
The Kellner, Plössl and Erfle eyepieces were optimized for pyoptics with
damped least squares from first order layouts in N-BAK4 and N-SF2 (the
Plössl kept symmetric: two identical cemented doublets, crowns facing each
other), for a 4 mm eye pupil, the objective 250 mm behind the field stop
and the apparent fields of the type; they show the behaviour of each type,
they are not copies of particular commercial designs.

Sign conventions are those of `raytrace` along the reversed path: light
travels in +z from the eye pupil (z < 0 in front of the first lens) to the
field stop. Field curvature and astigmatism are given as the accommodation
in diopters that a point of the field stop demands from the eye
(`field_curves()`): positive values mean that the beam leaving the
eyepiece diverges (a virtual image at a finite distance, which the eye can
accommodate), negative values a converging beam (which only a far-sighted
eye or a refocus can take).

Functions:

- huygens(), ramsden(), kellner(), ploessl(), erfle(), EYEPIECES: the
  designs as `Eyepiece` objects.
- Eyepiece.system(): the reversed `raytrace.System` with the eye pupil as
  the stop; Eyepiece.forward_surfaces(): the lens in the direction of the
  light (objective to eye) for telescopes; Eyepiece.eye_relief().
- apparent_field(), distortion(), lateral_colour(), field_curves(),
  angular_blur(): first order and image quality measures in the units of
  visual instruments (degrees, arcminutes, diopters).
- pupil_aberration(): where the chief rays of the forward path cross the
  axis behind the eyepiece (spherical aberration of the exit pupil, the
  cause of the "kidney beans" of wide angle eyepieces).

References
----------
R. Kingslake and R. B. Johnson, "Lens Design Fundamentals", 2nd ed.,
Academic Press (2010), chapter 14 (eyepieces).

W. J. Smith, "Modern Optical Engineering", 4th ed., McGraw-Hill (2008),
chapter 13 (eyepieces, magnifiers).

H. Rutten and M. van Venrooij, "Telescope Optics: Evaluation and Design",
Willmann-Bell (1988), chapters 21-23.

H. Dennis Taylor, "A System of Applied Optics", Macmillan (1906) (field
lens and lateral colour of two-lens eyepieces).
"""

import copy

import numpy as np

from . import raytrace as rt


WAVELENGTHS = {"F": 486.13e-6, "d": 587.56e-6, "C": 656.27e-6}  # mm
ARCMIN = np.pi/(180*60)


class Eyepiece:
    """An eyepiece described from the eye pupil to the field stop.

    Parameters
    ----------
    name : str
    surfaces : sequence of raytrace.Surface
        The lens surfaces from the eye side to the field stop side; the
        thickness of the last one is ignored (the field stop follows from
        the paraxial focus).
    apparent_field : number
        Full apparent field of the design in degrees.
    pupil : number, optional
        Eye pupil diameter of the design in mm.
    eye_relief : number, optional
        Distance from the eye pupil to the first vertex used by `system()`
        when no objective distance is given; default: the eye relief for an
        objective 250 mm behind the field stop (10 f for f = 25 mm).
    """

    def __init__(self, name, surfaces, apparent_field, pupil=4.0, eye_relief=None):
        self.name = name
        self.surfaces = [copy.copy(s) for s in surfaces]
        self.apparent_field = apparent_field
        self.pupil = pupil
        self._eye_relief = eye_relief

    def __repr__(self):
        return "Eyepiece({!r}, f={:.2f} mm, apparent field {:.0f} deg, eye relief {:.1f} mm)".format(
            self.name, self.focal_length, self.apparent_field, self.eye_relief())

    def _lens_system(self, eye_relief=0.0, pupil=None, field=None, wavelength=WAVELENGTHS["d"]):
        pupil = self.pupil if pupil is None else pupil
        field = self.apparent_field/2 if field is None else field
        S = [rt.Surface(np.inf, eye_relief, semi_diameter=pupil/2, label="eye pupil")]
        S += [copy.copy(s) for s in self.surfaces]
        S[-1].thickness = 0.0
        system = rt.System(S, 0, wavelength, field_angle=np.radians(field))
        system.surfaces[-1].thickness = system.paraxial().image_distance

        return system

    @property
    def focal_length(self):
        """Effective focal length in mm (d line)."""

        return self._lens_system().paraxial().efl

    def back_focus(self, wavelength=WAVELENGTHS["d"]):
        """Distance from the last vertex (objective side) to the image plane
        of the objective, the image plane of the reversed trace. This is
        the field stop for all types but the Huygens eyepiece, whose
        objective image would lie inside the eyepiece (negative values):
        the field lens catches the converging beam first, and the field
        stop sits at the real intermediate image in the focal plane of the
        eye lens (`intermediate_image()`)."""

        return self._lens_system(wavelength=wavelength).surfaces[-1].thickness

    def intermediate_image(self):
        """Distance of the real image between the lenses of a Huygens
        eyepiece (its field stop) behind the eye lens vertex, from the
        reversed paraxial trace (None if the reversed beam has no real
        focus inside the eyepiece)."""

        s = self._lens_system()
        ys, us = s._paraxial_trace(1.0, 0.0, 0, len(s.surfaces))
        z = np.cumsum([0.0] + [x.thickness for x in s.surfaces])
        for i in range(1, len(s.surfaces) - 1):
            if ys[i]*ys[i + 1] < 0 and s.n[i + 1] == 1.0:
                return z[i] - ys[i]/us[i] - z[1]

        return None

    def eye_relief(self, objective_distance=250.0, wavelength=WAVELENGTHS["d"]):
        """Distance from the first lens vertex to the exit pupil (the
        image of the objective).

        Parameters
        ----------
        objective_distance : number, optional
            Distance of the objective (the entrance pupil of the telescope)
            behind the field stop; np.inf: the eye-side focal point.
        wavelength : number, optional

        Returns
        -------
        number
        """

        if self._eye_relief is not None and objective_distance == 250.0:
            return self._eye_relief
        s = self._lens_system(wavelength=wavelength)
        M = s._matrix(1, len(s.surfaces))
        A, B, C, D = M.ravel()
        if np.isinf(objective_distance):
            return -D/C
        T = s.surfaces[-1].thickness + objective_distance

        return -(B + T*D)/(A + T*C)

    def system(self, pupil=None, field=None, objective_distance=250.0, eye_relief=None,
               wavelength=WAVELENGTHS["d"]):
        """The eyepiece traced backwards: eye pupil (stop), lenses, field
        stop (image plane in the paraxial focus).

        Parameters
        ----------
        pupil : number, optional
            Eye pupil diameter (default: the design pupil).
        field : number, optional
            Maximum apparent half field in degrees (default: the design).
        objective_distance : number, optional
            Sets the eye relief, see `eye_relief()`.
        eye_relief : number, optional
            Overrides the eye relief.
        wavelength : number, optional

        Returns
        -------
        raytrace.System
        """

        er = self.eye_relief(objective_distance, wavelength) if eye_relief is None else eye_relief
        s = self._lens_system(er, pupil, field, wavelength)
        if wavelength != WAVELENGTHS["d"]:
            # the field stop stays where it is for the d line
            s.surfaces[-1].thickness = self.back_focus()

        return s

    def scaled(self, focal_length):
        """The same design for another focal length."""

        k = focal_length/self.focal_length
        S = [copy.copy(s) for s in self.surfaces]
        for s in S:
            s.R, s.thickness = s.R*k, s.thickness*k
            if s.semi_diameter is not None:
                s.semi_diameter *= k

        return Eyepiece(self.name, S, self.apparent_field, self.pupil,
                        None if self._eye_relief is None else self._eye_relief*k)

    def forward_surfaces(self, wavelength=WAVELENGTHS["d"]):
        """The lens surfaces in the direction of the light in a telescope
        (field stop -> eye), with the media of the reversed path.

        The thickness of the last surface is the eye relief for an objective
        far away (np.inf), to be replaced by the user. Prepend a surface or
        a gap that ends at the field stop: the first lens vertex lies
        back_focus() behind it.

        Returns
        -------
        list of raytrace.Surface
        """

        S = self.surfaces
        media = [None]  # medium in front of each surface: None = air
        for s in S:
            media.append(s.material if s.material is not None else
                         (s.n_after if s.n_after is not None else media[-1]))
        out = []
        for i in reversed(range(len(S))):
            s = copy.copy(S[i])
            s.R = -s.R
            s.aspheric = tuple(-a for a in s.aspheric)
            before = media[i]
            if before is None or (np.isscalar(before) and before == 1.0):
                s.material, s.n_after = None, 1.0
            elif np.isscalar(before):
                s.material, s.n_after = None, before
            else:
                s.material, s.n_after = before, None
            s.thickness = S[i - 1].thickness if i > 0 else self.eye_relief(np.inf, wavelength)
            out.append(s)

        return out


# --- the designs (f = 25 mm, surfaces from the eye side) ---------------------

def _glass(name):
    from . import glasses
    return glasses.get(name)


def huygens(focal_length=25.0):
    """Huygens eyepiece: two plano-convex N-BK7 lenses, curved sides towards
    the objective, field lens focal length three times the eye lens focal
    length, about twice the eye lens focal length apart. The field stop lies
    between the lenses. Apparent field 40 degrees."""

    bk7 = _glass("N-BK7")
    S = [rt.Surface(np.inf, 2.5, material=bk7), rt.Surface(-8.80, 34.10, n_after=1.0),
         rt.Surface(np.inf, 3.0, material=bk7), rt.Surface(-25.90, 0.0, n_after=1.0)]

    return Eyepiece("Huygens", S, 40.0, pupil=4.0).scaled(focal_length)


def ramsden(focal_length=25.0):
    """Ramsden eyepiece: two equal plano-convex N-BK7 lenses, curved sides
    facing each other, two thirds of their focal length apart; the field
    stop lies a quarter of the focal length in front of the field lens.
    Apparent field 35 degrees."""

    bk7 = _glass("N-BK7")
    S = [rt.Surface(np.inf, 3.0, material=bk7), rt.Surface(-17.40, 20.30, n_after=1.0),
         rt.Surface(17.40, 3.0, material=bk7), rt.Surface(np.inf, 0.0, n_after=1.0)]

    return Eyepiece("Ramsden", S, 35.0, pupil=4.0).scaled(focal_length)


# (R, thickness, glass) from the eye side, f = 25 mm (scaled exactly by the
# builders); glass None: air behind the surface
DESIGNS = {
    "kellner": [(128.867, 5.0, "N-BAK4"), (-10.642, 1.5, "N-SF6"), (-20.412, 13.237, None),
                (21.816, 4.0, "N-BK7"), (np.inf, 0.0, None)],
    "ploessl": [(-287.5, 1.5, "N-SF2"), (28.7, 6.0, "N-BAK4"), (-24.5, 0.3, None),
                (24.5, 6.0, "N-BAK4"), (-28.7, 1.5, "N-SF2"), (287.5, 0.0, None)],
    "erfle": [(201.204, 1.5, "N-SF6"), (29.540, 8.0, "N-BAK4"), (-50.132, 0.5, None),
              (104.170, 7.0, "N-BAK4"), (-40.266, 0.5, None),
              (62.221, 8.0, "N-BAK4"), (-29.688, 1.5, "N-SF6"), (-50.914, 0.0, None)],
}


def _from_table(name, table, apparent_field):
    S = []
    for R, t, glass in table:
        if glass is None:
            S.append(rt.Surface(R, t, n_after=1.0))
        else:
            S.append(rt.Surface(R, t, material=_glass(glass)))

    return Eyepiece(name, S, apparent_field, pupil=4.0)


def kellner(focal_length=25.0):
    """Kellner eyepiece: a Ramsden with a cemented achromatic eye lens
    (N-BAK4 crown towards the eye, N-SF2 flint) and a plano-convex field
    lens. Apparent field 45 degrees."""

    return _from_table("Kellner", DESIGNS["kellner"], 45.0).scaled(focal_length)


def ploessl(focal_length=25.0):
    """Plössl (symmetrical) eyepiece: two identical cemented doublets
    (N-SF2 flint outside, N-BAK4 crown inside), crowns facing each other
    across a small air gap. Apparent field 50 degrees."""

    return _from_table("Plössl", DESIGNS["ploessl"], 50.0).scaled(focal_length)


def erfle(focal_length=25.0):
    """Erfle eyepiece: cemented doublet, biconvex singlet and cemented
    doublet (N-BAK4 and N-SF2). Apparent field 65 degrees."""

    return _from_table("Erfle", DESIGNS["erfle"], 65.0).scaled(focal_length)


def eyepieces(focal_length=25.0):
    """All five designs, ordered by apparent field."""

    return [f(focal_length) for f in (ramsden, huygens, kellner, ploessl, erfle)]


# --- measures -----------------------------------------------------------------

def _chief(system, theta, wavelength=None):
    s = system if wavelength is None else system.with_wavelength(wavelength)
    b = s.trace_pupil(theta, 0.0, 0.0, clip=False)
    I, d = s._image_hits(b)

    return I[0], d[0]


def apparent_field(system, field_stop_radius, wavelength=None):
    """Apparent half field angle (degrees) at which the real chief ray
    reaches the rim of the field stop.

    Parameters
    ----------
    system : raytrace.System
        Reversed eyepiece (`Eyepiece.system()`).
    field_stop_radius : number
    wavelength : number, optional

    Returns
    -------
    number
    """

    lo, hi = 0.0, np.radians(60.0)
    for _ in range(60):
        mid = 0.5*(lo + hi)
        h = _chief(system, mid, wavelength)[0][1]
        if np.isfinite(h) and h < field_stop_radius:
            lo = mid
        else:
            hi = mid

    return np.degrees(0.5*(lo + hi))


def distortion(system, fields):
    """Distortion h/(f tan(theta)) - 1 of the eyepiece at the apparent half
    field angles (degrees): the chief ray height h in the field stop against
    a distortion-free eyepiece.

    Returns
    -------
    array
    """

    f = system.paraxial().efl
    out = [_chief(system, np.radians(t))[0][1]/(f*np.tan(np.radians(t))) - 1 if t else 0.0
           for t in np.atleast_1d(fields)]

    return np.array(out)


def lateral_colour(system, fields, wavelengths=(WAVELENGTHS["F"], WAVELENGTHS["C"])):
    """Lateral colour in arcminutes: the difference of the chief ray
    heights of two wavelengths in the field stop, divided by f (the
    angular separation of the coloured images of a point at the rim, as
    the eye sees them).

    Returns
    -------
    array
    """

    f = system.paraxial().efl
    out = []
    for t in np.atleast_1d(fields):
        h1 = _chief(system, np.radians(t), wavelengths[0])[0][1]
        h2 = _chief(system, np.radians(t), wavelengths[1])[0][1]
        out.append((h1 - h2)/f/ARCMIN)

    return np.array(out)


def _focus_along_chief(system, theta, delta, wavelength=None):
    """Positions along the chief ray (measured as z in the image frame) of
    the tangential and sagittal foci from neighbouring rays at the pupil
    offsets +-delta."""

    s = system if wavelength is None else system.with_wavelength(wavelength)
    px = np.array([0.0, 0.0, 0.0, -delta, delta])
    py = np.array([0.0, -delta, delta, 0.0, 0.0])
    b = s.trace_pupil(theta, px, py, clip=False)
    I, d = s._image_hits(b)
    z = []
    for (i, j), e in (((1, 2), np.array([0.0, 1.0, 0.0])), ((3, 4), np.array([1.0, 0.0, 0.0]))):
        # distance in z behind the field stop where the transverse
        # separation of the neighbouring rays i and j vanishes
        a = np.dot(I[i] - I[j], e)
        b_ = np.dot(d[i]/d[i][2] - d[j]/d[j][2], e)
        z.append(-a/b_)

    return np.array(z)


def field_curves(system, fields, wavelength=None, delta=0.05):
    """Tangential and sagittal field curves of the eyepiece as the
    accommodation (diopters) that a point of the field stop demands: the
    distance dz (mm, along z) of the real ray foci from the field stop,
    1000 dz/f**2. Positive: the beam leaving the eyepiece diverges.

    Parameters
    ----------
    system : raytrace.System
        Reversed eyepiece.
    fields : array
        Apparent half field angles in degrees.
    wavelength : number, optional
    delta : number, optional
        Pupil offset of the neighbouring rays (fraction of the pupil).

    Returns
    -------
    tangential, sagittal : arrays
    """

    f = system.paraxial().efl
    zs = np.array([_focus_along_chief(system, np.radians(t), delta, wavelength)
                   for t in np.atleast_1d(fields)])

    return 1000*zs[:, 0]/f**2, 1000*zs[:, 1]/f**2


def angular_blur(system, fields, wavelengths=None, refocus=True, N=15):
    """RMS angular blur (arcminutes) of the eyepiece over the eye pupil,
    polychromatic, relative to the d line chief ray.

    Parameters
    ----------
    system : raytrace.System
        Reversed eyepiece.
    fields : array
        Apparent half field angles in degrees.
    wavelengths : sequence, optional
        Default F, d, C.
    refocus : bool, optional
        Accommodate: evaluate each field in its best focus (smallest RMS),
        as the eye does when it looks at that part of the field. False: in
        the field stop.
    N : int, optional
        Rays across the pupil diameter.

    Returns
    -------
    array
    """

    wavelengths = list(WAVELENGTHS.values()) if wavelengths is None else wavelengths
    f = system.paraxial().efl
    p = np.linspace(-1, 1, N)
    PX, PY = np.meshgrid(p, p)
    inside = PX**2 + PY**2 <= 1
    px, py = PX[inside], PY[inside]
    out = []
    for t in np.atleast_1d(fields):
        theta = np.radians(t)
        I0, d0 = _chief(system, theta)
        pts, dirs = [], []
        for w in wavelengths:
            s = system.with_wavelength(w)
            I, d = s._image_hits(s.trace_pupil(theta, px, py, clip=False))
            pts.append(I)
            dirs.append(d)
        I, d = np.concatenate(pts), np.concatenate(dirs)
        ok = np.all(np.isfinite(I), axis=1)
        I, d = I[ok], d[ok]

        def rms(dz):
            x = I[:, 0] + dz*d[:, 0]/d[:, 2] - (I0[0] + dz*d0[0]/d0[2])
            y = I[:, 1] + dz*d[:, 1]/d[:, 2] - (I0[1] + dz*d0[1]/d0[2])
            return np.sqrt(np.mean(x**2 + y**2))

        if refocus:
            from scipy.optimize import minimize_scalar
            dz = minimize_scalar(rms, bounds=(-0.05*f, 0.05*f), method="bounded",
                                 options={"xatol": 1e-5*f}).x
        else:
            dz = 0.0
        out.append(rms(dz)/f/ARCMIN)

    return np.array(out)


def pupil_aberration(eyepiece, fields, objective_distance=250.0, wavelength=WAVELENGTHS["d"]):
    """Spherical aberration of the exit pupil: where the chief rays of the
    forward path (from the centre of the objective through the field stop
    and the eyepiece) cross the axis, as the distance from the last lens
    vertex, for the apparent half field angles of the reversed trace.

    The real chief rays of the reversed system start at the centre of the
    eye pupil; traced forwards from the centre of the objective, the
    chief ray of a field point crosses the axis at the eye relief only near
    the axis. Wide angle eyepieces bring the outer chief rays to the axis
    closer to the lens: an eye at the paraxial pupil loses parts of the
    field to the rim of its pupil (kidney beans), an eye further back loses
    the rim of the field.

    Parameters
    ----------
    eyepiece : Eyepiece
    fields : array
        Apparent half field angles in degrees (paraxial: tan(theta) = h/f).
    objective_distance : number, optional
    wavelength : number, optional

    Returns
    -------
    z : array
        Axis crossing distances from the eye-side vertex (the paraxial
        value is the eye relief).
    angle : array
        Real angles (degrees) of the chief rays with the axis at the eye.
    """

    S = eyepiece.forward_surfaces(wavelength)
    bf = eyepiece.back_focus()
    L = objective_distance
    S[-1].thickness = 0.0
    # the objective (stop, a point source of chief rays at its centre) at
    # z = 0, its image plane L behind it, the first lens vertex
    # back_focus() behind that plane
    system = rt.System([rt.Surface(np.inf, L + bf, semi_diameter=1.0, label="objective")] + S, 0,
                       wavelength, object_distance=1.0, object_height=0.0)
    f = eyepiece.focal_length
    zs, angles = [], []
    for t in np.atleast_1d(fields):
        h = f*np.tan(np.radians(t))
        u = np.arctan(h/L)
        P = np.array([[0.0, 0.0, 0.0]])
        d = np.array([[0.0, np.sin(u), np.cos(u)]])
        b = system.trace(rt.Rays(P, d, np.zeros(1)), clip=False, last=len(system.surfaces) - 1)
        Q, k = b.points[-1, 0], b.directions[-1, 0]
        z_last = system.vertex_z[-1]
        t_cross = -Q[1]/k[1]
        zs.append(Q[2] + t_cross*k[2] - z_last)
        angles.append(np.degrees(np.arctan2(-k[1], k[2])))

    return np.array(zs), np.array(angles)


def telescope(eyepiece, objective_focal_length=250.0, objective_diameter=None, field=None,
              exit_pupil_distance=None, wavelength=WAVELENGTHS["d"], objective=None):
    """A Keplerian telescope in the direction of the light: a perfect
    objective (`raytrace.IdealLens`, the aperture stop) and the eyepiece,
    with the image plane in the eye pupil.

    Parameters
    ----------
    eyepiece : Eyepiece
    objective_focal_length : number, optional
    objective_diameter : number, optional
        Default: the design pupil of the eyepiece times the magnifying
        power (the exit pupil equals the design pupil).
    field : number, optional
        Maximum real half field angle in degrees; default: the apparent
        half field of the eyepiece divided by the magnifying power.
    exit_pupil_distance : number, optional
        Distance from the last eyepiece vertex to the image plane (the eye
        pupil); default: the paraxial eye relief.
    wavelength : number, optional
    objective : sequence of raytrace.Surface, optional
        A real objective instead of the ideal lens: its surfaces, the first
        one the stop, the thickness of the last one the distance to the
        objective's focal plane. Prisms (plane parallel blocks) may be
        included.

    Returns
    -------
    raytrace.System
        Object at infinity; the image plane is the eye pupil, where the
        output beams are collimated: their footprints show the exit pupil,
        their directions the apparent field. `angular_magnification()`
        traces the magnifying power.
    """

    f_e = eyepiece.focal_length
    M = objective_focal_length/f_e
    D = eyepiece.pupil*M if objective_diameter is None else objective_diameter
    field = eyepiece.apparent_field/2/M if field is None else field
    er = eyepiece.eye_relief(objective_focal_length, wavelength) if exit_pupil_distance is None \
        else exit_pupil_distance
    S = eyepiece.forward_surfaces(wavelength)
    S[-1].thickness = er
    if objective is None:
        front = [rt.IdealLens(objective_focal_length, objective_focal_length + eyepiece.back_focus(),
                              semi_diameter=D/2, label="objective")]
        stop = 0
    else:
        front = [copy.copy(s) for s in objective]
        front[-1].thickness = front[-1].thickness + eyepiece.back_focus()
        stop = 0
    system = rt.System(front + S, stop, wavelength, field_angle=np.radians(field))

    return system


def angular_magnification(system, field):
    """Ratio tan(theta')/tan(theta) of the real chief ray at the real field
    angle theta (degrees) of a telescope from `telescope()`: the angular
    magnification (negative: inverted image)."""

    b = system.trace_pupil(np.radians(field), 0.0, 0.0, clip=False)
    d = b.directions[-1, 0]

    return (d[1]/d[2])/np.tan(np.radians(field))


def pupil_footprint(system, field, n=41):
    """Points (x, y) where the rays of the full objective aperture at the
    real field angle (degrees) cross the image plane (the eye pupil plane)
    of a telescope from `telescope()`; vignetted rays are left out."""

    p = np.linspace(-1, 1, n)
    PX, PY = np.meshgrid(p, p)
    m = PX**2 + PY**2 <= 1
    b = system.trace_pupil(np.radians(field), PX[m], PY[m])
    I = b.points[-1]
    ok = b.valid & np.all(np.isfinite(I), axis=1)

    return I[ok, 0], I[ok, 1]


def eye_transmission(system, fields, eye_pupil, offset=0.0, n=61):
    """Fraction of the light of each real field angle (degrees) that enters
    an eye pupil of the given diameter, centred on the axis in the image
    plane of a telescope from `telescope()` moved by `offset` along the axis
    (positive: away from the eyepiece). Shows the kidney beaning of
    eyepieces with spherical aberration of the pupil."""

    p = np.linspace(-1, 1, n)
    PX, PY = np.meshgrid(p, p)
    m = PX**2 + PY**2 <= 1
    out = []
    for t in np.atleast_1d(fields):
        b = system.trace_pupil(np.radians(t), PX[m], PY[m], clip=False)
        I, d = b.points[-1], b.directions[-1]
        I = I + offset*d/d[:, 2:3]
        r = np.hypot(I[:, 0], I[:, 1])
        out.append(np.mean(np.isfinite(r) & (r <= eye_pupil/2)))

    return np.array(out)


def afocal_blur(system, fields, wavelengths=None, accommodate=True, N=15):
    """RMS angular blur (arcminutes) of an afocal instrument from
    `telescope()`, from the directions of the rays leaving the eyepiece,
    relative to the d line chief ray.

    Parameters
    ----------
    system : raytrace.System
        Telescope with the image plane at the eye pupil.
    fields : array
        Real half field angles in degrees.
    wavelengths : sequence, optional
        Default F, d, C.
    accommodate : bool, optional
        Remove the best vergence (the eye focuses): the part of the angular
        errors proportional to the ray position in the eye pupil.
    N : int, optional
        Rays across the objective.

    Returns
    -------
    array
    """

    wavelengths = list(WAVELENGTHS.values()) if wavelengths is None else wavelengths
    p = np.linspace(-1, 1, N)
    PX, PY = np.meshgrid(p, p)
    inside = PX**2 + PY**2 <= 1
    px, py = PX[inside], PY[inside]
    out = []
    for t in np.atleast_1d(fields):
        theta = np.radians(t)
        bc = system.trace_pupil(theta, 0.0, 0.0, clip=False)
        Ic, dc = bc.points[-1, 0], bc.directions[-1, 0]
        ac = dc[:2]/dc[2]
        pos, ang = [], []
        for w in wavelengths:
            s = system.with_wavelength(w)
            b = s.trace_pupil(theta, px, py, clip=False)
            I, d = b.points[-1], b.directions[-1]
            pos.append(I[:, :2] - Ic[:2])
            ang.append(d[:, :2]/d[:, 2:3] - ac)
        pos, ang = np.concatenate(pos), np.concatenate(ang)
        ok = np.all(np.isfinite(ang), axis=1) & np.all(np.isfinite(pos), axis=1)
        pos, ang = pos[ok], ang[ok]
        if accommodate:
            V = np.sum(pos*ang)/np.sum(pos*pos)
            ang = ang - V*pos
        out.append(np.sqrt(np.mean(np.sum(ang**2, axis=1)))/ARCMIN)

    return np.array(out)
