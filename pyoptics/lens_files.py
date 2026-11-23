#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Reading and writing lens prescriptions: patent tables, Zemax and CODE V.

A `Prescription` is what a lens file holds: the surfaces of a sequential
`raytrace.System`, the aperture stop, the entrance pupil diameter, the
wavelengths, the fields and the object distance. ``Prescription.system()``
builds the `System` at any of its wavelengths, ``Prescription.from_system()``
collects it from a `System`. The readers and writers ::

    from pyoptics import lens_files
    lens = lens_files.read("doublet.zmx")       # by extension: .zmx, .seq, else a table
    system = lens.system()                      # at the primary wavelength
    lens_files.write(system, "doublet.seq", wavelengths=(486.1e-6, 587.6e-6, 656.3e-6))
    text = lens_files.write_patent(system)      # a table as in a patent

handle three formats:

- Patent tables (`read_patent()`, `write_patent()`): plain text or CSV with
  one row per surface (surface number, radius, thickness, n_d, V_d,
  semi-diameter, glass) and a second table of conic constants and aspheric
  coefficients, as lens prescriptions are printed in patents and in the
  literature. The reader is tolerant of the usual variations: "∞" or "inf",
  blank cells for air, "1.2E-05", "1.2D-05" or "1.2×10^-5", aspheric
  coefficients as a table (K, A4, A6, ...) or as "K = ..., A4 = ..." per
  surface, "STO" and "*" marks in the surface column, whitespace columns
  aligned with their header, CSV and decimal commas. Metadata (wavelengths,
  entrance pupil diameter, fields, units) are comment lines ``# key: value``.
- Zemax OpticStudio sequential files (`read_zmx()`, `write_zmx()`, ``.zmx``,
  text; UTF-16 or UTF-8): surface types STANDARD, EVENASPH, TOROIDAL,
  BICONIC, COORDBRK and PARAXIAL, glasses by catalogue name
  (``GLAS N-BK7 ...``) or as model glasses (``GLAS ___BLANK 1 0 n_d V_d``),
  mirrors (``GLAS MIRROR``), the entrance pupil diameter (ENPD), the image
  space F-number (FNUM) or "float by stop size" (FLOA), field angles,
  object heights or paraxial or real image heights, and wavelengths.
  Tested on files exported by OpticStudio (Thorlabs and Edmund catalogue
  lenses, prescriptions from patents).
- CODE V sequence files (`read_seq()`, `write_seq()`, ``.seq``): the
  commands S, SO, SI, STO, CIR, ASP/CON with K and A ... J, YTO with RDX
  or CUX, REFL, the decentres XDE, YDE, ADE, BDE, CDE with DAR (decentre
  and return) or as basic decentres (all following surfaces), and the
  system data DIM, WL, REF, EPD, XAN/YAN and XOB/YOB.

Only the publicly documented and commonly seen structure of the files is
used; neither sample files nor glass catalogues of the vendors are needed.

Materials
---------
Every surface of a file names the medium behind it (blank: air). A glass
name is looked up in the catalogues of `pyoptics.glasses` (SCHOTT, OHARA,
HOYA, CDGM, SUMITA, HIKARI; case, hyphens and spaces do not matter, a CODE
V vendor suffix such as ``_SCHOTT`` selects the vendor); a name that is not
found falls back to the n_d and V_d given with it in the file (the Abbe
model of `materials.abbe_dispersion()`) with a warning. ``use_catalog=False``
always uses n_d and V_d. V_d = 0 marks a medium without dispersion (an
index n_after; a pyoptics convention, other programs may not accept it).
CODE V model glasses are glass codes ``nnnnnn.vvvvvv``: n_d = 1.nnnnnn,
V_d = vv.vvvv. Names of obsolete glasses that are not in the catalogues
("BK7", "SF11") fall back to their lead free successors ("N-BK7",
"N-SF11"), with a warning; a suffix such as "_MOLD" (molded lenses) is
dropped; fused silica under its usual names ("F_SILICA", "C79-80", "UVFS",
...) is `raytrace.fused_silica`. Placeholder indices that some programs
write for glasses they could not find (n_d = 1.5, V_d = 40) are only used
when nothing else is known.

The writers name catalogue glasses (`glasses.Glass`) by name, give (n_d,
V_d) materials as model glasses and constant indices with V_d = 0. Other
materials (functions n(wavelength_nm), `materials.Material`) are written as
the model glass with their n_d and V_d, with a warning: the dispersion is
then only approximated. ``index_offset`` is added to n_d.

Conventions
-----------
Radii, thicknesses and the sign rules are those of `raytrace` in all three
formats (R > 0: centre of curvature behind the vertex; thicknesses negative
behind mirrors). A plane is "inf" in a patent table, curvature 0 in Zemax
and radius 0 in CODE V. Field angles are in degrees in the files (radians
in pyoptics), wavelengths in nm (patent tables, CODE V) or um (Zemax).

A decentred or tilted surface (``decenter``, ``tilt``) is written to Zemax
as a coordinate break before it and one that undoes it behind it (the
second with order flag 1), and read back into a displaced surface; to CODE
V as XDE ... CDE with DAR. Tilts are taken to be right-handed rotations
about x, y and z in this order (``a_x > 0`` turns +z towards -y, see
`raytrace`) in both programs; the signs are not checked against the
programs themselves.

Not covered: multiple configurations, solves and pickups (their current
values are read), apertures other than circular semi-diameters, surface
types other than those above (Zernike, gratings, diffractive and Q-type
surfaces raise NotImplementedError), coatings, and units other than mm,
cm and m.

References
----------
Zemax OpticStudio User Manual (surface types, coordinate breaks).

CODE V Command Reference Manual, Synopsys (sequence file commands).

W. J. Smith, "Modern Lens Design", 2nd ed., McGraw-Hill (2005) (lens
prescription tables).
"""

import math
import os
import re
import shlex
import warnings

import numpy as np

from . import glasses as _glasses
from .materials import LAMBDA_C, LAMBDA_F, LAMBDA_d, Material
from .raytrace import (NM_PER_UNIT, BiconicSurface, CoordinateBreak, IdealLens, Surface, System,
                       ToroidalSurface, _field_xy)


__all__ = ["Prescription", "read", "write", "read_patent", "write_patent", "read_zmx",
           "write_zmx", "read_seq", "write_seq", "find_glass"]


# media -----------------------------------------------------------------------------------------

class _Medium:
    """The medium behind a surface as a file states it.

    kind: "air", "const" (index n), "model" (n_d, V_d), "glass" (a
    `glasses.Glass`), "mirror".
    """

    def __init__(self, kind, n=1.0, vd=0.0, glass=None):
        self.kind, self.n, self.vd, self.glass = kind, n, vd, glass

    def __eq__(self, other):
        return (self.kind, self.n, self.vd, self.glass) == \
            (other.kind, other.n, other.vd, other.glass)

    def __repr__(self):
        return "_Medium({!r}, {}, {}, {!r})".format(self.kind, self.n, self.vd, self.glass)

    @property
    def nd_vd(self):
        """(n_d, V_d) to write; V_d = 0 for media without dispersion."""

        if self.kind == "glass":
            return self.glass.n_d, self.glass.V_d
        if self.kind == "function":
            nd, nF, nC = (float(self.glass(w)) for w in (LAMBDA_d, LAMBDA_F, LAMBDA_C))
            return nd, (nd - 1)/(nF - nC)
        return self.n, (self.vd if self.kind == "model" else 0.0)

    def surface_kwargs(self):
        if self.kind == "air":
            return {"n_after": 1.0}
        if self.kind == "const":
            return {"n_after": self.n}
        if self.kind == "model":
            return {"material": (self.n, self.vd)}
        if self.kind in ("glass", "function"):
            return {"material": self.glass}
        return {}


AIR = _Medium("air")
MIRROR = _Medium("mirror")


def _medium_from_index(n):
    return AIR if n == 1.0 else _Medium("const", float(n))


def _medium_from_spec(spec, offset=0.0, where=""):
    """_Medium of a Surface's material or n_after."""

    if isinstance(spec, _glasses.Glass):
        if offset:
            warnings.warn("{}: glass {} with an index offset is written as a model glass."
                          .format(where, spec.name))
            return _Medium("model", spec.n_d + offset, spec.V_d)
        return _Medium("glass", glass=spec)
    if isinstance(spec, tuple) and len(spec) == 2:
        return _Medium("model", float(spec[0]) + offset, float(spec[1]))
    if isinstance(spec, Material) or callable(spec):
        f = spec.n_geometric if isinstance(spec, Material) else spec
        nd, nF, nC = (float(np.real(f(w))) for w in (LAMBDA_d, LAMBDA_F, LAMBDA_C))
        warnings.warn("{}: material {!r} is written as a model glass n_d = {:.6f}, V_d = {:.3f}; "
                      "its dispersion is only approximated.".format(where, spec, nd,
                                                                   (nd - 1)/(nF - nC)))
        return _Medium("model", nd + offset, (nd - 1)/(nF - nC))

    return _medium_from_index(float(spec) + offset)


def _media(prescription):
    """The medium behind each surface (mirrors: MIRROR), as the files
    state it: explicit on every surface."""

    current = _medium_from_index(prescription.n_object)
    out = []
    for i, s in enumerate(prescription.surfaces):
        if s.mirror:
            out.append(MIRROR)
            continue
        if s.changes_medium:
            spec = s.material if s.material is not None else s.n_after
            current = _medium_from_spec(spec, getattr(s, "index_offset", 0.0),
                                        "surface {}".format(i + 1))
        out.append(current)

    return out


def _norm_name(name):
    return re.sub(r"[^A-Z0-9]", "", name.upper())


_glass_index = None


def find_glass(name, vendors=None):
    """A catalogue glass of `pyoptics.glasses` by name.

    Case, hyphens, spaces and underscores do not matter ("NBK7", "n-bk7"
    and "N-BK7" are the same glass); a vendor suffix ("N-BK7_SCHOTT", as in
    CODE V) restricts the search to that vendor.

    Parameters
    ----------
    name : str
    vendors : sequence of str, optional
        Vendors to search first, in this order (e.g. those of a Zemax
        GCAT line); the others follow.

    Returns
    -------
    glasses.Glass or None
        None if no catalogue has the glass.
    """

    global _glass_index
    if _glass_index is None:
        _glass_index = {}
        for v in _glasses.VENDORS:
            for k in _glasses.load(v):
                _glass_index.setdefault(_norm_name(k), []).append((v, k))
    order = [v.lower() for v in (vendors or ()) if v.lower() in _glasses.VENDORS]
    base = name
    m = re.match(r"^(.*?)[_ ]({})$".format("|".join(_glasses.VENDORS)), name, re.I)
    if m:
        base, order = m.group(1), [m.group(2).lower()]
    hits = _glass_index.get(_norm_name(base), [])
    if m:
        hits = [h for h in hits if h[0] in order]
    if not hits:
        return None
    hits = sorted(hits, key=lambda h: order.index(h[0]) if h[0] in order else len(order))
    v, k = hits[0]

    return _glasses.load(v)[k]


def _special_material(name):
    """Materials that are not in the glass catalogues: fused silica (by
    its usual names in lens files) as `raytrace.fused_silica`."""

    from .raytrace import fused_silica
    key = _norm_name(name)
    if key in ("FSILICA", "SILICA", "FUSEDSILICA", "C7980", "C7980UV", "C7980IR", "CORNING7980",
               "UVFS", "SIO2", "LITHOSIL", "SUPRASIL", "HPFS7980", "QUARTZFUSED"):
        return fused_silica
    if key == "B270":     # SCHOTT B 270 crown (condenser lenses, windows)
        return (1.5230, 58.5)
    return None


def _resolve_medium(name, nd, vd, use_catalog, vendors=None, where=""):
    """_Medium from a glass name and/or n_d, V_d read from a file."""

    if name:
        if name.upper() in ("MIRROR", "REFL"):
            return MIRROR
        special = _special_material(name) if use_catalog else None
        if callable(special):
            return _Medium("function", glass=special)
        if special is not None:
            warnings.warn("{}: {} as the model glass n_d = {}, V_d = {}.".format(where, name,
                                                                                *special))
            return _Medium("model", *special)
        if use_catalog:
            g = find_glass(name, vendors)
            if g is None and "_" in name:   # "L-TIM28_MOLD": the glass of a molded lens
                g = find_glass(name.split("_")[0], vendors)
            if g is None and not name.upper().startswith("N-"):
                # obsolete lead containing glasses: their lead free successors
                g = find_glass("N-" + name, vendors)
                if g is not None:
                    warnings.warn("{}: glass {!r} is not in the catalogues, using {} instead."
                                  .format(where, name, g.name))
                    return _Medium("glass", glass=g)
            if g is not None:
                if nd and abs(nd - g.n_d) > 5e-4 and (nd, vd) != (1.5, 40.0):
                    warnings.warn("{}: glass {} has n_d = {:.5f} in the catalogue, {:.5f} in the "
                                  "file.".format(where, name, g.n_d, nd))
                return _Medium("glass", glass=g)
            if not nd:
                raise KeyError("{}: no glass {!r} in the catalogues and no n_d given."
                               .format(where, name))
            warnings.warn("{}: glass {!r} not in the catalogues, using n_d = {}, V_d = {}."
                          .format(where, name, nd, vd))
        elif not nd:
            raise ValueError("{}: glass {!r} without n_d.".format(where, name))
    if not nd or nd == 1.0 and not vd:
        return AIR
    if not vd:
        return _Medium("const", float(nd))

    return _Medium("model", float(nd), float(vd))


def _apply_media(raw, n_object):
    """Surface kwargs from the media of the surfaces in a file: a medium
    is set on the surface where it changes."""

    current = _medium_from_index(n_object)
    out = []
    for m in raw:
        if m.kind == "mirror":
            out.append({"mirror": True})
            continue
        out.append({} if m == current else m.surface_kwargs())
        current = m

    return out


# the prescription ------------------------------------------------------------------------------

def _supported(s, fmt, types):
    if type(s) not in types:
        raise NotImplementedError("{} files: surface type {} is not supported (only {})."
                                  .format(fmt, type(s).__name__,
                                          ", ".join(t.__name__ for t in types)))


class Prescription:
    """A lens as a lens file holds it.

    Parameters
    ----------
    surfaces : sequence of raytrace.Surface
        As for `raytrace.System`; the thickness of the last surface is the
        distance to the image plane.
    stop_index : int
    wavelengths : sequence of number, optional
        In the length unit (default: the d line).
    primary : int, optional
        Index of the primary wavelength.
    object_distance : number, optional
    fields : sequence, optional
        Fields (x, y) or y: angles in radians (object at infinity) or object
        heights.
    epd : number, optional
        Entrance pupil diameter; None: the semi-diameter of the stop
        surface sets the aperture.
    n_object : number, optional
    units : str, optional
    name : str, optional
    refocus : bool, optional
        Put the image plane into the paraxial focus in `system()` (files
        that do not give the image distance).
    """

    def __init__(self, surfaces, stop_index, wavelengths=None, primary=0,
                 object_distance=np.inf, fields=(0.0,), epd=None, n_object=1.0, units="mm",
                 name="", refocus=False):
        if units not in NM_PER_UNIT:
            raise ValueError("Unknown units {!r}.".format(units))
        self.surfaces = list(surfaces)
        self.stop_index = stop_index % len(self.surfaces)
        if wavelengths is None:
            wavelengths = (LAMBDA_d/NM_PER_UNIT[units],)
        self.wavelengths = tuple(float(w) for w in np.atleast_1d(wavelengths))
        self.primary = primary
        self.object_distance = object_distance
        self.fields = [_field_xy(f) for f in fields] or [(0.0, 0.0)]
        self.epd = epd
        self.n_object = n_object
        self.units = units
        self.name = name
        self.refocus = refocus

    def __repr__(self):
        return ("Prescription({!r}, {} surfaces, stop {}, wavelengths {} {}, fields {}, epd {})"
                .format(self.name, len(self.surfaces), self.stop_index, self.wavelengths,
                        self.units, self.fields, self.epd))

    @property
    def infinite_object(self):
        return np.isinf(self.object_distance)

    @property
    def max_field(self):
        """The field farthest from the axis, y alone if it has no x."""

        x, y = max(self.fields, key=lambda f: math.hypot(*f))
        return y if x == 0 else (x, y)

    def system(self, wavelength=None, **kwargs):
        """The `raytrace.System`.

        Parameters
        ----------
        wavelength : number, optional
            Default: the primary wavelength.
        **kwargs
            Passed on to `raytrace.System` (e.g. ray_aiming).

        Returns
        -------
        raytrace.System
        """

        wl = self.wavelengths[self.primary] if wavelength is None else wavelength
        kw = dict(object_distance=self.object_distance, n_object=self.n_object,
                  units=self.units,
                  entrance_pupil_radius=None if self.epd is None else self.epd/2)
        kw["field_angle" if self.infinite_object else "object_height"] = self.max_field
        kw.update(kwargs)
        system = System(self.surfaces, self.stop_index, wl, **kw)

        return system.at_paraxial_focus() if self.refocus else system

    def systems(self, **kwargs):
        """The systems at all wavelengths."""

        return [self.system(w, **kwargs) for w in self.wavelengths]

    @classmethod
    def from_system(cls, system, wavelengths=None, primary=None, fields=None, name=""):
        """The prescription of a `raytrace.System`.

        Parameters
        ----------
        system : raytrace.System
        wavelengths : sequence, optional
            Wavelengths to list in the file (default: that of the system).
        primary : int, optional
            Index of the primary wavelength (default: the one nearest to
            the wavelength of the system).
        fields : sequence, optional
            Default: the axis and the maximum field of the system.
        name : str, optional

        Returns
        -------
        Prescription
        """

        if wavelengths is None:
            wavelengths = (system.wavelength,)
        wavelengths = tuple(np.atleast_1d(wavelengths))
        if primary is None:
            primary = int(np.argmin(np.abs(np.array(wavelengths) - system.wavelength)))
        if fields is None:
            f = _field_xy(system.max_field)
            fields = [(0.0, 0.0)] + ([f] if any(f) else [])
        if any(system.image_decenter) or any(system.image_tilt):
            warnings.warn("The decentre and tilt of the image plane are not written.")
        ep_r = system._pupils()[1]

        return cls([s for s in system.surfaces], system.stop_index, wavelengths, primary,
                   system.object_distance, fields, 2*ep_r if np.isfinite(ep_r) else None,
                   system.n_object, system.units, name)


def _as_prescription(lens, **kwargs):
    if isinstance(lens, Prescription):
        if kwargs:
            raise TypeError("Options {} only apply to a System.".format(sorted(kwargs)))
        return lens
    return Prescription.from_system(lens, **kwargs)


# text helpers ----------------------------------------------------------------------------------

def _source_text(source):
    """File contents (path or text; UTF-16 with BOM, UTF-8 or Latin-1)."""

    if isinstance(source, os.PathLike) or (isinstance(source, str) and "\n" not in source
                                           and os.path.isfile(source)):
        with open(source, "rb") as f:
            data = f.read()
        if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
            return data.decode("utf-16")
        if data[:3] == b"\xef\xbb\xbf":
            return data[3:].decode("utf-8")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")

    return source


def _output(text, path):
    if path is not None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return text


def _g(x):
    """Number for a file: shortest repr that reads back exactly."""

    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return "{:.1f}".format(x)
    return repr(x)


def _deg(a):
    """Degrees for a file, without the rounding noise of the conversion."""

    return _g(round(math.degrees(a), 12))


def _wl_nm(p, w):
    return w*NM_PER_UNIT[p.units]


def _number_list(text):
    return [float(t) for t in re.split(r"[\s,;]+", text.strip()) if t]


# patent tables ---------------------------------------------------------------------------------

_EMPTY = {"", "-", "—", "–", "--"}


def _num(tok):
    """A number in the notations of patents; None for an empty cell."""

    t = tok.strip().replace("−", "-").replace("–", "-").replace("+-", "-")
    if t in _EMPTY:
        return None
    u = t.upper().lstrip("+")
    if u in ("∞", "INF", "INFINITY", "INFINITE", "PLANE", "FLAT", "PLANO"):
        return np.inf
    if u in ("-∞", "-INF", "-INFINITY"):
        return -np.inf
    t = re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "e", t)

    return float(t)


def _normalize_line(line):
    line = line.replace("−", "-").replace(" ", " ").replace("\xa0", " ")
    # 1.23×10^-5, 1.23 x 10^(-5), 1.23×10-5 -> 1.23e-5
    line = re.sub(r"(?<=\d)\s*[×]\s*10\s*\^?\s*\(?\s*([+-]?\d+)\s*\)?", r"e\1", line)
    line = re.sub(r"(?<=\d)\s*[xX*]\s*10\s*\^\s*\(?\s*([+-]?\d+)\s*\)?", r"e\1", line)
    line = re.sub(r"(?<=\d)\s*[eE]\s*([+-])\s*(?=\d)", r"e\1", line)
    line = re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "e", line)                 # Fortran 1.0D-05

    return line


_COLUMNS = [
    ("label", r"(surf(ace)?\.?(no\.?)?|si|no\.?|#|s|i|surface number)"),
    ("R", r"(r|radius|radius of curvature|ri|rdy|curvature radius)"),
    ("d", r"(d|thickness|thick|t|di|distance|thi|spacing|d/t)"),
    ("nd", r"(nd|n_d|n|ndi|index|refractive index|ne)"),
    ("vd", r"(vd|v_d|νd|ν|v|vdi|νdi|abbe|abbe number|ve|νe)"),
    ("sd", r"(sd|semi[- ]?diameter|semi[- ]?aperture|ea/2|ed/2|ca/2|h|y|clear semi[- ]?aperture|"
           r"semi[- ]?dia\.?|half[- ]?aperture)"),
    ("diam", r"(ed|ea|ca|diameter|clear aperture|effective diameter|dia\.?)"),
    ("glass", r"(glass|material|glass name|medium|type)"),
    ("K", r"(k|conic|κ|cc|kappa)"),
]


def _column_key(word):
    w = word.strip().lower()
    for key, pat in _COLUMNS:
        if re.fullmatch(pat, w):
            return key
    return _asphere_key(word)


def _asphere_key(word):
    """Key of an aspheric coefficient column or key: "K", "eps" or the power
    of r (4, 6, ...)."""

    w = word.strip()
    if re.fullmatch(r"(?i)k|κ|cc|kappa|conic( constant)?", w):
        return "K"
    if re.fullmatch(r"(?i)ε|eps(ilon)?", w):
        return "eps"
    m = re.fullmatch(r"(?i)a_?0*(\d+)", w)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"[A-HJ]", w):
        return 4 + 2*"ABCDEFGHJ".index(w)
    if re.fullmatch(r"A[D-G]", w):
        return 4 + 2*"DEFG".index(w[1])
    return None


_LABEL = re.compile(r"(?i)^(obj(ect)?|img|ima(ge)?|sto(p)?|s?\d+\s*\*?\s*(\(?\s*(sto|stop|s)\s*\)?)?"
                    r"\s*\*?|(sto|stop)\s*\d*\s*\*?)$")


def _split_cells(line, csv):
    """Cells and their character spans."""

    if csv:
        cells, pos = [], 0
        for c in line.split(","):
            cells.append((c.strip(), (pos, pos + len(c))))
            pos += len(c) + 1
        return cells
    return [(m.group(0), m.span()) for m in re.finditer(r"\S+", line)]


def _header(line, csv, asph=False):
    """Columns of a table header line (the main table, or with asph the
    table of aspheric coefficients), or None."""

    cells = _split_cells(line, csv)
    if not cells:
        return None
    if asph:
        keys = [_column_key(cells[0][0]) if _column_key(cells[0][0]) == "label" else None] \
            + [_asphere_key(c) for c, _ in cells[1:]]
        ok = keys[0] == "label" and any(k == "K" or isinstance(k, int) for k in keys[1:])
    else:
        keys = [_column_key(c) for c, _ in cells]
        # "Surface No.", "Surface number": one column
        while len(keys) > 1 and keys[0] == "label" and (
                keys[1] == "label" or cells[1][0].lower() in ("number", "nr.", "nr")):
            (c0, (a0, _)), (_, (_, b1)) = cells[:2]
            cells[:2] = [(c0, (a0, b1))]
            keys[:2] = ["label"]
        ok = keys[0] == "label" and "R" in keys and "d" in keys
    if ok and all(k is not None for k in keys):
        return [(k, span) for k, (_, span) in zip(keys, cells)]
    return None


def _assign(cells, columns):
    """Cells of a row to the columns: in order if all are there, else by
    overlap with the header spans (blank cells of aligned tables)."""

    if len(cells) >= len(columns):
        return {k: c for (k, _), (c, _) in zip(columns, cells)}
    out, last = {}, -1
    for c, (a, b) in cells:
        best, score = None, None
        for j, (k, (ha, hb)) in enumerate(columns):
            if j <= last:
                continue
            ov = min(b, hb) - max(a, ha)
            s = (ov, -abs((a + b) - (ha + hb))) if ov > 0 else (-np.inf, -abs(b - hb))
            if score is None or s > score:
                best, score = j, s
        if best is None:
            raise ValueError("Cannot place the cell {!r}.".format(c))
        out[columns[best][0]] = c
        last = best
    return out


def read_patent(source, units=None, wavelengths_nm=None, epd=None, fields=None, stop=None,
                use_catalog=True, csv=None):
    """Read a lens prescription table as printed in patents.

    The main table has a header line naming its columns (surface, R, d,
    n_d, V_d and optionally the semi-diameter, the diameter, the glass and
    the conic constant; the usual names such as "r", "radius", "Nd", "νd",
    "ED/2" are recognized) and one row per surface: n_d and V_d of the
    medium behind the surface, blank for air. A row "OBJ" gives the object
    distance (and index), a row "IMG" is the image plane. "STO" in the
    surface column marks the stop, "*" an aspheric surface. A line with
    "aspher" starts the aspheric data: a table (surface, K, A4, A6, ... or
    A, B, C, ...) or lines "K = ..., A4 = ..." after a line naming the
    surface ("surface 3", "3rd surface", "3:"). Comment lines ``# key:
    value`` give the name, units, wavelengths [nm] (the first is the
    primary one unless "primary wavelength" says otherwise), entrance
    pupil diameter, field angles [deg] ("field angles", "x field angles")
    or "object heights", and the stop surface. A missing image distance
    (no thickness on the last surface) puts the image into the paraxial
    focus.

    Parameters
    ----------
    source : str or path
        File name or the text itself.
    units, wavelengths_nm, epd, fields, stop : optional
        Override (or supply) the metadata: fields in radians or object
        heights, stop as the surface number of the table.
    use_catalog : bool, optional
        Look glass names up in the catalogues (else n_d, V_d).
    csv : bool, optional
        Comma separated (default: detected from the header line).

    Returns
    -------
    Prescription
    """

    text = _source_text(source)
    meta = {}
    rows, asph = [], {}
    columns, asph_columns, in_asph, current = None, None, False, None
    for raw in text.splitlines():
        line = _normalize_line(raw).rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("//"):
            m = re.match(r"^[#/]+\s*([^:=]+?)\s*[:=]\s*(.*)$", stripped)
            if m:
                meta[m.group(1).strip().lower()] = m.group(2).strip()
            continue
        if csv is None and columns is None:
            use_csv = stripped.count(",") >= 2 and _header(stripped, True) is not None
        else:
            use_csv = bool(csv) if csv is not None else use_csv
        if re.search(r"(?i)aspher", stripped) and not _header(line, use_csv, True):
            in_asph, asph_columns, current = True, None, None
            rest = re.split(r"(?i)aspher\w*\s*(coefficients?|data|constants?)?\s*:?", stripped,
                            maxsplit=1)[-1]
            if "=" not in rest:
                continue
            line = stripped = rest
        if not use_csv and columns is not None and not in_asph:
            line = re.sub(r"(?<=\d),(?=\d)", ".", line)
        if not in_asph:
            head = _header(line, use_csv)
            if head is not None and columns is None:
                columns = head
                continue
            if columns is None:
                continue
            cells = _split_cells(line, use_csv)
            if not _LABEL.match(cells[0][0].strip()):
                if rows:
                    in_asph = True  # free text after the table: look for aspheric data
                else:
                    continue
            else:
                row = _assign(cells, columns)
                rows.append(row)
                continue
        # aspheric data
        head = _header(line, use_csv, True)
        if head is not None:
            asph_columns = head
            continue
        m = (re.search(r"(?i)(\d+)\s*(st|nd|rd|th)?\.?\s*surface", line)
             or re.search(r"(?i)surface\s*(no\.?\s*)?(\d+)", line)
             or re.match(r"^\s*s?(\d+)\s*\*?\s*(\(?sto(p)?\)?)?\s*[:)]", line, re.I)
             or (re.match(r"^\s*s?(\d+)\s*\*?\s+(?=[A-Za-zκε]+_?\d*\s*=)", line)
                 if "=" in line else None))
        if m:
            current = re.sub(r"\D", "", [g for g in m.groups() if g and re.search(r"\d", g)][0])
        if "=" in line:
            for key, value in re.findall(r"([A-Za-zκε]+_?\d*)\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)",
                                         line):
                k = _asphere_key(key)
                if k is not None and current is not None:
                    asph.setdefault(current, {})[k] = float(value)
            continue
        if asph_columns is not None:
            cells = _split_cells(line.replace(",", " ") if not use_csv else line, use_csv)
            if _LABEL.match(cells[0][0].strip()):
                row = _assign(cells, asph_columns)
                label = re.sub(r"\D", "", row.pop("label", "")) or row.get("label", "")
                asph.setdefault(label, {}).update({k: _num(v) for k, v in row.items()
                                                   if _num(v) is not None})
    if columns is None:
        raise ValueError("No table header (surface, R, d, ...) found.")
    if not rows:
        raise ValueError("The table has no rows.")

    # metadata
    def get(*words):
        for k, v in meta.items():
            if all(w in k for w in words):
                return v
        return None

    units = units or (get("unit") or "mm").strip().lower()
    if wavelengths_nm is None:
        w = next((v for k, v in meta.items() if "wavelength" in k and "primary" not in k
                  and "reference" not in k), None)
        wavelengths_nm = _number_list(w) if w else [LAMBDA_d]
    primary = 0
    pw = None
    for k, v in meta.items():
        if "primary" in k or ("reference" in k and "wavelength" in k):
            pw = v
    if pw:
        x = float(re.split(r"[\s,]+", pw)[0])
        primary = int(x) - 1 if x == int(x) and 1 <= x <= len(wavelengths_nm) \
            else int(np.argmin(np.abs(np.array(wavelengths_nm) - x)))
    if epd is None:
        e = get("entrance pupil") or get("epd")
        epd = float(_number_list(e)[0]) if e else None
    name = get("name") or get("title") or ""

    # rows -> surfaces
    obj = None
    if re.match(r"(?i)^obj|^0$", rows[0]["label"].strip()):
        obj = rows.pop(0)
    image = None
    if rows and re.match(r"(?i)^im", rows[-1]["label"].strip()):
        image = rows.pop()
    object_distance = np.inf
    n_object = 1.0
    if obj is not None:
        d = _num(obj.get("d", ""))
        object_distance = np.inf if d is None else abs(d)
        n = _num(obj.get("nd", ""))
        n_object = 1.0 if n is None else n
    if fields is None:
        fx = fy = None
        heights = False
        for k, v in meta.items():
            if "field" in k or "object height" in k:
                heights = heights or "height" in k
                if re.search(r"\bx\b", k):
                    fx = v
                else:
                    fy = v
        if fy or fx:
            y = _number_list(fy) if fy else [0.0]*len(_number_list(fx))
            x = _number_list(fx) if fx else [0.0]*len(y)
            conv = (lambda a: a) if heights else math.radians
            fields = [(conv(a), conv(b)) for a, b in zip(x, y)]
        else:
            fields = [(0.0, 0.0)]

    media, kwargs, stop_index, numbers = [], [], None, []
    refocus = False
    for i, row in enumerate(rows):
        label = row["label"].strip()
        where = "surface {}".format(label)
        numbers.append(re.sub(r"\D", "", label) or None)
        if re.search(r"(?i)sto", label):
            stop_index = i
        R = _num(row.get("R", "")) if row.get("R") is not None else np.inf
        d = _num(row["d"]) if row.get("d") is not None else None
        if d is None:
            if i == len(rows) - 1:
                refocus, d = True, 0.0
            else:
                d = 0.0
        nd_cell = row.get("nd", "") or ""
        glass = (row.get("glass") or "").strip() or None
        if nd_cell.strip().upper() in ("MIRROR", "REFL", "REFLECT", "REFLECTIVE"):
            glass = "MIRROR"
            nd = None
        else:
            nd = _num(nd_cell)
        vd = _num(row.get("vd", "") or "")
        media.append(_resolve_medium(glass, nd, vd, use_catalog, where=where))
        sd = _num(row["sd"]) if row.get("sd") else None
        if sd is None and row.get("diam"):
            dm = _num(row["diam"])
            sd = None if dm is None else dm/2
        kw = {"R": R if R is not None else np.inf, "thickness": d, "semi_diameter": sd}
        if row.get("K"):
            kw["conic"] = _num(row["K"]) or 0.0
        kwargs.append(kw)
    if not any(numbers):  # no numbers in the surface column: count the rows
        numbers = [str(i + 1) for i in range(len(rows))]
    for i, number in enumerate(numbers):
        a = (asph.get(number) if number else None) or asph.get(rows[i]["label"].strip())
        if not a:
            continue
        if "eps" in a:
            kwargs[i]["conic"] = a.pop("eps") - 1
        if "K" in a:
            kwargs[i]["conic"] = a.pop("K")
        powers = sorted(a)
        if any(p % 2 or p < 2 for p in powers):
            raise NotImplementedError("surface {}: odd aspheric terms {} are not supported."
                                      .format(number, [p for p in powers if p % 2]))
        if a.get(2):
            raise NotImplementedError("surface {}: an r**2 aspheric term is not supported."
                                      .format(number))
        a.pop(2, None)
        if a:
            coef = [0.0]*((max(a) - 2)//2)
            for p, v in a.items():
                coef[(p - 4)//2] = v
            kwargs[i]["aspheric"] = tuple(coef)
    if stop is not None:
        stop_index = numbers.index(str(stop))
    if stop_index is None:
        warnings.warn("No stop marked (STO); the first surface is the stop.")
        stop_index = 0
    surfaces = [Surface(**kw, **mk) for kw, mk in zip(kwargs, _apply_media(media, n_object))]

    return Prescription(surfaces, stop_index, [w/NM_PER_UNIT[units] for w in wavelengths_nm],
                        primary, object_distance, fields, epd, n_object, units, name, refocus)


def _check_centred(p, fmt, types):
    for i, s in enumerate(p.surfaces):
        _supported(s, fmt, types)
        if s.displaced and type(s) is not CoordinateBreak and fmt == "Patent table":
            raise NotImplementedError("Patent table: surface {} is decentred or tilted."
                                      .format(i + 1))


def write_patent(lens, path=None, csv=False, **kwargs):
    """Write a lens prescription table as printed in patents.

    One row per surface (surface, R, d, n_d, V_d, semi-diameter, glass),
    an "OBJ" and an "IMG" row, a table of conic constants and aspheric
    coefficients (K, A4, A6, ...) for the aspheric surfaces (marked "*"),
    and the metadata as comment lines. Radii, thicknesses and indices are
    written with all their digits, so that `read_patent()` gives back the
    same system. Only centred spheres, conics and even aspheres.

    Parameters
    ----------
    lens : raytrace.System or Prescription
    path : str, optional
        File to write.
    csv : bool, optional
        Comma separated instead of aligned columns.
    **kwargs
        For `Prescription.from_system()` (wavelengths, fields, name).

    Returns
    -------
    str
        The text.
    """

    p = _as_prescription(lens, **kwargs)
    _check_centred(p, "Patent table", (Surface,))
    media = _media(p)
    out = ["# pyoptics lens prescription (patent table)"]
    if p.name:
        out.append("# name: {}".format(p.name))
    out.append("# units: {}".format(p.units))
    out.append("# wavelengths [nm]: {}".format(" ".join(_g(_wl_nm(p, w)) for w in p.wavelengths)))
    out.append("# primary wavelength: {}".format(p.primary + 1))
    if p.epd is not None:
        out.append("# entrance pupil diameter: {}".format(_g(p.epd)))
    xs = [f[0] for f in p.fields]
    ys = [f[1] for f in p.fields]
    if p.infinite_object:
        out.append("# field angles y [deg]: {}".format(" ".join(_deg(v) for v in ys)))
        if any(xs):
            out.append("# field angles x [deg]: {}".format(" ".join(_deg(v)
                                                                    for v in xs)))
    else:
        out.append("# object heights: {}".format(" ".join(_g(v) for v in ys)))
        if any(xs):
            out.append("# object heights x: {}".format(" ".join(_g(v) for v in xs)))
    out.append("")

    header = ["Surf", "R", "d", "nd", "vd", "sd", "glass"]
    table = [["OBJ", "inf", "inf" if p.infinite_object else _g(p.object_distance),
              "" if p.n_object == 1 else _g(p.n_object), "", "", ""]]
    asph_rows = []
    for i, (s, m) in enumerate(zip(p.surfaces, media)):
        aspheric = bool(s.conic) or any(s.aspheric)
        label = "{}{}{}".format(i + 1, "*" if aspheric else "",
                                "(STO)" if i == p.stop_index else "")
        nd = vd = glass = ""
        if m.kind == "mirror":
            glass = "MIRROR"
        elif m.kind != "air":
            n, v = m.nd_vd
            nd = _g(n)
            vd = _g(v) if v else ""
            glass = m.glass.name.replace(" ", "_") if m.kind == "glass" else ""
        table.append([label, "inf" if np.isinf(s.R) else _g(s.R), _g(s.thickness), nd, vd,
                      "" if s.semi_diameter is None else _g(s.semi_diameter), glass])
        if aspheric:
            asph_rows.append((str(i + 1), s.conic, s.aspheric))
    table.append(["IMG", "inf", "", "", "", "", ""])

    if csv:
        out.append(", ".join(header))
        out.extend(", ".join(r) for r in table)
    else:
        widths = [max(len(r[j]) for r in table + [header]) for j in range(len(header))]
        fmt = lambda r: "  ".join(c.rjust(w) for c, w in zip(r, widths)).rstrip()
        out.append(fmt(header))
        out.extend(fmt(r) for r in table)
    if asph_rows:
        n = max(len(a) for _, _, a in asph_rows)
        aheader = ["Surf", "K"] + ["A{}".format(2*j + 4) for j in range(n)]
        arows = [[lab, _g(K)] + [_g(a[j]) if j < len(a) else "0.0" for j in range(n)]
                 for lab, K, a in asph_rows]
        out += ["", "Aspheric coefficients"]
        if csv:
            out.append(", ".join(aheader))
            out.extend(", ".join(r) for r in arows)
        else:
            widths = [max(len(r[j]) for r in arows + [aheader]) for j in range(len(aheader))]
            fmt = lambda r: "  ".join(c.rjust(w) for c, w in zip(r, widths)).rstrip()
            out.append(fmt(aheader))
            out.extend(fmt(r) for r in arows)

    return _output("\n".join(out) + "\n", path)


# Zemax -----------------------------------------------------------------------------------------

_ZMX_UNITS = {"MM": "mm", "CM": "cm", "METER": "m", "METERS": "m", "M": "m"}


def _zmx_glas(tokens):
    """Name, n_d, V_d of a GLAS line."""

    name = tokens[1] if len(tokens) > 1 else ""
    nums = []
    for t in tokens[2:6]:
        try:
            nums.append(float(t))
        except ValueError:
            nums.append(0.0)
    nd = nums[2] if len(nums) > 2 else 0.0
    vd = nums[3] if len(nums) > 3 else 0.0
    if name.upper() == "___BLANK":
        name = ""

    return name, nd, vd


def read_zmx(source, use_catalog=True, aperture_margin=1.0):
    """Read a Zemax OpticStudio sequential lens file (.zmx).

    Parameters
    ----------
    source : str or path
        File name or the text itself.
    use_catalog : bool, optional
        Look glass names up in the catalogues of `pyoptics.glasses` (else
        their n_d, V_d in the file).
    aperture_margin : number or None, optional
        The semi-diameters (DIAM) of the surfaces times this factor are
        their apertures. Zemax computes automatic semi-diameters from the
        marginal rays of the fields; a margin slightly above 1 keeps those
        rays from being vignetted by rounding. In Zemax, semi-diameters do
        not vignette (only explicit surface apertures do); None keeps only
        the semi-diameter of the stop, so that no other surface clips rays
        (e.g. off-axis mirrors whose DIAM is not centred on the beam).

    Returns
    -------
    Prescription
    """

    text = _source_text(source)
    header, surfs, cur = {}, [], None
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        key = t[0].upper()
        if key == "SURF":
            cur = {"type": "STANDARD", "parm": {}, "curv": 0.0, "disz": 0.0, "glas": None,
                   "diam": None, "coni": 0.0, "stop": False, "label": None}
            surfs.append(cur)
            continue
        if not line[:1].isspace():
            cur = None
        if cur is None:
            header.setdefault(key, []).append(t[1:])
            continue
        if key == "TYPE":
            cur["type"] = t[1].upper()
        elif key == "CURV":
            cur["curv"] = float(t[1])
        elif key == "DISZ":
            cur["disz"] = np.inf if t[1].upper().startswith("INF") else float(t[1])
        elif key == "GLAS":
            cur["glas"] = t
        elif key == "DIAM":
            cur["diam"] = float(t[1])
        elif key == "CONI":
            cur["coni"] = float(t[1])
        elif key == "PARM":
            cur["parm"][int(t[1])] = float(t[2])
        elif key == "STOP":
            cur["stop"] = True
        elif key == "COMM" and len(t) > 1:
            cur["label"] = line.split(None, 1)[1].strip().strip('"') or None
    if len(surfs) < 3:
        raise ValueError("A .zmx file needs an object, a surface and an image.")

    def h(key, i=0, default=None):
        v = header.get(key)
        return v[0][i] if v and len(v[0]) > i else default

    unit = (h("UNIT") or "MM").upper()
    if unit not in _ZMX_UNITS:
        raise NotImplementedError("Lens unit {} is not supported.".format(unit))
    units = _ZMX_UNITS[unit]
    vendors = [v for line in header.get("GCAT", []) for v in line]
    ftyp = [int(float(x)) for x in header.get("FTYP", [[]])[0]]
    n_f = ftyp[2] if len(ftyp) > 2 else None
    n_w = ftyp[3] if len(ftyp) > 3 else None

    # wavelengths [um]
    if "WAVM" in header:
        wl = sorted((int(r[0]), float(r[1])) for r in header["WAVM"])
        wl_um = [w for _, w in wl][:n_w] if n_w else [w for _, w in wl]
    elif "WAVN" in header or "WAVL" in header:
        vals = [float(x) for x in (header.get("WAVN") or header.get("WAVL"))[0]]
        wl_um = vals[:n_w] if n_w else [v for v in vals if v > 0]
    else:
        wl_um = [LAMBDA_d/1e3]
    primary = int(h("PWAV", 0, 1)) - 1

    # fields
    def flist(key):
        return [float(x) for x in header.get(key, [[]])[0]]
    xf, yf = flist("XFLN"), flist("YFLN")
    if n_f is None:
        w = flist("FWGN")
        n_f = max([i + 1 for i, v in enumerate(w) if v]) if any(w) else \
            max([1] + [i + 1 for i, v in enumerate(yf) if v] + [i + 1 for i, v in enumerate(xf) if v])
    xf = (xf + [0.0]*n_f)[:n_f]
    yf = (yf + [0.0]*n_f)[:n_f]
    field_type = ftyp[0] if ftyp else 0
    if field_type == 0:
        fields = [(math.radians(a), math.radians(b)) for a, b in zip(xf, yf)]
    elif field_type in (1, 2, 3):
        fields = list(zip(xf, yf))   # object heights, or image heights converted below
    else:
        raise NotImplementedError("Field type {} (only angles 0, object heights 1 and "
                                  "paraxial or real image heights 2, 3).".format(field_type))

    f_number = None
    if "ENPD" in header:
        epd = float(h("ENPD"))
    elif "FLOA" in header:
        epd = None
    elif "FNUM" in header:
        epd, f_number = None, float(h("FNUM"))
    else:
        raise NotImplementedError("Only the system apertures ENPD, FNUM and FLOA are supported.")

    obj, image, body = surfs[0], surfs[-1], surfs[1:-1]
    object_distance = obj["disz"]
    if object_distance >= 1e10:
        object_distance = np.inf
    n_object = 1.0
    if obj["glas"]:
        m = _resolve_medium(*_zmx_glas(obj["glas"]), use_catalog, vendors, "object")
        n_object = m.nd_vd[0] if m.kind in ("const", "model", "glass") else 1.0
    if image["curv"]:
        warnings.warn("The curved image surface is read as a plane.")

    items = []  # (surface, medium, is_stop, cb_order)
    for i, s in enumerate(body):
        where = "surface {}".format(i + 1)
        medium = (_resolve_medium(*_zmx_glas(s["glas"]), use_catalog, vendors, where)
                  if s["glas"] else AIR)
        P = s["parm"]
        R = np.inf if s["curv"] == 0 else 1/s["curv"]
        sd = None if s["diam"] is None or (aperture_margin is None and not s["stop"]) \
            else s["diam"]*(1.0 if s["stop"] else aperture_margin)
        kw = dict(thickness=s["disz"], semi_diameter=sd, label=s["label"])
        typ = s["type"]
        order = None
        if typ in ("STANDARD", "EVENASPH"):
            asp = ()
            if typ == "EVENASPH":
                if P.get(1):
                    raise NotImplementedError("{}: an r**2 term (PARM 1) is not supported."
                                              .format(where))
                asp = [P.get(j, 0.0) for j in range(2, 9)]
                while asp and asp[-1] == 0:
                    asp.pop()
            surf = Surface(R, conic=s["coni"], aspheric=asp, **kw)
        elif typ == "TOROIDAL":
            if P.get(2):
                raise NotImplementedError("{}: a y**2 term (PARM 2) is not supported."
                                          .format(where))
            asp = [P.get(j, 0.0) for j in range(3, 9)]
            while asp and asp[-1] == 0:
                asp.pop()
            surf = ToroidalSurface(R, R_x=np.inf if not P.get(1) else P[1], conic=s["coni"],
                                   aspheric=asp, **kw)
        elif typ == "BICONIC":
            surf = BiconicSurface(R, R_x=np.inf if not P.get(1) else P[1],
                                  conic_x=P.get(2, 0.0), conic=s["coni"], **kw)
        elif typ == "COORDBRK":
            order = int(P.get(6, 0))
            surf = CoordinateBreak((P.get(1, 0.0), P.get(2, 0.0)),
                                   tuple(math.radians(P.get(j, 0.0)) for j in (3, 4, 5)),
                                   s["disz"], s["label"])
        elif typ == "PARAXIAL":
            surf = IdealLens(P.get(1, np.inf), **kw)
        else:
            raise NotImplementedError("{}: surface type {} is not supported.".format(where, typ))
        if medium.kind == "mirror" and typ in ("COORDBRK", "PARAXIAL"):
            raise ValueError("{}: a {} cannot be a mirror.".format(where, typ))
        items.append([surf, medium, s["stop"], order])

    items = _collapse_breaks(items)
    surfaces, media, stop_index = [], [], None
    for surf, medium, is_stop, order in items:
        if is_stop:
            stop_index = len(surfaces)
        if order == 1:
            chain = _reverse_break(surf)
            surfaces += chain
            media += [medium]*len(chain)
        else:
            surfaces.append(surf)
            media.append(medium)
    if stop_index is None:
        raise ValueError("No stop surface (STOP).")
    _set_media(surfaces, media, n_object)
    name = " ".join(header["NAME"][0]) if header.get("NAME") and header["NAME"][0] else ""

    p = Prescription(surfaces, stop_index, [w*1e3/NM_PER_UNIT[units] for w in wl_um], primary,
                     object_distance, fields, epd, n_object, units, name)
    if f_number is not None:   # image space F/# (object at infinity): EPD = f'/F#
        if not p.infinite_object:
            raise NotImplementedError("FNUM needs an object at infinity.")
        if p.surfaces[p.stop_index].semi_diameter is None:
            p.surfaces[p.stop_index].semi_diameter = 1.0
        p.epd = abs(p.system().paraxial().efl)/f_number
    if field_type in (2, 3):
        p.fields = _fields_from_image_heights(p, fields, real=field_type == 3)

    return p


def _fields_from_image_heights(p, heights, real):
    """Fields (angles or object heights) whose chief rays reach the given
    paraxial or real image heights (Zemax field types 2 and 3)."""

    out = []
    probe = 1e-3 if p.infinite_object else 1e-3*max(1.0, abs(p.object_distance))
    for hx, hy in heights:
        h = math.hypot(hx, hy)
        if h == 0:
            out.append((0.0, 0.0))
            continue
        p.fields = [(0.0, probe)]
        system = p.system()
        k = system.paraxial(probe).image_height/(math.tan(probe) if p.infinite_object
                                                    else probe)   # per tan(angle) or height
        f = h/k
        if p.infinite_object:
            f = math.atan(f)       # paraxial: image height proportional to tan(angle)
        if real:
            def height(field):
                bundle = system.trace_pupil(field, 0.0, 0.0)
                return system._image_hits(bundle)[0][0][1]
            f0, h0 = f, height(f)
            f1 = f*h/h0 if h0 else f*1.01
            for _ in range(30):
                h1 = height(f1)
                if not np.isfinite(h1) or abs(h1 - h) < 1e-12*max(1.0, h):
                    break
                f0, h0, f1 = f1, h1, f1 + (h - h1)*(f1 - f0)/(h1 - h0)
            if not np.isfinite(height(f1)):
                raise ValueError("No real chief ray reaches the image height {}.".format(h))
            f = f1
        out.append((f*hx/h, f*hy/h))

    return out


def _set_media(surfaces, media, n_object):
    for s, kw in zip(surfaces, _apply_media(media, n_object)):
        if kw.get("mirror"):
            s.mirror = True
        for k, v in kw.items():
            if k != "mirror":
                setattr(s, k, v)


def _reverse_break(cb):
    """A coordinate break with tilts first (z, y, x) and then the
    decentre (Zemax order flag 1) as a chain of `CoordinateBreak`s."""

    chain = []
    for j in (2, 1, 0):
        if cb.tilt[j]:
            t = [0.0, 0.0, 0.0]
            t[j] = cb.tilt[j]
            chain.append(CoordinateBreak(tilt=tuple(t), label=cb.label))
    if any(cb.decenter) or not chain:
        chain.append(CoordinateBreak(decenter=cb.decenter, label=cb.label))
    chain[-1].thickness = cb.thickness

    return chain


def _collapse_breaks(items):
    """A coordinate break, a surface without thickness and the coordinate
    break that undoes the first one (order flag 1, written for a displaced
    surface): one decentred and tilted surface."""

    out, i = [], 0
    while i < len(items):
        if i + 2 < len(items):
            (a, ma, sa, oa), (s, ms, ss, os_), (b, mb, sb, ob) = items[i:i + 3]
            if (oa == 0 and ob == 1 and os_ is None and a.thickness == 0 and s.thickness == 0
                    and not s.displaced and not sa and not sb
                    and np.allclose(a.decenter, [-x for x in b.decenter], rtol=0, atol=1e-14)
                    and np.allclose(a.tilt, [-x for x in b.tilt], rtol=0, atol=1e-14)):
                s.decenter, s.tilt, s.thickness = a.decenter, a.tilt, b.thickness
                out.append([s, ms, ss, None])
                i += 3
                continue
        out.append(items[i])
        i += 1

    return out


def _zmx_glas_line(m, gcat):
    if m.kind == "air":
        return None
    if m.kind == "mirror":
        return "GLAS MIRROR 0 0 1.5 40 0 0 0 0 0 0"
    nd, vd = m.nd_vd
    if m.kind == "glass":
        gcat.append(m.glass.vendor.upper())
        return "GLAS {} 0 0 {} {} 0 0 0 0 0 0".format(m.glass.name.replace(" ", "_"), _g(nd),
                                                      _g(vd))
    return "GLAS ___BLANK 1 0 {} {} 0 0 0 0 0 0".format(_g(nd), _g(vd))


def write_zmx(lens, path=None, **kwargs):
    """Write a Zemax OpticStudio sequential lens file (.zmx, text).

    Surfaces and `IdealLens`es become STANDARD or EVENASPH (conic and
    aspheric terms) and PARAXIAL surfaces, `ToroidalSurface` TOROIDAL,
    `BiconicSurface` (without polynomial terms) BICONIC,
    `CoordinateBreak` COORDBRK. A decentred or tilted surface is
    framed by two coordinate breaks.

    Parameters
    ----------
    lens : raytrace.System or Prescription
    path : str, optional
        File to write (UTF-8).
    **kwargs
        For `Prescription.from_system()` (wavelengths, fields, name).

    Returns
    -------
    str
        The text.
    """

    p = _as_prescription(lens, **kwargs)
    zunit = {"mm": "MM", "cm": "CM", "m": "METER"}.get(p.units)
    if zunit is None:
        raise NotImplementedError("Zemax files in {} are not supported.".format(p.units))
    types = (Surface, IdealLens, ToroidalSurface, BiconicSurface, CoordinateBreak)
    media = _media(p)
    gcat = []
    body = []  # lines per Zemax surface
    before = _medium_from_index(p.n_object)

    def surface(lines, typ, curv, disz, medium, stop=False, sd=None, conic=0.0, parms=(),
                label=None):
        out = []
        if label:
            out.append('  COMM {}'.format(label))
        if stop:
            out.append("  STOP")
        out += ["  TYPE {}".format(typ), "  CURV {} 0 0 0 0 \"\"".format(_g(curv))]
        out += ["  PARM {} {}".format(j, _g(v)) for j, v in parms]
        out.append("  DISZ {}".format("INFINITY" if np.isinf(disz) else _g(disz)))
        g = _zmx_glas_line(medium, gcat)
        if g:
            out.append("  " + g)
        if sd is not None:
            out.append("  DIAM {} 1 0 0 1 \"\"".format(_g(sd)))
        if conic:
            out.append("  CONI {}".format(_g(conic)))
        lines.append(out)

    for i, (s, m) in enumerate(zip(p.surfaces, media)):
        _supported(s, "Zemax", types)
        stop = i == p.stop_index
        medium = before if m.kind == "mirror" else m
        if type(s) is CoordinateBreak:
            surface(body, "COORDBRK", 0.0, s.thickness, m,
                    parms=[(1, s.decenter[0]), (2, s.decenter[1])]
                    + [(j + 3, round(math.degrees(s.tilt[j]), 12)) for j in range(3)] + [(6, 0.0)],
                    stop=stop, label=s.label)
            before = m
            continue
        displaced = s.displaced
        if displaced:
            surface(body, "COORDBRK", 0.0, 0.0, before,
                    parms=[(1, s.decenter[0]), (2, s.decenter[1])]
                    + [(j + 3, round(math.degrees(s.tilt[j]), 12)) for j in range(3)] + [(6, 0.0)])
        thickness = 0.0 if displaced else s.thickness
        if type(s) is IdealLens:
            surface(body, "PARAXIAL", 0.0, thickness, m, stop, s.semi_diameter,
                    parms=[(1, s.f), (2, 0.0)], label=s.label)
        elif type(s) is ToroidalSurface:
            parms = [(1, 0.0 if np.isinf(s.R_x) else s.R_x), (2, 0.0)] \
                + [(j + 3, a) for j, a in enumerate(s.aspheric)]
            if len(s.aspheric) > 6:
                raise NotImplementedError("Zemax TOROIDAL: at most 6 aspheric terms.")
            surface(body, "TOROIDAL", s.c, thickness, m, stop, s.semi_diameter, s.conic, parms,
                    s.label)
        elif type(s) is BiconicSurface:
            if any(s.aspheric) or any(s.aspheric_x):
                raise NotImplementedError("Zemax BICONIC has no polynomial terms.")
            surface(body, "BICONIC", s.c, thickness, m, stop, s.semi_diameter, s.conic,
                    [(1, 0.0 if np.isinf(s.R_x) else s.R_x), (2, s.conic_x)], s.label)
        elif any(s.aspheric):
            if len(s.aspheric) > 7:
                raise NotImplementedError("Zemax EVENASPH: at most 7 aspheric terms (r**16).")
            surface(body, "EVENASPH", s.c, thickness, m, stop, s.semi_diameter, s.conic,
                    [(1, 0.0)] + [(j + 2, a) for j, a in enumerate(s.aspheric)], s.label)
        else:
            surface(body, "STANDARD", s.c, thickness, m, stop, s.semi_diameter, s.conic,
                    label=s.label)
        if displaced:
            surface(body, "COORDBRK", 0.0, s.thickness, medium,
                    parms=[(1, -s.decenter[0]), (2, -s.decenter[1])]
                    + [(j + 3, -round(math.degrees(s.tilt[j]), 12)) for j in range(3)] + [(6, 1.0)])
        before = medium

    nf, nw = len(p.fields), len(p.wavelengths)
    head = ["VERS 190000 0 0", "MODE SEQ", "NAME {}".format(p.name),
            "UNIT {} X W X CM MR CPMM".format(zunit)]
    head.append("ENPD {}".format(_g(p.epd)) if p.epd is not None else "FLOA")
    obj_glas = _zmx_glas_line(_medium_from_index(p.n_object), gcat)
    if gcat:
        head.append("GCAT {}".format(" ".join(dict.fromkeys(gcat))))
    head += ["FTYP {} 0 {} {} 0 0 0".format(0 if p.infinite_object else 1, nf, nw),
             "XFLN " + " ".join((_deg(f[0]) if p.infinite_object else _g(f[0])) for f in p.fields),
             "YFLN " + " ".join((_deg(f[1]) if p.infinite_object else _g(f[1])) for f in p.fields),
             "FWGN " + " ".join("1" for _ in p.fields)]
    head += ["WAVM {} {} 1".format(j + 1, _g(_wl_nm(p, w)/1e3)) for j, w in enumerate(p.wavelengths)]
    head.append("PWAV {}".format(p.primary + 1))
    lines = list(head)
    lines += ["SURF 0", "  TYPE STANDARD", "  CURV 0.0 0 0 0 0 \"\"",
              "  DISZ {}".format("INFINITY" if p.infinite_object else _g(p.object_distance))]
    if obj_glas:
        lines.append("  " + obj_glas)
    for j, b in enumerate(body):
        lines.append("SURF {}".format(j + 1))
        lines += b
    lines += ["SURF {}".format(len(body) + 1), "  TYPE STANDARD", "  CURV 0.0 0 0 0 0 \"\"",
              "  DISZ 0"]

    return _output("\n".join(lines) + "\n", path)


# CODE V ----------------------------------------------------------------------------------------

_SEQ_UNITS = {"M": "mm", "C": "cm", "MM": "mm", "CM": "cm"}
_SEQ_ASPH = "ABCDEFGHJ"


def _seq_statements(text):
    for raw in text.splitlines():
        line = raw.split("!", 1)[0]
        for stmt in line.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                tokens = shlex.split(stmt, posix=True)
            except ValueError:
                tokens = stmt.replace("'", " ").replace('"', " ").split()
            if tokens:
                yield [tokens[0].upper()] + tokens[1:]


def _seq_glass(token):
    """(name, n_d, V_d) of a CODE V glass token."""

    m = re.fullmatch(r"(\d{3,})\.(\d+)", token)
    if m:
        a, b = m.groups()
        return None, 1 + int(a)/10**len(a), float("0." + b)*100
    m = re.fullmatch(r"(\d\.\d+):(\d+(\.\d*)?)", token)
    if m:
        return None, float(m.group(1)), float(m.group(2))
    return token, 0.0, 0.0


def read_seq(source, use_catalog=True):
    """Read a CODE V sequence file (.seq).

    Parameters
    ----------
    source : str or path
        File name or the text itself.
    use_catalog : bool, optional
        Look glass names up in the catalogues of `pyoptics.glasses`;
        catalogue glasses without a match raise KeyError (CODE V names do
        not carry n_d and V_d).

    Returns
    -------
    Prescription
    """

    text = _source_text(source)
    units, radius_mode = "mm", True
    wl_nm, ref, epd, name = [], 1, None, ""
    xan = yan = xob = yob = None
    surfs, cur = [], None

    def new(kind, args):
        nums = []
        rest = []
        for a in args:
            try:
                nums.append(float(a))
            except ValueError:
                rest.append(a)
        v = nums[0] if nums else 0.0
        if radius_mode:
            R = np.inf if v == 0 else v
        else:
            R = np.inf if v == 0 else 1/v
        s = {"kind": kind, "R": R, "t": nums[1] if len(nums) > 1 else 0.0,
             "glass": rest[0] if rest else None, "stop": kind == "STO", "cir": None,
             "conic": 0.0, "asp": {}, "yto": False, "rdx": np.inf,
             "dec": [0.0]*6, "dar": False, "label": None}
        if len(nums) > 2 and s["glass"] is None:
            s["glass"] = args[2]
        surfs.append(s)
        return s

    for t in _seq_statements(text):
        cmd, args = t[0], t[1:]
        if cmd in ("SO", "S", "SI") or (cmd == "STO" and args):
            cur = new(cmd, args)
        elif cmd == "STO":
            if cur is None:
                raise ValueError("STO before the first surface.")
            cur["stop"] = True
        elif cmd in ("TIT", "TITLE"):
            name = " ".join(args)
        elif cmd == "DIM":
            u = args[0].upper()
            if u not in _SEQ_UNITS:
                raise NotImplementedError("Units {} are not supported.".format(u))
            units = _SEQ_UNITS[u]
        elif cmd == "RDM":
            radius_mode = not (args and args[0].upper().startswith("N"))
        elif cmd == "CUM":
            radius_mode = False
        elif cmd == "WL":
            wl_nm = [float(a) for a in args]
        elif cmd == "REF":
            ref = int(float(args[0]))
        elif cmd == "EPD":
            epd = float(args[0])
        elif cmd in ("XAN", "YAN", "XOB", "YOB"):
            vals = [float(a) for a in args]
            if cmd == "XAN":
                xan = vals
            elif cmd == "YAN":
                yan = vals
            elif cmd == "XOB":
                xob = vals
            else:
                yob = vals
        elif cur is not None and cmd == "CIR" and args:
            try:
                cur["cir"] = float(args[-1])
            except ValueError:
                pass
        elif cur is not None and cmd == "K":
            cur["conic"] = float(args[0])
        elif cur is not None and len(cmd) == 1 and cmd in _SEQ_ASPH and args:
            cur["asp"][4 + 2*_SEQ_ASPH.index(cmd)] = float(args[0])
        elif cur is not None and cmd == "YTO":
            cur["yto"] = True
        elif cur is not None and cmd == "XTO":
            raise NotImplementedError("X toroids (XTO) are not supported.")
        elif cur is not None and cmd == "RDX":
            v = float(args[0])
            cur["rdx"] = np.inf if v == 0 else v
        elif cur is not None and cmd == "CUX":
            v = float(args[0])
            cur["rdx"] = np.inf if v == 0 else 1/v
        elif cur is not None and cmd in ("RDY", "CUY"):
            v = float(args[0])
            cur["R"] = np.inf if v == 0 else (v if cmd == "RDY" else 1/v)
        elif cur is not None and cmd == "THI":
            cur["t"] = float(args[0])
        elif cur is not None and cmd == "GLA":
            cur["glass"] = args[0]
        elif cur is not None and cmd == "REFL":
            cur["glass"] = "REFL"
        elif cur is not None and cmd in ("XDE", "YDE", "ZDE", "ADE", "BDE", "CDE"):
            cur["dec"][("XDE", "YDE", "ZDE", "ADE", "BDE", "CDE").index(cmd)] = float(args[0])
        elif cur is not None and cmd == "DAR":
            cur["dar"] = True
        elif cur is not None and cmd in ("BEN", "REV"):
            raise NotImplementedError("Decentre type {} is not supported.".format(cmd))
        elif cur is not None and cmd == "SLB":
            cur["label"] = " ".join(args)
        elif cmd == "GO":
            break

    if not surfs or surfs[0]["kind"] != "SO":
        raise ValueError("No object surface (SO).")
    obj = surfs[0]
    image = surfs[-1] if surfs[-1]["kind"] == "SI" else None
    body = surfs[1:-1] if image else surfs[1:]
    if not body:
        raise ValueError("No surfaces.")
    object_distance = np.inf if abs(obj["t"]) >= 1e10 else obj["t"]
    n_object = 1.0
    if obj["glass"]:
        m = _resolve_medium(*_seq_glass(obj["glass"]), use_catalog, where="object")
        n_object = m.nd_vd[0] if m.kind != "air" else 1.0
    if image is not None and not np.isinf(image["R"]):
        warnings.warn("The curved image surface is read as a plane.")

    surfaces, media, stop_index = [], [], None
    eff = _medium_from_index(n_object)  # medium in front of the surface
    for i, s in enumerate(body):
        where = "surface {}".format(i + 1)
        medium = _resolve_medium(*_seq_glass(s["glass"]), use_catalog, where=where) \
            if s["glass"] else AIR
        dx, dy, dz, a, b, c = s["dec"]
        if dz:
            raise NotImplementedError("{}: ZDE is not supported.".format(where))
        tilt = tuple(math.radians(v) for v in (a, b, c))
        displaced = any((dx, dy, a, b, c))
        asp = s["asp"]
        coef = [asp.get(p, 0.0) for p in range(4, 4 + 2*len(_SEQ_ASPH), 2)]
        while coef and coef[-1] == 0:
            coef.pop()
        kw = dict(thickness=s["t"], semi_diameter=s["cir"], conic=s["conic"], aspheric=coef,
                  label=s["label"])
        if displaced and s["dar"]:
            kw.update(decenter=(dx, dy), tilt=tilt)
        if s["yto"]:
            surf = ToroidalSurface(s["R"], R_x=s["rdx"], **kw)
        else:
            surf = Surface(s["R"], **kw)
        if displaced and not s["dar"]:
            plain_dummy = (np.isinf(s["R"]) and not coef and not s["conic"] and s["cir"] is None
                           and not s["yto"] and not s["stop"] and medium == eff)
            if plain_dummy:  # a coordinate break as written by write_seq()
                surfaces.append(CoordinateBreak((dx, dy), tilt, s["t"], s["label"]))
                media.append(medium)
                continue
            surfaces.append(CoordinateBreak((dx, dy), tilt))
            media.append(eff)
        if s["stop"]:
            stop_index = len(surfaces)
        surfaces.append(surf)
        media.append(medium)
        if medium.kind != "mirror":
            eff = medium
    if stop_index is None:
        raise ValueError("No stop surface (STO).")
    _set_media(surfaces, media, n_object)
    if not wl_nm:
        wl_nm = [LAMBDA_d]
    if yob is not None or xob is not None:
        ys = yob if yob is not None else [0.0]*len(xob)
        xs = xob if xob is not None else [0.0]*len(ys)
        fields = list(zip(xs, ys))
    else:
        ys = yan if yan is not None else ([0.0]*len(xan) if xan else [0.0])
        xs = xan if xan is not None else [0.0]*len(ys)
        fields = [(math.radians(x), math.radians(y)) for x, y in zip(xs, ys)]

    return Prescription(surfaces, stop_index, [w/NM_PER_UNIT[units] for w in wl_nm], ref - 1,
                        object_distance, fields, epd, n_object, units, name)


def _seq_glass_token(m):
    if m.kind == "air":
        return ""
    if m.kind == "mirror":
        return "REFL"
    if m.kind == "glass":
        return "{}_{}".format(m.glass.name.replace(" ", "_"), m.glass.vendor.upper())
    nd, vd = m.nd_vd
    a = int(round((nd - 1)*1e6))
    if not 0 < a < 1e6 or abs(1 + a/1e6 - nd) > 1e-12 or abs(round(vd*1e4)/1e4 - vd) > 1e-12:
        warnings.warn("CODE V glass code {:06d}.{:06d}: n_d and V_d are rounded to 6 and 4 "
                      "decimals.".format(a, int(round(vd*1e4))))
    return "{:06d}.{:06d}".format(a, int(round(vd*1e4)))


def write_seq(lens, path=None, **kwargs):
    """Write a CODE V sequence file (.seq).

    Spheres, conics and even aspheres (K, A ... J: r**4 ... r**20),
    `ToroidalSurface` (YTO with RDX), mirrors (REFL), decentred and tilted
    surfaces (XDE ... CDE with DAR) and `CoordinateBreak`s (dummy surfaces
    with a basic decentre). Model glasses are glass codes ``nnnnnn.vvvvvv``
    (n_d to 6, V_d to 4 decimals).

    Parameters
    ----------
    lens : raytrace.System or Prescription
    path : str, optional
        File to write.
    **kwargs
        For `Prescription.from_system()` (wavelengths, fields, name).

    Returns
    -------
    str
        The text.
    """

    p = _as_prescription(lens, **kwargs)
    dim = {"mm": "M", "cm": "C"}.get(p.units)
    if dim is None:
        raise NotImplementedError("CODE V files in {} are not supported.".format(p.units))
    types = (Surface, ToroidalSurface, CoordinateBreak)
    media = _media(p)
    out = ["! pyoptics lens prescription (CODE V sequence)", "LEN NEW"]
    if p.name:
        out.append("TIT '{}'".format(p.name.replace("'", "")))
    out += ["DIM {}".format(dim), "RDM",
            "WL " + " ".join(_g(_wl_nm(p, w)) for w in p.wavelengths),
            "REF {}".format(p.primary + 1),
            "WTW " + " ".join("1" for _ in p.wavelengths)]
    if p.epd is not None:
        out.append("EPD {}".format(_g(p.epd)))
    xs = [f[0] for f in p.fields]
    ys = [f[1] for f in p.fields]
    if p.infinite_object:
        if any(xs):
            out.append("XAN " + " ".join(_deg(v) for v in xs))
        out.append("YAN " + " ".join(_deg(v) for v in ys))
    else:
        if any(xs):
            out.append("XOB " + " ".join(_g(v) for v in xs))
        out.append("YOB " + " ".join(_g(v) for v in ys))
    og = _seq_glass_token(_medium_from_index(p.n_object))
    out.append("SO 0.0 {} {}".format("1e10" if p.infinite_object else _g(p.object_distance),
                                     og).rstrip())
    before = _medium_from_index(p.n_object)
    for i, (s, m) in enumerate(zip(p.surfaces, media)):
        _supported(s, "CODE V", types)
        if type(s) is CoordinateBreak:
            if i == p.stop_index:
                raise NotImplementedError("CODE V: a coordinate break cannot be the stop.")
            out.append("S 0.0 {} {}".format(_g(s.thickness), _seq_glass_token(m)).rstrip())
            for cmd, v in zip(("XDE", "YDE", "ADE", "BDE", "CDE"),
                              s.decenter + tuple(round(math.degrees(a), 12) for a in s.tilt)):
                if v:
                    out.append("  {} {}".format(cmd, _g(v)))
            before = m
            continue
        R = 0.0 if np.isinf(s.R) else s.R
        out.append("S {} {} {}".format(_g(R), _g(s.thickness), _seq_glass_token(m)).rstrip())
        if s.label:
            out.append("  SLB '{}'".format(str(s.label).replace("'", "")))
        if i == p.stop_index:
            out.append("  STO")
        if s.semi_diameter is not None:
            out.append("  CIR {}".format(_g(s.semi_diameter)))
        if type(s) is ToroidalSurface:
            out.append("  YTO")
            out.append("  RDX {}".format(_g(0.0 if np.isinf(s.R_x) else s.R_x)))
        if any(s.aspheric):
            if len(s.aspheric) > len(_SEQ_ASPH):
                raise NotImplementedError("CODE V: at most 9 aspheric terms (r**20).")
            out.append("  ASP")
            out.append("  K {}".format(_g(s.conic)))
            out.append("  " + "; ".join("{} {}".format(_SEQ_ASPH[j], _g(a))
                                        for j, a in enumerate(s.aspheric)))
        elif s.conic:
            out.append("  CON")
            out.append("  K {}".format(_g(s.conic)))
        if s.displaced:
            for cmd, v in zip(("XDE", "YDE", "ADE", "BDE", "CDE"),
                              s.decenter + tuple(round(math.degrees(a), 12) for a in s.tilt)):
                if v:
                    out.append("  {} {}".format(cmd, _g(v)))
            out.append("  DAR")
        before = before if m.kind == "mirror" else m
    out += ["SI 0.0 0.0", "GO"]

    return _output("\n".join(out) + "\n", path)


# dispatch --------------------------------------------------------------------------------------

def _format(path, fmt):
    if fmt is not None:
        return fmt.lower()
    ext = os.path.splitext(str(path))[1].lower()
    return {".zmx": "zmx", ".seq": "seq"}.get(ext, "patent")


def read(path, format=None, **kwargs):
    """Read a lens file: .zmx (Zemax), .seq (CODE V), else a patent table.

    Parameters
    ----------
    path : str or path
    format : {"zmx", "seq", "patent"}, optional
        Default: from the extension.
    **kwargs
        For the reader.

    Returns
    -------
    Prescription
    """

    fmt = _format(path, format)
    reader = {"zmx": read_zmx, "seq": read_seq, "patent": read_patent}[fmt]

    return reader(os.fspath(path) if isinstance(path, os.PathLike) else path, **kwargs)


def write(lens, path, format=None, **kwargs):
    """Write a lens file: .zmx (Zemax), .seq (CODE V), else a patent table
    (comma separated for .csv).

    Parameters
    ----------
    lens : raytrace.System or Prescription
    path : str or path
    format : {"zmx", "seq", "patent"}, optional
        Default: from the extension.
    **kwargs
        For `Prescription.from_system()` (wavelengths, fields, name).

    Returns
    -------
    str
        The text written.
    """

    fmt = _format(path, format)
    if fmt == "patent":
        return write_patent(lens, path, csv=str(path).lower().endswith(".csv"), **kwargs)

    return {"zmx": write_zmx, "seq": write_seq}[fmt](lens, path, **kwargs)
