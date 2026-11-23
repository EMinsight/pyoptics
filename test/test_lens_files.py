"""Tests of pyoptics.lens_files: patent tables, Zemax .zmx and CODE V .seq.

The files read here are written by the tests themselves (no sample files of
the vendors); round trips write a system, read it back and compare traces.
"""

import warnings

import numpy as np
import pytest

from pyoptics import glasses
from pyoptics import lens_files as lf
from pyoptics import raytrace as rt


WAVELENGTHS = (486.1327e-6, 587.5618e-6, 656.2725e-6)
WRITERS = {"patent": lf.write_patent, "zmx": lf.write_zmx, "seq": lf.write_seq}
READERS = {"patent": lf.read_patent, "zmx": lf.read_zmx, "seq": lf.read_seq}


def _same_trace(a, b, fields, atol=1e-9):
    """Trace the same rays through both systems and compare."""

    px, py = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    inside = px**2 + py**2 <= 1
    for f in fields:
        rays = a.rays(f, px[inside], py[inside])
        A, B = a.trace(rays), b.trace(rays)
        assert np.array_equal(A.valid, B.valid)
        assert A.valid.sum() > 10
        np.testing.assert_allclose(B.points[:, A.valid], A.points[:, A.valid], rtol=0, atol=atol)
        np.testing.assert_allclose(B.opl[..., A.valid], A.opl[..., A.valid], rtol=0, atol=atol)
    assert b.stop_index == a.stop_index or len(b.surfaces) != len(a.surfaces)
    assert b.surfaces[b.stop_index].semi_diameter == pytest.approx(
        a.surfaces[a.stop_index].semi_diameter, rel=1e-12)


def _round_trip(system, fmt, **kwargs):
    text = WRITERS[fmt](system, **kwargs)
    return READERS[fmt](text), text


def doublet():
    """Cemented catalogue glass doublet with an asphere, a stop and a model
    glass plate."""

    bk7, sf5 = glasses.get("N-BK7"), glasses.get("N-SF5")
    return rt.System([rt.Surface(61.07, 4.0, material=bk7, semi_diameter=12.5),
                      rt.Surface(-44.2, 2.5, material=sf5, semi_diameter=12.5),
                      rt.Surface(-128.2, 5.0, n_after=1.0, semi_diameter=12.5, conic=-0.5,
                                 aspheric=(1e-6, -2e-9, 3e-12)),
                      rt.Surface(np.inf, 3.0, semi_diameter=8.0),
                      rt.Surface(50.0, 3.0, material=(1.6, 40.0), semi_diameter=10.0),
                      rt.Surface(np.inf, 80.0, n_after=1.0, semi_diameter=10.0)],
                     3, WAVELENGTHS[1], field_angle=np.radians(3.0))


def mangin_cassegrain():
    """Mirrors (negative thicknesses behind them) and a Mangin mirror (a
    mirror inside glass)."""

    bk7 = glasses.get("N-BK7")
    return rt.System([rt.Surface(np.inf, 100.0, semi_diameter=25.0),
                      rt.Surface(-300.0, 5.0, material=bk7, semi_diameter=26.0),
                      rt.Surface(-320.0, -5.0, mirror=True, semi_diameter=26.0, conic=-0.2),
                      rt.Surface(-300.0, -80.0, n_after=1.0, semi_diameter=26.0),
                      rt.Surface(-120.0, 120.0, mirror=True, semi_diameter=9.0, conic=-3.0)],
                     0, WAVELENGTHS[1], field_angle=np.radians(0.3))


def finite_conjugates():
    """Finite object in a liquid, object heights, an index without
    dispersion, lengths in cm."""

    return rt.System([rt.Surface(-1.2, 0.4, n_after=1.7, semi_diameter=0.9),
                      rt.Surface(-1.0, 0.1, n_after=1.0, semi_diameter=0.9),
                      rt.Surface(np.inf, 0.1, semi_diameter=0.35),
                      rt.Surface(2.0, 0.3, material=(1.5168, 64.17), semi_diameter=0.6),
                      rt.Surface(-2.0, 3.0, n_after=1.0, semi_diameter=0.6)],
                     2, 5.875618e-5, object_distance=0.8, object_height=0.05, n_object=1.33,
                     units="cm")


@pytest.mark.parametrize("fmt", ["patent", "zmx", "seq"])
@pytest.mark.parametrize("make", [doublet, mangin_cassegrain, finite_conjugates])
def test_round_trip_centred(fmt, make):
    system = make()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p, text = _round_trip(system, fmt, wavelengths=(WAVELENGTHS if system.units == "mm" else
                                                        tuple(w/10 for w in WAVELENGTHS)),
                              name="test lens")
    assert p.name == "test lens"
    assert p.primary == 1
    assert p.units == system.units
    assert p.n_object == system.n_object
    back = p.system()
    assert back.wavelength == pytest.approx(system.wavelength, rel=1e-14)
    assert back.object_distance == system.object_distance
    m = system.max_field
    _same_trace(system, back, [0.0, 0.7*m, m], atol=1e-9*(10 if fmt == "seq" else 1))
    np.testing.assert_allclose(back.n, system.n, rtol=0, atol=1e-12)
    # the other wavelengths see the same dispersion
    for w in p.wavelengths:
        np.testing.assert_allclose(p.system(w).n, system.with_wavelength(w).n, rtol=0,
                                   atol=1e-12 if fmt != "seq" else 1e-9)
    assert back.paraxial().efl == pytest.approx(system.paraxial().efl, rel=1e-12)


def test_catalogue_glasses_by_name():
    p = lf.read_zmx(lf.write_zmx(doublet()))
    assert p.surfaces[0].material is glasses.get("N-BK7")
    seq = lf.write_seq(doublet())
    assert "N-BK7_SCHOTT" in seq and "600000.400000" in seq
    assert lf.read_seq(seq).surfaces[1].material is glasses.get("N-SF5")
    # without the catalogues: the Abbe model of n_d, V_d
    p = lf.read_zmx(lf.write_zmx(doublet()), use_catalog=False)
    assert p.surfaces[0].material == (1.5168, 64.17)
    assert lf.find_glass("nbk7_schott") is glasses.get("N-BK7")
    assert lf.find_glass("S-BSL 7", vendors=["OHARA"]).vendor.upper() == "OHARA"
    assert lf.find_glass("no such glass") is None


def tilted_system(biconic=True, ideal=True):
    """Toroid, biconic, ideal lens, a decentred and tilted lens and a fold
    mirror between coordinate breaks."""

    a = np.pi/4
    surfaces = [rt.Surface(np.inf, 10.0, semi_diameter=6.0),
                rt.ToroidalSurface(40.0, 4.0, R_x=60.0, n_after=1.5, conic=-0.3,
                                   aspheric=(1e-6,), semi_diameter=8.0),
                rt.Surface(-80.0, 10.0, n_after=1.0, semi_diameter=8.0,
                           decenter=(0.1, -0.2), tilt=(0.01, -0.02, 0.3)),
                rt.CoordinateBreak(tilt=(a, 0.0, 0.0)),
                rt.Surface(np.inf, 0.0, mirror=True, semi_diameter=12.0),
                rt.CoordinateBreak(tilt=(a, 0.0, 0.0), thickness=-20.0)]
    if biconic:
        surfaces += [rt.BiconicSurface(-50.0, -3.0, R_x=-70.0, conic_x=-0.5, conic=0.2,
                                       material=(1.6, 45.0), semi_diameter=8.0),
                     rt.Surface(np.inf, -10.0, n_after=1.0, semi_diameter=8.0)]
    if ideal:
        surfaces.append(rt.IdealLens(-60.0, -50.0, semi_diameter=10.0))
    else:
        surfaces[-1].thickness = -60.0
    return rt.System(surfaces, 0, WAVELENGTHS[1], field_angle=(np.radians(1.0), np.radians(2.0)))


def test_round_trip_zemax_tilted():
    system = tilted_system()
    p, text = _round_trip(system, "zmx")
    assert "TYPE TOROIDAL" in text and "TYPE BICONIC" in text and "TYPE PARAXIAL" in text
    assert text.count("TYPE COORDBRK") == 4      # two for the tilted lens, two for the fold
    back = p.system()
    # the coordinate breaks around the tilted lens collapse into a displaced surface again
    assert [type(s) for s in back.surfaces] == [type(s) for s in system.surfaces]
    assert back.surfaces[2].decenter == pytest.approx(system.surfaces[2].decenter, abs=1e-15)
    assert back.surfaces[2].tilt == pytest.approx(system.surfaces[2].tilt, abs=1e-14)
    assert p.fields[1] == pytest.approx((np.radians(1.0), np.radians(2.0)), abs=1e-15)
    _same_trace(system, back, [0.0, system.max_field])


def test_zemax_coordinate_break_order_flag():
    """Order flag 1 (tilts z, y, x, then the decentre) undoes order 0."""

    text = lf.write_zmx(tilted_system()).replace("PARM 6 1.0", "PARM 6 1")
    # swap the surface after the first coordinate break for a dummy: no collapse
    lines = text.splitlines()
    i = lines.index("SURF 4")
    j = lines.index("SURF 5")
    lines[i + 1:j] = ["  TYPE STANDARD", "  CURV 0.0", "  DISZ 0.5", "  DIAM 8.0"]
    p = lf.read_zmx("\n".join(lines))
    cb = [s for s in p.surfaces if isinstance(s, rt.CoordinateBreak)]
    # 1 (decenter, tilt) + 4 for the inverse (3 tilts, 1 decentre) + 2 for the fold
    assert len(cb) == 7
    frames = p.system().frames()[0]
    # the frame after the inverse chain is rotated back to the axis frame
    k = next(i for i, s in enumerate(p.surfaces) if isinstance(s, rt.CoordinateBreak)) + 6
    assert all(isinstance(s, rt.CoordinateBreak) for s in p.surfaces[k - 4:k])
    o, R = frames[k - 1]
    np.testing.assert_allclose(R, np.eye(3), atol=1e-14)


def test_round_trip_codev_tilted():
    system = tilted_system(biconic=False, ideal=False)
    p, text = _round_trip(system, "seq")
    assert "YTO" in text and "DAR" in text and "REFL" in text
    back = p.system()
    assert [type(s) for s in back.surfaces] == [type(s) for s in system.surfaces]
    _same_trace(system, back, [0.0, system.max_field])


def test_unsupported_surfaces():
    z = rt.System([rt.ZernikeSurface(np.inf, 5.0, zernike={5: 0.01}, semi_diameter=5.0)], 0,
                  550e-6)
    for writer in WRITERS.values():
        with pytest.raises(NotImplementedError):
            writer(z)
    with pytest.raises(NotImplementedError):
        lf.write_patent(tilted_system())
    with pytest.raises(NotImplementedError):
        lf.write_seq(tilted_system())


def test_other_materials_are_model_glasses():
    system = rt.System([rt.Surface(30.0, 4.0, material=rt.fused_silica, semi_diameter=5.0),
                        rt.Surface(-30.0, 50.0, n_after=1.0, semi_diameter=5.0)], 0,
                       WAVELENGTHS[1])
    with pytest.warns(UserWarning, match="dispersion is only approximated"):
        text = lf.write_zmx(system)
    back = lf.read_zmx(text).system()
    nd, vd = back.surfaces[0].material
    assert nd == pytest.approx(rt.fused_silica(587.5618), abs=1e-12)
    assert vd == pytest.approx(67.8, abs=0.3)


PATENT = """
Example 1   f = 100   Fno = 4   (no metadata block)

 Surface No.      r          d         Nd        νd      ED/2
   OBJ           ∞          ∞
    1          39.65       5.20     1.62041    60.3     14.0
    2        -186.4        1.92                         14.0
    3*        -64.1        1,50     1.62004    36.3     12.0
    4          38.10       4.00                         11.0
    STO          ∞         3.50                          9.8
    6         119.8        4.40     1.62041    60.3     12.0
    7*        -49.6       82.70                         12.0
   IMG           ∞

Aspheric coefficients
3rd surface
  K = -1.2345, A4 = 2.5×10^-6, A6 = -1.5 x 10^-9
  A8 = 3.0D-12
Surface 7: ε = 0.5  A4 = -1.0E-06
"""


def test_read_patent_table_variants():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = lf.read_patent(PATENT, epd=25.0, fields=[0.0, np.radians(15.0)],
                           wavelengths_nm=[587.5618])
    s = p.surfaces
    assert len(s) == 7 and p.stop_index == 4
    assert s[0].R == 39.65 and s[0].material == (1.62041, 60.3) and s[0].semi_diameter == 14.0
    assert s[1].R == -186.4 and s[1].n_after == 1.0 and s[1].material is None
    assert s[2].thickness == 1.5                       # decimal comma
    assert s[2].conic == -1.2345
    assert s[2].aspheric == pytest.approx((2.5e-6, -1.5e-9, 3.0e-12), rel=1e-15)
    assert s[4].R == np.inf and s[4].semi_diameter == 9.8
    assert s[6].conic == pytest.approx(-0.5) and s[6].aspheric == (-1.0e-6,)
    assert s[6].thickness == 82.7
    system = p.system()
    assert system.paraxial().f_number == pytest.approx(system.paraxial().efl/25.0)
    assert np.isfinite(system.rms_opd(np.radians(15.0)))


def test_read_patent_table_aspheric_table_and_csv():
    text = """# units: mm
# wavelengths [nm]: 546.07
# entrance pupil diameter: 10
# object heights: 0 2.5
Surf, R, d, nd, vd, sd, glass
OBJ, inf, 50, , , ,
1(STO), 20, 3, 1.5168, 64.17, 6, N-BK7
2*, -20, 40, , , 6,

Aspheric coefficients
Surf, K, A, B, C, D
2*, -1, 1e-5, 0, 0, 2e-14
"""
    p = lf.read_patent(text)
    assert p.object_distance == 50.0 and p.fields[1] == (0.0, 2.5) and p.epd == 10.0
    assert p.stop_index == 0
    assert p.surfaces[0].material is glasses.get("N-BK7")
    assert p.surfaces[1].conic == -1 and p.surfaces[1].aspheric == (1e-5, 0.0, 0.0, 2e-14)
    assert p.wavelengths == pytest.approx((546.07e-6,), rel=1e-15)


def test_patent_table_without_image_distance_refocuses():
    text = """R d nd vd
1 50 5 1.5168 64.17
2 -50
"""
    text = "Surf " + text
    with pytest.warns(UserWarning, match="No stop"):
        p = lf.read_patent(text, epd=10.0)
    system = p.system()
    assert system.surfaces[-1].thickness > 40.0
    assert system.surfaces[-1].thickness == pytest.approx(system.paraxial().image_distance)


def test_zemax_utf16_and_unknown_glass(tmp_path):
    text = """VERS 140000 0 0
MODE SEQ
NAME My singlet
NOTE 0 something
UNIT MM X W X CM MR CPMM
ENPD 20
GCAT SCHOTT
FTYP 0 0 1 1 0 0 0
YFLN 0 0 0 0 0 0 0 0 0 0 0 0
WAVM 1 0.55 1
PWAV 1
SURF 0
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ INFINITY
SURF 1
  STOP
  COMM front
  TYPE STANDARD
  CURV 0.02 0 0 0 0 ""
  HIDE 0 0 0 0 0 0 0 0 0 0
  DISZ 5
  GLAS MYGLASS 0 0 1.6 50 0 0 0 0 0 0
  DIAM 10 1 0 0 1 ""
SURF 2
  TYPE STANDARD
  CURV -0.01 0 0 0 0 ""
  DISZ 60
  DIAM 10 0 0 0 1 ""
SURF 3
  TYPE STANDARD
  CURV 0.0 0 0 0 0 ""
  DISZ 0
TOL TOFF 0 0 0 0 0 0 0
"""
    path = tmp_path / "singlet.zmx"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-16"))
    with pytest.warns(UserWarning, match="MYGLASS"):
        p = lf.read(path)
    assert p.name == "My singlet" and p.epd == 20.0 and p.wavelengths == (0.55e-3,)
    s = p.surfaces
    assert s[0].R == 50.0 and s[0].material == (1.6, 50.0) and s[0].label == "front"
    assert s[1].R == -100.0 and s[1].n_after == 1.0
    assert len(p.fields) == 1


def test_codev_hand_written():
    text = """! a singlet
RDM; LEN
TIT 'Singlet'
DIM M
WL 656.3 587.6 486.1
REF 2
EPD 20.0
YAN 0.0 5.0
SO 0.0 1e+10
STO 0.0 5.0
S 50.0 5.0 'NBK7_SCHOTT'
  CIR 11
S -100.0 90.0 ; K -2.0
SI 0 0
GO
"""
    p = lf.read_seq(text)
    assert p.name == "Singlet" and p.primary == 1 and p.stop_index == 0
    assert p.wavelengths == pytest.approx((656.3e-6, 587.6e-6, 486.1e-6), rel=1e-15)
    assert p.fields[1] == pytest.approx((0.0, np.radians(5.0)))
    s = p.surfaces
    assert s[1].material is glasses.get("N-BK7") and s[1].semi_diameter == 11.0
    assert s[2].conic == -2.0 and s[2].n_after == 1.0
    assert np.isinf(p.object_distance)


def test_read_write_by_extension(tmp_path):
    system = doublet()
    for name in ("lens.zmx", "lens.seq", "lens.txt", "lens.csv"):
        path = tmp_path / name
        lf.write(system, path, wavelengths=WAVELENGTHS)
        back = lf.read(path).system()
        _same_trace(system, back, [0.0, system.max_field])
    assert "," in (tmp_path / "lens.csv").read_text()


def test_patent_unnumbered_stop_row():
    """A stop row without a number (as in many phone lens patents) does not
    shift the aspheric data of the numbered surfaces."""

    text = """Surf r d nd vd
STO inf -0.2
1* 2.0 0.6 1.535 56.2
2* -7.0 3.0

Aspheric coefficients
Surf K A4
1 -1.5 0.04
2 0.0 0.12
"""
    p = lf.read_patent(text, epd=1.5)
    assert p.stop_index == 0 and p.surfaces[0].conic == 0.0 and p.surfaces[0].aspheric == ()
    assert p.surfaces[1].conic == -1.5 and p.surfaces[1].aspheric == (0.04,)
    assert p.surfaces[2].aspheric == (0.12,)


def _singlet_zmx(header_aperture="ENPD 20", ftyp="FTYP 0 0 2 1 0 0 0", yfln="YFLN 0 5",
                 glass="GLAS N-BK7 0 0 1.5168 64.17 0 0 0 0 0 0"):
    return """VERS 190000 0 0
MODE SEQ
UNIT MM X W X CM MR CPMM
{}
{}
XFLN 0 0
{}
WAVM 1 0.5875618 1
PWAV 1
SURF 0
  TYPE STANDARD
  CURV 0.0
  DISZ INFINITY
SURF 1
  STOP
  TYPE STANDARD
  CURV 0.02
  DISZ 5
  {}
  DIAM 12
SURF 2
  TYPE STANDARD
  CURV -0.005
  DISZ 80
  DIAM 30
SURF 3
  TYPE STANDARD
  CURV 0.0
  DISZ 0
""".format(header_aperture, ftyp, yfln, glass)


def test_zemax_fnum_and_image_height_fields():
    ref = lf.read_zmx(_singlet_zmx())
    efl = ref.system().paraxial().efl
    p = lf.read_zmx(_singlet_zmx("FNUM 4 0"))
    assert p.epd == pytest.approx(efl/4, rel=1e-12)
    assert p.system().paraxial().f_number == pytest.approx(4.0, rel=1e-9)
    # field type 2: paraxial image height 3 mm
    p = lf.read_zmx(_singlet_zmx(ftyp="FTYP 2 0 2 1 0 0 0", yfln="YFLN 0 3"))
    s = p.system()
    assert p.fields[0] == (0.0, 0.0)
    assert s.paraxial(p.fields[1][1]).image_height == pytest.approx(3.0, rel=1e-9)
    # field type 3: the real chief ray reaches 3 mm on the image plane
    p = lf.read_zmx(_singlet_zmx(ftyp="FTYP 3 0 2 1 0 0 0", yfln="YFLN 0 3"))
    s = p.system()
    chief = s.trace_pupil(p.fields[1][1], 0.0, 0.0)
    assert s._image_hits(chief)[0][0][1] == pytest.approx(3.0, abs=1e-9)


def test_zemax_glass_fallbacks():
    with pytest.warns(UserWarning, match="using N-BK7 instead"):
        p = lf.read_zmx(_singlet_zmx(glass="GLAS BK7"))
    assert p.surfaces[0].material is glasses.get("N-BK7")
    # placeholder indices of unresolved glasses do not override the catalogue
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = lf.read_zmx(_singlet_zmx(glass="GLAS SF11 0 0 1.5 4.0E+1 0 0 0 0 0 0"))
    assert p.surfaces[0].material is glasses.get("SF11")
    p = lf.read_zmx(_singlet_zmx(glass="GLAS F_SILICA 0 0 1.458464 67.82 0 0 0 0 0 0"))
    assert p.surfaces[0].material is rt.fused_silica
    p = lf.read_zmx(_singlet_zmx(glass="GLAS L-TIM28_MOLD 0 0 1.5 40 0 0 0 0 0 0"))
    assert p.surfaces[0].material.name == "L-TIM28"


def test_zemax_semi_diameters_do_not_have_to_clip():
    p = lf.read_zmx(_singlet_zmx(), aperture_margin=None)
    assert p.surfaces[0].semi_diameter == 12.0           # the stop
    assert p.surfaces[1].semi_diameter is None
    p = lf.read_zmx(_singlet_zmx(), aperture_margin=1.01)
    assert p.surfaces[1].semi_diameter == pytest.approx(30.3)
