"""Tests of pyoptics.eyepieces: first order data of the five eyepieces,
the eye relief and the afocal telescope (notebook 171, eyepieces)."""

import numpy as np
import pytest

from pyoptics import eyepieces as ep


@pytest.fixture(scope="module")
def designs():
    return ep.eyepieces(25.0)


def test_five_designs_scaled(designs):
    assert [e.name for e in designs] == ["Ramsden", "Huygens", "Kellner", "Plössl", "Erfle"]
    for e in designs:
        assert e.focal_length == pytest.approx(25.0, rel=1e-6)
    assert ep.erfle(20.0).focal_length == pytest.approx(20.0, rel=1e-6)


def test_eye_relief_order(designs):
    er = {e.name: e.eye_relief(np.inf) for e in designs}
    # the Huygens eyepiece has the shortest eye relief, the Plössl the longest
    assert er["Huygens"] < er["Ramsden"] < er["Kellner"] < er["Plössl"]
    assert 5.0 < er["Huygens"] < 10.0 and 20.0 < er["Plössl"] < 25.0


def test_eye_relief_is_exit_pupil_of_objective(designs):
    # the image of an objective far away lies near the back focal point
    # of the eyepiece on the eye side; closer objectives push it out
    e = designs[3]
    assert e.eye_relief(1e6) == pytest.approx(e.eye_relief(np.inf), abs=1e-3)
    assert e.eye_relief(250.0) > e.eye_relief(np.inf)
    # Newton's equation for the pupil image: the objective at the distance
    # d in front of the focal plane is imaged f^2/d behind the back focus
    for d in (150.0, 250.0, 600.0):
        assert e.eye_relief(d) - e.eye_relief(np.inf) == pytest.approx(25.0**2/d, rel=1e-6)


def test_huygens_has_virtual_field_stop(designs):
    huygens = designs[1]
    assert huygens.back_focus() < 0          # the image lies between the lenses
    assert designs[0].back_focus() > 0       # Ramsden: in front of the field lens


def test_telescope_magnification():
    tel = ep.telescope(ep.kellner(25.0), objective_focal_length=250.0)
    m = ep.angular_magnification(tel, np.radians(0.5))
    assert abs(m) == pytest.approx(10.0, rel=0.02)


def test_wide_field_needs_more_lenses(designs):
    # the Erfle covers its 65 degree field with less blur at the edge than
    # the Ramsden at the same field
    ramsden, erfle = designs[0], designs[4]
    theta = np.radians(30.0)
    b_r = ep.angular_blur(ramsden.system(field=theta), [theta])[0]
    b_e = ep.angular_blur(erfle.system(field=theta), [theta])[0]
    assert b_e < b_r
