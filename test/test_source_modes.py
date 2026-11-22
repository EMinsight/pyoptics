import numpy as np
import pytest

from pyoptics import fieldtrace as ft
from pyoptics import source_modes as sm
from pyoptics import glasses
from pyoptics.ultrafast import fwhm, gaussian_broadening, group_index, gvd


WL = 1e-3


def rms_width(I, x):
    return np.sqrt(np.sum(I*x**2)/np.sum(I))


def test_hermite_gauss_unit_power_and_orthogonal():
    x = np.linspace(-4, 4, 256)
    f0 = sm.hermite_gauss(0, 0, 0.8, WL, x, x)
    f1 = sm.hermite_gauss(1, 0, 0.8, WL, x, x)
    assert f0.power() == pytest.approx(1.0)
    assert f0.overlap(f1) < 1e-12
    g = ft.gaussian(0.8, WL, x, x)
    assert f0.overlap(g) == pytest.approx(1.0, abs=1e-10)
    # one dimension, away from the waist: the same as fieldtrace.gaussian
    f = sm.hermite_gauss(0, 0, 0.3, WL, x, None, distance=200.0)
    assert f.overlap(ft.gaussian(0.3, WL, x, None, distance=200.0)) == pytest.approx(1.0, abs=1e-9)


def test_gsm_parameters_and_m2():
    w0, kappa = sm.gsm_parameters(0.5, np.inf)
    assert kappa == 0 and w0 == pytest.approx(1.0)        # coherent: 1/e**2 radius = 2 sigma_s
    w0, kappa = sm.gsm_parameters(0.5, 0.4)
    assert sm.gsm_m2(0.5, 0.4) == pytest.approx((1 + kappa)/(1 - kappa))


@pytest.mark.parametrize("z", [0.0, 500.0, 2000.0])
def test_gaussian_schell_spreads_as_analytic(z):
    ss, sg = 0.5, 0.4
    x = np.linspace(-8, 8, 1024)
    ens = sm.gaussian_schell(ss, sg, WL, x, fraction=0.9999)
    steps = [ft.Propagate(z, method="spw")] if z else []
    res = sm.trace(steps + [ft.Detector("d")], ens)
    xx, _ = res.grid("d")
    I = res.irradiance("d", xx)
    assert rms_width(I, xx) == pytest.approx(sm.gsm_width(ss, sg, z, WL), rel=2e-3)


def test_gaussian_schell_degree_of_coherence():
    ss, sg = 0.5, 0.4
    x = np.linspace(-6, 6, 512)
    res = sm.trace([ft.Detector("d")], sm.gaussian_schell(ss, sg, WL, x, fraction=0.99999))
    mu = res.degree_of_coherence("d", 0.0, x)
    for dx in (0.2, 0.4, 0.8):
        i = np.argmin(np.abs(x - dx))
        assert abs(mu[i]) == pytest.approx(np.exp(-x[i]**2/(2*sg**2)), abs=5e-3)
    res0 = res.degree_of_coherence("d", 0.0, np.array([0.0]))
    assert abs(res0[0]) == pytest.approx(1.0)


def test_multimode_laser_thermal_is_gaussian_schell():
    x = np.linspace(-6, 6, 400)
    W, M2 = 1.0, 3.0
    ens = sm.multimode_laser(W, M2, WL, x, fraction=0.99999)
    res = sm.trace([ft.Detector("d")], ens)
    I = res.irradiance("d", x)
    assert 2*rms_width(I, x) == pytest.approx(W, rel=1e-3)
    gsm = sm.gaussian_schell(W/2, W/np.sqrt(M2**2 - 1), WL, x)
    assert gsm.info["kappa"][0] == pytest.approx(ens.info["kappa"][0])


def test_multimode_coupling_into_fibre_is_fundamental_share():
    # a fibre mode matched to the embedded Gaussian takes exactly the TEM00 power
    x = np.linspace(-3, 3, 128)
    W, M2 = 1.0, 1.5
    ens = sm.multimode_laser(W, (M2, M2), WL, x, x, fraction=0.9999)
    w0 = ens.info["w0"][0]
    mode = ft.gaussian(w0, WL, x, x)
    res = sm.trace([ft.Detector("fibre", quantities=("coupling",), mode=mode)], ens)
    kappa = (M2 - 1)/(M2 + 1)
    assert res.coupling("fibre") == pytest.approx((1 - kappa)**2, rel=2e-3)


def test_explicit_modes_m2():
    x = np.linspace(-4, 4, 64)
    ens = sm.multimode_laser(0.5, None, WL, x, x, modes={(0, 0): 0.5, (1, 0): 0.5})
    assert ens.info["M2"] == pytest.approx((2.0, 1.0))
    assert len(ens) == 2 and np.allclose(ens.weights, 0.5)


def test_point_source_exact_spherical_wave():
    x = np.linspace(-1, 1, 101)
    f = sm.point_source((0.1, -0.05), 20.0, WL, x, x)
    Ex, _ = f.E()
    X, Y = np.meshgrid(x, x)
    r = np.sqrt((X - 0.1)**2 + (Y + 0.05)**2 + 20.0**2)
    ref = np.exp(1j*2*np.pi/WL*r)/r
    ref = ref/np.sqrt(np.sum(np.abs(ref)**2)*(x[1] - x[0])**2)
    np.testing.assert_allclose(Ex, ref, atol=1e-8*np.abs(ref).max())


def test_van_cittert_zernike_strip():
    a, d = 0.5, 200.0
    x = np.linspace(-3, 3, 512)
    res = sm.trace([ft.Detector("d")], sm.lambertian_disk(a, d, WL, x, None, n_points=300, m=0))
    mu = res.degree_of_coherence("d", 0.0, x)
    for dx in (0.05, 0.1, 0.3):
        i = np.argmin(np.abs(x - dx))
        u = 2*np.pi/WL*a*x[i]/d
        assert abs(mu[i]) == pytest.approx(abs(np.sin(u)/u), abs=1e-4)


def test_lambertian_disk_2d_points():
    x = np.linspace(-1, 1, 16)
    ens = sm.lambertian_disk(0.1, 10.0, WL, x, x, n_points=20, rng=1)
    pos = ens.info["positions"]
    assert pos.shape == (20, 2) and np.all(np.hypot(*pos.T) <= 0.1)


def test_polychromatic_weights_and_chain_factory():
    wls, w = sm.gaussian_spectrum(WL, 50e-6, n=5)
    assert np.sum(w) == pytest.approx(1.0) and wls[2] == pytest.approx(WL)
    x = np.linspace(-2, 2, 64)
    ens = sm.polychromatic(lambda wl: ft.gaussian(0.5, wl, x), wls, w, power=2.0)
    seen = []

    def chain(wl):
        seen.append(wl)
        return [ft.Propagate(10.0), ft.Detector("d")]

    res = sm.trace(chain, ens)
    assert len(seen) == 5
    assert res.power("d") == pytest.approx(2.0, rel=1e-6)


def test_young_double_slit_visibility_falls_with_bandwidth():
    # broadband light: fringes far from the centre wash out
    x = np.linspace(-1, 1, 2048)
    slits = lambda u: ((np.abs(u - 0.2) < 0.02) | (np.abs(u + 0.2) < 0.02)).astype(float)
    xd = np.linspace(-16, 16, 6401)
    chain = [ft.Aperture(mask=slits), ft.Propagate(2000.0, method="fresnel_ft"),
             ft.Detector("screen", x=xd)]
    mono = sm.trace(chain, sm.coherent(ft.Field(np.ones_like(x), x, None, WL)))
    wls, w = sm.gaussian_spectrum(WL, 0.4*WL, n=21)     # coherence length 2.5 wavelengths
    broad = sm.trace(chain, sm.polychromatic(lambda wl: ft.Field(np.ones_like(x), x, None, wl),
                                             wls, w))

    def visibility(I, centre):
        sel = np.abs(xd - centre) < 2.6
        return (I[sel].max() - I[sel].min())/(I[sel].max() + I[sel].min())

    Im, Ib = mono.irradiance("screen", xd), broad.irradiance("screen", xd)
    assert visibility(Im, 0.0) > 0.99 and visibility(Ib, 0.0) > 0.8
    # 12 mm off axis the slits differ in path by 2.4 wavelengths
    assert visibility(Im, 12.0) > 0.99 and visibility(Ib, 12.0) < 0.25


def test_pulse_group_delay_and_dispersion_in_glass():
    bk7 = glasses.get("N-BK7")
    wl0, L = 800e-6, 10.0
    x = np.linspace(-1, 1, 32)
    pulse = sm.gaussian_pulse(lambda wl: ft.gaussian(0.3, wl, x, x), wl0, 20.0, n=24)
    chain = lambda wl: [ft.Interface(bk7.n(wl*1e6)), ft.Propagate(L, method="fresnel"),
                        ft.Interface(1.0), ft.Detector("d")]
    res = sm.trace(chain, pulse)
    assert res.group_delay("d") == pytest.approx(group_index(bk7.n, 800.0)*L/sm.C_MM_PER_FS,
                                                 rel=1e-4)
    t = np.linspace(-150, 150, 601)
    I = res.intensity("d", t, np.array([0.0]), np.array([0.0]))[:, 0]
    assert fwhm(t, I) == pytest.approx(gaussian_broadening(20.0, gvd(bk7.n, 800.0)*L), rel=5e-3)
    # the fluence (over one period of the time window) is the incoherent sum
    T = pulse.time_window()
    t = np.linspace(-T/2, T/2, 1000, endpoint=False)
    I = res.intensity("d", t, np.array([0.0]), np.array([0.0]))[:, 0]
    dt = t[1] - t[0]
    F = sum(w*np.abs(sm.field_at(rec.field, np.array([0.0]), np.array([0.0]))[0][0])**2
            for rec, w in zip(res.records("d"), pulse.weights))
    assert np.sum(I)*dt == pytest.approx(F, rel=1e-3)


def test_pulse_with_gdd_is_longer():
    x = np.linspace(-1, 1, 16)
    p0 = sm.gaussian_pulse(lambda wl: ft.gaussian(0.3, wl, x), 800e-6, 10.0, n=48)
    p1 = sm.gaussian_pulse(lambda wl: ft.gaussian(0.3, wl, x), 800e-6, 10.0, n=48, gdd=100.0)
    t = np.linspace(-80, 80, 801)
    r0, r1 = (sm.trace([ft.Detector("d")], p) for p in (p0, p1))
    d0 = fwhm(t, r0.intensity("d", t, np.array([0.0]))[:, 0])
    d1 = fwhm(t, r1.intensity("d", t, np.array([0.0]))[:, 0])
    assert d0 == pytest.approx(10.0, rel=1e-3)
    assert d1 == pytest.approx(gaussian_broadening(10.0, 100.0), rel=1e-3)


def test_ensemble_validation_and_truncation():
    x = np.linspace(-1, 1, 8)
    f = ft.gaussian(0.3, WL, x)
    with pytest.raises(ValueError):
        sm.Ensemble([])
    with pytest.raises(ValueError):
        sm.Ensemble([f], [-1.0])
    ens = sm.Ensemble([f, f, f], [0.7, 0.2, 0.1])
    assert len(ens.truncated(0.85)) == 2
    assert isinstance(sm.trace([ft.Detector("d")], f), sm.EnsembleResult)


def test_field_at_matches_E_and_is_not_aliased():
    x = np.linspace(-1, 1, 64)
    f = ft.Field(np.exp(-x**2), x, None, WL, alpha=(2000.0, 0.0), beta=(30.0, 0.0), gamma=1.0)
    Ex, _ = sm.field_at(f, x)
    np.testing.assert_allclose(Ex, f.E()[0], atol=1e-12)
    xm = 0.5*(x[1:] + x[:-1])
    Em, _ = sm.field_at(f, xm)
    ref = np.exp(-xm**2)*np.exp(1j*(2000*xm**2 + 30*xm + 1))
    np.testing.assert_allclose(Em, ref, atol=2e-3)
