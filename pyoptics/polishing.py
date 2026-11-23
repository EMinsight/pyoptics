#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Surface errors left by grinding and polishing, as maps for the ray trace.

An optical surface departs from its design on all lateral scales, and the
scales are usually split by their effect on the light (ISO 10110-5 and -8;
Aikens, Wolfe and Lawson):

- **Figure** (form, low spatial frequencies, a few cycles over the
  aperture): power, astigmatism and other low order Zernikes, a turned
  edge. It is a wavefront aberration and lowers the Strehl ratio and the
  core of the PSF.
- **Mid-spatial frequencies** (MSF, periods from about 0.1 mm to a few mm,
  roughly 10 to a few hundred cycles over the aperture): the footprint of
  sub-aperture tools (CNC pitch or bonnet polishing, magnetorheological
  finishing, diamond turning). A periodic ripple diffracts like a weak phase
  grating: satellites of the PSF at the angle wavelength/period, rings for
  a spiral tool path, and a loss in the MTF at mid frequencies, often while
  the figure looks excellent (Youngworth and Stone; Tamkin and Milster).
- **Roughness** (high spatial frequencies, periods below about 0.1 mm):
  scatter at large angles out of the image altogether. It is described by
  its RMS sigma and the total integrated scatter (TIS) and enters a
  sequential trace as a loss of the specular power (`specular_fraction()`).

The functions here build such errors on a grid, to be handed to
`raytrace.SurfaceMap` (sag maps for `raytrace.MapSurface`, energy maps for
``energy_map`` of any `raytrace.Surface`):

- `tool_path_ripple()`: the residual of a sub-aperture tool with a Gaussian
  removal profile (tool influence function, TIF) that moves along a raster
  or a spiral path with the pitch p. Preston's law (removal proportional to
  pressure, speed and dwell) makes the removal the sum of the TIF over the
  tracks. Its periodic part follows from the Poisson sum ::

      sum_k g(u - k p) = (D/p) [1 + 2 sum_m exp(-2 pi**2 m**2 s**2/p**2) cos(2 pi m u/p)]

  for a Gaussian track profile g of the area D (removal depth D/p on
  average) and the standard deviation s: the ripple falls off steeply when
  the tool is wider than the pitch (`ripple_amplitude()`). u is the
  distance across the tracks: x' (raster lines along y', turned by
  `angle`) or r - p phi/(2 pi) for an Archimedean spiral.
- `turned_down_edge()`: the roll-off of full-aperture or sub-aperture
  polishing at the rim, where the pressure under a tool that overhangs the
  edge rises (Kim, Park, Kim and Burge): a polynomial dip
  -depth ((r - r_start)/(r_edge - r_start))**power beyond r_start.
- `random_surface()`: a Gaussian random surface with an isotropic power
  spectral density (PSD), e.g. the fractal power law
  ``psd_power_law()`` A/f**B of polished glass (Church; B of about 1.5 to 2.5
  at mid frequencies), by filtering white noise in the Fourier domain.
- `radial_psd()`, `band_rms()`: the radially averaged 2D PSD of a map and
  the RMS in a band of spatial frequencies (Parseval).
- `remove_zernikes()`: the least squares fit of Fringe Zernikes over a
  disk, to split a map into figure and the rest.
- `specular_fraction()`: the specular power left behind a rough surface,
  exp(-(2 pi sigma delta/wavelength)**2) (the Debye-Waller factor of a
  Gaussian surface; delta = 2 n cos(theta) for a reflection,
  ``|n1 cos(theta1) - n2 cos(theta2)|`` for a refraction), i.e. 1 - TIS.
- `defect_map()`: a pixelated transmittance map of digs and scratches
  (MIL-PRF-13830B, ISO 10110-7) for an energy map.

Notebook 414 (polishing errors) puts them on a fast asphere and follows
them into the PSF and the MTF.

Conventions: maps are sampled as ``Z[i, j]`` at (x[j], y[i]) (the image
convention of `raytrace.SurfaceMap`), lengths in one unit (mm in
`raytrace`), spatial frequencies in cycles per length unit. The 2D PSD
is normalized such that its integral over the frequency plane is the
variance of the surface (units length**4).

References
----------
F. W. Preston, "The theory and design of plate glass polishing machines",
J. Soc. Glass Technol. 11, 214-256 (1927).

E. L. Church, "Fractal surface finish", Appl. Opt. 27, 1518-1526 (1988).

J. M. Bennett and L. Mattsson, "Introduction to Surface Roughness and
Scattering", 2nd ed., Optical Society of America (1999).

D. M. Aikens, C. R. Wolfe and J. K. Lawson, "Use of power spectral density
(PSD) functions in specifying optics for the National Ignition Facility",
Proc. SPIE 2576, 281-292 (1995).

R. N. Youngworth and B. D. Stone, "Simple estimates for the effects of
mid-spatial-frequency surface errors on image quality", Appl. Opt. 39,
2198-2209 (2000).

J. M. Tamkin and T. D. Milster, "Effects of structured mid-spatial
frequency surface errors on image performance", Appl. Opt. 49, 6522-6536
(2010).

D. W. Kim, W. H. Park, S.-W. Kim and J. H. Burge, "Parametric modeling of
edge effects for polishing tool influence functions", Opt. Express 17,
5656-5665 (2009).

D. D. Walker, D. Brooks, A. King, R. Freeman, R. Morton, G. McCavana and
S.-W. Kim, "The 'Precessions' tooling for polishing and figuring flat,
spherical and aspheric surfaces", Opt. Express 11, 958-964 (2003).

C. R. Dunn and D. D. Walker, "Pseudo-random tool paths for CNC
sub-aperture polishing and other applications", Opt. Express 16,
18942-18949 (2008).

G. W. Forbes, "Never-ending struggles with mid-spatial frequencies",
Proc. SPIE 9525, 95251B (2015).
"""

import numpy as np
from numpy import pi

from .bases import fringe_zernike


FWHM_TO_SIGMA = 1/(2*np.sqrt(2*np.log(2)))


def ripple_amplitude(pitch, tif_fwhm, removal, harmonic=1):
    """Amplitude of the harmonic m of the ripple that a Gaussian tool leaves
    on parallel tracks: 2 D exp(-2 pi**2 m**2 s**2/p**2).

    Parameters
    ----------
    pitch : number
        Track spacing p.
    tif_fwhm : number
        Full width at half maximum of the tool's removal profile across the
        track (s = FWHM/2.355).
    removal : number
        Mean removal depth D.
    harmonic : int, optional

    Returns
    -------
    amplitude : number
        Half the peak-to-valley of the harmonic (length units).
    """

    s = tif_fwhm*FWHM_TO_SIGMA

    return 2*removal*np.exp(-2*pi**2*harmonic**2*s**2/pitch**2)


def tool_path_ripple(X, Y, pitch, tif_fwhm, removal, path="raster", angle=0.0,
                     centre=(0.0, 0.0), n_harmonics=20):
    """Sag ripple left by a Gaussian tool on a raster or spiral path.

    Parameters
    ----------
    X, Y : arrays
        Points on the surface (local frame).
    pitch : number
        Track spacing (raster) or advance per turn (spiral).
    tif_fwhm : number
        FWHM of the removal profile across the track.
    removal : number
        Mean removal depth.
    path : {"raster", "spiral"}, optional
        Raster: straight tracks along y' (x turned by `angle`), ripple along
        x'. Spiral: an Archimedean spiral about `centre`; within about one
        pitch of the centre, where the tracks wind up, the ripple is tapered
        by 1 - exp(-(r/p)**2) (its slopes would diverge there).
    angle : number, optional
        Raster direction in radians (tracks along the direction angle + pi/2).
    centre : (x0, y0), optional
        Centre of the spiral.
    n_harmonics : int, optional
        Harmonics kept in the Poisson sum.

    Returns
    -------
    dz : array
        Sag departure (material removed is negative) with zero mean over a
        period: the mean removal is not included.
    """

    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if path == "raster":
        phase = 2*pi*(X*np.cos(angle) + Y*np.sin(angle))/pitch
        m_phi = np.zeros_like(phase)
    elif path == "spiral":
        Xc, Yc = X - centre[0], Y - centre[1]
        phase = 2*pi*np.hypot(Xc, Yc)/pitch
        m_phi = np.arctan2(Yc, Xc)
    else:
        raise ValueError("path must be 'raster' or 'spiral'.")
    dz = np.zeros(np.broadcast(phase, m_phi).shape)
    for m in range(1, n_harmonics + 1):
        a = ripple_amplitude(pitch, tif_fwhm, removal, m)
        if a < 1e-17*abs(removal):
            break
        dz = dz - a*np.cos(m*(phase - m_phi))
    if path == "spiral":
        # the tracks of a spiral wind up at its centre, where the azimuth
        # would make the slopes diverge: taper the ripple within a pitch
        dz = dz*(1 - np.exp(-(phase/(2*pi))**2))

    return dz


def turned_down_edge(X, Y, r_start, r_edge, depth, power=2.0, centre=(0.0, 0.0)):
    """Roll-off at the rim: -depth ((r - r_start)/(r_edge - r_start))**power
    for r > r_start, zero inside.

    Parameters
    ----------
    X, Y : arrays
    r_start : number
        Radius where the roll-off begins.
    r_edge : number
        Radius at which the dip reaches `depth` (usually the clear aperture
        or the physical rim; beyond it the polynomial continues).
    depth : number
        Depth of the dip at r_edge (> 0 turned down, < 0 turned up).
    power : number, optional
    centre : (x0, y0), optional

    Returns
    -------
    dz : array
    """

    r = np.hypot(np.asarray(X, dtype=float) - centre[0], np.asarray(Y, dtype=float) - centre[1])
    u = np.clip((r - r_start)/(r_edge - r_start), 0.0, None)

    return -depth*u**power


def psd_power_law(f, A, B, f_min=0.0, f_max=np.inf):
    """Isotropic 2D PSD A/f**B between f_min and f_max (zero outside).

    Parameters
    ----------
    f : array
        Radial spatial frequency (cycles per length unit).
    A : number
        Amplitude (length**(4 - B)).
    B : number
        Exponent (B > 2: the variance converges at high frequencies).
    f_min, f_max : numbers, optional
        Band.

    Returns
    -------
    psd : array
    """

    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore"):
        psd = A/f**B
    band = (f >= f_min) & (f <= f_max) & (f > 0)

    return np.where(band, psd, 0.0)


def _frequencies(x, y):
    dx, dy = x[1] - x[0], y[1] - y[0]
    FX, FY = np.meshgrid(np.fft.fftfreq(len(x), dx), np.fft.fftfreq(len(y), dy))

    return FX, FY, dx, dy


def random_surface(x, y, psd, rms=None, seed=None):
    """Gaussian random surface with an isotropic 2D PSD on an equidistant
    grid (periodic over the window).

    Parameters
    ----------
    x, y : 1d arrays
        Equidistant grid.
    psd : callable
        psd(f) of the radial frequency f, normalized such that its integral
        over the frequency plane is the variance (e.g. `psd_power_law()`
        with a partial).
    rms : number, optional
        Scale the result to this RMS (the PSD then only sets the shape).
    seed : int or numpy Generator, optional

    Returns
    -------
    Z : 2d array, shape (len(y), len(x))
        Zero mean.
    """

    rng = np.random.default_rng(seed)
    FX, FY, dx, dy = _frequencies(x, y)
    Ny, Nx = FX.shape
    df = 1/(Nx*dx*Ny*dy)
    amp = np.sqrt(np.asarray(psd(np.hypot(FX, FY)), dtype=float)*df)
    amp[0, 0] = 0.0
    noise = (rng.standard_normal(FX.shape) + 1j*rng.standard_normal(FX.shape))/np.sqrt(2)
    # complex coefficients with <|c|**2> = psd*df; the real part keeps half
    # of the variance of each pair (f, -f), hence sqrt(2):
    Z = np.real(np.fft.ifft2(noise*amp))*Nx*Ny*np.sqrt(2)
    Z -= Z.mean()
    if rms is not None:
        Z *= rms/np.sqrt(np.mean(Z**2))

    return Z


def radial_psd(x, y, Z, n_bins=None, window=True):
    """Radially averaged 2D PSD of a map.

    Parameters
    ----------
    x, y : 1d arrays
        Equidistant grid.
    Z : 2d array
        Map (NaN samples count as zero, after removing the mean).
    n_bins : int, optional
        Frequency bins (default: half the smaller grid size).
    window : bool, optional
        Apply a 2D Hann window (with its power loss compensated) against the
        leakage of non-periodic maps.

    Returns
    -------
    f : array
        Bin centres (cycles per length unit).
    psd : array
        Mean 2D PSD in the bins (length**4).
    """

    Z = np.array(Z, dtype=float)
    ok = np.isfinite(Z)
    Z[ok] -= Z[ok].mean()
    Z[~ok] = 0.0
    Ny, Nx = Z.shape
    if window:
        w = np.outer(np.hanning(Ny), np.hanning(Nx))
        Z = Z*w/np.sqrt(np.mean(w**2))
    FX, FY, dx, dy = _frequencies(x, y)
    P = np.abs(np.fft.fft2(Z))**2*dx*dy/(Nx*Ny)
    f = np.hypot(FX, FY).ravel()
    n_bins = n_bins or min(Nx, Ny)//2
    f_max = min(0.5/dx, 0.5/dy)
    edges = np.linspace(0.0, f_max, n_bins + 1)
    idx = np.digitize(f, edges) - 1
    keep = (idx >= 0) & (idx < n_bins)
    s = np.bincount(idx[keep], P.ravel()[keep], n_bins)
    c = np.bincount(idx[keep], minlength=n_bins)
    with np.errstate(invalid="ignore"):
        psd = s/c

    return 0.5*(edges[1:] + edges[:-1]), psd


def band_rms(x, y, Z, f_low=0.0, f_high=np.inf):
    """RMS of the part of a map in a band of radial spatial frequencies
    (Fourier filter over the periodic window; Parseval).

    Parameters
    ----------
    x, y : 1d arrays
    Z : 2d array
        NaN samples count as zero after removing the mean.
    f_low, f_high : numbers, optional

    Returns
    -------
    rms : number
    """

    return np.sqrt(np.mean(band_filter(x, y, Z, f_low, f_high)**2))


def band_filter(x, y, Z, f_low=0.0, f_high=np.inf):
    """The part of a map with radial spatial frequencies in [f_low, f_high)
    (sharp Fourier filter, periodic window).

    Parameters
    ----------
    x, y : 1d arrays
    Z : 2d array
    f_low, f_high : numbers, optional

    Returns
    -------
    Z_band : 2d array
    """

    Z = np.array(Z, dtype=float)
    ok = np.isfinite(Z)
    Z[ok] -= Z[ok].mean()
    Z[~ok] = 0.0
    FX, FY, _, _ = _frequencies(x, y)
    f = np.hypot(FX, FY)

    return np.real(np.fft.ifft2(np.fft.fft2(Z)*((f >= f_low) & (f < f_high))))


def remove_zernikes(X, Y, Z, norm_radius, n_terms=36, centre=(0.0, 0.0)):
    """Least squares fit of the Fringe Zernikes Z_1 ... Z_n over the disk of
    radius norm_radius, and the residual.

    Parameters
    ----------
    X, Y, Z : 2d arrays
        Points and map; points outside the disk or with NaN are ignored.
    norm_radius : number
    n_terms : int, optional
    centre : (x0, y0), optional

    Returns
    -------
    residual : 2d array
        Z minus the fit inside the disk, NaN outside.
    coefficients : array
        coefficients[j - 1] of Z_j (length units).
    """

    u = (np.asarray(X, dtype=float) - centre[0])/norm_radius
    v = (np.asarray(Y, dtype=float) - centre[1])/norm_radius
    inside = (u**2 + v**2 <= 1.0) & np.isfinite(Z)
    A = np.column_stack([fringe_zernike(j, u[inside], v[inside]) for j in range(1, n_terms + 1)])
    c = np.linalg.lstsq(A, Z[inside], rcond=None)[0]
    res = np.full(np.shape(Z), np.nan)
    res[inside] = Z[inside] - A @ c

    return res, c


def specular_fraction(sigma, wavelength, n1=1.0, n2=None, theta=0.0):
    """Specular power fraction behind a rough surface (Gaussian heights,
    scalar theory): exp(-(2 pi sigma delta/wavelength)**2), i.e. 1 - TIS.

    Parameters
    ----------
    sigma : number or array
        RMS roughness (length units, the band that scatters out of the
        image).
    wavelength : number
        Vacuum wavelength (same units).
    n1 : number, optional
        Index of the incident medium.
    n2 : number, optional
        Index behind a refracting surface; None: a reflection (delta =
        2 n1 cos(theta)).
    theta : number or array, optional
        Angle of incidence (radians).

    Returns
    -------
    fraction : array
    """

    cos1 = np.cos(theta)
    if n2 is None:
        delta = 2*n1*cos1
    else:
        sin2 = n1*np.sin(theta)/n2
        delta = np.abs(n1*cos1 - n2*np.sqrt(np.clip(1 - sin2**2, 0.0, None)))

    return np.exp(-(2*pi*np.asarray(sigma, dtype=float)*delta/wavelength)**2)


def defect_map(X, Y, digs=(), scratches=(), transmittance=0.0):
    """Pixelated energy map of digs (disks) and scratches (line segments)
    that block (or attenuate) the light.

    Parameters
    ----------
    X, Y : 2d arrays
        Grid points.
    digs : sequence of (x, y, diameter), optional
    scratches : sequence of (x0, y0, x1, y1, width), optional
    transmittance : number, optional
        Power fraction passed by a defect (0: opaque).

    Returns
    -------
    T : 2d array
        1 outside the defects.
    """

    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    hit = np.zeros(X.shape, dtype=bool)
    for x0, y0, d in digs:
        hit |= np.hypot(X - x0, Y - y0) <= d/2
    for x0, y0, x1, y1, w in scratches:
        ex, ey = x1 - x0, y1 - y0
        L2 = ex**2 + ey**2
        t = np.clip(((X - x0)*ex + (Y - y0)*ey)/L2, 0.0, 1.0)
        hit |= np.hypot(X - x0 - t*ex, Y - y0 - t*ey) <= w/2

    return np.where(hit, transmittance, 1.0)
