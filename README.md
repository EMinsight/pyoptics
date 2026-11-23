# pyoptics

[![tests](https://github.com/aeberspaecher/pyoptics/actions/workflows/tests.yml/badge.svg)](https://github.com/aeberspaecher/pyoptics/actions/workflows/tests.yml)
[![docs](https://github.com/aeberspaecher/pyoptics/actions/workflows/docs.yml/badge.svg)](https://aeberspaecher.github.io/pyoptics/)
![Python 3.9 to 3.13](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

<!-- docs:about-start -->
**Wave optics, ray optics and lens design, Maxwell solvers, metrology, lasers
and lithography simulation in Python.**

pyoptics is a toolbox for optical simulation built on NumPy and SciPy: scalar
and vectorial wave propagation, Gaussian beams and modes, thin films and
polarization, imaging and aberrations, sequential and non-sequential ray
tracing with lens optimization and tolerancing, FDTD and FDFD solvers,
wavefront metrology, lasers, nonlinear and fibre optics, and a full
lithography imaging stack. Its blocks are small and independent, so a
propagator, a Zernike basis or a ray traced lens can be reused in new
contexts. The numerics are checked against analytic results throughout.
<!-- docs:about-end -->

![Fresnel diffraction behind a square aperture, a ray traced lithography lens, and the aerial image of contact holes](docs/images/readme_overview.png)

**[Documentation](https://aeberspaecher.github.io/pyoptics/)** ·
**[Example notebooks](notebooks/README.md)** ·
**[API reference](https://aeberspaecher.github.io/pyoptics/api/index.html)** ·
**[Conventions](https://aeberspaecher.github.io/pyoptics/conventions.html)**

## Features

| Area | What you get | Main modules |
|---|---|---|
| **Wave propagation** | Rayleigh-Sommerfeld, Fresnel and Fraunhofer propagators, free output windows, tilted planes, vectorial high NA focusing, analytic phase handling, WPM, field tracing chains, partially coherent sources | `propagators`, `tilted`, `analytic_phase`, `wpm`, `fieldtrace`, `source_modes` |
| **Beams and modes** | Hermite-, Laguerre- and Ince-Gaussian beams, ABCD matrices and resonators, M² after ISO 11146, mode decomposition, vortices, Bessel and Airy beams, ultrashort pulses | `beams`, `abcd`, `beam_quality`, `modes`, `ince`, `vortices`, `ultrafast` |
| **Thin films and polarization** | Fresnel coefficients, coherent and incoherent multilayers, Jones and Mueller calculus, materials and dispersion | `interfaces`, `thin_films`, `polarization`, `materials` |
| **Crystals and modulators** | Anisotropic crystals and the 4×4 method, polarizing prisms, Pockels and Kerr cells, liquid crystals, acousto-optics | `crystals`, `electrooptics`, `liquid_crystals`, `acoustooptics` |
| **Maxwell solvers** | FDTD in 1D, 2D and 3D with PML, dispersive metals, Bloch boundaries and mode sources; FDFD with resonances and far fields | `fdtd`, `fdfd`, `billiard`, `cloak` |
| **Imaging and microscopy** | PSF, MTF, Strehl, Zernike and other bases, partially coherent and vectorial imaging, phase contrast, DIC, confocal, ptychography, super-resolution, OCT, tweezers | `imaging`, `bases`, `vectorial_imaging`, `microscopy`, `superresolution`, `oct`, `tweezers` |
| **Ray tracing and lens design** | Sequential trace with conics, aspheres, freeforms, diffractive and GRIN surfaces, Seidel sums, OPD and Zernikes, polarization ray trace, damped least squares and global optimization, tolerancing, glass catalogues, zoom, scan and phone lenses, off-axis mirrors, CAD export | `raytrace`, `lens_optimization`, `tolerancing`, `glasses`, `qaspheres`, `zoom`, `off_axis`, `nodal`, `cad` |
| **Stray light** | Non-sequential ray tracing with scattering models, ghosts and coatings | `nsrt`, `bsdf` |
| **Rays and waves** | Ray field summation, Gaussian beamlets, Wigner rays, field tracing through lenses | `raywave`, `fieldtrace` |
| **Wavefront metrology** | Phase shifting and Fourier fringe analysis, holography, Shack-Hartmann, phase retrieval, unwrapping, TIE, IFTA, CGH null tests, wavefront shaping, Schlieren | `interferometry`, `holography`, `shack_hartmann`, `phase_retrieval`, `unwrap`, `TIE`, `ifta`, `cgh` |
| **Surface and film metrology** | White light interferometry, ellipsometry, scatterometry with RCWA, deflectometry | `csi`, `ellipsometry`, `scatterometry`, `deflectometry` |
| **Illumination and radiometry** | Radiometry and photometry, étendue, nonimaging concentrators, Köhler illumination, homogenizers, LEDs, freeform beam shaping, speckle | `radiometry`, `illumination`, `freeform`, `speckle`, `colour` |
| **Cameras and photography** | Bokeh, sunstars, flare, the sensor, depth of field, image restoration and correction, light field and lensless cameras | `photography`, `sensor`, `restoration`, `lens_correction`, `lightfield`, `lensless` |
| **Lasers** | Rate equations, threshold and output coupling, Q-switching and mode-locking, laser diodes, Fox–Li resonator modes | `lasers`, `resonators` |
| **Nonlinear and fibre optics** | SHG and phase matching, OPOs, Kerr effect; fibre modes, dispersion, Bragg gratings, links and the split-step NLSE | `nonlinear`, `fibres`, `fibrecomm`, `splitstep` |
| **Detectors** | Responsivity, noise, NEP and D*, photon statistics | `detectors` |
| **Astronomy and atmosphere** | Turbulence and adaptive optics, stellar interferometry, coronagraphs, telescopes, segmented mirrors, gravitational lensing, rainbows | `turbulence`, `coherence`, `coronagraph`, `telescopes`, `segmented`, `lensing`, `rainbow` |
| **Spectroscopy** | Grating, prism and echelle spectrometers, Fourier transform spectroscopy, Fabry–Perot filters, spectral lines | `spectrometers`, `spectroscopy`, `echelle` |
| **AR/VR, HUDs and 3D sensing** | Waveguide combiners, volume gratings, pancake lenses, windshield HUDs; LiDAR link budgets, FMCW, beam scanners, structured light | `arvr`, `lidar`, `structured_light` |
| **Lithography** | Abbe and Hopkins imaging, thin and rigorous masks (RCWA, FDTD), resist, process windows, ray traced projection lenses and illuminators, Jones pupils, source optimization and OPC, stochastics, EUV | `litho` |

The [module overview](https://aeberspaecher.github.io/pyoptics/modules.html)
in the documentation describes every module in detail.

## Installation

pyoptics is not on PyPI yet. It has one Cython extension, so you need a C
compiler (`build-essential` on Debian/Ubuntu, `xcode-select --install` on
macOS, the Microsoft C++ Build Tools on Windows). Then:

```bash
git clone https://github.com/aeberspaecher/pyoptics.git
cd pyoptics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[plot]"
```

Optional extras can be combined, for example `".[plot,fast,test]"`:

| Extra | Installs | For |
|---|---|---|
| `plot` | matplotlib | plotting tools |
| `fast` | numba | compiled ray trace kernel, several times faster ray traces and ray aiming |
| `nufft` | finufft | faster exact propagation onto planes tilted about a general axis |
| `hdf5` | h5py | streaming large FDTD snapshot stacks to compressed HDF5 files |
| `tie` | sympy | the TIE module |
| `notebooks` | matplotlib, JupyterLab, ipywidgets | running the example notebooks |
| `test` | pytest, pytest-xdist, matplotlib, sympy | running the test suite |
| `docs` | Sphinx and extensions | building the documentation |

The [installation guide](https://aeberspaecher.github.io/pyoptics/install.html)
walks through every step on Linux, macOS and Windows and covers common build
errors.

## Quickstart

Propagate a Gaussian beam by one Rayleigh range with the angular spectrum
method and compare it with the analytic beam (lengths in µm):

```python
import numpy as np
from pyoptics import propagators as prop
from pyoptics.beams import gauss_laguerre
from pyoptics.utils import grid2d

wl, w0 = 0.5, 5.0                      # wavelength and waist [µm]
z_R = np.pi*w0**2/wl                   # Rayleigh range
x, y, X, Y = grid2d(-80, 80, -80, 80, 512)

U0 = gauss_laguerre(0, 0, x, y, 0.0, w0, z_R, wl)          # waist at z = 0
U, _, _ = prop.propagate_padded(prop.rayleigh_sommerfeld_I_TF, U0, x, y, z_R, wl)
U_exact = gauss_laguerre(0, 0, x, y, z_R, w0, z_R, wl)

print(f"max. error: {np.abs(U - U_exact).max()/np.abs(U_exact).max():.1e}")
# max. error: 2.5e-04
```

Ray trace a plano-convex BK7 singlet (f = 100 mm, f/5) and look at its
spherical aberration (lengths in mm):

```python
import numpy as np
from pyoptics import raytrace as rt

BK7 = (1.5168, 64.17)                  # (n_d, V_d)
lens = rt.System([rt.Surface(51.68, 5.0, material=BK7, semi_diameter=10.5),
                  rt.Surface(np.inf, 95.0, n_after=1.0, semi_diameter=10.5)],
                 0, 587.6e-6, entrance_pupil_radius=10.0).at_paraxial_focus()

print(f"EFL:           {lens.paraxial().efl:.2f} mm")
print(f"W040 (Seidel): {lens.seidel(0.0).W040:.2f} waves")
print(f"RMS OPD:       {lens.rms_opd(0.0):.3f} waves, Strehl {lens.strehl(0.0):.2f}")
# EFL:           100.00 mm
# W040 (Seidel): -4.62 waves
# RMS OPD:       1.388 waves, Strehl 0.03
```

Image 45 nm lines and spaces with an ArF water immersion scanner (lengths in
nm):

```python
from pyoptics.litho import Illumination, MaskLayout, ThinMask, Projector, LithoSimulation

sim = LithoSimulation(MaskLayout.lines_spaces(pitch=90.0, cd=45.0),
                      ThinMask.binary(),
                      Illumination.dipole_for_pitch(90.0, 193.0, 1.35, polarization="y"),
                      Projector(NA=1.35, wavelength=193.0, n=1.437))
print(f"NILS {sim.nils():.2f}, dose to size {sim.dose_to_size(threshold=0.3):.2f}")
# NILS 2.82, dose to size 0.85
```

## Example notebooks

The [`notebooks/`](notebooks/README.md) folder holds 173 executed Jupyter
notebooks, one per topic. They are stored with their outputs, so the
figures show directly on GitHub, and they are rendered in the
[documentation](https://aeberspaecher.github.io/pyoptics/examples.html).
Most of them compare the numerics with analytic results.

The notebooks are numbered in reading order: the hundreds digit is the part,
the tens digit the chapter.

| Numbers | Part | Chapters |
|---|---|---|
| 000–099 | Start here | foundations and conventions, geometrical optics primer |
| 100–199 | Waves | propagation, beams and modes, coherence |
| 200–299 | Light in matter | thin films, polarization and crystals, FDTD, FDFD |
| 300–399 | Imaging and microscopy | image formation and quality, microscopy |
| 400–499 | Ray tracing and lens design | basics, optimization and tolerancing, aspheres, lens systems, stray light, rays and waves, field tracing |
| 500–599 | Metrology and computational imaging | interferometry, wavefront sensing, ellipsometry and scatterometry, holography and phase retrieval |
| 600–699 | Illumination, radiometry and cameras | radiometry and illumination, photography |
| 700–799 | Lasers, nonlinear optics, fibres and detectors | lasers, nonlinear optics, fibre optics, detectors |
| 800–899 | Lithography | imaging and resist, masks, lens and illuminator, OPC and stochastics, EUV |
| 900–999 | Instruments | telescopes and astronomy, spectroscopy, AR/VR and HUDs, 3D sensing |

**Where to start.** New to optics: notebook 001 (introduction to
geometrical optics). Know optics, new to pyoptics: notebook 000
(foundations and conventions) and the quickstart above. Looking for a topic:
the table above or the
[learning paths](https://aeberspaecher.github.io/pyoptics/learning_paths.html),
which suggest routes for different backgrounds.

To run them yourself, install the `notebooks` extra and start JupyterLab:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab notebooks/
```

## Status

<!-- docs:status-start -->
pyoptics 0.1 is a research and teaching toolbox, not a production library.
It covers a lot of ground quickly: most modules are less than a year old,
and their API may change between versions. What holds them together is
testing: more than 3000 pytest tests check the numerics against analytic
results and published data, in GitHub Actions on Python 3.9, 3.11 and 3.13
for every push and pull request. The results have not been compared
systematically with commercial codes; where a notebook does compare, it
says so. There is no release on PyPI yet.

The [module overview](https://aeberspaecher.github.io/pyoptics/modules.html)
gives each module a maturity tier: **core** (stable API, used everywhere),
**tested** (checked against analytic or published results, API may change),
**showcase** (written for one or two notebooks, narrow) or **unfinished**.
Planned work is in
[`WIP.md`](https://github.com/aeberspaecher/pyoptics/blob/master/WIP.md),
finished work in
[`CHANGELOG.md`](https://github.com/aeberspaecher/pyoptics/blob/master/CHANGELOG.md).
<!-- docs:status-end -->

## Conventions

pyoptics uses the time dependence exp(−iωt) throughout, so a plane wave
travelling in +z is exp(+ikz), and `fft.FT()` works on centered arrays. Sign
conventions for lenses, ABCD matrices, ray tracing, absorbing media and
polarization are collected on the
[conventions page](https://aeberspaecher.github.io/pyoptics/conventions.html);
read it before comparing results with a textbook.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest -n auto test
```

The test suite (more than 3000 tests) takes a few minutes on four cores with
`-n auto` (pytest-xdist). Each test process then uses a single thread, see
`test/conftest.py`: by default the FFTs, the numba kernels and BLAS each use
all cores, which is fastest for one process but slows parallel processes
down many times over. Use `pyoptics.utils.set_threads(1)` in your own
process pools for the same reason. After editing `pyoptics/wpm.pyx`,
run the install command again to recompile the extension. Docstrings follow
the NumPy style; the documentation is built with Sphinx from `docs/`:

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The figure at the top of this README is made by
[`docs/images/make_readme_figure.py`](docs/images/make_readme_figure.py).

## License

pyoptics is released under the [BSD 3-Clause License](LICENSE).
Copyright 2014–2026 Alexander Eberspächer.
