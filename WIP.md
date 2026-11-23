# WIP

Open work, grouped by the parts of the notebooks (see `notebooks/toc.json`).
Each item says how large it is (small, medium, large) and which module it
touches. Finished work moves to [`CHANGELOG.md`](CHANGELOG.md).

## Across the package

* **medium**, all modules: Parameters/Returns docstring sections for
  single-argument functions, methods and properties (see the report-only
  numpydoc check in the docs workflow); then make the check blocking per
  module
* **medium**, docs: weekly re-execution of all notebooks in CI with run
  times, failing on errors; a header cell per notebook (level,
  prerequisites, modules, run time, previous and next); notebooks listed on
  the API page of each module they use
* **medium**, docs: user guide pages for the core workflows, one
  bibliography (sphinxcontrib-bibtex), a validation page per module,
  doctests, a glossary, `CITATION.cff`
* **small**, `bases`: basis class factory for a numbering scheme plus a
  callable

## 1xx Waves

* **small**, `masks`: rectangle with rounded edges; rotation angle for
  apertures; Fresnel numbers for general annular apertures
* **medium**, `wpm`: Fresnel transmission correction at index steps,
  absorbing boundaries; symmetrized step (alternate the two operator
  orderings) against the power drift in narrow waveguides (notebook 105,
  wave propagation method, section 12)

## 2xx Light in matter

* **large**, `fdfd`: 3D (needs an iterative solver); dispersive media
  inside the eigenvalue problem (nonlinear eigenproblem; now evaluated at
  the guess); symmetry boundaries (half and quarter domains for
  mirror-symmetric cavities); better subpixel averaging of curved boundaries
  (lowers the numerical Q floor of whispering gallery modes, about 1e7 at
  20 cells per wavelength)
* **large**, `fdtd`: 3D mode sources and monitors (needs the mode solver
  on a 2D cross section); subpixel averaging for dispersive media
  (`FDTDMask` puts edges on nodes instead); a GPU or MPI backend for large
  3D mask cells; mode sources with dispersion-matched auxiliary grids (no
  backward leakage away from the centre wavelength); a second-order Debye
  update; better accuracy for metal boundaries in the Hz mode (first order
  now); `analytical.cylinder_field` for H along the axis (the series is in
  notebook 220 (FDTD basics) and test_fdtd.py)

## 5xx Metrology

* **small**, `TIE`: more tests

## 8xx Lithography

* **large**, `litho`: RCWA for 2D masks (the FDTD mask model covers them,
  slowly); ray traced stress birefringence (DUV, see the roadmap in the
  project files)
* **medium**, `litho`, `polygons`: masks with more than two regions
  (alternating PSM, tri-tone) for polygon layouts; non-Manhattan polygons in
  `FDTDMask` (subpixel averaging of oblique edges) and in the fragment based
  OPC
