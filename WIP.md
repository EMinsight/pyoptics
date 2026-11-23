# WIP

## Low-hanging fruit

* rectangle with rounded egdes aperture

* rotation angle in apertures

* Fresnel numbers for general annular apertures

## Medium

* Lithography (DUV, see the roadmap in the project files): RCWA for 2D masks (the FDTD
  mask model covers them, slowly), ray traced stress birefringence
* Lithography layouts: masks with more than two regions (alternating PSM,
  tri-tone) for polygon layouts; non-Manhattan polygons in `FDTDMask` (subpixel
  averaging of oblique edges) and in the fragment based OPC
* More tests for `TIE`
* WPM: Fresnel transmission correction at index steps, absorbing boundaries;
  symmetrized step (alternate the two operator orderings) against the power
  drift in narrow waveguides (notebook 105, wave propagation method, section 12)
* Docstrings: Parameters/Returns for single-argument functions, methods and
  properties (see the report-only numpydoc check in the docs workflow); then
  make the check blocking per module
* Documentation: user guide pages per topic, bibliography, doctests, weekly
  re-execution of the notebooks
* Incoherence in thin films (spatial and temporal incoherence)
* FDFD, next: 3D (needs an iterative solver), dispersive media inside the
  eigenvalue problem (nonlinear eigenproblem; now evaluated at the guess),
  symmetry boundaries (half and quarter domains for mirror-symmetric
  cavities), better subpixel averaging of curved boundaries (lowers the
  numerical Q floor of whispering gallery modes, about 1e7 at 20 cells per
  wavelength), Husimi functions and ray dynamics for deformed cavities
* FDTD, next milestones (done: 1D/2D core, numba kernels, dispersive media,
  waveguide modes, mode sources/monitors, S-parameters, Q extraction,
  Bloch boundaries, diffraction orders vs. RCWA, band structures,
  photonic-crystal devices, grating coupler, metalens, 3D with the Mie
  sphere and the nanojet, 3D diffraction orders, the litho mask model
  `litho.FDTDMask`; notebooks 220 (FDTD basics), 221 (FDTD metals and plasmons), 223 (FDTD photonic circuits), 222 (FDTD periodic structures), 224 (FDTD devices), 225 (3D FDTD and nanojet) and 812 (FDTD mask for lithography)): 3D mode
  sources and monitors (needs the mode solver on a 2D cross section);
  subpixel averaging for dispersive media (FDTDMask puts edges on nodes
  instead); a GPU or MPI backend for large 3D mask cells; mode sources with dispersion-matched auxiliary
  grids (no backward leakage away from the centre wavelength); a second-order Debye
  update; better accuracy for metal boundaries in the Hz mode (first order
  now); `analytical.cylinder_field` for H along the axis (the series is in
  notebook 220 (FDTD basics) and test_fdtd.py)
* Basis class factory for numbering scheme plus callable

## More challenging

## DONE

* Ince-Gaussian beams (`ince`: Ince polynomials, even, odd and helical IG
  modes, HG and LG content; notebook 114 (Ince-Gaussian beams))

* General astigmatic beams (`abcd`: 4x4 matrices, beam matrix Q,
  `collins_4x4()`; `beam_quality`: ten second moments, intrinsic M² and
  twist; notebooks 111 (ABCD matrices), sections 11 and 12, and 07 (beam
  quality), section 4)

* Radiometry and photometry (`radiometry`: black bodies, V(λ) and V'(λ)
  fits, luminous efficacy, view factors, the integrating sphere, the
  camera equation and exposure; `photography.retrofocus()` and the image
  side relative illumination; notebooks 600 (radiometry and photometry) and 601 (radiometry of images))

* Trigonometric interpolation (`interpolator`: 1D/2D Fourier interpolation,
  FFT upsampling, Lanczos sigma factors and other windows against Gibbs,
  exact threshold CDs from coarse grids; notebook 803, Fourier interpolation of aerial images)
* Microscopy (`microscopy`: Abbe imaging with Köhler illumination, weak
  object transfer, Zernike phase contrast, Nomarski DIC, confocal PSFs and
  optical sectioning, Fourier ptychography with pupil recovery; notebooks
  300 (Abbe image formation) and 310 (phase contrast and DIC) to 312 (Fourier ptychography))
* Fresnel lenses (`fresnel`: zones from a parent surface, ideal and real
  facet ray tracing with draft, shadowing and riser losses, exact facet
  design and the fit back to an asphere, kinoforms; notebook 421, Fresnel lenses)
* Conversion between rays and wave fields (`raywave`: ray field summation
  with Gouy phases, Gaussian beamlets, phase gradient and local spectrum
  rays, Wigner function; notebooks 450 (rays and waves), 453 (laser head DOE) and 454 (eye aberrometer))
* Propagation with analytic handling of the quadratic and linear phase and
  of grid offsets after Wyrowski and coworkers (`analytic_phase`:
  semi-analytical and homeomorphic Fourier transforms, parallel and tilted
  planes, notebook 103, analytic phase propagation)
* Docstrings: NumPy-style Parameters/Returns sections for every public
  function with two or more arguments; report-only `numpydoc lint` in the docs
  workflow (configured in `pyproject.toml`)

* Rays to waves at the focus of a lithography lens (notebook 451, lens focus from rays to waves): traced
  exit pupil plus angular spectrum through focus, ray caustics vs. wave
  caustics, axial colour; the angular spectrum propagators drop evanescent
  waves for z < 0

* Coated laser mirror (notebook 203): quarter wave layers at oblique incidence
  (`coatings.Layer.quarter_wave(angle=...)`), vectorial focus of polarization
  traced rays (`raytrace.System.vectorial_focus`)

* Radius of curvature measurement with a Fizeau interferometer (notebook 502, radius interferometer):
  double pass systems (`raytrace.double_pass`), optical path on a detector
  grid (`System.ray_map`), displaced test spheres
  (`interferometry.fit_sphere_displacement`)

* Polarization ray trace with energy in `raytrace` (`System.trace_polarization`):
  coatings with thickness profiles on any surface (`coatings`), coated
  mirrors, absorbing, uniaxial (retardance, dichroism) and cubic crystal
  media (`materials`); `litho.lens.lens_jones_pupil` built on it

* Lithography: lens aberrations from the ray tracer across the scanner
  slit, Zernike sensitivities (`litho.lens`, notebook 820, lens aberrations across the slit)
* Lithography: Hopkins TCC and SOCS imaging (`litho.hopkins`, notebook 802, Hopkins imaging and SOCS)
* Lithography: swing curves, BARC optimization and a resist model
  (exposure, PEB, Mack development; `litho.wafer`, `litho.resist`,
  notebook 804, resist and swing curves)
* Lithography: rigorous mask diffraction with 1D RCWA and boundary layer
  calibration (`litho.rcwa`, notebook 810, rigorous mask)
* Lithography: rigorous FDTD mask model for 1D and 2D layouts
  (`litho.FDTDMask`, notebook 812, FDTD mask for lithography)
* FDFD solver in 2D (`fdfd`: driven solves, resonances, mode tracing, far
  fields; notebooks 230 (FDFD basics) and 231 (limaçon microcavity), the limaçon microcavity)
* Lithography: Jones pupils (Pauli-Zernike, orientation Zernikes, polar
  decomposition) and the polarization ray trace of a lens with coatings
  and CaF2 intrinsic birefringence (`litho.jones_pupil`, `litho.lens`,
  notebook 821, Jones pupils)
* Lithography: source optimization of pixelated sources (`litho.source_opt`,
  notebook 830, source optimization)
* Lithography: OPC of Manhattan polygons (bias tables, SRAFs, model-based
  OPC with fragments and EPE; `litho.opc`, notebook 831)
* Lithography: geometric (chief ray) and energetic telecentricity of a
  coated lens across the slit, the real pupil and its apodization in the
  projector, pattern shift through focus (`litho.lens.lens_telecentricity`,
  `lens_apodization`, notebook 822, telecentricity across the slit)
* Lens design: Delano (y-ybar) diagrams (`delano`), a least squares lens
  optimizer (`lens_optimization`), first order layout and correction of a
  20 element KrF lens of real NA 0.7 with 10 aspheres, from the layout and
  from Caldwell's lens, asphere placement by predicted gain, the axial
  position of the NA stop (`raytrace.delano_krf_lens`, notebook 405, Delano diagrams)
* Lens design: classical damped least squares optimizer with variables
  (curvatures, thicknesses, conics, aspheres, model glasses), solves,
  operands (ray errors, spots, wavefront, Seidel, colour) and Lagrange
  multiplier constraints (`lens_optimization.DampedLeastSquares`, notebook 410, lens optimization:
  achromat, Cooke triplet, Lister and plan microscope objectives)
* Glass catalogues of SCHOTT, OHARA, HOYA, CDGM, SUMITA, HIKARI with partial
  dispersions; snapping model glasses to real ones (`glasses`, notebook 404, glass catalogues)

* FFT module
* fitting bases to scattered data
* Chebyshev basis
* Müller matrices and some Stokes vectors
* elliptical aperture
* catch z=0 case in beam function
* "ensure_meshgrid()" function or similar
* modern packaging (pyproject.toml, replaces waf)
* Python 3 port
* test suite against analytical results, CI on Python 3.9, 3.11 and 3.13
* propagator tests against Mahajan's formula
* consistent phase convention exp(i(k·r − ωt)), documented in the README
* basic Jones vectors and matrices
* example notebooks
* image processors: Scaler, CameraNoiser (EMVA 1288 camera model), Centerer
* padding/unpadding routines (`utils.pad()`, `unpad()`)
* image shift (`processing.Shifter`)
* rotation angle in basis functions, abstract base class for tensor product
  bases (`bases.TensorProductBasis`)
* Gaussian beam parameters according to ISO 11146 (`beam_quality`)
* analytical Airy disk and OTF/MTF for circular apertures
* PSF, OTF & MTF tool with pupil masks and phases, Strehl ratio and Maréchal
  approximations (`imaging`)
* transfer functions of thin film systems for the angular spectrum
  (`thin_films.stack_transfer_functions()`)
* notebooks 302 (PSF, MTF and Strehl), 112 (beam quality) and 202 (stacks as transfer functions) (PSF/MTF/Strehl, beam quality, stack transfer
  functions)
* vectorial propagation: E_z from div E = 0, vectorial angular spectrum,
  vectorized scalar propagators, Mansuripur's high NA focusing checked
  against Richards-Wolf (notebook 104, vectorial diffraction)
* vectorial imaging (Abbe, s/p/unpolarized, lithography notebook 800, vectorial imaging in lithography),
  polarization converters x/y <-> TE/TM, Stokes -> Jones, DOP, IPS
* lithography toolbox `litho` (sources, mask layouts, thin and thin film
  mask stacks, boundary layer, resist on a thin film stack with TE/TM
  Fresnel transmission, NILS, CD, MEEF, Bossung curves, process windows;
  notebooks 800 (vectorial imaging in lithography) and 801 (lithography toolbox)), pupil apodization (apodization Zernikes) and KrF
  material data
* ABCD matrices (`abcd`): rays, cardinal points, Gaussian beams via q,
  resonators, Collins integral (notebook 111, ABCD matrices)
* 2D phase unwrapping (`unwrap`): least squares (DCT and weighted PCG),
  quality guided, residues (notebook 501, phase unwrapping); reference guided (TIE) and
  two-wavelength unwrapping (notebook 531, holographic microscopy)
* interferometry (`interferometry`): phase-shifting algorithms (N-step,
  4-step, Carré, Hariharan), Takeda's Fourier-transform method,
  interferogram simulation, shifter miscalibration sensitivity (notebook 500, interferometry)
* digital holography (`holography`): off-axis, phase-shifting and Gabor
  reconstruction, refocusing, autofocus, quantitative phase (notebook 530, digital holography)
* chirp-z transform (`fft.czt()`, `zoom_FT()`, `zoom_inv_FT()`), shifted
  angular spectrum, scaled Fresnel/Fraunhofer propagators, far field in
  chosen directions, zoomed PSFs (notebook 101, chirp-z transform)
* sequential ray tracing (`raytrace`): spheres, conics, aspheres, mirrors,
  paraxial pupils and Seidel sums checked against `abcd`, OPD maps, Zernike
  fits, PSF/MTF/Strehl via `imaging`, spot diagrams and ray fans
  (notebook 400, ray tracing to OPD)
* tilts and decentres in `raytrace` (decentred/tilted surfaces, coordinate
  breaks, fold mirrors, tilted image planes), telecentric objects, the
  Caldwell KrF lithography lens as an example (notebook 400, ray tracing to OPD)
* axial (z) shift sensitivity of the Caldwell lens elements
  (`System.with_axial_shift`; focus, Z9, magnification; notebook 400, ray tracing to OPD)
* radius and refractive index sensitivities (`System.with_radius_change`,
  `System.with_index_change`, `Surface(index_offset=...)`; per surface and
  per element of the Caldwell lens in notebook 400, ray tracing to OPD)
* water immersion ArF lens (NA 1.35, 8x, 11 aspheres) from Zeiss patent
  US 7,187,503 B2, second embodiment (`raytrace.zeiss_arfi_lens()`); ray
  aiming by continuation for the rim rays of high NA lenses
* phase retrieval (`phase_retrieval`): error reduction and hybrid
  input-output with support, reality and positivity constraints,
  shrink-wrap, near-field (Fresnel) CDI via the propagators, phase
  diversity (multi-plane Gerchberg-Saxton and Zernike fit, paraxial or
  exact defocus) from through-focus PSFs; first tests for `ifta`
  (notebook 532, phase retrieval)
* mode analysis and fibres (`modes`): HG/LG decomposition with fitted basis,
  M² from the mode content, LP modes of step-index fibres, Marcuse mode
  field radius, coupling efficiency and misalignment sweeps (notebook 113, mode analysis)
* propagation onto tilted planes (`tilted`): exact non-uniform sum and
  Matsushima's spectrum rotation, scalar and vectorial, tilted source to
  parallel target, beam evaluation (ISO 11146 widths) on inclined planes
  (notebook 102, tilted planes)
* Sphinx documentation (`docs/`, Furo theme, notebooks via myst-nb, API
  reference, built on every PR with warnings as errors, published to GitHub
  Pages)
* KrF illuminator for the Caldwell lens (`litho.illuminator`): REMA
  objective of US 5,982,558 (misprinted air space 19 reconstructed from the
  ray angle table), zoom-axicon of US 5,675,401 re-glassed for 248 nm,
  `raytrace.Axicon` cones, invented source, diffuser, coupling lens and
  glass rod; sigma, annular settings, pupil fill per field point,
  uniformity and telecentricity against the lens (notebook 823, KrF illuminator)
* Zernike aberration gallery (notebook 303): a ray traced lens for each Fringe
  Zernike Z2 to Z16 with ray fans, rays and waves through focus and PSFs;
  `raytrace.ZernikeSurface` (Fringe Zernike sag departure) and
  `System.draw(plane="xz")`
