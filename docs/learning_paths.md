# Learning paths

The {doc}`examples` are numbered in reading order: the hundreds digit is
the part, the tens digit the chapter. Reading a chapter from its first
notebook to its last is a good default, and the applied chapters list
below what to read first. The paths on this page are for readers who want
a cross-section through several parts. Each path has a **core** to read in
order, followed by optional side trips. The notebooks stand on their own,
so leaving a path or mixing two is fine; where a notebook builds on an
earlier one, its introduction says so.

Most readers profit from {doc}`conventions` and {doc}`000 Foundations and conventions <notebooks/000_foundations_conventions>` early: they fix the
signs of phases, radii, angles and Zernikes that every other notebook uses.
Readers new to optics can leave notebook 000 (foundations and conventions) for later and read it as a
summary, and start instead with {doc}`001 Introduction to geometrical optics <notebooks/001_geometrical_optics_intro>`:
the lens equation, real and virtual images, principal planes, the
magnifier, microscope and telescope, the Seidel aberrations and the
wavefront error, each formula checked with the ray tracer.

A few notebooks serve almost every path:

- {doc}`300 Abbe image formation <notebooks/300_abbe_image_formation>` is the entry to all of imaging.
- {doc}`302 PSF, MTF and Strehl <notebooks/302_psf_mtf_strehl>` defines the image quality measures that the later notebooks report.
- {doc}`450 Rays and waves <notebooks/450_rays_and_waves>`, {doc}`451 Lens focus from rays to waves <notebooks/451_lens_focus_waves>` and {doc}`452 Gaussian beam decomposition <notebooks/452_gaussian_beam_decomposition>` connect ray tracing with wave optics.

## Undergraduate in a first optics course

For readers who know the textbook theory (Hecht level) and want to see it
work. The core follows the order of a typical lecture, which is also the
order of the numbers.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`001 Introduction to geometrical optics <notebooks/001_geometrical_optics_intro>` | Lens equation, real and virtual images, principal planes, magnifier, microscope and telescope, Seidel aberrations and wavefront error: the ray optics of the first weeks of the course. |
| 2 | {doc}`100 Free-space propagation <notebooks/100_propagation>` | Slits, apertures, Talbot effect and the Poisson spot: diffraction you can check by hand. |
| 3 | {doc}`110 Gaussian beams <notebooks/110_gaussian_beams>` | Gaussian beams as the paraxial building block, and the Gouy phase. |
| 4 | {doc}`111 ABCD matrices <notebooks/111_abcd_matrices>` | Paraxial rays and beams through lenses and resonators with 2×2 matrices. |
| 5 | {doc}`120 Coherence and stellar interferometry <notebooks/120_coherence_stellar_interferometry>` | Spatial and temporal coherence, and how they measure stars. |
| 6 | {doc}`200 Interfaces and thin films <notebooks/200_interfaces_thin_films>` | Fresnel coefficients, Brewster angle, total internal reflection, soap films. |
| 7 | {doc}`210 Jones and Mueller calculus <notebooks/210_jones_mueller_calculus>` | Polarization with Jones vectors and Stokes parameters. |
| 8 | {doc}`300 Abbe image formation <notebooks/300_abbe_image_formation>` | Why an imaging system has a resolution limit: diffraction orders and the pupil. |
| 9 | {doc}`301 Imaging and Zernikes <notebooks/301_imaging_zernikes>` | The pupil function, the PSF and aberrations described by Zernike polynomials. |
| 10 | {doc}`302 PSF, MTF and Strehl <notebooks/302_psf_mtf_strehl>` | PSF, MTF and Strehl ratio, the language of image quality. |
| 11 | {doc}`500 Interferometry <notebooks/500_interferometry>` | How interferometers turn fringes into wavefronts. |

**Side trips:** {doc}`906 Rainbows and glories <notebooks/906_rainbows_and_glories>` (Descartes' ray picture and Airy's wave picture of
the same bow), {doc}`450 Rays and waves <notebooks/450_rays_and_waves>` (a lab rule explained with rays and waves), {doc}`115 Bessel and Airy beams <notebooks/115_bessel_airy_self_healing>`
(beams that seem not to diffract), {doc}`600 Radiometry and photometry <notebooks/600_radiometry_photometry>` (watts, lumens and why a wall looks equally bright from any distance), {doc}`910 Grating and prism spectrometers <notebooks/910_grating_prism_spectrometers>` sections 1 and 3 (prism against grating, the slit and the resolving power) and {doc}`912 Fabry–Perot and filters <notebooks/912_fabry_perot_filters>` section 1 (the Fabry–Perot as the multiple-beam sequel to two-beam interference),
{doc}`720 Light in a fibre <notebooks/720_fibre_rays_step_graded>` (sections 1 to 4: total internal reflection, the NA and why pulses spread in a
multimode fibre) and {doc}`725 A fibre link <notebooks/725_fibre_link>` (section 1: why telecom uses 1550 nm). Afterwards, {doc}`000 Foundations and conventions <notebooks/000_foundations_conventions>` shows where everything
fits on the ladder from Maxwell's equations to rays.

## Curious reader

For readers who want the story behind a phenomenon and are happy to read the
figures without running the code. No order is needed.

| Notebook | Story |
|---|---|
| {doc}`115 Bessel and Airy beams <notebooks/115_bessel_airy_self_healing>` | Beams that heal behind an obstacle. |
| {doc}`116 Optical vortices <notebooks/116_optical_vortices>` | Light with a twisted wavefront. |
| {doc}`233 Carpet cloak <notebooks/233_carpet_cloak>` | Hiding a bump under a carpet with a graded material. |
| {doc}`402 Sine and Herschel conditions <notebooks/402_sine_herschel_lagrange>` | Why a paraboloid has comet-shaped stars, why a microscope sees one plane sharply, and how remote focusing gets around it. |
| {doc}`421 Fresnel lenses <notebooks/421_fresnel_lenses>` | From lighthouse lenses to solar concentrators. |
| {doc}`450 Rays and waves <notebooks/450_rays_and_waves>` | Which way round a plano-convex lens should go. |
| {doc}`511 Schlieren and knife-edge <notebooks/511_schlieren_knife_edge>` | Foucault's razor blade test and seeing air flow. |
| {doc}`536 Lensless imaging <notebooks/536_lensless_coded_aperture_imaging>` | Cameras without lenses: pinholes, masks and diffusers. |
| {doc}`601 Radiometry of images <notebooks/601_radiometry_of_images>` | Why a telescope never shows the Moon brighter than the naked eye, why the corners of a photo are darker, and where the sunny 16 rule comes from. |
| {doc}`602 Étendue <notebooks/602_etendue_brightness_theorem>` | Why a burning glass cannot make anything hotter than the Sun. |
| {doc}`606 Speckle reduction <notebooks/606_speckle_reduction>` | Why a laser spot glitters, and how to stop it. |
| {doc}`610 The sensor <notebooks/610_sensor>` | Where moiré and false colour come from, what the low pass filter in front of a sensor does, and why some lenses darken the corners on digital cameras. |
| {doc}`611 Bokeh <notebooks/611_bokeh>` | Why the background of a fast lens shows cat's eyes, soap bubbles and onion rings. |
| {doc}`612 Sunstars <notebooks/612_sunstars>` | Why a stopped down lens turns the Sun into a star, and how many rays it gets. |
| {doc}`613 Flare and ghosts <notebooks/613_flare_ghosts>` | Where the coloured polygons come from when the Sun is in the frame, and what lens coatings do about them. |
| {doc}`614 Depth of field and tilt-shift <notebooks/614_tilt_shift_depth_of_field>` | How much is sharp, when stopping down stops helping, and how a tilted lens puts a whole landscape in focus at f/2.8. |
| {doc}`616 The light field camera <notebooks/616_light_field_camera>` | How a camera with microlenses in front of the sensor can be focused after the picture was taken, and what it pays for that. |
| {doc}`900 Telescope zoo <notebooks/900_telescope_zoo>` | Why big telescopes are Ritchey–Chrétiens, why a Newtonian has comet-shaped stars at the edge, and how the Schmidt camera sees a wide field with a spherical mirror. |
| {doc}`901 Segmented mirrors <notebooks/901_segmented_mirrors>` | Where the six plus two spikes of JWST images come from, and how 18 mirror segments are aligned to a few nanometres. |
| {doc}`902 Atmospheric dispersion corrector <notebooks/902_atmospheric_dispersion_corrector>` | Why stars near the horizon look like little rainbows, and how two rotating prisms in a telescope undo it. |
| {doc}`903 Twinkling stars and AO <notebooks/903_twinkling_stars_adaptive_optics>` | Why stars twinkle, and how telescopes undo it. |
| {doc}`904 Coronagraphs <notebooks/904_coronagraphs>` | How a telescope can see a planet ten billion times fainter than the star next to it. |
| {doc}`905 Einstein rings <notebooks/905_einstein_rings>` | How a galaxy bends light like a lens, turning a distant quasar into a ring or a cross, and why a star passing behind another brightens. |
| {doc}`906 Rainbows and glories <notebooks/906_rainbows_and_glories>` | Why the rainbow is where it is, and the glory around an aircraft's shadow. |
| {doc}`912 Fabry–Perot and filters <notebooks/912_fabry_perot_filters>` | How the Sun is filmed in the light of a single spectral line with a stack of birefringent plates (section 4). |
| {doc}`913 Spectral lines and absorption <notebooks/913_spectral_lines_absorption>` | Why the CO₂ in a metre of air is already black at 4.3 µm, and why doubling it does not double its absorption (sections 3 and 4). |
| {doc}`914 Echelle spectrograph <notebooks/914_echelle_spectrograph>` | How a spectrograph fits the whole visible spectrum at a resolution of 100 000 onto one detector, and why big telescopes need big gratings. |

## Lens designer

For readers who use a commercial lens design program and want to compare
pyoptics with it, or go from the ray trace to the wave optical image.
Readers who want a refresher of first order optics and the Seidel
aberrations first can start with {doc}`001 Introduction to geometrical optics <notebooks/001_geometrical_optics_intro>`.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`000 Foundations and conventions <notebooks/000_foundations_conventions>` | Sign conventions for radii, angles, OPD and Zernikes. |
| 2 | {doc}`111 ABCD matrices <notebooks/111_abcd_matrices>` | Paraxial layout with ABCD matrices. |
| 3 | {doc}`302 PSF, MTF, Strehl <notebooks/302_psf_mtf_strehl>` | PSF, MTF and Strehl from the pupil. |
| 4 | {doc}`303 Zernike aberration gallery <notebooks/303_zernike_aberration_gallery>` | What each Zernike term looks like in a real lens. |
| 5 | {doc}`400 Ray tracing to OPD <notebooks/400_ray_tracing_opd>` | Exact ray tracing, OPD maps, Zernikes over the field. |
| 6 | {doc}`401 Seidel aberrations <notebooks/401_seidel_caldwell_lens>` | Seidel sums surface by surface: where correction and tension come from. |
| 7 | {doc}`402 Sine and Herschel conditions <notebooks/402_sine_herschel_lagrange>` | The Lagrange invariant, Conrady's OSC, aplanatic points and why the sine and Herschel conditions exclude each other. |
| 8 | {doc}`403 Magnification, distortion, telecentricity <notebooks/403_magnification_distortion_telecentricity>` | Magnification, distortion and telecentricity as Zernikes. |
| 9 | {doc}`404 Glass catalogues <notebooks/404_glass_catalogues>` | Glass maps, partial dispersion and an apochromat. |
| 10 | {doc}`405 Delano diagrams <notebooks/405_litho_delano_lens_design>` | First order layout with $y$-$\bar y$ diagrams. |
| 11 | {doc}`410 Lens optimization <notebooks/410_lens_optimization>` | Damped least squares optimization, from a singlet to a microscope objective. |
| 12 | {doc}`413 Monte Carlo tolerancing <notebooks/413_monte_carlo_tolerancing>` | Monte Carlo tolerancing. |
| 13 | {doc}`824 Lens tolerances and wafer yield <notebooks/824_lens_tolerances_wafer_yield>` | Tilt, wedge, irregularity and inhomogeneity tolerances, manipulators with limited strokes, inverse sensitivity and tolerance allocation, on a lithography lens. |

**Lens design classics:** {doc}`431 Zoom lenses <notebooks/431_zoom_lenses>` (mechanical compensation, a patent zoom with real glasses, cams and their tolerances, inner focusing and focus breathing), {doc}`420 Forbes Q aspheres <notebooks/420_forbes_q_aspheres>` (slope-orthogonal asphere descriptions, manufacturability and better conditioned optimization), {doc}`430 Phone camera lens with Forbes Q polynomials <notebooks/430_phone_camera_lens_forbes_q>` (a five element f/1.8 plastic phone lens designed with Q-bfs surfaces by widening the field step by step, manufacturing limits on the Q terms, Zernike wavefronts at three wavelengths and Q2D freeform terms), {doc}`435 Progressive spectacle lenses <notebooks/435_progressive_lenses>` (the Minkwitz theorem, hard and soft designs, the lens traced as worn and seen by the eye of {doc}`454 Eye aberrometer <notebooks/454_eye_aberrometer>`), {doc}`434 Scan lenses <notebooks/434_scan_lenses>` (an F-theta and a telecentric scan lens with damped least squares), {doc}`422 Hybrid diffractive lenses <notebooks/422_hybrid_diffractive_lenses>` (kinoforms in the ray tracer, a hybrid achromat and apochromat and a thermal hybrid singlet), {doc}`411 Multi-configuration optimization <notebooks/411_multiconfig_optimization>` (one prescription for several zoom positions or object distances, after {doc}`410 Lens optimization <notebooks/410_lens_optimization>` and {doc}`431 Zoom lenses <notebooks/431_zoom_lenses>`) and {doc}`412 Global optimization and saddle points <notebooks/412_global_optimization_saddle_points>` (the many local minima of a Cooke triplet, the saddle points between them, Bociort's saddle-point construction from doublets to triplets, basin hopping against random starts), and {doc}`436 Off-axis mirrors and nodal aberrations <notebooks/436_off_axis_mirrors_nodal_aberrations>` (off-axis segments, nodal aberration theory of a misaligned Ritchey–Chrétien with constant coma and binodal astigmatism, an unobscured three-mirror anastigmat with ring fields and freeform mirrors, after {doc}`900 Telescope zoo <notebooks/900_telescope_zoo>`).

**From rays to waves:** {doc}`451 Lens focus from rays to waves <notebooks/451_lens_focus_waves>`, {doc}`450 Rays and waves <notebooks/450_rays_and_waves>`, {doc}`103 Analytic phase propagation <notebooks/103_analytic_phase_propagation>` (propagating strongly curved
wavefronts), {doc}`452 Gaussian beam decomposition <notebooks/452_gaussian_beam_decomposition>`.
**Polarization and coatings:** {doc}`406 Polarization ray tracing <notebooks/406_polarization_raytrace>`, {doc}`203 Coated laser mirror <notebooks/203_coated_laser_mirror>`.
**Broadband light:** {doc}`117 Ultrashort pulses <notebooks/117_ultrashort_pulses_lenses>`.
**Camera lenses:** {doc}`611 Bokeh <notebooks/611_bokeh>` (a double Gauss, vignetting, out-of-focus rendering), {doc}`612 Sunstars <notebooks/612_sunstars>` (diffraction at the iris), {doc}`613 Flare and ghosts <notebooks/613_flare_ghosts>` (double reflections, coatings), {doc}`610 The sensor <notebooks/610_sensor>` (pixels, Bayer mosaic, low pass filter, microlenses), {doc}`614 Depth of field and tilt-shift <notebooks/614_tilt_shift_depth_of_field>` (hyperfocal distance, optimum aperture, Scheimpflug), {doc}`616 The light field camera <notebooks/616_light_field_camera>` (microlenses, views, refocusing), {doc}`601 Radiometry of images <notebooks/601_radiometry_of_images>` (the camera equation, cos⁴ and how a retrofocus lens beats it, exposure and photons per pixel). {doc}`306 Image restoration and PSF deconvolution <notebooks/306_image_restoration_deconvolution>` (field and colour dependent PSFs of a phone lens, Wiener, Richardson–Lucy, TV and cross-channel deconvolution, the Bayer mosaic, wrong PSFs and a learned unrolled solver), {doc}`615 Distortion, lateral colour and lens shading correction <notebooks/615_distortion_shading_correction>` (the distortion of the phone lens and its warp, a warp per colour channel, warp before or after the restoration, relative illumination, chief ray angle and microlens shift, colour shading from the IR cut filter, ISP gain tables).

## Lithography and semiconductor engineer

For readers who work with scanners, masks or process windows. The core
builds up an imaging simulation one part at a time.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`300 Abbe image formation <notebooks/300_abbe_image_formation>` | Abbe's method, partial coherence and the pupil cutoff. |
| 2 | {doc}`302 PSF, MTF, Strehl <notebooks/302_psf_mtf_strehl>` | Image quality measures used throughout. |
| 3 | {doc}`800 Vectorial imaging in lithography <notebooks/800_vectorial_imaging_lithography>` | Hyper-NA imaging: s, p and unpolarized lines and spaces. |
| 4 | {doc}`801 Lithography toolbox <notebooks/801_lithography_toolbox>` | The litho toolbox: sources, masks, stacks, process windows. |
| 5 | {doc}`802 Hopkins imaging and SOCS <notebooks/802_hopkins_socs>` | Hopkins imaging, TCCs and SOCS for fast mask iterations. |
| 6 | {doc}`804 Resist and swing curves <notebooks/804_litho_resist_swing>` | A chemically amplified resist and swing curves. |
| 7 | {doc}`810 Rigorous mask (RCWA) <notebooks/810_litho_rigorous_mask>` | Rigorous mask diffraction with RCWA, mask 3D effects. |
| 8 | {doc}`811 Mask models compared <notebooks/811_litho_mask_models>` | Kirchhoff, film, boundary layer and RCWA side by side. |
| 9 | {doc}`812 FDTD mask for lithography <notebooks/812_litho_fdtd_mask>` | An FDTD mask for lines and contact holes. |
| 10 | {doc}`821 Jones pupils <notebooks/821_litho_jones_pupil>` | Polarization of the projection lens as a Jones pupil. |

**The projection lens and the scanner:** {doc}`451 Lens focus from rays to waves <notebooks/451_lens_focus_waves>`, {doc}`820 Lens aberrations across the slit <notebooks/820_litho_lens_aberrations>`, {doc}`822 Telecentricity across the slit <notebooks/822_litho_telecentricity>`, {doc}`403 Magnification, distortion, telecentricity <notebooks/403_magnification_distortion_telecentricity>`
(overlay), {doc}`405 Delano diagrams <notebooks/405_litho_delano_lens_design>`, {doc}`401 Seidel aberrations <notebooks/401_seidel_caldwell_lens>`, {doc}`413 Monte Carlo tolerancing <notebooks/413_monte_carlo_tolerancing>`, {doc}`824 Lens tolerances and wafer yield <notebooks/824_lens_tolerances_wafer_yield>` (from lens tolerances to CD uniformity, best focus and overlay, and what the manipulators recover), {doc}`823 KrF illuminator <notebooks/823_litho_illuminator>`, {doc}`605 Homogenizers <notebooks/605_homogenizers>`.
**Computational lithography:** {doc}`830 Source optimization <notebooks/830_litho_source_optimization>`, then {doc}`831 OPC <notebooks/831_litho_opc>`, then {doc}`805 Arbitrary mask layouts <notebooks/805_litho_arbitrary_layouts>` (polygons at any angle, layouts from gdstk/gdspy).

**Beyond the mean image:** {doc}`832 Litho stochastics, flare and focus blur <notebooks/832_litho_stochastics_flare_focus_blur>` (photon and acid shot noise, LER, LWR and their spectra, local CDU and missing contacts; flare from lens roughness and pattern density; focus blur from the laser bandwidth and the stage), after {doc}`804 Resist and swing curves <notebooks/804_litho_resist_swing>`.

**EUV:** {doc}`840 EUV multilayers and reflective masks <notebooks/840_euv_multilayers_reflective_masks>` (Mo/Si Bragg mirrors with Ru cap, their Jones pupil; a TaBN or low-n absorber at a 6° chief ray: shadowing, best focus through pitch, pattern shift, telecentricity) → {doc}`841 EUV projector and pupil <notebooks/841_euv_projector_pupil>` (NA 0.33 against anamorphic NA 0.55, central obscuration and forbidden pitches, a Schwarzschild mirror objective with graded coatings) → {doc}`842 EUV stochastics and flare <notebooks/842_euv_stochastics_flare>` (13.5 nm photon shot noise, secondary electron blur, Z factor, contact LCDU and flare scaling as 1/λ²), after {doc}`832 Litho stochastics, flare and focus blur <notebooks/832_litho_stochastics_flare_focus_blur>`.
**Wafer metrology:** {doc}`520 Ellipsometry <notebooks/520_ellipsometry>` (film thicknesses and optical constants), then {doc}`521 Scatterometry <notebooks/521_scatterometry>` (CD, height and sidewall angle of printed lines with RCWA).

## Laser and beam delivery engineer

For readers who build laser heads, fibre couplings or beam shapers.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`110 Gaussian beams <notebooks/110_gaussian_beams>` | Gaussian and higher order beams. |
| 2 | {doc}`112 Beam quality, M² <notebooks/112_beam_quality>` | Beam widths and M² after ISO 11146. |
| 3 | {doc}`111 ABCD matrices <notebooks/111_abcd_matrices>` | Beam propagation and resonator stability. |
| 4 | {doc}`113 Mode analysis <notebooks/113_mode_analysis>` | Mode content and fibre coupling. |
| 5 | {doc}`452 Gaussian beam decomposition <notebooks/452_gaussian_beam_decomposition>` | A spatial filter with Gaussian beam decomposition. |
| 6 | {doc}`203 Coated laser mirror <notebooks/203_coated_laser_mirror>` | A coated 45° fold mirror: retardance and its effect on the focus. |
| 7 | {doc}`534 IFTA beam shaping <notebooks/534_ifta_beam_shaping>` | Designing a phase-only DOE with IFTA. |
| 8 | {doc}`608 Freeform beam shaping <notebooks/608_freeform_beam_shaping>` | Refractive flat-top shapers and freeform lenses. |
| 9 | {doc}`453 Laser head DOE <notebooks/453_laser_head_doe>` | A diffractive 5 × 5 spot splitter in a processing head. |
| 10 | {doc}`434 Scan lenses <notebooks/434_scan_lenses>` | F-theta and telecentric scan lenses behind two galvos, the bow of the scan lines and its lookup table. |
| 11 | {doc}`115 Bessel and Airy beams <notebooks/115_bessel_airy_self_healing>` | Bessel and Airy beams. |
| 12 | {doc}`116 Optical vortices <notebooks/116_optical_vortices>` | Vortex beams and mode converters. |
| 13 | {doc}`117 Ultrashort pulses <notebooks/117_ultrashort_pulses_lenses>` | Femtosecond pulses through lenses, prisms and gratings. |
| 14 | {doc}`314 Optical tweezers <notebooks/314_optical_tweezers>` | Optical forces in a tight focus. |

**Side trips:** {doc}`104 Vectorial diffraction <notebooks/104_vectorial_diffraction>` (high NA foci), {doc}`102 Tilted planes <notebooks/102_tilted_planes>`, {doc}`101 Chirp-z transform <notebooks/101_chirp_z_transform>`; for fibre delivery,
{doc}`720 Light in a fibre <notebooks/720_fibre_rays_step_graded>` (NA, étendue and bend loss of multimode fibres), {doc}`721 Fibre modes <notebooks/721_fibre_modes>`
(single-mode condition and mode field diameter) and {doc}`724 Fibre Bragg gratings <notebooks/724_fibre_bragg_gratings>` (gratings that stabilize lasers).

## Metrology, microscopy and computational imaging

For readers who measure wavefronts or build imaging systems that rely on
computation. Three shorter paths share this section.

**Measuring wavefronts and surfaces**

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`305 Image processing <notebooks/305_image_processing>` | Camera noise, quantization and centering. |
| 2 | {doc}`501 Phase unwrapping <notebooks/501_phase_unwrapping>` | Phase unwrapping. |
| 3 | {doc}`500 Interferometry <notebooks/500_interferometry>` | Phase shifting and Fourier fringe analysis. |
| 4 | {doc}`502 Radius interferometer <notebooks/502_radius_interferometer>` | Radius of curvature with a Fizeau interferometer. |
| 5 | {doc}`510 Shack-Hartmann vs phase diversity <notebooks/510_shack_hartmann_vs_phase_diversity>` | Shack-Hartmann sensor against phase diversity. |
| 6 | {doc}`454 Eye aberrometer <notebooks/454_eye_aberrometer>` | A Shack-Hartmann aberrometer behind a schematic eye. |
| 7 | {doc}`532 Phase retrieval <notebooks/532_phase_retrieval>` | Fienup's algorithms and phase diversity. |
| 8 | {doc}`533 TIE <notebooks/533_tie>` | The transport-of-intensity equation. |
| 9 | {doc}`530 Digital holography <notebooks/530_digital_holography>` | Digital holography. |
| 10 | {doc}`531 Holographic microscopy <notebooks/531_holographic_microscopy>` | Holographic microscopy, combining notebooks 501 (phase unwrapping), 530 (digital holography) and 533 (TIE). |
| 11 | {doc}`511 Schlieren and knife-edge <notebooks/511_schlieren_knife_edge>` | The knife-edge test. |
| 12 | {doc}`505 OCT <notebooks/505_optical_coherence_tomography>` | Depth from coherence: OCT. |

**Microscopy:** {doc}`300 Abbe image formation <notebooks/300_abbe_image_formation>`, {doc}`310 Phase contrast and DIC <notebooks/310_phase_contrast_dic>`, {doc}`311 Confocal microscopy <notebooks/311_confocal_microscopy>`, {doc}`604 Köhler and critical illumination <notebooks/604_koehler_critical_illumination>`, {doc}`313 Super-resolution microscopy <notebooks/313_super_resolution_microscopy>`, {doc}`312 Fourier ptychography <notebooks/312_fourier_ptychography>`.

**Scattering, turbulence and lensless imaging:** {doc}`535 Wavefront shaping <notebooks/535_wavefront_shaping>`, {doc}`903 Twinkling stars and AO <notebooks/903_twinkling_stars_adaptive_optics>`, {doc}`536 Lensless imaging <notebooks/536_lensless_coded_aperture_imaging>`.

**Spectroscopy:** {doc}`910 Grating and prism spectrometers <notebooks/910_grating_prism_spectrometers>` (dispersive spectrometers, the traced Czerny–Turner, grating efficiency), then {doc}`911 Fourier transform spectroscopy <notebooks/911_fourier_transform_spectroscopy>` (the FTIR: apodization, sampling, phase correction, Jacquinot and Fellgett), then {doc}`913 Spectral lines and absorption <notebooks/913_spectral_lines_absorption>` (line shapes, the curve of growth, and why concentrations from unresolved lines go wrong); {doc}`912 Fabry–Perot and filters <notebooks/912_fabry_perot_filters>` for etalons and filters and {doc}`915 Imaging spectroscopy <notebooks/915_imaging_spectroscopy>` for pushbroom, filter and snapshot imaging spectrometers.

**Astronomical instruments:** {doc}`904 Coronagraphs <notebooks/904_coronagraphs>` (Lyot, APLC and vortex coronagraphs, pointing and aberration sensitivity, a dark hole with a deformable mirror), {doc}`900 Telescope zoo <notebooks/900_telescope_zoo>` (Newtonian, Cassegrain, Ritchey–Chrétien and Schmidt camera against the Seidel and Schwarzschild formulas), {doc}`901 Segmented mirrors <notebooks/901_segmented_mirrors>` (hexagon diffraction, piston and tip-tilt errors, dispersed fringe sensing and phase retrieval), {doc}`905 Einstein rings <notebooks/905_einstein_rings>` (gravitational lensing as a GRIN medium and as a thin lens: rings, microlensing, time delays and caustics), {doc}`914 Echelle spectrograph <notebooks/914_echelle_spectrograph>` (grating equation in high orders, blaze, free spectral range, cross-dispersion, the echellogram and the resolving power against the slit), {doc}`902 Atmospheric dispersion corrector <notebooks/902_atmospheric_dispersion_corrector>` (refraction and dispersion of air against the zenith angle, counter-rotating Amici prisms, a glass pair matched to air, the residual against the zenith angle).

## Nanophotonics and computational electromagnetics

For readers who want to know which solver to use when, and where the
cheaper ones stop being valid.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`000 Foundations and conventions <notebooks/000_foundations_conventions>` | The ladder of approximations and the modules for each rung. |
| 2 | {doc}`100 Free-space propagation <notebooks/100_propagation>` | FFT propagation in homogeneous media. |
| 3 | {doc}`105 Wave propagation method <notebooks/105_wave_propagation_method>` | The wave propagation method in inhomogeneous media. |
| 4 | {doc}`202 Stacks as transfer functions <notebooks/202_stack_transfer_functions>` | Layer stacks as transfer functions for beams. |
| 5 | {doc}`220 FDTD basics <notebooks/220_fdtd_basics>` | The Yee grid, absorbing boundaries, first FDTD runs. |
| 6 | {doc}`221 FDTD metals and plasmons <notebooks/221_fdtd_metals_plasmons>` | Dispersive metals and surface plasmons. |
| 7 | {doc}`223 FDTD photonic circuits <notebooks/223_fdtd_photonic_circuits>` | Waveguides, couplers and ring resonators. |
| 8 | {doc}`222 FDTD periodic structures <notebooks/222_fdtd_periodic_structures>` | Bloch boundaries, gratings and band structures. |
| 9 | {doc}`224 FDTD devices <notebooks/224_fdtd_devices>` | Photonic crystal devices, a grating coupler and a metalens. |
| 10 | {doc}`225 3D FDTD and nanojet <notebooks/225_fdtd_3d_nanojet>` | 3D FDTD against Mie theory. |
| 11 | {doc}`230 FDFD basics <notebooks/230_fdfd_basics>` | FDFD: one frequency at a time. |
| 12 | {doc}`231 Limaçon microcavity <notebooks/231_fdfd_limacon_microcavity>` | A deformed microcavity with FDFD. |
| 13 | {doc}`232 Phase space, Husimi and ray chaos <notebooks/232_phase_space_husimi_chaos>` | Wigner and Husimi functions; the modes of chaotic cavities against billiard rays. |
| 14 | {doc}`233 Carpet cloak <notebooks/233_carpet_cloak>` | Transformation optics. |

**Cross-checks between methods:** {doc}`810 Rigorous mask (RCWA) <notebooks/810_litho_rigorous_mask>`, {doc}`811 Mask models compared <notebooks/811_litho_mask_models>` and {doc}`812 FDTD mask for lithography <notebooks/812_litho_fdtd_mask>` (RCWA against
FDTD), {doc}`407 GRIN optics <notebooks/407_grin_optics>` (rays against FDFD in graded index media), {doc}`910 Grating and prism spectrometers <notebooks/910_grating_prism_spectrometers>` section 4 (RCWA efficiency of a blazed grating, TE and TM, Wood–Rayleigh anomalies).

## Applied chapters

The applied chapters are best read in order. Each one builds on a few
notebooks from the earlier parts; read those first, and take the side trips
afterwards.

| Chapter | Read first | Side trips |
|---|---|---|
| Radiometry and illumination (600–608) | {doc}`400 Ray tracing to OPD <notebooks/400_ray_tracing_opd>` for the ray tracer | {doc}`200 Interfaces and thin films <notebooks/200_interfaces_thin_films>` and {doc}`406 Polarization ray tracing <notebooks/406_polarization_raytrace>` (coatings and Fresnel losses), {doc}`421 Fresnel lenses <notebooks/421_fresnel_lenses>`, {doc}`823 KrF illuminator <notebooks/823_litho_illuminator>` (a complete illuminator, ray traced), {doc}`601 Radiometry of images <notebooks/601_radiometry_of_images>` |
| Lasers (700–703) | {doc}`111 ABCD matrices <notebooks/111_abcd_matrices>`, {doc}`113 Mode analysis <notebooks/113_mode_analysis>` | {doc}`117 Ultrashort pulses <notebooks/117_ultrashort_pulses_lenses>`, {doc}`913 Spectral lines and absorption <notebooks/913_spectral_lines_absorption>` (line shapes behind the cross sections), {doc}`912 Fabry–Perot and filters <notebooks/912_fabry_perot_filters>` (the resonator as a filter), {doc}`712 Kerr effect and self-focusing <notebooks/712_kerr_effect_self_focusing>` (section 6: Kerr-lens mode-locking) |
| Nonlinear optics (710–712) | {doc}`211 Crystal optics <notebooks/211_crystal_optics>` (index ellipsoid and walk-off) | {doc}`702 Q-switching and mode-locking <notebooks/702_q_switching_mode_locking>` (the master equation behind Kerr-lens mode-locking), {doc}`117 Ultrashort pulses <notebooks/117_ultrashort_pulses_lenses>`, {doc}`722 Dispersion in fibres <notebooks/722_fibre_dispersion>` (the β₂ that balances SPM in a soliton), {doc}`726 SPM and solitons in fibres <notebooks/726_fibre_spm_solitons>` |
| Fibre optics (720–726) | {doc}`113 Mode analysis <notebooks/113_mode_analysis>` (LP modes and fibre coupling) | {doc}`200 Interfaces and thin films <notebooks/200_interfaces_thin_films>` (total internal reflection and its phase), {doc}`407 GRIN optics <notebooks/407_grin_optics>`, {doc}`223 FDTD photonic circuits <notebooks/223_fdtd_photonic_circuits>` (waveguides on a chip), {doc}`117 Ultrashort pulses <notebooks/117_ultrashort_pulses_lenses>`, {doc}`203 Coated laser mirror <notebooks/203_coated_laser_mirror>` (Bragg mirrors), {doc}`463 Field tracing applications <notebooks/463_field_tracing_applications>` (coupling into a single-mode fibre) |
| Detectors (730) | {doc}`600 Radiometry and photometry <notebooks/600_radiometry_photometry>`, {doc}`725 Fibre link <notebooks/725_fibre_link>` (receiver noise) | {doc}`911 Fourier transform spectroscopy <notebooks/911_fourier_transform_spectroscopy>` (the multiplex advantage), {doc}`120 Coherence and stellar interferometry <notebooks/120_coherence_stellar_interferometry>` (intensity interferometry after Hanbury Brown and Twiss), {doc}`711 Parametric amplification and OPOs <notebooks/711_parametric_amplification_opo>` (photon pairs), {doc}`930 Time-of-flight LiDAR <notebooks/930_time_of_flight_lidar>`, {doc}`305 Image processing <notebooks/305_image_processing>` |
| Polarization, crystals and modulators (210–214) | {doc}`200 Interfaces and thin films <notebooks/200_interfaces_thin_films>` | {doc}`725 Fibre link <notebooks/725_fibre_link>` (the modulator in a 10 Gb/s link), {doc}`915 Imaging spectroscopy <notebooks/915_imaging_spectroscopy>` (LC tunable filter and AOTF), {doc}`922 Pancake lenses <notebooks/922_pancake_lenses>` (real retarders in a VR lens), {doc}`406 Polarization ray tracing <notebooks/406_polarization_raytrace>` (crystals in a sequential ray trace), {doc}`912 Fabry–Perot and filters <notebooks/912_fabry_perot_filters>` (birefringent filters) |
| Interferometry (500–505) and polarimetric metrology (520–521) | {doc}`200 Interfaces and thin films <notebooks/200_interfaces_thin_films>`, {doc}`305 Image processing <notebooks/305_image_processing>` | {doc}`810 Rigorous mask <notebooks/810_litho_rigorous_mask>` (the RCWA behind scatterometry), {doc}`512 Deflectometry <notebooks/512_deflectometry>` and {doc}`933 Triangulation and structured light <notebooks/933_triangulation_structured_light>` (surface slopes and fringe projection), {doc}`420 Forbes Q aspheres <notebooks/420_forbes_q_aspheres>` (the polynomials deflectometry measures) |
| AR/VR and head-up displays (920–923) | {doc}`200 Interfaces and thin films <notebooks/200_interfaces_thin_films>`, {doc}`210 Jones and Mueller calculus <notebooks/210_jones_mueller_calculus>` | {doc}`810 Rigorous mask <notebooks/810_litho_rigorous_mask>` (the method behind the grating efficiencies), {doc}`910 Grating and prism spectrometers <notebooks/910_grating_prism_spectrometers>` (Coddington astigmatism in a Czerny–Turner), {doc}`421 Fresnel lenses <notebooks/421_fresnel_lenses>` (the VR lens before pancakes), {doc}`602 Étendue <notebooks/602_etendue_brightness_theorem>` (why a small projector cannot fill a large eye box) |
| 3D sensing and LiDAR (930–933) | {doc}`600 Radiometry and photometry <notebooks/600_radiometry_photometry>`, {doc}`730 Detectors and photon statistics <notebooks/730_detectors_photon_statistics>` | {doc}`725 Fibre link <notebooks/725_fibre_link>` (APD noise and Q factors), {doc}`602 Étendue <notebooks/602_etendue_brightness_theorem>` (why a single mode receiver sees no sunlight), {doc}`606 Speckle reduction <notebooks/606_speckle_reduction>`, {doc}`500 Interferometry <notebooks/500_interferometry>` and {doc}`501 Phase unwrapping <notebooks/501_phase_unwrapping>` (behind fringe projection), {doc}`534 IFTA beam shaping <notebooks/534_ifta_beam_shaping>` (the fan-out grating) |

## Developer extending pyoptics

For readers who want to add a module or a notebook. These notebooks show
how the existing code is organised and which conventions new code has to
keep.

| Step | Notebook | Why |
|---|---|---|
| 1 | {doc}`000 Foundations and conventions <notebooks/000_foundations_conventions>` | The conventions every module follows, checked numerically. Read with {doc}`conventions`. |
| 2 | {doc}`100 Free-space propagation <notebooks/100_propagation>` | Grids, sampling and the basic propagators. |
| 3 | {doc}`101 Chirp-z transform <notebooks/101_chirp_z_transform>` | Zoomed and shifted transforms. |
| 4 | {doc}`103 Analytic phase propagation <notebooks/103_analytic_phase_propagation>` | Splitting off an analytic phase to save samples. |
| 5 | {doc}`304 Basis functions <notebooks/304_basis_functions>` | The common interface of the basis sets. |
| 6 | {doc}`400 Ray tracing to OPD <notebooks/400_ray_tracing_opd>` | The data model of the ray tracer. |
| 7 | {doc}`801 Lithography toolbox <notebooks/801_lithography_toolbox>` | A simulation built from exchangeable components. |
| 8 | {doc}`802 Hopkins imaging and SOCS <notebooks/802_hopkins_socs>` | Abbe against Hopkins: caching what does not change. |
| 9 | {doc}`220 FDTD basics <notebooks/220_fdtd_basics>` | The layout of a time domain solver. |
| 10 | {doc}`230 FDFD basics <notebooks/230_fdfd_basics>` | The same grid as a sparse linear system. |
| 11 | {doc}`305 Image processing <notebooks/305_image_processing>` | Turning simulated fields into realistic measurements. |

See also {doc}`modules` for the maturity of each module.
