# Examples

The example notebooks illustrate the modules with examples whose physics
is easy to follow, and compare the numerics with analytic results wherever
an analytic solution exists. They are shown here with their stored outputs;
to run them yourself, see [Running the notebooks](#running-the-notebooks).
For suggested reading orders by background and interest, see
{doc}`learning_paths`.

## Foundations and conventions

From Maxwell's equations down to rays, with the modules for each level of approximation and numerical checks of all sign conventions. A first course in geometrical optics (images, principal planes, simple instruments, Seidel aberrations and wavefront error) for readers new to optics.

- {doc}`notebooks/000_foundations_conventions`
- {doc}`notebooks/001_geometrical_optics_intro`

```{toctree}
:hidden:
:caption: Foundations

notebooks/000_foundations_conventions
notebooks/001_geometrical_optics_intro
```

## Propagation and diffraction

Scalar and vectorial propagation of fields between planes, onto tilted planes and through inhomogeneous media.

- {doc}`notebooks/100_propagation`
- {doc}`notebooks/104_vectorial_diffraction`
- {doc}`notebooks/102_tilted_planes`
- {doc}`notebooks/101_chirp_z_transform`
- {doc}`notebooks/105_wave_propagation_method`
- {doc}`notebooks/103_analytic_phase_propagation`

```{toctree}
:hidden:
:caption: Propagation and diffraction

notebooks/100_propagation
notebooks/104_vectorial_diffraction
notebooks/102_tilted_planes
notebooks/101_chirp_z_transform
notebooks/105_wave_propagation_method
notebooks/103_analytic_phase_propagation
```

## Maxwell solvers: FDTD and FDFD

Full-wave solutions of Maxwell's equations on the Yee grid, in the time domain (`fdtd`) and one frequency at a time (`fdfd`).

- {doc}`notebooks/220_fdtd_basics`
- {doc}`notebooks/221_fdtd_metals_plasmons`
- {doc}`notebooks/223_fdtd_photonic_circuits`
- {doc}`notebooks/222_fdtd_periodic_structures`
- {doc}`notebooks/224_fdtd_devices`
- {doc}`notebooks/225_fdtd_3d_nanojet`
- {doc}`notebooks/230_fdfd_basics`
- {doc}`notebooks/231_fdfd_limacon_microcavity`
- {doc}`notebooks/232_phase_space_husimi_chaos`
- {doc}`notebooks/233_carpet_cloak`
- {doc}`notebooks/812_litho_fdtd_mask` (in *Lithography*)

```{toctree}
:hidden:
:caption: Maxwell solvers: FDTD and FDFD

notebooks/220_fdtd_basics
notebooks/221_fdtd_metals_plasmons
notebooks/223_fdtd_photonic_circuits
notebooks/222_fdtd_periodic_structures
notebooks/224_fdtd_devices
notebooks/225_fdtd_3d_nanojet
notebooks/230_fdfd_basics
notebooks/231_fdfd_limacon_microcavity
notebooks/232_phase_space_husimi_chaos
notebooks/233_carpet_cloak
```

## Beams, modes and paraxial optics

Gaussian and structured beams, beam quality, resonators, modes and ultrashort pulses.

- {doc}`notebooks/110_gaussian_beams`
- {doc}`notebooks/112_beam_quality`
- {doc}`notebooks/111_abcd_matrices`
- {doc}`notebooks/113_mode_analysis`
- {doc}`notebooks/115_bessel_airy_self_healing`
- {doc}`notebooks/116_optical_vortices`
- {doc}`114 Ince-Gaussian beams <notebooks/114_ince_gaussian_beams>`
- {doc}`notebooks/117_ultrashort_pulses_lenses`

```{toctree}
:hidden:
:caption: Beams, modes and paraxial optics

notebooks/110_gaussian_beams
notebooks/112_beam_quality
notebooks/111_abcd_matrices
notebooks/113_mode_analysis
notebooks/115_bessel_airy_self_healing
notebooks/116_optical_vortices
notebooks/114_ince_gaussian_beams
notebooks/117_ultrashort_pulses_lenses
```

## Fibre optics and optical communication

Light guided in glass fibres and the basics of sending bits through them:
the acceptance angle and rays in step and graded-index cores, fibre modes
and the single-mode condition, material, waveguide and intermodal
dispersion with real GeO₂–SiO₂ data, pulse broadening and the bit-rate
limit, fibre Bragg gratings, a basic link with its receiver noise, eye
diagrams and budgets, and self-phase modulation, solitons and
supercontinuum. Related: LP modes and fibre coupling in
{doc}`notebooks/113_mode_analysis`, waveguides in
{doc}`notebooks/223_fdtd_photonic_circuits`, GRIN rods in
{doc}`notebooks/407_grin_optics` and pulses in glass in
{doc}`notebooks/117_ultrashort_pulses_lenses`.

- {doc}`notebooks/720_fibre_rays_step_graded`
- {doc}`notebooks/721_fibre_modes`
- {doc}`notebooks/722_fibre_dispersion`
- {doc}`notebooks/723_fibre_pulse_broadening`
- {doc}`notebooks/724_fibre_bragg_gratings`
- {doc}`notebooks/725_fibre_link`
- {doc}`726 SPM and solitons in fibres <notebooks/726_fibre_spm_solitons>`

```{toctree}
:hidden:
:caption: Fibre optics and optical communication

notebooks/720_fibre_rays_step_graded
notebooks/721_fibre_modes
notebooks/722_fibre_dispersion
notebooks/723_fibre_pulse_broadening
notebooks/724_fibre_bragg_gratings
notebooks/725_fibre_link
notebooks/726_fibre_spm_solitons
```

## Thin films, coatings and polarization

Fresnel coefficients, layer stacks and coatings, Jones and Mueller calculus.

- {doc}`notebooks/200_interfaces_thin_films`
- {doc}`notebooks/210_jones_mueller_calculus`
- {doc}`notebooks/201_layer_stacks`
- {doc}`notebooks/202_stack_transfer_functions`
- {doc}`notebooks/203_coated_laser_mirror`

```{toctree}
:hidden:
:caption: Thin films, coatings and polarization

notebooks/200_interfaces_thin_films
notebooks/210_jones_mueller_calculus
notebooks/201_layer_stacks
notebooks/202_stack_transfer_functions
notebooks/203_coated_laser_mirror
```

## Imaging, aberrations and detection

Pupils and Zernike polynomials, PSF, MTF and Strehl ratio, detectors and lensless imaging.

- {doc}`notebooks/301_imaging_zernikes`
- {doc}`notebooks/304_basis_functions`
- {doc}`notebooks/302_psf_mtf_strehl`
- {doc}`notebooks/305_image_processing`
- {doc}`notebooks/303_zernike_aberration_gallery`
- {doc}`notebooks/536_lensless_coded_aperture_imaging`

```{toctree}
:hidden:
:caption: Imaging, aberrations and detection

notebooks/301_imaging_zernikes
notebooks/304_basis_functions
notebooks/302_psf_mtf_strehl
notebooks/305_image_processing
notebooks/303_zernike_aberration_gallery
notebooks/536_lensless_coded_aperture_imaging
```

## Ray tracing and lens design

Sequential and polarization ray tracing, from rays to waves, glass catalogues, lens optimization and tolerancing, and what a lens must conserve: the Lagrange invariant, the sine and the Herschel condition, and field tracing: rays through the lenses, waves where diffraction matters. Three lens design classics close the section: a mechanically compensated zoom from a patent, Forbes Q aspheres, and a progressive spectacle lens seen by a model eye.

- {doc}`notebooks/400_ray_tracing_opd`
- {doc}`notebooks/406_polarization_raytrace`
- {doc}`notebooks/451_lens_focus_waves`
- {doc}`notebooks/450_rays_and_waves`
- {doc}`notebooks/453_laser_head_doe`
- {doc}`notebooks/454_eye_aberrometer`
- {doc}`notebooks/421_fresnel_lenses`
- {doc}`notebooks/452_gaussian_beam_decomposition`
- {doc}`notebooks/410_lens_optimization`
- {doc}`notebooks/404_glass_catalogues`
- {doc}`notebooks/413_monte_carlo_tolerancing`
- {doc}`notebooks/407_grin_optics`
- {doc}`notebooks/402_sine_herschel_lagrange`
- {doc}`notebooks/460_field_tracing`
- {doc}`notebooks/463_field_tracing_applications`
- {doc}`462 Local plane interface approximation <notebooks/462_local_plane_interface_approximation>`
- {doc}`notebooks/431_zoom_lenses`
- {doc}`notebooks/420_forbes_q_aspheres`
- {doc}`notebooks/435_progressive_lenses`
- {doc}`notebooks/434_scan_lenses`
- {doc}`422 Hybrid diffractive lenses <notebooks/422_hybrid_diffractive_lenses>`
- {doc}`411 Multi-configuration optimization <notebooks/411_multiconfig_optimization>`
- {doc}`412 Global optimization and saddle points <notebooks/412_global_optimization_saddle_points>`
- {doc}`436 Off-axis mirrors and nodal aberrations <notebooks/436_off_axis_mirrors_nodal_aberrations>`
- {doc}`430 Phone camera lens with Forbes Q polynomials <notebooks/430_phone_camera_lens_forbes_q>`
- {doc}`415 CAD export of lens systems <notebooks/415_cad_export_lens_systems>`
- {doc}`440 Non-sequential ray tracing <notebooks/440_non_sequential_ray_tracing>`
- {doc}`441 Stray light in Caldwell's lens <notebooks/441_stray_light_caldwell_lens>`
- {doc}`461 Source models in field tracing <notebooks/461_source_models_field_tracing>`
- {doc}`432 Eyepieces <notebooks/432_eyepieces>`
- {doc}`433 Binoculars and erecting prisms <notebooks/433_binoculars_erecting_prisms>`
- {doc}`414 Polishing errors <notebooks/414_polishing_errors>`

```{toctree}
:hidden:
:caption: Ray tracing and lens design

notebooks/400_ray_tracing_opd
notebooks/406_polarization_raytrace
notebooks/451_lens_focus_waves
notebooks/450_rays_and_waves
notebooks/453_laser_head_doe
notebooks/454_eye_aberrometer
notebooks/421_fresnel_lenses
notebooks/452_gaussian_beam_decomposition
notebooks/410_lens_optimization
notebooks/404_glass_catalogues
notebooks/413_monte_carlo_tolerancing
notebooks/407_grin_optics
notebooks/402_sine_herschel_lagrange
notebooks/460_field_tracing
notebooks/463_field_tracing_applications
notebooks/462_local_plane_interface_approximation
notebooks/431_zoom_lenses
notebooks/420_forbes_q_aspheres
notebooks/435_progressive_lenses
notebooks/434_scan_lenses
notebooks/422_hybrid_diffractive_lenses
notebooks/411_multiconfig_optimization
notebooks/412_global_optimization_saddle_points
notebooks/436_off_axis_mirrors_nodal_aberrations
notebooks/430_phone_camera_lens_forbes_q
notebooks/415_cad_export_lens_systems
notebooks/440_non_sequential_ray_tracing
notebooks/441_stray_light_caldwell_lens
notebooks/461_source_models_field_tracing
notebooks/432_eyepieces
notebooks/433_binoculars_erecting_prisms
notebooks/414_polishing_errors
```

## Interferometry, holography and wavefront sensing

Measuring phase: interferometers, coherence, holography, wavefront sensors and optical testing.

- {doc}`notebooks/501_phase_unwrapping`
- {doc}`notebooks/500_interferometry`
- {doc}`notebooks/502_radius_interferometer`
- {doc}`notebooks/120_coherence_stellar_interferometry`
- {doc}`notebooks/510_shack_hartmann_vs_phase_diversity`
- {doc}`notebooks/530_digital_holography`
- {doc}`notebooks/531_holographic_microscopy`
- {doc}`notebooks/511_schlieren_knife_edge`
- {doc}`notebooks/505_optical_coherence_tomography`

```{toctree}
:hidden:
:caption: Interferometry, holography and wavefront sensing

notebooks/501_phase_unwrapping
notebooks/500_interferometry
notebooks/502_radius_interferometer
notebooks/120_coherence_stellar_interferometry
notebooks/510_shack_hartmann_vs_phase_diversity
notebooks/530_digital_holography
notebooks/531_holographic_microscopy
notebooks/511_schlieren_knife_edge
notebooks/505_optical_coherence_tomography
```

## Phase retrieval and beam shaping

Recovering phase from intensities, and shaping intensities with phase.

- {doc}`notebooks/532_phase_retrieval`
- {doc}`notebooks/533_tie`
- {doc}`notebooks/534_ifta_beam_shaping`
- {doc}`notebooks/535_wavefront_shaping`

```{toctree}
:hidden:
:caption: Phase retrieval and beam shaping

notebooks/532_phase_retrieval
notebooks/533_tie
notebooks/534_ifta_beam_shaping
notebooks/535_wavefront_shaping
```

## Microscopy

Image formation in the microscope, contrast methods, confocal and super-resolution microscopy, optical traps.

- {doc}`notebooks/300_abbe_image_formation`
- {doc}`notebooks/310_phase_contrast_dic`
- {doc}`notebooks/311_confocal_microscopy`
- {doc}`notebooks/312_fourier_ptychography`
- {doc}`notebooks/314_optical_tweezers`
- {doc}`notebooks/313_super_resolution_microscopy`
- {doc}`notebooks/531_holographic_microscopy` (in *Interferometry, holography and wavefront sensing*)

```{toctree}
:hidden:
:caption: Microscopy

notebooks/300_abbe_image_formation
notebooks/310_phase_contrast_dic
notebooks/311_confocal_microscopy
notebooks/312_fourier_ptychography
notebooks/314_optical_tweezers
notebooks/313_super_resolution_microscopy
```

## Lithography

Projection lithography from the mask to the resist: imaging models, mask models, lens and illumination design, source and mask optimization.

- {doc}`notebooks/800_vectorial_imaging_lithography`
- {doc}`notebooks/801_lithography_toolbox`
- {doc}`notebooks/820_litho_lens_aberrations`
- {doc}`notebooks/802_hopkins_socs`
- {doc}`notebooks/804_litho_resist_swing`
- {doc}`notebooks/810_litho_rigorous_mask`
- {doc}`notebooks/811_litho_mask_models`
- {doc}`notebooks/812_litho_fdtd_mask`
- {doc}`notebooks/821_litho_jones_pupil`
- {doc}`notebooks/830_litho_source_optimization`
- {doc}`notebooks/831_litho_opc`
- {doc}`notebooks/805_litho_arbitrary_layouts`
- {doc}`notebooks/822_litho_telecentricity`
- {doc}`notebooks/823_litho_illuminator`
- {doc}`notebooks/405_litho_delano_lens_design`
- {doc}`notebooks/401_seidel_caldwell_lens`
- {doc}`notebooks/403_magnification_distortion_telecentricity`
- {doc}`notebooks/803_litho_fourier_interpolation_cd`
- {doc}`824 Lens tolerances and wafer yield <notebooks/824_lens_tolerances_wafer_yield>`
- {doc}`832 Litho stochastics, flare and focus blur <notebooks/832_litho_stochastics_flare_focus_blur>`
- {doc}`840 EUV multilayers and reflective masks <notebooks/840_euv_multilayers_reflective_masks>`
- {doc}`841 EUV projector and pupil <notebooks/841_euv_projector_pupil>`
- {doc}`842 EUV stochastics and flare <notebooks/842_euv_stochastics_flare>`

```{toctree}
:hidden:
:caption: Lithography

notebooks/800_vectorial_imaging_lithography
notebooks/801_lithography_toolbox
notebooks/820_litho_lens_aberrations
notebooks/802_hopkins_socs
notebooks/804_litho_resist_swing
notebooks/810_litho_rigorous_mask
notebooks/811_litho_mask_models
notebooks/812_litho_fdtd_mask
notebooks/821_litho_jones_pupil
notebooks/830_litho_source_optimization
notebooks/831_litho_opc
notebooks/805_litho_arbitrary_layouts
notebooks/822_litho_telecentricity
notebooks/823_litho_illuminator
notebooks/405_litho_delano_lens_design
notebooks/401_seidel_caldwell_lens
notebooks/403_magnification_distortion_telecentricity
notebooks/803_litho_fourier_interpolation_cd
notebooks/824_lens_tolerances_wafer_yield
notebooks/832_litho_stochastics_flare_focus_blur
notebooks/840_euv_multilayers_reflective_masks
notebooks/841_euv_projector_pupil
notebooks/842_euv_stochastics_flare
```

## Illumination optics

Radiometry and photometry, étendue, nonimaging concentrators, microscope
illumination, homogenizers, speckle reduction, LED optics, freeform beam
shaping and the brightness of images.

- {doc}`notebooks/600_radiometry_photometry`
- {doc}`notebooks/602_etendue_brightness_theorem`
- {doc}`notebooks/603_nonimaging_concentrators`
- {doc}`notebooks/604_koehler_critical_illumination`
- {doc}`notebooks/605_homogenizers`
- {doc}`notebooks/606_speckle_reduction`
- {doc}`notebooks/607_led_optics_monte_carlo`
- {doc}`notebooks/608_freeform_beam_shaping`
- {doc}`notebooks/601_radiometry_of_images`

```{toctree}
:hidden:
:caption: Illumination optics

notebooks/600_radiometry_photometry
notebooks/602_etendue_brightness_theorem
notebooks/603_nonimaging_concentrators
notebooks/604_koehler_critical_illumination
notebooks/605_homogenizers
notebooks/606_speckle_reduction
notebooks/607_led_optics_monte_carlo
notebooks/608_freeform_beam_shaping
notebooks/601_radiometry_of_images
```

## Atmospheric optics

Rainbows, glories and the turbulent atmosphere.

- {doc}`notebooks/906_rainbows_and_glories`
- {doc}`notebooks/903_twinkling_stars_adaptive_optics`

```{toctree}
:hidden:
:caption: Atmospheric optics

notebooks/906_rainbows_and_glories
notebooks/903_twinkling_stars_adaptive_optics
```

## Photography

The optics of cameras: the out-of-focus image of a lens (bokeh), the
star its iris makes of a bright light (sunstars), the ghosts its
surfaces make of it (flare), what the sensor does to the image,
what is sharp (depth of field and tilt-shift), how a light field
camera focuses after the exposure, and how much of the blur of a phone
lens deconvolution can undo.

- {doc}`notebooks/611_bokeh`
- {doc}`notebooks/612_sunstars`
- {doc}`notebooks/613_flare_ghosts`
- {doc}`notebooks/610_sensor`
- {doc}`notebooks/614_tilt_shift_depth_of_field`
- {doc}`notebooks/616_light_field_camera`
- {doc}`306 Image restoration and PSF deconvolution <notebooks/306_image_restoration_deconvolution>`
- {doc}`615 Distortion, lateral colour and lens shading correction <notebooks/615_distortion_shading_correction>`

```{toctree}
:hidden:
:caption: Photography

notebooks/611_bokeh
notebooks/612_sunstars
notebooks/613_flare_ghosts
notebooks/610_sensor
notebooks/614_tilt_shift_depth_of_field
notebooks/616_light_field_camera
notebooks/306_image_restoration_deconvolution
notebooks/615_distortion_shading_correction
```

## Astronomy

Optics for imaging planets next to their stars: coronagraphs with
apodized pupils, occulting disks, vortex masks and Lyot stops, and a dark
hole dug with a deformable mirror; and the telescopes themselves: the
Newtonian, the classical Cassegrain, the Ritchey–Chrétien and the Schmidt
camera against the third order theory of conic mirrors, and segmented
primaries with their diffraction spikes and phasing; and gravity as a
lens: light bending traced through the index of a mass, Einstein rings,
microlensing light curves and the Einstein cross; and the echelle
spectrograph of the planet hunters, with its blaze, stacked orders and
resolving power; and the atmospheric dispersion corrector that undoes the
rainbow the air makes of every star.

- {doc}`notebooks/904_coronagraphs`
- {doc}`notebooks/900_telescope_zoo`
- {doc}`notebooks/901_segmented_mirrors`
- {doc}`notebooks/905_einstein_rings`
- {doc}`notebooks/914_echelle_spectrograph`
- {doc}`notebooks/902_atmospheric_dispersion_corrector`

```{toctree}
:hidden:
:caption: Astronomy

notebooks/904_coronagraphs
notebooks/900_telescope_zoo
notebooks/901_segmented_mirrors
notebooks/905_einstein_rings
notebooks/914_echelle_spectrograph
notebooks/902_atmospheric_dispersion_corrector
```

## Spectroscopy

Spectrometers and what they measure: prisms, gratings and the
Czerny–Turner traced with grating surfaces, grating efficiency and order
sorting; the Fourier transform spectrometer with apodization, laser
sampling, phase correction and the Jacquinot and Fellgett advantages;
Fabry–Perot etalons, interference filters and the Lyot filter; line
shapes, a model CO₂ band, the curve of growth and Beer–Lambert behind a
real instrument; and imaging spectrometers: smile and keystone of the
Offner and the Czerny–Turner, filter cameras and snapshot CASSI. The
echelle spectrograph is under Astronomy.

- {doc}`notebooks/910_grating_prism_spectrometers`
- {doc}`notebooks/911_fourier_transform_spectroscopy`
- {doc}`notebooks/912_fabry_perot_filters`
- {doc}`notebooks/913_spectral_lines_absorption`
- {doc}`notebooks/915_imaging_spectroscopy`
- {doc}`notebooks/914_echelle_spectrograph`

```{toctree}
:hidden:
:caption: Spectroscopy

notebooks/910_grating_prism_spectrometers
notebooks/911_fourier_transform_spectroscopy
notebooks/912_fabry_perot_filters
notebooks/913_spectral_lines_absorption
notebooks/915_imaging_spectroscopy
```

## AR/VR and head-up displays

The optics of augmented and virtual reality and of head-up displays:
waveguide combiners analysed in k-space, with slanted in-couplers from
RCWA, pupil replication over the eye box and the rainbow artefact; volume
holographic gratings with Kogelnik's coupled wave theory checked against
RCWA and thin films; pancake lenses that fold a VR eyepiece with
polarization, with real quartz retarders and their ghosts; and windshield
head-up displays with the double image, the wedge and a ray traced
freeform mirror.

- {doc}`920 AR waveguide combiners <notebooks/920_ar_waveguide_combiners>`
- {doc}`921 Volume holographic gratings <notebooks/921_volume_holographic_gratings>`
- {doc}`922 Pancake lenses <notebooks/922_pancake_lenses>`
- {doc}`923 Head-up displays <notebooks/923_head_up_displays>`

```{toctree}
:hidden:
:caption: AR/VR and head-up displays

notebooks/920_ar_waveguide_combiners
notebooks/921_volume_holographic_gratings
notebooks/922_pancake_lenses
notebooks/923_head_up_displays
```

## 3D sensing and LiDAR

How machines measure depth with light: the link budget of a
time-of-flight LiDAR against sunlight, detector noise and the class 1
eye safety limits that favour 1550 nm; FMCW LiDAR with its coherent
receiver, Doppler, speckle fading and the demands on the laser; the beam
scanners that sweep LiDARs over the scene; and triangulation, from laser
spot sensors and fringe projection to the dot projector of a phone's
depth camera.

- {doc}`930 Time-of-flight LiDAR <notebooks/930_time_of_flight_lidar>`
- {doc}`931 FMCW LiDAR <notebooks/931_fmcw_lidar>`
- {doc}`932 Beam scanning <notebooks/932_beam_scanning>`
- {doc}`933 Triangulation and structured light <notebooks/933_triangulation_structured_light>`

```{toctree}
:hidden:
:caption: 3D sensing and LiDAR

notebooks/930_time_of_flight_lidar
notebooks/931_fmcw_lidar
notebooks/932_beam_scanning
notebooks/933_triangulation_structured_light
```

## Surface and thin-film metrology

How fabs and optics shops measure surfaces and films: coherence scanning
(white-light) interferometry for surface heights from nanometres to
millimetres, spectroscopic ellipsometry and its model fits for film
thicknesses and optical constants, scatterometry of line gratings too
small to image, phase measuring deflectometry for the shape and the
ripples of specular freeforms, and the null test of aspheres with a
computer-generated hologram, from the hologram design to the surface form
deviation.

- {doc}`504 Coherence scanning interferometry (white light interferometry, WLI) <notebooks/504_coherence_scanning_interferometry>`
- {doc}`520 Ellipsometry <notebooks/520_ellipsometry>`
- {doc}`521 Scatterometry <notebooks/521_scatterometry>`
- {doc}`512 Deflectometry <notebooks/512_deflectometry>`
- {doc}`503 CGH null testing of aspheres <notebooks/503_cgh_null_testing_aspheres>`

```{toctree}
:hidden:
:caption: Surface and thin-film metrology

notebooks/504_coherence_scanning_interferometry
notebooks/520_ellipsometry
notebooks/521_scatterometry
notebooks/512_deflectometry
notebooks/503_cgh_null_testing_aspheres
```

## Crystal optics and light modulators

Light in anisotropic and gyrotropic media and the devices that switch,
steer and filter it: the index ellipsoid and double refraction, polarizing
prisms, optical activity and Faraday isolators, Pockels and Kerr cells and
the Mach–Zehnder modulator of every fibre link, liquid crystal cells from
the elastic theory of the director to the viewing angle of a TN display,
and acousto-optic modulators, deflectors and tunable filters. Related:
Lyot and Šolc filters with real plates in
{doc}`912 Fabry–Perot and filters <notebooks/912_fabry_perot_filters>`, a
real modulator in {doc}`725 Fibre link <notebooks/725_fibre_link>`, and an
LC tunable filter and an AOTF in
{doc}`915 Imaging spectroscopy <notebooks/915_imaging_spectroscopy>`.

- {doc}`211 Crystal optics <notebooks/211_crystal_optics>`
- {doc}`212 Electro-optic modulators <notebooks/212_electro_optic_modulators>`
- {doc}`213 Liquid crystals <notebooks/213_liquid_crystals>`
- {doc}`214 Acousto-optics <notebooks/214_acousto_optics>`

```{toctree}
:hidden:
:caption: Crystal optics and light modulators

notebooks/211_crystal_optics
notebooks/212_electro_optic_modulators
notebooks/213_liquid_crystals
notebooks/214_acousto_optics
```

## Lasers

How a laser works and what limits it: gain and saturation, the threshold
and the best output coupler, the modes of the resonator and the width of a
single one, relaxation oscillations, Q-switched and mode-locked pulses, the
laser diode of every fibre link with its chirp and noise, and the
transverse modes of real resonators with finite mirrors (Fox and Li). Related: the
resonator in {doc}`111 ABCD matrices <notebooks/111_abcd_matrices>`, pulses
in {doc}`117 Ultrashort pulses <notebooks/117_ultrashort_pulses_lenses>`, and
the directly modulated laser of {doc}`725 Fibre link <notebooks/725_fibre_link>`.

- {doc}`700 Laser fundamentals <notebooks/700_laser_fundamentals>`
- {doc}`702 Q-switching and mode-locking <notebooks/702_q_switching_mode_locking>`
- {doc}`703 Laser diodes <notebooks/703_laser_diodes>`
- {doc}`701 Laser resonator modes <notebooks/701_laser_resonator_modes>`

```{toctree}
:hidden:
:caption: Lasers

notebooks/700_laser_fundamentals
notebooks/702_q_switching_mode_locking
notebooks/703_laser_diodes
notebooks/701_laser_resonator_modes
```

## Nonlinear optics

Light that changes the medium it travels in: second harmonic generation
and phase matching in birefringent and periodically poled crystals,
parametric amplifiers and oscillators that turn one laser into a tunable
source, and the Kerr effect with self-focusing, self-phase modulation and
the Kerr lens of femtosecond lasers. Related: crystal optics in
{doc}`211 Crystal optics <notebooks/211_crystal_optics>`, mode-locked
lasers in {doc}`702 Q-switching and mode-locking <notebooks/702_q_switching_mode_locking>`,
and solitons and supercontinuum in
{doc}`726 SPM and solitons in fibres <notebooks/726_fibre_spm_solitons>`.

- {doc}`710 Second harmonic generation and phase matching <notebooks/710_second_harmonic_generation>`
- {doc}`711 Parametric amplification and OPOs <notebooks/711_parametric_amplification_opo>`
- {doc}`712 Kerr effect and self-focusing <notebooks/712_kerr_effect_self_focusing>`

```{toctree}
:hidden:
:caption: Nonlinear optics

notebooks/710_second_harmonic_generation
notebooks/711_parametric_amplification_opo
notebooks/712_kerr_effect_self_focusing
```

## Detectors and photon statistics

Where the numbers of a detector come from and what the photons
themselves do: responsivity and band gap cutoffs of Si, InGaAs and
HgCdTe, absorption depth and quantum efficiency, bandwidth against
capacitance, NEP and D* with the background limit of infrared detectors,
photon counting with SPADs and their dead time, Poisson, super- and
sub-Poissonian light, the beam splitter with photons and the
Hong–Ou–Mandel dip. Related: the receiver of a link in
{doc}`725 Fibre link <notebooks/725_fibre_link>`, SPAD histograms in
{doc}`930 Time-of-flight LiDAR <notebooks/930_time_of_flight_lidar>`,
the Fellgett advantage in
{doc}`911 Fourier transform spectroscopy <notebooks/911_fourier_transform_spectroscopy>`
and intensity interferometry in
{doc}`120 Coherence and stellar interferometry <notebooks/120_coherence_stellar_interferometry>`.

- {doc}`730 Detectors and photon statistics <notebooks/730_detectors_photon_statistics>`

```{toctree}
:hidden:
:caption: Detectors and photon statistics

notebooks/730_detectors_photon_statistics
```

## Running the notebooks

```{include} ../notebooks/README.md
:start-after: <!-- docs:running-start -->
:end-before: <!-- docs:running-end -->
```
