# Examples

The example notebooks illustrate the modules with examples whose physics
is easy to follow, and compare the numerics with analytic results wherever
an analytic solution exists. They are shown here with their stored outputs;
to run them yourself, see [Running the notebooks](#running-the-notebooks).

The notebooks are numbered in reading order: the hundreds digit is the
part, the tens digit the chapter, the last digit the place in the chapter.
Reading a chapter from its first notebook to its last is a good default; for
routes across the parts by background and interest, see {doc}`learning_paths`.

## 0xx Start here

Where everything fits, and a first course in rays for readers new to optics.

### Orientation (000–001)

From Maxwell's equations down to rays with the modules for each level of approximation and checks of all sign conventions, and a first course in geometrical optics.

- {doc}`000 Foundations and conventions <notebooks/000_foundations_conventions>`
- {doc}`001 Introduction to geometrical optics <notebooks/001_geometrical_optics_intro>`

```{toctree}
:hidden:
:caption: 0xx Start here

notebooks/000_foundations_conventions
notebooks/001_geometrical_optics_intro
```

## 1xx Waves: propagation, beams and coherence

Fields between planes, from the angular spectrum to tilted planes and inhomogeneous media; Gaussian and structured beams; coherence.

### Free-space propagation (100–105)

Scalar and vectorial propagation of fields between planes, onto tilted planes and through inhomogeneous media.

- {doc}`100 Free-space propagation <notebooks/100_propagation>`
- {doc}`101 Chirp-z transform <notebooks/101_chirp_z_transform>`
- {doc}`102 Tilted planes <notebooks/102_tilted_planes>`
- {doc}`103 Analytic phase propagation <notebooks/103_analytic_phase_propagation>`
- {doc}`104 Vectorial diffraction <notebooks/104_vectorial_diffraction>`
- {doc}`105 Wave propagation method <notebooks/105_wave_propagation_method>`

### Beams and modes (110–117)

Gaussian and structured beams, ABCD matrices and resonators, beam quality, mode decompositions and ultrashort pulses.

- {doc}`110 Gaussian beams <notebooks/110_gaussian_beams>`
- {doc}`111 ABCD matrices <notebooks/111_abcd_matrices>`
- {doc}`112 Beam quality <notebooks/112_beam_quality>`
- {doc}`113 Mode analysis <notebooks/113_mode_analysis>`
- {doc}`114 Ince-Gaussian beams <notebooks/114_ince_gaussian_beams>`
- {doc}`115 Bessel and Airy beams <notebooks/115_bessel_airy_self_healing>`
- {doc}`116 Optical vortices <notebooks/116_optical_vortices>`
- {doc}`117 Ultrashort pulses <notebooks/117_ultrashort_pulses_lenses>`

### Coherence (120)

Spatial and temporal coherence, and how they measure stars.

- {doc}`120 Coherence and stellar interferometry <notebooks/120_coherence_stellar_interferometry>`

```{toctree}
:hidden:
:caption: 1xx Waves: propagation, beams and coherence

notebooks/100_propagation
notebooks/101_chirp_z_transform
notebooks/102_tilted_planes
notebooks/103_analytic_phase_propagation
notebooks/104_vectorial_diffraction
notebooks/105_wave_propagation_method
notebooks/110_gaussian_beams
notebooks/111_abcd_matrices
notebooks/112_beam_quality
notebooks/113_mode_analysis
notebooks/114_ince_gaussian_beams
notebooks/115_bessel_airy_self_healing
notebooks/116_optical_vortices
notebooks/117_ultrashort_pulses_lenses
notebooks/120_coherence_stellar_interferometry
```

## 2xx Light in matter

Interfaces, thin films and polarization; anisotropic crystals and the modulators built from them; full-wave Maxwell solvers (FDTD and FDFD) for structures on the scale of the wavelength.

### Interfaces and thin films (200–203)

Fresnel coefficients, multilayer stacks and coatings.

- {doc}`200 Interfaces and thin films <notebooks/200_interfaces_thin_films>`
- {doc}`201 Layer stacks <notebooks/201_layer_stacks>`
- {doc}`202 Stacks as transfer functions <notebooks/202_stack_transfer_functions>`
- {doc}`203 Coated laser mirror <notebooks/203_coated_laser_mirror>`

### Polarization, crystals and modulators (210–214)

Jones and Mueller calculus, anisotropic and gyrotropic media, and the devices that switch and steer light with them.

- {doc}`210 Jones and Mueller calculus <notebooks/210_jones_mueller_calculus>`
- {doc}`211 Crystal optics <notebooks/211_crystal_optics>`
- {doc}`212 Electro-optic modulators <notebooks/212_electro_optic_modulators>`
- {doc}`213 Liquid crystals <notebooks/213_liquid_crystals>`
- {doc}`214 Acousto-optics <notebooks/214_acousto_optics>`

### FDTD (220–225)

Full-wave solutions of Maxwell's equations on the Yee grid in the time domain (`fdtd`).

- {doc}`220 FDTD basics <notebooks/220_fdtd_basics>`
- {doc}`221 FDTD metals and plasmons <notebooks/221_fdtd_metals_plasmons>`
- {doc}`222 FDTD periodic structures <notebooks/222_fdtd_periodic_structures>`
- {doc}`223 FDTD photonic circuits <notebooks/223_fdtd_photonic_circuits>`
- {doc}`224 FDTD devices <notebooks/224_fdtd_devices>`
- {doc}`225 3D FDTD and nanojet <notebooks/225_fdtd_3d_nanojet>`

### FDFD and nanophotonics (230–233)

Maxwell's equations one frequency at a time (`fdfd`): resonances, microcavities and a carpet cloak.

- {doc}`230 FDFD basics <notebooks/230_fdfd_basics>`
- {doc}`231 Limaçon microcavity <notebooks/231_fdfd_limacon_microcavity>`
- {doc}`232 Husimi functions and ray chaos <notebooks/232_phase_space_husimi_chaos>`
- {doc}`233 Carpet cloak <notebooks/233_carpet_cloak>`

```{toctree}
:hidden:
:caption: 2xx Light in matter

notebooks/200_interfaces_thin_films
notebooks/201_layer_stacks
notebooks/202_stack_transfer_functions
notebooks/203_coated_laser_mirror
notebooks/210_jones_mueller_calculus
notebooks/211_crystal_optics
notebooks/212_electro_optic_modulators
notebooks/213_liquid_crystals
notebooks/214_acousto_optics
notebooks/220_fdtd_basics
notebooks/221_fdtd_metals_plasmons
notebooks/222_fdtd_periodic_structures
notebooks/223_fdtd_photonic_circuits
notebooks/224_fdtd_devices
notebooks/225_fdtd_3d_nanojet
notebooks/230_fdfd_basics
notebooks/231_fdfd_limacon_microcavity
notebooks/232_phase_space_husimi_chaos
notebooks/233_carpet_cloak
```

## 3xx Imaging and microscopy

How an image forms and how good it is: diffraction orders and the pupil, PSF, MTF and Strehl, Zernike aberrations, image processing; then the microscope and its contrast methods.

### Image formation and quality (300–306)

Abbe's diffraction orders and the pupil, PSF, MTF and Strehl ratio, Zernike and other bases, image processing and restoration.

- {doc}`300 Abbe image formation <notebooks/300_abbe_image_formation>`
- {doc}`301 Imaging and Zernikes <notebooks/301_imaging_zernikes>`
- {doc}`302 PSF, MTF and Strehl <notebooks/302_psf_mtf_strehl>`
- {doc}`303 Zernike aberration gallery <notebooks/303_zernike_aberration_gallery>`
- {doc}`304 Basis functions <notebooks/304_basis_functions>`
- {doc}`305 Image processing <notebooks/305_image_processing>`
- {doc}`306 Image restoration and PSF deconvolution <notebooks/306_image_restoration_deconvolution>`

### Microscopy (310–314)

Contrast methods, confocal and super-resolution microscopy, Fourier ptychography and optical traps.

- {doc}`310 Phase contrast and DIC <notebooks/310_phase_contrast_dic>`
- {doc}`311 Confocal microscopy <notebooks/311_confocal_microscopy>`
- {doc}`312 Fourier ptychography <notebooks/312_fourier_ptychography>`
- {doc}`313 Super-resolution microscopy <notebooks/313_super_resolution_microscopy>`
- {doc}`314 Optical tweezers <notebooks/314_optical_tweezers>`

```{toctree}
:hidden:
:caption: 3xx Imaging and microscopy

notebooks/300_abbe_image_formation
notebooks/301_imaging_zernikes
notebooks/302_psf_mtf_strehl
notebooks/303_zernike_aberration_gallery
notebooks/304_basis_functions
notebooks/305_image_processing
notebooks/306_image_restoration_deconvolution
notebooks/310_phase_contrast_dic
notebooks/311_confocal_microscopy
notebooks/312_fourier_ptychography
notebooks/313_super_resolution_microscopy
notebooks/314_optical_tweezers
```

## 4xx Ray tracing and lens design

Sequential ray tracing and aberration theory, optimization and tolerancing, aspheres and diffractive surfaces, complete lens systems, stray light, and the bridge between rays and waves.

### Ray tracing basics (400–407)

Sequential ray tracing to the wavefront, Seidel aberrations, the sine and Herschel conditions, telecentricity, glasses, Delano diagrams, polarization ray tracing and GRIN media.

- {doc}`400 Ray tracing to OPD <notebooks/400_ray_tracing_opd>`
- {doc}`401 Seidel aberrations <notebooks/401_seidel_caldwell_lens>`
- {doc}`402 Sine and Herschel conditions <notebooks/402_sine_herschel_lagrange>`
- {doc}`403 Magnification, distortion and telecentricity <notebooks/403_magnification_distortion_telecentricity>`
- {doc}`404 Glass catalogues <notebooks/404_glass_catalogues>`
- {doc}`405 Delano diagrams <notebooks/405_litho_delano_lens_design>`
- {doc}`406 Polarization ray tracing <notebooks/406_polarization_raytrace>`
- {doc}`407 GRIN optics <notebooks/407_grin_optics>`

### Optimization and tolerancing (410–415)

Damped least squares, several configurations, the landscape of local minima, Monte Carlo tolerancing, polishing errors and CAD export.

- {doc}`410 Lens optimization <notebooks/410_lens_optimization>`
- {doc}`411 Multi-configuration optimization <notebooks/411_multiconfig_optimization>`
- {doc}`412 Global optimization and saddle points <notebooks/412_global_optimization_saddle_points>`
- {doc}`413 Monte Carlo tolerancing <notebooks/413_monte_carlo_tolerancing>`
- {doc}`414 Polishing errors <notebooks/414_polishing_errors>`
- {doc}`415 CAD export of lens systems <notebooks/415_cad_export_lens_systems>`

### Aspheres and special surfaces (420–422)

Forbes Q polynomials, Fresnel lenses and diffractive surfaces.

- {doc}`420 Forbes Q aspheres <notebooks/420_forbes_q_aspheres>`
- {doc}`421 Fresnel lenses <notebooks/421_fresnel_lenses>`
- {doc}`422 Hybrid diffractive lenses <notebooks/422_hybrid_diffractive_lenses>`

### Lens systems (430–436)

Complete designs: a phone camera lens, zoom lenses, eyepieces and binoculars, scan lenses, spectacle lenses and off-axis mirror systems.

- {doc}`430 Phone camera lens with Forbes Q polynomials <notebooks/430_phone_camera_lens_forbes_q>`
- {doc}`431 Zoom lenses <notebooks/431_zoom_lenses>`
- {doc}`432 Eyepieces <notebooks/432_eyepieces>`
- {doc}`433 Binoculars and erecting prisms <notebooks/433_binoculars_erecting_prisms>`
- {doc}`434 Scan lenses <notebooks/434_scan_lenses>`
- {doc}`435 Progressive spectacle lenses <notebooks/435_progressive_lenses>`
- {doc}`436 Off-axis mirrors and nodal aberrations <notebooks/436_off_axis_mirrors_nodal_aberrations>`

### Stray light (440–441)

Non-sequential ray tracing, ghosts and scatter.

- {doc}`440 Non-sequential ray tracing <notebooks/440_non_sequential_ray_tracing>`
- {doc}`441 Stray light in Caldwell's lens <notebooks/441_stray_light_caldwell_lens>`

### Rays and waves (450–454)

Converting between ray ensembles and wave fields: ray field summation, foci of real lenses, Gaussian beamlets, DOEs and the eye.

- {doc}`450 Rays and waves <notebooks/450_rays_and_waves>`
- {doc}`451 Lens focus from rays to waves <notebooks/451_lens_focus_waves>`
- {doc}`452 Gaussian beam decomposition <notebooks/452_gaussian_beam_decomposition>`
- {doc}`453 Laser head DOE <notebooks/453_laser_head_doe>`
- {doc}`454 Eye aberrometer <notebooks/454_eye_aberrometer>`

### Field tracing (460–463)

Rays through the lenses, waves where diffraction matters: chains of propagation operators chosen step by step.

- {doc}`460 Field tracing <notebooks/460_field_tracing>`
- {doc}`461 Source models in field tracing <notebooks/461_source_models_field_tracing>`
- {doc}`462 Local plane interface approximation <notebooks/462_local_plane_interface_approximation>`
- {doc}`463 Field tracing applications <notebooks/463_field_tracing_applications>`

```{toctree}
:hidden:
:caption: 4xx Ray tracing and lens design

notebooks/400_ray_tracing_opd
notebooks/401_seidel_caldwell_lens
notebooks/402_sine_herschel_lagrange
notebooks/403_magnification_distortion_telecentricity
notebooks/404_glass_catalogues
notebooks/405_litho_delano_lens_design
notebooks/406_polarization_raytrace
notebooks/407_grin_optics
notebooks/410_lens_optimization
notebooks/411_multiconfig_optimization
notebooks/412_global_optimization_saddle_points
notebooks/413_monte_carlo_tolerancing
notebooks/414_polishing_errors
notebooks/415_cad_export_lens_systems
notebooks/420_forbes_q_aspheres
notebooks/421_fresnel_lenses
notebooks/422_hybrid_diffractive_lenses
notebooks/430_phone_camera_lens_forbes_q
notebooks/431_zoom_lenses
notebooks/432_eyepieces
notebooks/433_binoculars_erecting_prisms
notebooks/434_scan_lenses
notebooks/435_progressive_lenses
notebooks/436_off_axis_mirrors_nodal_aberrations
notebooks/440_non_sequential_ray_tracing
notebooks/441_stray_light_caldwell_lens
notebooks/450_rays_and_waves
notebooks/451_lens_focus_waves
notebooks/452_gaussian_beam_decomposition
notebooks/453_laser_head_doe
notebooks/454_eye_aberrometer
notebooks/460_field_tracing
notebooks/461_source_models_field_tracing
notebooks/462_local_plane_interface_approximation
notebooks/463_field_tracing_applications
```

## 5xx Metrology and computational imaging

Measuring phase, surfaces and films: interferometers, wavefront sensors, ellipsometry and scatterometry, holography, phase retrieval and imaging without lenses.

### Interferometry (500–505)

Phase from fringes, unwrapping, optical testing of spheres and aspheres, white light interferometry and OCT.

- {doc}`500 Interferometry <notebooks/500_interferometry>`
- {doc}`501 Phase unwrapping <notebooks/501_phase_unwrapping>`
- {doc}`502 Radius interferometer <notebooks/502_radius_interferometer>`
- {doc}`503 CGH null testing of aspheres <notebooks/503_cgh_null_testing_aspheres>`
- {doc}`504 Coherence scanning interferometry <notebooks/504_coherence_scanning_interferometry>`
- {doc}`505 OCT <notebooks/505_optical_coherence_tomography>`

### Wavefront and slope sensing (510–512)

Shack-Hartmann, phase diversity, Schlieren and deflectometry.

- {doc}`510 Shack-Hartmann vs phase diversity <notebooks/510_shack_hartmann_vs_phase_diversity>`
- {doc}`511 Schlieren and knife-edge <notebooks/511_schlieren_knife_edge>`
- {doc}`512 Deflectometry <notebooks/512_deflectometry>`

### Polarimetric metrology (520–521)

Ellipsometry and scatterometry of films and gratings.

- {doc}`520 Ellipsometry <notebooks/520_ellipsometry>`
- {doc}`521 Scatterometry <notebooks/521_scatterometry>`

### Holography, phase retrieval and computational imaging (530–536)

Recovering phase from intensities, shaping intensities with phase, focusing through scattering media and imaging without lenses.

- {doc}`530 Digital holography <notebooks/530_digital_holography>`
- {doc}`531 Holographic microscopy <notebooks/531_holographic_microscopy>`
- {doc}`532 Phase retrieval <notebooks/532_phase_retrieval>`
- {doc}`533 TIE <notebooks/533_tie>`
- {doc}`534 IFTA beam shaping <notebooks/534_ifta_beam_shaping>`
- {doc}`535 Wavefront shaping <notebooks/535_wavefront_shaping>`
- {doc}`536 Lensless imaging <notebooks/536_lensless_coded_aperture_imaging>`

```{toctree}
:hidden:
:caption: 5xx Metrology and computational imaging

notebooks/500_interferometry
notebooks/501_phase_unwrapping
notebooks/502_radius_interferometer
notebooks/503_cgh_null_testing_aspheres
notebooks/504_coherence_scanning_interferometry
notebooks/505_optical_coherence_tomography
notebooks/510_shack_hartmann_vs_phase_diversity
notebooks/511_schlieren_knife_edge
notebooks/512_deflectometry
notebooks/520_ellipsometry
notebooks/521_scatterometry
notebooks/530_digital_holography
notebooks/531_holographic_microscopy
notebooks/532_phase_retrieval
notebooks/533_tie
notebooks/534_ifta_beam_shaping
notebooks/535_wavefront_shaping
notebooks/536_lensless_coded_aperture_imaging
```

## 6xx Illumination, radiometry and cameras

Watts and lumens, étendue and the design of illumination optics; the optics of cameras and photography.

### Radiometry and illumination (600–608)

Radiometry and photometry, étendue, nonimaging concentrators, Köhler illumination, homogenizers, speckle, LEDs and freeform beam shaping.

- {doc}`600 Radiometry and photometry <notebooks/600_radiometry_photometry>`
- {doc}`601 Radiometry of images <notebooks/601_radiometry_of_images>`
- {doc}`602 Étendue <notebooks/602_etendue_brightness_theorem>`
- {doc}`603 Nonimaging concentrators <notebooks/603_nonimaging_concentrators>`
- {doc}`604 Köhler and critical illumination <notebooks/604_koehler_critical_illumination>`
- {doc}`605 Homogenizers <notebooks/605_homogenizers>`
- {doc}`606 Speckle reduction <notebooks/606_speckle_reduction>`
- {doc}`607 LED optics <notebooks/607_led_optics_monte_carlo>`
- {doc}`608 Freeform beam shaping <notebooks/608_freeform_beam_shaping>`

### Cameras and photography (610–616)

The sensor, bokeh, sunstars, flare, depth of field, image correction and the light field camera.

- {doc}`610 The sensor <notebooks/610_sensor>`
- {doc}`611 Bokeh <notebooks/611_bokeh>`
- {doc}`612 Sunstars <notebooks/612_sunstars>`
- {doc}`613 Flare and ghosts <notebooks/613_flare_ghosts>`
- {doc}`614 Depth of field and tilt-shift <notebooks/614_tilt_shift_depth_of_field>`
- {doc}`615 Distortion, lateral colour and lens shading correction <notebooks/615_distortion_shading_correction>`
- {doc}`616 Light field camera <notebooks/616_light_field_camera>`

```{toctree}
:hidden:
:caption: 6xx Illumination, radiometry and cameras

notebooks/600_radiometry_photometry
notebooks/601_radiometry_of_images
notebooks/602_etendue_brightness_theorem
notebooks/603_nonimaging_concentrators
notebooks/604_koehler_critical_illumination
notebooks/605_homogenizers
notebooks/606_speckle_reduction
notebooks/607_led_optics_monte_carlo
notebooks/608_freeform_beam_shaping
notebooks/610_sensor
notebooks/611_bokeh
notebooks/612_sunstars
notebooks/613_flare_ghosts
notebooks/614_tilt_shift_depth_of_field
notebooks/615_distortion_shading_correction
notebooks/616_light_field_camera
```

## 7xx Lasers, nonlinear optics, fibres and detectors

How lasers work, light that changes the medium it travels in, light guided in fibres, and the detectors at the end of the line.

### Lasers (700–703)

Gain, threshold and modes, Q-switching and mode-locking, laser diodes.

- {doc}`700 Laser fundamentals <notebooks/700_laser_fundamentals>`
- {doc}`701 Laser resonator modes <notebooks/701_laser_resonator_modes>`
- {doc}`702 Q-switching and mode-locking <notebooks/702_q_switching_mode_locking>`
- {doc}`703 Laser diodes <notebooks/703_laser_diodes>`

### Nonlinear optics (710–712)

Second harmonic generation, parametric amplification and the Kerr effect.

- {doc}`710 Second harmonic generation and phase matching <notebooks/710_second_harmonic_generation>`
- {doc}`711 Parametric amplification and OPOs <notebooks/711_parametric_amplification_opo>`
- {doc}`712 Kerr effect and self-focusing <notebooks/712_kerr_effect_self_focusing>`

### Fibre optics (720–726)

Light guided in glass fibres and the basics of sending bits through them.

- {doc}`720 Light in a fibre <notebooks/720_fibre_rays_step_graded>`
- {doc}`721 Fibre modes <notebooks/721_fibre_modes>`
- {doc}`722 Dispersion in fibres <notebooks/722_fibre_dispersion>`
- {doc}`723 Pulse broadening <notebooks/723_fibre_pulse_broadening>`
- {doc}`724 Fibre Bragg gratings <notebooks/724_fibre_bragg_gratings>`
- {doc}`725 Fibre link <notebooks/725_fibre_link>`
- {doc}`726 SPM and solitons in fibres <notebooks/726_fibre_spm_solitons>`

### Detectors (730)

Where the numbers of a detector come from and what photons do to them.

- {doc}`730 Detectors and photon statistics <notebooks/730_detectors_photon_statistics>`

```{toctree}
:hidden:
:caption: 7xx Lasers, nonlinear optics, fibres and detectors

notebooks/700_laser_fundamentals
notebooks/701_laser_resonator_modes
notebooks/702_q_switching_mode_locking
notebooks/703_laser_diodes
notebooks/710_second_harmonic_generation
notebooks/711_parametric_amplification_opo
notebooks/712_kerr_effect_self_focusing
notebooks/720_fibre_rays_step_graded
notebooks/721_fibre_modes
notebooks/722_fibre_dispersion
notebooks/723_fibre_pulse_broadening
notebooks/724_fibre_bragg_gratings
notebooks/725_fibre_link
notebooks/726_fibre_spm_solitons
notebooks/730_detectors_photon_statistics
```

## 8xx Lithography

Projection lithography from the mask to the resist: imaging and resist models, rigorous masks, the projection lens and the illuminator, source and mask optimization, stochastics, and EUV.

### Imaging and resist (800–805)

Vectorial and Hopkins imaging, process windows, resist and arbitrary layouts.

- {doc}`800 Vectorial imaging in lithography <notebooks/800_vectorial_imaging_lithography>`
- {doc}`801 Lithography toolbox <notebooks/801_lithography_toolbox>`
- {doc}`802 Hopkins imaging and SOCS <notebooks/802_hopkins_socs>`
- {doc}`803 Fourier interpolation of aerial images <notebooks/803_litho_fourier_interpolation_cd>`
- {doc}`804 Resist and swing curves <notebooks/804_litho_resist_swing>`
- {doc}`805 Arbitrary mask layouts <notebooks/805_litho_arbitrary_layouts>`

### Masks (810–812)

Rigorous mask diffraction with RCWA and FDTD, compared with thin mask models.

- {doc}`810 Rigorous mask <notebooks/810_litho_rigorous_mask>`
- {doc}`811 Mask models compared <notebooks/811_litho_mask_models>`
- {doc}`812 FDTD mask for lithography <notebooks/812_litho_fdtd_mask>`

### Projection lens and illuminator (820–824)

The ray traced projection lens across the slit, Jones pupils, telecentricity, the illuminator and lens tolerances.

- {doc}`820 Lens aberrations across the slit <notebooks/820_litho_lens_aberrations>`
- {doc}`821 Jones pupils <notebooks/821_litho_jones_pupil>`
- {doc}`822 Telecentricity across the slit <notebooks/822_litho_telecentricity>`
- {doc}`823 KrF illuminator <notebooks/823_litho_illuminator>`
- {doc}`824 Lens tolerances and wafer yield <notebooks/824_lens_tolerances_wafer_yield>`

### Optimization and stochastics (830–832)

Source optimization, OPC, and the noise of photons and acids.

- {doc}`830 Source optimization <notebooks/830_litho_source_optimization>`
- {doc}`831 OPC <notebooks/831_litho_opc>`
- {doc}`832 Litho stochastics, flare and focus blur <notebooks/832_litho_stochastics_flare_focus_blur>`

### EUV (840–842)

Multilayer mirrors, reflective masks, anamorphic pupils and EUV stochastics.

- {doc}`840 EUV multilayers and reflective masks <notebooks/840_euv_multilayers_reflective_masks>`
- {doc}`841 EUV projector and pupil <notebooks/841_euv_projector_pupil>`
- {doc}`842 EUV stochastics and flare <notebooks/842_euv_stochastics_flare>`

```{toctree}
:hidden:
:caption: 8xx Lithography

notebooks/800_vectorial_imaging_lithography
notebooks/801_lithography_toolbox
notebooks/802_hopkins_socs
notebooks/803_litho_fourier_interpolation_cd
notebooks/804_litho_resist_swing
notebooks/805_litho_arbitrary_layouts
notebooks/810_litho_rigorous_mask
notebooks/811_litho_mask_models
notebooks/812_litho_fdtd_mask
notebooks/820_litho_lens_aberrations
notebooks/821_litho_jones_pupil
notebooks/822_litho_telecentricity
notebooks/823_litho_illuminator
notebooks/824_lens_tolerances_wafer_yield
notebooks/830_litho_source_optimization
notebooks/831_litho_opc
notebooks/832_litho_stochastics_flare_focus_blur
notebooks/840_euv_multilayers_reflective_masks
notebooks/841_euv_projector_pupil
notebooks/842_euv_stochastics_flare
```

## 9xx Instruments

Telescopes and the atmosphere, spectrometers, AR/VR and head-up displays, and 3D sensing.

### Telescopes, atmosphere and astronomy (900–906)

Telescope designs, segmented mirrors, the atmosphere and adaptive optics, coronagraphs, gravitational lenses and rainbows.

- {doc}`900 Telescope zoo <notebooks/900_telescope_zoo>`
- {doc}`901 Segmented mirrors <notebooks/901_segmented_mirrors>`
- {doc}`902 Atmospheric dispersion corrector <notebooks/902_atmospheric_dispersion_corrector>`
- {doc}`903 Twinkling stars and AO <notebooks/903_twinkling_stars_adaptive_optics>`
- {doc}`904 Coronagraphs <notebooks/904_coronagraphs>`
- {doc}`905 Einstein rings <notebooks/905_einstein_rings>`
- {doc}`906 Rainbows and glories <notebooks/906_rainbows_and_glories>`

### Spectroscopy (910–915)

Spectrometers and what they measure.

- {doc}`910 Grating and prism spectrometers <notebooks/910_grating_prism_spectrometers>`
- {doc}`911 Fourier transform spectroscopy <notebooks/911_fourier_transform_spectroscopy>`
- {doc}`912 Fabry–Perot and filters <notebooks/912_fabry_perot_filters>`
- {doc}`913 Spectral lines and absorption <notebooks/913_spectral_lines_absorption>`
- {doc}`914 Echelle spectrograph <notebooks/914_echelle_spectrograph>`
- {doc}`915 Imaging spectroscopy <notebooks/915_imaging_spectroscopy>`

### AR/VR and head-up displays (920–923)

Waveguide combiners, volume gratings, pancake lenses and windshield HUDs.

- {doc}`920 AR waveguide combiners <notebooks/920_ar_waveguide_combiners>`
- {doc}`921 Volume holographic gratings <notebooks/921_volume_holographic_gratings>`
- {doc}`922 Pancake lenses <notebooks/922_pancake_lenses>`
- {doc}`923 Head-up displays <notebooks/923_head_up_displays>`

### 3D sensing and LiDAR (930–933)

Time-of-flight and FMCW LiDAR, beam scanners and structured light.

- {doc}`930 Time-of-flight LiDAR <notebooks/930_time_of_flight_lidar>`
- {doc}`931 FMCW LiDAR <notebooks/931_fmcw_lidar>`
- {doc}`932 Beam scanning <notebooks/932_beam_scanning>`
- {doc}`933 Triangulation and structured light <notebooks/933_triangulation_structured_light>`

```{toctree}
:hidden:
:caption: 9xx Instruments

notebooks/900_telescope_zoo
notebooks/901_segmented_mirrors
notebooks/902_atmospheric_dispersion_corrector
notebooks/903_twinkling_stars_adaptive_optics
notebooks/904_coronagraphs
notebooks/905_einstein_rings
notebooks/906_rainbows_and_glories
notebooks/910_grating_prism_spectrometers
notebooks/911_fourier_transform_spectroscopy
notebooks/912_fabry_perot_filters
notebooks/913_spectral_lines_absorption
notebooks/914_echelle_spectrograph
notebooks/915_imaging_spectroscopy
notebooks/920_ar_waveguide_combiners
notebooks/921_volume_holographic_gratings
notebooks/922_pancake_lenses
notebooks/923_head_up_displays
notebooks/930_time_of_flight_lidar
notebooks/931_fmcw_lidar
notebooks/932_beam_scanning
notebooks/933_triangulation_structured_light
```

## Running the notebooks

```{include} ../notebooks/README.md
:start-after: <!-- docs:running-start -->
:end-before: <!-- docs:running-end -->
```
