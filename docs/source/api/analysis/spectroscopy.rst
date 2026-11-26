Spectroscopy Analysis
=====================

Frequency axes and chemical-shift calibration
---------------------------------------------

The spectral frequency axis itself (:func:`~hypermri.reconstruction.axes.get_freq_axis`
and :func:`~hypermri.reconstruction.axes.center_freq_axis`) belongs to the
reconstruction and is documented there (:ref:`api-spectral-axes`); both names
can also be imported from ``hypermri.analysis.spectroscopy.frequency`` and
``hypermri.analysis.spectroscopy``.

The functions below convert positions between Hz and ppm. Absolute
calibration uses a regression over the paired axes, including its intercept;
relative ppm intervals subtract the converted zero position and keep the
calibration sign. Ascending and descending axes keep their direction.
``ppm_centered_at_0=True`` centres both calibration axes before the
regression; frequency arguments are then relative to the centred Hz axis.

.. automodule:: hypermri.analysis.spectroscopy.frequency
   :members: freq_to_index, get_bw_from_freq_axis, freq_hz_to_ppm, freq_ppm_to_hz, add_fit_freqs_ppm

Signal processing and peak detection
------------------------------------

Effective NSA for echo design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numerical NSA calculations live in ``hypermri.analysis.spectroscopy.nsa`` and
can be imported without notebook controls. The existing interactive dashboard
remains ``hypermri.visualization.nsa_dashboard.interactive_nsa_simulation``.
Its callbacks call the same calculation used by scripts. The former dashboard
aliases ``calculate_nsa`` and ``get_freq_hz`` were removed on 15 November 2026.

The numerical API takes echo times and T2-star values in seconds and field
strength in Tesla. The dashboard retains milliseconds for its timing controls
and kHz for bandwidth. NSA is dimensionless and assumes independent
equal-variance echo noise with fixed frequencies; it does not estimate measured
noise or frequency-fit uncertainty. The existing phantom offsets, 0.61/0.39
coupling weights, singular-design zeros and clipping policy are preserved.

.. automodule:: hypermri.analysis.spectroscopy.nsa
   :members: get_freq_hz, calculate_nsa

.. automodule:: hypermri.analysis.spectroscopy.signals
   :members:

.. automodule:: hypermri.analysis.spectroscopy.processing
   :members:

.. automodule:: hypermri.analysis.spectroscopy.peaks
   :members:

.. automodule:: hypermri.analysis.spectroscopy.metabolites
   :members:

.. automodule:: hypermri.analysis.spectroscopy.kinetics
   :members:

.. automodule:: hypermri.analysis.spectroscopy.single_voxel
   :members:
   :undoc-members:
   :show-inheritance:

Frequency shifting and alignment
--------------------------------

``shift_frequency`` multiplies FIDs by a phase ramp
``exp(2j * pi * offset_hz * t)`` and works on plain NumPy arrays with any
sampling axis. Spectral input in ``fftshift`` order is transformed to the time
domain and back. ``align_spectra_by_fit_frequency`` is the labelled-data
workflow: it derives per-voxel offsets from fitted frequencies, reads timing
and ``signal_domain`` from the data attributes, and applies the same
``shift_frequency`` operation along ``acquisition_idx``.

.. code-block:: python

   import numpy as np
   from hypermri.analysis.spectroscopy import shift_frequency

   shifted_spectrum = shift_frequency(
       spectrum, 10.0, dwell_time_s=dwell_time_s, signal_domain="spectral"
   )

.. automodule:: hypermri.analysis.spectroscopy.alignment
   :members: shift_frequency, align_spectra_by_fit_frequency

Calibrated pH conversions
-------------------------

The existing pH/ppm/Hz conversion functions and calibration constants now live
in ``hypermri.analysis.spectroscopy.ph``; the former
``hypermri.functions.pHconverter`` module was removed on 15 November 2026. Field strength remains in Tesla and measured reference
frequency in MHz; this relocation changes no formula or calibration value.

.. automodule:: hypermri.analysis.spectroscopy.ph
   :members: pH_to_ppm, ppm_to_pH, freq_to_pH, pH_to_freq, OMPD5_freq2pH, OMPD_pH_from_c1_c5, OMPD_pH_from_peak_shift, OMPD_pH_from_absolute_position

CSI acquisition-order diagnostics
----------------------------------

The signal-decay viewer uses these widget-free operations for mean magnitude,
spectral band power, their normalized cumulative sums, linear tail fits and its
shared encoding-order permutation. Curve indices refer to acquired k-space
points; they are not the FID sampling-time axis. Source arrays, coordinates and
metadata are preserved.

Band power averages ``|FFT(FID)|^2 / n_samples`` over chosen frequency ranges,
for example the metabolite lines. Noise outside the ranges no longer enters the
curve, so weak signal at the edge of k-space stays visible where the
whole-FID magnitude has already reached its noise floor. Pure noise gives the
flat level ``2 sigma^2`` for any range; the acquired point where the curve meets
that level marks the end of measurable signal. In the viewer, choose
``Band power`` and type ranges such as ``-330:-150, 200:420`` in Hz, or pass
``bands_hz`` to ``CSI.plot_signal_decay``.

.. automodule:: hypermri.analysis.spectroscopy.signal_decay
   :members: signal_decay_curves, spectral_band_power, csi_order_indices

CSI peak detection without controls (Beta)
------------------------------------------

The interactive selector and automatic fitting pipeline share one detection
operation. It reports candidates, stable clusters and initial Lorentz search
ranges without running a fit or constructing a figure or widget:

.. code-block:: python

   from hypermri.analysis.spectroscopy import (
       CSIPeakDetectionOptions,
       detect_csi_peaks,
   )

   options = CSIPeakDetectionOptions(b0_shift_tolerance_hz=6, max_peaks=2)
   detected = detect_csi_peaks(csi_data, options)
   centres_hz = [peak["center_hz"] for peak in detected["cluster_summary"]]

Frequency ranges use Hz; initial T2-star values and ranges use seconds.
Detection noise comes from the magnitude-spectrum edges. The optional legacy
FID-tail amplitude filter retains its existing behavior and is separate from
the trusted-noise fit-quality workflow. An empty result lets the interactive
selector retain manual peaks; the automatic fitting pipeline retains its error.
Source samples, labelled axes, units and precision are preserved. Existing
frequency resolution may append the documented frequency-range alias attributes.
Neither consumer changes its defaults, peak selection or result publication.

.. autoclass:: hypermri.analysis.spectroscopy.pipeline.CSIPeakDetectionOptions

.. autofunction:: hypermri.analysis.spectroscopy.pipeline.detect_csi_peaks

Selected-voxel refitting without controls (Beta)
------------------------------------------------

``refit_spectroscopy_voxel`` is the analysis operation used by ReviewFit. It
prepares a selected Lorentz spectrum, runs the existing frequency/T2 search
and refines amplitudes at the final parameters. Scripts use the same operation
without constructing a viewer or fit session:

.. code-block:: python

   from hypermri.analysis.spectroscopy import (
       SpectroscopyRefitEdit,
       refit_spectroscopy_voxel,
   )

   reviewed = refit_spectroscopy_voxel(
       data,
       fit_params,
       fit_results,
       voxel_indices={"read_idx": 1, "phase_idx": 2, "slice_idx": 0},
       edits={
           "Pyruvate": SpectroscopyRefitEdit(
               initial_freq_hz=15,
               freq_search_range_hz=4,
               initial_t2_s=0.03,
               min_t2_s=0.005,
               max_t2_s=0.1,
           ),
       },
   )
   selected_results = reviewed["fit_results"]

Select every non-metabolite result dimension with more than one position.
``data_indices`` can separately select signal channels or repetitions. Reviewed
frequencies and widths use Hz; T2-star values and bounds use seconds. Lines
without a reviewed row search the selected voxel's ``metabs_freq_ranges``
width and ``min_t2_s``/``max_t2_s`` T2-star bounds, as in the full fit; every
line uses the voxel's ``metabs_t2_ranges`` T2-star width. Every line's
frequency bounds are its start -/+ half its width, and a width of 0 freezes
that parameter. Disabled
metabolites keep their earlier estimates. Inputs, result precision, units,
selected coordinates and provenance are preserved; callers own persistence.

Aligned seconds metadata takes precedence over legacy full sampling axes.
Full axes are cut once, with existing dwell-time and positional unit-step
fallbacks retained. This function adds no preprocessing or domain conversion.
Its result records effective settings and search diagnostics. ``status`` reports
whether refitting was invoked; fit quality remains a separate operation.
Amplitude-refinement failure is reported while retaining the original fitter
amplitudes. Reconstructed signals and uncertainty are not calculated here;
their result fields are explicitly ``None``.

.. autoclass:: hypermri.analysis.spectroscopy.refitting.SpectroscopyRefitEdit

.. autofunction:: hypermri.analysis.spectroscopy.refitting.refit_spectroscopy_voxel

Step-by-step trace of one FID fit (Beta)
----------------------------------------

``fit_spectrum_trace`` refits one selected spectrum with the optimization
history switched on and returns every evaluated state of that fit: the
measured FID and spectrum, the model of each step, every frequency and T2*
grid with its bounds and choice, the complex amplitudes and the phase law
``angle(c * exp(-2j*pi*f*tau))`` of each line. It runs the same single-spectrum
fit as ``fit_data_pseudo_inv``; inputs are not changed. Frequencies use Hz,
T2* values and times use seconds. Voigt line shapes are not supported.

.. code-block:: python

   from hypermri.analysis.spectroscopy import fit_spectrum_trace

   trace = fit_spectrum_trace(
       data, fit_params=params, selection={"x_idx": 3}, fit_results=results
   )
   trace["entries"][-1]["frequency_hz"]

:class:`hypermri.visualization.fit_monitor.FitMonitor` serves the trace as a
local browser page; the ``interactive_fit`` workflow starts it with its
Fit Monitor button.

.. autofunction:: hypermri.analysis.spectroscopy.fit_trace.fit_spectrum_trace

Simple fitting, noise and uncertainty
-------------------------------------

The domain entry point keeps the ordinary numerical fit unchanged when no
noise is supplied.  A scalar or labelled per-spectrum quadrature sigma adds
cheap conditional diagnostics without weighting or changing the fitted
parameters:

.. code-block:: python

   from hypermri.analysis.spectroscopy import (
       compare_metabolite_relevance,
       estimate_fit_covariance,
       estimate_fit_local_uncertainty,
       estimate_fit_uncertainty,
       estimate_noise,
       fit_spectrum,
   )

   noise = estimate_noise(rf_off_data, method="rf_off")
   fit = fit_spectrum(data, fit_params, noise=noise)

   # fit_params is the unchanged prepared Dataset used for this fit.
   local_errors = estimate_fit_covariance(fit, fit_params)
   amplitude_std = local_errors.amplitude_abs_std
   frequency_std_hz = local_errors.frequency_std
   t2_std_s = local_errors.t2_std

   # Reportable local intervals with explicit nonregular-case statuses.
   local_intervals = estimate_fit_local_uncertainty(fit, fit_params)
   frequency_low_hz = local_intervals.frequency_interval_low
   frequency_high_hz = local_intervals.frequency_interval_high

   relevance = compare_metabolite_relevance(
       data,
       fit_params,
       fit,
       targets=["Bicarbonate"],
   )

   fit_params.attrs["uncertainty_bootstrap_samples"] = 999
   uncertainty = estimate_fit_uncertainty(
       data,
       fit_params,
       fit,
       selection=roi_mask,
   )

The relevance result is an observed M1/M0 RSS improvement, not a p-value.
``estimate_fit_covariance`` adds joint local standard errors, covariance and
correlation without refitting. Frequencies and T2* participate in the full
Jacobian unless fixed by the supplied fit parameters. Read
``covariance_status`` and ``amplitude_abs_status`` before interpreting errors:
rank deficiency, active bounds and magnitudes near zero do not receive
apparently precise magnitude errors. These are local one-standard-deviation
estimates, not calibrated confidence intervals or automatic Q2 eligibility.
Noise must describe the same reconstruction, filtering and scaling as the
fitted samples; apply any CSI k-space filter to the RF-off reference as well.

Bootstrap work starts only when ``estimate_fit_uncertainty`` is called.  For
multi-spectrum data it requires a labelled selection or an explicit
``selection="all"``.

``eliminate_unnecessary_lines`` decides, per spectrum, which lines of one
overlapping group the data need. It starts from the full fit and refits
without each remaining candidate. The line whose removal costs least is
removed if ``delta_chi2 - delta_k * ln(2 N)`` is negative, where
``delta_chi2`` is the RSS increase divided by the noise variance, ``delta_k``
the number of parameters of the line and ``N`` the number of fitted complex
samples (Bayesian information criterion). This repeats until no line can be
removed; at least one candidate always stays:

.. code-block:: python

   from hypermri.analysis.spectroscopy import eliminate_unnecessary_lines

   checked = eliminate_unnecessary_lines(
       data,
       fit_params,
       candidates=["OMPD1_1", "OMPD1_2", "OMPD1_3"],
       noise=noise,
   )
   removed = checked.necessity_removed.sel(metabolite="OMPD1_3")

The result is the final fit with removed lines skipped, plus the decisions,
the removal step, ``necessity_delta_chi2`` and ``necessity_threshold``. Without
a phase constraint, spurious lines that cancel each other can still look
necessary, so combine the test with the phase band described in
:doc:`fitting`. For Bruker FID scans, ``phase_band_tau_s_from_scan``
returns the band's delay ``phase_band_tau_s`` from the echo and dwell times. It
warns that the value is not verified for methods other than
``<User:lucaCSI4>``.

``add_line_from_residual`` is the opposite step: it looks for one line the
model lacks. It fits the current lines, proposes a new line at the largest
peak of the residual spectrum inside ``search_range_hz`` (at least
``min_peak_separation_hz`` from every fitted line) and refits all lines
from the first fit with the new line added. The new line is kept where
``delta_chi2 > (k_M1 - k_M0) * ln(2 N)`` (4 parameters, about 25 for 256
samples), the same information criterion as above; elsewhere the result
equals the first fit and the new line is skipped with amplitude 0. Without
``noise`` it uses the acquisition-tail sigma the input carries:

.. code-block:: python

   from hypermri.analysis.spectroscopy import add_line_from_residual

   searched = add_line_from_residual(
       data, fit_params, search_range_hz=(320.0, 380.0), new_metab_name="extra"
   )
   kept = searched.line_search_accepted

On synthetic spectra a line of area 1 in noise of 0.05 per quadrature is
found within 0.2 Hz, and in 200 noise-only spectra no line was kept. The
review plugin's "Add line from residual" button runs the search for the
current voxel and, when the line is kept, adds it to the model at its fitted
frequency; the next refit fits it. Whether a line found this way belongs in
the metabolite model remains a scientific decision.

.. autofunction:: hypermri.analysis.spectroscopy.fitting.fit_spectrum

.. autofunction:: hypermri.analysis.spectroscopy.noise_estimation.estimate_noise

``estimate_noise`` is the only general noise estimator.  The per-source
functions ``estimate_noise_from_complex_samples``,
``estimate_noise_from_rf_off``, ``estimate_noise_from_frames`` and
``estimate_noise_from_outside_mask`` and the former
``hypermri.utils.physics.estimate_noise`` were removed on 15 November 2026.
Call ``estimate_noise(data, method=...)`` with ``"complex_samples"``,
``"rf_off"``, ``"noise_frames"`` or ``"outside_object"`` instead.  For
magnitude data, such as the background of a magnitude image, pass
``sample_type="magnitude"``; the result is the Rayleigh scale that the removed
physics helper returned for ``component="magnitude"``.  Its printed summary
has no replacement; ``return_pipeline=True`` reports the same statistics.

.. autofunction:: hypermri.analysis.spectroscopy.relevance.compare_metabolite_relevance

.. autofunction:: hypermri.analysis.spectroscopy.relevance.eliminate_unnecessary_lines

.. autofunction:: hypermri.analysis.spectroscopy.relevance.add_line_from_residual

.. automodule:: hypermri.analysis.spectroscopy.phase_timing
   :members: phase_band_tau_s_from_scan, VALIDATED_PHASE_BAND_SEQUENCES,
      PULSE_ACQUIRE_DEAD_TIME_SEQUENCES

.. autofunction:: hypermri.analysis.spectroscopy.uncertainty.estimate_fit_uncertainty

.. autofunction:: hypermri.analysis.spectroscopy.covariance.estimate_fit_covariance

.. autofunction:: hypermri.analysis.spectroscopy.local_uncertainty.estimate_fit_local_uncertainty

Advanced calibrated outer-k-space noise
---------------------------------------

``estimate_csi_noise`` is the explicitly named advanced noise source
``"automatic_outer_kspace"``.  It needs a validated population calibration and
returns raw k-space sigma that must be propagated through the reconstruction
before it describes fitted spectra.  Neither ``estimate_noise`` nor the quality
workflow selects it on its own.

.. autofunction:: hypermri.analysis.spectroscopy.noise_estimation.estimate_csi_noise

Acquisition-tail noise
----------------------

In a centric CSI acquisition the late acquired k-space points carry no signal.
``estimate_acquisition_tail_noise`` finds the first acquired point from which
both the whole-FID energy and the metabolite-band power of
``spectral_band_power`` are consistent with noise, and returns the noise
standard deviation per real or imaginary quadrature in acquired signal units.
The last FID sample is excluded. ``propagate_acquisition_tail_noise`` carries
this value through the linear reconstruction (acquired-point weights of the
filter and ``signal_end``, the spatial FFT, a user amplitude scale and the mask
weight) to the reconstructed voxel. Its result is named
``noise_sigma_quadrature``, carries ``noise_source="acquisition_tail"`` and the
estimate's provenance, and can be passed as ``noise`` to ``fit_spectrum``:

.. code-block:: python

   from hypermri.analysis.spectroscopy import (
       estimate_acquisition_tail_noise,
       propagate_acquisition_tail_noise,
   )

   acquired = estimate_acquisition_tail_noise(
       raw_matrix,  # (FID sample, acquired point), acquisition order
       dwell_time_s=csi.handler.to_csi_acquisition().dwell_time_s,
       bands_hz=[(-330, -150), (200, 420)],
   )
   weights = np.r_[np.ones(600), np.zeros(raw_matrix.shape[1] - 600)]
   voxel_sigma = propagate_acquisition_tail_noise(
       acquired, point_weights=weights, n_spatial_points=raw_matrix.shape[1]
   )

.. autofunction:: hypermri.analysis.spectroscopy.noise_estimation.estimate_acquisition_tail_noise

.. autofunction:: hypermri.analysis.spectroscopy.noise_estimation.propagate_acquisition_tail_noise

.. autoclass:: hypermri.analysis.spectroscopy.noise_estimation.AcquisitionTailNoise
   :members: provenance

The same noise onset suggests the CSI cut-off ``signal_end``, the number of
acquired points kept by the reconstruction: ``suggest_signal_end`` returns the
onset (plus an optional margin, default 0). Keeping fewer points loses signal;
keeping more adds noise-only points. ``CSI.plot_signal_decay`` prints the
suggestion and marks it in the decay panel as "Suggested end"; no default
``signal_end`` changes.

.. code-block:: python

   from hypermri.analysis.spectroscopy import suggest_signal_end

   signal_end = suggest_signal_end(
       raw_matrix,
       dwell_time_s=csi.handler.to_csi_acquisition().dwell_time_s,
       bands_hz=[(-330, -150), (200, 420)],
   )

.. autofunction:: hypermri.analysis.spectroscopy.noise_estimation.suggest_signal_end

Reduced chi-squared of the fit
------------------------------

Every ``fit_data_pseudo_inv`` result reports how well the model explains each
spectrum relative to the noise:

* ``fit_n_free_parameters``: real parameters the fit adjusted (two per active
  complex amplitude, one per free frequency and T2*, shared T2* and phase
  counted once, plus the Voigt width);
* ``fit_residual_degrees_of_freedom``: ``2 * n_fitted_samples -
  fit_n_free_parameters``;
* ``fit_reduced_chi_squared``: ``fit_rss / sigma**2 /
  fit_residual_degrees_of_freedom``.

For white noise of the right sigma and a model that explains the signal, the
reduced chi-squared is near 1 (spread ``sqrt(2 / degrees of freedom)``).
Clearly larger values mean signal is left in the residual or sigma is too
small; clearly smaller values mean sigma is too large.

The fit uses an acquisition-tail sigma only when the input carries it as the
coordinate ``noise_sigma_quadrature`` with ``noise_source="acquisition_tail"``.
``attach_acquisition_tail_noise`` checks the propagated sigma (source, positive
finite values, voxel dimensions and coordinates) and attaches it;
``scan.interactive_fit(noise=voxel_sigma)`` does the same for the data of the
interactive workflow, so its fit, frequency alignment and single-voxel refits
all use this sigma. ``refit_spectroscopy_voxel`` (the "Refit" button of the
review tab) recomputes ``fit_rss``, ``fit_n_free_parameters``,
``fit_residual_degrees_of_freedom`` and ``fit_reduced_chi_squared`` of the
refitted voxel for its enabled lines and effective search windows. Without the
coordinate the reduced chi-squared is NaN. Pass other noise
explicitly with ``fit_spectrum(noise=...)`` or ``calculate_reduced_chi_squared``.

.. code-block:: python

   from hypermri.analysis.spectroscopy import attach_acquisition_tail_noise

   fids = attach_acquisition_tail_noise(csi.primary_data, noise=voxel_sigma)
   fitted = fit_data_pseudo_inv(fids, fit_params)
   fitted.fit_reduced_chi_squared  # finite, noise_source="acquisition_tail"

.. autofunction:: hypermri.analysis.spectroscopy.noise_estimation.attach_acquisition_tail_noise

.. autofunction:: hypermri.analysis.spectroscopy.fit_diagnostics.calculate_reduced_chi_squared

.. autofunction:: hypermri.analysis.fitting.models.spectroscopy.fid_model.count_free_parameters

.. autofunction:: hypermri.analysis.fitting.models.spectroscopy.fid_model.fitted_sample_count

Finding problem voxels
~~~~~~~~~~~~~~~~~~~~~~

``summarize_fit_bound_flags`` lists, per voxel, the lines whose frequency or
T2* ended at a search bound (from ``fit_freq_status`` and ``fit_t2_status``).
``order_fit_problem_voxels`` orders the fitted voxels for review: voxels with
such a flag first, then by descending ``fit_reduced_chi_squared``. The review
tab uses both: its parameter table has a "Bound" column, a label shows the
voxel's ``chi2/N`` and the alignment shift ``frequency_alignment_offset_hz``
applied to its data, the spatial overlay offers ``fit_reduced_chi_squared``,
and "Next problem voxel" steps through the order.

.. code-block:: python

   from hypermri.analysis.spectroscopy import order_fit_problem_voxels

   order = order_fit_problem_voxels(fitted)
   worst = {dim: int(order[dim][0]) for dim in order.attrs["spectrum_dims"]}
   fitted.fit_amps.isel(worst)

.. autofunction:: hypermri.analysis.spectroscopy.fit_diagnostics.summarize_fit_bound_flags

.. autofunction:: hypermri.analysis.spectroscopy.fit_diagnostics.order_fit_problem_voxels

Settings that produced a fit
----------------------------

A fit result copies the attributes and voxel coordinates of its input, so the
settings applied before the fit are stored with it without being passed again:

* ``signal_end``, ``signal_end_axis`` and ``signal_end_selection`` from
  ``CSI.create_filtered_data``: the number of kept points (``"none"`` without
  truncation), whether they count acquired k-space points or FID samples, and
  whether the first or the highest-signal points were kept;
* ``applied_mask`` from ``apply_mask_to``: the name of the mask;
* the coordinate ``frequency_alignment_offset_hz`` from
  ``align_spectra_by_fit_frequency``: the total shift in Hz applied to each
  voxel (it adds up over repeated alignments and returns to zero when one is
  undone), next to the attrs ``frequency_alignment_method`` and
  ``frequency_alignment_global_mean_hz``;
* ``lineshape`` (``"lorentz"`` or ``"voigt"``), always written by
  ``fit_data_pseudo_inv``;
* ``fit_start_initial_freqs_hz`` and ``fit_start_initial_t2s_s`` (plus the
  Voigt ``fit_start_initial_gauss_fwhm_hz``) from ``fit_best_of_starts``: the
  start maps of every candidate along ``start_candidate``.

All values are plain numbers, strings or float64 arrays, so they survive a
NetCDF round trip. ``apply_mask_to(..., mean_along_space=True)`` keeps its
legacy behaviour and drops the input attributes; use ``reduction="mean"`` to
keep them.

Advanced peak quality
---------------------

The stricter Q0/Q1/Q2 workflow additionally audits the noise-reference origin
and can run a calibrated full-refit detection bootstrap.  These functions are
also available directly from ``hypermri.analysis.spectroscopy``.

.. autofunction:: hypermri.analysis.spectroscopy.quality.make_rf_off_noise_reference

.. autofunction:: hypermri.analysis.spectroscopy.quality.make_noise_frame_reference

.. autofunction:: hypermri.analysis.spectroscopy.quality.make_outside_object_noise_reference

.. autofunction:: hypermri.analysis.spectroscopy.quality.evaluate_spectroscopy_fit_quality

The decision thresholds are passed as an ordinary mapping of the options that
differ from the defaults, for example ``config={"detection_fit_snr": 8.0}``.
One function owns the defaults and validates the mapping.

.. autofunction:: hypermri.analysis.spectroscopy.quality.normalize_spectroscopy_quality_options

The former ``SpectroscopyQualityConfig`` dataclass was removed on 15 November
2026; pass the mapping instead.  The quality functions reject any other
``config`` type with a ``TypeError``.
