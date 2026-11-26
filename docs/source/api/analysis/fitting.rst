Fitting Algorithms
==================

.. note::

   The canonical fitting API and new fitting pipelines are **Beta**. Result
   schemas and model identifiers may change in a ``0.x`` minor release.

Canonical Beta API
------------------

The former ``DATA_PRECISION`` constant of ``hypermri.analysis.fitting`` and
``hypermri.analysis.fitting.fit_maps``, which never controlled fitting
precision, was removed on 15 November 2026. Use the explicit precision
options of the relevant operation;
shared conversion rules are documented in :mod:`hypermri.reconstruction.precision`.

.. automodule:: hypermri.analysis.fitting.api
   :members:
   :show-inheritance:

Worker execution and labelled results
-------------------------------------

Worker options live in ``hypermri.analysis.fitting.execution``; the common
voxel loop lives in ``hypermri.analysis.fitting.voxelwise``. Scientific callbacks
own the model calculation, while ``hypermri.analysis.fitting.result_metadata``
preserves the existing result labels, provenance and reconstruction inputs.
The former ``analysis.fitting.core`` submodules were removed on
15 November 2026. DCE signal/concentration conversion is owned by
``hypermri.analysis.dce.conversions``. This relocation retains worker defaults,
precision, callback behavior and all scientific output contracts.

.. automodule:: hypermri.analysis.fitting.execution
   :members: ExecutionConfig, copy_execution_params, run_indexed_tasks

.. automodule:: hypermri.analysis.fitting.voxelwise
   :members: VoxelwiseExecutor

.. automodule:: hypermri.analysis.fitting.result_metadata
   :members: dataset_reconstruction_array, dataset_reconstruction_values, select_reconstruction_array, set_reconstruction_variable, attach_reconstruction_schema, inherit_input_xarray_metadata, attach_fit_metadata

Magnitude-distribution fitting
------------------------------

The existing histogram-based Rayleigh fit now lives in the named
``hypermri.analysis.fitting.distributions`` module; the previous
``hypermri.utils.mathematics.utils_fit`` module was removed on 15 November
2026. Parameter units follow input signal units; the original
unconstrained fit, covariance and optional static diagnostic plot are preserved.

.. automodule:: hypermri.analysis.fitting.distributions
   :members: rayleigh_pdf_fit, fit_rayleigh

Prepared phase-IDEAL volumes
----------------------------

``hypermri.analysis.fitting.phase_ideal.PHIDEALSolver`` requires a prepared source object.
Its ``data`` is ordered echo, read, phase, slice; ``TEs`` supplies one echo time
per echo in milliseconds. Explicit constructor ``echo_timings`` use seconds
and must agree with source times. Constructor and per-call alternative signals
must have the source shape. Standalone arrays now raise an actionable
``ValueError`` rather than failing later through a missing source. Existing
source, signal, phase, field-map and mask arrays remain unchanged.

``solve`` requires source ``unwrapped_phase`` in radians with the data shape.
When UMPIRE frequency initialization is enabled, both volume operations require
a prepared spatial ``_fmap`` in Hz. An optional ``mask_dict['snr']`` excludes
voxels where True; absent masks include all voxels. Metadata requirements are
checked before voxel processing. A per-call ``optional_signal_data`` applies
only to that call and does not replace the solver's default signal.

``evaluation_solve`` retains the legacy three-frequency workflow and now returns
every voxel in read, phase, slice, component-count, parameter order. Component
counts descend from three to one. Each seven-field record contains three padded
frequencies in Hz, three complex amplitudes in acquired units and the iteration
count; relaxation adds T2-star in seconds before the iteration count. Masked
voxels and unused components retain NaNs. The one-/two-frequency evaluation
variants were not consistently supported and are rejected explicitly; ``solve``
retains its one-to-three-frequency limits.

The frequency selection and numerical voxel solver are unchanged. UMPIRE phase
unwrapping is still unavailable in this package build: ``solve`` can raise
``NotImplementedError`` when its phase-fit assessment requests it, even with a
prepared source and UMPIRE frequency initialization disabled. No replacement
unwrapping algorithm is implied by the volume-orchestration correction.

.. autoclass:: hypermri.analysis.fitting.phase_ideal.PHIDEALSolver
   :members: solve, evaluation_solve

Direct quantitative map fitting
-------------------------------

Use the scientific function directly when the acquisition model is known::

    from hypermri.analysis.quantitative import T1FitOptions, fit_t1

    result = fit_t1(inversion_data, T1FitOptions(t1_values_s=(0.5, 0.7, 0.9)))

The inversion-time coordinate is in milliseconds; the trial T1 grid is seconds
and the fitted T1 map is milliseconds. ``fit_adc`` uses magnitude log-linear
regression with b-value coordinates in s/mm² and reports ADC in mm²/s.
``fit_t2`` fits magnitude echo decays with unweighted nonlinear least squares
in signal space, without an offset, using echo coordinates explicitly in seconds.
It reports ``t2`` in ms, ``s0`` in signal units, ``rss_t2`` in squared signal
units and ``echo_times_s`` for reconstruction. At least three finite magnitudes
strictly above threshold (default 1e-5) and two distinct echo times are required
per voxel. Bounds are S0 >= 0 and 0 < T2 <= 5 s; S0 has no finite upper bound.
The numerical optimizer enforces the fixed T2 bound, recorded in
``maximum_t2_s``. Stored T2 maps use ms and cannot exceed 5000 ms. Refit existing
results explicitly to apply this changed scientific constraint. All echoes
are considered; no uncertainty or physiological quality estimate is calculated.
Failed numerical fits retain NaNs. The named, generic/prepared and legacy registry
paths delegate to the same operation; ``result.hp.reconstruct()`` evaluates the
documented model. ``T2FitOptions`` controls threshold, dynamic dimension and
existing shared voxel execution settings. This quantitative MSME model has
separate physics from spectroscopy T2-star fitting.
``fit_b1`` uses reference-power coordinates in W, periods in sqrt(W), pulse
duration in ms and flip angle in degrees. It selects the existing independent
or region-growing solver. ``fit_ideal`` uses relative frequencies in Hz and
equally spaced centered echo times from the supplied spacing in seconds.

The frozen option containers hold supplied settings; ``None`` leaves a setting
to the existing model owner. Legacy mappings and prepared attribute Datasets
remain supported. A Dataset passed to a named fit is already prepared and is
not prepared again. T1 mappings and ``T1FitOptions`` support ``weight_by_signal``,
``bounds_A``, ``bounds_B``, ``rel_bounds_A``, ``rel_bounds_B``,
``auto_calc_rel_bound``, ``inversion_imperfection`` and ``regularization_alpha``.
The typed coefficient fields use lowercase names (``bounds_a``, ``bounds_b``,
``rel_bounds_a``, ``rel_bounds_b``), translated once to the established mapping keys.
Before this correction, unprepared T1 requests silently lost those settings;
such requests can now change fitted outputs. Omitted settings retain the old
defaults, and explicitly prepared T1 fits retain their existing behavior.

IDEAL mappings and ``IDEALFitOptions`` preserve ``solve_for``,
``init_f_offset_Hz``, ``init_Theta``, ``stop_f_offset_hz``, ``stop_theta`` and
``max_iterations``. The typed initialization names are ``init_f_offset_hz`` and
``init_theta``. Previously unprepared requests silently lost these settings;
such requests can now change fitted outputs. Omission/None retains the actual
solver defaults: both ``Field_offset`` and ``Theta`` are optimized. The earlier
parameter-info text described only ``Field_offset``; this correction changes
the description, not the numerical default.

Field initialization and convergence increments use Hz. Theta is a dimensionless
complex alternating-echo log-amplitude/phase modulation; ``stop_theta`` provides
the respective real/imaginary increment thresholds. ``max_iterations`` is a
positive integer. Solve selection must be a nonempty sequence of distinct
``Field_offset``/``Theta`` names. Selecting only Theta omits ``field_offset``;
the existing result schema exposes no Theta map. No status, uncertainty or generic
IDEAL reconstruction is added. T2-star correction/optimization is not implemented:
quantitative preparation/fitting rejects ``init_T2_star_s`` and ``stop_T2_star_s``
with an actionable ValueError, including on the prepared path. Remove these
settings for field/Theta fitting; relaxation requires a separate T2-star model.
The lower-level legacy solver is unchanged.

T1 coefficients and absolute bounds have input signal units. Relative bounds
scale by each voxel's maximum signal magnitude and override absolute bounds;
automatic bounds override supplied relative bounds. Complex coefficients apply
the same limits to real and imaginary parts independently. Any finite coefficient
bound selects bounded least squares instead of ridge regularization. With no
finite bounds, positive ``regularization_alpha`` penalizes all fitted coefficients.
Weighting uses ``abs(signal)/(max(abs(signal))+1e-8)`` as a dimensionless heuristic,
not a calibrated noise likelihood. ``rss_t1`` reports the correspondingly weighted
residual sum of squares in squared signal units and excludes the ridge penalty.
Supplied settings are copied into prepared parameters and fit provenance.

The generic facade selects these four functions from an ordinary dictionary.
Compatible registry entries only translate prepared attributes and delegate;
named fits do not return through the facade. Existing solvers, numerical defaults,
precision, labelled coordinates, NaN failure behavior and reconstruction contracts
are unchanged. No additional fit status or uncertainty is inferred. Two additional
attributes record copied prepared settings (``fit_prepared_options``) and the
actual package version (``fit_package_version``); implicit numerical defaults
remain owned by that version's scientific model. Existing provenance remains.

B1 region-growing diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Request both region growing and debug output to retain the fitting order::

    from hypermri.analysis.quantitative import fit_b1

    result = fit_b1(calibration, {
        "fa_deg": 90.0,
        "trf_ms": 1.0,
        "sint": 1.0,
        "use_region_growing": True,
        "seed_voxel": {"read_idx": 0, "phase_idx": 0},
        "return_debug_info": True,
    })
    order = result["fit_order_map"]
    prepared_options = result.attrs["fit_prepared_options"]

``fit_order_map`` is a saved data variable on ``(read_idx, phase_idx)`` with
the input coordinates. Its float64 values record the zero-based traversal
order: the seed is 0 and unvisited voxels are NaN. The same debug request
retains ``stop_index_map_iter1`` and the boolean ``region_mask``. These three
maps are omitted when ``return_debug_info=False`` (the default); enabling
``store_debug_info`` alone does not request these dataset variables.

The ordinary maps are ``amplitude``, ``period``, ``refpower_90deg_W``,
``rss_b1``, ``stop_index`` and ``n_valid_samples``. With two-iteration region
growing (the default), their ``_iter1`` counterparts retain the first-pass
results. All ordinary maps retain the non-dynamic input dimensions, including
singleton dimensions; the debug maps retain their existing two-dimensional
layout. ``has_init_params_map`` records the presence of the internal neighbor
initialization dictionary; that dictionary has never been a saved dataset
variable.

``fit_prepared_options`` and ``fit_package_version`` are retained regardless
of the debug setting by ``fit_b1``, the generic/legacy fitting facades and the
surface-coil fitting helper. The options attribute is an independent nested
copy of the prepared settings. These attributes describe newly fitted results;
loading an older result does not add them retroactively.

.. automodule:: hypermri.analysis.quantitative.fitting
   :members: fit_t1, fit_adc, fit_b1, fit_ideal, fit_t2

.. automodule:: hypermri.analysis.quantitative.options
   :members: T1FitOptions, ADCFitOptions, B1FitOptions, IDEALFitOptions, T2FitOptions

.. autofunction:: hypermri.analysis.fitting.models.quantitative.t2.prepare_t2_fit_params_dataset

.. autofunction:: hypermri.analysis.fitting.models.quantitative.t2.monoexponential_t2_model

.. autofunction:: hypermri.analysis.fitting.models.quantitative.t1.get_t1_params_info

.. autofunction:: hypermri.analysis.fitting.models.quantitative.t1.prepare_t1_fit_params_dataset

.. autofunction:: hypermri.analysis.fitting.models.quantitative.ideal.get_ideal_params_info

.. autofunction:: hypermri.analysis.fitting.models.quantitative.ideal.prepare_ideal_fit_params_dataset

.. autofunction:: hypermri.analysis.fitting.models.quantitative.ideal.fit_ideal_dataset

Shared spectroscopy execution
-----------------------------

For the FID model, ``api.fit(data, model="FID", fit_params=params)`` and
``spectroscopy.fit_data_pseudo_inv(data, params)`` use the same
spectroscopy workflow. It aligns parameter maps, runs the single-spectrum
solver, and assembles results with units and input metadata. Explicit T2*
bounds are handled consistently through both entry points.

Serial fitting, serial fitting with a progress bar, and joblib workers receive
the same aligned input and parameter maps. The sample dimension is placed last
internally, while result dimensions follow the non-spectral input order. These
execution paths return raw solver outputs to one shared result-assembly step,
so units, coordinates and metadata are attached consistently.

The workflow lives in :mod:`hypermri.analysis.fitting.spectroscopy`. Direct
spectroscopy callers can import its functions without selecting a registry key::

    from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv

    params = def_fit_params(fid_data, fit_params=options)
    result = fit_data_pseudo_inv(fid_data, params)

The numerical model, parameter preparation and scientific defaults live in
:mod:`hypermri.analysis.fitting.models.spectroscopy.fid_model`. That module
does not import fitting workflows or call the generic API. The domain entry
``hypermri.analysis.spectroscopy.fit_spectrum`` adds supplied-noise diagnostics
around the same labelled workflow and numerical solver.

Use ``fid_model`` for direct numerical operations. The scientific model
identifiers ``FID`` and ``fid_pseudo_inv`` select the same registered model.

.. automodule:: hypermri.analysis.fitting.models.spectroscopy.fid_model
   :members: get_fid_pseudo_inv_params_info, def_fit_params, fid_basefunc, solve_for_complex_amplitudes, fit_freq_pseudo_inv, fit_t2_pseudo_inv, fid_model, reconstruct_fid_signal, final_fit_errors

Spectroscopy frequency parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supply initial frequencies in metabolite order using ``metabs_freqs_hz``.
The notebook spelling ``metabs_freqs_Hz`` is accepted and normalized to that
name. If both spellings are supplied, their values must agree.

Explicit Hz values need only a sampling-time axis, for example::

    params = api.prepare_fit_params(
        fid_data,
        model="FID",
        fit_params={
            "metabs": ["pyruvate", "lactate"],
            "metabs_freqs_hz": [-70.0, 857.0],
        },
    )
    result = api.fit(fid_data, model="FID", fit_params=params)

When no ppm axis is present, derived ``metabs_freqs_ppm`` entries remain
``None``. Explicit ``metabs_freqs_ppm`` values retain precedence over Hz values
and require a calibrated ``freq_axis_ppm`` coordinate. Missing entries use
the corresponding Hz value, or a known chemical shift if calibration exists.
Scalar values address the first metabolite; shorter lists leave the remaining
entries unspecified. Invalid values, extra entries and missing required
calibration raise a descriptive ``ValueError`` during parameter preparation.

Search settings and iterative refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

T2* guesses, bounds and search widths accept a scalar for every metabolite or
a one-dimensional sequence with exactly one entry per metabolite. The legacy
``metabs_t2s`` key remains supported; ``initial_t2s_s`` takes precedence when
both are provided. Unlike frequency lists, these sequences cannot omit entries.

Preparation checks the following constraints before running the solver:

* ``min_t2_s`` is positive, ``max_t2_s >= min_t2_s``, and each initial T2*
  lies inside its bounds. All values must be finite.
* ``metabs_freq_ranges`` and ``metabs_t2_ranges`` are finite, non-negative
  **full widths**. A width of 10 Hz means initial frequency +/- 5 Hz. Zero
  freezes the corresponding parameter.
* ``npoints``, ``niter`` and ``rep_fitting`` are positive integers;
  ``zoomfactor`` is finite and positive.
* ``min_peak_separation_hz`` and ``amp_ridge_lambda`` are finite and
  non-negative. ``snr_threshold`` follows the same rule or may be ``None``.
* ``search_order`` is ``"amplitude"`` (default) or ``"metabolite"``, and
  ``joint_refinement`` is a boolean (default True); see
  `Search execution and diagnostics`_.
* The start frequency and the full width own the frequency bounds:
  ``freq_range_lower_bound_hz`` and ``freq_range_upper_bound_hz`` are always
  ``initial_freqs_hz -/+ metabs_freq_ranges / 2``. Stored bounds that differ,
  for example after ``initial_freqs_hz`` or ``metabs_freq_ranges`` was edited
  in place, are replaced with a ``UserWarning`` naming the operation and the
  lines. ``fit_data_pseudo_inv``, ``refit_spectroscopy_voxel`` and the
  uncertainty and diagnostic functions apply this rule. Change the start or
  the range, not the bounds. ``refit_spectroscopy_voxel`` computes the bounds
  of every line from its start and the selected voxel's width (or the
  reviewed row's width) and does not read stored bounds.
* In every spectrum, the start frequencies (``initial_freqs_hz``) of lines
  that are not skipped lie at least ``min_peak_separation_hz`` apart. The fit
  rejects every frequency set with a closer pair, so a start of that kind
  would give the first frequency step no valid comparison value, and that step
  could move a line that already sits on its peak. ``fit_data_pseudo_inv``
  checks this before the first spectrum is fitted, and
  ``add_metab_to_fit_params`` checks it for the new line against the existing
  starts (after refreshing them from ``fit_results``). The error names the
  pair, the spectrum and the number of affected spectra. Move the starts
  apart, lower ``min_peak_separation_hz``, or skip or remove one line.
* ``skip_fitting`` contains booleans or 0/1 values. True excludes a metabolite
  before numerical fitting. A scalar applies to every metabolite; edit the
  prepared map for voxel-specific exclusions. Missing or invalid mask values
  are rejected even when introduced after parameter preparation.

Invalid settings now raise parameter-specific ``ValueError`` messages rather
than reaching the search with unusable values. Existing valid defaults and
numerical search methods are unchanged.

Use ``initialise_fit_params_from_results`` to reuse fitted guesses and optionally
change the full search widths. ``None`` keeps the existing widths; scalars and
per-metabolite sequences are supported, as are labelled DataArrays varying by
voxel. Results and range maps must have matching coordinate labels in the same
order as the parameter maps; dimension order may differ. Mismatches are rejected
before assignment can introduce missing values. The function returns a copy
and recalculates frequency bounds around the new guesses.

Adding a line
~~~~~~~~~~~~~

``add_metab_to_fit_params`` is the one owner of adding a line to prepared
parameters; the Add button of the review plugin (``ReviewFitPlugin``) calls
it too. It returns a copy with the new line and leaves its input unchanged.
Settings that are not given take the ``def_fit_params`` defaults: a 70 Hz
search window, a 25 ms start T2*, a 40 ms T2* range, a 5 ms minimum and a
70 ms maximum (1 s for a Voigt fit). With ``freeze_other_freqs=False`` and
``freeze_other_t2s=False``, adding a line therefore gives the same parameters
as preparing the extended line set with ``def_fit_params``; the default
``True`` narrows the windows of the existing lines around their starts. The
function also extends ``metabs``, ``metabs_freqs_hz``, ``metabs_freqs_ppm``
and, with ``t2_coupling``, the coupling groups. It raises ``ValueError``
before changing anything when the name is in use, the start lies outside the
frequency axis, the window is wider than the spectral width, the T2* start
lies outside its bounds, the start is closer than
``min_peak_separation_hz`` to another line, or the coupling is invalid
(unknown leader, a leader that is itself coupled, a ratio that is not
positive).

.. code-block:: python

    params = add_metab_to_fit_params(
        180.0,
        new_metab_name="pyr_2",
        fit_params=params,
        t2_coupling=("pyr", 1.0),
    )

``remove_metab_from_fit_params`` is its counterpart and the owner of removing
a line; the review session (``FitSession``) applies staged removals through
it. The line leaves every variable with a ``metabolite`` dimension, the
per-line option attributes (``metabs``, ``metabs_freqs_hz``,
``metabs_freqs_ppm`` and the other per-line start options) and
``t2_coupling``, so removing a line gives the same parameters as preparing the
reduced line set with ``def_fit_params``. Removing a line that leads the T2*
of another line raises ``ValueError``: remove the dependent lines first.

.. code-block:: python

    params = remove_metab_from_fit_params(params, metab_name="pyr_2")

Start candidates
~~~~~~~~~~~~~~~~

The frequency, T2* and Voigt-width search starts from the prepared start
values and accepts only steps that lower the objective, so a fit can end in a
local minimum that depends on its start. Amplitudes have no start: they are
solved linearly at every step. Three kinds of start are built with the
functions above, and ``fit_data_pseudo_inv`` fits any of them unchanged:

* global starts: ``def_fit_params`` (the default);
* from previous results: ``initialise_fit_params_from_results``;
* nested: ``add_metab_to_fit_params`` with ``fit_results`` of the model
  without the new line and both freeze flags ``False``. The existing lines
  start at their fitted values. Because the new line can take a zero
  amplitude, the nested fit with the default temporal objective does not
  end with a higher RSS than the model without it (tested for Lorentz,
  Voigt and phase-band fits).

``fit_best_of_starts`` fits several candidates of the same model and keeps,
per spectrum, the result with the lowest ``fit_objective`` (for the default
temporal objective this is ``fit_rss`` and thus the lowest chi-squared). Ties
go to the candidate listed first, and a spectrum in which every candidate
failed keeps candidate 0. ``fit_start_candidate`` records the kept candidate
and ``fit_start_objective`` the objective of every candidate;
``fit_start_initial_freqs_hz`` and ``fit_start_initial_t2s_s`` keep every
candidate's start maps so the starts can be repeated. The candidates
must share metabolites, option attrs, skipped lines and line order; they may
differ in their start and search-range maps. This is an option: the default
fit uses the global starts only. Run time is the sum of the candidate fits.

.. code-block:: python

    nested = add_metab_to_fit_params(
        25.0,
        new_metab_name="c",
        fit_params=two_line_params,
        fit_results=two_line_fit,
        freeze_other_freqs=False,
        freeze_other_t2s=False,
    )
    best = fit_best_of_starts(signals, start_params=[three_line_params, nested])

Coupled T2* values
~~~~~~~~~~~~~~~~~~

``t2_coupling`` ties the T2* of a dependent line to a leader line, for example
two components of one resonance that share their decay:

.. code-block:: python

    params = def_fit_params(signals, {"metabs": ["pyr", "pyr_2"], ...})
    params = params.assign_attrs(t2_coupling={"pyr_2": ("pyr", 1.0)})
    result = fit_data_pseudo_inv(signals, params)

Each entry means ``T2*(dependent) = t2_ratio * T2*(leader)`` with
``t2_ratio > 0``; a plain leader name means ``t2_ratio = 1.0``. In Voigt fits
the ratio applies to the Lorentz part, and each line keeps the shared Gaussian
width. Only the leader's T2* is searched. Every candidate sets both lines and is
rejected when either leaves its own ``[min_t2_s, max_t2_s]``; the dependent's
``initial_t2s_s`` and ``metabs_t2_ranges`` are not used. Fitting stops before
work when ``t2_ratio`` times the leader's start T2* lies outside the dependent's
bounds in any spectrum. A leader with ``metabs_t2_ranges`` 0 freezes the
dependent. If the leader is skipped or removed by ``snr_threshold``, the first
remaining dependent in metabolite order becomes the leader. It searches with
the old leader's ``metabs_t2_ranges`` times its own ``t2_ratio``, so a frozen
group stays frozen, and the other dependents follow it with their ``t2_ratio``
divided by its ``t2_ratio``. All T2* ratios within the group are kept; a lone
remaining dependent is fitted independently. Names must be fitted metabolites,
a line cannot follow itself and a leader cannot itself be a dependent. The
option is recorded in the result attrs, used by the bootstrap uncertainty and
the interactive refit, and rejected with ``detection_target`` and by the local
covariance.

A ratio band couples more loosely. Instead of one ``t2_ratio``, give a
``(low, high)`` pair with ``0 < low <= high``:

.. code-block:: python

    params = params.assign_attrs(t2_coupling={"pyr_2": ("pyr", (0.75, 1.25))})

Every accepted fit state then keeps
``low <= T2*(dependent) / T2*(leader) <= high``. Unlike a fixed ratio, the
dependent keeps its own ``initial_t2s_s`` and ``metabs_t2_ranges`` and is
searched itself; its candidates are rejected when the ratio leaves the band or
the T2* leaves its own ``[min_t2_s, max_t2_s]``. A step of the leader moves the
dependent with its current ratio. The leader is then searched once more on
its own: its banded dependents keep their T2* and the band limits the leader's
candidates. Without this second step a weak leader could change its T2* only
together with a strong dependent and would follow it. In Voigt fits the band
applies to the Lorentz part; the Gaussian step keeps each banded line's total
width and rejects widths whose derived ratio leaves the band. The start ratio of
``initial_t2s_s`` must lie in the band in every spectrum, otherwise fitting
stops before work. Start maps taken from an earlier fit, for example from
``add_metab_to_fit_params``, often miss the band in some voxels. Set
``t2_coupling_clip_start=True`` to move such starts to the nearest admissible
value instead: a banded dependent's start is clipped to ``low`` or ``high``
times its leader's start, intersected with its own ``[min_t2_s, max_t2_s]``,
and a leader's start is clipped so that ``t2_ratio`` times it lies within the
bounds of every fixed dependent. Admissible starts are not changed, the
caller's parameters are not modified, and spectra without any admissible start
still raise. ``(r, r)`` is the fixed ratio ``r``. A banded dependent never
takes over a missing leader: it follows a fixed dependent that does, with its
band divided by that line's ``t2_ratio``, and is fitted independently when no
fixed dependent of the group remains. Fits whose true ratio lies outside the
band end at the band edge, so check results that sit on ``low`` or ``high``.

Leader and dependent are metabolite names, but the coupling describes physical
lines. When the frequency ranges of coupled lines overlap, the frequency
search could move a name onto another line of its group, for example the
leader name onto the weak line, and the fit would then apply the T2*
constraint to the wrong pair. Names therefore belong to frequency positions:
the k-th lowest line of a coupling group (a leader and its dependents) carries
the group's name with the k-th lowest initial frequency. As for the
relabelling below, this order uses the mean of ``initial_freqs_hz`` over all
spectra, with equal means ordered by metabolite name, so every spectrum uses
the same order even where its own starts differ. Every frequency candidate is
scored in this assignment, and the T2* values, coupling roles and
``phase_band_tolerance_deg`` entries stay with the names. A line that passes
another line of its group thus takes over that line's name and role instead
of exchanging the constraint.

Each line keeps its own frequency window: its ``metabs_freq_ranges`` and the
bounds around its start move with the line when it passes another line, and
the line is only searched inside that window. Windows of a group may
therefore have any widths and positions, for example a narrow window for a
line added later inside the wide window of an earlier one. A frozen line
(``metabs_freq_ranges`` 0) keeps its frequency even when another line passes
it, but then reports under the other name. For coupled lines
``fit_component_index`` equals the metabolite index; it does not record which
line started where. Start coupled lines in their expected frequency order.

Phase band for overlapping lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With free complex amplitudes, overlapping lines can cancel each other: two
large lines with opposite phases may describe noise or a lineshape error
instead of signal. The phase band ties the listed lines to one phase law:

.. code-block:: python

    from hypermri.analysis.spectroscopy import phase_band_tau_s_from_scan

    params = params.assign_attrs(
        phase_band_tolerance_deg={"OMPD1_1": 3.0, "OMPD1_2": 3.0, "C1": 20.0},
        phase_band_tau_s=phase_band_tau_s_from_scan(scan),
    )
    result = fit_data_pseudo_inv(signals, params)

Each listed line ``k`` gets the complex amplitude
``a_k = A_k * exp(1j * (phi0 + 2*pi*f_k*tau + delta_k))`` with ``A_k >= 0``
and ``abs(delta_k) <= tolerance_k``. ``f_k`` is the fitted line frequency in
Hz. ``phi0`` is a zero-order phase, solved per spectrum and shared by all
listed lines. ``tau = phase_band_tau_s`` is the known delay in seconds from the
time at which all lines share one phase (for example the excitation pulse
centre) to ``t = 0`` of ``sampling_time_axis_s``; it is not fitted.
Tolerances lie in ``[0, 90]`` degrees: 0 puts the line exactly on the law,
larger values allow for lineshape and phase errors. Because ``A_k >= 0``, a
listed line cannot turn to the opposite phase to cancel a neighbour. Lines that
are not listed keep free complex amplitudes.

For Bruker ``<User:lucaCSI4>`` CSI scans as loaded by hypermri,
:func:`~hypermri.analysis.spectroscopy.phase_timing.phase_band_tau_s_from_scan`
returns ``tau = TE + dwell`` (``PVM_EchoTime`` plus ``PVM_DigDw``). Free fits
of seven well-resolved pyruvate CSI scans at 7 T follow this rule with an rms
deviation of 8 microseconds (TE 1.10-1.26 ms, dwell 0.19-0.50 ms); ``tau = TE``
is about 0.38 ms too short. Other Bruker methods get the same rule and a
``UserWarning`` that the value has not been verified. The rule then also counts
the points removed at loading (``cut_off``) and, for stock methods, the
digital filter delay of ``PVM_DigShift`` points that their raw FID still
contains: ``tau = TE + (1 + cut_off - PVM_DigShift) * dwell``, which is
``TE + 3 * dwell`` for the default ``cut_off`` of 70 and ``PVM_DigShift`` 68.
``<Bruker:NSPECT>`` stores its echo time as ``EchoTime``.
``<Bruker:SINGLEPULSE>`` has no echo time; its pulse program starts the
acquisition ``DeadTime`` after the end of the excitation pulse, so ``TE`` is
replaced by the excitation pulse length times its rephasing factor plus
``DeadTime`` (0.128 ms * 50 % + 0.05 ms = 0.114 ms for a typical block pulse).
The rule assumes a
pulse-acquire FID and does not apply to spin-echo methods such as PRESS. Check
an unverified value against free fits of several well-separated lines before
relying on it.

For every frequency and T2* candidate the amplitudes are solved under the
band. Free lines are projected out, each listed amplitude is a non-negative
combination of the directions at ``-tolerance``, 0 and ``+tolerance``
(non-negative least squares), and ``phi0`` is searched on a 24-point grid with
a bounded refinement to 1e-4 rad. When the unconstrained least-squares
amplitudes already satisfy the band, that is when one common ``phi0`` puts
every non-zero listed amplitude inside its tolerance arc, they are the
constrained optimum and are returned without the search; this applies to
roughly half of the solves in a typical fit and gives the same amplitudes to
1e-9 relative. For a line with tolerance 0 the search reaches the optimum
only to its 1e-4 rad precision, so the two can differ by up to about 1e-4
relative in amplitude, with the early exit never giving a higher residual.
The band requires ``amp_solver='pinv'``.
Tolerances follow the line names when lines are skipped or removed by
``snr_threshold``. The options are recorded in the result attrs, used by the
bootstrap uncertainty and the interactive refit, and rejected with
``detection_target`` and by the local covariance. To decide which listed lines
the data actually need, see ``eliminate_unnecessary_lines`` in
:doc:`spectroscopy`.

``scripts/example_t2_coupling_and_phase_band.py`` fits synthetic doublets
with and without ``t2_coupling`` and the phase band and prints the bias and
scatter of each setting, including a wrong ``tau`` and a band that excludes
the true T2* ratio.

Metabolite order of overlapping lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the frequency search ranges of two lines overlap, the solver can return
the lower line under the label of the higher one in some voxels. By default
(``keep_fit_order=False``) ``fit_data_pseudo_inv`` therefore relabels the
fitted components per spectrum so that their fitted frequencies follow the
order of the initial frequencies. The target order uses the mean of
``initial_freqs_hz`` over all spectra, so spatially varying starts are
allowed; equal means are ordered by metabolite name. For example, with
``metabs=["OMPD1_1", "OMPD1_2"]`` started at 10 and 12 Hz, ``OMPD1_1`` always
holds the lower fitted frequency.

All per-metabolite outputs (amplitudes, frequencies, T2*, widths, search
diagnostics, amplitude-filter flags) move together, so the fitted signal is
unchanged. ``fit_component_index`` records which solver component, counted in
metabolite order, is reported under each label. The optimization history keeps
the solver order. Only lines with the same constraint role are exchanged:
skipped lines, lines in ``t2_coupling`` (leaders and dependents) and lines with
different ``phase_band_tolerance_deg`` entries keep their label; coupled lines
already get their names by frequency position during the fit (see
`Coupled T2* values`_). Statistical
mode (``detection_target``) does not reorder. Set ``keep_fit_order=True`` to
keep the solver order.

Amplitude solver selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameter preparation and direct ``solve_for_complex_amplitudes`` calls use
the same solver names, aliases and precedence rules:

* ``amp_solver="pinv"`` uses the pseudoinverse of the complex FID basis.
  ``"pseudo_inv"`` and ``"pseudoinv"`` are aliases.
* ``amp_solver="normal_eq"`` retains the historical plain-transpose formula
  ``pinv(X.T @ X) @ X.T @ y``. ``"normal"`` is an alias. For complex data,
  this differs from the conjugate-transpose normal equations.
* ``amp_solver="ridge"`` uses conjugate-transpose normal equations plus
  ``amp_ridge_lambda * I``. ``"tikhonov"`` is an alias. A singular solve
  falls back to the pseudoinverse of the regularized matrix.

An explicit ``amp_solver`` takes precedence over ``pseudo_inv``. When omitted
or ``None``, ``pseudo_inv=True`` selects pinv (the default); ``False`` selects
normal_eq. Names ignore case and surrounding whitespace. Ridge lambda defaults
to zero and affects only ridge; preparation validates it as finite and
non-negative. Direct amplitude calls retain their historical behavior of
clamping negative lambda to zero.

Amplitude solving returns temporal component signals with shape
``(samples, metabolites)`` and the complex amplitudes in metabolite order,
even for spectral input. Sum the components along axis 1 to reconstruct the FID.

Search execution and diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Frequency and T2* searches share spectrum preparation and candidate evaluation.
Each search converts spectral input to an FID once, caches the measured spectrum,
and reuses the component signals returned by the amplitude solver. Frequency
separation, T2* validity, grid traversal and bound clipping remain explicit in
their respective search functions.

Each grid step searches one line at a time while the others keep their current
values, so the order of the lines can change the result when lines overlap.
``search_order`` sets this order for every frequency, T2* and Voigt
Lorentz-width step:

* ``"amplitude"`` (default) solves the amplitudes once at the start of each
  step and visits the line with the largest fitted signal first. The signal of
  a line is the norm of its fitted component FID, so a narrow line can rank
  above a broad line of larger area. Equal signals keep metabolite order. The
  order is fixed within a step and is computed again at the next step. This
  costs one amplitude solve per step with two or more lines, which is small
  against the ``npoints`` candidates of each line; single-line fits solve
  nothing extra.
* ``"metabolite"`` visits the lines in metabolite order and adds no
  calculation.

``"amplitude"`` makes the visiting order independent of how the metabolite list
is written. In a synthetic two-line study (spacings 3 to 8 Hz, area ratios 1 to
10, start errors 0 to 4 Hz, 54 cases), listing the weaker line first gave a
worse final residual than listing the stronger line first in 28 cases and a
better one in 12. With ``"amplitude"`` both listings ended within 0.1 % of the
strong-first residual, at about 1 % more run time. Neither order is best in
every case: in a noise-free doublet whose weak line must pass the strong line
inside a narrow window, the grid alone ended up to 1.2 Hz off the weak line
in ``"amplitude"`` order after the default 11 repetitions, while
``"metabolite"`` found it exactly; the joint refinement described below
recovers it in both orders.
The optimization history records the visiting order in its
``metabolite_index`` field; diagnostics rows stay in metabolite order.

**Migration:** fits made before November 2026 used metabolite order. Set
``search_order="metabolite"`` to reproduce them. Fits with one line, and fits
whose lines are visited strongest-first in metabolite order anyway, do not
change. Where lines overlap, frequencies, T2* values and amplitudes can move,
and with ``snr_threshold`` a weak line near the threshold can be removed or
kept differently.

One-line steps cannot move two overlapping lines together. A weak line next to
a strong one can therefore stop in a state from which every single move is
worse although a joint move is better, for example when the strong line's
true frequency lies between two grid points. With ``joint_refinement=True``
(default) the search ends with one local least-squares step
(``scipy.optimize.least_squares``, trust-region reflective) over all free
frequencies and T2* values together, started at the grid result. It
minimizes the grid objective, solves the amplitudes linearly as the grid
does, and keeps every absolute bound, frozen parameter, fixed T2* ratio and
ratio band. The Gaussian FWHM of a Voigt fit stays at its grid value. The
result replaces the grid state only if its objective is strictly lower, the
coupled lines keep their order and the minimum peak separation holds;
otherwise the grid result stays. The step is local: it repairs a grid result
near the right minimum, but it does not find a line the grid left in the
wrong place. The optimization history records it as event
``"joint_refinement"``, accepted or not.

For the FID objective with the ``pinv`` amplitude solver the step uses
analytic derivatives of the residual with respect to the frequencies, T2*
values and ratio bands, in which the linear amplitude solve, including the
common phase and the band edges of a phase band, is accounted for (variable
projection). The magnitude objective (``use_sse_mag=True``) and the other
amplitude solvers use finite differences. Without a phase band both give
the same result to 1e-6 Hz. With a phase band the objective carries the
1e-4 rad tolerance of the common-phase search; both then stop within about
0.01 Hz and 0.05 ms of each other, and the analytic step needs fewer
amplitude solves (on a four-line synthetic fit with three banded lines the
whole fit took 0.82 s instead of 1.02 s, and 0.44 s together with the early
exit of the phase-band solve described above).

In the 54-case synthetic two-line study above (noise-free), the grid alone
ended more than 0.1 Hz off the truth in 45 cases and the refinement in none;
the residual fell in every case, and the total run time rose from 6.9 s to
7.2 s. With complex Gaussian noise (standard deviation 0.02), the cases more
than 0.5 Hz off fell from 30 to 8 and the residual fell in all 54 cases.

**Migration:** fits made before 13 November 2026 used the grid alone. Set
``joint_refinement=False`` to reproduce them. Results now lie between the grid
points: frequencies and T2* values change by up to a fraction of the final
grid spacing where the grid was near the minimum, and by several Hz where it
had stopped in a coupled-line state as described above. Results of serial and
multi-process fits can differ at the 1e-9 relative level, the precision with
which a continuous minimum is determined.

The legacy ``fit_r2_freq`` and ``fit_r2_t2`` arrays contain **sums of squared
errors (SSE)**, not statistical R-squared values. Their ``r2_data`` axis means:

.. list-table:: Search diagnostic columns
   :header-rows: 1
   :widths: 15 85

   * - Index
     - Stored value
   * - ``0``
     - Candidate frequency in Hz or candidate T2* in seconds.
   * - ``1``
     - Sum of squared complex FID residual magnitudes.
   * - ``2``
     - Sum of squared differences between measured and fitted spectrum magnitudes.

Both errors are recorded. ``use_sse_mag=False`` selects candidates using column
1; ``True`` uses column 2. Rejected candidates have infinite errors. Frozen
parameters keep zero-filled diagnostic error slots because their grids were not
evaluated; those zeros do not indicate a perfect fit. Existing variable names,
shapes and legacy unit metadata remain unchanged for compatibility.

Results at a search bound
~~~~~~~~~~~~~~~~~~~~~~~~~

A fitted frequency or T2* that ends at the edge of its search window is set
by the window, not by the data. ``fit_data_pseudo_inv`` therefore reports,
per spectrum and line, ``fit_freq_status`` (against
``freq_range_lower_bound_hz`` and ``freq_range_upper_bound_hz``) and
``fit_t2_status`` (against ``min_t2_s`` and ``max_t2_s``), in the style of
the Voigt ``fit_gauss_fwhm_status``. Each label is, in order of precedence:

* ``not_fitted``: the line was skipped or the spectrum's fit did not succeed;
* ``fixed``: the search width is zero. A fixed-ratio ``t2_coupling``
  dependent follows its leader's T2* search, so it is ``fixed`` only when
  the leader is; a ratio-band dependent uses its own width;
* ``at_lower_bound`` or ``at_upper_bound``: the value lies within 1e-3 Hz
  or 1e-6 s of the bound;
* ``estimated``: the value lies inside the bounds.

Each line is compared with the window it was searched in. After the fitted
components are relabelled by frequency (see
`Metabolite order of overlapping lines`_) the label moves with its
component. ``refit_spectroscopy_voxel`` recomputes both labels of the
refitted lines against the refit's own windows; disabled lines keep their
labels.

Before fitting, ``fit_data_pseudo_inv`` emits a ``UserWarning`` that names
the lines and limits when

* a searched frequency window (``metabs_freq_ranges > 0``) is narrower than
  the spectral point spacing ``bw_hz / nsamplepoints``
  (``1 / (n_samples * dwell)``). Such a window cannot move the line by one
  point, so the fit usually ends at its edge. Use a window of at least one
  point spacing, or 0 to freeze the frequency;
* the windows of two searched lines that are not in one ``t2_coupling``
  group overlap in a spectrum where neither line is skipped, so both lines
  can reach the same peak.

The fit runs unchanged after either warning.

Outer refinement and amplitude exclusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The single-spectrum workflow has three stages: select the fitted sample prefix,
run alternating frequency/T2* searches, then optionally refit retained components.
A shortened spectral input is converted to an FID before truncation. The input
arrays and original parameter maps remain unchanged.

Each of the ``rep_fitting`` passes proposes frequencies followed by T2* values.
Both stages score the complete parameter set and retain their separate best
scores. Only strict improvements are accepted; ties retain the previous values.
Amplitudes are updated with an accepted T2* step. Search widths shrink by
``zoomfactor`` after each stage even if its proposal was rejected. The existing
selection rules and diagnostic axis order are retained.

``skip_fitting=True`` removes the selected component before amplitude solving,
frequency/T2* searches and peak-separation checks. Each voxel fits only its active
metabolites; results retain the complete original metabolite axis. Skipped
components have zero amplitude, their initial frequency/T2* guesses and NaN
diagnostics. Those guesses and their derived linewidths are placeholders, not
estimates. ``fit_low_snr_refitted`` stays false for manually skipped components;
the input ``skip_fitting`` map identifies their exclusion. With every component
skipped, the fitter bypasses numerical work and reconstructs a zero signal.

Positive ``snr_threshold`` compares the absolute complex amplitude with a
threshold; no noise estimate is involved. Components strictly below it are
flagged in ``fit_low_snr_refitted`` and their amplitudes become zero. If some
components remain, fit that subset once more from its fitted guesses, using
the **original** widths and bounds. Thresholding is disabled during this refit.
If all components are excluded, no subset fit is run. Frequencies, T2* values
and diagnostics of excluded components retain their first-fit values.
This amplitude filter applies only to components active after ``skip_fitting``.

All-zero signals bypass numerical searching. They return zero amplitudes,
initial frequency/T2* guesses, NaN diagnostics and false exclusion flags.
These unchanged fallback values do not indicate successful estimation.

A signal with any NaN or infinite sample, in the real or imaginary part, is
not fitted either. Its active components return NaN amplitudes, frequencies,
T2* values and linewidths, and a Voigt fit also returns a NaN Gaussian FWHM;
``fit_status`` is ``invalid_input``. Skipped components keep their placeholders.
``initialise_fit_params_from_results`` and "Init from fit" keep the previous
start values wherever a fitted value is NaN, so a following fit stays valid.

``mask_fit_dataset_where_input_nonfinite(dataset, data)`` returns a copy of
fit parameters or results in which every floating-point or complex variable
is NaN at these voxels, including start values, search ranges and bounds.
Boolean, integer and string variables such as ``skip_fitting`` and
``fit_status`` are unchanged. Masked parameters can be fitted again: the
NaN voxels are skipped as above, the others are fitted normally::

    from hypermri.analysis.spectroscopy import mask_fit_dataset_where_input_nonfinite

    params = mask_fit_dataset_where_input_nonfinite(params, fid_data)
    result = mask_fit_dataset_where_input_nonfinite(
        fit_data_pseudo_inv(fid_data, params), fid_data
    )

The interactive fit workflow applies this to the stored ``fit_params_review``
and ``fit_results_review`` while the "NaN where data NaN" box beside "Save
Review Snapshot" is ticked (the default). Untick it to store the values
unchanged; the setting applies to the next fit, alignment or snapshot.

Absorption and magnitude linewidths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every fit reports two linewidths per line, both in Hz:

``fit_fwhm_hz``
    FWHM of the absorption spectrum (real part of the phased line),
    ``1/(pi*T2*)`` for a Lorentz line.
``fit_magnitude_fwhm_hz``
    FWHM of the magnitude spectrum, ``sqrt(3)/(pi*T2*)`` for a Lorentz line.
    Read this width when spectra are shown in magnitude mode.

Both are computed from the fitted widths alone and describe an infinitely
long line. Line broadening applied to the data before the fit is already
contained in ``fit_t2s`` and therefore in both widths. Broadening caused by
truncating the acquired FID is not included. The ``attrs["definition"]`` of
each variable states its formula. ``estimate_fit_covariance`` reports
``fwhm_std`` and ``magnitude_fwhm_std = sqrt(3) * fwhm_std``;
``estimate_fit_local_uncertainty`` reports ``fwhm_interval_low/high`` and
``magnitude_fwhm_interval_low/high`` from the same T2* endpoints.
``refit_spectroscopy_voxel`` and the review refit recompute both widths of
the refitted voxel from its new ``fit_t2s``.

Optional Voigt lineshape
~~~~~~~~~~~~~~~~~~~~~~~~

The default ``lineshape="lorentz"`` fits every component as
``a * exp(2j*pi*f*t) * exp(-t/T2)``. ``lineshape="voigt"`` multiplies every
component by the Gaussian decay ``exp(-(pi*G*t)**2 / (4*ln 2))``, whose
spectral FWHM is exactly ``G`` in Hz. ``G`` is one width per spectrum, shared
by all metabolites, for example to describe field inhomogeneity within a
voxel. ``G = 0`` reproduces the Lorentz model, and amplitudes remain line
areas at ``t = 0``. The Lorentz default is unchanged when the option is
omitted::

    params = def_fit_params(
        fid_data,
        fit_params={
            "metabs": ["pyruvate", "lactate"],
            "lineshape": "voigt",
            "gauss_initial_fwhm_hz": 10.0,
        },
    )
    result = fit_data_pseudo_inv(fid_data, params)

``gauss_initial_fwhm_hz`` (default 0 Hz) fills the per-spectrum start map
``initial_gauss_fwhm_hz``; ``gauss_max_fwhm_hz`` (default 40 Hz) is the upper
bound and 0 Hz the lower bound. Each ``rep_fitting`` pass runs three steps,
each accepted only on a strict improvement:

1. the frequency search, as in Lorentz mode;
2. a search of every Lorentz FWHM ``1/(pi*T2)`` on a grid in Hz with ``G``
   fixed. Its full width starts at the range allowed by ``min_t2_s`` and
   ``max_t2_s``; ``metabs_t2_ranges`` only enables (> 0) or freezes (0) it;
3. a search of ``G`` over ``gauss_search_width_hz`` (default 40 Hz). Every
   candidate keeps the total Voigt FWHM of each line and moves width between
   its Gaussian and Lorentz parts. ``gauss_search_width_hz=0`` keeps ``G``
   fixed at its start value.

All search widths shrink by ``zoomfactor`` after each pass. Because a line
may be mostly Gaussian, ``max_t2_s`` defaults to 1 s instead of 0.07 s in
Voigt mode.

Voigt results contain ``fit_t2s`` and ``fit_lorentz_fwhm_hz = 1/(pi*fit_t2s)``
for the Lorentz part only, the shared ``fit_gauss_fwhm_hz`` and its
``fit_gauss_fwhm_status`` (``estimated``, ``at_lower_bound``,
``at_upper_bound``, ``fixed`` or ``not_fitted``, and ``neighbour_median``
after the neighbour fill below). ``fit_fwhm_hz`` is the
total absorption Voigt FWHM from the Olivero-Longbothum approximation
``0.5346*f_L + sqrt(0.2166*f_L**2 + G**2)``, which is accurate to about 0.02 %.
``fit_magnitude_fwhm_hz`` is the magnitude FWHM of the Voigt line, solved
numerically to full precision from the Faddeeva function. The dispersion
part also widens the Gaussian part: a pure Gaussian line (``f_L = 0``) has a
magnitude FWHM of about ``1.785*G``.
For strongly overlapping lines the fitted ``G`` can depend on its start
value; compare a start near the expected width with the default.

In weak voxels the fitted ``G`` often ends at 0 Hz (``at_lower_bound``) or
the fit fails. The optional second pass
``refit_gauss_width_from_neighbours(data, fit_params, fit_results,
plane_dims=("read_idx", "phase_idx"))`` fixes ``G`` in each such voxel to
the median of its 8 neighbours in that plane whose status is ``estimated``
and fits only these voxels again. Their ``fit_gauss_fwhm_status`` becomes
``neighbour_median``; voxels without an estimated neighbour, and all other
voxels, keep their results. The filled width comes from the neighbours, not
from the voxel's own data, so the pass runs only when called.

``reconstruct_fitted_signal``, ``fit_results.reconstruct()``,
``add_metab_to_fit_params`` and ``initialise_fit_params_from_results`` carry
``G``. Seeding from results, ``estimate_fit_uncertainty`` and
``compare_metabolite_relevance`` require fit parameters and results of the
same lineshape. For Voigt fits the bootstrap ``t2`` intervals refer to the
Lorentz part, and no interval is reported for ``G``.

The following operations assume Lorentz lines and raise ``ValueError`` for a
Voigt fit before any computation: ``detection_target``,
``store_optimization_history``, ``estimate_fit_covariance``,
``estimate_fit_local_uncertainty``, ``calculate_spectroscopy_fit_diagnostics``,
``fit_spectrum`` with ``noise``, ``evaluate_spectroscopy_fit_quality``,
and ``run_csi_fitting_pipeline(run_quality=True)``.

The interactive ``review_fit``, ``interactive_fit``, ``ReviewFitPlugin``,
``InteractiveFitWorkflowPlugin`` and ``CSIPeakSelectorPlugin`` accept Voigt
fits; they raise ``ValueError`` when fit parameters and results have different
lineshapes. ``refit_spectroscopy_voxel`` refits one Voigt voxel with its own
``G``, starting from the voxel's fitted ``fit_gauss_fwhm_hz``, and updates
``fit_gauss_fwhm_hz``, ``fit_lorentz_fwhm_hz``, ``fit_fwhm_hz``,
``fit_magnitude_fwhm_hz`` and ``fit_gauss_fwhm_status``. In the review table, T2* is the Lorentz part and the
column ``G FWHM`` shows ``G``. The peak selector takes ``lineshape`` and the
``gauss_*`` options as settings, or adopts them from Voigt fit parameters.

The "workers" field above the peak table of ``interactive_fit`` sets the
number of parallel fit processes that Run Fit passes to
``fit_data_pseudo_inv`` as ``n_jobs``: ``1`` fits serially, ``-1`` uses all
cores. It starts at ``-1`` unless ``fit_kwargs`` sets ``n_jobs``. Serial and
parallel fits give the same results; the status line shows the worker count.

In the "Generate Parameters" tab of ``interactive_fit``, "Lineshape" chooses
Lorentz or Voigt, and "Max Gauss FWHM [Hz]" sets ``gauss_max_fwhm_hz``. Each
peak row can name a "T2* leader": equal "Ratio low" and "Ratio high" give a
fixed ratio ``T2*(row) / T2*(leader)`` in ``t2_coupling``, different values a
ratio band. "Clip coupled T2* starts" sets ``t2_coupling_clip_start``. Rows
with "Phase band" checked enter ``phase_band_tolerance_deg`` with their
"Tol [deg]". An empty "Phase band tau [ms]" is derived once with
``phase_band_tau_s_from_scan`` when the parameters are initialized; if that
fails, the tab asks for the value. Loaded fit parameters fill these controls.
The peak rows form one table whose column groups are "Frequency [Hz]",
"T2* [ms]", "Fit", "T2* coupling" and "Phase band". "Range" is the full
frequency search width around the start value and is used as typed, also
below one step of the frequency axis; 0 keeps the frequency fixed. "Freeze"
keeps T2* at its start value (``metabs_t2_ranges`` 0) and disables every T2*
field. With "Init from fit" the start frequency and T2* of every voxel come
from the earlier fit results, so their fields are disabled. The T2* "Range"
field (ms) is then available: filled, it bounds T2* of every voxel to its
fitted start value ± half the range and disables Min and Max; empty, Min and
Max apply, and a fitted start T2* outside them is moved to the nearer bound,
so the bounds can be narrowed after a first fit. The output area names the
lines and the number of voxels whose start was moved.
The peak-detection block starts folded; the peak-selector setting
``peak_detection_collapsed=False`` opens it.

In ``review_fit``, the "Spectrum" row sets how the fit of the current voxel
is drawn: "Fit" chooses "Sum" (the summed model), "Individual" (the lines
stacked on each other, strongest first) or "Separate" (each line on its own),
as ``fit_mode`` does in ``plot``. "Peak positions" draws a dashed vertical
line in the metabolite's colour at each fitted frequency, in the unit of the
frequency axis; it is hidden in the FID view. The SSE always refers to the
summed fit.

Results of ``review_fit`` and ``interactive_fit`` carry ``fit_freqs_ppm``.
``add_fit_freqs_ppm`` recomputes it from ``fit_freqs`` and the result's Hz and
ppm axes, so refitted, shifted and added lines have current ppm values.

Reconstruct the total fitted signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``reconstruct_fitted_signal`` to compare a spectroscopy fit directly with
its input data::

    from hypermri.analysis.fitting import reconstruct_fitted_signal

    fitted_signal = reconstruct_fitted_signal(fit_results, data)
    residual = data - fitted_signal
    rss = (abs(residual) ** 2).sum(fit_results.attrs["spectral_dim"])

This Beta function sums the fitted metabolite components and preserves the
input dimensions, their order, and all coordinates. It evaluates the model at
the times stored with the fit, while retaining the input's sample labels.
Thus integer sample indices are not accidentally aligned with seconds during
xarray arithmetic. Explicit input time axes must match the fitted grid up to
a constant time origin. Non-spectral coordinate mismatches are rejected.

The output follows ``data.attrs["signal_domain"]``: a summed FID for
``"temporal"`` input or its centered FFT for ``"spectral"`` input. If that
attribute is absent, the fit's domain is used. Pass exactly the samples used
for fitting; the function does not crop or extrapolate automatically. Failed
component values propagate to the total signal.

The existing ``fit_results.reconstruct()`` accessor still returns individual
metabolite FIDs with time values as ``acquisition_idx`` labels. It remains
available for callers that need that component-wise representation.

.. automodule:: hypermri.analysis.fitting.spectroscopy
   :members:
   :show-inheritance:

Model-key entry points
----------------------

:mod:`hypermri.analysis.fitting.fit_maps` keeps two entry points that select
a quantitative model by its registry key: ``def_fit_params`` builds the
default parameter dataset and ``fit_quantitative_maps`` runs the voxel-wise
fit. Both delegate once to :mod:`hypermri.analysis.fitting.api`. Model
functions are imported from their owner modules; see :doc:`../public`.

.. automodule:: hypermri.analysis.fitting.fit_maps
   :members:
   :show-inheritance:
