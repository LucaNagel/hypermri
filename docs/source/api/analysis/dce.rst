DCE Analysis
============

.. note::

   DCE fitting and its pipeline interfaces are **Beta**. Model dispatch,
   parameter schemas, and scientific reference values may change in a ``0.x``
   minor release.

This page documents the refactored DCE analysis package under
``hypermri.analysis.dce``.

Signal and concentration conversions
------------------------------------

``hypermri.analysis.dce.conversions`` owns the existing SPGR and linear
enhancement conversions and the shared relaxivity default in 1/(mM s).
The former ``hypermri.analysis.fitting.core.conversions`` module was removed
on 15 November 2026. The relocation preserves saturation handling,
calculation precision, labelled arithmetic and the low-level
``signal_to_concentration`` exception-to-NaN behavior. Workflow entry points
continue to validate their options and resolve the default relaxivity.

.. automodule:: hypermri.analysis.dce.conversions
   :members: DEFAULT_R1_PER_MM_S, resolve_relaxivity, signal_to_concentration, concentration_to_signal

Public fitting surface
----------------------

``run_dce_pipeline`` exposes its auxiliary options explicitly as keyword-only
parameters: ``fitted_t1``, ``motion_correct_dce``, ``renal_masks``,
``gfr_conversion``, ``aif_t1_ms`` and ``aif_t1_s``. Existing positional
parameters remain supported. The compatibility aliases ``blood_to_plasma``
and ``hematocrit`` fill only unspecified canonical AIF options; an explicit
canonical value takes precedence. Unknown or misspelled keyword names raise
``TypeError`` before processing, so check the signature when migrating a call
that previously supplied ignored extra keywords. Scientific defaults, units,
model selection, numerical results and provenance retain their contracts.

.. automodule:: hypermri.analysis.dce.fitting
   :members:
   :undoc-members:
   :show-inheritance:

Model curves and scientific options
-----------------------------------

The numerical implementations live in ``hypermri.analysis.dce.models``. Each
model has an immutable options object for concentration-curve fitting, with
units and parameter bounds documented below. An unspecified field leaves the
existing model default or data-dependent guess in force. This does not
standardize temporal selectors: Tofts retains negative-end slicing, Patlak
does not accept explicit index/time lists, and Sourbron accepts explicit
indices but no explicit time list.

For example, the Sourbron curve function accepts either its existing mapping
or the typed scientific options::

    from hypermri.analysis.dce.models.sourbron import (
        SourbronFitOptions,
        fit_sourbron_separable_compartment_curve,
    )

    options = SourbronFitOptions(fit_start_time_s=10.0, max_nfev=300)
    fitted = fit_sourbron_separable_compartment_curve(
        tissue_mm, aif_concentration=plasma_mm, time_s=times_s,
        fit_params=options,
    )

Tofts uses ``options=ToftsFitOptions(...)`` on its numerical curve helper;
Patlak, Sourbron, 2C-IRF and empirical curve functions accept their respective
options as ``fit_params``. Dataset and pipeline interfaces keep their existing
mapping contracts. All DCE models use the same voxel loop for serial and joblib
execution, while each model retains its fitter and metadata schema. Pass the
same scientific window mapping to quality evaluation; quality
uses the selected model's temporal rules. Patlak's graphical RSS is dimensionless
and remains distinct from concentration-space quality RSS in mM².

One explicit DCE dictionary selects the named workflow and its model-owned
reconstruction, quality, metadata and dtype rules. Named fits call domain
execution directly; the generic fitting facade delegates there. Existing fitting
registry entries and quantitative import paths remain compatible. Pipeline
backend provenance for Patlak, Tofts and Sourbron now records this domain route.

Imports from ``analysis.fitting.models.quantitative`` for Tofts, Patlak and
Sourbron, and the existing curve/reconstruction imports from ``fitting_support``,
remain aliases of these implementations through the HyperMRI 0.2.0 compatibility
window. Units, precision and the documented identifiability limitations of
2C-IRF, finite-transit Sourbron and empirical timing retain their contracts.

The empirical fit additionally reports ``half_rise_time_s`` as
``bolus_arrival_s + rise_time_s``. This is the absolute acquisition time in
seconds when the log-logistic rise term reaches 0.5 before applying baseline
and washout. It is derived from the existing fit and is not the half-height
time of the full concentration curve. The original timing parameters and
six-parameter optimizer are preserved. Inspect ``fit_success`` before using
the timing map; individual arrival/rise parameters can trade off strongly,
and uncertainty of the derived time is not estimated.

Extended-Tofts and reduced-Sourbron dataset workflows additionally report
``ve_equivalent = Ktrans / kep = Ft * Tt`` without clipping, in float64 with
dimensionless units ``1``. Existing ``ve_2c`` and ``ve_extended`` retain their
historical clipping to [0, 1]. The boolean ``ve_was_clipped`` marks equivalent
values outside that interval; it is false for NaN/unavailable values and does
not assess fit success or physical validity. This is a derived output, not an
additional optimizer parameter. Fits, predictions and scientific bounds are
preserved. These equivalent-volume fields do not apply to finite-transit
Sourbron (``estimate_tp=True``).

Extended Tofts and reduced Sourbron share one extended-compartment equation
and equivalent-alias definitions. The parameter correspondence is Ktrans = Ft
and kep = 1/Tt. Each model retains its native rate/time kernel arithmetic,
parameter bounds, initial guesses and optimizer settings. A deterministic
plasma function is evaluated once per extended-Tofts model call. The
finite-plasma-transit Sourbron calculation remains separate.

With ``estimate_tp=True``, Sourbron now linearly interpolates the sampled
arterial plasma AIF onto one uniform internal grid with step at most 0.1 s.
Both plasma and tubular convolutions use that grid and trapezoidal endpoint
correction, and only the final tissue curve is returned at acquisition times.
The initial states are zero at the first supplied sample; no earlier AIF is
extrapolated. The curve fitter prepares this input once for its optimizer.
Maps, reconstruction and QC use the same calculation. Result attributes
``sourbron_convolution_method``, ``sourbron_integration_max_step_s`` and
``sourbron_aif_interpolation`` record this numerical policy. The internal step
is a numerical resolution, not a physical lower Tp bound; accuracy depends on
the inputs and time scales.

This correction changes finite-Tp fitted values, predictions and
derived plasma flow/extraction outputs relative to the former acquisition-grid
double convolution. Scientific starts, bounds, fit windows and optimizer are
retained. The reduced Sourbron model and other DCE models keep their numerical
results. Independent synthetic tests at Tp=5 s recover vp, Ft, Tt and Tp within
0.5% on both test grids. Finer integration does not resolve uncertainty or noisy
parameter identifiability; optimizer success is not an accuracy guarantee.

2C-IRF recovery limits
~~~~~~~~~~~~~~~~~~~~~~

The 2C-IRF fitter uses a single bounded optimizer start. ``fit_success`` describes
optimizer termination; a high R² or small RSS measures curve agreement and does
not establish identifiable or accurate GFR/RPF. Flow, delay and dispersion
parameters can trade off, and sample-grid convolution can bias recovery.
The same limits apply to direct curves, labelled maps and pipeline outputs.

The independent dense-quadrature test uses an invented noise-free curve with
GFR/RPF 0.018/0.032 1/s, vascular delay/dispersion 4 s/0.16 1/s and tubular
delay/dispersion 8 s/0.08 1/s. On uniform 4 s and non-uniform acquisition grids
over 176 s, default fits report success and R² above 0.9999 but GFR about
0.0121/0.0110 1/s. The reference integrates the linearly interpolated sampled
AIF at 0.01 s spacing; a 0.0025 s reference retains the discrepancy. A start
at the known synthetic parameters estimates GFR more accurately but has a
larger residual. Thus selecting the smallest residual across additional starts
does not guarantee correct GFR. These cases are not general clinical error limits.

The existing optimizer, starts, bounds, sampling rule and outputs are retained.
No automatic multi-start selection, identifiability threshold or uncertainty
estimate is added. See ``fit_two_c_irf_curve`` below for the input and status
contracts.

.. automodule:: hypermri.analysis.dce.models.tofts
   :members: ToftsFitOptions, tofts_model_1comp, extended_tofts_model_2comp, fit_single_voxel_tofts_helper, build_tofts_fit_mask, tofts_model_reconstruction

.. automodule:: hypermri.analysis.dce.models.patlak
   :members: PatlakFitOptions, fit_single_voxel_patlak_helper, patlak_model

.. automodule:: hypermri.analysis.dce.models.sourbron
   :members: SourbronFitOptions, sourbron_separable_compartment_model, fit_sourbron_separable_compartment_curve, sourbron_model_reconstruction

.. automodule:: hypermri.analysis.dce.models.two_c_irf
   :members: TwoCIrfFitOptions, two_c_irf_retention_function, two_c_irf_model, fit_two_c_irf_curve, two_c_irf_reconstruction

.. automodule:: hypermri.analysis.dce.models.empirical_log_logistic
   :members: EmpiricalLogLogisticFitOptions, empirical_log_logistic_model, fit_empirical_log_logistic_curve, empirical_log_logistic_reconstruction

Renal workflows
---------------

.. automodule:: hypermri.analysis.dce.renal
   :members:
   :undoc-members:
   :show-inheritance:

AIF helpers
-----------

.. automodule:: hypermri.analysis.dce.aif
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing
-------------

.. automodule:: hypermri.analysis.dce.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Denoising
---------

Local TPCA, GPCA and MPPCA denoising and their DCE wrappers are unavailable
pending redistribution permission for their source code. The removed
``hypermri.analysis.dce.denoising`` module has no automatic replacement.
See :doc:`denoising` for the removal scope and the separate global PCA API.

Motion correction
-----------------

.. automodule:: hypermri.analysis.dce.motion_correction
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline internals
------------------

.. automodule:: hypermri.analysis.dce.pipeline
   :members:
   :undoc-members:
   :show-inheritance:
