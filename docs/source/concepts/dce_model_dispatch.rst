DCE Models and Dispatch
=======================

The DCE domain selects its five existing models through one explicit dictionary
in ``hypermri.analysis.dce.model_selection``. Named workflows call the scientific
operation directly. The generic fitting facade delegates to that domain path;
legacy registry entries remain compatibility views of the same model definitions.

Shipped models
--------------

Use the named functions in :mod:`hypermri.analysis.dce.fitting` for concentration
maps. Times are seconds and concentrations are mM; spatial labels and coordinates
are preserved. :doc:`../api/analysis/dce` documents the public contracts and
the available model options.

.. list-table::
   :header-rows: 1
   :widths: 22 38 10 30

   * - Model
     - Named dataset function
     - AIF
     - Main parameters
   * - Patlak
     - ``fit_patlak_model``
     - Required
     - ``K_trans`` (1/s), ``Vp`` (fraction)
   * - Standard Tofts
     - ``fit_standard_tofts_model``
     - Required
     - ``Ktrans_standard``, ``kep_standard`` (1/s)
   * - Extended Tofts
     - ``fit_extended_tofts_model``
     - Required
     - ``Ktrans_extended``, ``kep_extended`` (1/s), ``vp_plasma`` (fraction)
   * - Sourbron
     - ``fit_sourbron_model``
     - Required
     - ``vp`` (fraction), ``Ft`` (1/s), ``Tt`` (s); optional ``Tp`` (s)
   * - 2C-IRF
     - ``fit_two_c_irf_model``
     - Required
     - ``GFR``, ``RPF`` (1/s), vascular/tubular delay (s) and dispersion (1/s)
   * - Empirical log-logistic
     - ``fit_empirical_log_logistic_model``
     - None
     - Arrival/rise/half-rise times (s), shape, washout (1/s), amplitude/baseline (mM)

Standard and extended Tofts are variants of one model module; ``fit_tofts_model``
also supports fitting both. Reduced Sourbron is equivalent to extended Tofts
under the documented parameter conversion. Its optional finite plasma-transit
extension is a separate scientific calculation. Both stages use an internal
uniform grid with spacing at most 0.1 s and a linearly interpolated AIF, before
returning acquisition samples. Noise and limited AIF information can still
limit parameter recovery. Successful 2C-IRF optimization
and a high R² do not establish accurate or identifiable parameters; its independent
recovery limitations are documented in the tutorial. No uncertainty guarantee is
inferred from fit success. The empirical model describes curve shape and needs
no AIF; its derived half-rise time applies to the rise term.

Dispatch structure
------------------

For the standard DCE workflow, the call chain is:

1. :func:`hypermri.analysis.dce.fitting.run_dce_pipeline`
2. ``hypermri.analysis.dce.shared._normalize_legacy_dce_request``
3. ``hypermri.analysis.dce.pipeline._run_dce_pipeline_core``
4. one of the DCE wrappers in ``hypermri.analysis.dce.fitting_support``
5. ``hypermri.analysis.dce.execution._fit_dce_maps`` for the legacy-prepared
   Patlak, Tofts and Sourbron contracts
6. the shared voxel executor and the selected numerical model module

2C-IRF and empirical workflows also use the shared voxel loop, retaining their
existing direct metadata assembly. Named DCE fits do not route back through the
generic facade or fitting registry. Generic parameter preparation for those two
direct workflows remains unsupported; use their named domain functions.

High-level flow
---------------

The public pipeline performs these steps in order:

1. normalize the requested fit type and any legacy aliases
2. prepare metadata such as TR, flip angle, and time axis
3. align or load the T1 map
4. combine channels when needed
5. compute the baseline ``s0_map``
6. prepare the AIF when the selected model requires one
7. convert the dynamic DCE series from signal space to concentration space
8. dispatch to the selected model fitter
9. reconstruct model predictions and compute QC metrics
10. attach alias variables and metadata for downstream use

Canonical locations
-------------------

Use these files as the main map of the DCE stack:

- ``src/hypermri/analysis/dce/fitting.py``: public imports
- ``src/hypermri/analysis/dce/model_selection.py``: model dictionary, named workflow
  selection and compatibility variant aliases
- ``src/hypermri/analysis/dce/shared.py``: historical request normalization and
  dispatch provenance derived from the dictionary
- ``src/hypermri/analysis/dce/pipeline.py``: high-level orchestration
- ``src/hypermri/analysis/dce/execution.py``: preparation and numerical map execution
- ``src/hypermri/analysis/dce/fitting_support.py``: named wrappers and shared QC
  metric assembly
- ``src/hypermri/analysis/dce/models/``: model equations, scientific options,
  quality inputs, reconstruction translation, units and metadata
- ``src/hypermri/analysis/fitting/fit_maps.py``: retained compatibility entry points

Alias handling
--------------

These routes are currently implemented:

- ``standard_tofts`` and ``standard_ktrans_ve`` route to Tofts with ``model_mode="1comp"``
- ``extended_tofts``, ``extended_tofts_plasma``, and ``extended_plasma_tofts`` route to Tofts with ``model_mode="2comp"``
- ``kidney_two_compartment_outflow`` and related aliases are deprecated and are treated as aliases of extended Tofts
- ``sourbron`` routes to the Sourbron separable renal compartment fitter
- ``empirical_log_logistic`` and ``log_logistic_empirical`` route to the direct empirical fitter
- ``2c_irf`` routes to the direct 2C-IRF fitter

Why this split exists
---------------------

Historically, DCE fitting logic accumulated in ``fit_maps.py`` together with
other quantitative fitting routines. The DCE package retains the
compatibility entry points while owning the scientific workflow in a
dedicated package with:

- explicit AIF preparation
- signal-to-concentration conversion
- clearer public fit names
- centralized alias handling
- model-specific QC outputs

Adding a concentration model requires its module and one dictionary entry.
The module supplies the scientific voxel callback, ordered outputs and dtypes,
reconstruction translation, quality inputs and descriptions. Its named dataset
function supplies the public workflow. Current adapters are ordinary functions;
there is no dynamic registration service or universal fitting framework.

Structural consolidation preserves scientific starts, bounds, convolution,
clipping, labelled coordinates and precision, including the documented 2C-IRF
recovery limits. The pipeline's ``fit_backend_function`` and
``fit_dispatch_summary`` now describe the actual domain execution route for
Patlak, Tofts and Sourbron. Other provenance and model outputs are retained.

Recommended navigation pattern
------------------------------

When investigating a DCE fit, trace it in this order:

1. public function in ``hypermri.analysis.dce.fitting``
2. alias handling in ``hypermri.analysis.dce.shared``
3. dispatch in ``hypermri.analysis.dce.pipeline``
4. wrapper in ``hypermri.analysis.dce.fitting_support``
5. domain execution and the equation in ``hypermri.analysis.dce.models``
