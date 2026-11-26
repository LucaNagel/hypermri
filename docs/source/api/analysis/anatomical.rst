Anatomical Analysis
===================

The unused ``interpolate_array`` function and its package export have been
removed with the owner's explicit authorization. ``define_xyz_grid`` and
``CSI.get_plotting_grid`` remain available with their documented legacy
limitations; this removal does not change their geometry or numerical output.

Physical coverage of selected mask planes
-----------------------------------------

When selecting one plane with ``isel(slice_idx=0, drop=True)``, preserve the
array's ``slice_thickness_mm`` attribute. Mask projection uses half this
thickness as its default tolerance normal to the source plane. A 2 mm slice
therefore covers points up to 1 mm on either side, including the boundary.
Previously, removing the dimension reduced this tolerance to 0.001 mm even
when the thickness was known, which could exclude physically covered voxels.

An explicit finite, nonnegative ``plane_tolerance_mm`` overrides the default.
If thickness is unavailable after removing ``slice_idx``, supply that tolerance;
projection now reports an actionable error instead of assuming 0.001 mm.
When providing ``source_data`` separately from a mask, its geometry and thickness
define the source plane. Keeping a singleton ``slice_idx`` retains the existing
inactive-axis resolution contract. Input arrays and geometry are not modified.

.. automodule:: hypermri.analysis.anatomical.complex_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: hypermri.analysis.anatomical.trajectory
   :members:

Import ``plot_voxel_grids`` and ``add_scalebar`` from
``hypermri.visualization.anatomical``. Since 15 November 2026 the
``hypermri.analysis.anatomical`` package no longer re-exports them, nor
``ScaleBar``, ``ORIENTATION_TO_DIM``, ``ensure_standard_spectroscopy_dataarray``
or incidental names such as ``np``, ``plt``, ``xr``, ``logger``,
``Optional`` and ``TYPE_CHECKING``.

.. automodule:: hypermri.visualization.anatomical
   :members: plot_voxel_grids, add_scalebar

.. automodule:: hypermri.analysis.anatomical.masks
   :members:
   :undoc-members:
   :show-inheritance:

Single-voxel B1 calibration
---------------------------

``fit_single_voxel`` retains the historical iterative period search for complex
single-voxel calibration signals and delegates its calculation to
``hypermri.analysis.quantitative.fit_b1_calibration``. Both return the complex
amplitude, period in sqrt(W), fitted samples and two arrays of shape
``(niter, npoints)`` containing residual sums of squares and candidate periods.
The historical ``local_r2`` result contains RSS, rather than R-squared.
This restores the five-element calibration result; the incompatible adapter to
the quantitative map solver previously raised ``TypeError`` before fitting.

The initial center and half-width default to 3.0 sqrt(W). Each iteration
recenters on the best period seen so far and divides the half-width by the zoom
factor. The minimum candidate period remains 0.25 sqrt(W). Calculation uses
complex128 signals and float64 powers and diagnostics, preserves input samples,
and keeps their supplied order. Invalid arrays and search options raise
``ValueError``; an all-zero signal retains the historical zero amplitude and
curve with NaN period and search histories.

The numerical function accepts an exclusive ``stop_index`` for fitting a
segment and returns zeros afterwards. The retained ``stop_index_extra`` setting
does not change this calculation, matching its historical behavior. No automatic
flip-angle cutoff or pulse scaling is introduced. ``plot=True`` on the retained
anatomical function displays the final curve and stored search histories after
the calculation. Quantitative B1 map fitting keeps its existing solver and
defaults.

The spectral ``b1_calibration`` workflow exposes ``local_stop``,
``initial_guess_cent``, ``initial_guess_extent`` and ``use_multiprocessing`` as
explicit keyword-only options. Its default ``local_stop=-1`` continues to omit
the final amplitude/power sample after spectral fitting, as chosen by the owner
on 11 October 2026. Negative ends retain Python slicing semantics; ``None`` or
an end at/beyond the series length selects all samples. At least three samples
must remain. Selection does not truncate the acquired FID, and full-signal plots
do not index beyond the power array.

The workflow's center defaults to 3.1 sqrt(W). An omitted or ``None`` half-width
continues to follow the supplied center, preserving the previously effective
window. Explicit ``None`` for both selects the numerical solver's 3.0 sqrt(W)
window. An explicit half-width now takes effect independently for every
receiver; the former caller incorrectly replaced it with the center. This
correction can change results for calls specifying a half-width, but leaves
implicit windows and the modern B1 map solver unchanged. Invalid windows and
selections raise ``ValueError`` before spectral fitting; unknown keywords raise
``TypeError`` instead of being silently ignored. Spectral options still belong
in ``fit_params``. The first four positional arguments and the legacy scalar
return for the last fitted receiver remain supported.

.. automodule:: hypermri.analysis.quantitative.b1_calibration
   :members: fit_b1_calibration

.. automodule:: hypermri.analysis.anatomical.b1_mapping
   :members:
   :undoc-members:
   :show-inheritance:

ROI display and volume
-----------------------

.. automodule:: hypermri.analysis.anatomical.roi
   :members: raw_masked_mean, mask_volume_ml
