"""Expose spectroscopy operations without importing interactive dependencies.

Domain modules own numerical fitting, signal processing and diagnostics. Public
and historical re-exports resolve on first access and retain object identity.
No fitting, scan reading or widget construction occurs at package import.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    "..fitting.spectroscopy": (
        "_fit_single_spectrum_pseudo_inv add_metab_to_fit_params def_fit_params "
        "fid_basefunc fid_model fit_best_of_starts fit_data_pseudo_inv "
        "initialise_fit_params_from_results mask_fit_dataset_where_input_nonfinite "
        "phase_fit_results "
        "reconstruct_fitted_signal solve_for_complex_amplitudes "
    ),
    ".alignment": ("align_spectra_by_fit_frequency shift_frequency "),
    ".signals": "lorentzian sqrt_lorentzian generate_fid",
    ".processing": "norm conv_func multi_dim_linebroadening multi_dim_zeropadding apply_lb zeropad_data",
    ".metabolites": "METABOLITE_CS_PPM get_metab_cs_ppm",
    ".peaks": "find_npeaks find_npeaks_nd",
    ".kinetics": "calculate_pyruvate_to_lactate_auc",
    ".frequency": (
        "freq_to_index get_bw_from_freq_axis get_freq_axis freq_hz_to_ppm freq_ppm_to_hz center_freq_axis "
        "add_fit_freqs_ppm "
    ),
    ".covariance": ("estimate_fit_covariance "),
    ".fit_diagnostics": (
        "noise_sigma_for_fit order_fit_problem_voxels summarize_fit_bound_flags "
    ),
    ".fitting": ("fit_spectrum "),
    ".fit_trace": ("FIT_TRACE_VERSION fit_spectrum_trace "),
    ".local_uncertainty": ("estimate_fit_local_uncertainty "),
    ".noise": ("inspect_csi_noise_candidates "),
    ".noise_calibration": ("calibrate_csi_noise_selection make_csi_noise_reference "),
    ".noise_estimation": (
        "ACQUISITION_NOISE_MODEL ACQUISITION_TAIL_NOISE_SOURCE AcquisitionTailNoise "
        "attach_acquisition_tail_noise estimate_acquisition_tail_noise estimate_csi_noise estimate_noise "
        "propagate_acquisition_tail_noise suggest_signal_end "
    ),
    ".pipeline": (
        "CSIPeakDetectionOptions detect_csi_peaks run_csi_auto_fitting_pipeline "
        "run_csi_fitting_pipeline "
    ),
    ".quality": (
        "DETECTION_BASES PEAK_QUALITY_VERSION QUALITY_CLASSES QUALITY_FLAGS "
        "QUALITY_STATUS_MESSAGES QUALITY_STATUSES "
        "calculate_spectroscopy_fit_diagnostics "
        "classify_spectroscopy_fit_quality estimate_explicit_noise_reference "
        "evaluate_spectroscopy_fit_quality make_noise_frame_reference "
        "make_outside_object_noise_reference make_rf_off_noise_reference "
        "missing_spectroscopy_noise_reference "
        "normalize_spectroscopy_quality_options "
        "resolve_spectroscopy_noise_reference "
        "run_selective_spectroscopy_detection_tests "
        "select_full_detection_test_mask "
    ),
    ".phase_timing": (
        "PULSE_ACQUIRE_DEAD_TIME_SEQUENCES VALIDATED_PHASE_BAND_SEQUENCES "
        "phase_band_tau_s_from_scan "
    ),
    ".refitting": ("SpectroscopyRefitEdit refit_spectroscopy_voxel "),
    ".relevance": (
        "add_line_from_residual compare_metabolite_relevance "
        "eliminate_unnecessary_lines "
    ),
    ".schema": (
        "SPECTROSCOPY_DIM_ALIASES STANDARD_SPECTROSCOPY_DIMS "
        "ensure_standard_spectroscopy_dataarray "
    ),
    ".single_voxel": (
        "LOG_MODES define_extent_press init_default_logger logger np "
        "show_voxel_overlap "
    ),
    ".uncertainty": ("estimate_fit_uncertainty "),
}
_IMPORTS = {
    name: module for module, names in _EXPORTS.items() for name in names.split()
}
__all__ = sorted(name for name in _IMPORTS if not name.startswith("_"))


def __getattr__(name: str):
    """Resolve a canonical or historical spectroscopy export lazily.

    Parameters
    ----------
    name : str
        Name previously re-exported by the spectroscopy package.

    Returns
    -------
    object
        Original implementation, cached after successful import.

    Raises
    ------
    AttributeError
        If the requested name is not a spectroscopy export.
    ImportError
        If dependencies of the requested operation are unavailable.
    """
    target = _IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(target, __name__), name)
    globals()[name] = value
    return value
