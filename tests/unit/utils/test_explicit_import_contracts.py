"""Import contracts with explicitly documented additive domain APIs."""

import ast
import importlib
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from types import ModuleType

import pytest

_IMPORT_CONTRACTS = {
    "hypermri.analysis.anatomical": {
        "hypermri.analysis.anatomical.b1_mapping": (
            "DB_MSG DEBUG_FILE DEBUG_LOG b1_calibration "
            "debug_log debug_log_b1_mapping find_nth_highest_signal_voxel "
            "fit_b1_surface_coil fit_quantitative_maps fit_single_voxel "
            "interactive_b1_fit_viewer plot_refpow_map_overlay "
            "refpow_from_sin_fit refpow_parameter_adjustment "
        ),
        "hypermri.analysis.anatomical.masks": (
            "align_mask_to_data_grid apply_aligned_mask apply_projected_mask "
            "extract_values_with_aligned_mask project_mask_to_data_grid "
        ),
    },
    "hypermri.analysis.denoising": {
        "hypermri.analysis.denoising.algorithms": ("pca_denoise "),
        "hypermri.analysis.denoising.patch_pca": (
            "PCANoiseClassification compute_marchenko_pastur_upper_edge "
            "denoise_gpca denoise_mppca denoise_tpca estimate_gpca_rank "
            "estimate_mppca_rank estimate_tpca_rank "
        ),
    },
    "hypermri.analysis.spectroscopy": {
        "hypermri.analysis.fitting.spectroscopy": (
            "_fit_single_spectrum_pseudo_inv add_metab_to_fit_params "
            "def_fit_params fid_basefunc fid_model fit_best_of_starts fit_data_pseudo_inv "
            "initialise_fit_params_from_results mask_fit_dataset_where_input_nonfinite "
            "phase_fit_results "
            "reconstruct_fitted_signal solve_for_complex_amplitudes "
        ),
        "hypermri.analysis.spectroscopy.alignment": (
            "align_spectra_by_fit_frequency shift_frequency "
        ),
        "hypermri.analysis.spectroscopy.frequency": (
            "center_freq_axis freq_hz_to_ppm freq_ppm_to_hz freq_to_index "
            "get_bw_from_freq_axis get_freq_axis "
        ),
        "hypermri.analysis.spectroscopy.pipeline": (
            "CSIPeakDetectionOptions detect_csi_peaks "
            "run_csi_auto_fitting_pipeline run_csi_fitting_pipeline "
        ),
        "hypermri.analysis.spectroscopy.refitting": (
            "SpectroscopyRefitEdit refit_spectroscopy_voxel "
        ),
        "hypermri.analysis.spectroscopy.single_voxel": (
            "LOG_MODES define_extent_press init_default_logger logger "
            "np show_voxel_overlap "
        ),
    },
    "hypermri.utils": {
        "hypermri.utils.mri": ("combine_channels "),
        "hypermri.utils.plot": (
            "disable_usetex_if_unavailable install_matplotlib_usetex_fallback "
        ),
        "hypermri.utils.system.decorators": (
            "functools init_default_logger logger onlycallonce optional_arg_decorator "
        ),
    },
    "hypermri.utils.mathematics": {
        "hypermri.utils.mathematics.calculations": (
            "apply_gaussian_smoothing calc_sampling_time_axis calculate_fwhm "
            "compute_weighted_mean_and_std find_max_and_index gaussian "
            "get_histogram init_default_logger logger moving_average np part_func "
            "polynomial_fit xr "
        ),
    },
    "hypermri.utils.time": {
        "hypermri.utils.time.temporal": (
            "Any Optional Tuple calc_inversion_time_multi_slice "
            "calc_per_image_timeaxis_s calc_sampling_time_axis create_time_axis "
            "get_dwell_time_s init_default_logger interp1d "
            "interpolate_temporal_data logger np resample_time_series "
            "smooth_temporal_data "
        ),
    },
}


_NAMESPACE_CONTRACTS = [
    (
        "hypermri.utils",
        (
            "combine_channels disable_usetex_if_unavailable functools "
            "init_default_logger install_matplotlib_usetex_fallback logger "
            "onlycallonce optional_arg_decorator "
        ),
        None,
    ),
    (
        "hypermri.analysis.spectroscopy",
        (
            "ACQUISITION_NOISE_MODEL ACQUISITION_TAIL_NOISE_SOURCE "
            "AcquisitionTailNoise DETECTION_BASES FIT_TRACE_VERSION LOG_MODES METABOLITE_CS_PPM "
            "PEAK_QUALITY_VERSION QUALITY_CLASSES QUALITY_FLAGS QUALITY_STATUSES "
            "PULSE_ACQUIRE_DEAD_TIME_SEQUENCES "
            "QUALITY_STATUS_MESSAGES SPECTROSCOPY_DIM_ALIASES "
            "STANDARD_SPECTROSCOPY_DIMS "
            "VALIDATED_PHASE_BAND_SEQUENCES add_fit_freqs_ppm "
            "add_metab_to_fit_params align_spectra_by_fit_frequency apply_lb "
            "attach_acquisition_tail_noise "
            "calculate_pyruvate_to_lactate_auc "
            "calculate_spectroscopy_fit_diagnostics calibrate_csi_noise_selection "
            "center_freq_axis classify_spectroscopy_fit_quality "
            "compare_metabolite_relevance conv_func def_fit_params "
            "define_extent_press eliminate_unnecessary_lines "
            "ensure_standard_spectroscopy_dataarray "
            "estimate_acquisition_tail_noise estimate_csi_noise "
            "estimate_explicit_noise_reference estimate_fit_covariance "
            "estimate_fit_local_uncertainty estimate_fit_uncertainty "
            "estimate_noise evaluate_spectroscopy_fit_quality "
            "fid_basefunc fid_model find_npeaks find_npeaks_nd fit_data_pseudo_inv "
            "fit_spectrum fit_spectrum_trace freq_hz_to_ppm freq_ppm_to_hz freq_to_index generate_fid "
            "get_bw_from_freq_axis get_freq_axis get_metab_cs_ppm "
            "init_default_logger initialise_fit_params_from_results "
            "inspect_csi_noise_candidates logger lorentzian "
            "make_csi_noise_reference make_noise_frame_reference "
            "make_outside_object_noise_reference make_rf_off_noise_reference "
            "mask_fit_dataset_where_input_nonfinite "
            "missing_spectroscopy_noise_reference multi_dim_linebroadening "
            "multi_dim_zeropadding noise_sigma_for_fit norm "
            "normalize_spectroscopy_quality_options np "
            "order_fit_problem_voxels phase_band_tau_s_from_scan "
            "phase_fit_results "
            "propagate_acquisition_tail_noise reconstruct_fitted_signal "
            "resolve_spectroscopy_noise_reference run_csi_auto_fitting_pipeline "
            "run_csi_fitting_pipeline run_selective_spectroscopy_detection_tests "
            "select_full_detection_test_mask shift_frequency show_voxel_overlap "
            "solve_for_complex_amplitudes sqrt_lorentzian suggest_signal_end "
            "summarize_fit_bound_flags zeropad_data "
        ),
        None,
    ),
    (
        "hypermri.analysis.anatomical",
        (
            "DB_MSG DEBUG_FILE DEBUG_LOG align_mask_to_data_grid "
            "apply_aligned_mask apply_projected_mask b1_calibration "
            "bruker2complex calc_k_space_trajectory_from_measurement debug_log "
            "debug_log_b1_mapping extract_values_with_aligned_mask "
            "find_nth_highest_signal_voxel fit_b1_surface_coil "
            "fit_quantitative_maps fit_single_voxel interactive_b1_fit_viewer "
            "plot_refpow_map_overlay project_mask_to_data_grid "
            "refpow_from_sin_fit refpow_parameter_adjustment "
        ),
        None,
    ),
    (
        "hypermri.analysis.denoising",
        (
            "PCANoiseClassification compute_marchenko_pastur_upper_edge "
            "denoise_gpca denoise_mppca denoise_tpca estimate_gpca_rank "
            "estimate_mppca_rank estimate_tpca_rank pca_denoise "
        ),
        [
            "PCANoiseClassification",
            "compute_marchenko_pastur_upper_edge",
            "denoise_gpca",
            "denoise_mppca",
            "denoise_tpca",
            "estimate_gpca_rank",
            "estimate_mppca_rank",
            "estimate_tpca_rank",
            "pca_denoise",
        ],
    ),
    (
        "hypermri.utils.mathematics",
        (
            "apply_gaussian_smoothing calc_sampling_time_axis calculate_fwhm "
            "compute_weighted_mean_and_std find_max_and_index gaussian "
            "get_histogram init_default_logger logger moving_average np part_func "
            "polynomial_fit xr "
        ),
        [
            "gaussian",
            "get_histogram",
            "compute_weighted_mean_and_std",
            "find_max_and_index",
            "moving_average",
            "polynomial_fit",
            "apply_gaussian_smoothing",
        ],
    ),
    (
        "hypermri.utils.time",
        (
            "Any Optional Tuple calc_inversion_time_multi_slice "
            "calc_per_image_timeaxis_s calc_sampling_time_axis create_time_axis "
            "get_dwell_time_s init_default_logger interp1d "
            "interpolate_temporal_data logger np resample_time_series "
            "smooth_temporal_data "
        ),
        [
            "get_dwell_time_s",
            "calc_sampling_time_axis",
            "calc_inversion_time_multi_slice",
            "create_time_axis",
            "resample_time_series",
            "interpolate_temporal_data",
        ],
    ),
]


# Documented domain additions retain every captured name.
_APPROVED_NAMESPACE_ADDITIONS = {
    "hypermri.analysis.anatomical": {"extract_masked_data"},
    "hypermri.analysis.spectroscopy": {
        "CSIPeakDetectionOptions",
        "detect_csi_peaks",
        "SpectroscopyRefitEdit",
        "refit_spectroscopy_voxel",
        "fit_best_of_starts",
        "add_line_from_residual",
    },
}


def _import(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module(name)


@pytest.mark.parametrize(
    "target, source, names",
    [
        (target, source, names)
        for target, owners in _IMPORT_CONTRACTS.items()
        for source, names in owners.items()
    ],
)
def test_existing_reexports_keep_their_implementation_owner(target, source, names):
    target_module, source_module = _import(target), _import(source)
    with warnings.catch_warnings():
        # Compatibility modules warn for every name they still provide.
        warnings.simplefilter("ignore", DeprecationWarning)
        for name in names.split():
            assert getattr(target_module, name) is getattr(source_module, name)


@pytest.mark.parametrize("target, public_names, exported_names", _NAMESPACE_CONTRACTS)
def test_existing_namespace_and_star_import_contract_are_unchanged(
    target, public_names, exported_names
):
    module = _import(target)

    # Python attaches imported child modules to their package dynamically.
    # Their presence depends on the rest of the process, not these re-exports.
    def exported_names_in(values):
        return {
            name
            for name, value in values.items()
            if not name.startswith("_")
            and not (
                isinstance(value, ModuleType)
                and value.__name__.startswith(target + ".")
            )
        }

    expected_names = set(public_names.split()) | _APPROVED_NAMESPACE_ADDITIONS.get(
        target, set()
    )
    if target in {
        "hypermri.utils",
        "hypermri.analysis.spectroscopy",
        "hypermri.analysis.anatomical",
    }:
        # Historical star exports are made explicit so they can resolve lazily.
        exported_names = sorted(expected_names)
    # An explicit __all__ can retain a lazy re-export without eager imports.
    resolved = dict(vars(module))
    for name in exported_names or []:
        resolved[name] = getattr(module, name)
    assert exported_names_in(resolved) == expected_names
    assert getattr(module, "__all__", None) == exported_names
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        exec(f"from {target} import *", {}, namespace)
    if exported_names is not None:
        assert set(namespace) == set(exported_names)
    else:
        assert exported_names_in(namespace) == expected_names


def test_source_imports_are_explicit():
    source_root = Path(__file__).resolve().parents[3] / "src"
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        assert not any(
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "*" for alias in node.names)
            for node in ast.walk(tree)
        ), str(path.relative_to(source_root))


@pytest.mark.parametrize(
    "name",
    [
        "ORIENTATION_TO_DIM",
        "Optional",
        "ScaleBar",
        "TYPE_CHECKING",
        "add_scalebar",
        "ensure_standard_spectroscopy_dataarray",
        "init_default_logger",
        "logger",
        "np",
        "plot_voxel_grids",
        "plt",
        "xr",
    ],
)
def test_anatomical_package_no_longer_reexports_incidental_names(name):
    anatomical = importlib.import_module("hypermri.analysis.anatomical")
    assert name not in anatomical.__all__
    with pytest.raises(AttributeError):
        getattr(anatomical, name)


def test_anatomical_plotting_helpers_stay_in_visualization():
    visualization = importlib.import_module("hypermri.visualization.anatomical")
    assert callable(visualization.plot_voxel_grids)
    assert callable(visualization.add_scalebar)


def test_unused_anatomical_interpolation_is_removed():
    """The removal is explicit; the shared grid helper remains available."""
    anatomical = importlib.import_module("hypermri.analysis.anatomical")
    grids = importlib.import_module("hypermri.utils.coordinates.grids")
    assert not hasattr(anatomical, "interpolate_array")
    assert callable(grids.define_xyz_grid)


@pytest.mark.parametrize(
    "image_first", [True, False], ids=["image_first", "coordinates_first"]
)
def test_image_and_coordinate_import_orders_preserve_coronal_geometry(
    image_first, package_subprocess_env
):
    """Fresh imports retain the FOV and grid values on invented coronal metadata."""
    first, second = (
        "hypermri.utils.image.transforms",
        "hypermri.utils.coordinates.grids",
    )
    if not image_first:
        first, second = second, first
    code = textwrap.dedent(
        f"""
        import importlib
        from types import SimpleNamespace
        import numpy as np

        importlib.import_module({first!r})
        importlib.import_module({second!r})
        transforms = importlib.import_module("hypermri.utils.image.transforms")
        grids = importlib.import_module("hypermri.utils.coordinates.grids")
        coordinates = importlib.import_module("hypermri.utils.coordinates")
        assert callable(coordinates.define_imagematrix_parameters)
        assert not hasattr(transforms, "define_imagematrix_parameters")
        scan = SimpleNamespace(
            matrix=np.array([4, 3, 2]),
            fov_mm=np.array([8.0, 9.0, 6.0]),
            spatial_dimension="3D",
            patient_position="Head_Prone",
            slice_orientation="coronal",
            read_orientation="L_R",
        )
        np.testing.assert_allclose(
            transforms.define_imageFOV_parameters(scan),
            [9.0, 8.0, 6.0], rtol=0, atol=1e-12,
        )
        expected = ([-4.5, -1.5, 1.5], [-4.0, -2.0, 0.0, 2.0], [-3.0, 0.0])
        for line, repeated, reference in zip(
            grids.define_xyz_grid(scan), grids.define_xyz_grid(scan), expected, strict=True
        ):
            np.testing.assert_allclose(line, reference, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(line, repeated)
            assert line.dtype == np.float64
        np.testing.assert_array_equal(scan.fov_mm, [8.0, 9.0, 6.0])
        np.testing.assert_array_equal(scan.matrix, [4, 3, 2])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=package_subprocess_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
