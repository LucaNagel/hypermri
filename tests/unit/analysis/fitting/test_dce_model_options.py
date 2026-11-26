"""F2 curve-option and import contracts using invented mM curves and seconds.

Options are compared with explicit legacy inputs; float64 fit results use
rtol 1e-12/atol 1e-14 to detect numerical changes beyond rounding. The broader
The recovery tests retain independent dense-grid references and scientific
recovery tolerances and characterize the remaining scientific limitations.
"""

import importlib
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from hypermri.analysis.dce import fit_window, fitting_support
from hypermri.analysis.dce.models import (
    empirical_log_logistic,
    patlak,
    sourbron,
    tofts,
    two_c_irf,
)

OPTIONS = [
    patlak.PatlakFitOptions,
    sourbron.SourbronFitOptions,
    tofts.ToftsFitOptions,
    two_c_irf.TwoCIrfFitOptions,
    empirical_log_logistic.EmpiricalLogLogisticFitOptions,
]
TIMES_S = np.cumsum(np.r_[0.0, np.resize([2.0, 3.0, 4.0], 39)])
AIF_MM = (TIMES_S / 12) ** 3 * np.exp(-TIMES_S / 4)
NOISE_MM = np.random.default_rng(20261006).normal(0, 0.0001, len(TIMES_S))


def _assert_same_result(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        assert np.asarray(actual[name]).shape == np.asarray(expected[name]).shape
        assert np.asarray(actual[name]).dtype == np.asarray(expected[name]).dtype
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=1e-12, atol=1e-14, equal_nan=True
        )


@pytest.mark.parametrize("option_type", OPTIONS)
@pytest.mark.parametrize("explicit", [False, True])
def test_options_are_frozen_overrides_without_duplicate_defaults(option_type, explicit):
    options = option_type(fit_start_time_s=5.0) if explicit else option_type()
    assert options.to_fit_params() == ({"fit_start_time_s": 5.0} if explicit else {})
    with pytest.raises(FrozenInstanceError):
        options.fit_start_time_s = 9.0
    translated = options.to_fit_params()
    translated["fit_start_time_s"] = 12.0
    assert options.fit_start_time_s == (5.0 if explicit else None)
    with pytest.raises(TypeError, match="unknown_setting"):
        option_type(unknown_setting=1)


@pytest.mark.parametrize("model", ["patlak", "sourbron", "tofts"])
def test_removed_quantitative_dce_import_paths_are_absent(model):
    """The DCE models are imported only from ``hypermri.analysis.dce.models``."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(
            f"hypermri.analysis.fitting.models.quantitative.{model}"
        )
    canonical = importlib.import_module(f"hypermri.analysis.dce.models.{model}")
    assert canonical.MODEL_ID == model


def test_fitting_support_reexports_existing_curve_and_reconstruction_names():
    for module, names in [
        (
            two_c_irf,
            [
                "two_c_irf_model",
                "two_c_irf_retention_function",
                "fit_two_c_irf_curve",
                "two_c_irf_reconstruction",
            ],
        ),
        (
            empirical_log_logistic,
            [
                "empirical_log_logistic_model",
                "fit_empirical_log_logistic_curve",
                "empirical_log_logistic_reconstruction",
            ],
        ),
    ]:
        for name in names:
            assert getattr(fitting_support, name) is getattr(module, name)
    assert fitting_support.select_fit_window is fit_window.select_fit_window


@pytest.mark.parametrize("estimate_tp", [False, True])
def test_sourbron_options_follow_existing_bounds_window_and_transit_variant(
    estimate_tp,
):
    options = sourbron.SourbronFitOptions(
        fit_start_index=2,
        fit_end_time_s=TIMES_S[-2],
        aif_threshold_mm=1e-7,
        estimate_tp=estimate_tp,
        max_nfev=100,
        f_scale_mm=0.03,
    )
    curve = (
        sourbron.sourbron_separable_compartment_model(
            TIMES_S,
            AIF_MM,
            plasma_volume_fraction=0.07,
            tubular_flow_s=0.02,
            tubular_transit_time_s=12,
            plasma_transit_time_s=4 if estimate_tp else None,
        )
        + NOISE_MM
    )
    original = curve.copy()
    actual = sourbron.fit_sourbron_separable_compartment_curve(
        curve,
        aif_concentration=AIF_MM,
        time_s=TIMES_S,
        fit_params=options,
    )
    expected = sourbron.fit_sourbron_separable_compartment_curve(
        curve,
        aif_concentration=AIF_MM,
        time_s=TIMES_S,
        fit_params={
            "fit_start_index": 2,
            "fit_end_time_s": TIMES_S[-2],
            "aif_threshold": 1e-7,
            "estimate_tp": estimate_tp,
            "max_nfev": 100,
            "f_scale": 0.03,
        },
    )
    _assert_same_result(actual, expected)
    np.testing.assert_array_equal(curve, original)


def test_patlak_options_preserve_graphical_regression_and_threshold():
    curve = (
        0.02 * cumulative_trapezoid(AIF_MM, TIMES_S, initial=0)
        + 0.1 * AIF_MM
        + NOISE_MM
    )
    options = patlak.PatlakFitOptions(
        fit_start_index=1.5, fit_end_time_s=90, aif_threshold_mm=0.001
    )
    actual = patlak.fit_single_voxel_patlak_helper(curve, AIF_MM, TIMES_S, options)
    expected = patlak.fit_single_voxel_patlak_helper(
        curve,
        AIF_MM,
        TIMES_S,
        {"fit_start_index": 1.5, "fit_end_time_s": 90, "aif_threshold": 0.001},
    )
    _assert_same_result(actual, expected)


@pytest.mark.parametrize("model_type", ["1comp", "2comp"])
def test_tofts_options_match_explicit_starts_bounds_and_window(model_type):
    cp = interp1d(TIMES_S, AIF_MM, fill_value="extrapolate")
    options = tofts.ToftsFitOptions(
        fit_end_index=-2,
        initial_1comp_per_s=(0.03, 0.1),
        lower_1comp_per_s=(0.0, 0.0),
        upper_1comp_per_s=(1.0, 2.0),
        initial_2comp=(0.03, 0.1, 0.05),
        lower_2comp=(0.0, 0.0, 0.0),
        upper_2comp=(1.0, 2.0, 0.8),
    )
    curve = (
        tofts.tofts_model_1comp(TIMES_S, 0.02, 0.08, cp)
        if model_type == "1comp"
        else tofts.extended_tofts_model_2comp(TIMES_S, 0.02, 0.08, 0.07, cp)
    ) + NOISE_MM
    input_mask = np.arange(len(TIMES_S)) >= 2
    original_mask = input_mask.copy()
    actual = tofts.fit_single_voxel_tofts_helper(
        TIMES_S,
        curve,
        cp,
        model_type,
        fit_mask=input_mask,
        options=options,
    )
    expected = tofts.fit_single_voxel_tofts_helper(
        TIMES_S,
        curve,
        cp,
        model_type,
        x0_1comp=(0.03, 0.1),
        bounds_1comp=((0.0, 0.0), (1.0, 2.0)),
        x0_2comp=(0.03, 0.1, 0.05),
        bounds_2comp=((0.0, 0.0, 0.0), (1.0, 2.0, 0.8)),
        fit_mask=input_mask & (np.arange(len(TIMES_S)) < len(TIMES_S) - 2),
    )
    for obtained, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(obtained, reference, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(input_mask, original_mask)
    with pytest.raises(ValueError, match="not both"):
        tofts.fit_single_voxel_tofts_helper(
            TIMES_S, curve, cp, x0_1comp=(0.1, 0.1), options=options
        )


@pytest.mark.parametrize(
    ("option_type", "curve_fit", "curve"),
    [
        (
            two_c_irf.TwoCIrfFitOptions,
            two_c_irf.fit_two_c_irf_curve,
            two_c_irf.two_c_irf_model(
                TIMES_S,
                AIF_MM,
                gfr_s=0.015,
                rpf_s=0.05,
                vascular_delay_s=2,
                vascular_dispersion_rate_s=0.2,
                tubular_delay_s=4,
                tubular_dispersion_rate_s=0.05,
            ),
        ),
        (
            empirical_log_logistic.EmpiricalLogLogisticFitOptions,
            empirical_log_logistic.fit_empirical_log_logistic_curve,
            empirical_log_logistic.empirical_log_logistic_model(
                TIMES_S, 4, 12, 3, 0.02, 0.2, 0.01
            ),
        ),
    ],
)
def test_six_parameter_curve_options_preserve_defaults_and_explicit_selectors(
    option_type, curve_fit, curve
):
    frames = np.arange(2, len(TIMES_S) - 2)
    options = option_type(fit_indices=frames, max_nfev=100, f_scale_mm=0.03)
    frames[0] = 0  # Options must retain their own immutable copy.
    kwargs = {"time_points_s": TIMES_S}
    if curve_fit is two_c_irf.fit_two_c_irf_curve:
        kwargs["aif_concentration"] = AIF_MM
    actual = curve_fit(curve + NOISE_MM, fit_params=options, **kwargs)
    expected = curve_fit(
        curve + NOISE_MM,
        fit_params={
            "fit_indices": np.arange(2, len(TIMES_S) - 2),
            "max_nfev": 100,
            "f_scale": 0.03,
        },
        **kwargs,
    )
    _assert_same_result(actual, expected)
    assert options.fit_indices[0] == 2


@pytest.mark.parametrize(
    "options",
    [
        two_c_irf.TwoCIrfFitOptions(gfr_guess_per_s=0.02, flow_upper_bound_per_s=2.0),
        sourbron.SourbronFitOptions(initial_parameters=[0.1, 0.02, 30.0]),
        empirical_log_logistic.EmpiricalLogLogisticFitOptions(
            lower_bounds=[0, 0.1, 1, 0, 0, -1]
        ),
    ],
)
def test_scientific_overrides_keep_units_and_parameter_order(options):
    mapping = options.to_fit_params()
    if isinstance(options, two_c_irf.TwoCIrfFitOptions):
        assert mapping == {"gfr_guess": 0.02, "flow_upper_bound": 2.0}
    else:
        assert isinstance(next(iter(mapping.values())), tuple)


@pytest.mark.parametrize(
    ("option_type", "kwargs"),
    [
        (sourbron.SourbronFitOptions, {"initial_parameters": (0.1, 0.02)}),
        (sourbron.SourbronFitOptions, {"estimate_tp": True, "lower_bounds": (0, 0, 1)}),
        (two_c_irf.TwoCIrfFitOptions, {"initial_parameters": (1, 2)}),
        (
            empirical_log_logistic.EmpiricalLogLogisticFitOptions,
            {"upper_bounds": (1, 2)},
        ),
        (tofts.ToftsFitOptions, {"lower_1comp_per_s": (0, 0)}),
        (tofts.ToftsFitOptions, {"initial_2comp": (0.1, 0.1)}),
    ],
)
def test_option_vector_contracts_fail_before_fitting(option_type, kwargs):
    with pytest.raises(ValueError, match="exactly|paired lower and upper"):
        option_type(**kwargs)


@pytest.mark.parametrize(
    "option_type, selector",
    [
        (patlak.PatlakFitOptions, "fit_indices"),
        (patlak.PatlakFitOptions, "fit_times_s"),
        (sourbron.SourbronFitOptions, "fit_times_s"),
    ],
)
def test_options_do_not_advertise_selectors_ignored_by_the_model(option_type, selector):
    with pytest.raises(TypeError, match=selector):
        option_type(**{selector: (1,)})
