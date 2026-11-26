"""Regression evidence for DCE quality windows using invented mM curves and seconds.

The explicit frame lists below characterize the existing model fitters. With
zero model predictions, RSS/RMSE/R² and information criteria have independent
analytic references. Relative tolerance 1e-12 covers float64 rounding without
concealing the effect of evaluating a different set of samples.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.dce.fitting_support import compute_fit_quality_metrics


@pytest.mark.parametrize("temporal_first", [False, True])
@pytest.mark.parametrize(
    ("model", "options", "frames", "parameter_count"),
    [
        ("tofts", {"fit_end_index": -2}, [0, 1, 2, 3], 2),
        ("tofts", {"fit_stop_index": -2}, [0, 1, 2, 3], 2),
        ("tofts", {"fit_end_index": -9}, [], 2),
        ("patlak", {"fit_indices": [0]}, list(range(6)), 2),
        ("patlak", {"fit_times_s": [20]}, list(range(6)), 2),
        ("patlak", {"fit_start_index": 1.5}, [2, 3, 4, 5], 2),
        ("patlak", {"fit_stop_index": -2}, [], 2),
        ("sourbron", {"fit_times_s": [20]}, list(range(6)), 3),
        (
            "sourbron",
            {"fit_times_s": [20], "fit_indices": [0, 1, 3, 5]},
            [0, 1, 3, 5],
            3,
        ),
    ],
)
def test_quality_metrics_use_existing_model_frames(
    model, options, frames, parameter_count, temporal_first
):
    times = np.array([0.0, 2.0, 5.0, 9.0, 14.0, 20.0])
    observed = np.arange(1.0, 7.0)
    concentration = xr.DataArray(
        observed[None, :],
        dims=("read_idx", "repetition_idx"),
        coords={
            "read_idx": [4.0],
            "repetition_idx": [8, 9, 10, 11, 12, 13],
            "time_s": ("repetition_idx", times),
        },
        attrs={"units": "mM"},
    )
    if temporal_first:
        concentration = concentration.transpose("repetition_idx", "read_idx")
    parameters_by_model = {
        "tofts": {"Ktrans_1c": 0.0, "kep_1c": 0.1},
        "patlak": {"K_trans": 0.0, "Vp": 0.0},
        "sourbron": {"vp": 0.0, "Ft": 0.0, "Tt": 60.0},
    }
    parameters = xr.Dataset(
        {
            name: ("read_idx", [value])
            for name, value in parameters_by_model[model].items()
        },
        coords={"read_idx": [4.0]},
        attrs={"model_mode": "1comp"},
    )
    original_concentration = concentration.copy(deep=True)
    original_parameters = parameters.copy(deep=True)
    aif = np.ones(6)
    original_options = dict(options)
    metrics = compute_fit_quality_metrics(
        parameters,
        concentration,
        fit_type=model,
        aif_concentration=aif,
        aif_time_s=times,
        fit_params=options,
    )

    expected_mask = np.isin(np.arange(6), frames)
    np.testing.assert_array_equal(metrics.fit_window_mask, expected_mask)
    assert metrics.fit_window_mask.dims == ("repetition_idx",)
    assert metrics.fit_window_mask.dtype == bool
    xr.testing.assert_identical(
        metrics.fit_window_mask.repetition_idx, concentration.repetition_idx
    )
    xr.testing.assert_identical(metrics.fit_window_mask.time_s, concentration.time_s)
    rss = np.sum(observed[frames] ** 2)
    assert metrics.rss.item() == pytest.approx(rss, rel=1e-12)
    if frames:
        count = len(frames)
        assert metrics.rmse.item() == pytest.approx(np.sqrt(rss / count), rel=1e-12)
        tss = np.sum((observed[frames] - np.mean(observed[frames])) ** 2)
        assert metrics.r_squared.item() == pytest.approx(1 - rss / tss, rel=1e-12)
        assert metrics.aic.item() == pytest.approx(
            count * np.log(rss / count) + 2 * parameter_count, rel=1e-12
        )
        assert metrics.bic.item() == pytest.approx(
            count * np.log(rss / count) + parameter_count * np.log(count), rel=1e-12
        )
        assert metrics.converged.item()
        assert not metrics.insufficient_signal.item()
    else:
        for name in ("rmse", "r_squared", "aic", "bic"):
            assert np.isnan(metrics[name].item())
        assert metrics.insufficient_signal.item()
        assert not metrics.converged.item()
    assert metrics.failed.item() == (not metrics.converged.item())
    xr.testing.assert_identical(concentration, original_concentration)
    xr.testing.assert_identical(parameters, original_parameters)
    np.testing.assert_array_equal(aif, np.ones(6))
    np.testing.assert_array_equal(times, concentration.time_s)
    assert options == original_options
