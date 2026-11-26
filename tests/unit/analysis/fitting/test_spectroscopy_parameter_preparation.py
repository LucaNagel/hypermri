"""Small checks for parameter bounds and safe reuse of previous spectroscopy fits."""

import warnings
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import spectroscopy
from hypermri.analysis.fitting.models.spectroscopy import fid_model as fid_pseudo_inv
from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    fit_data_pseudo_inv,
    initialise_fit_params_from_results,
)


@pytest.fixture
def fid_data() -> xr.DataArray:
    """Provide two small labelled spectra without requiring ppm calibration."""
    return xr.DataArray(
        np.ones((2, 8), dtype=complex),
        dims=("voxel", "acquisition_idx"),
        coords={"voxel": [10, 20], "acquisition_idx": np.arange(8) * 0.001},
        attrs={"signal_domain": "temporal"},
    )


def _prepare(data: xr.DataArray, **options) -> xr.Dataset:
    """Prepare two separated metabolites with optional test-specific settings."""
    return def_fit_params(
        data, {"metabs": ["a", "b"], "metabs_freqs_hz": [0.0, 200.0], **options}
    )


def test_scalar_settings_alias_precedence_and_solver_options(fid_data):
    """Broadcast scalars, keep T2* alias precedence and preserve the options dict."""
    options = {
        "metabs": ["a", "b"],
        "metabs_freqs_hz": [0.0, 200.0],
        "metabs_t2s": [0.02, 0.03],
        "initial_t2s_s": 0.025,
        "metabs_freq_ranges": [0.0, 40.0],
        "min_t2_s": 0.01,
        "max_t2_s": 0.04,
        "skip_fitting": [False, True],
        "pseudo_inv": False,
    }
    original = deepcopy(options)
    params = def_fit_params(fid_data, options)
    np.testing.assert_array_equal(params.initial_t2s_s, np.full((2, 2), 0.025))
    np.testing.assert_array_equal(params.freq_range_lower_bound_hz[:, 0], [0, 180])
    np.testing.assert_array_equal(params.freq_range_upper_bound_hz[:, 0], [0, 220])
    assert params.skip_fitting.dtype == bool
    assert params.attrs["amp_solver"] == "normal_eq"
    assert options == original

    legacy = _prepare(fid_data, metabs_t2s=[0.02, 0.03], amp_solver="tikhonov")
    np.testing.assert_array_equal(legacy.initial_t2s_s[:, 0], [0.02, 0.03])
    assert legacy.attrs["amp_solver"] == "ridge"


@pytest.mark.parametrize(
    ("options", "error_name"),
    [
        ({"metabs_freq_ranges": -1}, "metabs_freq_ranges"),
        ({"metabs_t2_ranges": [[0.01], [0.02]]}, "metabs_t2_ranges"),
        ({"initial_t2s_s": np.nan}, "initial_t2s_s"),
        ({"min_t2_s": 0}, "min_t2_s"),
        ({"min_t2_s": 0.05, "max_t2_s": 0.01}, "max_t2_s"),
        ({"initial_t2s_s": 0.2}, "initial_t2s_s"),
        ({"npoints": 2.5}, "npoints"),
        ({"niter": 0}, "niter"),
        ({"zoomfactor": 0}, "zoomfactor"),
        ({"amp_ridge_lambda": np.nan}, "amp_ridge_lambda"),
        ({"min_peak_separation_hz": -1}, "min_peak_separation_hz"),
        ({"skip_fitting": [0, 2]}, "skip_fitting"),
    ],
)
def test_invalid_settings_fail_during_preparation(fid_data, options, error_name):
    """Reject distinct invalid settings before any optimizer work is started."""
    with pytest.raises(ValueError, match=error_name):
        _prepare(fid_data, **options)


def test_too_close_start_frequencies_fail_before_fitting(fid_data, monkeypatch):
    """Reject a start pair closer than min_peak_separation_hz in one voxel.

    The fit scores such a pair with infinite error, so the first frequency
    step would have no valid comparison value. The error names the pair, the
    positional index of its voxel (coordinate value 20 is position 1) and the
    number of affected spectra before any spectrum is fitted.
    """

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Fit preparation must reject the start values first.")

    params = _prepare(fid_data, metabs_freqs_hz=[0.0, 5.0], metabs_freq_ranges=20.0)
    params = params.copy(deep=True)
    params["initial_freqs_hz"].loc[{"voxel": 20, "metabolite": "b"}] = 3.0
    params = fid_pseudo_inv._update_frequency_bounds(params)
    monkeypatch.setattr(spectroscopy, "_fit_single_spectrum_pseudo_inv", unexpected_fit)
    with pytest.raises(
        ValueError,
        match=r"fit_data_pseudo_inv: .*'a' \(0 Hz\) and 'b' \(3 Hz\) are 3 Hz "
        r"apart in the spectrum at positions \{'voxel': 1\} \(for isel\), "
        r"closer than min_peak_separation_hz=5 Hz "
        r"\(1 of 2 spectra affected\)",
    ):
        fit_data_pseudo_inv(fid_data, params)


def test_too_close_starts_name_positions_when_coordinates_span_dimensions():
    """Report positional indices when coordinates share dimension names.

    Regression for CSI fit parameters: their ``read_idx``, ``phase_idx`` and
    ``slice_idx`` coordinates hold 3-D voxel positions in mm over all three
    dimensions, so building the message from coordinate values raised
    "can only convert an array of size 1 to a Python scalar" instead of the
    intended error. Fixture: 3 x 2 x 1 voxels, starts 100 and 99 Hz in every
    voxel, separation 3 Hz.
    """
    shape = (3, 2, 1)
    position_mm = np.broadcast_to(np.arange(3.0)[:, None, None], shape)
    voxel_dims = ("read_idx", "phase_idx", "slice_idx")
    starts = xr.DataArray(
        np.stack([np.full(shape, 100.0), np.full(shape, 99.0)]),
        dims=("metabolite", *voxel_dims),
        coords={
            "metabolite": ["OMPD1_1", "OMPD1_2"],
            **{dim: (voxel_dims, position_mm) for dim in voxel_dims},
        },
    )
    params = xr.Dataset({"initial_freqs_hz": starts})

    with pytest.raises(
        ValueError,
        match=r"add_metab_to_fit_params: .*'OMPD1_2' \(99 Hz\) and 'OMPD1_1' "
        r"\(100 Hz\) are 1 Hz apart in the spectrum at positions "
        r"\{'read_idx': 0, 'phase_idx': 0, 'slice_idx': 0\} \(for isel\).*"
        r"\(6 of 6 spectra affected\)",
    ):
        spectroscopy._check_start_separation(
            params, 3.0, operation="add_metab_to_fit_params"
        )


@pytest.mark.parametrize("case", ["equal", "skipped", "lower", "disabled"])
def test_start_separation_check_admits_valid_starts(fid_data, case):
    """Allow a distance equal to the separation, skipped lines and lower limits."""
    params = _prepare(fid_data, metabs_freqs_hz=[0.0, 5.0], metabs_freq_ranges=20.0)
    params = params.copy(deep=True)
    if case != "equal":
        params["initial_freqs_hz"].loc[{"voxel": 20, "metabolite": "b"}] = 3.0
        params = fid_pseudo_inv._update_frequency_bounds(params)
    if case == "skipped":
        params["skip_fitting"].loc[{"voxel": 20, "metabolite": "b"}] = True
    elif case in ("lower", "disabled"):
        params.attrs["min_peak_separation_hz"] = 2.0 if case == "lower" else 0.0
    result = fit_data_pseudo_inv(fid_data, params)
    assert result.fit_freqs.shape == (2, 2)


def test_frequency_bounds_follow_start_and_range(fid_data):
    """Stored bounds that differ are replaced and named; matching ones pass."""
    params = _prepare(fid_data, metabs_freqs_hz=[0.0, 5.0], metabs_freq_ranges=20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        same = fid_pseudo_inv._derive_frequency_bounds(params, operation="op")
    xr.testing.assert_identical(same, params)

    stale = params.copy(deep=True)
    stale["initial_freqs_hz"].loc[{"voxel": 20, "metabolite": "b"}] = 7.0
    before = stale.copy(deep=True)
    with pytest.warns(UserWarning, match=r"^op: stored frequency bounds of \['b'\]"):
        derived = fid_pseudo_inv._derive_frequency_bounds(stale, operation="op")
    xr.testing.assert_identical(stale, before)
    bounds = derived[["freq_range_lower_bound_hz", "freq_range_upper_bound_hz"]]
    assert bounds.sel(voxel=20, metabolite="b").to_array().values.tolist() == [
        -3.0,
        17.0,
    ]
    xr.testing.assert_identical(derived.sel(metabolite="a"), stale.sel(metabolite="a"))

    no_range = stale.drop_vars("metabs_freq_ranges")
    assert fid_pseudo_inv._derive_frequency_bounds(no_range, operation="op") is no_range


def test_zero_search_widths_keep_parameters_fixed(fid_data):
    """A tiny real fit preserves frozen frequencies and T2* at equal bounds."""
    params = _prepare(
        fid_data,
        initial_t2s_s=0.025,
        min_t2_s=0.025,
        max_t2_s=0.025,
        metabs_freq_ranges=0.0,
        metabs_t2_ranges=0.0,
        rep_fitting=1,
        niter=1,
        npoints=3,
    )
    result = fit_data_pseudo_inv(fid_data, params, show_tqdm=False)
    xr.testing.assert_equal(
        result.fit_freqs, params.initial_freqs_hz.transpose("voxel", "metabolite")
    )
    xr.testing.assert_equal(
        result.fit_t2s, params.initial_t2s_s.transpose("voxel", "metabolite")
    )


def test_refinement_preserves_labels_and_recalculates_bounds(fid_data):
    """Reuse transposed results and labelled widths without mutating either input."""
    params = _prepare(fid_data)
    results = xr.Dataset(
        {
            "fit_freqs": (params.initial_freqs_hz + 3).transpose("voxel", "metabolite"),
            "fit_t2s": (params.initial_t2s_s * 0.9).transpose("voxel", "metabolite"),
        }
    )
    original_params, original_results = params.copy(deep=True), results.copy(deep=True)
    widths = xr.DataArray(
        [[0.0, 12.0], [20.0, 0.0]],
        dims=("voxel", "metabolite"),
        coords={"voxel": [10, 20], "metabolite": ["a", "b"]},
    )
    updated = initialise_fit_params_from_results(params, results, widths, [0, 0.01])
    xr.testing.assert_equal(
        updated.initial_freqs_hz,
        results.fit_freqs.transpose(*params.initial_freqs_hz.dims),
    )
    xr.testing.assert_equal(
        updated.freq_range_lower_bound_hz, updated.initial_freqs_hz - widths / 2
    )
    xr.testing.assert_equal(
        updated.freq_range_upper_bound_hz, updated.initial_freqs_hz + widths / 2
    )
    np.testing.assert_array_equal(updated.metabs_t2_ranges[:, 0], [0, 0.01])
    unchanged_widths = initialise_fit_params_from_results(params, results)
    xr.testing.assert_identical(
        unchanged_widths.metabs_freq_ranges, params.metabs_freq_ranges
    )
    xr.testing.assert_identical(params, original_params)
    xr.testing.assert_identical(results, original_results)


@pytest.mark.parametrize("source", ["fit_results", "search_widths"])
def test_refinement_rejects_missing_coordinate_labels(fid_data, source):
    """Reject missing voxels instead of silently introducing NaN parameter maps."""
    params = _prepare(fid_data)
    results = params[["initial_freqs_hz", "initial_t2s_s"]].rename(
        {"initial_freqs_hz": "fit_freqs", "initial_t2s_s": "fit_t2s"}
    )
    widths = params.metabs_freq_ranges
    if source == "fit_results":
        results = results.assign_coords(voxel=[20, 30])
    else:
        widths = widths.assign_coords(voxel=[20, 30])
    with pytest.raises(ValueError, match="coordinate"):
        initialise_fit_params_from_results(params, results, freq_ranges_hz=widths)


@pytest.mark.parametrize("width", [-1.0, np.inf])
def test_refinement_rejects_invalid_search_widths(fid_data, width):
    """Apply the same finite, non-negative width rule when preparing a refit."""
    params = _prepare(fid_data)
    results = params[["initial_freqs_hz", "initial_t2s_s"]].rename(
        {"initial_freqs_hz": "fit_freqs", "initial_t2s_s": "fit_t2s"}
    )
    with pytest.raises(ValueError, match="t2_ranges_s"):
        initialise_fit_params_from_results(params, results, t2_ranges_s=width)
