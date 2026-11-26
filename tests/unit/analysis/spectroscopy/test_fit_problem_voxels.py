"""Order fitted voxels for review by bound flags and reduced chi-squared.

Fixtures are invented result datasets on a 2 x 3 voxel grid with two lines;
the order follows from the stored statuses and values alone, so exact
comparison is appropriate. One fit-based case checks the contract with real
``fit_data_pseudo_inv`` results.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv
from hypermri.analysis.spectroscopy import (
    order_fit_problem_voxels,
    summarize_fit_bound_flags,
)

DIMS = ("x", "y", "metabolite")


def _results(*, chi2=None, with_status=True, x_first=True):
    shape = (2, 3, 2)
    coords = {"x": [10, 11], "y": [0, 1, 2], "metabolite": ["pyr", "lac"]}
    freq_status = np.full(shape, "estimated", dtype=object)
    t2_status = np.full(shape, "estimated", dtype=object)
    freq_status[0, 2, 1] = "at_upper_bound"
    t2_status[1, 0, 0] = "at_lower_bound"
    t2_status[1, 0, 1] = "at_upper_bound"
    freq_status[1, 1, :] = "not_fitted"
    t2_status[1, 1, :] = "not_fitted"
    t2_status[0, 1, 0] = "fixed"
    rss = np.ones(shape[:2])
    rss[1, 1] = np.nan
    if chi2 is None:
        chi2 = np.array([[1.1, 3.0, 0.9], [2.0, np.nan, 5.0]])
    results = xr.Dataset(
        {
            "fit_freqs": (DIMS, np.zeros(shape)),
            "fit_rss": (DIMS[:2], rss),
            "fit_reduced_chi_squared": (DIMS[:2], np.asarray(chi2, dtype=float)),
        },
        coords=coords,
    )
    if with_status:
        results["fit_freq_status"] = (DIMS, freq_status.astype(str))
        results["fit_t2_status"] = (DIMS, t2_status.astype(str))
    if not x_first:
        results = results.transpose("metabolite", "y", "x")
    return results


def test_flags_list_each_line_at_a_bound_in_metabolite_order():
    flags = summarize_fit_bound_flags(_results())

    assert flags.dims == ("x", "y")
    assert flags.coords["x"].values.tolist() == [10, 11]
    assert flags.values.tolist() == [
        ["", "", "lac frequency at_upper_bound"],
        ["pyr T2* at_lower_bound; lac T2* at_upper_bound", "", ""],
    ]


def test_flags_need_a_status_variable():
    with pytest.raises(KeyError, match="neither fit_freq_status nor fit_t2_status"):
        summarize_fit_bound_flags(_results(with_status=False))


def test_order_puts_flagged_voxels_first_then_highest_chi2():
    results = _results()
    original = results.copy(deep=True)

    order = order_fit_problem_voxels(results)

    positions = list(zip(order.x.values.tolist(), order.y.values.tolist(), strict=True))
    # Flagged: (1, 0) chi2 2.0 before (0, 2) chi2 0.9; then 5.0, 3.0, 1.1;
    # (1, 1) is not fitted and has no flag, so it is left out.
    assert positions == [(1, 0), (0, 2), (1, 2), (0, 1), (0, 0)]
    np.testing.assert_array_equal(
        order.fit_reduced_chi_squared.values, [2.0, 0.9, 5.0, 3.0, 1.1]
    )
    assert order.fit_bound_flags.values[1] == "lac frequency at_upper_bound"
    assert order.attrs["spectrum_dims"] == ["x", "y"]
    xr.testing.assert_identical(results, original)


def test_order_does_not_depend_on_dimension_order():
    order = order_fit_problem_voxels(_results(x_first=False))

    assert order.attrs["spectrum_dims"] == ["y", "x"]
    positions = list(zip(order.x.values.tolist(), order.y.values.tolist(), strict=True))
    assert positions == [(1, 0), (0, 2), (1, 2), (0, 1), (0, 0)]


def test_without_chi2_or_statuses_the_order_is_index_order():
    results = _results(chi2=np.full((2, 3), np.nan), with_status=False)

    order = order_fit_problem_voxels(results)

    positions = list(zip(order.x.values.tolist(), order.y.values.tolist(), strict=True))
    assert positions == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)]
    assert set(order.fit_bound_flags.values) == {""}


def test_order_needs_rss_and_frequencies():
    with pytest.raises(KeyError, match="fit_rss"):
        order_fit_problem_voxels(_results().drop_vars("fit_rss"))


def test_order_reads_real_fit_results():
    # Two noise-free 30 ms lines, 1024 points at 1 kHz; the line at 6 Hz lies
    # outside the +/- 2 Hz window and ends at its upper bound.
    time_s = np.arange(1024) / 1000.0
    fids = np.array(
        [np.exp(-time_s / 0.03 + 2j * np.pi * f * time_s) for f in (0.5, 6.0)]
    )
    data = xr.DataArray(
        fids,
        dims=("x", "acquisition_idx"),
        coords={"x": [0, 1]},
        attrs={"signal_domain": "temporal", "sampling_time_axis_s": time_s},
    )
    params = def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a"],
            "metabs_freqs_hz": [0.0],
            "metabs_freq_ranges": [4.0],
            "rep_fitting": 2,
            "npoints": 11,
        },
    )

    order = order_fit_problem_voxels(fit_data_pseudo_inv(data, params))

    assert order.x.values.tolist() == [1, 0]
    assert order.fit_bound_flags.values.tolist() == [
        "a frequency at_upper_bound",
        "",
    ]
