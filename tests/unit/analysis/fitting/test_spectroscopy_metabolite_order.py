"""Characterize the per-spectrum metabolite ordering after the FID fit.

``keep_fit_order=False`` (the default) relabels fitted components so that
fitted frequencies follow the order of the mean initial frequencies. The fit
fixtures are synthetic noise-free two-voxel FIDs: 256 samples at 4 kHz with
lines at 0 and 20 Hz (areas 10 and 6) and a separated line at -300 Hz. The
search ranges of 'b' and 'c' overlap, and voxel 1 starts them in reversed
order, so the solver order and the label order disagree in that voxel.
Relabelling only permutes outputs, so permuted values are compared exactly and
the reconstructed signal to floating-point precision (rtol 1e-12).
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _update_frequency_bounds,
)
from hypermri.analysis.fitting.spectroscopy import (
    _component_index_by_frequency,
    _exchangeable_metabolite_groups,
    def_fit_params,
    fit_data_pseudo_inv,
    reconstruct_fitted_signal,
)

FREQS_HZ = np.array([-300.0, 0.0, 20.0])
AMPLITUDES = np.array([10.0, 10.0, 6.0]) * np.exp(1j * np.array([0.3, -0.2, 0.5]))


def two_voxel_fid():
    """Return two identical labelled noise-free FIDs of three lines."""
    t = np.arange(256) / 4000.0
    lines = AMPLITUDES[:, None] * np.exp(2j * np.pi * FREQS_HZ[:, None] * t - t / 0.03)
    data = xr.DataArray(
        np.repeat(lines.sum(0)[None], 2, axis=0),
        dims=["x", "acquisition_idx"],
        coords={"x": [0, 1]},
    )
    data.attrs["sampling_time_axis_s"] = t
    return data


def overlapping_params(data, **options):
    """Start 'b' below 'c' in voxel 0 and above it in voxel 1 (60 Hz ranges)."""
    params = def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a", "b", "c"],
            "metabs_freqs_hz": [-299.0, 2.0, 18.0],
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": [8.0, 60.0, 60.0],
            "min_t2_s": 0.005,
            "max_t2_s": 0.1,
            **options,
        },
    )
    initial = params["initial_freqs_hz"].copy()
    initial.loc[{"x": 1, "metabolite": "b"}] = 18.0
    initial.loc[{"x": 1, "metabolite": "c"}] = 2.0
    return _update_frequency_bounds(params.assign(initial_freqs_hz=initial))


def test_component_index_sorts_active_components_within_groups():
    """Active lines are sorted per spectrum; skipped and NaN lines keep labels."""
    freqs_hz = np.array(
        [
            [5.0, 1.0, 9.0, 3.0],  # swap 0 and 1; 2 and 3 are another group
            [1.0, 5.0, 3.0, 9.0],  # already ordered
            [5.0, 1.0, 9.0, 3.0],  # 0 skipped: nothing left to exchange in its group
            [np.nan, 1.0, 9.0, 3.0],  # 0 not finite: same
            [2.0, 2.0, 9.0, 3.0],  # equal frequencies keep target order
        ]
    )
    skipped = np.zeros_like(freqs_hz, dtype=bool)
    skipped[2, 0] = True
    index = _component_index_by_frequency(
        freqs_hz, skipped, groups=[np.array([0, 1]), np.array([3, 2])]
    )
    np.testing.assert_array_equal(
        index,
        [[1, 0, 2, 3], [0, 1, 3, 2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
    )


def test_partially_skipped_group_sorts_remaining_components():
    """A voxel-specific skip removes one line; the remaining lines are sorted."""
    index = _component_index_by_frequency(
        np.array([[9.0, 7.0, 5.0]]),
        np.array([[False, True, False]]),
        groups=[np.array([0, 1, 2])],
    )
    np.testing.assert_array_equal(index, [[2, 1, 0]])


def test_groups_use_mean_initial_frequency_then_name():
    """Inhomogeneous starts use their mean; equal means are ordered by name."""
    data = two_voxel_fid()
    params = overlapping_params(data)
    # Means: a -299, b 10, c 10 -> b and c tie and are ordered by name.
    groups = _exchangeable_metabolite_groups(params)
    assert [group.tolist() for group in groups] == [[0, 1, 2]]
    params = params.assign_coords(metabolite=["z", "c", "b"])
    groups = _exchangeable_metabolite_groups(params)
    assert [group.tolist() for group in groups] == [[0, 2, 1]]


def test_groups_respect_constraint_roles():
    """Coupled lines keep their labels; phase-band lines group by tolerance."""
    data = two_voxel_fid()
    coupled = overlapping_params(data, t2_coupling={"c": "b"})
    assert _exchangeable_metabolite_groups(coupled) == []
    banded = overlapping_params(
        data, phase_band_tolerance_deg={"b": 5.0, "c": 5.0}, phase_band_tau_s=0.0
    )
    assert [group.tolist() for group in _exchangeable_metabolite_groups(banded)] == [
        [1, 2]
    ]
    mixed = overlapping_params(
        data, phase_band_tolerance_deg={"b": 5.0, "c": 10.0}, phase_band_tau_s=0.0
    )
    assert _exchangeable_metabolite_groups(mixed) == []


def test_fit_labels_follow_initial_frequency_order():
    """Sorted labels are a per-voxel permutation of the solver output."""
    data = two_voxel_fid()
    params = overlapping_params(data, store_optimization_history=True)
    raw = fit_data_pseudo_inv(data, params.assign_attrs(keep_fit_order=True))
    result = fit_data_pseudo_inv(data, params)

    assert "fit_component_index" not in raw
    assert raw.attrs["keep_fit_order"] is True
    assert result.attrs["keep_fit_order"] is False
    freqs = result.fit_freqs
    assert bool((freqs.sel(metabolite="b") <= freqs.sel(metabolite="c")).all())
    # Voxel 1 starts 'b' above 'c', so the solver order is reversed there.
    raw_freqs = raw.fit_freqs.sel(x=1)
    assert raw_freqs.sel(metabolite="b") > raw_freqs.sel(metabolite="c")
    np.testing.assert_array_equal(
        result.fit_component_index.transpose("x", "metabolite"),
        [[0, 1, 2], [0, 2, 1]],
    )

    for name in ("fit_amps", "fit_freqs", "fit_t2s", "fit_fwhm_hz", "fit_r2_freq"):
        expected = raw[name].sel(x=1).isel(metabolite=[0, 2, 1])
        np.testing.assert_array_equal(
            result[name].sel(x=1).values, expected.values, err_msg=name
        )
        xr.testing.assert_identical(result[name].sel(x=0), raw[name].sel(x=0))
        assert result[name].dims == raw[name].dims
        assert result[name].attrs == raw[name].attrs
    xr.testing.assert_identical(result.fit_history_freqs, raw.fit_history_freqs)
    xr.testing.assert_identical(result.fit_rss, raw.fit_rss)
    np.testing.assert_allclose(
        reconstruct_fitted_signal(result, data),
        reconstruct_fitted_signal(raw, data),
        rtol=1e-12,
    )


def test_keep_fit_order_must_be_boolean():
    """The option is validated before fitting."""
    data = two_voxel_fid()
    params = overlapping_params(data).assign_attrs(keep_fit_order="no")
    with pytest.raises(ValueError, match="keep_fit_order must be boolean"):
        fit_data_pseudo_inv(data, params)


def test_dask_results_are_sorted_lazily():
    """Dask-backed input keeps the relabelling lazy and gives the same labels."""
    pytest.importorskip("dask")
    data = two_voxel_fid()
    params = overlapping_params(data)
    eager = fit_data_pseudo_inv(data, params)
    lazy = fit_data_pseudo_inv(data.chunk({"x": 1}), params)
    assert lazy.fit_component_index.chunks is not None
    xr.testing.assert_identical(lazy.compute().fit_freqs, eager.fit_freqs)
    xr.testing.assert_identical(
        lazy.compute().fit_component_index, eager.fit_component_index
    )
