"""Noise-free doublets that the FID grid search alone leaves off their lines.

Three observations and their causes, found 13 November 2026:

1. Frequency bounds are stored variables. Changing ``initial_freqs_hz`` or
   ``metabs_freq_ranges`` of a parameter set in place kept the bounds of the
   values given to ``def_fit_params``, and the search stayed inside them.
   The fit now derives the bounds from the start and the range and warns
   when stored bounds differ; with matching bounds every start of the
   passing-line case of ``test_spectroscopy_t2_coupling_roles.py`` reaches
   the truth.
2. The first frequency step searches the strong line while the weak line is
   still at its start. If the strong line's true frequency lies between two
   grid points, the strong line ends a fraction of the grid spacing off, the
   weak line moves to the other side of the strong peak to absorb that error,
   and later one-line steps keep this state. A grid that contains the truth
   recovers both lines.
3. In ``"amplitude"`` search order the strong line moves first and ends
   0.33 Hz off while the weak line is still far from its peak. The
   alternating frequency and T2* steps then approach the truth along a
   narrow valley more slowly than the windows shrink. ``"metabolite"`` order
   moves the weak line first and reaches the truth.

The joint refinement after the grid passes (``joint_refinement=True``, the
default) moves all lines together and recovers cases 2 and 3; with
``joint_refinement=False`` both failures are reproduced.

Fixture provenance: noise-free FIDs, so the true parameters give a zero
residual. Case 1 uses ``two_voxel_doublet`` (256 samples at 4 kHz; strong
line S area 20 and T2* 30 ms, weak line W area 2 and T2* 25.5 ms, 8 Hz above;
voxel 1 shifted by +8 Hz). Case 2 uses the synthetic study of the default
search order decision: 512 samples at 4 kHz, S at 0 Hz (area 20, phase
-0.2 rad, T2* 30 ms) and W at 8 Hz (area 2, phase 0.5 rad, T2* 25 ms), both
starts 2 Hz too high, 40 Hz ranges and otherwise default settings.

Tolerances: recovered lines lie within 0.1 Hz and 0.1 ms of the truth, the
grid resolution after the zoomed passes. The failures are several Hz or ms
off, far outside these tolerances.
"""

import warnings

import numpy as np
import pytest
import xarray as xr
from test_spectroscopy_t2_coupling_roles import (
    BAND_COUPLING,
    NESTED_RANGES_HZ,
    SHIFT_HZ,
    doublet_params,
    two_voxel_doublet,
)

from hypermri.analysis.fitting.models.spectroscopy.fid_model import (
    _update_frequency_bounds,
)
from hypermri.analysis.fitting.spectroscopy import def_fit_params, fit_data_pseudo_inv

FREQUENCY_TOLERANCE_HZ = 0.1
T2_TOLERANCE_S = 1e-4


def passing_line_params(start_below_hz, *, recompute_bounds):
    """Return case-1 parameters: S frozen, W started below S with 50 Hz.

    The ranges and starts are changed in place after ``def_fit_params``, as
    in ``test_frozen_line_keeps_its_frequency_when_passed``; the frequency
    bounds are recomputed only when ``recompute_bounds`` is true.
    """
    data = two_voxel_doublet()
    params = doublet_params(data, metabs_freqs_hz=[0.0, -start_below_hz]).copy(
        deep=True
    )
    params["metabs_freq_ranges"].loc[{"metabolite": "OMPD1_1"}] = 0.0
    params["metabs_freq_ranges"].loc[{"metabolite": "OMPD1_2"}] = 50.0
    params["initial_freqs_hz"].loc[{"metabolite": "OMPD1_1"}] = [0.0, SHIFT_HZ]
    params["initial_freqs_hz"].loc[{"metabolite": "OMPD1_2"}] = [
        -start_below_hz,
        SHIFT_HZ - start_below_hz,
    ]
    params = params.assign_attrs(t2_coupling=BAND_COUPLING, t2_coupling_clip_start=True)
    if recompute_bounds:
        params = _update_frequency_bounds(params)
    return data, params


def test_in_place_start_change_keeps_the_old_frequency_bounds():
    _, params = passing_line_params(6.0, recompute_bounds=False)

    # def_fit_params gave W the start -6 Hz and the 40 Hz range of the fixture.
    np.testing.assert_array_equal(
        params.freq_range_upper_bound_hz.sel(metabolite="OMPD1_2"), [14.0, 14.0]
    )


def test_stale_bounds_are_replaced_with_a_warning():
    data, params = passing_line_params(6.0, recompute_bounds=False)

    with pytest.warns(
        UserWarning, match=r"fit_data_pseudo_inv.*\['OMPD1_1', 'OMPD1_2'\]"
    ):
        result = fit_data_pseudo_inv(data, params)

    # W lies at 16 Hz in voxel 1, above the stale bound of 14 Hz.
    np.testing.assert_allclose(
        result.fit_freqs.sel(metabolite="OMPD1_1").values,
        [SHIFT_HZ, 2 * SHIFT_HZ],
        atol=FREQUENCY_TOLERANCE_HZ,
    )


@pytest.mark.parametrize("start_below_hz", [5.0, 6.0, 7.0, 10.0, 14.0])
def test_passing_line_reaches_the_weak_line_within_its_bounds(start_below_hz):
    data, params = passing_line_params(start_below_hz, recompute_bounds=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = fit_data_pseudo_inv(data, params)

    np.testing.assert_allclose(
        result.fit_freqs.sel(metabolite="OMPD1_1").values,
        [SHIFT_HZ, 2 * SHIFT_HZ],
        atol=FREQUENCY_TOLERANCE_HZ,
    )
    np.testing.assert_allclose(
        result.fit_t2s.values, [[0.0255, 0.030]] * 2, atol=T2_TOLERANCE_S
    )


def study_doublet():
    """Return the noise-free study doublet with S at 0 Hz and W at 8 Hz."""
    time_s = np.arange(512) / 4000.0
    fid = 20.0 * np.exp(-0.2j) * np.exp(-time_s / 0.030) + 2.0 * np.exp(0.5j) * np.exp(
        2j * np.pi * 8.0 * time_s - time_s / 0.025
    )
    data = xr.DataArray(fid[None], dims=["x", "acquisition_idx"], coords={"x": [0]})
    data.attrs["sampling_time_axis_s"] = time_s
    return data


def fit_study_doublet(npoints, *, joint_refinement=True):
    """Fit the study doublet from starts 2 Hz too high with ``npoints``."""
    data = study_doublet()
    params = def_fit_params(
        data,
        {
            "metabs": ["strong", "weak"],
            "metabs_freqs_hz": [2.0, 10.0],
            "metabs_freq_ranges": 40.0,
            "npoints": npoints,
            "joint_refinement": joint_refinement,
        },
    )
    return fit_data_pseudo_inv(data, params)


def test_study_doublet_is_recovered_when_the_first_grid_holds_the_truth():
    # 41 points over 40 Hz: 1 Hz spacing, the truth lies on the first grid.
    result = fit_study_doublet(41)

    np.testing.assert_allclose(
        result.fit_freqs.values[0], [0.0, 8.0], atol=FREQUENCY_TOLERANCE_HZ
    )


def test_grid_alone_takes_the_weak_line_off_between_grid_points():
    # 31 points: 1.33 Hz spacing, the truth lies midway between two points.
    result = fit_study_doublet(31, joint_refinement=False)

    strong, weak = result.fit_freqs.values[0]
    assert abs(strong) > 5.0
    assert abs(weak - 8.0) > 8.0


def test_study_doublet_is_recovered_between_grid_points():
    result = fit_study_doublet(31)

    np.testing.assert_allclose(
        result.fit_freqs.values[0], [0.0, 8.0], atol=FREQUENCY_TOLERANCE_HZ
    )


def fit_nested_windows(search_order, *, joint_refinement=True):
    """Fit voxel 1 of the nested-window doublet without ``t2_coupling``."""
    data = two_voxel_doublet().isel(x=[1])
    params = doublet_params(
        data,
        metabs_freq_ranges=NESTED_RANGES_HZ,
        search_order=search_order,
        joint_refinement=joint_refinement,
    )
    return fit_data_pseudo_inv(data, params)


def test_nested_windows_reach_the_truth_in_metabolite_order():
    result = fit_nested_windows("metabolite")

    np.testing.assert_allclose(
        result.fit_freqs.values[0],
        [SHIFT_HZ, 2 * SHIFT_HZ],
        atol=FREQUENCY_TOLERANCE_HZ,
    )
    np.testing.assert_allclose(
        result.fit_t2s.values[0], [0.030, 0.0255], atol=T2_TOLERANCE_S
    )


def test_grid_alone_stops_short_of_nested_windows_in_amplitude_order():
    result = fit_nested_windows("amplitude", joint_refinement=False)

    assert np.max(np.abs(result.fit_freqs.values[0] - [SHIFT_HZ, 2 * SHIFT_HZ])) > (
        FREQUENCY_TOLERANCE_HZ
    )


def test_nested_windows_reach_the_truth_in_amplitude_order():
    result = fit_nested_windows("amplitude")

    np.testing.assert_allclose(
        result.fit_freqs.values[0],
        [SHIFT_HZ, 2 * SHIFT_HZ],
        atol=FREQUENCY_TOLERANCE_HZ,
    )
    np.testing.assert_allclose(
        result.fit_t2s.values[0], [0.030, 0.0255], atol=T2_TOLERANCE_S
    )
