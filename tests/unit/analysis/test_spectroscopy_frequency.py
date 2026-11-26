"""Characterize the frequency/ppm group across its move into its own module.

All axes and metadata are invented. Independent calibration is
Hz = 100 * (ppm - 170), with its sign reversed for descending ppm. Conversion
checks allow 1e-12 rounding; bins, nearest-index choices and return types retain
their discrete legacy contracts. This migration does not standardize them.
"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy.frequency import (
    _convert_ppm_ndarray,
    _convert_ppm_value,
    _ppm_delta_to_hz,
    center_freq_axis,
    freq_hz_to_ppm,
    freq_ppm_to_hz,
    freq_to_index,
    get_bw_from_freq_axis,
    get_freq_axis,
)


def _context(**overrides):
    metadata = {"center_ppm": 170.0, "dwell_time_s": 0.001, "cut_off": 1}
    attributes = {
        "bandwidth_hz": 1000.0,
        "bandwidth_ppm": 10.0,
        "n_acquisition": 5,
        "get": metadata.get,
        **overrides,
    }
    return SimpleNamespace(**attributes), metadata


@pytest.mark.parametrize("descending", [False, True])
def test_nearest_index_ties_and_endpoint_selection(descending):
    axis = np.array([-200.0, -100.0, 0.0, 100.0])
    if descending:
        axis = axis[::-1]
    before = axis.copy()
    assert freq_to_index(0.0, axis) == (1 if descending else 2)
    assert freq_to_index(-150.0, axis) == (2 if descending else 0)
    assert freq_to_index(-1000.0, axis) == (3 if descending else 0)
    np.testing.assert_array_equal(axis, before)


@pytest.mark.parametrize("descending", [False, True])
def test_bandwidth_counts_the_first_sample_width(descending):
    # Nonuniform spacing deliberately records the existing first-step rule.
    axis = np.array([-200.0, -100.0, 50.0, 300.0])
    if descending:
        axis = axis[::-1]
    expected = 750.0 if descending else 600.0
    np.testing.assert_allclose(get_bw_from_freq_axis(axis), expected, rtol=1e-12)


def test_bandwidth_context_route_and_exclusive_inputs():
    context, _ = _context()
    np.testing.assert_allclose(
        get_bw_from_freq_axis(None, context, unit="ppm"), 10.0, rtol=1e-12
    )
    with pytest.raises(ValueError, match="either"):
        get_bw_from_freq_axis(None)
    with pytest.raises(ValueError, match="AND"):
        get_bw_from_freq_axis(np.arange(4), context)
    with pytest.raises(IndexError):
        get_bw_from_freq_axis(np.array([1.0]))


@pytest.mark.parametrize("descending", [False, True])
def test_centering_preserves_precision_and_centers_both_directions(descending):
    axis = np.array([-200.0, -100.0, 0.0, 100.0], dtype=np.float32)
    if descending:
        axis = axis[::-1]
    before = axis.copy()
    expected = (
        [150.0, 50.0, -50.0, -150.0] if descending else [-150.0, -50.0, 50.0, 150.0]
    )
    result = center_freq_axis(axis)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(axis, before)


@pytest.mark.parametrize("unit", ["Hz", "HZ", "ppm", "PPM"])
def test_context_sampling_precedence_and_preservation(unit):
    context, metadata = _context()
    before = deepcopy(vars(context))
    metadata_before = metadata.copy()
    result = get_freq_axis(context, unit=unit, dwell_time_s=99.0)
    expected = np.array([-500.0, -250.0, 0.0, 250.0])
    if unit.lower() == "ppm":
        expected = expected / 100 + 170
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert result.dtype == np.float64
    assert vars(context) == before
    assert metadata == metadata_before
    repeated = get_freq_axis(context, unit=unit, npoints=4, cut_off=3)
    np.testing.assert_allclose(repeated, result, rtol=1e-12, atol=1e-12)


def test_explicit_tuple_length_and_cutoff_keep_requested_bins():
    np.testing.assert_allclose(
        get_freq_axis(unit="Hz", npoints=(4,), dwell_time_s=0.001, cut_off=3),
        [-500.0, -250.0, 0.0, 250.0],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("cut_off", [5, 70])
def test_cut_off_without_remaining_points_names_the_lengths(cut_off):
    """Regression test: a CPMG echo of 10 points minus a 70-point
    cut-off reached numpy as a negative FFT length."""
    context, _ = _context()
    with pytest.raises(ValueError, match=f"cut_off={cut_off} .*n_acquisition=5"):
        get_freq_axis(context, unit="Hz", cut_off=cut_off)


@pytest.mark.parametrize("bandwidth", [None, 0.0])
def test_unavailable_ppm_calibration_returns_none(bandwidth):
    context, _ = _context(bandwidth_ppm=bandwidth)
    assert get_freq_axis(context, unit="ppm") is None


def test_missing_hz_metadata_retains_none_and_empty_list_distinction():
    context, _ = _context(bandwidth_hz=None)
    assert get_freq_axis(context, unit="Hz") == []
    del context.bandwidth_hz
    assert get_freq_axis(context, unit="Hz") is None


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("kind", ["scalar", "list", "array"])
def test_hz_to_ppm_affine_calibration_return_types_and_input_preservation(
    flipped, kind
):
    hz = np.array([-200.0, -100.0, 0.0, 100.0])
    ppm = np.array([168.0, 169.0, 170.0, 171.0])
    value = {"scalar": 50.0, "list": [-50.0, 50.0], "array": np.array([[-50.0, 50.0]])}[
        kind
    ]
    before = deepcopy(value)
    result = freq_hz_to_ppm(value, hz, ppm, ppm_axis_flipped=flipped)
    expected = (
        169.0 - np.asarray(value) / 100 if flipped else 170 + np.asarray(value) / 100
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert isinstance(
        result,
        list if kind == "list" else np.ndarray if kind == "array" else np.floating,
    )
    np.testing.assert_array_equal(value, before)
    np.testing.assert_array_equal(hz, [-200, -100, 0, 100])
    np.testing.assert_array_equal(ppm, [168, 169, 170, 171])


@pytest.mark.parametrize("flipped", [False, True])
@pytest.mark.parametrize("values", [170.5, [170.5], [169.0, None, 171.0]])
def test_ppm_to_hz_scalar_list_and_none_contract(flipped, values):
    hz = np.array([-200.0, -100.0, 0.0, 100.0])
    ppm = np.array([168.0, 169.0, 170.0, 171.0])
    before = deepcopy(values)
    result = freq_ppm_to_hz(values, hz, ppm, ppm_axis_flipped=flipped)
    if isinstance(values, list) and len(values) > 1:
        assert isinstance(result, list)
        assert result[1] is None
        expected = [0.0, -200.0] if flipped else [-100.0, 100.0]
        np.testing.assert_allclose(
            [result[0], result[2]], expected, rtol=1e-12, atol=1e-12
        )
    else:
        assert isinstance(result, np.floating)
        np.testing.assert_allclose(
            result, -150.0 if flipped else 50.0, rtol=1e-12, atol=1e-12
        )
    assert values == before


def test_conversion_centering_and_missing_context_contracts():
    hz = np.array([-200.0, -100.0, 0.0, 100.0])
    ppm = np.array([168.0, 169.0, 170.0, 171.0])
    np.testing.assert_allclose(
        freq_hz_to_ppm(50.0, hz, ppm, ppm_centered_at_0=True), 0.5, rtol=1e-12
    )
    context, _ = _context()
    np.testing.assert_allclose(freq_hz_to_ppm(50.0, scan=context), 170.5, rtol=1e-12)
    np.testing.assert_allclose(freq_ppm_to_hz(170.5, scan=context), 50.0, rtol=1e-12)
    with pytest.raises(ValueError, match="Provide either"):
        freq_hz_to_ppm(1.0)
    with pytest.raises(ValueError, match="Provide either"):
        freq_ppm_to_hz(1.0)


def test_absolute_and_relative_ppm_conversion_keep_shape_dtype_and_label_policy():
    hz = np.array([-200.0, -100.0, 0.0, 100.0])
    ppm = np.array([168.0, 169.0, 170.0, 171.0])
    value = xr.DataArray(
        [[169.0, 170.0], [171.0, 168.0]],
        dims=("voxel", "metabolite"),
        coords={"voxel": [4.0, -4.0]},
        attrs={"units": "ppm"},
    )
    before = value.copy(deep=True)
    result = _convert_ppm_value(value, hz_axis=hz, ppm_axis=ppm)
    assert (
        type(result) is np.ndarray
    )  # This helper deliberately returns unlabelled values.
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [[-100, 0], [100, -200]], rtol=1e-12, atol=1e-12)
    delta = _ppm_delta_to_hz(np.array([[1.0, -2.0]]), hz_axis=hz, ppm_axis=ppm[::-1])
    np.testing.assert_allclose(delta, [[-100, 200]], rtol=1e-12, atol=1e-12)
    assert _convert_ppm_value(None, hz_axis=None, ppm_axis=None) is None
    assert _convert_ppm_ndarray(np.empty((0, 2)), hz_axis=hz, ppm_axis=ppm).shape == (
        0,
        2,
    )
    xr.testing.assert_identical(value, before)


def _imaging_context(**overrides):
    """Non-spectroscopic scan: no bandwidths, one acquisition point per read.

    Invented values: transmit centre at 170 ppm, 100 MHz reference frequency,
    so Hz = 100 * (ppm - 170) as in the spectroscopic fixtures above.
    """
    return _context(
        bandwidth_hz=None,
        bandwidth_ppm=None,
        n_acquisition=1,
        reference_frequency_mhz=100.0,
        **overrides,
    )


def test_ppm_to_hz_metadata_line_matches_axis_regression():
    spectral, _ = _context(reference_frequency_mhz=100.0)
    imaging, _ = _imaging_context()
    values = [166.0, 170.0, 171.5]
    np.testing.assert_allclose(
        freq_ppm_to_hz(values, scan=imaging),
        freq_ppm_to_hz(values, scan=spectral),
        rtol=1e-12,
    )
    np.testing.assert_allclose(freq_ppm_to_hz(166.08, scan=imaging), -392.0, rtol=1e-12)
    assert freq_ppm_to_hz([170.0, None, 171.0], scan=imaging)[1] is None


def test_ppm_to_hz_metadata_line_prefers_handler_and_preserves_context():
    imaging, metadata = _imaging_context()
    imaging.handler = SimpleNamespace(reference_frequency_mhz=50.0)
    imaging.reference_frequency_mhz = 100.0
    before, metadata_before = deepcopy(vars(imaging)), metadata.copy()
    np.testing.assert_allclose(freq_ppm_to_hz(171.0, scan=imaging), 50.0)
    assert vars(imaging).keys() == before.keys()
    assert metadata == metadata_before


@pytest.mark.parametrize("missing", ["reference_frequency_mhz", "center_ppm"])
def test_ppm_to_hz_metadata_line_names_missing_metadata(missing):
    imaging, metadata = _imaging_context()
    if missing == "center_ppm":
        del metadata["center_ppm"]
    else:
        imaging.reference_frequency_mhz = None
    with pytest.raises(ValueError, match="reference_frequency_mhz"):
        freq_ppm_to_hz(166.08, scan=imaging)


def test_ppm_to_hz_supplied_axes_never_use_metadata_line():
    imaging, _ = _imaging_context()
    hz = np.array([-200.0, -100.0, 0.0, 100.0])
    ppm = np.array([168.0, 169.0, 170.0, 171.0])
    np.testing.assert_allclose(
        freq_ppm_to_hz(170.5, hz, ppm, scan=imaging), 50.0, rtol=1e-12
    )


def test_hz_to_ppm_metadata_line_matches_axis_regression():
    spectral, _ = _context(reference_frequency_mhz=100.0)
    imaging, _ = _imaging_context()
    values = [-392.0, 0.0, 150.0]
    np.testing.assert_allclose(
        freq_hz_to_ppm(values, scan=imaging),
        freq_hz_to_ppm(values, scan=spectral),
        rtol=1e-12,
    )
    np.testing.assert_allclose(freq_hz_to_ppm(-392.0, scan=imaging), 166.08)
    np.testing.assert_allclose(
        freq_hz_to_ppm(np.array([[0.0, 100.0]]), scan=imaging, ppm_centered_at_0=True),
        [[0.0, 1.0]],
        atol=1e-12,
    )


def test_ppm_hz_metadata_round_trip_and_missing_metadata_error():
    imaging, _ = _imaging_context()
    np.testing.assert_allclose(
        freq_hz_to_ppm(freq_ppm_to_hz(166.08, scan=imaging), scan=imaging),
        166.08,
        rtol=1e-12,
    )
    imaging.reference_frequency_mhz = None
    with pytest.raises(ValueError, match="freq_hz_to_ppm.*reference_frequency_mhz"):
        freq_hz_to_ppm(10.0, scan=imaging)


def test_add_fit_freqs_ppm_converts_fit_frequencies_with_result_axes():
    """fit_freqs_ppm follows fit_freqs; attrs stay without legacy aliases."""
    from hypermri.analysis.spectroscopy.frequency import add_fit_freqs_ppm

    hz = np.array([-200.0, 0.0, 200.0])
    results = xr.Dataset(
        {"fit_freqs": (("x", "metabolite"), [[100.0, -50.0]])},
        attrs={"freq_axis_hz": hz, "freq_axis_ppm": 170.0 + hz / 100.0},
    )
    attrs_before = set(results.attrs)

    updated = add_fit_freqs_ppm(results)

    np.testing.assert_allclose(updated["fit_freqs_ppm"], [[171.0, 169.5]], atol=1e-12)
    assert updated["fit_freqs_ppm"].dims == ("x", "metabolite")
    assert updated["fit_freqs_ppm"].attrs["unit"] == "ppm"
    assert set(results.attrs) == attrs_before
    assert "fit_freqs_ppm" not in results


def test_add_fit_freqs_ppm_leaves_results_without_axes_unchanged():
    from hypermri.analysis.spectroscopy.frequency import add_fit_freqs_ppm

    results = xr.Dataset({"fit_freqs": (("metabolite",), [100.0])})

    assert add_fit_freqs_ppm(results) is results
