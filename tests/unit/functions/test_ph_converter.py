import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy.ph import (
    LAMOR_FREQUENCY_C13,
    PARAMS_OMPD_C1,
    PARAMS_OMPD_C5,
    OMPD5_freq2pH,
    OMPD_pH_from_absolute_position,
    OMPD_pH_from_c1_c5,
    OMPD_pH_from_peak_shift,
    freq_to_pH,
    pH_to_freq,
    pH_to_ppm,
    ppm_to_pH,
)


def test_ppm_and_ph_conversions_are_inverse_for_scalars_and_arrays():
    ph_values = np.array([6.7, 7.0, 7.4])

    shifts = pH_to_ppm(ph_values, PARAMS_OMPD_C5)

    np.testing.assert_allclose(ppm_to_pH(shifts, PARAMS_OMPD_C5), ph_values)
    assert ppm_to_pH(pH_to_ppm(7.1, PARAMS_OMPD_C5), PARAMS_OMPD_C5) == pytest.approx(
        7.1
    )


def test_frequency_conversion_round_trip_uses_larmor_frequency():
    ph_value = 7.2

    frequency_hz = pH_to_freq(
        ph_value, PARAMS_OMPD_C5, larmor_frequency_mhz=LAMOR_FREQUENCY_C13
    )

    assert frequency_hz == pytest.approx(
        pH_to_ppm(ph_value, PARAMS_OMPD_C5) * LAMOR_FREQUENCY_C13
    )
    assert freq_to_pH(
        frequency_hz, PARAMS_OMPD_C5, larmor_frequency_mhz=LAMOR_FREQUENCY_C13
    ) == pytest.approx(ph_value)
    assert OMPD5_freq2pH(
        frequency_hz, larmor_frequency_mhz=LAMOR_FREQUENCY_C13
    ) == pytest.approx(ph_value)


def test_frequency_conversion_can_derive_larmor_frequency_from_fieldstrength():
    ph_value = 7.2

    frequency_hz = pH_to_freq(ph_value, PARAMS_OMPD_C5, fieldstrength_t=7.0)

    assert freq_to_pH(
        frequency_hz,
        PARAMS_OMPD_C5,
        fieldstrength_t=7.0,
    ) == pytest.approx(ph_value)
    assert OMPD5_freq2pH(frequency_hz, fieldstrength_T=7.0) == pytest.approx(ph_value)


@pytest.mark.parametrize("reference_mhz", [1.0, 75.468, 150.0])
def test_midpoint_has_independent_hz_scale(reference_mhz):
    """At pH=pKa, delta is the average endpoint shift: Hz=ppm*MHz.

    A round trip alone would miss a shared factor-of-one-million error.
    """
    pka, acid_ppm, base_ppm = PARAMS_OMPD_C5
    expected_hz = (acid_ppm + base_ppm) / 2 * reference_mhz
    options = {"larmor_frequency_mhz": reference_mhz}
    assert pH_to_freq(pka, PARAMS_OMPD_C5, **options) == pytest.approx(expected_hz)
    assert freq_to_pH(expected_hz, PARAMS_OMPD_C5, **options) == pytest.approx(pka)
    assert OMPD5_freq2pH(expected_hz, **options) == pytest.approx(pka)


def test_positional_field_strength_and_default_keep_tesla_contract():
    """Retain positional Tesla, lowercase keyword and historical uppercase alias."""
    ph = np.array([6.7, 7.0, 7.4])
    expected = pH_to_ppm(ph, PARAMS_OMPD_C5) * (10.7084 * 7.0)
    for options in ({}, {"fieldstrength_t": 7.0}, {"fieldstrength_T": 7.0}):
        np.testing.assert_allclose(pH_to_freq(ph, PARAMS_OMPD_C5, **options), expected)
        np.testing.assert_allclose(freq_to_pH(expected, PARAMS_OMPD_C5, **options), ph)
        np.testing.assert_allclose(OMPD5_freq2pH(expected, **options), ph)
    np.testing.assert_allclose(pH_to_freq(ph, PARAMS_OMPD_C5, 7.0), expected)
    np.testing.assert_allclose(pH_to_freq(ph, PARAMS_OMPD_C5, 14.0), 2 * expected)


def test_frequency_conversion_preserves_labelled_arrays():
    """Keep voxel coordinates when these helpers receive xarray parameter maps."""
    ph = xr.DataArray([6.7, 7.4], dims="voxel", coords={"voxel": ["a", "b"]})
    frequency = pH_to_freq(ph, PARAMS_OMPD_C5, larmor_frequency_mhz=75.468)
    xr.testing.assert_allclose(frequency, pH_to_ppm(ph, PARAMS_OMPD_C5) * 75.468)
    xr.testing.assert_allclose(
        OMPD5_freq2pH(frequency, larmor_frequency_mhz=75.468), ph
    )


@pytest.mark.parametrize(
    "options",
    [
        {"fieldstrength_t": 7.0, "fieldstrength_T": 3.0},
        {"fieldstrength_t": 7.0, "larmor_frequency_mhz": 75.468},
        {"fieldstrength_t": np.nan},
        {"fieldstrength_T": -7.0},
        {"larmor_frequency_mhz": 0.0},
        {"larmor_frequency_mhz": np.inf},
        {"larmor_frequency_mhz": [75.468, 150.0]},
        {"larmor_frequency_mhz": 75.468 + 1j},
    ],
)
def test_invalid_or_conflicting_reference_is_rejected(options):
    for converter, value in ((pH_to_freq, 7.0), (freq_to_pH, 1000.0)):
        with pytest.raises(ValueError):
            converter(value, PARAMS_OMPD_C5, **options)
    with pytest.raises(ValueError):
        OMPD5_freq2pH(1000.0, **options)


def _c1_c5_positions(ph_value, offset_ppm):
    """Peak positions from the published C1/C5 calibrations plus a common offset."""
    return (
        pH_to_ppm(ph_value, PARAMS_OMPD_C1) + offset_ppm,
        pH_to_ppm(ph_value, PARAMS_OMPD_C5) + offset_ppm,
    )


def test_c1_c5_ph_recovers_ph_for_2d_maps_independent_of_offset():
    ph_map = np.array([[6.5, 6.9, 7.0], [7.2, 7.4, 6.6]])
    c1, c5 = _c1_c5_positions(ph_map, offset_ppm=-123.4)

    np.testing.assert_allclose(OMPD_pH_from_c1_c5(c1, c5), ph_map, atol=1e-9)


def test_c1_c5_ph_hz_matches_ppm_and_keeps_xarray_labels():
    ph_map = np.array([[6.6, 7.1], [7.3, 6.9]])
    c1, c5 = _c1_c5_positions(ph_map, offset_ppm=10.0)
    c1_hz = xr.DataArray(c1 * LAMOR_FREQUENCY_C13, dims=("y", "x"))
    c5_hz = xr.DataArray(c5 * LAMOR_FREQUENCY_C13, dims=("y", "x"))

    result = OMPD_pH_from_c1_c5(
        c1_hz, c5_hz, unit="Hz", larmor_frequency_mhz=LAMOR_FREQUENCY_C13
    )

    assert result.dims == ("y", "x")
    np.testing.assert_allclose(result.values, ph_map, atol=1e-9)


def test_c1_c5_ph_out_of_range_is_nan_without_warning_and_unit_is_validated(recwarn):
    assert np.isnan(OMPD_pH_from_c1_c5(0.0, 2.0))
    assert not recwarn.list
    with pytest.raises(ValueError, match="unit"):
        OMPD_pH_from_c1_c5(0.0, 5.0, unit="khz")


@pytest.mark.parametrize(
    "peak, params", [("C1", PARAMS_OMPD_C1), ("C5", PARAMS_OMPD_C5)]
)
def test_peak_shift_ph_recovers_ph_map_in_ppm_and_hz(peak, params):
    # C1 spans only 0.46 ppm, so stay inside its calibrated range.
    ph_map = xr.DataArray(np.array([[6.6, 7.0], [7.2, 6.8]]), dims=("y", "x"))
    reference = 123.4 + pH_to_ppm(7.4, params)
    positions = 123.4 + pH_to_ppm(ph_map, params)

    ppm = OMPD_pH_from_peak_shift(reference, positions, 7.4, peak=peak)
    hz = OMPD_pH_from_peak_shift(
        reference * LAMOR_FREQUENCY_C13,
        positions * LAMOR_FREQUENCY_C13,
        7.4,
        peak=peak,
        unit="Hz",
        larmor_frequency_mhz=LAMOR_FREQUENCY_C13,
    )

    assert ppm.dims == ("y", "x")
    np.testing.assert_allclose(ppm.values, ph_map.values, atol=1e-9)
    np.testing.assert_allclose(hz.values, ph_map.values, atol=1e-9)


def test_peak_shift_ph_validates_options_and_flags_out_of_range(recwarn):
    assert np.isnan(OMPD_pH_from_peak_shift(0.0, 5.0, 7.0, peak="C1"))
    assert not recwarn.list
    with pytest.raises(ValueError, match="peak"):
        OMPD_pH_from_peak_shift(0.0, 0.1, 7.0, peak="C3")
    with pytest.raises(ValueError, match="unit"):
        OMPD_pH_from_peak_shift(0.0, 0.1, 7.0, unit="khz")


@pytest.mark.parametrize(
    "peak, params", [("C1", PARAMS_OMPD_C1), ("C5", PARAMS_OMPD_C5)]
)
def test_absolute_position_ph_recovers_ph_and_honours_urea_reference(peak, params):
    ph_map = xr.DataArray(np.array([[6.6, 7.0], [7.2, 6.8]]), dims=("y", "x"))
    result = OMPD_pH_from_absolute_position(
        170.0 + pH_to_ppm(ph_map, params), peak=peak, urea_ppm=170.0
    )
    assert result.dims == ("y", "x")
    np.testing.assert_allclose(result.values, ph_map.values, atol=1e-9)
    default_axis = OMPD_pH_from_absolute_position(
        163.5 + pH_to_ppm(7.0, params), peak=peak
    )
    np.testing.assert_allclose(default_axis, 7.0, atol=1e-9)


def test_absolute_position_ph_hz_needs_scan_and_validates_options():
    with pytest.raises(ValueError, match="scan"):
        OMPD_pH_from_absolute_position(100.0, unit="Hz")
    with pytest.raises(ValueError, match="peak"):
        OMPD_pH_from_absolute_position(8.0, peak="C3")
    with pytest.raises(ValueError, match="unit"):
        OMPD_pH_from_absolute_position(8.0, unit="khz")


def test_absolute_position_ph_hz_uses_scan_metadata_line():
    class Scan(dict):
        handler = type("H", (), {"reference_frequency_mhz": LAMOR_FREQUENCY_C13})()

    scan = Scan(center_ppm=170.0)
    ppm = 163.5 + pH_to_ppm(7.1, PARAMS_OMPD_C5)
    hz = (ppm - 170.0) * LAMOR_FREQUENCY_C13
    result = OMPD_pH_from_absolute_position(hz, peak="C5", unit="Hz", scan=scan)
    np.testing.assert_allclose(result, 7.1, atol=1e-9)
