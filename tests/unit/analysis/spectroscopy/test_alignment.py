"""Contracts for phase-ramp frequency shifting on NumPy and xarray input.

All fixtures are synthetic single-exponential FIDs. Offsets are whole multiples
of the spectral bin width (1 / (n * dwell)), so a shift moves a peak exactly by
whole bins. Floating-point comparisons use rtol=1e-10, well above FFT rounding
for 64 complex points.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy import shift_frequency
from hypermri.analysis.spectroscopy.alignment import (
    _shift_data,
    align_spectra_by_fit_frequency,
)
from hypermri.utils.mri.fourier import spectral_fft

NPOINTS = 64
DWELL_S = 1e-3
BIN_HZ = 1.0 / (NPOINTS * DWELL_S)


def _fid(frequency_hz: float) -> np.ndarray:
    time_s = np.arange(NPOINTS) * DWELL_S
    return np.exp(2j * np.pi * frequency_hz * time_s - time_s / 0.02)


def _peak_hz(spectrum: np.ndarray) -> float:
    freq_axis_hz = (np.arange(NPOINTS) - NPOINTS // 2) * BIN_HZ
    return float(freq_axis_hz[np.argmax(np.abs(spectrum))])


def test_scalar_offset_moves_peak_by_offset():
    fid = _fid(2 * BIN_HZ)
    original = fid.copy()

    shifted = shift_frequency(fid, 5 * BIN_HZ, dwell_time_s=DWELL_S)

    assert _peak_hz(spectral_fft(shifted)) == pytest.approx(7 * BIN_HZ)
    np.testing.assert_array_equal(fid, original)


def test_spectral_and_temporal_input_give_the_same_shift():
    fid = _fid(3 * BIN_HZ)

    from_fid = shift_frequency(fid, -4 * BIN_HZ, dwell_time_s=DWELL_S)
    from_spectrum = shift_frequency(
        spectral_fft(fid), -4 * BIN_HZ, dwell_time_s=DWELL_S, signal_domain="Spectral"
    )

    np.testing.assert_allclose(from_spectrum, spectral_fft(from_fid), rtol=1e-10)


def test_per_spectrum_offsets_and_axis_choice():
    fids = np.stack([_fid(0.0), _fid(0.0), _fid(0.0)])
    offsets_hz = np.array([1.0, -2.0, 3.0]) * BIN_HZ

    shifted = shift_frequency(fids, offsets_hz, dwell_time_s=DWELL_S)
    shifted_axis0 = shift_frequency(fids.T, offsets_hz, dwell_time_s=DWELL_S, axis=0)

    peaks = [_peak_hz(spectral_fft(row)) for row in shifted]
    np.testing.assert_allclose(peaks, offsets_hz)
    np.testing.assert_allclose(shifted_axis0, shifted.T, rtol=1e-10)


def test_sampling_time_axis_matches_dwell_time():
    fid = _fid(BIN_HZ)
    time_axis_s = np.arange(NPOINTS) * DWELL_S

    np.testing.assert_allclose(
        shift_frequency(fid, 2 * BIN_HZ, sampling_time_axis_s=time_axis_s),
        shift_frequency(fid, 2 * BIN_HZ, dwell_time_s=DWELL_S),
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dwell_time_s": DWELL_S, "signal_domain": "ppm"}, "signal_domain"),
        ({}, "sampling time axis"),
        ({"dwell_time_s": -1.0}, "dwell_time_s"),
    ],
)
def test_invalid_options_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        shift_frequency(_fid(0.0), 1.0, **kwargs)


def test_offsets_that_do_not_broadcast_raise():
    fids = np.stack([_fid(0.0), _fid(0.0)])
    with pytest.raises(ValueError, match="does not broadcast"):
        shift_frequency(fids, np.zeros(3), dwell_time_s=DWELL_S)


@pytest.mark.parametrize("signal_domain", ["temporal", "spectral"])
def test_xarray_wrapper_matches_numpy_kernel(signal_domain):
    fids = np.stack([_fid(0.0), _fid(BIN_HZ)])
    values = spectral_fft(fids) if signal_domain == "spectral" else fids
    data = xr.DataArray(
        values.T,
        dims=("acquisition_idx", "read_idx"),
        coords={"read_idx": [10, 20]},
        attrs={"dwell_time_s": DWELL_S, "signal_domain": signal_domain},
    )
    offsets_hz = xr.DataArray(
        [2 * BIN_HZ, -BIN_HZ], dims=("read_idx",), coords={"read_idx": [10, 20]}
    )

    shifted = _shift_data(
        data,
        offsets_hz,
        global_mean_hz=0.0,
        sampling_time_axis_s=None,
        dwell_time_s=None,
    )

    expected = shift_frequency(
        values,
        offsets_hz.values,
        dwell_time_s=DWELL_S,
        signal_domain=signal_domain,
    )
    assert shifted.dims == data.dims
    np.testing.assert_array_equal(shifted["read_idx"], [10, 20])
    np.testing.assert_allclose(shifted.values, expected.T, rtol=1e-10)
    assert shifted.attrs["signal_domain"] == signal_domain


def test_shift_keeps_the_attached_voxel_noise():
    # A frequency shift is a unit-modulus phase ramp, so the per-voxel noise
    # sigma is unchanged and must stay attached for later fits.
    data = xr.DataArray(
        np.stack([_fid(0.0), _fid(BIN_HZ)]).T,
        dims=("acquisition_idx", "read_idx"),
        coords={"read_idx": [10, 20]},
        attrs={"dwell_time_s": DWELL_S, "signal_domain": "temporal"},
    )
    sigma = xr.DataArray(
        [0.5, 2.0], dims="read_idx", attrs={"noise_source": "acquisition_tail"}
    )
    data = data.assign_coords(noise_sigma_quadrature=sigma)
    shifted = _shift_data(
        data,
        xr.DataArray([BIN_HZ, -BIN_HZ], dims="read_idx", coords={"read_idx": [10, 20]}),
        global_mean_hz=0.0,
        sampling_time_axis_s=None,
        dwell_time_s=None,
    )
    xr.testing.assert_identical(
        shifted.noise_sigma_quadrature.drop_vars("frequency_alignment_offset_hz"),
        data.noise_sigma_quadrature,
    )


def _voxel_data_and_results(order):
    """Return 2 x 3 voxels of FIDs in ``order`` and their one-line fit results."""
    coords = {"x": [0, 1], "y": [5, 6, 7]}
    fids = np.broadcast_to(_fid(0.0), (2, 3, NPOINTS))
    data = xr.DataArray(
        fids,
        dims=("x", "y", "acquisition_idx"),
        coords=coords,
        attrs={"dwell_time_s": DWELL_S, "signal_domain": "temporal"},
    ).transpose(*order)
    results = xr.Dataset(
        {
            "fit_freqs": (("x", "y", "metabolite"), np.zeros((2, 3, 1))),
            "fit_amps": (("x", "y", "metabolite"), np.ones((2, 3, 1))),
        },
        coords={**coords, "metabolite": ["a"]},
    )
    return data, results


OFFSETS_HZ = xr.DataArray(
    np.arange(6.0).reshape(2, 3) * BIN_HZ,
    dims=("x", "y"),
    coords={"x": [0, 1], "y": [5, 6, 7]},
)


@pytest.mark.parametrize(
    "order", [("x", "y", "acquisition_idx"), ("y", "acquisition_idx", "x")]
)
def test_alignment_records_the_total_applied_offset(order):
    # Fits of aligned data inherit the coordinate, so stored results say how
    # far each voxel was shifted; repeated calls add up, undoing returns zero.
    data, results = _voxel_data_and_results(order)
    once, once_results, _, _ = align_spectra_by_fit_frequency(
        data, results, frequency_offsets_hz=OFFSETS_HZ
    )
    twice, twice_results, _, _ = align_spectra_by_fit_frequency(
        once, once_results, frequency_offsets_hz=2 * OFFSETS_HZ
    )

    for aligned in (twice, twice_results):
        applied = aligned.frequency_alignment_offset_hz
        assert applied.attrs["unit"] == "Hz"
        assert applied.dtype == np.float64
        xr.testing.assert_allclose(
            applied.transpose("x", "y").reset_coords(drop=True),
            3 * OFFSETS_HZ,
        )
    assert twice.frequency_alignment_offset_hz.dims == tuple(
        dim for dim in order if dim != "acquisition_idx"
    )
    undone, undone_results, _, _ = align_spectra_by_fit_frequency(
        twice, twice_results, frequency_offsets_hz=-3 * OFFSETS_HZ
    )
    np.testing.assert_allclose(undone.frequency_alignment_offset_hz, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        undone_results.frequency_alignment_offset_hz, 0.0, atol=1e-12
    )
