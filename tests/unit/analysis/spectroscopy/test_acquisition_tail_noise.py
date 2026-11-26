"""Acquisition-tail noise estimate and its propagation to reconstructed voxels.

Fixtures are synthetic: a centric CSI acquisition in acquisition order whose
signal decays over the acquired points and stops at a known point, plus white
complex Gaussian noise of known per-quadrature sigma. Tolerances are stated as
multiples of the sampling standard error of the respective estimate.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy.fit_diagnostics import noise_sigma_for_fit
from hypermri.analysis.spectroscopy.noise_estimation import (
    AcquisitionTailNoise,
    estimate_acquisition_tail_noise,
    propagate_acquisition_tail_noise,
)
from hypermri.reconstruction.centred_dft import centered_ifft
from hypermri.sequences._csi_kspace_filter import SampleWindow, window_and_truncate

DWELL_S = 1 / 3000.0
SIGMA = 3.0
SIGNAL_STOP = 250
BANDS_HZ = [(100.0, 200.0), (-400.0, -300.0)]


def _noise(shape, seed):
    rng = np.random.default_rng(seed)
    return SIGMA * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def _centric_acquisition(n_fid=128, n_points=600, amplitude=200.0, seed=0):
    """Return a (FID sample, acquired point) matrix with signal before SIGNAL_STOP."""
    t = np.arange(n_fid)[:, None] * DWELL_S
    lines = np.exp(2j * np.pi * 150.0 * t) + 0.5 * np.exp(-2j * np.pi * 350.0 * t)
    fid = lines * np.exp(-t / 0.02)
    points = np.arange(n_points)
    weight = amplitude * np.exp(-points / 50.0) * (points < SIGNAL_STOP)
    return fid * weight + _noise((n_fid, n_points), seed)


def test_tail_estimate_recovers_sigma_and_an_onset_after_the_signal():
    matrix = _centric_acquisition()
    result = estimate_acquisition_tail_noise(
        matrix, dwell_time_s=DWELL_S, bands_hz=BANDS_HZ
    )

    assert isinstance(result, AcquisitionTailNoise)
    # The signal is below one noise SD per sample from about point 210 on;
    # it ends at SIGNAL_STOP, after which every point is noise only.
    assert 180 <= result.onset_point <= SIGNAL_STOP + 10
    assert result.n_points == 600 - result.onset_point
    assert result.n_fid_samples == 127
    assert result.n_samples == 127 * result.n_points
    # Four sampling standard errors of the Rayleigh estimate.
    tolerance = 4 * SIGMA / np.sqrt(2 * result.n_samples)
    assert result.sigma_quadrature == pytest.approx(SIGMA, abs=tolerance)
    assert result.sigma_standard_error == pytest.approx(
        result.sigma_quadrature / np.sqrt(2 * result.n_samples)
    )
    for cross_check in (result.sigma_real, result.sigma_imag):
        assert cross_check == pytest.approx(SIGMA, abs=2 * tolerance)
    assert abs(result.z_band) < 2
    assert result.bands_hz == ((100.0, 200.0), (-400.0, -300.0))


def test_band_detector_finds_weak_late_signal_that_the_energy_misses():
    # Weak narrow-band signal up to SIGNAL_STOP: invisible in the whole-FID
    # energy, concentrated by the band detector.
    matrix = _centric_acquisition(amplitude=4.0, seed=3)
    with_bands = estimate_acquisition_tail_noise(
        matrix, dwell_time_s=DWELL_S, bands_hz=BANDS_HZ
    )
    whole_width = estimate_acquisition_tail_noise(matrix, dwell_time_s=DWELL_S)

    assert with_bands.onset_point >= whole_width.onset_point
    assert with_bands.onset_point > 50


def test_last_fid_sample_is_excluded():
    matrix = _centric_acquisition(seed=5)
    matrix[-1] *= 1e-3
    result = estimate_acquisition_tail_noise(
        matrix, dwell_time_s=DWELL_S, bands_hz=BANDS_HZ
    )

    tolerance = 4 * SIGMA / np.sqrt(2 * result.n_samples)
    assert result.sigma_quadrature == pytest.approx(SIGMA, abs=tolerance)


def test_tail_estimate_is_deterministic_and_does_not_mutate_input():
    matrix = _centric_acquisition(seed=7)
    before = matrix.copy()
    first = estimate_acquisition_tail_noise(matrix, dwell_time_s=DWELL_S)
    second = estimate_acquisition_tail_noise(matrix, dwell_time_s=DWELL_S)

    assert first == second
    np.testing.assert_array_equal(matrix, before)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dwell_time_s": None}, "dwell_time_s"),
        ({"dwell_time_s": 0.0}, "dwell_time_s"),
        ({"dwell_time_s": DWELL_S, "min_noise_points": 700}, "min_noise_points"),
        ({"dwell_time_s": DWELL_S, "bands_hz": [(5000.0, 6000.0)]}, "no frequency"),
    ],
)
def test_invalid_tail_input_fails_with_actionable_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        estimate_acquisition_tail_noise(_centric_acquisition(), **kwargs)


def test_acquisition_without_signal_free_points_fails():
    matrix = _centric_acquisition(n_points=200)
    with pytest.raises(ValueError, match="consistent with noise"):
        estimate_acquisition_tail_noise(matrix, dwell_time_s=DWELL_S, bands_hz=BANDS_HZ)


def test_real_matrix_is_rejected():
    with pytest.raises(ValueError, match="complex"):
        estimate_acquisition_tail_noise(np.ones((8, 100)), dwell_time_s=DWELL_S)


def _reconstruct(matrix, window, shape, scale):
    """Apply the CSI filter and spatial FFT used by the reconstruction."""
    filtered = window_and_truncate(matrix, axis=1, window=window)
    n_fid = matrix.shape[0]
    order = np.random.default_rng(11).permutation(shape[0] * shape[1])
    grid = np.zeros((n_fid, shape[0] * shape[1]), dtype=complex)
    grid[:, order] = filtered
    grid = grid.reshape(n_fid, *shape)
    return scale * centered_ifft(grid, (1, 2))


@pytest.mark.parametrize("filter_type", ["None", "Hamming"])
def test_propagated_voxel_noise_matches_a_noise_only_reconstruction(filter_type):
    shape = (20, 30)
    n_points = shape[0] * shape[1]
    kspace = _noise((65, n_points), seed=13)
    window = SampleWindow(
        filter_type=filter_type,
        decay_constant=0.01,
        alpha=0.54,
        center_position=None,
        signal_end=400,
        remove_lowest_signal=False,
        apply_flipangle_correction=False,
        flipangle_deg=None,
    )
    scale, mask = 1e-2, 0.7
    voxels = mask * _reconstruct(kspace, window, shape, scale)[:-1]

    acquired = estimate_acquisition_tail_noise(kspace, dwell_time_s=DWELL_S)
    assert acquired.onset_point == 0
    weights = window_and_truncate(
        np.ones((1, n_points), dtype=complex), axis=1, window=window
    )[0].real
    predicted = propagate_acquisition_tail_noise(
        acquired,
        point_weights=weights,
        n_spatial_points=n_points,
        amplitude_scale=scale,
        mask_weight=mask,
    )

    empirical = np.sqrt(np.mean(np.abs(voxels) ** 2) / 2)
    # Both the estimate and the empirical value come from the same 64 x 600
    # complex samples; 4 standard errors of one estimate bound their ratio.
    tolerance = 4 / np.sqrt(2 * voxels.size)
    assert float(predicted) / empirical == pytest.approx(1.0, abs=tolerance)
    assert predicted.attrs["noise_n_kept_points"] == 400
    assert predicted.attrs["noise_source"] == "acquisition_tail"
    assert predicted.attrs["noise_amplitude_scale"] == scale


def test_uniform_weights_reproduce_the_diagnosed_formula():
    acquired = estimate_acquisition_tail_noise(
        _noise((64, 300), seed=17), dwell_time_s=DWELL_S
    )
    weights = np.r_[np.ones(200), np.zeros(100)]
    mask = xr.DataArray([0.5, 1.0], dims="voxel")
    sigma = propagate_acquisition_tail_noise(
        acquired,
        point_weights=weights,
        n_spatial_points=300,
        amplitude_scale=1e-2,
        mask_weight=mask,
    )

    expected_power = mask**2 * 2 * acquired.sigma_quadrature**2 * 200 / 300**2 * 1e-4
    np.testing.assert_allclose(2 * sigma**2, expected_power, rtol=1e-12)
    assert sigma.dims == ("voxel",)


def test_voxel_noise_keeps_provenance_through_the_fit_hand_off_and_netcdf(
    tmp_path,
):
    acquired = estimate_acquisition_tail_noise(
        _centric_acquisition(seed=19), dwell_time_s=DWELL_S, bands_hz=BANDS_HZ
    )
    sigma = propagate_acquisition_tail_noise(
        acquired, point_weights=np.ones(600), n_spatial_points=600
    )
    template = xr.DataArray(
        np.zeros((2, 3)),
        dims=("metabolite", "x"),
        coords={"metabolite": ["a", "b"], "x": [0, 1, 2]},
    )
    for_fit = noise_sigma_for_fit(sigma, template, require_valid=True)

    assert for_fit.name == "noise_sigma_quadrature"
    assert for_fit.attrs["noise_source"] == "acquisition_tail"
    assert for_fit.attrs["noise_onset_point"] == acquired.onset_point

    path = tmp_path / "noise.nc"
    xr.Dataset({"noise_sigma_quadrature": for_fit}).to_netcdf(path)
    with xr.open_dataset(path) as loaded:
        attrs = loaded.noise_sigma_quadrature.attrs
        assert attrs["noise_source"] == "acquisition_tail"
        assert attrs["noise_n_samples"] == acquired.n_samples
        np.testing.assert_allclose(attrs["noise_bands_hz"], [100, 200, -400, -300])


def test_propagation_rejects_invalid_inputs():
    acquired = estimate_acquisition_tail_noise(
        _noise((16, 100), seed=23), dwell_time_s=DWELL_S
    )
    with pytest.raises(TypeError, match="AcquisitionTailNoise"):
        propagate_acquisition_tail_noise(
            3.0, point_weights=np.ones(100), n_spatial_points=100
        )
    with pytest.raises(ValueError, match="point_weights"):
        propagate_acquisition_tail_noise(
            acquired, point_weights=np.array([np.nan]), n_spatial_points=100
        )
    with pytest.raises(ValueError, match="n_spatial_points"):
        propagate_acquisition_tail_noise(
            acquired, point_weights=np.ones(100), n_spatial_points=0
        )
