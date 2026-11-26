"""Characterize the optional neighbour fill of the Voigt Gaussian width.

Fixtures are 3 x 3 voxel planes of synthetic FIDs: 256 samples at 4 kHz,
three lines at -300, 0 and 250 Hz with one Gaussian FWHM of 14 Hz and
complex white noise of sigma 0.02, as in ``test_spectroscopy_voigt_lineshape``. The status of
single voxels is edited after the first fit, so the tests do not depend on a
noise realisation that drives a fit to the 0 Hz bound.
"""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import (
    _median_of_estimated_neighbours,
    def_fit_params,
    fit_data_pseudo_inv,
    refit_gauss_width_from_neighbours,
)

PLANE = ("read_idx", "phase_idx")
FREQS_HZ = np.array([-300.0, 0.0, 250.0])
AMPLITUDES = np.array([10.0, 30.0, 3.0]) * np.exp(1j * np.array([0.3, -0.2, 0.5]))
T2S_S = np.array([0.05, 0.04, 0.06])
GAUSS_FWHM_HZ = 14.0


def plane_fid(seed=0):
    """Return 3 x 3 voxels of noisy three-line FIDs with a 14 Hz Gaussian FWHM."""
    rng = np.random.default_rng(seed)
    t = np.arange(256) / 4000.0
    lines = AMPLITUDES[:, None] * np.exp(
        2j * np.pi * FREQS_HZ[:, None] * t - t / T2S_S[:, None]
    )
    signal = lines.sum(0) * np.exp(
        -((np.pi * GAUSS_FWHM_HZ * t) ** 2) / (4 * np.log(2))
    )
    noise = 0.02 * (
        rng.standard_normal((3, 3, t.size)) + 1j * rng.standard_normal((3, 3, t.size))
    )
    data = xr.DataArray(
        signal + noise,
        dims=[*PLANE, "acquisition_idx"],
        coords={"read_idx": np.arange(3), "phase_idx": np.arange(3)},
    )
    data.attrs["sampling_time_axis_s"] = t
    return data


def voigt_params(data, **options):
    """Prepare fit parameters for the three synthetic lines."""
    return def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a", "b", "c"],
            "metabs_freqs_hz": list(FREQS_HZ + np.array([2.0, -1.5, 1.0])),
            "initial_t2s_s": 0.03,
            "metabs_freq_ranges": 20.0,
            **options,
        },
    )


@pytest.fixture(scope="module")
def plane_fit():
    """Return data, Voigt parameters and results of the 3 x 3 plane."""
    data = plane_fid()
    params = voigt_params(data, lineshape="voigt")
    return data, params, fit_data_pseudo_inv(data, params)


def test_median_uses_only_estimated_neighbours():
    """The centre takes the median of its estimated neighbours; edges have fewer."""
    gauss = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 6.0], [7.0, 100.0, 9.0]]), dims=PLANE
    )
    status = xr.full_like(gauss, "estimated", dtype=object)
    status[1, 1] = "at_lower_bound"
    status[2, 1] = "at_upper_bound"

    median = _median_of_estimated_neighbours(gauss, status, PLANE)

    assert float(median[1, 1]) == pytest.approx(np.median([1, 2, 3, 4, 6, 7, 9]))
    # The corner (0, 0) sees 2 and 4; the centre and the bound voxel do not count.
    assert float(median[0, 0]) == pytest.approx(3.0)


def test_median_is_nan_without_estimated_neighbours():
    gauss = xr.DataArray(np.ones((2, 2)), dims=PLANE)
    status = xr.full_like(gauss, "not_fitted", dtype=object)

    assert np.isnan(_median_of_estimated_neighbours(gauss, status, PLANE)).all()


def test_lower_bound_voxel_is_refitted_with_the_neighbour_median(plane_fit):
    """Only the target voxel changes; its width is fixed at the median."""
    data, params, results = plane_fit
    edited = results.copy(deep=True)
    edited["fit_gauss_fwhm_hz"][1, 1] = 0.0
    edited["fit_gauss_fwhm_status"][1, 1] = "at_lower_bound"
    expected_hz = float(
        np.median(np.delete(results["fit_gauss_fwhm_hz"].values.ravel(), 4))
    )

    filled = refit_gauss_width_from_neighbours(data, params, edited, plane_dims=PLANE)

    assert str(filled["fit_gauss_fwhm_status"][1, 1].values) == "neighbour_median"
    assert float(filled["fit_gauss_fwhm_hz"][1, 1]) == pytest.approx(expected_hz)
    assert str(filled["fit_status"][1, 1].values) == "success"
    # The 14 Hz data refitted with a width near 14 Hz recover the line areas.
    np.testing.assert_allclose(
        np.abs(filled["fit_amps"][1, 1]),
        np.abs(results["fit_amps"][1, 1]),
        rtol=0.01,
    )
    others = np.ones((3, 3), dtype=bool)
    others[1, 1] = False
    for name in ("fit_gauss_fwhm_hz", "fit_amps", "fit_freqs", "fit_t2s"):
        np.testing.assert_array_equal(
            filled[name].values[others], edited[name].values[others], err_msg=name
        )
    assert (filled["fit_gauss_fwhm_status"].values[others] == "estimated").all()
    assert "gauss_neighbour_fill" in filled.attrs
    assert edited["fit_gauss_fwhm_status"][1, 1] == "at_lower_bound"


def test_without_targets_the_results_are_copied(plane_fit):
    data, params, results = plane_fit

    filled = refit_gauss_width_from_neighbours(data, params, results, plane_dims=PLANE)

    xr.testing.assert_identical(
        filled.drop_attrs(deep=False), results.drop_attrs(deep=False)
    )


def test_lorentz_fit_is_rejected():
    data = plane_fid()
    params = voigt_params(data)
    results = fit_data_pseudo_inv(data, params)

    with pytest.raises(ValueError, match="needs a Voigt fit"):
        refit_gauss_width_from_neighbours(data, params, results, plane_dims=PLANE)


def test_missing_plane_dimension_is_named(plane_fit):
    data, params, results = plane_fit

    with pytest.raises(ValueError, match=r"missing \['slice_idx'\]"):
        refit_gauss_width_from_neighbours(
            data, params, results, plane_dims=("read_idx", "slice_idx")
        )
