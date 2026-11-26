"""Focused checks for explicit spectroscopy frequencies and missing calibration."""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import def_fit_params


@pytest.fixture
def fid_data() -> xr.DataArray:
    """Provide a small FID with a sampling axis and no frequency calibration."""
    return xr.DataArray(
        np.ones(16, dtype=np.complex128),
        dims="acquisition_idx",
        coords={"acquisition_idx": np.arange(16) * 0.0004},
        attrs={"signal_domain": "temporal"},
    )


@pytest.mark.parametrize("key", ["metabs_freqs_hz", "metabs_freqs_Hz"])
def test_explicit_hz_accepts_arrays_and_scalar_zero_without_ppm_axis(fid_data, key):
    """Honor Hz arrays and zero-valued scalar frequencies through either spelling."""
    params = def_fit_params(
        fid_data,
        {"metabs": ["pyruvate", "lactate"], key: np.array([-70.0, 857.0])},
    )
    np.testing.assert_array_equal(params["initial_freqs_hz"], [-70.0, 857.0])
    assert params.attrs["metabs_freqs_ppm"] == [None, None]
    assert params.attrs["freq_axis_ppm"] is None
    assert "metabs_freqs_Hz" not in params.attrs

    zero = def_fit_params(fid_data, {"metabs": "pyruvate", key: 0.0})
    assert zero["initial_freqs_hz"].item() == 0.0


def test_hz_aliases_must_agree_and_do_not_mutate_input_options(fid_data):
    """Accept matching aliases and reject conflicting values before fitting."""
    options = {
        "metabs": "pyruvate",
        "metabs_freqs_hz": [0.0],
        "metabs_freqs_Hz": 0.0,
    }
    result = def_fit_params(fid_data, options)
    assert result["initial_freqs_hz"].item() == 0.0
    assert options["metabs_freqs_Hz"] == 0.0

    with pytest.raises(ValueError, match="metabs_freqs_hz.*metabs_freqs_Hz"):
        def_fit_params(fid_data, {**options, "metabs_freqs_Hz": 10.0})


def test_ppm_precedence_and_partial_frequency_defaults_are_preserved(fid_data):
    """Use explicit ppm before Hz, then supplied Hz, then known chemical shifts."""
    calibrated = fid_data.assign_coords(
        freq_axis_hz=("acquisition_idx", np.linspace(-1000, 1000, 16)),
        freq_axis_ppm=("acquisition_idx", np.linspace(160, 190, 16)),
    )
    params = def_fit_params(
        calibrated,
        {
            "metabs": ["pyruvate", "lactate", "alanine"],
            "metabs_freqs_ppm": [172.0],
            "metabs_freqs_hz": [123.0, 0.0],
        },
    )
    np.testing.assert_allclose(params["initial_freqs_hz"], [-200.0, 0.0, 100.0])


@pytest.mark.parametrize("options", [{}, {"metabs_freqs_ppm": [171.076]}])
def test_ppm_conversion_without_an_axis_gives_actionable_error(fid_data, options):
    """Report missing calibration before an unusable frequency reaches the solver."""
    with pytest.raises(ValueError, match="freq_axis_ppm.*metabs_freqs_hz"):
        def_fit_params(fid_data, {"metabs": "pyruvate", **options})


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [[10.0]], [10.0, 20.0]])
def test_invalid_hz_values_are_rejected_before_fitting(fid_data, values):
    """Reject nonfinite values and shapes that do not match the metabolite list."""
    with pytest.raises(ValueError, match="metabs_freqs_hz"):
        def_fit_params(fid_data, {"metabs": "pyruvate", "metabs_freqs_hz": values})
