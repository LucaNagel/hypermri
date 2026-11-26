"""F7 characterization on invented FIDs; no scanner input or defaults changed.

Complete domain/workflow results use rtol 1e-12, atol 1e-14 in signal units.
This guards routing equivalence, not a physiological recovery guarantee.
"""

import importlib

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import spectroscopy
from hypermri.analysis.fitting.models.spectroscopy import fid_model
from hypermri.analysis.spectroscopy.fitting import fit_spectrum


def _invented_input(transpose=False):
    times = np.arange(24, dtype=float) * 0.001
    signal = (1.2 + 0.3j) * np.exp(2j * np.pi * 40 * times) * np.exp(-times / 0.025)
    data = xr.DataArray(
        np.stack([signal, 0.8 * signal]),
        dims=("voxel", "acquisition_idx"),
        coords={
            "voxel": [4, 1],
            "acquisition_idx": np.arange(times.size),
            "sampling_time_axis_s": ("acquisition_idx", times),
        },
        attrs={"signal_domain": "temporal", "unit": "a.u.", "invented_fixture": True},
    )
    if transpose:
        data = data.transpose()
    options = {
        "metabs": ["invented_peak"],
        "metabs_freqs_hz": [40.0],
        "initial_t2s_s": [0.025],
        "metabs_freq_ranges": 0.0,
        "metabs_t2_ranges": 0.0,
        "min_t2_s": 0.025,
        "max_t2_s": 0.025,
        "npoints": 3,
        "niter": 1,
        "rep_fitting": 1,
    }
    return data, options


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_spectrum_domain_and_prepared_workflow_preserve_full_results(
    transpose, parallel
):
    data, options = _invented_input(transpose)
    before = data.copy(deep=True)
    prepared = spectroscopy.def_fit_params(data, options)
    prepared_before = prepared.copy(deep=True)
    expected = spectroscopy.fit_data_pseudo_inv(data, prepared, n_jobs=1)
    actual = fit_spectrum(
        data, prepared, n_jobs=2 if parallel else 1, parallel_backend="threading"
    )
    xr.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert actual.fit_amps.dtype == np.complex128
    assert actual.fit_freqs.dtype == np.float64
    xr.testing.assert_identical(actual.voxel, data.voxel)
    xr.testing.assert_identical(data, before)
    xr.testing.assert_identical(prepared, prepared_before)


@pytest.mark.parametrize(
    "module",
    [
        "hypermri.analysis.fitting.spectroscopic_fitting",
        "hypermri.analysis.fitting.models.spectroscopy.fid_pseudo_inv",
    ],
)
def test_removed_spectroscopy_import_paths_are_absent(module):
    """Workflow names live in ``fitting.spectroscopy``, numerics in ``fid_model``."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_numerical_module_owns_no_workflow_functions():
    assert not hasattr(fid_model, "fit_data_pseudo_inv")
    assert not hasattr(fid_model, "initialise_fit_params_from_results")
    assert fid_model.MODEL_ID == "fid_pseudo_inv"
