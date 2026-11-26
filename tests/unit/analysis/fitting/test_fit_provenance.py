"""Fit results record the caller-side settings that produced them.

The fixture is a synthetic 2 x 2 voxel grid of one damped line at 0 Hz with
known per-voxel frequency offsets. The input carries the truncation and mask
attributes that ``CSI.create_filtered_data`` and ``apply_mask_to`` write, and
it is aligned with :func:`align_spectra_by_fit_frequency` before the fit, so the
results must carry all of them without the caller passing them again.
"""

import pickle

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    fit_best_of_starts,
    fit_data_pseudo_inv,
)
from hypermri.analysis.spectroscopy.alignment import align_spectra_by_fit_frequency

DWELL_S = 1 / 2000.0
N_SAMPLES = 64
INPUT_ATTRS = {
    "signal_domain": "temporal",
    "signal_end": 40,
    "signal_end_axis": "acquired_points",
    "signal_end_selection": "first",
    "applied_mask": "tumour",
}
OFFSETS_HZ = xr.DataArray(
    [[3.0, -2.0], [1.0, 0.0]], dims=("x", "y"), coords={"x": [0, 1], "y": [4, 5]}
)


def _data(order):
    t = np.arange(N_SAMPLES) * DWELL_S
    fid = 10.0 * np.exp(-t / 0.03)
    data = xr.DataArray(
        np.broadcast_to(fid, (2, 2, N_SAMPLES)),
        dims=("x", "y", "acquisition_idx"),
        coords={"x": [0, 1], "y": [4, 5]},
        attrs={**INPUT_ATTRS, "sampling_time_axis_s": t},
    )
    return data.transpose(*order)


def _params(data, freq_hz=0.0):
    return def_fit_params(
        data=data,
        fit_params={
            "metabs": ["a"],
            "metabs_freqs_hz": [freq_hz],
            "rep_fitting": 1,
            "npoints": 5,
        },
    )


def _aligned_fit(order):
    data = _data(order)
    results = fit_data_pseudo_inv(data, _params(data))
    aligned, _, _, _ = align_spectra_by_fit_frequency(
        data, results, frequency_offsets_hz=OFFSETS_HZ
    )
    best = fit_best_of_starts(
        aligned, start_params=[_params(aligned), _params(aligned, freq_hz=4.0)]
    )
    return aligned, best


@pytest.fixture(scope="module")
def aligned_fit():
    return _aligned_fit(("x", "y", "acquisition_idx"))


def test_results_carry_truncation_mask_alignment_starts_and_lineshape(aligned_fit):
    _, best = aligned_fit
    for key, value in INPUT_ATTRS.items():
        assert best.attrs[key] == value
    assert best.attrs["lineshape"] == "lorentz"
    xr.testing.assert_allclose(
        best.frequency_alignment_offset_hz.reset_coords(drop=True), OFFSETS_HZ
    )
    assert best.fit_start_initial_freqs_hz.attrs["unit"] == "Hz"
    np.testing.assert_allclose(
        best.fit_start_initial_freqs_hz.sel(metabolite="a").transpose(
            "start_candidate", ...
        ),
        np.array([0.0, 4.0])[:, None, None] * np.ones((2, 2, 2)),
    )
    assert best.fit_start_initial_t2s_s.attrs["unit"] == "s"
    assert "fit_start_initial_gauss_fwhm_hz" not in best


def test_provenance_survives_snapshot_and_netcdf(aligned_fit, tmp_path):
    _, best = aligned_fit
    xr.testing.assert_identical(pickle.loads(pickle.dumps(best)), best)

    provenance = xr.Dataset(
        {
            name: best[name]
            for name in ("fit_start_initial_freqs_hz", "fit_start_initial_t2s_s")
        },
        attrs={key: best.attrs[key] for key in (*INPUT_ATTRS, "lineshape")},
    )
    path = tmp_path / "provenance.nc"
    provenance.to_netcdf(path)
    with xr.open_dataset(path) as loaded:
        xr.testing.assert_identical(loaded.load(), provenance)


def test_dimension_order_does_not_change_the_provenance(aligned_fit):
    _, reference = aligned_fit
    _, permuted = _aligned_fit(("y", "acquisition_idx", "x"))
    for name in (
        "frequency_alignment_offset_hz",
        "fit_start_initial_freqs_hz",
        "fit_start_initial_t2s_s",
    ):
        xr.testing.assert_allclose(
            permuted[name].transpose(*reference[name].dims).reset_coords(drop=True),
            reference[name].reset_coords(drop=True),
        )
    assert permuted.attrs["signal_end"] == reference.attrs["signal_end"]
