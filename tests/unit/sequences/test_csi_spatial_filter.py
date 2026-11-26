"""Physical CSI spatial filter on raw and labelled data, without scanner files or widgets.

Invented paired encodings use delta-k=1/FOV. Double-precision quadratures and
independent Fourier references use 1e-12 absolute tolerance in arbitrary units.
"""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spatial_filtering import SpatialFilterOptions
from hypermri.reconstruction.csi_acquisition import CSIAcquisition
from hypermri.sequences.base_spectroscopy import BaseSpectroscopy
from hypermri.sequences.CSI import CSI


@pytest.fixture
def raw_sequence():
    # Nonuniform paired order with distinct Read/Phase FOV and every plane present.
    read, phase = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    order = np.array([5, 0, 8, 11, 1, 4, 7, 2, 10, 3, 6, 9])
    read, phase = read.ravel()[order], phase.ravel()[order]
    acquisition = CSIAcquisition(
        encoding_order="invented paired order",
        matrix=(3, 4),
        fov_mm=(30.0, 20.0),
        fov_offset_mm=(0.0, 0.0, 0.0),
        read_positions=read,
        phase_positions=phase,
        n_points=2,
        dwell_time_s=None,
        reference_frequency_mhz=None,
        center_ppm=0.0,
    )
    sequence = SimpleNamespace(
        n_acquisition=2,
        n_repetitions=2,
        handler=SimpleNamespace(
            matrix=np.array([3, 4, 2]),
            fov_mm=np.array([30.0, 20.0, 8.0]),
            n_slices=2,
            n_receivers=2,
            method={"Method": "<User:lucaCSI>"},
            to_csi_acquisition=lambda: acquisition,
        ),
    )
    # Rep/slice/point/channel/FID. Unique complex values expose any axis mixing.
    blocks = np.arange(1, 2 * 2 * 12 * 2 * 2 + 1).reshape(2, 2, 12, 2, 2)
    raw = blocks.astype(complex) * (1 + 0.5j)
    sequence.handler.rawdatajob0 = raw.ravel().copy()
    sequence.filter_signal = BaseSpectroscopy.filter_signal.__get__(sequence)
    return sequence, raw, read, phase


@pytest.mark.parametrize("matrix_input", [False, True])
@pytest.mark.parametrize("profile", ["gaussian", "fermi"])
def test_raw_physical_weights_follow_paired_positions_and_keep_every_plane(
    raw_sequence, matrix_input, profile
):
    sequence, raw, read, phase = raw_sequence
    radius = np.hypot((read - 1) / 30.0, (phase - 2) / 20.0)
    options = (
        SpatialFilterOptions(sigma_per_mm=0.1)
        if profile == "gaussian"
        else SpatialFilterOptions(
            kind="fermi", radius_per_mm=0.08, transition_per_mm=0.02
        )
    )
    weights = (
        np.exp(-0.5 * (radius / 0.1) ** 2)
        if profile == "gaussian"
        else 1 / (1 + np.exp((radius - 0.08) / 0.02))
    )
    source = raw.reshape(-1, 2).T if matrix_input else raw.ravel()
    before = source.copy()
    actual = sequence.filter_signal(data=source, spatial_filter=options)
    expected = raw * weights[None, None, :, None, None]
    np.testing.assert_allclose(actual, expected.ravel(), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(sequence.handler.rawdatajob0, raw.ravel())


def test_scanner_spatial_raw_layout_matches_independent_fourier_reference(raw_sequence):
    sequence, raw, read, phase = raw_sequence
    sequence.handler.method["Method"] = "<Bruker:CSI>"
    grid = np.empty((2, 2, 3, 4, 2, 2), dtype=complex)
    grid[:, :, read, phase] = raw
    spectrum = np.fft.fftn(np.fft.ifftshift(grid, axes=(2, 3)), axes=(2, 3))
    k_read = np.array([-1, 0, 1]) / 30.0
    k_phase = np.array([-2, -1, 0, 1]) / 20.0
    weights = np.exp(-0.5 * (k_read[:, None] ** 2 + k_phase[None, :] ** 2) / 0.1**2)
    expected = np.fft.fftshift(
        np.fft.ifftn(spectrum * weights[None, None, :, :, None, None], axes=(2, 3)),
        axes=(2, 3),
    )[:, :, read, phase].ravel()
    actual = sequence.filter_signal(
        spatial_filter=SpatialFilterOptions(sigma_per_mm=0.1)
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_reconstructed_raw_and_create_filtered_data_keep_physical_provenance(
    raw_sequence, monkeypatch
):
    sequence, raw, _, _ = raw_sequence
    template = xr.DataArray(
        np.ones((2, 3, 4)), dims=("acquisition_idx", "read_idx", "phase_idx")
    )
    sequence.data = template
    seen = []

    def reconstruct(_strategy, handler):
        seen.append(handler.rawdatajob0.copy())
        return template.copy(deep=True)

    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.csi.BrukerCsiRecoStrategy.reconstruct",
        reconstruct,
    )
    options = SpatialFilterOptions(sigma_per_mm=0.1)
    params = {
        "spatial_filter": options,
        "filter_type": "Exponential",
        "decay_constant": 0,
    }
    actual = CSI.create_filtered_data(sequence, filter_params=params)
    expected_raw = sequence.filter_signal(data=raw.ravel(), **params)
    np.testing.assert_allclose(seen[0], expected_raw, atol=1e-12, rtol=0)
    record = json.loads(actual.attrs["spatial_filter"])
    assert record["spacing_mm"] == [10, 5]
    assert record["options"]["sigma_per_mm"] == 0.1
    np.testing.assert_array_equal(sequence.handler.rawdatajob0, raw.ravel())
    xr.testing.assert_identical(sequence.data, template)


def test_spatial_filter_calibration_errors_never_fall_back_to_truncation(raw_sequence):
    sequence, _, _, _ = raw_sequence
    sequence.data = xr.DataArray(np.ones(2), dims="acquisition_idx")
    sequence.handler.method["Method"] = "unknown"
    with pytest.raises(ValueError, match="verified"):
        CSI.create_filtered_data(
            sequence, spatial_filter=SpatialFilterOptions(sigma_per_mm=0.1)
        )
    with pytest.raises(ValueError, match="physical widths"):
        CSI.create_filtered_data(
            sequence,
            filter_params={"spatial_filter_type": "Gaussian", "spatial_width": 0.5},
        )
    with pytest.raises(ValueError, match="physical widths"):
        sequence.filter_signal(spatial_filter_type="Fermi", spatial_width=0.5)


def test_raw_missing_fov_requires_explicit_spacing(raw_sequence):
    sequence, raw, _, _ = raw_sequence
    options = SpatialFilterOptions(sigma_per_mm=0.1)
    expected = sequence.filter_signal(spatial_filter=options)
    without_fov = dataclasses.replace(
        sequence.handler.to_csi_acquisition(), fov_mm=None
    )
    sequence.handler.to_csi_acquisition = lambda: without_fov
    with pytest.raises(ValueError, match="fov_mm"):
        sequence.filter_signal(spatial_filter=options)
    np.testing.assert_allclose(
        sequence.filter_signal(spatial_filter=options, spatial_spacing_mm=(10, 5)),
        expected,
        atol=1e-12,
        rtol=0,
    )
    with pytest.raises(ValueError, match="metadata"):
        sequence.filter_signal(
            data=raw.ravel()[:-2],
            spatial_filter=options,
            spatial_spacing_mm=(10, 5),
        )


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (
            {"end_index": 3},
            {
                "signal_end": 3,
                "signal_end_axis": "acquired_points",
                "signal_end_selection": "first",
            },
        ),
        (
            {"filter_params": {"signal_end": 2, "remove_lowest_signal": True}},
            {
                "signal_end": 2,
                "signal_end_axis": "acquired_points",
                "signal_end_selection": "highest_mean_signal",
            },
        ),
        (
            {"filter_axis": 0},
            {
                "signal_end": "none",
                "signal_end_axis": "fid_samples",
                "signal_end_selection": "first",
            },
        ),
    ],
)
def test_create_filtered_data_records_the_truncation(
    raw_sequence, monkeypatch, tmp_path, options, expected
):
    # Fit results inherit these attrs, so a stored fit names its signal_end;
    # plain ints and strings keep the record storable in NetCDF.
    sequence, _, _, _ = raw_sequence
    template = xr.DataArray(
        np.ones((2, 3, 4)), dims=("acquisition_idx", "read_idx", "phase_idx")
    )
    sequence.data = template
    monkeypatch.setattr(
        "hypermri.reconstruction.strategies.bruker.csi.BrukerCsiRecoStrategy.reconstruct",
        lambda _strategy, handler: template.copy(deep=True),
    )

    actual = CSI.create_filtered_data(sequence, **options)

    assert {key: actual.attrs[key] for key in expected} == expected
    assert "filtered_end_index" not in actual.attrs
    path = tmp_path / "filtered.nc"
    actual.to_netcdf(path)
    with xr.open_dataarray(path) as loaded:
        assert {key: loaded.attrs[key] for key in expected} == expected
