"""Characterize ``filter_signal`` for every filter type, input layout and domain.

The expected values come from the explicit reference below, written from the
documented contract: an optional flip-angle correction, a window along the
filter axis, then either zeroing the tail after ``signal_end`` or zeroing the
points with the lowest mean magnitude. Raw vectors with repetitions apply the
truncation separately in each repetition block. Labelled CSI k-space is
filtered in acquired order, and spatial-domain input is first moved to
k-space with the exact inverse of the reconstruction FFT.

Fixtures are small seeded synthetic arrays without scanner metadata. The
reference reached the shipped implementation within ``rtol=1e-12`` before
``filter_signal`` was split into named steps; the tolerance only allows for
floating-point summation order in the lowest-signal metric.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.reconstruction.centred_dft import centered_fft, centered_ifft
from hypermri.reconstruction.csi_acquisition import CSIAcquisition
from hypermri.sequences.base_spectroscopy import BaseSpectroscopy

RTOL = 1e-12
DECAY_PER_SAMPLE = 0.3
HAMMING_ALPHA = 0.54
N_FID_SAMPLES = 4
N_PHASE, N_READ = 3, 2
# Acquired order of the (read, phase) encoding positions; deliberately not
# the storage order, so a wrong permutation changes which points are kept.
READ_POSITIONS = np.array([1, 0, 1, 0, 1, 0])
PHASE_POSITIONS = np.array([2, 2, 0, 1, 1, 0])
FILTER_TYPES = ["None", "Exponential", "Hamming", "Hann"]
TRUNCATIONS = {
    "none": {},
    "tail": {"signal_end": 2},
    "lowest": {"signal_end": 2, "remove_lowest_signal": True},
}


def _complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _window(n, filter_type):
    k = np.arange(n)
    if filter_type == "Exponential":
        return np.exp(-DECAY_PER_SAMPLE * k)
    phase = 2 * np.pi * np.clip(k, 0, n - 1) / max(n - 1, 1)
    if filter_type == "Hamming":
        return HAMMING_ALPHA - (1 - HAMMING_ALPHA) * np.cos(phase)
    if filter_type == "Hann":
        return 0.5 * (1 - np.cos(phase))
    return np.ones(n)


def _reference(values, axis, filter_type, truncation, n_blocks=1, flip_deg=None):
    """Filter ``values`` along ``axis`` following the documented contract."""
    out = np.moveaxis(np.array(values, dtype=complex), axis, -1)
    n = out.shape[-1]
    if flip_deg is not None:
        out = out / np.cos(np.deg2rad(flip_deg)) ** np.arange(n)
    out = out * _window(n, filter_type)
    keep = truncation.get("signal_end")
    if keep is not None and keep < n:
        size = n // n_blocks
        for block in range(n_blocks):
            start = block * size
            if truncation.get("remove_lowest_signal"):
                metric = np.abs(out[..., start : start + size])
                metric = metric.reshape(-1, size).mean(axis=0)
                out[..., np.argsort(metric)[: size - keep] + start] = 0
            else:
                out[..., start + keep : start + size] = 0
    return np.moveaxis(out, -1, axis)


def _sequence(n_repetitions=1, handler=None):
    acquisition = CSIAcquisition(
        encoding_order="synthetic",
        matrix=(N_READ, N_PHASE),
        fov_mm=(20.0, 30.0),
        fov_offset_mm=(0.0, 0.0, 0.0),
        read_positions=READ_POSITIONS,
        phase_positions=PHASE_POSITIONS,
        n_points=N_FID_SAMPLES,
        dwell_time_s=None,
        reference_frequency_mhz=None,
        center_ppm=0.0,
    )
    return SimpleNamespace(
        n_acquisition=N_FID_SAMPLES,
        n_repetitions=n_repetitions,
        flipangle_deg=12.0,
        handler=handler or SimpleNamespace(to_csi_acquisition=lambda: acquisition),
    )


def _labelled_kspace(rng):
    """Return CSI k-space whose measured centre equals the nominal centre."""
    values = _complex(rng, (N_FID_SAMPLES, N_PHASE, N_READ, 2))
    values[:, N_PHASE // 2, N_READ // 2, :] += 20.0
    return xr.DataArray(
        values,
        dims=("acquisition_idx", "phase_idx", "read_idx", "repetition_idx"),
        coords={"acquisition_idx": np.arange(N_FID_SAMPLES) * 0.5},
        attrs={"units": "a.u."},
    ).transpose("repetition_idx", "read_idx", "acquisition_idx", "phase_idx")


def _filter_kspace_in_acquired_order(kspace, filter_type, truncation):
    working = kspace.transpose("acquisition_idx", "phase_idx", "read_idx", ...)
    flat = working.values.reshape(N_FID_SAMPLES, N_PHASE * N_READ, -1)
    order = PHASE_POSITIONS * N_READ + READ_POSITIONS
    filtered = np.empty_like(flat)
    filtered[:, order] = _reference(flat[:, order], 1, filter_type, truncation)
    return working.copy(data=filtered.reshape(working.shape)).transpose(*kspace.dims)


def _case(kind, rng, filter_type, truncation):
    """Return the call keywords, sequence and expected result of one case."""
    if kind == "raw_vector_repetitions":
        matrix = _complex(rng, (N_FID_SAMPLES, 6))
        expected = _reference(matrix, 1, filter_type, truncation, n_blocks=2)
        return (
            {"data": matrix.T.flatten()},
            _sequence(n_repetitions=2),
            expected.T.flatten(),
        )
    if kind == "raw_matrix_fid_axis":
        matrix = _complex(rng, (N_FID_SAMPLES, 6))
        expected = _reference(matrix, 0, filter_type, truncation)
        return (
            {"data": matrix, "filter_axis": 0},
            _sequence(n_repetitions=2),
            expected.T.flatten(),
        )
    if kind == "numpy_nd":
        values = _complex(rng, (N_FID_SAMPLES, 3, 5))
        expected = _reference(values, 2, filter_type, truncation)
        return {"data": values, "filter_axis": 2}, _sequence(), expected
    kspace = _labelled_kspace(rng)
    if kind == "labelled_kspace":
        expected = _filter_kspace_in_acquired_order(kspace, filter_type, truncation)
        return {"data": kspace}, _sequence(), expected
    if kind == "labelled_spatial":
        axes = [kspace.get_axis_num(d) for d in ("read_idx", "phase_idx")]
        spatial = kspace.copy(data=centered_ifft(kspace.values.copy(), axes))
        spatial.attrs["domain"] = "spatial"
        filtered = _filter_kspace_in_acquired_order(kspace, filter_type, truncation)
        expected = spatial.copy(data=centered_ifft(filtered.values.copy(), axes))
        return {"data": spatial}, _sequence(), expected
    if kind == "labelled_acquisition_dim":
        axis = kspace.get_axis_num("acquisition_idx")
        expected = kspace.copy(
            data=_reference(kspace.values, axis, filter_type, truncation)
        )
        return (
            {"data": kspace, "filter_dim": "acquisition_idx"},
            _sequence(),
            expected,
        )
    raise AssertionError(kind)


KINDS = [
    "raw_vector_repetitions",
    "raw_matrix_fid_axis",
    "numpy_nd",
    "labelled_kspace",
    "labelled_spatial",
    "labelled_acquisition_dim",
]


@pytest.mark.parametrize("truncation", TRUNCATIONS, ids=str)
@pytest.mark.parametrize("filter_type", FILTER_TYPES)
@pytest.mark.parametrize("kind", KINDS)
def test_filter_signal_matches_reference(kind, filter_type, truncation):
    rng = np.random.default_rng(20261112)
    kwargs, sequence, expected = _case(kind, rng, filter_type, TRUNCATIONS[truncation])
    source = kwargs["data"]
    source_copy = source.copy(deep=True) if hasattr(source, "dims") else source.copy()

    result = BaseSpectroscopy.filter_signal(
        sequence,
        filter_type=filter_type,
        decay_constant=DECAY_PER_SAMPLE,
        alpha=HAMMING_ALPHA,
        **TRUNCATIONS[truncation],
        **kwargs,
    )

    if isinstance(expected, xr.DataArray):
        assert result.dims == expected.dims
        for name, coord in source.coords.items():
            np.testing.assert_array_equal(result.coords[name], coord)
        np.testing.assert_allclose(result.values, expected.values, rtol=RTOL, atol=0)
        expected_attrs = dict(source.attrs)
        if kind != "labelled_acquisition_dim":
            expected_attrs["filter_fid_order_mode"] = "encoding_order"
            expected_attrs["filter_kspace_order_mode"] = "encoding_order"
        assert result.attrs == expected_attrs
        assert source.identical(source_copy)
    else:
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=0)
        np.testing.assert_array_equal(source, source_copy)


@pytest.mark.parametrize("kind", ["raw_vector_repetitions", "labelled_kspace"])
@pytest.mark.parametrize("flipangle_deg", [None, 20.0])
def test_flipangle_correction_precedes_window_and_truncation(kind, flipangle_deg):
    """None uses the sequence's 12 degrees; an explicit angle takes precedence."""
    rng = np.random.default_rng(7)
    truncation = TRUNCATIONS["tail"]
    kwargs, sequence, _ = _case(kind, rng, "Hann", truncation)
    effective = 12.0 if flipangle_deg is None else flipangle_deg

    result = BaseSpectroscopy.filter_signal(
        sequence,
        filter_type="Hann",
        apply_flipangle_correction=True,
        flipangle_deg=flipangle_deg,
        **truncation,
        **kwargs,
    )

    if kind == "raw_vector_repetitions":
        matrix = kwargs["data"].reshape(6, N_FID_SAMPLES).T
        expected = _reference(
            matrix, 1, "Hann", truncation, n_blocks=2, flip_deg=effective
        ).T.flatten()
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=0)
    else:
        kspace = kwargs["data"]
        working = kspace.transpose("acquisition_idx", "phase_idx", "read_idx", ...)
        flat = working.values.reshape(N_FID_SAMPLES, N_PHASE * N_READ, -1)
        order = PHASE_POSITIONS * N_READ + READ_POSITIONS
        filtered = np.empty_like(flat)
        filtered[:, order] = _reference(
            flat[:, order], 1, "Hann", truncation, flip_deg=effective
        )
        expected = working.copy(data=filtered.reshape(working.shape))
        np.testing.assert_allclose(
            result.transpose(*working.dims).values,
            expected.values,
            rtol=RTOL,
            atol=0,
        )


@pytest.mark.parametrize(
    ("mode", "handler", "expected_mode"),
    [
        ("native", None, "native_order"),
        ("auto", "failing", "native_order_fallback"),
    ],
)
def test_native_and_fallback_orders_filter_in_storage_order(
    mode, handler, expected_mode
):
    rng = np.random.default_rng(3)
    kspace = _labelled_kspace(rng)
    failing = SimpleNamespace(
        to_csi_acquisition=lambda: (_ for _ in ()).throw(RuntimeError("missing"))
    )
    sequence = _sequence(handler=failing if handler == "failing" else None)

    result = BaseSpectroscopy.filter_signal(
        sequence,
        data=kspace,
        filter_type="Exponential",
        decay_constant=DECAY_PER_SAMPLE,
        signal_end=2,
        spatial_order_mode=mode,
    )

    working = kspace.transpose("acquisition_idx", "phase_idx", "read_idx", ...)
    flat = working.values.reshape(N_FID_SAMPLES, N_PHASE * N_READ, -1)
    expected = _reference(flat, 1, "Exponential", TRUNCATIONS["tail"])
    np.testing.assert_allclose(
        result.transpose(*working.dims).values.reshape(flat.shape),
        expected,
        rtol=RTOL,
        atol=0,
    )
    assert result.attrs["filter_kspace_order_mode"] == expected_mode
