"""Independent analytic controls for consolidated helper families."""

from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.spectroscopy.signals import propagate_isochromats
from hypermri.reconstruction.axes import (
    _label_cpmg_echo_train,
)
from hypermri.utils.mathematics.calculations import (
    calc_sampling_time_axis as legacy_axis,
)
from hypermri.utils.mri.fourier import spectral_fft, spectral_ifft
from hypermri.vendors._jcamp import (
    _parse_paravision_array,
    _parse_scalar,
    _parse_topspin_array,
    _parse_topspin_scalar,
)


@pytest.mark.parametrize("size", [7, 8])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_fourier_impulse_and_inverse_preserve_scale_and_order(size, axis):
    """A temporal impulse gives a unit spectrum; roundtrip tolerance 1e-14."""
    values = np.zeros((size, size), complex)
    selection = [slice(None), slice(None)]
    selection[axis % 2] = 0
    values[tuple(selection)] = 1
    before = values.copy()
    spectrum = spectral_fft(values, axis)
    np.testing.assert_allclose(spectrum, 1, atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        spectral_ifft(spectrum, axis), values, atol=1e-14, rtol=0
    )
    np.testing.assert_array_equal(values, before)


def test_time_axis_legacy_defaults_and_metadata_remain_distinct():
    np.testing.assert_array_equal(legacy_axis(), [])
    np.testing.assert_allclose(
        legacy_axis(npoints=4), [0, 0.001, 0.002, 0.003], atol=1e-15, rtol=0
    )
    obj = SimpleNamespace(
        n_acquisition=6, cut_off=2, sampling_time_axis_s=np.arange(6) + 9
    )
    np.testing.assert_allclose(
        legacy_axis(scan=obj, bw_hz=100), [0, 0.01, 0.02, 0.03], atol=1e-15, rtol=0
    )


@pytest.mark.parametrize("count, error", [(-1, ValueError), (3.5, TypeError)])
def test_legacy_time_axis_retains_invalid_sample_count_rejection(count, error):
    with pytest.raises(error):
        legacy_axis(npoints=count)


@pytest.mark.parametrize(
    "options",
    [
        {"dwell_time_s": 0.001, "bw_hz": 1000},
        {"dwell_time_s": 0},
        {"dwell_time_s": -1},
        {"bw_hz": 0},
        {"bw_hz": np.inf},
    ],
)
def test_legacy_time_axis_delegates_strict_explicit_timing_validation(options):
    with pytest.raises(ValueError):
        legacy_axis(npoints=3, **options)


def test_jcamp_dialect_shapes_and_record_boundary_are_retained():
    pv = StringIO("1 2\n3 4\n##$Next=7\n")
    top = StringIO("1 2\n3 4\n##$Next=7\n")
    np.testing.assert_array_equal(
        _parse_paravision_array(pv, "( 2, 2 )\n"), [[1, 2], [3, 4]]
    )
    np.testing.assert_array_equal(_parse_topspin_array(top, "(0..3)\n"), [1, 2, 3, 4])
    assert pv.readline() == top.readline() == "##$Next=7\n"
    short = StringIO("1\n##$Next=7\n")
    np.testing.assert_array_equal(_parse_topspin_array(short, "(0..3)\n"), [1])
    assert short.readline() == "##$Next=7\n"


@pytest.mark.parametrize(
    ("text", "paravision", "topspin"),
    [
        ("42", 42, 42),
        ("-3", -3, -3),
        ("1.5e3", 1500.0, 1500.0),
        (" 7 ", 7, 7),
        ("abc\n", "abc", "abc"),
        (" text \n", " text ", "text"),
        ("<Bruker>", "<Bruker>", "Bruker"),
        ("<12>", "<12>", 12),
        ("<>", "<>", ""),
    ],
)
def test_jcamp_scalar_dialects_differ_only_in_delimiter_stripping(
    text, paravision, topspin
):
    """ParaVision keeps angle brackets and outer spaces; TopSpin strips both.

    Invented scalar tokens; values and Python types are compared exactly.
    """
    for parsed, expected in (
        (_parse_scalar(text), paravision),
        (_parse_topspin_scalar(text), topspin),
    ):
        assert parsed == expected
        assert type(parsed) is type(expected)


def test_shared_evolution_broadcasts_times_without_mutation():
    times = np.array([0.0, 0.001])[:, None]
    vectors, phases = propagate_isochromats([2.0, 3.0], [0.0, 250.0], times)
    np.testing.assert_allclose(vectors, [[2, 3], [2, 3j]], atol=1e-14, rtol=0)
    np.testing.assert_allclose(phases, [[0, 0], [0, np.pi / 2]], atol=1e-14, rtol=0)


def test_cpmg_raw_and_scanner_echo_labels_share_a_contract():
    """Invented three-echo spectral samples survive metadata-only normalization."""
    handler = SimpleNamespace(type="CPMG", method={"EffectiveTE": [10.0, 20.0, 30.0]})
    values = np.arange(12).reshape(4, 3)
    raw = xr.DataArray(
        values.reshape(4, 1, 1, 3, 1, 1),
        dims=(
            "acquisition_idx",
            "repetition_idx",
            "slice_idx",
            "phase_idx",
            "channel_idx",
            "read_idx",
        ),
    )
    scanner = xr.DataArray(values, dims=("acquisition_idx", "echo_idx"))
    r = _label_cpmg_echo_train(raw, handler)
    s = _label_cpmg_echo_train(scanner, handler)
    assert (
        r.dims
        == s.dims
        == ("acquisition_idx", "echo_idx", "channel_idx", "repetition_idx")
    )
    np.testing.assert_array_equal(r.values.squeeze(), values)
    np.testing.assert_allclose(r.echo_idx, [0.01, 0.02, 0.03], atol=1e-15, rtol=0)
    xr.testing.assert_identical(r.echo_idx, s.echo_idx)


def test_phase_rotation_and_zero_fill_preserve_samples_and_precision():
    from hypermri.utils.mri.signal import rotate_phase, zero_fill_temporal

    values = np.array([[1 + 2j, 3 - 4j]], dtype=np.complex64)
    before = values.copy()
    rotated = rotate_phase(values, np.pi / 2)
    np.testing.assert_allclose(rotated, [[-2 + 1j, 4 + 3j]], atol=1e-6, rtol=0)
    padded = zero_fill_temporal(values, 2)
    assert padded.dtype == values.dtype
    np.testing.assert_array_equal(padded, [[1 + 2j, 3 - 4j, 0, 0]])
    np.testing.assert_array_equal(values, before)


def test_ideal_test_signals_keep_distinct_echo_time_conventions():
    from hypermri.analysis.fitting.ideal import IDEAL_Solver
    from hypermri.analysis.fitting.phase_ideal import create_test_signal

    solver = IDEAL_Solver.__new__(IDEAL_Solver)
    solver.num_Echoes = 3
    solver.echo_spacing_s = 0.001
    solver.relative_frequencies = np.array([0.0, 250.0])
    amplitudes = np.array([2.0, 3.0])
    np.testing.assert_allclose(
        solver.create_test_signal(amplitudes, 3),
        [2 - 3j, 5, 2 + 3j],
        atol=1e-14,
        rtol=0,
    )
    explicit = create_test_signal(
        amplitudes, solver.relative_frequencies, np.array([0.0, 0.001, 0.002])
    )
    assert explicit.shape == (3, 1, 1, 1)
    np.testing.assert_allclose(explicit.squeeze(), [5, 2 + 3j, -1], atol=1e-14, rtol=0)
