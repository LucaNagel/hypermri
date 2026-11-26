from types import SimpleNamespace

import numpy as np
import pytest

from hypermri.utils.time.temporal import (
    calc_inversion_time_multi_slice,
    calc_per_image_timeaxis_s,
    calc_sampling_time_axis,
    get_dwell_time_s,
)


def test_get_dwell_time_uses_bandwidth_or_data_object_metadata():
    assert get_dwell_time_s(bw_hz=2_000) == pytest.approx(0.0005)
    assert get_dwell_time_s(sampling_time_axis_s=np.array([0.1, 0.4])) == pytest.approx(
        0.3
    )
    assert get_dwell_time_s(scan={"dwell_time_s": 0.002}) == pytest.approx(0.002)


def test_calc_sampling_time_axis_supports_offsets_and_precalculated_axes():
    np.testing.assert_allclose(
        calc_sampling_time_axis(
            no_points=3, dwell_time_s=0.25, start_with_0=False, t_echo_s=1.0
        ),
        [1.25, 1.5, 1.75],
    )

    scan = SimpleNamespace(sampling_time_axis_s=np.array([0.2, 0.5]))
    np.testing.assert_allclose(
        calc_sampling_time_axis(scan=scan, npoints=2), [0.2, 0.5]
    )


@pytest.mark.parametrize("bandwidth_name", ["bw_hz", "bandwidth_hz"])
def test_canonical_sampling_axis_requires_one_explicit_timing_source(bandwidth_name):
    """Analytic 1000 Hz sampling gives millisecond spacing, with no default."""
    np.testing.assert_allclose(
        calc_sampling_time_axis(no_points=3, **{bandwidth_name: 1_000}),
        [0.0, 0.001, 0.002],
        atol=1e-15,
        rtol=0,
    )
    with pytest.raises(ValueError, match="requires dwell_time_s or bandwidth"):
        calc_sampling_time_axis(no_points=3)
    with pytest.raises(ValueError, match="either dwell_time_s or bandwidth"):
        calc_sampling_time_axis(
            no_points=3, dwell_time_s=0.001, **{bandwidth_name: 1_000}
        )


@pytest.mark.parametrize("name", ["dwell_time_s", "bw_hz", "bandwidth_hz"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf])
def test_sampling_axis_rejects_nonpositive_and_nonfinite_timing(name, value):
    with pytest.raises(ValueError, match="finite and positive"):
        calc_sampling_time_axis(no_points=3, **{name: value})


@pytest.mark.parametrize("name", ["dwell_time_s", "bw_hz", "bandwidth_hz", "t_echo_s"])
@pytest.mark.parametrize("value", [True, "0.001", [0.001], np.array(0.001), 1j])
def test_sampling_axis_rejects_non_real_scalar_inputs(name, value):
    options = {"no_points": 3, name: value}
    if name == "t_echo_s":
        options["dwell_time_s"] = 0.001
    with pytest.raises(TypeError, match="real scalar"):
        calc_sampling_time_axis(**options)


@pytest.mark.parametrize("name", ["npoints", "no_points"])
@pytest.mark.parametrize("value", [True, 3.0, "3", [3], np.array([3]), np.array(3)])
def test_sampling_axis_rejects_noninteger_counts(name, value):
    with pytest.raises(TypeError):
        calc_sampling_time_axis(dwell_time_s=0.001, **{name: value})


@pytest.mark.parametrize("name", ["npoints", "no_points"])
def test_sampling_axis_rejects_negative_counts_but_preserves_empty_axes(name):
    with pytest.raises(ValueError, match="nonnegative npoints"):
        calc_sampling_time_axis(dwell_time_s=0.001, **{name: -1})
    axis = calc_sampling_time_axis(dwell_time_s=0.001, **{name: 0})
    assert axis.shape == (0,)
    assert axis.dtype == np.float64


@pytest.mark.parametrize(
    "options",
    [
        {"npoints": 3, "no_points": 3, "dwell_time_s": 0.001},
        {"bw_hz": 1000, "bandwidth_hz": 1000},
    ],
)
def test_sampling_axis_rejects_duplicate_aliases(options):
    with pytest.raises(ValueError, match="supply only one"):
        calc_sampling_time_axis(**options)


@pytest.mark.parametrize(
    "start_with_0, expected",
    [
        (True, [1.0, 1.25, 1.5]),
        (False, [1.25, 1.5, 1.75]),
    ],
)
def test_canonical_sampling_axis_explicit_dwell_and_echo_offset(start_with_0, expected):
    """Synthetic 0.25 s spacing and a 1 s origin shift are independent."""
    np.testing.assert_allclose(
        calc_sampling_time_axis(
            no_points=np.int64(3),
            dwell_time_s=np.float64(0.25),
            start_with_0=start_with_0,
            t_echo_s=1.0,
        ),
        expected,
        atol=1e-15,
        rtol=0,
    )
    assert calc_sampling_time_axis(dwell_time_s=0.001).shape == (512,)
    np.testing.assert_allclose(
        calc_sampling_time_axis(npoints=3, bw_hz=1000, t_echo_s=-0.001),
        [-0.001, 0, 0.001],
        atol=1e-15,
        rtol=0,
    )


@pytest.mark.parametrize("offset", [np.nan, np.inf, -np.inf])
def test_sampling_axis_rejects_nonfinite_offset(offset):
    with pytest.raises(ValueError, match="t_echo_s.*finite"):
        calc_sampling_time_axis(npoints=3, dwell_time_s=0.001, t_echo_s=offset)


def test_sampling_axis_rejects_invalid_start_flag_and_unrepresentable_times():
    with pytest.raises(TypeError, match="boolean start_with_0"):
        calc_sampling_time_axis(npoints=3, dwell_time_s=0.001, start_with_0=1)
    with pytest.raises(ValueError, match="increasing float64 times"):
        calc_sampling_time_axis(npoints=3, dwell_time_s=0.001, t_echo_s=1e20)
    with pytest.raises(ValueError, match="increasing float64 times"):
        calc_sampling_time_axis(npoints=3, dwell_time_s=np.finfo(np.float64).max)


def test_canonical_sampling_axis_metadata_and_cached_axis_precedence():
    """Synthetic metadata supplies count and dwell unless a cached axis fits."""

    # Metadata count is n_acquisition - cut_off (5 - 2); dwell_time_s metadata
    # is seconds. A cached sequence axis wins when its length matches npoints.
    class Metadata(SimpleNamespace):
        def get(self, name, default=None):
            return getattr(self, name, default)

    metadata = Metadata(n_acquisition=5, cut_off=2, dwell_time_s=0.004)
    np.testing.assert_allclose(
        calc_sampling_time_axis(scan=metadata),
        [0.0, 0.004, 0.008],
        atol=1e-15,
        rtol=0,
    )

    cached = SimpleNamespace(
        n_acquisition=5,
        cut_off=2,
        dwell_time_s=0.004,
        sampling_time_axis_s=np.array([0.1, 0.3, 0.7]),
    )
    np.testing.assert_array_equal(
        calc_sampling_time_axis(scan=cached, npoints=3), [0.1, 0.3, 0.7]
    )


def test_cached_axis_retains_irregular_spacing_offsets_and_input_dtype():
    """Invented MGE-like echo times stay authoritative over valid overrides."""
    cached = np.array([0.1, 0.3, 0.7], dtype=np.float32)
    before = cached.copy()
    scan = SimpleNamespace(sampling_time_axis_s=cached)
    actual = calc_sampling_time_axis(
        scan=scan,
        npoints=3,
        dwell_time_s=0.004,
        t_echo_s=1.0,
        start_with_0=False,
    )
    np.testing.assert_array_equal(actual, before)
    np.testing.assert_array_equal(cached, before)
    assert actual.dtype == np.float32
    with pytest.raises(ValueError, match="either dwell_time_s or bandwidth"):
        calc_sampling_time_axis(scan=scan, dwell_time_s=0.001, bw_hz=1000)
    with pytest.raises(ValueError, match="finite and positive"):
        calc_sampling_time_axis(scan=scan, dwell_time_s=0)


@pytest.mark.parametrize(
    "cached",
    [
        np.array(0.1),
        np.array([[0.1, 0.2]]),
        [0.1, np.nan],
        [0.1, 0.1],
        [0.2, 0.1],
        [0.1j, 0.2j],
        ["0.1", "0.2"],
    ],
)
def test_sampling_axis_rejects_invalid_cached_axis(cached):
    with pytest.raises(ValueError, match="sampling_time_axis_s"):
        calc_sampling_time_axis(scan=SimpleNamespace(sampling_time_axis_s=cached))


def test_sampling_axis_metadata_fallback_and_explicit_timing_contract():
    metadata = SimpleNamespace(
        n_acquisition=5,
        cut_off=2,
        dwell_time_s=0.004,
        sampling_time_axis_s=np.array([0.1, 0.2]),
    )
    np.testing.assert_allclose(
        calc_sampling_time_axis(scan=metadata, npoints=3),
        [0, 0.004, 0.008],
        atol=1e-15,
        rtol=0,
    )
    for options in ({"dwell_time_s": 0.002}, {"bw_hz": 500}, {"bandwidth_hz": 500}):
        np.testing.assert_allclose(
            calc_sampling_time_axis(scan=metadata, no_points=3, **options),
            [0, 0.002, 0.004],
            atol=1e-15,
            rtol=0,
        )
    for missing in (SimpleNamespace(), SimpleNamespace(n_acquisition=3)):
        with pytest.raises(ValueError, match="cannot determine npoints"):
            calc_sampling_time_axis(scan=missing, dwell_time_s=0.001)
    with pytest.raises(ValueError, match="requires dwell_time_s or bandwidth"):
        calc_sampling_time_axis(scan=SimpleNamespace(n_acquisition=3, cut_off=0))
    for invalid in (0, -1, np.inf, np.nan):
        with pytest.raises(ValueError, match="scan.dwell_time_s.*finite and positive"):
            calc_sampling_time_axis(
                scan=SimpleNamespace(n_acquisition=3, cut_off=0, dwell_time_s=invalid)
            )


def test_per_image_axis_and_inversion_time_validate_missing_metadata():
    scan = SimpleNamespace(n_repetitions=3, scan_duration_s=6.0)
    np.testing.assert_allclose(calc_per_image_timeaxis_s(scan), [0.0, 2.0, 4.0])
    np.testing.assert_allclose(
        calc_per_image_timeaxis_s(scan, start_with_0=False), [2.0, 4.0, 6.0]
    )
    assert calc_per_image_timeaxis_s() is None

    with pytest.raises(ValueError, match="Data object is None"):
        calc_inversion_time_multi_slice(None)

    assert calc_inversion_time_multi_slice({"refpulse_duration_ms": 0}) == 0
