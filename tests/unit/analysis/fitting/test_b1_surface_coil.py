"""Surface-coil B1 fitting uses the canonical fitting route on labelled data."""

import numpy as np
import xarray as xr

from hypermri.analysis.anatomical import b1_mapping
from hypermri.analysis.fitting import api


def _invented_b1_scan() -> tuple[xr.DataArray, np.ndarray]:
    powers_w = np.array([8.0, 1.0, 5.0, 2.0, 7.0, 3.0, 6.0, 4.0])
    amplitude = np.array(
        [[1.0 + 0.2j, 0.8 - 0.3j, 1.2 + 0.1j], [0.6 + 0.4j, 1.1 - 0.2j, 0.9 + 0.3j]],
        dtype=np.complex64,
    )
    signal = amplitude[..., np.newaxis] * np.sin(np.pi / 4.0 * np.sqrt(powers_w))
    data = xr.DataArray(
        signal.astype(np.complex64),
        dims=("read_idx", "phase_idx", "rfpower_idx"),
        coords={
            "read_idx": [9, 2],
            "phase_idx": [4, 1, 7],
            "rfpower_idx": powers_w,
        },
        attrs={"unit": "a.u.", "signal_domain": "temporal", "fixture": "invented"},
    )
    return data, amplitude


def test_surface_coil_fit_sorts_and_uses_canonical_api_without_mutation(monkeypatch):
    data, expected_amplitude = _invented_b1_scan()
    data_before = data.copy(deep=True)
    fit_options = {
        "fa_deg": 90.0,
        "trf_ms": 1.0,
        "sint": 1.0,
        "period_min": 2.0,
        "period_max": 6.0,
        "period_samples": 201,
        "use_multiprocessing": False,
    }
    options_before = dict(fit_options)
    sorted_input = data.sortby("rfpower_idx")
    api_options = {**fit_options, "dynamic_dim": "rfpower_idx"}
    expected = api.fit(sorted_input, "b1", api_options)

    def reject_legacy(*args, **kwargs):
        raise AssertionError("Surface-coil fitting used the deprecated fitting facade")

    monkeypatch.setattr(
        b1_mapping, "fit_quantitative_maps", reject_legacy, raising=False
    )
    actual = b1_mapping.fit_b1_surface_coil(
        data,
        fit_params=fit_options,
        dynamic_dim="rfpower_idx",
    )

    assert set(actual.data_vars) == set(expected.data_vars)
    for name in expected.data_vars:
        xr.testing.assert_equal(actual[name], expected[name])
    xr.testing.assert_equal(actual.coords.to_dataset(), expected.coords.to_dataset())
    for name in ("unit", "signal_domain", "fixture", "model_name", "dynamic_dim"):
        assert actual.attrs[name] == expected.attrs[name]
    xr.testing.assert_identical(data, data_before)
    assert fit_options == options_before
    assert data.dtype == np.complex64
    assert actual["amplitude"].dtype == expected["amplitude"].dtype
    np.testing.assert_allclose(actual["period"], 4.0, rtol=0.0, atol=0.02)
    np.testing.assert_allclose(
        actual["amplitude"], expected_amplitude, rtol=1e-6, atol=1e-6
    )
