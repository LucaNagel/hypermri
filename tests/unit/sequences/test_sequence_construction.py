"""Construction of a sequence object from one vendor handler.

The scan is a synthetic ParaVision FLASH series written by the test: two
2 x 2 slices of 16-bit integers 0..7 without geometry or raw data.  Its state
is compared by object identity, exact equality of parameters, and
``xarray.DataArray.identical`` for arrays; no floating-point operation is
involved, so no tolerance applies.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from hypermri.loading import scans as factory
from hypermri.loading.experiment import MRIExperiment
from hypermri.sequences.FLASH import FLASH
from hypermri.vendors import get_vendor_handler

# State that is shared by design or names the construction route.
_ROUTE_KEYS = {
    "handler",
    "_sequence_requested_load_data",
    "_sequence_reco_options",
    "_sequence_log_mode",
}


def _assert_equal(actual, expected, key: str) -> None:
    assert type(actual) is type(expected), key
    if isinstance(actual, (xr.DataArray, xr.Dataset)):
        assert actual.identical(expected), key
    elif isinstance(actual, np.ndarray):
        assert actual.dtype == expected.dtype, key
        np.testing.assert_array_equal(actual, expected, err_msg=key)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), key
        for name in actual:
            _assert_equal(actual[name], expected[name], f"{key}.{name}")
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), key
        for index, (item, other) in enumerate(zip(actual, expected, strict=True)):
            _assert_equal(item, other, f"{key}[{index}]")
    else:
        assert actual == expected, key


def _assert_same_state(actual: MRIExperiment, expected: MRIExperiment) -> None:
    actual_state = {k: v for k, v in vars(actual).items() if k not in _ROUTE_KEYS}
    expected_state = {k: v for k, v in vars(expected).items() if k not in _ROUTE_KEYS}
    _assert_equal(actual_state, expected_state, "state")


def test_sequence_rejects_an_experiment_with_a_pointer_to_its_handler(
    synthetic_flash_scan,
):
    experiment = MRIExperiment(synthetic_flash_scan)

    with pytest.raises(TypeError, match=r"pass scan\.handler instead"):
        FLASH(experiment)


def test_sequence_construction_rejects_unknown_arguments(synthetic_flash_scan):
    handler = get_vendor_handler(str(synthetic_flash_scan))

    with pytest.raises(TypeError, match="positional argument"):
        FLASH(handler, "ignored")
    with pytest.raises(
        TypeError, match=r"'reco_option'; supported keywords are: .*reco_options"
    ):
        FLASH(handler, reco_option={})


def test_sequence_from_handler_takes_the_path_from_the_handler(synthetic_flash_scan):
    handler = get_vendor_handler(str(synthetic_flash_scan))

    scan = FLASH(handler)

    assert scan.path == synthetic_flash_scan.resolve()
    assert scan._sequence_requested_load_data is False


def test_sequence_from_handler_constructs_no_second_handler(
    synthetic_flash_scan, monkeypatch
):
    handler = get_vendor_handler(str(synthetic_flash_scan))

    def fail(*_args, **_kwargs):
        raise AssertionError("A given handler must be used as it is")

    monkeypatch.setattr(MRIExperiment, "_get_vendor_handler", fail)

    scan = FLASH(handler, log_mode="debug", force_vendor="Bruker")

    assert scan.handler is handler
    assert handler.log_mode == "debug"
    assert scan._forced_vendor == "bruker"
    assert MRIExperiment(synthetic_flash_scan, handler=handler).handler is handler


@pytest.mark.parametrize("mode", ["lazy", "minimal", "full"])
def test_load_scan_equals_direct_handler_construction(
    synthetic_flash_scan, monkeypatch, mode
):
    """``load_scan`` builds the same object as constructing from the handler."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expected = FLASH(
            get_vendor_handler(str(synthetic_flash_scan), load_data=mode == "full"),
            path=synthetic_flash_scan,
            load_data=mode,
            reco_options={"minimal_loading": mode == "minimal"},
        )
    factory._apply_sequence_load_policy(
        expected,
        mode,
        category="anatomical",
        reco_options={"minimal_loading": mode == "minimal"},
    )
    calls = []
    real_get_vendor_handler = factory.get_vendor_handler

    def counting_get_vendor_handler(**kwargs):
        calls.append(kwargs)
        return real_get_vendor_handler(**kwargs)

    monkeypatch.setattr(factory, "get_vendor_handler", counting_get_vendor_handler)

    actual = factory.load_scan(synthetic_flash_scan, mode=mode)

    assert type(actual) is FLASH
    assert len(calls) == 1
    assert type(actual.handler) is type(expected.handler)
    assert sorted(actual.reconstructions) == sorted(expected.reconstructions)
    _assert_same_state(actual, expected)
