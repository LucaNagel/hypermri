"""Tests of the fits saved on plotted objects by the browser viewer."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.visualization.saved_fits import (
    SavedFit,
    check_fit_name,
    save_fit,
    saved_fits,
)


def _fit():
    values = xr.DataArray(np.ones((2, 3)), dims=("read_idx", "phase_idx"))
    return SavedFit("t2", xr.Dataset({"t2": values}))


def test_fit_is_copied_onto_a_scan_attribute_and_replaced_by_name():
    scan = SimpleNamespace()
    assert saved_fits(scan) == {} and not hasattr(scan, "saved_fits")
    fit = _fit()
    save_fit(scan, "liver", fit)
    stored = scan.saved_fits["liver"]
    assert stored.results.identical(fit.results)
    fit.results["t2"][0, 0] = 5.0
    assert stored.results["t2"].values[0, 0] == 1.0

    save_fit(scan, "liver", SavedFit("adc", fit.results))
    assert saved_fits(scan) is scan.saved_fits
    assert list(scan.saved_fits) == ["liver"]
    assert scan.saved_fits["liver"].kind == "adc"


def test_fit_is_stored_in_the_attrs_of_a_dataarray():
    data = xr.DataArray(np.zeros(3), dims="read_idx")
    save_fit(data, "a", _fit())
    assert saved_fits(data) is data.attrs["saved_fits"]
    assert saved_fits(None) == {}
    with pytest.raises(ValueError, match="needs the plotted"):
        save_fit(None, "a", _fit())


def test_names_and_contents_are_checked():
    assert check_fit_name("  pyr-lac_2 ") == "pyr-lac_2"
    for name in ("", "2a", "a b", "a.b"):
        with pytest.raises(ValueError, match="not valid"):
            check_fit_name(name)
    with pytest.raises(ValueError, match="kind"):
        SavedFit("t3", xr.Dataset())
    with pytest.raises(TypeError, match="results"):
        SavedFit("t2", xr.DataArray(1.0))
    with pytest.raises(TypeError, match="params"):
        SavedFit("spectral", xr.Dataset(), params={})
