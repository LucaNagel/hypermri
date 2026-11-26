"""Removing a line from prepared fit parameters and from a review session.

Fixtures are synthetic: a 2-voxel FID with 64 points at 2 kHz and three lines
at 0, 100 and 200 Hz, where line ``b`` follows the T2* of ``a``.
"""

import copy

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting.spectroscopy import (
    def_fit_params,
    remove_metab_from_fit_params,
)

TIME_S = np.arange(64) / 2000.0
LINES = {
    "a": {"freq_hz": 0.0, "t2_s": 0.02},
    "b": {"freq_hz": 100.0, "t2_s": 0.03},
    "c": {"freq_hz": 200.0, "t2_s": 0.04},
}


def _data():
    return xr.DataArray(
        np.ones((2, TIME_S.size)),
        dims=("x", "acquisition_idx"),
        coords={"x": [0, 1]},
        attrs={"signal_domain": "temporal", "sampling_time_axis_s": TIME_S},
    )


def _params(names, coupling=None):
    options = {
        "metabs": list(names),
        "metabs_freqs_hz": [LINES[name]["freq_hz"] for name in names],
        "initial_t2s_s": [LINES[name]["t2_s"] for name in names],
    }
    if coupling:
        options["t2_coupling"] = coupling
    return def_fit_params(data=_data(), fit_params=options)


@pytest.mark.parametrize(
    ("removed", "expected_lines", "expected_coupling"),
    [
        ("c", ["a", "b"], {"b": "a"}),
        ("b", ["a", "c"], None),
    ],
)
def test_removing_a_line_equals_preparing_the_reduced_set(
    removed, expected_lines, expected_coupling
):
    full = _params(["a", "b", "c"], coupling={"b": "a"})
    before = copy.deepcopy(full)

    reduced = remove_metab_from_fit_params(full, metab_name=removed)

    xr.testing.assert_identical(
        reduced, _params(expected_lines, coupling=expected_coupling)
    )
    xr.testing.assert_identical(full, before)


def test_removing_a_leader_with_dependents_is_rejected():
    full = _params(["a", "b", "c"], coupling={"b": "a"})
    with pytest.raises(ValueError, match=r"'a' leads the T2\* of \['b'\]"):
        remove_metab_from_fit_params(full, metab_name="a")


@pytest.mark.parametrize(
    ("name", "message"), [("d", "is not in fit_params"), ("a", "only line")]
)
def test_unknown_or_last_line_is_rejected(name, message):
    params = _params(["a"]) if name == "a" else _params(["a", "b"])
    with pytest.raises(ValueError, match=message):
        remove_metab_from_fit_params(params, metab_name=name)
