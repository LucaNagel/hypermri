"""Verify IDEAL frequency spellings through preparation and the actual fit path."""

import numpy as np
import pytest
import xarray as xr

from hypermri.analysis.fitting import api
from hypermri.analysis.fitting.ideal import IDEAL_Solver
from hypermri.analysis.fitting.models.quantitative import ideal


@pytest.mark.parametrize("key", ["relative_frequencies_hz", "relative_frequencies_Hz"])
def test_frequency_alias_reaches_solver_and_preserves_prepared_metadata(
    key, monkeypatch
):
    """Both accepted input spellings reach the solver with the same Hz values."""
    data = xr.DataArray(np.ones(3, dtype=complex), dims="echo_idx")
    frequencies = np.array([20.0])
    options = {key: frequencies, "echo_spacing_s": 0.001}
    prepared = api.prepare_fit_params(data, "ideal", options)
    # Preserve the existing prepared-dataset key for callers inspecting attrs.
    np.testing.assert_array_equal(
        prepared.attrs["relative_frequencies_Hz"], frequencies
    )

    def solve(self, *args, **kwargs):
        np.testing.assert_array_equal(self.relative_frequencies, frequencies)
        return xr.Dataset({"fit_amps": ("metabolite", np.array([1 + 2j]))})

    monkeypatch.setattr(IDEAL_Solver, "solve_xarray", solve)
    assert api.fit(data, "ideal", options).fit_amps.item() == 1 + 2j
    assert ideal.fit_ideal_dataset(data, options)[0].fit_amps.item() == 1 + 2j
    assert list(options) == [key, "echo_spacing_s"]


def test_conflicting_frequency_aliases_are_rejected_before_fitting():
    data = xr.DataArray(np.ones(3, dtype=complex), dims="echo_idx")
    options = {
        "relative_frequencies_hz": [0.0],
        "relative_frequencies_Hz": [5.0],
        "echo_spacing_s": 0.001,
    }
    for call in (ideal.prepare_ideal_fit_params_dataset, ideal.fit_ideal_dataset):
        with pytest.raises(ValueError, match="Conflicting relative_frequencies"):
            call(data, options)
    options["relative_frequencies_Hz"] = [0.0]
    prepared = ideal.prepare_ideal_fit_params_dataset(data, options)
    np.testing.assert_array_equal(prepared.attrs["relative_frequencies_Hz"], [0.0])
