"""CSI filtering fails early instead of substituting another operation.

An unknown window name and a failing ``filter_signal`` previously returned
unwindowed samples or a truncated template that looked like a filtered
result. Both now raise. Synthetic namespaces stand in for scanner objects.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hypermri.sequences._csi_kspace_filter import FILTER_TYPES, SampleWindow
from hypermri.sequences.base_spectroscopy import BaseSpectroscopy
from hypermri.sequences.CSI import CSI


def _window(filter_type):
    return SampleWindow(
        filter_type=filter_type,
        decay_constant=0.01,
        alpha=0.54,
        center_position=None,
        signal_end=None,
        remove_lowest_signal=False,
        apply_flipangle_correction=False,
        flipangle_deg=None,
    )


@pytest.mark.parametrize("filter_type", FILTER_TYPES)
def test_documented_filter_names_are_accepted(filter_type):
    assert _window(filter_type).filter_type == filter_type


@pytest.mark.parametrize("filter_type", ["hann", "Gaussian", "", None])
def test_unknown_filter_name_lists_the_valid_names(filter_type):
    with pytest.raises(ValueError, match="'None', 'Exponential', 'Hann', 'Hamming'"):
        _window(filter_type)


def test_filter_signal_rejects_unknown_filter_name_before_filtering():
    sequence = SimpleNamespace()
    raw = np.arange(1, 13, dtype=complex).reshape(3, 4)
    original = raw.copy()

    with pytest.raises(ValueError, match="filter_type 'Hanning'"):
        BaseSpectroscopy.filter_signal(
            sequence, data=raw, filter_type="Hanning", signal_end=2
        )
    np.testing.assert_array_equal(raw, original)


def _csi_with_failing_filter(error):
    def filter_signal(**_):
        raise error

    return SimpleNamespace(
        data=xr.DataArray(np.ones((4, 3)), dims=("acquisition_idx", "read_idx")),
        filter_signal=filter_signal,
    )


@pytest.mark.parametrize(
    "error",
    [
        ValueError("filter_type 'Hanning' is not supported"),
        RuntimeError("reconstruction failed"),
    ],
)
def test_create_filtered_data_propagates_filter_errors(error):
    sequence = _csi_with_failing_filter(error)

    with pytest.raises(type(error), match=str(error)):
        CSI.create_filtered_data(sequence, end_index=2)


def test_create_filtered_data_rejects_unknown_name_from_filter_params():
    sequence = SimpleNamespace(
        data=xr.DataArray(np.ones((4, 3)), dims=("acquisition_idx", "read_idx")),
    )
    sequence.filter_signal = BaseSpectroscopy.filter_signal.__get__(sequence)
    sequence.handler = SimpleNamespace(rawdatajob0=np.ones(12, dtype=complex))

    with pytest.raises(ValueError, match="filter_type 'Gauss'"):
        CSI.create_filtered_data(
            sequence, filter_params={"filter_type": "Gauss", "signal_end": 2}
        )
