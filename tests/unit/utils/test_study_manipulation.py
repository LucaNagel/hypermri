from types import SimpleNamespace

import numpy as np
import xarray as xr

from hypermri.sequences.base_anatomy import BaseAnatomy
from hypermri.utils.study.manipulation import _stack_scan_objects


class DummyAnatomySequence(BaseAnatomy):
    """Minimal sequence type used to exercise stacked scan construction."""


def test_stacked_sequence_builds_viewer_only_on_request(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hypermri.visualization.api.plot",
        lambda *a, **kw: calls.append((a, kw)) or "plot-result",
    )
    scans = [
        SimpleNamespace(
            data=xr.DataArray(
                np.full((2, 2), value),
                dims=("read_idx", "phase_idx"),
            )
        )
        for value in (1.0, 2.0)
    ]

    stacked = _stack_scan_objects(
        scan_objects=scans,
        sort_values=[10, 20],
        scan_indices_sorted=[1, 2],
        scan_type_class=DummyAnatomySequence,
        scan_type_name="DummyAnatomySequence",
        stack_dim_name="stack_idx",
        sort_by=None,
        match_orientation_to=None,
        average_repetitions=False,
        return_scan_object=True,
    )

    assert isinstance(stacked, DummyAnatomySequence)
    assert calls == []
    assert "_plotter" not in vars(stacked)
    assert stacked.plot() == "plot-result"
    assert calls[0][0][0] is stacked
