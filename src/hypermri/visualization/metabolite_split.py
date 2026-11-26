"""Split alternating repetitions into two metabolites for display.

Some sequences excite two metabolites in turn, for example pyruvate in even
and lactate in odd repetitions. For display, these repetitions are split
into a dimension ``metabolite_idx`` of size two, so that each metabolite has
its own time axis. ``plot`` and ``plot_scan`` use this module to decide
whether to split (:func:`resolve_metabolite_split`) and to split
(:func:`split_interleaved_metabolites`). The split is a view change only;
the scan's data are not modified.
"""

from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "INTERLEAVED_METABOLITE_SEQUENCE",
    "METABOLITE_SPLIT_DIM",
    "resolve_metabolite_split",
    "split_interleaved_metabolites",
]

#: Sequence label (lower case, without angle brackets) of the sequence whose
#: repetitions alternate between two metabolites; it is split by default.
INTERLEAVED_METABOLITE_SEQUENCE = "user:fisp_grad_1metab_cs"

#: Name of the dimension that holds the two metabolites after the split.
METABOLITE_SPLIT_DIM = "metabolite_idx"


def _iter_sequence_label_values(value: Any):
    """Yield scalar label values from possibly nested method metadata."""
    if value is None:
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_sequence_label_values(item)
        return
    try:
        if isinstance(value, np.ndarray):
            for item in value.ravel():
                yield from _iter_sequence_label_values(item)
            return
    except Exception:
        pass
    yield value


def _normalize_sequence_label(value: Any) -> str:
    """Normalize scanner sequence labels such as '<User:foo>'."""
    label = str(value).strip().strip("'\"")
    if label.startswith("<") and label.endswith(">"):
        label = label[1:-1]
    return label.strip().lower()


def _sequence_labels(scan: Any) -> set[str]:
    """Return lowercase sequence labels available on a sequence object."""
    labels = set()
    for value in (
        type(scan).__name__,
        getattr(scan, "type", ""),
        getattr(scan, "sequence_name", ""),
    ):
        for label_value in _iter_sequence_label_values(value):
            labels.add(_normalize_sequence_label(label_value))

    method = getattr(getattr(scan, "handler", None), "method", None)
    if isinstance(method, dict):
        for key in ("Method", "method", "PVM_SequenceName", "PVM_Method"):
            value = method.get(key)
            for label_value in _iter_sequence_label_values(value):
                labels.add(_normalize_sequence_label(label_value))

    return {label for label in labels if label and label != "none"}


def _can_split_interleaved_metabolites(data: xr.DataArray) -> bool:
    return (
        "repetition_idx" in data.dims and int(data.sizes.get("repetition_idx", 0)) >= 2
    )


def _supports_interleaved_metabolite_split(scan: Any, data: xr.DataArray) -> bool:
    """Return True for the two-metabolite interleaved FISP sequence."""
    return INTERLEAVED_METABOLITE_SEQUENCE in _sequence_labels(
        scan
    ) and _can_split_interleaved_metabolites(data)


def _is_default_index_axis(values: np.ndarray) -> bool:
    values = np.asarray(values)
    if values.ndim != 1 or values.size <= 1:
        return False
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        np.allclose(numeric, np.arange(values.size), rtol=0.01, atol=1e-9)
        or np.allclose(numeric, np.arange(1, values.size + 1), rtol=0.01, atol=1e-9)
    )


def _coord_values_along_dim(data: xr.DataArray, dim: str) -> np.ndarray | None:
    if dim not in data.coords:
        return None
    coord = data.coords[dim]
    if coord.ndim <= 1:
        return np.asarray(coord.values)
    selector = {coord_dim: 0 for coord_dim in coord.dims if coord_dim != dim}
    return np.asarray(coord.isel(selector).values)


def split_interleaved_metabolites(
    data: xr.DataArray,
    *,
    repetition_dim: str = "repetition_idx",
    metabolite_dim: str = METABOLITE_SPLIT_DIM,
) -> xr.DataArray:
    """Split alternating repetitions into a two-entry metabolite dimension.

    Even repetitions (0, 2, 4, ...) become metabolite 0 and odd ones
    metabolite 1, both renumbered from 0 along ``repetition_dim``. An odd
    last repetition has no partner and is dropped.

    Parameters
    ----------
    data : xarray.DataArray
        Samples with the alternating repetitions along ``repetition_dim``.
        Not modified.
    repetition_dim : str, optional
        Dimension whose entries alternate between the two metabolites.
    metabolite_dim : str, optional
        Name of the new dimension of size two, inserted right after
        ``repetition_dim``.

    Returns
    -------
    xarray.DataArray
        New array with ``metabolite_dim`` (coordinate 0, 1) and half as many
        repetitions, or ``data`` itself when it has no ``repetition_dim`` or
        fewer than two repetitions. A physical repetition axis in seconds is
        kept per metabolite as coordinate ``time``. The attributes of
        ``data`` are copied, plus ``metabolite_split_source_dim``,
        ``metabolite_split_source_size`` and
        ``metabolite_split_dropped_repetitions``.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> data = xr.DataArray(np.arange(4.0), dims=("repetition_idx",))
    >>> split_interleaved_metabolites(data).values.tolist()
    [[0.0, 1.0], [2.0, 3.0]]
    """
    if repetition_dim not in data.dims:
        return data

    repetition_count = int(data.sizes[repetition_dim])
    pair_count = repetition_count // 2
    if pair_count < 1:
        return data

    trimmed_count = pair_count * 2
    pieces = []
    repetition_coord = _coord_values_along_dim(data, repetition_dim)
    repetition_attrs = dict(getattr(data.coords.get(repetition_dim), "attrs", {}))
    has_physical_repetition_coord = (
        repetition_coord is not None
        and repetition_coord.size >= trimmed_count
        and not _is_default_index_axis(repetition_coord[:trimmed_count])
    )

    for metabolite_index in range(2):
        piece = data.isel({repetition_dim: slice(metabolite_index, trimmed_count, 2)})
        piece = piece.assign_coords({repetition_dim: np.arange(pair_count)})

        if (
            has_physical_repetition_coord
            and "time_s" not in piece.coords
            and "time" not in piece.coords
        ):
            unit = str(repetition_attrs.get("units", repetition_attrs.get("unit", "")))
            if unit.lower() in {"s", "sec", "secs", "second", "seconds"}:
                piece = piece.assign_coords(
                    {
                        "time": (
                            repetition_dim,
                            repetition_coord[metabolite_index:trimmed_count:2],
                        )
                    }
                )
                piece.coords["time"].attrs.update({"unit": "s", "units": "s"})

        if repetition_dim in piece.coords:
            piece.coords[repetition_dim].attrs.update(repetition_attrs)
        pieces.append(piece)

    split = xr.concat(
        pieces,
        dim=xr.DataArray(
            np.arange(2, dtype=int),
            dims=(metabolite_dim,),
            name=metabolite_dim,
            attrs={"long_name": "Metabolite"},
        ),
    )

    desired_order: list[str] = []
    for dim in data.dims:
        desired_order.append(dim)
        if dim == repetition_dim:
            desired_order.append(metabolite_dim)
    split = split.transpose(*[dim for dim in desired_order if dim in split.dims])
    split.attrs.update(data.attrs)
    split.attrs["metabolite_split_source_dim"] = repetition_dim
    split.attrs["metabolite_split_source_size"] = repetition_count
    split.attrs["metabolite_split_dropped_repetitions"] = (
        repetition_count - trimmed_count
    )
    return split


def resolve_metabolite_split(
    scan: Any, data: xr.DataArray, requested: bool | None
) -> tuple[bool, bool]:
    """Decide whether alternating repetitions can be and are split.

    Parameters
    ----------
    scan : object
        Scan whose type, ``sequence_name`` or Bruker ``method`` names the
        sequence. Only read.
    data : xarray.DataArray
        Samples that would be split; a split needs at least two entries
        along ``repetition_idx``.
    requested : bool or None
        The caller's ``split_metabolites`` choice. None splits the sequence
        :data:`INTERLEAVED_METABOLITE_SEQUENCE` automatically; True splits
        any data with at least two repetitions; False never splits.

    Returns
    -------
    available : bool
        True when the split can be offered: the data come from the
        interleaved sequence, or ``requested`` is True, and have at least
        two repetitions.
    enabled : bool
        True when the split is to be shown now: ``requested`` when given,
        otherwise ``available``. Only meaningful together with
        ``available``.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> data = xr.DataArray(np.zeros(4), dims=("repetition_idx",))
    >>> resolve_metabolite_split(object(), data, None)
    (False, False)
    >>> resolve_metabolite_split(object(), data, True)
    (True, True)
    """
    automatic = _supports_interleaved_metabolite_split(scan, data)
    forced = bool(requested) and _can_split_interleaved_metabolites(data)
    available = automatic or forced
    enabled = bool(requested) if requested is not None else available
    return available, enabled
