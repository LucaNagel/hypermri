"""Calculate supplied-domain magnitude ROI curves without constructing a viewer.

Notes
-----
Mask selection and display axes belong to the caller. Labelled weighting and
magnitude reduction delegate to ``masks.extract_masked_data``; this module
never reconstructs, preprocesses or transforms the supplied signal.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .masks import extract_masked_data

__all__ = ["raw_masked_mean", "mask_statistics", "mask_volume_ml"]

_LOGGER = logging.getLogger(__name__)


def _sanitize_mask_values(values) -> np.ndarray:
    """Retain the editor's float32 clipped fractional coverage policy."""
    weights = np.asarray(values, dtype=np.float32)
    weights = np.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(weights, 0.0, 1.0)


def _weighted_nanmean(values, weights) -> float:
    """Reduce finite positions using the established float32 coverage weights."""
    vals = np.asarray(values, dtype=np.float64)
    w = _sanitize_mask_values(weights).astype(np.float64, copy=False)
    valid = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return float("nan")
    w_valid = w[valid]
    weight_sum = float(np.sum(w_valid))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return float("nan")
    return float(np.sum(vals[valid] * w_valid) / weight_sum)


def raw_masked_mean(
    data: xr.DataArray,
    mask: xr.DataArray,
    *,
    secondary_dim: str,
    selectors: Mapping[str, int],
    preview_slice: tuple[str, int] | None = None,
) -> np.ndarray:
    """Calculate a coverage-weighted mean magnitude along the supplied axis.

    Magnitude is taken per voxel before averaging, including absolute values
    for real samples. Opposite complex phases therefore do not cancel. No FFT,
    phase/channel/PCA processing or display-part selection occurs. Committed
    masks use the full remaining spatial volume; previews use the displayed
    plane. Weighting and labelled grid validation have one owner in
    :func:`hypermri.analysis.anatomical.masks.extract_masked_data`.

    Parameters
    ----------
    data : xarray.DataArray
        Numeric supplied signal with a named secondary dimension. Any dimension
        order is accepted. Units, coordinate values/direction, attributes and
        acquired storage precision remain unchanged.
    mask : xarray.DataArray
        Dimensionless coverage on the same labelled spatial grid, in any order.
        Omitted spatial dimensions broadcast; present sizes and coordinates must
        match. Editor weights are rounded to float32 and clipped to [0, 1]; NaN
        and negative infinity become zero, positive infinity becomes one.
    secondary_dim : str
        Dimension sampled in original positional order. A spatial secondary
        dimension also indexes the committed mask at each position.
    selectors : mapping of str to int
        Explicit positional selections of other dimensions. Remaining dimensions
        are averaged spatially. Negative indices use xarray semantics; invalid
        selections yield NaN rather than averaging a different channel.
    preview_slice : tuple of str and int, optional
        Displayed slicing dimension and positional index. None uses the full
        committed volume. If it equals the secondary dimension, only that
        selected position has preview coverage; other positions remain NaN.

    Returns
    -------
    numpy.ndarray
        Eager float64 means with shape ``(data.sizes[secondary_dim],)`` in input
        signal units and axis order. Magnitude and weighting use at least double
        precision. Empty/all-invalid coverage yields NaN independently per
        position; nonfinite signal values contribute neither signal nor weight.

    Raises
    ------
    TypeError
        If the signal is not numeric or mask weights are not real numbers.
    ValueError
        If the secondary dimension is absent, or mask dimensions/coordinates
        do not match the selected signal grid. Flattened positional masks are
        rejected; project masks from other grids before this operation.
    IndexError
        If the committed mask lacks a required secondary-axis position.

    Notes
    -----
    Deterministic and non-mutating, with no cache, file reads, rendering or
    implicit processing. Use ``extract_masked_data`` directly for coherent
    complex sums/means or unreduced labelled values. This viewer convenience
    samples the supplied domain even when other viewer panels display an FFT.

    Examples
    --------
    >>> signal = xr.DataArray([[3j, 6j], [-9j, -12j]],
    ...                       dims=("read_idx", "echo_idx"))
    >>> coverage = xr.DataArray([0.25, 0.75], dims="read_idx")
    >>> raw_masked_mean(signal, coverage, secondary_dim="echo_idx", selectors={})
    array([ 7.5, 10.5])
    """
    if not isinstance(data, xr.DataArray) or not np.issubdtype(data.dtype, np.number):
        raise TypeError("raw_masked_mean requires a numeric xarray DataArray.")
    if (
        not isinstance(mask, xr.DataArray)
        or not (
            np.issubdtype(mask.dtype, np.number) or np.issubdtype(mask.dtype, np.bool_)
        )
        or np.issubdtype(mask.dtype, np.complexfloating)
    ):
        raise TypeError("raw_masked_mean mask must contain real coverage weights.")
    if secondary_dim not in data.dims:
        raise ValueError(
            f"raw_masked_mean secondary_dim {secondary_dim!r} must name a data dimension."
        )
    means = np.full(data.sizes[secondary_dim], np.nan)
    coverage = mask.astype(np.float32)
    for i in range(len(means)):
        selection = {**selectors, secondary_dim: i}
        if preview_slice is not None:
            preview_dim, preview_index = preview_slice
            if preview_dim == secondary_dim:
                resolved_index = preview_index % len(means) if len(means) else 0
                if not -len(means) <= preview_index < len(means) or i != resolved_index:
                    continue
            selection[preview_dim] = preview_index
        try:
            data_slice = data.isel(selection)
        except (IndexError, ValueError) as error:
            _LOGGER.debug("ROI data selection failed at position %s: %s", i, error)
            continue
        mask_use = (
            coverage.isel({secondary_dim: i})
            if preview_slice is None and secondary_dim in coverage.dims
            else coverage
        )
        if data_slice.ndim == 0:
            data_slice = data_slice.expand_dims("_roi_voxel")
            mask_use = mask_use.expand_dims("_roi_voxel")
        mean = extract_masked_data(
            data_slice,
            mask_use,
            reduction="mean",
            signal_part="magnitude",
            spatial_dims=data_slice.dims,
        )
        means[i] = float(mean.values)
    return means


MASK_STATISTICS_COLUMNS = (
    "mask",
    "variable",
    "mean",
    "std",
    "n_voxels",
    "weight_sum",
    "units",
)


def mask_statistics(
    maps: xr.Dataset,
    masks: xr.Dataset | Mapping[str, xr.DataArray],
    *,
    variables: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize parameter maps inside masks by their coverage-weighted mean and spread.

    For every mask and every map variable this gives the weighted mean and
    the weighted standard deviation over the voxels of the mask, for example
    the mean fitted frequency and T2* of each line inside a drawn ROI. Each
    voxel counts with its coverage, the fraction of the voxel inside the
    drawn outline, so a voxel at the border counts less than one inside.

    With coverage weights :math:`w_i` and values :math:`x_i`::

        mean = sum(w_i * x_i) / sum(w_i)
        std  = sqrt(sum(w_i * (x_i - mean) ** 2) / sum(w_i))

    ``std`` describes the spread of the values in the mask (population
    standard deviation), not the uncertainty of the mean. Voxels whose value
    is not finite (failed fit, NaN) are left out of both and of
    ``n_voxels``.

    Parameters
    ----------
    maps : xarray.Dataset
        Parameter maps on the voxel grid of the masks, for example the fit
        results of a CSI scan (``fit_freqs`` in Hz, ``fit_t2s`` in s,
        complex ``fit_amps``). Every mask dimension must be a dimension of a
        used variable, with the same size; the voxels are matched by
        position along these dimensions. Further dimensions such as
        ``metabolite`` or ``repetitions`` are kept: each of their
        positions gives its own row. Complex variables are summarized by
        their magnitude. Not modified.
    masks : xarray.Dataset or mapping of str to xarray.DataArray
        Coverage in [0, 1] by mask name, for example ``scan.masks`` or
        :attr:`ScanView.masks`. NaN counts as 0, values are clipped to
        [0, 1].
    variables : sequence of str, optional
        Variables of ``maps`` to summarize. None takes every numeric
        variable that has all mask dimensions; string variables such as fit
        status labels are skipped.

    Returns
    -------
    pandas.DataFrame
        One row per mask, variable and position of the further dimensions.
        Columns ``mask``, ``variable``, one column per further dimension
        (its coordinate value, or the position without a coordinate; empty
        for variables without that dimension), then ``mean`` and ``std`` in
        the units of the variable, ``n_voxels`` (voxels with coverage above
        0 and a finite value), ``weight_sum`` (their summed coverage, the
        weighted voxel count) and ``units`` (``attrs["units"]`` or ``""``).
        Rows follow the order of ``masks``, then ``variables``. ``mean``
        and ``std`` are NaN when no voxel of the mask has a finite value.

    Raises
    ------
    ValueError
        If there is no mask, a mask is not on the grid of ``maps`` (missing
        dimension or other size), a chosen variable is missing or not
        numeric, or no variable has all mask dimensions.

    Notes
    -----
    Deterministic and eager; neither input is modified. The masks cover the
    whole volume they were drawn in (all slices).

    Examples
    --------
    >>> table = mask_statistics(csi.fit_results, csi.masks)  # doctest: +SKIP
    >>> table[table.variable == "fit_freqs"][["mask", "metabolite", "mean", "std"]]  # doctest: +SKIP
    """
    coverages = dict(
        masks.data_vars.items() if isinstance(masks, xr.Dataset) else masks.items()
    )
    if not coverages:
        raise ValueError("mask_statistics needs at least one mask.")
    mask_dims = {dim for coverage in coverages.values() for dim in coverage.dims}
    if variables is None:
        names = [
            str(name)
            for name, values in maps.data_vars.items()
            if mask_dims.issubset(values.dims)
            and np.issubdtype(values.dtype, np.number)
        ]
        if not names:
            raise ValueError(
                "mask_statistics: no numeric variable of the maps has all mask "
                f"dimensions {sorted(mask_dims)}; the maps have "
                f"{ {str(n): list(v.dims) for n, v in maps.data_vars.items()} }."
            )
    else:
        names = [str(name) for name in variables]
        for name in names:
            if name not in maps:
                raise ValueError(
                    f"mask_statistics: no variable {name!r} in the maps; "
                    f"there are {sorted(map(str, maps.data_vars))}."
                )
            if not np.issubdtype(maps[name].dtype, np.number):
                raise ValueError(
                    f"mask_statistics: variable {name!r} is not numeric "
                    f"({maps[name].dtype}); leave it out of `variables`."
                )

    rows = []
    for mask_name, coverage in coverages.items():
        for name in names:
            values = maps[name]
            for dim in coverage.dims:
                if dim not in values.dims or values.sizes[dim] != coverage.sizes[dim]:
                    raise ValueError(
                        f"mask_statistics: mask {mask_name!r} has {dim!r} of size "
                        f"{coverage.sizes[dim]}, variable {name!r} has "
                        f"{dict(values.sizes)}; draw the mask on the grid of the "
                        "maps."
                    )
            rows.extend(_mask_rows(str(mask_name), name, values, coverage))
    table = pd.DataFrame(rows)
    further = [c for c in table.columns if c not in MASK_STATISTICS_COLUMNS]
    order = ["mask", "variable", *further, *MASK_STATISTICS_COLUMNS[2:]]
    return table[order]


def _mask_rows(
    mask_name: str, name: str, values: xr.DataArray, coverage: xr.DataArray
) -> list[dict]:
    """Return the statistics rows of one variable inside one mask."""
    voxel_dims = list(coverage.dims)
    further = [dim for dim in values.dims if dim not in voxel_dims]
    weights = _sanitize_mask_values(coverage.values).astype(np.float64).ravel()
    samples = values.transpose(*voxel_dims, *further).values
    if np.iscomplexobj(samples):
        samples = np.abs(samples)
    shape = tuple(values.sizes[dim] for dim in further)
    samples = np.asarray(samples, dtype=np.float64).reshape(weights.size, -1)
    rows = []
    for column, position in enumerate(np.ndindex(shape)):
        x = samples[:, column]
        valid = np.isfinite(x) & (weights > 0)
        w = weights[valid]
        weight_sum = float(w.sum())
        if weight_sum > 0:
            mean = float(np.sum(w * x[valid]) / weight_sum)
            std = float(np.sqrt(np.sum(w * (x[valid] - mean) ** 2) / weight_sum))
        else:
            mean = std = float("nan")
        row = {"mask": mask_name, "variable": name}
        for dim, index in zip(further, position, strict=True):
            label = values[dim].values[index] if dim in values.coords else index
            row[dim] = label.item() if isinstance(label, np.generic) else label
        row.update(
            mean=mean,
            std=std,
            n_voxels=int(valid.sum()),
            weight_sum=weight_sum,
            units=str(values.attrs.get("units", "")),
        )
        rows.append(row)
    return rows


def _extract_dim_coord_values(
    data: xr.DataArray,
    dim_name: str,
) -> np.ndarray | None:
    """Return 1D coordinate values for a dimension when available."""
    if dim_name not in data.coords:
        return None

    coord = data.coords[dim_name]
    try:
        if coord.ndim <= 1:
            values = np.asarray(coord.values, dtype=np.float64)
        else:
            selector = {d: 0 for d in coord.dims if d != dim_name}
            values = np.asarray(coord.isel(selector).values, dtype=np.float64)
    except Exception:
        return None

    values = np.ravel(values)
    finite = values[np.isfinite(values)]
    return finite if finite.size else None


def _looks_like_default_index_axis(values: np.ndarray) -> bool:
    """Return True when an axis appears to be a plain voxel index axis."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size <= 1:
        return False
    zero_based = np.arange(arr.size, dtype=np.float64)
    one_based = np.arange(1, arr.size + 1, dtype=np.float64)
    return bool(np.allclose(arr, zero_based) or np.allclose(arr, one_based))


def _spacing_mm_for_dim(data, scan, dim_name: str) -> float | None:
    """Estimate physical spacing in mm for one spatial dimension."""
    if isinstance(data, xr.DataArray) and dim_name in data.dims:
        coord_values = _extract_dim_coord_values(data, dim_name)
        if coord_values is not None and coord_values.size > 1:
            coord = data.coords.get(dim_name)
            unit = ""
            if coord is not None:
                unit = (
                    str(coord.attrs.get("unit", coord.attrs.get("units", "")))
                    .strip()
                    .lower()
                )
            diffs = np.diff(coord_values)
            diffs = np.abs(diffs[np.isfinite(diffs)])
            diffs = diffs[diffs > 0]
            if diffs.size:
                spacing = float(np.nanmedian(diffs))
                if np.isfinite(spacing) and spacing > 0:
                    if unit in {
                        "mm",
                        "millimeter",
                        "millimeters",
                        "millimetre",
                        "millimetres",
                    }:
                        return spacing
                    if not unit and not _looks_like_default_index_axis(coord_values):
                        return spacing

    if scan is None:
        return None

    if dim_name == "slice_idx":
        try:
            thickness = float(
                getattr(scan, "slice_thickness_mm", None)
                or scan.get("slice_thickness_mm", None)
            )
            if np.isfinite(thickness) and thickness > 0:
                return thickness
        except Exception:
            pass

    dim_to_index = {"read_idx": 0, "phase_idx": 1, "slice_idx": 2}
    axis_index = dim_to_index.get(dim_name)
    if axis_index is None:
        return None

    try:
        fov = np.atleast_1d(np.asarray(scan.fov_mm, dtype=float))
        matrix = np.atleast_1d(np.asarray(scan.matrix, dtype=float))
        if axis_index < fov.size and axis_index < matrix.size and matrix[axis_index]:
            spacing = float(abs(fov[axis_index] / matrix[axis_index]))
            if np.isfinite(spacing) and spacing > 0:
                return spacing
    except Exception:
        pass

    return None


def mask_volume_ml(
    mask: xr.DataArray,
    data: xr.DataArray,
    scan=None,
) -> float | None:
    """Estimate fractional ROI volume using the editor's existing spacing policy.

    Parameters
    ----------
    mask : xarray.DataArray
        Spatial fractional coverage. Values are sanitized/clipped to float32
        in [0, 1], retaining the editor's weighted voxel-count precision.
    data : xarray.DataArray
        Labelled source grid with read_idx, phase_idx and optional slice_idx.
        Millimetre coordinate differences use their absolute median; unlabelled
        non-index axes retain the legacy millimetre interpretation. Direction,
        samples, metadata and mask values are not mutated.
    scan : object or None, optional
        Existing acquisition context for unresolved spacing: slice thickness
        in mm, then fov_mm divided by matrix counts. No scans are loaded.

    Returns
    -------
    float or None
        Weighted volume in mL (1000 cubic mm per mL), or None when spacing is
        unavailable. Empty coverage gives zero when geometry is available.

    Notes
    -----
    Pure, eager and deterministic. This preserves the editor's spacing estimate;
    it does not define physical plane tolerance for projection or change
    swapped anatomy-overlay drawing. No synthetic thickness is introduced
    for a missing spatial dimension.

    Examples
    --------
    >>> mask_volume_ml(mask, anatomy.primary_data, anatomy)  # doctest: +SKIP
    """
    dims = list(mask.dims)
    dims.extend(
        d
        for d in ("read_idx", "phase_idx", "slice_idx")
        if d in data.dims and d not in dims
    )
    spacing = [_spacing_mm_for_dim(data, scan, d) for d in dims]
    if any(v is None for v in spacing):
        return None
    count = float(np.sum(_sanitize_mask_values(mask.values)))
    result = count * float(np.prod(spacing)) / 1000.0
    return result if np.isfinite(result) else None
