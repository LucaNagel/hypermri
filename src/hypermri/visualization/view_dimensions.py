"""Choose which dimensions a viewer shows as image plane and as curve.

Every viewer shows labelled data in up to two panels: an image of two spatial
dimensions and a curve along one further dimension (a spectrum along
``acquisition_idx``, or a curve along echoes, repetitions, b-values and so on).
The functions here decide which dimensions fill these panels. They only look
at dimension names and sizes; they read no samples and change nothing.

The browser viewer and the render data preparation (``PlotDataModel``) both
use these functions, so the choice offered in a menu is the choice that is
drawn.

Examples
--------
>>> import numpy as np
>>> import xarray as xr
>>> data = xr.DataArray(
...     np.zeros((4, 4, 1, 5)),
...     dims=("read_idx", "phase_idx", "slice_idx", "bvalue_idx"),
... )
>>> displayed_spatial_dims(data, slicing_dim="slice_idx")
['phase_idx', 'read_idx']
>>> curve_dim_choices(data, ["phase_idx", "read_idx"])
['bvalue_idx']
>>> default_curve_dim(data, ["phase_idx", "read_idx"])
'bvalue_idx'
"""

import xarray as xr

__all__ = [
    "CURVE_DIM_PREFERENCE",
    "PLANE_DIMS_BY_SLICING_DIM",
    "SPATIAL_DIMS",
    "curve_dim_choices",
    "default_curve_dim",
    "displayed_spatial_dims",
    "view_plane_choices",
]

#: Spatial dimension names of the package's data model.
SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")

#: Image plane for each through-plane ("slicing") dimension, as
#: ``[vertical, horizontal]`` dimension names before display rotation.
PLANE_DIMS_BY_SLICING_DIM = {
    "slice_idx": ("phase_idx", "read_idx"),
    "read_idx": ("slice_idx", "phase_idx"),
    "phase_idx": ("slice_idx", "read_idx"),
}

#: Order in which a curve dimension is chosen when none was requested.
CURVE_DIM_PREFERENCE = (
    "inversion_idx",
    "bvalue_idx",
    "acquisition_idx",
    "echo_idx",
    "repetition_idx",
    "channel_idx",
    "slice_idx",
)


def displayed_spatial_dims(
    data: xr.DataArray, slicing_dim: str | None = None
) -> list[str]:
    """Return the two spatial dimensions shown as image, vertical first.

    Parameters
    ----------
    data : xarray.DataArray
        Labelled data; only its dimension names and sizes are read.
    slicing_dim : str or None, optional
        Through-plane dimension chosen by the user (``"slice_idx"``,
        ``"read_idx"`` or ``"phase_idx"``). It fixes the plane even when the
        dimension itself is no longer in ``data`` (for example after a single
        slice was selected), so the image keeps its orientation.

    Returns
    -------
    list of str
        ``[vertical, horizontal]`` dimension names before display rotation.
        Without a usable ``slicing_dim``: data with all three spatial
        dimensions shows the ``slice_idx`` plane, data with two shows the
        larger one vertically (read before phase before slice on equal size),
        and data with fewer returns the spatial dimensions it has (zero or
        one entry; no image can be drawn).
    """
    plane = PLANE_DIMS_BY_SLICING_DIM.get(slicing_dim or "", ())
    chosen = [dim for dim in plane if dim in data.dims]
    if len(chosen) == 2:
        return chosen

    present = [dim for dim in SPATIAL_DIMS if dim in data.dims]
    if len(present) == 3:
        return list(PLANE_DIMS_BY_SLICING_DIM["slice_idx"])
    if len(present) < 2:
        return present
    return sorted(present, key=lambda dim: data.sizes[dim], reverse=True)


def view_plane_choices(data: xr.DataArray) -> list[str]:
    """Return the through-plane dimensions a user may choose for the image.

    Parameters
    ----------
    data : xarray.DataArray
        Labelled data; only its dimension names are read.

    Returns
    -------
    list of str
        Slicing dimensions in the order ``slice_idx``, ``read_idx``,
        ``phase_idx`` whose two remaining spatial dimensions exist. Empty
        unless all three spatial dimensions are present.
    """
    present = [dim for dim in SPATIAL_DIMS if dim in data.dims]
    return [
        dim
        for dim in ("slice_idx", "read_idx", "phase_idx")
        if dim in present and len(present) == 3
    ]


def curve_dim_choices(data: xr.DataArray, spatial_dims: list[str]) -> list[str]:
    """Return the dimensions the curve panel may run along, sorted by name.

    Parameters
    ----------
    data : xarray.DataArray
        Labelled data; only its dimension names and sizes are read.
    spatial_dims : list of str
        The two dimensions shown as image; they cannot carry the curve.

    Returns
    -------
    list of str
        Dimensions with more than one entry that are not shown as image. A
        dimension with one entry (a single slice, say) would draw one point
        and is therefore not offered.
    """
    return sorted(
        str(dim) for dim in data.dims if data.sizes[dim] > 1 and dim not in spatial_dims
    )


def default_curve_dim(data: xr.DataArray, spatial_dims: list[str]) -> str | None:
    """Return the curve dimension used when none was requested.

    Parameters
    ----------
    data : xarray.DataArray
        Labelled data; only its dimension names and sizes are read.
    spatial_dims : list of str
        The two dimensions shown as image.

    Returns
    -------
    str or None
        The first entry of :data:`CURVE_DIM_PREFERENCE` among
        :func:`curve_dim_choices`, else the first choice by name, else None
        when no dimension has more than one entry outside the image.
    """
    choices = curve_dim_choices(data, spatial_dims)
    for dim in CURVE_DIM_PREFERENCE:
        if dim in choices:
            return dim
    return choices[0] if choices else None
