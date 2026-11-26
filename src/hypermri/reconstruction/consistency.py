"""Geometric agreement of two reconstructions of one scan.

A scan can be reconstructed from its raw data and from the images the scanner
stores, for example the Bruker ``2dseq`` file.  Both reconstructions describe
the same voxels, so they must order ``read_idx``, ``phase_idx`` and
``slice_idx`` alike, have the same number of positions along each, agree in
their mm coordinates and have the same orientation attributes.
:func:`check_reconstruction_geometry` checks this before
:meth:`hypermri.loading.experiment.MRIExperiment.reconstruct` stores a
reconstruction next to another one.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from hypermri.utils.coordinates.transformations import _POSITION_ATOL_MM

__all__ = ["check_reconstruction_geometry"]

_SPATIAL_DIMS = ("read_idx", "phase_idx", "slice_idx")
_SPATIAL_COORDS = (*_SPATIAL_DIMS, "x", "y", "z")
_ORIENTATION_ATTRS = ("patient_pos", "slice_orientation", "read_orientation")


def check_reconstruction_geometry(
    candidate: xr.DataArray,
    reference: xr.DataArray,
    *,
    candidate_name: str,
    reference_name: str,
) -> None:
    """Raise ``ValueError`` unless two reconstructions describe the same voxels.

    The check compares the spatial dimensions ``read_idx``, ``phase_idx``
    and ``slice_idx``, their mm coordinates and the orientation attributes
    of two reconstructions of one scan.  It collects every disagreement and
    reports them in one error.  It is a pure function of its inputs.

    * Order: the spatial dimensions present in both arrays follow each
      other in the same order.  Other dimensions may lie between them.
    * Sizes: each spatial dimension present in both arrays has the same
      number of positions.  A spatial dimension present in only one array is
      not compared, such as the single slice that the ``2dseq`` fallback
      squeezes or the CPMG echoes that the raw reconstruction stores along
      ``phase_idx``.
    * Coordinates: ``read_idx``, ``phase_idx``, ``slice_idx``, ``x``, ``y``
      and ``z`` agree within 1e-3 mm where both arrays have the coordinate
      with the attribute ``unit="mm"``.  Size-1 dimensions of a coordinate
      are ignored, so a 2D grid of a single slice matches a 3D grid with one
      slice.  Coordinates in other units, such as integer indices, are not
      compared.
    * Orientation: ``patient_pos``, ``slice_orientation`` and
      ``read_orientation`` are equal where both arrays have the attribute
      and it is not None.

    Non-spatial dimensions, their coordinates, data values and dtypes are
    not compared.

    Parameters
    ----------
    candidate
        The reconstruction to be stored, with spatial dimensions named
        ``read_idx``, ``phase_idx`` and ``slice_idx``.  It is not modified.
    reference
        A reconstruction of the same scan that is already stored, labelled
        like ``candidate``.  It is not modified.
    candidate_name
        Name of the reconstruction type of ``candidate``, for example
        ``"raw"``, used in the error message.
    reference_name
        Name of the reconstruction type of ``reference``, for example
        ``"seq2d"``, used in the error message.

    Returns
    -------
    None
        Successful validation returns without modifying either array.

    Raises
    ------
    ValueError
        If the spatial order, a spatial size, an mm coordinate or an
        orientation attribute of ``candidate`` differs from ``reference``.
        The message lists each difference.

    Examples
    --------
    .. code-block:: pycon

        >>> import numpy as np
        >>> import xarray as xr
        >>> from hypermri.reconstruction.consistency import (
        ...     check_reconstruction_geometry,
        ... )
        >>> seq2d = xr.DataArray(
        ...     np.zeros((4, 3, 2)), dims=("read_idx", "phase_idx", "slice_idx")
        ... )
        >>> check_reconstruction_geometry(
        ...     seq2d.copy(), seq2d, candidate_name="raw", reference_name="seq2d"
        ... )
        >>> check_reconstruction_geometry(
        ...     seq2d.transpose("read_idx", "slice_idx", "phase_idx"),
        ...     seq2d,
        ...     candidate_name="raw",
        ...     reference_name="seq2d",
        ... )
        Traceback (most recent call last):
            ...
        ValueError: The 'raw' reconstruction does not describe the voxels of the stored 'seq2d' reconstruction: spatial order ('read_idx', 'slice_idx', 'phase_idx') instead of ('read_idx', 'phase_idx', 'slice_idx'). ...
    """
    problems = [
        *_spatial_dimension_problems(candidate, reference),
        *_coordinate_problems(candidate, reference),
        *_orientation_problems(candidate, reference),
    ]
    if problems:
        raise ValueError(
            f"The {candidate_name!r} reconstruction does not describe the voxels "
            f"of the stored {reference_name!r} reconstruction: "
            f"{'; '.join(problems)}.  Both reconstructions of a scan must order "
            "read_idx, phase_idx and slice_idx alike, with equal sizes, mm "
            f"coordinates within {_POSITION_ATOL_MM} mm and equal orientation "
            "attributes.  Check the reconstruction options and the parameter "
            "files of the scan; to keep a reconstruction of another geometry, "
            f"first remove {reference_name!r} from the experiment's "
            "reconstructions."
        )


def _spatial_dimension_problems(
    candidate: xr.DataArray, reference: xr.DataArray
) -> list[str]:
    common = [
        dim for dim in _SPATIAL_DIMS if dim in candidate.dims and dim in reference.dims
    ]
    candidate_order = tuple(dim for dim in candidate.dims if dim in common)
    reference_order = tuple(dim for dim in reference.dims if dim in common)
    problems = []
    if candidate_order != reference_order:
        problems.append(f"spatial order {candidate_order} instead of {reference_order}")
    for dim in common:
        if candidate.sizes[dim] != reference.sizes[dim]:
            problems.append(
                f"{dim} has {candidate.sizes[dim]} instead of "
                f"{reference.sizes[dim]} positions"
            )
    return problems


def _coordinate_problems(candidate: xr.DataArray, reference: xr.DataArray) -> list[str]:
    problems = []
    for name in _SPATIAL_COORDS:
        if not (
            _has_mm_coordinate(candidate, name) and _has_mm_coordinate(reference, name)
        ):
            continue
        candidate_mm = candidate[name].variable.squeeze()
        reference_mm = reference[name].variable.squeeze()
        if dict(candidate_mm.sizes) != dict(reference_mm.sizes):
            problems.append(
                f"coordinate {name} spans {dict(candidate_mm.sizes)} instead of "
                f"{dict(reference_mm.sizes)}"
            )
            continue
        # Both reconstructions take their mm coordinates from the parameters
        # of one scan, so a deviation beyond the geometry position tolerance
        # means that they describe different voxels.
        close = np.isclose(
            candidate_mm.transpose(*reference_mm.dims).to_numpy(),
            reference_mm.to_numpy(),
            rtol=0.0,
            atol=_POSITION_ATOL_MM,
            equal_nan=True,
        )
        if not close.all():
            problems.append(
                f"coordinate {name} differs by more than {_POSITION_ATOL_MM} mm "
                f"at {np.count_nonzero(~close)} of {close.size} positions"
            )
    return problems


def _has_mm_coordinate(data: xr.DataArray, name: str) -> bool:
    return name in data.coords and data[name].attrs.get("unit") == "mm"


def _orientation_problems(
    candidate: xr.DataArray, reference: xr.DataArray
) -> list[str]:
    problems = []
    for name in _ORIENTATION_ATTRS:
        candidate_value = candidate.attrs.get(name)
        reference_value = reference.attrs.get(name)
        if candidate_value is None or reference_value is None:
            continue
        if np.asarray(candidate_value).tolist() != np.asarray(reference_value).tolist():
            problems.append(
                f"attribute {name} is {candidate_value!r} instead of "
                f"{reference_value!r}"
            )
    return problems
